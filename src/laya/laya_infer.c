/*
 * `--infer`: the first complete native end-to-end path.
 *
 *   request.json
 *     -> laya_build_sequence (input_ids + marker positions)
 *     -> laya_encoder_forward (ModernBERT, 28 layers)
 *     -> laya_decision_forward (type embedding, 2-layer TransformerEncoder
 *        head, marker gather, scorer, calibration, answers, action head)
 *     -> JSON with raw logits, calibrated probabilities, choice/score/noul and
 *        the action probability.
 *
 * One question per run, which is the bring-up contract: the decision head's
 * padding mask is built for a single sequence. Batching is a later concern.
 */

#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>

#include "decision.h"
#include "hd_cuda.h"
#include "hd_json.h"
#include "laya_sequence.h"
#include "modernbert.h"

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)n + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)n, f);
    buf[got] = '\0';
    fclose(f);
    return buf;
}

static void print_f32(FILE *out, float v) {
    if (isnan(v) || isinf(v)) fprintf(out, "\"%s\"", isnan(v) ? "nan" : "inf");
    else fprintf(out, "%.6g", (double)v);
}

int infer_request(const laya_model *model, const char *request_path,
                  const char *dump_spec, const char *out_path) {
    if (!request_path) {
        fprintf(stderr, "error: --infer requires --request PATH\n");
        return 2;
    }
    char *text = read_file(request_path);
    if (!text) {
        fprintf(stderr, "error: cannot read %s\n", request_path);
        return 1;
    }
    const char *jerr = NULL;
    hd_json *root = hd_json_parse(text, &jerr);
    free(text);
    if (!root) {
        fprintf(stderr, "error: bad request: %s\n", jerr ? jerr : "parse");
        return 1;
    }

    const hd_json *state = hd_json_get(root, "state");
    const hd_json *questions = hd_json_get(root, "questions");
    if (!questions || questions->type != HD_JSON_OBJECT || questions->u.object.count == 0) {
        fprintf(stderr, "error: request has no questions\n");
        hd_json_free(root);
        return 1;
    }
    char *qid = strdup(questions->u.object.keys[0]);
    const hd_json *qdef = questions->u.object.values[0];
    const char *qtype_str = hd_json_string(hd_json_get(qdef, "type"));
    int qtype = laya_qtype_of(qtype_str);
    hd_json *q = laya_question_internal(qdef);
    if (!q || qtype < 0) {
        fprintf(stderr, "error: bad question %s\n", qid);
        hd_json_free(q);
        hd_json_free(root);
        free(qid);
        return 1;
    }

    int *ids = NULL, *markers = NULL;
    size_t n = 0, nm = 0;
    laya_status st = laya_build_sequence(state, q, model->cfg.max_len,
                                         model->cfg.head_max_len, &ids, &n, &markers, &nm);
    hd_json_free(q);
    hd_json_free(root);
    if (st != LAYA_OK) {
        fprintf(stderr, "error: build_sequence: %s\n", laya_last_error());
        free(qid);
        return 1;
    }
    if (nm == 0) {
        fprintf(stderr, "error: no option markers built\n");
        free(ids); free(markers); free(qid);
        return 1;
    }

    int d = model->cfg.hidden_size;
    int64_t *ids64 = malloc(n * sizeof(int64_t));
    for (size_t i = 0; i < n; i++) ids64[i] = ids[i];
    int32_t *markers32 = malloc(nm * sizeof(int32_t));
    for (size_t i = 0; i < nm; i++) markers32[i] = markers[i];

    void *ids_dev = NULL, *enc_dev = NULL, *markers_dev = NULL;
    cudaMalloc(&ids_dev, n * sizeof(int64_t));
    cudaMalloc(&enc_dev, n * (size_t)d * 2);
    cudaMalloc(&markers_dev, nm * sizeof(int32_t));
    cudaMemcpy(ids_dev, ids64, n * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(markers_dev, markers32, nm * sizeof(int32_t), cudaMemcpyHostToDevice);
    free(ids64);

    laya_encoder enc;
    memset(&enc, 0, sizeof(enc));
    st = laya_encoder_forward(&enc, model, ids_dev, (int)n, enc_dev);
    if (st != LAYA_OK) {
        fprintf(stderr, "error: encoder: %s\n", laya_encoder_last_error());
        goto fail;
    }

    /* every token is real: build_sequence emits a dense, right-padded-free
     * sequence for a single question. */
    uint8_t *tok_valid = calloc(n, 1);
    for (size_t i = 0; i < n; i++) tok_valid[i] = 1;
    uint8_t *marker_valid = calloc(nm, 1);
    for (size_t i = 0; i < nm; i++) marker_valid[i] = 1;

    laya_decision dec;
    memset(&dec, 0, sizeof(dec));
    laya_decision_result res;
    st = laya_decision_forward(&dec, model, enc_dev, (int)n, markers_dev, (int)nm,
                               marker_valid, qtype, tok_valid, &res);
    free(tok_valid);
    free(marker_valid);
    if (st != LAYA_OK) {
        fprintf(stderr, "error: decision head: %s\n", laya_decision_last_error());
        goto fail;
    }

    FILE *out = out_path ? fopen(out_path, "wb") : stdout;
    if (!out) {
        fprintf(stderr, "error: cannot write %s\n", out_path);
        goto fail;
    }

    fputs("{\n \"question_id\": \"", out);
    fputs(qid, out);
    fprintf(out, "\",\n \"qtype\": %d,\n \"n_options\": %d,\n", qtype, res.n_options);

    fputs(" \"input_ids\": [", out);
    for (size_t i = 0; i < n; i++) fprintf(out, "%s%d", i ? ", " : "", ids[i]);
    fputs("],\n \"marker_pos\": [", out);
    for (size_t i = 0; i < nm; i++) fprintf(out, "%s%d", i ? ", " : "", markers[i]);
    fputs("],\n", out);

    fputs(" \"raw_logits\": [", out);
    for (int i = 0; i < res.n_options; i++) {
        if (i) fputs(", ", out);
        print_f32(out, res.raw_logits[i]);
    }
    fputs("],\n \"probabilities\": [", out);
    for (int i = 0; i < res.n_options; i++) {
        if (i) fputs(", ", out);
        print_f32(out, res.probs[i]);
    }
    fputs("],\n", out);

    fprintf(out, " \"choice\": %d,\n", res.choice);
    fprintf(out, " \"confidence\": ");
    print_f32(out, res.confidence);
    fputs(",\n \"score\": ", out);
    print_f32(out, res.score);
    fputs(",\n \"noul\": ", out);
    print_f32(out, res.noul);
    fputs(",\n", out);

    fputs(" \"act_logits\": [", out);
    fprintf(out, "%g, %g", (double)res.act_logits[0], (double)res.act_logits[1]);
    fputs("],\n \"act_probability\": ", out);
    print_f32(out, res.act_probability);
    fputs("\n", out);
    fputs("}\n", out);
    if (out_path) fclose(out);

    /* Encoder final hidden state tap, for the one encoder boundary still
     * compared end to end. The decision-head boundaries are tapped inside
     * decision.c on the same env var. */
    {
        const char *dbg = getenv("LAYA_DEBUG_PROBE");
        if (dbg && dbg[0]) {
            char path[512];
            snprintf(path, sizeof(path), "%s.enc_final.bin", dbg);
            size_t nb = (size_t)n * d * 2;
            uint8_t *host = malloc(nb);
            cudaMemcpy(host, enc_dev, nb, cudaMemcpyDeviceToHost);
            FILE *f = fopen(path, "wb");
            if (f) { fwrite(host, 1, nb, f); fclose(f); }
            free(host);
        }
    }

    laya_decision_free(&dec);
    laya_encoder_free(&enc);
    cudaFree(ids_dev);
    cudaFree(enc_dev);
    cudaFree(markers_dev);
    free(ids);
    free(markers);
    free(qid);
    return 0;

fail:
    laya_encoder_free(&enc);
    cudaFree(ids_dev);
    cudaFree(enc_dev);
    cudaFree(markers_dev);
    free(ids);
    free(markers);
    free(qid);
    return 1;
}
