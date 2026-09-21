/*
 * `--infer`: the complete native end-to-end path, batched.
 *
 *   request.json
 *     -> laya_build_sequence per question (input_ids + marker positions)
 *     -> collate into [B, L] with pad_token_id right padding
 *     -> laya_encoder_forward_batch (ModernBERT, 28 layers, rows = B*L)
 *     -> laya_decision_forward_batch (type embedding, 2-layer
 *        TransformerEncoder head, marker gather, scorer, calibration,
 *        answers, action head)
 *     -> JSON with one entry per question: type, choice/score/noul,
 *        probabilities, confidence, act_probability.
 *
 * This mirrors `collate_items` + `DecisionModel.forward` + `system_one`: all
 * questions go through ONE forward, exactly as the Python model does.
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

#define INFER_MAX_Q 256

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
    (void)dump_spec;
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
    int B = (int)questions->u.object.count;
    if (B > INFER_MAX_Q) {
        fprintf(stderr, "error: %d questions exceeds the %d supported\n", B, INFER_MAX_Q);
        hd_json_free(root);
        return 1;
    }

    const int d = model->cfg.hidden_size;
    int L = 0, kmax = 0;
    int **q_ids = calloc((size_t)B, sizeof(int *));
    size_t *q_n = calloc((size_t)B, sizeof(size_t));
    int **q_mk = calloc((size_t)B, sizeof(int *));
    size_t *q_nm = calloc((size_t)B, sizeof(size_t));
    int *qtype = calloc((size_t)B, sizeof(int));
    char **qids = calloc((size_t)B, sizeof(char *));

    int rc = 1;
    for (int b = 0; b < B; b++) {
        qids[b] = strdup(questions->u.object.keys[b]);
        const hd_json *qdef = questions->u.object.values[b];
        qtype[b] = laya_qtype_of(hd_json_string(hd_json_get(qdef, "type")));
        hd_json *q = laya_question_internal(qdef);
        if (!q || qtype[b] < 0) {
            fprintf(stderr, "error: bad question %s\n", qids[b]);
            hd_json_free(q);
            goto cleanup;
        }
        laya_status st = laya_build_sequence(state, q, model->cfg.max_len,
                                             model->cfg.head_max_len,
                                             &q_ids[b], &q_n[b], &q_mk[b], &q_nm[b]);
        hd_json_free(q);
        if (st != LAYA_OK) {
            fprintf(stderr, "error: build_sequence: %s\n", laya_last_error());
            goto cleanup;
        }
        if (q_nm[b] == 0) {
            fprintf(stderr, "error: no option markers built for %s\n", qids[b]);
            goto cleanup;
        }
        if ((int)q_n[b] > L) L = (int)q_n[b];
        if ((int)q_nm[b] > kmax) kmax = (int)q_nm[b];
    }
    hd_json_free(root);
    root = NULL;

    /* ---- collate: input_ids [B,L], attention_mask [B,L], markers [B,kmax] ---- */
    const int R = B * L, M = B * kmax;
    const int pad = model->cfg.pad_token_id;
    int64_t *ids64 = malloc((size_t)R * sizeof(int64_t));
    uint8_t *tok_valid = calloc((size_t)R, 1);
    int64_t *mflat = malloc((size_t)M * sizeof(int64_t));
    uint8_t *mmask = calloc((size_t)M, 1);
    for (int b = 0; b < B; b++) {
        for (int s = 0; s < L; s++) {
            int real = s < (int)q_n[b];
            ids64[(size_t)b * L + s] = real ? q_ids[b][s] : pad;
            tok_valid[(size_t)b * L + s] = real ? 1 : 0;
        }
        for (int j = 0; j < kmax; j++) {
            int real = j < (int)q_nm[b];
            mflat[(size_t)b * kmax + j] = real ? (int64_t)(b * L + q_mk[b][j]) : (int64_t)(b * L);
            mmask[(size_t)b * kmax + j] = real ? 1 : 0;
        }
    }

    void *ids_dev = NULL, *enc_dev = NULL, *markers_dev = NULL;
    cudaMalloc(&ids_dev, (size_t)R * sizeof(int64_t));
    cudaMalloc(&enc_dev, (size_t)R * (size_t)d * 2);
    cudaMalloc(&markers_dev, (size_t)M * sizeof(int64_t));
    cudaMemcpy(ids_dev, ids64, (size_t)R * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(markers_dev, mflat, (size_t)M * sizeof(int64_t), cudaMemcpyHostToDevice);
    free(ids64);

    laya_encoder enc;
    memset(&enc, 0, sizeof(enc));
    laya_status st = laya_encoder_forward_batch(&enc, model, ids_dev, B, L,
                                                tok_valid, enc_dev);
    if (st != LAYA_OK) {
        fprintf(stderr, "error: encoder: %s\n", laya_encoder_last_error());
        goto fail;
    }

    laya_decision dec;
    memset(&dec, 0, sizeof(dec));
    laya_decision_result *res = calloc((size_t)B, sizeof(laya_decision_result));
    st = laya_decision_forward_batch(&dec, model, enc_dev, B, L, markers_dev, kmax,
                                     mmask, qtype, tok_valid, res);
    if (st != LAYA_OK) {
        fprintf(stderr, "error: decision head: %s\n", laya_decision_last_error());
        goto fail_dec;
    }

    FILE *out = out_path ? fopen(out_path, "wb") : stdout;
    if (!out) {
        fprintf(stderr, "error: cannot write %s\n", out_path);
        goto fail_dec;
    }

    fputs("{\n \"questions\": {\n", out);
    for (int b = 0; b < B; b++) {
        const laya_decision_result *r = &res[b];
        if (b) fputs(",\n", out);
        fputs("  \"", out);
        fputs(qids[b], out);
        fprintf(out, "\": {\n   \"type\": %d,\n", r->qtype);
        fputs("   \"probabilities\": [", out);
        for (int i = 0; i < r->n_options; i++) {
            if (i) fputs(", ", out);
            print_f32(out, r->probs[i]);
        }
        fputs("],\n   \"confidence\": ", out);
        print_f32(out, r->confidence);
        fputs(",\n", out);
        if (r->qtype == 0) {
            fprintf(out, "   \"choice\": %d,\n", r->choice);
        } else if (r->qtype == 1) {
            fputs("   \"score\": ", out);
            print_f32(out, r->score);
            fputs(",\n", out);
        } else {
            fputs("   \"noul\": ", out);
            print_f32(out, r->noul);
            fputs(",\n", out);
        }
        fputs("   \"act_probability\": ", out);
        print_f32(out, r->act_probability);
        fputs(",\n", out);
        fputs("   \"raw_logits\": [", out);
        for (int i = 0; i < r->n_options; i++) {
            if (i) fputs(", ", out);
            print_f32(out, r->raw_logits[i]);
        }
        fputs("]\n  }", out);
    }
    fputs("\n }\n}\n", out);
    if (out_path) fclose(out);

    /* Encoder final hidden state tap, for the [B*L, d] boundary comparison.
     * The decision-head boundaries are tapped inside decision.c. */
    {
        const char *dbg = getenv("LAYA_DEBUG_PROBE");
        if (dbg && dbg[0]) {
            char path[512];
            snprintf(path, sizeof(path), "%s.enc_final.bin", dbg);
            size_t nb = (size_t)R * d * 2;
            uint8_t *host = malloc(nb);
            cudaMemcpy(host, enc_dev, nb, cudaMemcpyDeviceToHost);
            FILE *f = fopen(path, "wb");
            if (f) { fwrite(host, 1, nb, f); fclose(f); }
            free(host);
        }
    }

    free(res);
    laya_decision_free(&dec);
    laya_encoder_free(&enc);
    cudaFree(ids_dev);
    cudaFree(enc_dev);
    cudaFree(markers_dev);
    rc = 0;
    goto cleanup;

fail_dec:
    free(res);
    laya_decision_free(&dec);
fail:
    laya_encoder_free(&enc);
    cudaFree(ids_dev);
    cudaFree(enc_dev);
    cudaFree(markers_dev);
    rc = 1;

cleanup:
    if (root) hd_json_free(root);
    free(tok_valid);
    free(mflat);
    free(mmask);
    for (int b = 0; b < B; b++) {
        free(q_ids[b]);
        free(q_mk[b]);
        free(qids[b]);
    }
    free(q_ids);
    free(q_n);
    free(q_mk);
    free(q_nm);
    free(qtype);
    free(qids);
    return rc;
}
