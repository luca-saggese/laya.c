/*
 * Batched Laya end-to-end inference.
 *
 * A parsed System One request ({"state", "questions"}) becomes:
 *
 *   request questions
 *     -> laya_build_sequence per question (input_ids + marker positions)
 *     -> collate into [B, L] with pad_token_id right padding
 *     -> laya_encoder_forward_batch (ModernBERT, 28 layers, rows = B*L)
 *     -> laya_decision_forward_batch (type embedding, 2-layer
 *        TransformerEncoder head, marker gather, scorer, calibration,
 *        answers, action head)
 *     -> one typed result per question
 *
 * All questions go through ONE forward, exactly as the Python model does.
 * This is the single forward path: the `--infer` harness and the HTTP
 * server both call laya_infer_run and never build their own forward.
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
#include "laya_infer.h"
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

static char **option_keys_of(const hd_json *qdef, int qtype, int n_opts) {
    char **keys = calloc((size_t)n_opts, sizeof(char *));
    if (!keys) return NULL;
    if (qtype == LAYA_QTYPE_CHOICE) {
        const hd_json *crit = hd_json_get(qdef, "criteria");
        if (!crit || crit->type != HD_JSON_OBJECT ||
            (int)crit->u.object.count != n_opts) {
            free(keys);
            return NULL;
        }
        for (int i = 0; i < n_opts; i++)
            keys[i] = strdup(crit->u.object.keys[i]);
    } else {
        for (int i = 0; i < n_opts; i++) {
            char tmp[24];
            snprintf(tmp, sizeof tmp, "%d", i);
            keys[i] = strdup(tmp);
        }
    }
    return keys;
}

void laya_infer_results_free(laya_infer_result *res, int n) {
    if (!res) return;
    for (int i = 0; i < n; i++) {
        if (res[i].option_keys) {
            for (int k = 0; k < res[i].n_options; k++) free(res[i].option_keys[k]);
            free(res[i].option_keys);
        }
        if (res[i].option_labels) {
            for (int k = 0; k < res[i].n_options; k++) free(res[i].option_labels[k]);
            free(res[i].option_labels);
        }
        free(res[i].probs);
        free(res[i].raw_logits);
    }
    free(res);
}

laya_status laya_infer_run(const laya_model *model, const hd_json *root,
                           laya_infer_result **out_res, int *out_n,
                           long *out_input_tokens) {
    *out_res = NULL;
    *out_n = 0;
    if (out_input_tokens) *out_input_tokens = 0;

    const hd_json *state = hd_json_get(root, "state");
    const hd_json *questions = hd_json_get(root, "questions");
    if (!questions || questions->type != HD_JSON_OBJECT ||
        questions->u.object.count == 0) {
        laya_set_error("request has no questions");
        return LAYA_ERR_PARSE;
    }
    int B = (int)questions->u.object.count;
    if (B > LAYA_INFER_MAX_Q) {
        laya_set_error("%d questions exceeds the %d supported", B,
                       LAYA_INFER_MAX_Q);
        return LAYA_ERR_UNSUPPORTED;
    }

    const int d = model->cfg.hidden_size;
    int L = 0, kmax = 0;
    int64_t *ids64 = NULL;
    uint8_t *tok_valid = NULL;
    int64_t *mflat = NULL;
    uint8_t *mmask = NULL;
    void *ids_dev = NULL, *enc_dev = NULL, *markers_dev = NULL;
    laya_decision_result *res = NULL;
    laya_infer_result *out = NULL;
    int **q_ids = calloc((size_t)B, sizeof(int *));
    size_t *q_n = calloc((size_t)B, sizeof(size_t));
    int **q_mk = calloc((size_t)B, sizeof(int *));
    size_t *q_nm = calloc((size_t)B, sizeof(size_t));
    int *qtype = calloc((size_t)B, sizeof(int));
    char **qids = calloc((size_t)B, sizeof(char *));
    laya_options *qopts = calloc((size_t)B, sizeof(laya_options));
    char ***qkeys = calloc((size_t)B, sizeof(char **));
    laya_status st = LAYA_OK;

    if (!q_ids || !q_n || !q_mk || !q_nm || !qtype || !qids || !qopts ||
        !qkeys) {
        st = LAYA_ERR_OOM;
        goto cleanup;
    }

    for (int b = 0; b < B; b++) {
        qids[b] = strdup(questions->u.object.keys[b]);
        const hd_json *qdef = questions->u.object.values[b];
        qtype[b] = laya_qtype_of(hd_json_string(hd_json_get(qdef, "type")));
        hd_json *q = laya_question_internal(qdef);
        if (!q || qtype[b] < 0) {
            hd_json_free(q);
            laya_set_error("bad question '%s'", qids[b] ? qids[b] : "?");
            st = LAYA_ERR_PARSE;
            goto cleanup;
        }
        /* Rendering is needed for the wire answer labels; build_sequence
         * renders the same options internally. */
        laya_render_options(q, &qopts[b]);
        qkeys[b] = option_keys_of(qdef, qtype[b], (int)qopts[b].n_opts);
        st = laya_build_sequence(state, q, model->cfg.max_len,
                                 model->cfg.head_max_len,
                                 &q_ids[b], &q_n[b], &q_mk[b], &q_nm[b]);
        hd_json_free(q);
        if (st != LAYA_OK) goto cleanup;
        if (q_nm[b] == 0 || qopts[b].n_opts != q_nm[b] || !qkeys[b]) {
            laya_set_error("option markers do not match rendered options for "
                           "'%s'", qids[b]);
            st = LAYA_ERR_PARSE;
            goto cleanup;
        }
        if ((int)q_n[b] > L) L = (int)q_n[b];
        if ((int)q_nm[b] > kmax) kmax = (int)q_nm[b];
    }

    /* ---- collate: input_ids [B,L], attention_mask [B,L], markers [B,kmax] ---- */
    const int R = B * L, M = B * kmax;
    const int pad = model->cfg.pad_token_id;
    long real_tokens = 0;

    ids64 = malloc((size_t)R * sizeof(int64_t));
    tok_valid = calloc((size_t)R, 1);
    mflat = malloc((size_t)M * sizeof(int64_t));
    mmask = calloc((size_t)M, 1);
    if (!ids64 || !tok_valid || !mflat || !mmask) {
        free(ids64); free(tok_valid); free(mflat); free(mmask);
        st = LAYA_ERR_OOM;
        goto cleanup;
    }
    for (int b = 0; b < B; b++) {
        for (int s = 0; s < L; s++) {
            int real = s < (int)q_n[b];
            ids64[(size_t)b * L + s] = real ? q_ids[b][s] : pad;
            tok_valid[(size_t)b * L + s] = real ? 1 : 0;
            if (real) real_tokens++;
        }
        for (int j = 0; j < kmax; j++) {
            int real = j < (int)q_nm[b];
            mflat[(size_t)b * kmax + j] =
                real ? (int64_t)(b * L + q_mk[b][j]) : (int64_t)(b * L);
            mmask[(size_t)b * kmax + j] = real ? 1 : 0;
        }
    }

    if (cudaMalloc(&ids_dev, (size_t)R * sizeof(int64_t)) != cudaSuccess ||
        cudaMalloc(&enc_dev, (size_t)R * (size_t)d * 2) != cudaSuccess ||
        cudaMalloc(&markers_dev, (size_t)M * sizeof(int64_t)) != cudaSuccess) {
        laya_set_error("cudaMalloc failed: %s",
                       cudaGetErrorString(cudaGetLastError()));
        st = LAYA_ERR_OOM;
        goto cleanup;
    }
    cudaMemcpy(ids_dev, ids64, (size_t)R * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(markers_dev, mflat, (size_t)M * sizeof(int64_t), cudaMemcpyHostToDevice);
    free(ids64); ids64 = NULL;

    laya_encoder enc;
    memset(&enc, 0, sizeof(enc));
    st = laya_encoder_forward_batch(&enc, model, ids_dev, B, L, tok_valid, enc_dev);
    if (st != LAYA_OK) {
        laya_set_error("encoder: %s", laya_encoder_last_error());
        laya_encoder_free(&enc);
        goto cleanup;
    }

    laya_decision dec;
    memset(&dec, 0, sizeof(dec));
    res = calloc((size_t)B, sizeof(laya_decision_result));
    if (!res) {
        laya_encoder_free(&enc);
        st = LAYA_ERR_OOM;
        goto cleanup;
    }
    st = laya_decision_forward_batch(&dec, model, enc_dev, B, L, markers_dev, kmax,
                                     mmask, qtype, tok_valid, res);
    if (st != LAYA_OK) {
        laya_set_error("decision head: %s", laya_decision_last_error());
        free(res); res = NULL;
        laya_decision_free(&dec);
        laya_encoder_free(&enc);
        goto cleanup;
    }

    const char *dbg = getenv("LAYA_DEBUG_PROBE");
    if (dbg && dbg[0]) {
        char path[512];
        snprintf(path, sizeof(path), "%s.enc_final.bin", dbg);
        size_t nb = (size_t)R * d * 2;
        uint8_t *host = malloc(nb);
        if (host) {
            cudaMemcpy(host, enc_dev, nb, cudaMemcpyDeviceToHost);
            FILE *f = fopen(path, "wb");
            if (f) { fwrite(host, 1, nb, f); fclose(f); }
            free(host);
        }
    }

    /* ---- copy into the caller-owned result array (wire friendly) ---- */
    out = calloc((size_t)B, sizeof(*out));
    if (!out) {
        free(res);
        laya_decision_free(&dec);
        laya_encoder_free(&enc);
        st = LAYA_ERR_OOM;
        goto cleanup;
    }
    for (int b = 0; b < B; b++) {
        const laya_decision_result *r = &res[b];
        laya_infer_result *o = &out[b];
        const int nv = (int)qopts[b].n_opts;   /* true (unpadded) options */
        o->qtype = r->qtype;
        o->n_options = nv;
        o->choice = r->choice;
        o->score = r->score;
        o->noul = r->noul;
        o->confidence = r->confidence;
        o->act_probability = r->act_probability;
        o->probs = malloc((size_t)o->n_options * sizeof(float));
        o->raw_logits = malloc((size_t)o->n_options * sizeof(float));
        o->option_keys = calloc((size_t)o->n_options, sizeof(char *));
        o->option_labels = calloc((size_t)o->n_options, sizeof(char *));
        if (!o->probs || !o->raw_logits || !o->option_keys || !o->option_labels) {
            laya_infer_results_free(out, B);
            free(res);
            laya_decision_free(&dec);
            laya_encoder_free(&enc);
            st = LAYA_ERR_OOM;
            goto cleanup;
        }
        memcpy(o->probs, r->probs, (size_t)o->n_options * sizeof(float));
        memcpy(o->raw_logits, r->raw_logits, (size_t)o->n_options * sizeof(float));
        for (int k = 0; k < o->n_options; k++) {
            o->option_keys[k] = qkeys[b][k] ? strdup(qkeys[b][k]) : NULL;
            if (qopts[b].opts[k]) o->option_labels[k] = strdup(qopts[b].opts[k]);
        }
    }

    free(res);
    laya_decision_free(&dec);
    laya_encoder_free(&enc);
    free(tok_valid);
    free(mflat);
    free(mmask);
    if (ids_dev) cudaFree(ids_dev);
    if (enc_dev) cudaFree(enc_dev);
    if (markers_dev) cudaFree(markers_dev);

    *out_res = out;
    *out_n = B;
    if (out_input_tokens) *out_input_tokens = real_tokens;
    st = LAYA_OK;

cleanup:
    if (ids64) free(ids64);
    if (st != LAYA_OK) {
        if (res) free(res);
        if (out) laya_infer_results_free(out, B);
        free(tok_valid);
        free(mflat);
        free(mmask);
        if (ids_dev) cudaFree(ids_dev);
        if (enc_dev) cudaFree(enc_dev);
        if (markers_dev) cudaFree(markers_dev);
    }
    for (int b = 0; b < B; b++) {
        free(q_ids ? q_ids[b] : NULL);
        free(q_mk ? q_mk[b] : NULL);
        free(qids ? qids[b] : NULL);
        if (qopts) laya_options_free(&qopts[b]);
        if (qkeys && qkeys[b]) {
            for (int k = 0; q_nm && k < (int)q_nm[b]; k++) free(qkeys[b][k]);
            free(qkeys[b]);
        }
    }
    free(q_ids);
    free(q_n);
    free(q_mk);
    free(q_nm);
    free(qtype);
    free(qids);
    free(qopts);
    free(qkeys);
    return st;
}

/* ------------------------------------------------------------------ */
/* `--infer` harness                                                   */
/* ------------------------------------------------------------------ */

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

    const hd_json *questions = hd_json_get(root, "questions");
    laya_infer_result *res = NULL;
    int n = 0;
    long tokens = 0;
    laya_status st = laya_infer_run(model, root, &res, &n, &tokens);
    if (st != LAYA_OK) {
        fprintf(stderr, "error: %s\n", laya_last_error());
        hd_json_free(root);
        return 1;
    }

    FILE *out = out_path ? fopen(out_path, "wb") : stdout;
    if (!out) {
        fprintf(stderr, "error: cannot write %s\n", out_path);
        laya_infer_results_free(res, n);
        hd_json_free(root);
        return 1;
    }

    fputs("{\n \"questions\": {\n", out);
    for (int b = 0; b < n; b++) {
        const laya_infer_result *r = &res[b];
        const char *qid = questions->u.object.keys[b];
        if (b) fputs(",\n", out);
        fputs("  \"", out);
        fputs(qid, out);
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

    laya_infer_results_free(res, n);
    hd_json_free(root);
    return 0;
}
