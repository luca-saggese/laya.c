#define _POSIX_C_SOURCE 200809L

/*
 * `--encode`: run the native ModernBERT encoder on one request and dump the
 * hidden states the Python oracle can also dump, so the two can be diffed
 * entry by entry.
 *
 * This is a parity harness, not part of the inference path: it is the only
 * place that copies activations back to the host.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>

#include "hd_cuda.h"
#include "hd_json.h"
#include "laya.h"
#include "laya_sequence.h"
#include "laya_tokenizer.h"
#include "model.h"
#include "modernbert.h"

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *text = malloc((size_t)sz + 1);
    if (!text || fread(text, 1, (size_t)sz, f) != (size_t)sz) {
        fclose(f);
        free(text);
        return NULL;
    }
    fclose(f);
    text[sz] = '\0';
    return text;
}

/* Python repr of a float32: shortest form that round-trips. */
static void print_f32(FILE *out, float v) {
    if (v != v) { fputs("NaN", out); return; }
    if (v == (float)1e300 * (float)1e300) { fputs("Infinity", out); return; }
    if (v == -(float)1e300 * (float)1e300) { fputs("-Infinity", out); return; }
    fprintf(out, "%.9g", (double)v);
}

/* Writes a [rows, cols] bf16 device buffer as a JSON array of arrays. */
static void dump_matrix(FILE *out, const void *dev, int rows, int cols) {
    size_t n = (size_t)rows * cols;
    uint16_t *host = malloc(n * 2);
    if (!host) { fputs("null", out); return; }
    cudaMemcpy(host, dev, n * 2, cudaMemcpyDeviceToHost);
    fputs("[", out);
    for (int r = 0; r < rows; r++) {
        fputs(r ? ",\n  [" : "  [", out);
        for (int c = 0; c < cols; c++) {
            if (c) fputs(", ", out);
            print_f32(out, laya_bf16_to_f32(host[(size_t)r * cols + c]));
        }
        fputs("]", out);
    }
    fputs("]", out);
    free(host);
}

int encode_request(const laya_model *model, const char *request_path,
                   const char *dump_spec, const char *out_path) {
    if (!request_path) {
        fprintf(stderr, "error: --encode requires --request PATH\n");
        return 2;
    }
    char *text = read_file(request_path);
    if (!text) {
        fprintf(stderr, "error: cannot read %s\n", request_path);
        return 1;
    }
    const char *err = NULL;
    hd_json *root = hd_json_parse(text, &err);
    free(text);
    if (!root) {
        fprintf(stderr, "error: bad request: %s\n", err ? err : "parse");
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
    hd_json *q = laya_question_internal(questions->u.object.values[0]);
    if (!q) {
        fprintf(stderr, "error: bad question %s\n", qid);
        hd_json_free(root);
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

    unsigned mask = 0;
    int want_final = 0;
    if (dump_spec) {
        char *spec = strdup(dump_spec), *save = NULL;
        for (char *t = strtok_r(spec, ",", &save); t; t = strtok_r(NULL, ",", &save)) {
            while (*t == ' ') t++;
            if (strcmp(t, "embeddings") == 0) mask |= LAYA_DUMP_EMBEDDINGS;
            else if (strcmp(t, "layer:0") == 0) mask |= LAYA_DUMP_LAYER0;
            else if (strcmp(t, "layer:1") == 0) mask |= LAYA_DUMP_LAYER1;
            else if (strcmp(t, "layer:15") == 0) mask |= LAYA_DUMP_LAYER15;
            else if (strcmp(t, "layer:27") == 0) mask |= LAYA_DUMP_LAYER27;
            else if (strcmp(t, "final") == 0) want_final = 1;
            else fprintf(stderr, "warning: unknown dump '%s'\n", t);
        }
        free(spec);
    }

    /* laya_gather_rows takes int64 ids, matching the o1.c embedding path. */
    int64_t *ids64 = malloc(n * sizeof(int64_t));
    for (size_t i = 0; i < n; i++) ids64[i] = ids[i];
    void *ids_dev = NULL;
    void *out_dev = NULL;
    cudaMalloc(&ids_dev, n * sizeof(int64_t));
    cudaMalloc(&out_dev, n * (size_t)model->cfg.hidden_size * 2);
    cudaMemcpy(ids_dev, ids64, n * sizeof(int64_t), cudaMemcpyHostToDevice);
    free(ids64);

    laya_encoder enc;
    memset(&enc, 0, sizeof(enc));
    laya_encoder_set_dump(&enc, mask);
    st = laya_encoder_forward(&enc, model, ids_dev, (int)n, out_dev);
    if (st != LAYA_OK) {
        fprintf(stderr, "error: encoder: %s\n", laya_encoder_last_error());
        cudaFree(ids_dev);
        cudaFree(out_dev);
        free(ids);
        free(markers);
        free(qid);
        return 1;
    }

    FILE *out = out_path ? fopen(out_path, "wb") : stdout;
    if (!out) {
        fprintf(stderr, "error: cannot write %s\n", out_path);
        cudaFree(ids_dev);
        cudaFree(out_dev);
        free(ids);
        free(markers);
        free(qid);
        return 1;
    }

    int d = model->cfg.hidden_size;
    fputs("{\n \"items\": [{\"question_id\": \"", out);
    fputs(qid, out);
    fputs("\", \"input_ids\": [", out);
    for (size_t i = 0; i < n; i++) fprintf(out, "%s%d", i ? ", " : "", ids[i]);
    fputs("], \"marker_pos\": [", out);
    for (size_t i = 0; i < nm; i++) fprintf(out, "%s%d", i ? ", " : "", markers[i]);
    fputs("]}],\n", out);

    if (enc.ws.dump_emb) {
        fputs(" \"embeddings\": [", out);
        dump_matrix(out, enc.ws.dump_emb, (int)n, d);
        fputs("],\n", out);
    }
    if (enc.ws.dump_l0 || enc.ws.dump_l1 || enc.ws.dump_l15 || enc.ws.dump_l27) {
        fputs(" \"layers\": {", out);
        int first = 1;
        if (enc.ws.dump_l0) {
            fputs("\"0\": [", out);
            dump_matrix(out, enc.ws.dump_l0, (int)n, d);
            fputs("]", out);
            first = 0;
        }
        if (enc.ws.dump_l1) {
            fputs(first ? "\"1\": [" : ", \"1\": [", out);
            dump_matrix(out, enc.ws.dump_l1, (int)n, d);
            fputs("]", out);
            first = 0;
        }
        if (enc.ws.dump_l15) {
            fputs(first ? "\"15\": [" : ", \"15\": [", out);
            dump_matrix(out, enc.ws.dump_l15, (int)n, d);
            fputs("]", out);
            first = 0;
        }
        if (enc.ws.dump_l27) {
            fputs(first ? "\"27\": [" : ", \"27\": [", out);
            dump_matrix(out, enc.ws.dump_l27, (int)n, d);
            fputs("]", out);
            first = 0;
        }
        fputs("},\n", out);
    }
    if (want_final) {
        fputs(" \"final\": [", out);
        dump_matrix(out, out_dev, (int)n, d);
        fputs("]\n", out);
    }
    fputs("}\n", out);
    if (out_path) fclose(out);

    laya_encoder_free(&enc);
    cudaFree(ids_dev);
    cudaFree(out_dev);
    free(ids);
    free(markers);
    free(qid);
    return 0;
}