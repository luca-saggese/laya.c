#ifndef LAYA_MODEL_H
#define LAYA_MODEL_H

/*
 * Laya model: GGUF metadata + resident weight bindings.
 *
 * The residency flow is copied from o1.c (hd_weights_to_device_gguf):
 *   open GGUF -> parse metadata -> tensor directory -> resident arena
 *   -> pinned staging upload -> bind native pointers -> keep resident.
 *
 * Laya-specific additions are limited to (a) reading the laya.* /
 * modernbert.* metadata into a plain config struct and (b) binding the
 * per-layer/per-head tensor pointers the forward pass needs.
 */

#include <stddef.h>
#include <stdint.h>

#include <cuda_runtime.h>

#include "laya.h"
#include "hd_gguf.h"
#include "hd_gemm.h"

#define LAYA_MAX_LAYERS 64
#define LAYA_MAX_HEAD_LAYERS 8
#define LAYA_TEMP_BUCKETS 12 /* 3 qtypes x 4 option-count buckets */

typedef struct {
    int hidden_size;
    int n_layers;
    int n_heads;
    int head_dim;
    int intermediate_size;
    int max_position_embeddings;
    int vocab_size;
    int local_attention;          /* config.local_attention (window = /2) */
    int sliding_window;           /* local_attention / 2 */
    int global_attn_every_n_layers;
    float layer_norm_eps;
    float global_rope_theta;
    float local_rope_theta;
    int cls_token_id;
    int sep_token_id;
    int mask_token_id;
    int pad_token_id;

    int max_len;
    int head_max_len;
    int n_head_layers;
    int n_act;
    int n_types;                  /* type_emb rows */
    float temperature[3];
    float temperature_by_options[LAYA_TEMP_BUCKETS];
    uint8_t layer_is_global[LAYA_MAX_LAYERS];
} laya_config;

typedef struct {
    /* ---- resident device storage (one arena, as in o1.c) ---- */
    void *arena_ptr;
    int64_t arena_bytes;
    int64_t n_allocs;
    int64_t device_bytes;
    cudaStream_t upload_stream;

    /* ---- device identity ---- */
    int device_id;
    char device_name[256];
    int compute_major;
    int compute_minor;
    int64_t device_mem_bytes;

    /* ---- persistent GEMM runtime (cuBLAS/cuBLASLt) ---- */
    struct laya_gemm_runtime *gemm;
} laya_cuda_runtime;

typedef struct {
    void *ptr;
    int64_t nbytes;
    char name[160];
} laya_tensor;

typedef struct {
    laya_tensor attn_norm;
    laya_tensor mlp_norm;
    laya_tensor Wqkv;
    laya_tensor Wo;
    laya_tensor Wi;
    laya_tensor mlp_Wo;
    int is_global;
} laya_layer_weights;

typedef struct {
    laya_tensor norm1_w, norm1_b;
    laya_tensor norm2_w, norm2_b;
    laya_tensor linear1_w, linear1_b;
    laya_tensor linear2_w, linear2_b;
    laya_tensor in_proj_w, in_proj_b;
    laya_tensor out_proj_w, out_proj_b;
} laya_head_layer_weights;

typedef struct {
    laya_tensor scorer0_w, scorer0_b;
    laya_tensor scorer1_w, scorer1_b;
    laya_tensor scorer3_w, scorer3_b;
    laya_tensor act0_w, act0_b;
    laya_tensor act2_w, act2_b;
} laya_decision_weights;

typedef struct {
    /* config */
    laya_config cfg;

    /* host-side tensor directory (kept for inspection / --inspect) */
    laya_gguf_file gguf;

    /* bindings */
    laya_tensor tok_embeddings;
    laya_tensor emb_norm;
    laya_tensor final_norm;
    laya_layer_weights layers[LAYA_MAX_LAYERS];
    laya_tensor type_emb;
    laya_head_layer_weights head[LAYA_MAX_HEAD_LAYERS];
    laya_decision_weights decision;

    laya_cuda_runtime cuda;
} laya_model;

/* Opens the GGUF, parses metadata and loads every weight into a single
 * resident CUDA arena. The model stays resident until laya_model_free. */
laya_status laya_model_load(const char *gguf_path, int device_id,
                            laya_model *out);

/* Prints the resident-model summary used by the `--inspect` smoke check. */
void laya_model_report(const laya_model *m);

void laya_model_free(laya_model *m);

const char *laya_model_last_error(void);

#endif /* LAYA_MODEL_H */
