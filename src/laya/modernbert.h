#ifndef LAYA_MODERNBERT_H
#define LAYA_MODERNBERT_H

/*
 * ModernBERT encoder forward path for Laya.
 *
 * This is a direct, model-specific port of the original
 * `answerdotai/ModernBERT-large` encoder as instantiated by
 * `_reference/laya/laya/common.py`. It deliberately mirrors the torch
 * module structure instead of introducing a graph executor:
 *
 *   embeddings: norm(tok_embeddings(ids))
 *   layer:      h += attn(attn_norm(h))
 *               h += mlp(mlp_norm(h))
 *   final:      final_norm(h)
 *
 * All GEMM / RoPE / norm / activation work is delegated to the primitives
 * copied from o1.c; this file only sequences them and owns the workspace.
 */

#include <stdint.h>

#include "model.h"

/*
 * Persistent scratch for the encoder forward pass. Everything is allocated
 * once by laya_encoder_init and reused for every request, so the steady-state
 * forward contains no cudaMalloc/cudaFree (the o1.c residency policy).
 *
 * Layout notes:
 *  - `h` is the sequence-major activation [seq, hidden].
 *  - qkv holds the fused projection [seq, 3*hidden]; it is also reused for
 *    the Wi output [seq, 2*intermediate].
 *  - q/k/v are head-major [heads, seq, head_dim] because that is what the
 *    attention primitive consumes.
 */
typedef struct {
    int rows;                /* capacity in flattened rows: batch * item_seq */
    int item_seq;            /* capacity per item (sequence length)          */
    int hidden;
    int n_heads;
    int head_dim;
    int intermediate;

    void *h;                 /* [rows, hidden] bf16 */
    void *res;               /* [rows, hidden] bf16 */
    void *norm;              /* [rows, hidden] bf16 */
    void *qkv;               /* scratch [rows, max(3*hidden, 2*inter)] bf16 */
    int64_t scratch_bytes;   /* capacity of the qkv/Wi scratch slot */
    void *q;                 /* [n_heads, rows, head_dim] bf16 */
    void *k;
    void *v;
    void *attn_out;          /* [n_heads, rows, head_dim] -> merged */
    void *q_rot;             /* rotation output, separate from q/k    */
    void *k_rot;             /*   because the rope kernel is not safe
                              *   to run in place (it reads the pair) */
    void *attn_ctx;          /* [rows, hidden] bf16 (Wo output) */
    void *mlp_gate;          /* [rows, intermediate] bf16 */
    void *mlp_up;            /* [rows, intermediate] bf16 */
    void *mlp_act;           /* [rows, intermediate] bf16 */
    void *mlp_out;           /* [rows, hidden] bf16 */
    void *rope_cos;          /* [item_seq, head_dim] f32, global theta */
    void *rope_sin;
    void *rope_cos2;         /* [item_seq, head_dim] f32, sliding theta */
    void *rope_sin2;
    void *mask_full;         /* [batch, item_seq, item_seq] bf16, block-diagonal */
    void *mask_sliding;      /* [batch, item_seq, item_seq] bf16 window + diagonal */
    void *scores;            /* [heads, rows, item_seq] f32 attention scratch */
    void *probs;             /* [heads, rows, item_seq] bf16 */
    void *valid;             /* [rows] uint8 real-token flags (mask build)  */

    /* optional parity taps (allocated only when requested) */
    void *dump_emb;          /* [seq, hidden] after the embedding stage */
    void *dump_l0;           /* [seq, hidden] after layer 0 (global)     */
    void *dump_l1;           /* [seq, hidden] after layer 1 (sliding)    */
    void *dump_l15;          /* sparse encoder checkpoints               */
    void *dump_l27;

    int64_t bytes;
} laya_encoder_ws;

typedef struct {
    laya_encoder_ws ws;
    int initialized;
    unsigned dump_mask;      /* LAYA_DUMP_* taps, set before init */
} laya_encoder;

/* Validates that every tensor the forward dereferences exists with the exact
 * shape ModernBERT needs, so a layout bug fails the load, not the answer. */
laya_status laya_encoder_check_weights(const laya_model *m);

/* Human-readable reason for the last failing encoder call. */
const char *laya_encoder_last_error(void);

/* Parity taps: ask the encoder to keep a copy of the embedding stage and/or
 * the output of layer 0 / layer 1, so a single forward can be diffed against
 * the Python oracle's hook dump. Call before the first forward. */
enum {
    LAYA_DUMP_EMBEDDINGS = 1 << 0,
    LAYA_DUMP_LAYER0     = 1 << 1,
    LAYA_DUMP_LAYER1     = 1 << 2,
    LAYA_DUMP_LAYER15    = 1 << 3,
    LAYA_DUMP_LAYER27    = 1 << 4,
};
void laya_encoder_set_dump(laya_encoder *enc, unsigned mask);

/* Allocates the workspace for `batch` items of sequence length `seq`. The
 * activation dimension is flattened (rows = batch * seq); attention runs one
 * item at a time over [1, seq, seq] masks, which is what keeps the existing
 * batch-1 SDPA primitive usable. Idempotent for smaller sizes. */
laya_status laya_encoder_init(laya_encoder *enc, const laya_model *m, int batch,
                              int seq);

void laya_encoder_free(laya_encoder *enc);

/*
 * Runs the encoder on `batch` token rows already resident on the device.
 * `ids_dev` is int64 [batch*seq] (row-major, item-major), `attn_dev` is a host
 * uint8 [batch*seq] mask (1 = real token) used to pad the attention masks, and
 * `out_dev` receives bf16 [batch*seq, hidden].
 * `attn_dev` may be NULL, meaning every token is real.
 */
laya_status laya_encoder_forward_batch(laya_encoder *enc, const laya_model *m,
                                       const void *ids_dev, int batch, int seq,
                                       const uint8_t *attn_dev, void *out_dev);

/* Single-item convenience wrapper (batch = 1). */
laya_status laya_encoder_forward(laya_encoder *enc, const laya_model *m,
                                 const void *ids_dev, int seq, void *out_dev);

#endif /* LAYA_MODERNBERT_H */