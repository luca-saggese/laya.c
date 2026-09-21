#ifndef LAYA_DECISION_H
#define LAYA_DECISION_H

/*
 * Laya typed decision head forward path.
 *
 * This is a direct port of `_reference/laya/laya/common.py:DecisionModel`,
 * mirroring the torch module structure:
 *
 *   h = encoder_final
 *   h = h + type_emb(qtype)[:, None, :]
 *   h = head_layer_0(h, src_key_padding_mask=pad)
 *   h = head_layer_1(h, src_key_padding_mask=pad)
 *   m = gather(h, marker_pos)
 *   logits = scorer(m)                       # LayerNorm, Linear, GELU, Linear
 *   logits = masked_fill(~marker_mask, -1e4)
 *
 * Post-processing (softmax, temperature calibration, top-2 / entropy features,
 * choice / score / noul answers) is small scalar arithmetic and is done on the
 * host, exactly as the Python agent does.
 *
 * Every GPU operation is an existing primitive (LayerNorm, GEMM, head
 * split/merge, attention, exact GELU, gather). This file is orchestration:
 * it owns the workspace and sequences the primitives, and adds nothing but the
 * two genuinely missing elementwise kernels (ReLU, broadcast type add).
 *
 * IMPORTANT: `head` is a stock `nn.TransformerEncoderLayer`, so it is a
 * pre-norm, standard-MHA block with a 4*d FFN and ReLU activation. It is NOT a
 * ModernBERT block: no RoPE, no GeGLU, no sliding window, no causal mask.
 */

#include <stdint.h>

#include "model.h"

/* Results of the decision head for a single question, matching the Python
 * agent's answer semantics (agent.py). */
typedef struct {
    int qtype;               /* 0 choice, 1 score, 2 noul (QTYPES)          */
    float raw_logits[64];    /* scorer output, invalid options masked       */
    int n_options;           /* number of real (unpadded) options           */
    int valid[64];           /* marker_mask                                 */

    float probs[64];         /* temperature-calibrated probabilities        */
    float confidence;        /* confidence_from_probs                       */
    int choice;              /* argmax, or -1 when qtype != choice          */
    float score;             /* expected value, when qtype == score         */
    float noul;              /* p[1], when qtype == noul                    */

    float act_probability;   /* softmax(act_logits)[0]                      */
    float act_logits[2];
} laya_decision_result;

/* ---- workspace ---------------------------------------------------- */

typedef struct {
    int rows;                /* flattened rows: batch * item_seq            */
    int item_seq;            /* tokens per item                             */
    int hidden;
    int n_heads;
    int head_dim;
    int ffn;                 /* 4 * hidden                                  */
    int n_head_layers;       /* capacity in head layers                     */
    int d_act;               /* hidden + 4                                  */
    int n_batch;             /* capacity in questions                       */

    void *h;                 /* [rows, hidden] residual stream (bf16)        */
    void *norm;              /* [rows, hidden] LayerNorm output              */
    void *qkv;               /* [rows, 3*hidden] fused in_proj               */
    void *q;                 /* [heads, rows, hd] head-major                 */
    void *k;
    void *v;
    void *attn_out;          /* [heads, rows, hd] attention output           */
    void *merged;            /* [rows, hidden] head_merge output             */
    void *proj;              /* [rows, hidden] out_proj / linear2 output     */
    void *ff;                /* [rows, ffn] linear1 output                   */
    void *ff_act;            /* [rows, ffn] ReLU output                      */
    void *mask;              /* [batch, item_seq, item_seq] bf16 pad mask    */
    void *valid;             /* [rows] uint8 real-token flags                */
    void *scores;            /* fp32 attention score scratch                 */
    void *probs;             /* [heads, rows, item_seq] bf16 softmax probs   */

    void *m_hidden;          /* [n_markers_cap, hidden] gather output        */
    int n_markers_cap;       /* capacity in markers: batch * kmax            */
    void *m_idx_dev;         /* int64 [n_markers_cap] flattened row indices  */
    void *type_vec;          /* [batch, hidden] gathered type embeddings     */
    void *type_idx_dev;      /* int64 [n_batch] qtype indices                */
    void *act_vec;           /* [d_act] pooled + 4 features, fp32->bf16      */
    void *act1;              /* [256] action hidden                          */
    void *act1_act;          /* [256] GELU output                            */

    int64_t bytes;
} laya_decision_ws;

typedef struct {
    laya_decision_ws ws;
    int initialized;
} laya_decision;

/* Validates every decision-head tensor the forward dereferences. */
laya_status laya_decision_check_weights(const laya_model *m);

const char *laya_decision_last_error(void);

/* Allocates the workspace for (batch x seq) tokens and (batch x kmax) markers.
 * Idempotent for smaller sizes. */
laya_status laya_decision_init(laya_decision *dec, const laya_model *m,
                               int batch, int seq, int n_markers);

void laya_decision_free(laya_decision *dec);

/*
 * Runs the decision head on an encoder output already resident on the device.
 * `enc_out_dev` is bf16 [batch*seq, hidden] and is not modified: the
 * type-embedding add writes into the decision workspace.
 *
 * `markers_dev` is int64 [batch*kmax] with the FLATTENED row index
 * (b*seq + marker_pos), `marker_mask` is a host uint8 array of length
 * batch*kmax (1 = real option, 0 = padding), `qtypes` is [batch] and
 * `tok_valid` is a host uint8 array of length batch*seq used as the head's
 * src_key_padding_mask (NULL = every token of every item is valid).
 * `out` receives one result per question.
 */
laya_status laya_decision_forward_batch(laya_decision *dec, const laya_model *m,
                                        const void *enc_out_dev, int batch,
                                        int seq, const void *markers_dev,
                                        int kmax, const uint8_t *marker_mask,
                                        const int32_t *qtypes,
                                        const uint8_t *tok_valid,
                                        laya_decision_result *out);

/* Single-question convenience wrapper (batch = 1, marker positions relative). */
laya_status laya_decision_forward(laya_decision *dec, const laya_model *m,
                                  const void *enc_out_dev, int seq,
                                  const void *markers_dev, int n_markers,
                                  const uint8_t *marker_mask, int qtype,
                                  const uint8_t *tok_valid,
                                  laya_decision_result *out);

#endif /* LAYA_DECISION_H */
