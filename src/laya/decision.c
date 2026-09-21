/*
 * Laya typed decision head: host orchestration over the o1.c-derived CUDA
 * primitives.
 *
 * Mirrors `_reference/laya/laya/common.py:DecisionModel` exactly:
 *
 *   h  = enc_out + type_emb(qtype)[None, :]
 *   h  = head_layer(h, src_key_padding_mask=pad)     x n_head_layers
 *   m  = gather(h, marker_pos)
 *   lo = scorer(m)                                   LN / GEMM / GELU / GEMM
 *   lo = masked_fill(~marker_mask, -1e4)
 *
 * The head layer is a stock `nn.TransformerEncoderLayer` (pre-norm, standard
 * MHA with a fused in_proj, 4*d FFN, ReLU, padding mask, no causal mask).
 * It is NOT a ModernBERT block: no RoPE, no GeGLU, no sliding window.
 *
 * REUSE TABLE (o1.c primitive -> decision use):
 *   laya_layernorm     (hd_vision_layernorm)  norm1 / norm2 / scorer.0
 *   laya_linear        (hd_linear / cuBLASLt) in_proj, out_proj, ff, scorer, act
 *   laya_residual_add  (hd_residual_add)      attention + FFN residuals
 *   laya_qkv_split     (hd_head_split family) fused in_proj [S,3d] -> [S,3,H,hd]
 *   laya_head_split    (hd_head_split)        per-projection [S,H,hd] -> [H,S,hd]
 *   laya_head_merge    (hd_head_merge)        [H,S,hd] -> [S,H,hd]
 *   laya_attention_ws  (hd SDPA contract)     non-causal MHA with additive mask
 *   laya_gelu          (hd_vision_gelu_exact) scorer + action-head GELU
 *   laya_gather_rows   (hd_gather_rows)       type_emb(qtype) lookup
 *   laya_gather_pos    (hd_gather_pos)        marker gather (row gather)
 * NEW device code (nothing equivalent exists in o1.c):
 *   laya_relu          FFN activation (TransformerEncoderLayer default)
 *   laya_bcast_add     h + type_emb(qtype)[None, :]
 *
 * The fused in_proj GEMM output [S, 3d] holds Q|K|V as three contiguous
 * d-wide column blocks (torch chunks in_proj_weight along dim 0), which is
 * exactly the [S,3,H,hd] layout laya_qkv_split reads.
 *
 * Post-processing (temperature calibration, softmax, top-2/entropy features,
 * choice / score / noul answers) is scalar arithmetic on the host, exactly as
 * the Python agent does; it is negligible next to the encoder.
 */

#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>

#include "decision.h"
#include "hd_cuda.h"
#include "modernbert.h"

static __thread char dec_errbuf[512];

static laya_status dec_fail(laya_status st, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(dec_errbuf, sizeof(dec_errbuf), fmt, ap);
    va_end(ap);
    return st;
}

const char *laya_decision_last_error(void) { return dec_errbuf; }

/* Wraps a primitive, reporting the CUDA error state of that op only.
 * `bias` is a device bf16 pointer, applied by the GEMM bias epilogue; the
 * stock TransformerEncoderLayer has biases on every linear, so all callers
 * pass one.
 *
 * cuBLASLt has no valid algorithm for the tiny scorer/action shapes
 * (M = marker count, N up to 1024), so small-M GEMMs take the verified
 * reference path instead. The scorer/action heads are a negligible share of
 * the runtime, so this is not a performance concern. */
static laya_status linear(const laya_tensor *w, const void *bias,
                          const void *x, void *y, int M, int N, int K) {
    laya_cuda_clear_error();
    if (M < 16) {
        laya_linear_reference(x, w->ptr, bias, y, M, N, K, 1);
    } else {
        laya_linear(x, w->ptr, bias, y, M, N, K, 1);
    }
    const char *err = laya_cuda_last_error();
    if (err && err[0]) return dec_fail(LAYA_ERR_RUNTIME, "linear %s: %s", w->name, err);
    return LAYA_OK;
}

static laya_status check(const char *op) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        return dec_fail(LAYA_ERR_RUNTIME, "%s: %s", op, cudaGetErrorString(e));
    return LAYA_OK;
}

/* Parity tap. Enabled by LAYA_DEBUG_PROBE=<prefix>; slot disambiguates the
 * two head layers, which share one bake-off filename otherwise. */
static void dump_dev(const char *fmt, const void *dev, size_t nbytes, int slot) {
    const char *dbg = getenv("LAYA_DEBUG_PROBE");
    if (!dbg || !dbg[0]) return;
    cudaDeviceSynchronize();
    uint8_t *host = malloc(nbytes);
    if (!host) return;
    if (cudaMemcpy(host, dev, nbytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        free(host);
        return;
    }
    char path[512];
    snprintf(path, sizeof(path), fmt, dbg, slot);
    FILE *f = fopen(path, "wb");
    if (f) { fwrite(host, 1, nbytes, f); fclose(f); }
    free(host);
}

/* ---- lifecycle ---------------------------------------------------- */

void laya_decision_free(laya_decision *dec) {
    if (!dec) return;
    if (dec->ws.h) {
        cudaFree(dec->ws.h);
        memset(&dec->ws, 0, sizeof(dec->ws));
    }
    dec->initialized = 0;
}

laya_status laya_decision_check_weights(const laya_model *m) {
    const laya_config *c = &m->cfg;
    if (!m->type_emb.ptr) return dec_fail(LAYA_ERR_MISSING, "missing decision.type_emb.weight");
    if (m->type_emb.shape[1] != c->hidden_size)
        return dec_fail(LAYA_ERR_MISSING, "type_emb cols %lld != %d",
                        (long long)m->type_emb.shape[1], c->hidden_size);
    for (int i = 0; i < c->n_head_layers; i++) {
        const laya_head_layer_weights *w = &m->head[i];
        if (!w->norm1_w.ptr || !w->norm1_b.ptr || !w->norm2_w.ptr || !w->norm2_b.ptr ||
            !w->linear1_w.ptr || !w->linear1_b.ptr || !w->linear2_w.ptr || !w->linear2_b.ptr ||
            !w->in_proj_w.ptr || !w->in_proj_b.ptr || !w->out_proj_w.ptr || !w->out_proj_b.ptr)
            return dec_fail(LAYA_ERR_MISSING, "decision head layer %d incomplete", i);
        if (w->in_proj_w.shape[0] != 3 * c->hidden_size || w->in_proj_w.shape[1] != c->hidden_size)
            return dec_fail(LAYA_ERR_MISSING, "head %d in_proj [%lld,%lld] != [%d,%d]", i,
                            (long long)w->in_proj_w.shape[0], (long long)w->in_proj_w.shape[1],
                            3 * c->hidden_size, c->hidden_size);
        if (w->linear1_w.shape[0] != 4 * c->hidden_size)
            return dec_fail(LAYA_ERR_MISSING, "head %d linear1 out %lld != %d", i,
                            (long long)w->linear1_w.shape[0], 4 * c->hidden_size);
        if (w->linear2_w.shape[0] != c->hidden_size || w->linear2_w.shape[1] != 4 * c->hidden_size)
            return dec_fail(LAYA_ERR_MISSING, "head %d linear2 [%lld,%lld] != [%d,%d]", i,
                            (long long)w->linear2_w.shape[0], (long long)w->linear2_w.shape[1],
                            c->hidden_size, 4 * c->hidden_size);
        if (w->out_proj_w.shape[0] != c->hidden_size || w->out_proj_w.shape[1] != c->hidden_size)
            return dec_fail(LAYA_ERR_MISSING, "head %d out_proj [%lld,%lld] != [%d,%d]", i,
                            (long long)w->out_proj_w.shape[0], (long long)w->out_proj_w.shape[1],
                            c->hidden_size, c->hidden_size);
    }
    const laya_decision_weights *dw = &m->decision;
    if (!dw->scorer0_w.ptr || !dw->scorer0_b.ptr || !dw->scorer1_w.ptr || !dw->scorer1_b.ptr ||
        !dw->scorer3_w.ptr || !dw->scorer3_b.ptr || !dw->act0_w.ptr || !dw->act0_b.ptr ||
        !dw->act2_w.ptr || !dw->act2_b.ptr)
        return dec_fail(LAYA_ERR_MISSING, "decision scorer/act_head tensors incomplete");
    if (dw->scorer0_w.shape[0] != c->hidden_size ||
        dw->scorer1_w.shape[0] != c->hidden_size || dw->scorer1_w.shape[1] != c->hidden_size ||
        dw->scorer3_w.shape[0] != 1 || dw->scorer3_w.shape[1] != c->hidden_size)
        return dec_fail(LAYA_ERR_MISSING, "scorer shapes do not match hidden %d", c->hidden_size);
    if (dw->act0_w.shape[0] != 256 || dw->act0_w.shape[1] != c->hidden_size + 4)
        return dec_fail(LAYA_ERR_MISSING, "act_head.0 [%lld,%lld] != [256,%d]",
                        (long long)dw->act0_w.shape[0], (long long)dw->act0_w.shape[1],
                        c->hidden_size + 4);
    if (dw->act2_w.shape[0] != c->n_act || dw->act2_w.shape[1] != 256)
        return dec_fail(LAYA_ERR_MISSING, "act_head.2 [%lld,%lld] != [%d,256]",
                        (long long)dw->act2_w.shape[0], (long long)dw->act2_w.shape[1], c->n_act);
    return LAYA_OK;
}

laya_status laya_decision_init(laya_decision *dec, const laya_model *m,
                               int batch, int seq, int n_markers) {
    if (seq <= 0 || batch <= 0)
        return dec_fail(LAYA_ERR_RUNTIME, "decision_init: batch=%d seq=%d", batch, seq);
    if (n_markers <= 0) n_markers = 1;
    if (n_markers > 1024) return dec_fail(LAYA_ERR_RUNTIME, "too many markers: %d", n_markers);
    if (dec->initialized && dec->ws.item_seq >= seq && dec->ws.n_batch >= batch &&
        dec->ws.n_markers_cap >= n_markers && dec->ws.n_head_layers >= m->cfg.n_head_layers)
        return LAYA_OK;
    laya_decision_free(dec);

    const laya_config *c = &m->cfg;
    int d = c->hidden_size, H = c->n_heads, hd = c->head_dim;
    int ffn = 4 * d;                 /* head FFN width: 4096 for d=1024 */
    size_t S = (size_t)seq, B = (size_t)batch, R = B * S, M = (size_t)n_markers;

    size_t bf   = R * (size_t)d * 2;          /* one [R, d] bf16 activation    */
    size_t ffb  = R * (size_t)ffn * 2;        /* one [R, ffn] bf16 activation  */
    size_t qkvb = R * (size_t)(3 * d) * 2;    /* fused in_proj output [R, 3d]  */
    size_t hmaj = (size_t)H * R * (size_t)hd * 2;  /* head-major activation    */
    size_t maskb = B * S * S * 2;             /* additive pad mask [B,S,S]     */
    size_t actvec = (((size_t)d + 4) + 1) & ~(size_t)1; /* bf16 [d+4], aligned */

    /*
     * Workspace capacity: the largest single consumer here is the FFN
     * (R * 4d), which is wider than the fused QKV (R * 3d). Sizing by the
     * FFN is what keeps this from repeating the encoder's Wi overflow.
     */
    size_t bytes = 0;
    bytes += bf * 4;                 /* h, norm, merged, proj     */
    bytes += qkvb;                   /* qkv                       */
    bytes += hmaj * 4;               /* q, k, v, attn_out         */
    bytes += ffb * 2;                /* ff, ff_act                */
    bytes += maskb;                  /* additive attention mask   */
    bytes += (size_t)H * R * S * 4;  /* fp32 attention scores     */
    bytes += (size_t)H * R * S * 2;  /* bf16 attention probs      */
    bytes += M * (size_t)d * 2;      /* marker hidden             */
    bytes += M * 8;                  /* int64 flattened row idx   */
    bytes += B * (size_t)d * 2;      /* type_vec                  */
    bytes += actvec;                 /* act_vec [d+4] bf16        */
    bytes += 256 * 2 * 2;            /* act1, act1_act [256]      */
    bytes += B * 8 + 8;              /* int64 type indices        */
    bytes += ((R + 255) & ~(size_t)255); /* uint8 valid flags     */

    size_t total = ((bytes + 255) & ~(size_t)255) + 256;

    void *base = NULL;
    cudaError_t e = cudaMalloc(&base, total);
    if (e != cudaSuccess)
        return dec_fail(LAYA_ERR_OOM, "decision workspace cudaMalloc(%zu): %s", total,
                        cudaGetErrorString(e));

    laya_decision_ws *ws = &dec->ws;
    memset(ws, 0, sizeof(*ws));
    ws->rows = (int)R;
    ws->item_seq = seq;
    ws->hidden = d;
    ws->n_heads = H;
    ws->head_dim = hd;
    ws->ffn = ffn;
    ws->n_head_layers = c->n_head_layers;
    ws->d_act = d + 4;
    ws->n_markers_cap = n_markers;
    ws->n_batch = batch;
    ws->bytes = (int64_t)total;

    size_t vbytes = (R + 255) & ~(size_t)255;
    uint8_t *p = (uint8_t *)base;
    ws->h = p;             p += bf;
    ws->norm = p;          p += bf;
    ws->merged = p;        p += bf;
    ws->proj = p;          p += bf;
    ws->qkv = p;           p += qkvb;
    ws->q = p;             p += hmaj;
    ws->k = p;             p += hmaj;
    ws->v = p;             p += hmaj;
    ws->attn_out = p;      p += hmaj;
    ws->ff = p;            p += ffb;
    ws->ff_act = p;        p += ffb;
    ws->mask = p;          p += maskb;
    ws->scores = p;        p += (size_t)H * R * S * 4;
    ws->probs = p;         p += (size_t)H * R * S * 2;
    ws->m_hidden = p;      p += M * (size_t)d * 2;
    ws->type_vec = p;      p += B * (size_t)d * 2;
    ws->act_vec = p;       p += actvec;
    ws->act1 = p;          p += 256 * 2;
    ws->act1_act = p;      p += 256 * 2;
    p = (uint8_t *)(((uintptr_t)p + 7) & ~(uintptr_t)7);
    ws->type_idx_dev = p;  p += B * 8 + 8;
    ws->m_idx_dev = p;     p += M * 8;
    ws->valid = p;         p += vbytes;

    if ((size_t)(p - (uint8_t *)base) > total)
        return dec_fail(LAYA_ERR_RUNTIME, "decision workspace layout overflow");

    dec->initialized = 1;
    return LAYA_OK;
}

/* ---- additive padding mask ---------------------------------------- */

/*
 * `nn.TransformerEncoderLayer(src_key_padding_mask=pad)` masks padded KEYS for
 * every query. torch converts the bool mask to a float additive bias of -inf
 * on masked keys and 0.0 elsewhere; the fp32 softmax then gives exp(-inf-mx)=0.
 *
 * Built with the encoder's block-diagonal mask primitive so that item b never
 * attends to item c's keys, which is exactly the structure the oracle's
 * [B,1,L,L] bias has.
 */
static laya_status build_pad_mask(laya_decision_ws *ws, int batch, int seq) {
    laya_attn_mask_batch(ws->mask, ws->valid, batch, seq, seq, 0);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        return dec_fail(LAYA_ERR_RUNTIME, "pad mask: %s", cudaGetErrorString(e));
    return LAYA_OK;
}

/* ---- one stock TransformerEncoderLayer ---------------------------- */

static laya_status head_layer(laya_decision *dec, const laya_model *m,
                              const laya_head_layer_weights *w, int layer_index,
                              int batch, int seq) {
    laya_decision_ws *ws = &dec->ws;
    const laya_config *c = &m->cfg;
    const int d = c->hidden_size, H = c->n_heads, hd = c->head_dim;
    const int S = seq, B = batch, R = B * S, ffn = 4 * d;
    const float eps = c->layer_norm_eps;
    const float scale = 1.0f / sqrtf((float)hd);
    laya_status st;

    /* ---- pre-norm + self-attention ---- */
    laya_layernorm(ws->h, w->norm1_w.ptr, w->norm1_b.ptr, ws->norm, R, d, eps);
    if ((st = check("head norm1")) != LAYA_OK) return st;

    /* Fused QKV: in_proj_weight is [3d, d], so M=R, N=3d, K=d. */
    if ((st = linear(&w->in_proj_w, w->in_proj_b.ptr, ws->norm, ws->qkv, R, 3 * d, d)) != LAYA_OK) return st;

    if ((st = build_pad_mask(ws, B, S)) != LAYA_OK) return st;

    /* Layout and SDPA are per item: head-major buffers and the [S,S] mask are
     * indexed item-locally, so each item is passed as a contiguous slice. */
    for (int b = 0; b < B; b++) {
        const size_t off_qkv = (size_t)b * S * 3 * d * 2;
        const size_t off_h = (size_t)b * H * S * hd * 2;
        const size_t off_msk = (size_t)b * S * S * 2;
        const size_t off_sc = (size_t)b * H * S * S;
        /* q, k, v = fused.chunk(3, -1) -> the verified [S,3,H,hd] split. */
        laya_qkv_split((uint8_t *)ws->qkv + off_qkv, (uint8_t *)ws->q + off_h,
                       (uint8_t *)ws->k + off_h, (uint8_t *)ws->v + off_h, S, H, hd);
        if ((st = check("head qkv split")) != LAYA_OK) return st;
        laya_attention_ws((uint8_t *)ws->q + off_h, (uint8_t *)ws->k + off_h,
                          (uint8_t *)ws->v + off_h,
                          (const uint8_t *)ws->mask + off_msk,
                          (float *)ws->scores + off_sc, ws->probs + off_sc,
                          (uint8_t *)ws->attn_out + off_h, H, S, hd, scale);
        if ((st = check("head attention")) != LAYA_OK) return st;
        laya_head_merge((uint8_t *)ws->attn_out + off_h,
                        (uint8_t *)ws->merged + (size_t)b * S * d * 2, S, H, hd);
    }
    if ((st = linear(&w->out_proj_w, w->out_proj_b.ptr, ws->merged, ws->proj, R, d, d)) != LAYA_OK) return st;

    laya_residual_add(ws->h, ws->proj, ws->h, (size_t)R * d);
    if ((st = check("head attn residual")) != LAYA_OK) return st;

    /* ---- pre-norm + FFN: Linear(4d) -> ReLU -> Linear(d) ---- */
    laya_layernorm(ws->h, w->norm2_w.ptr, w->norm2_b.ptr, ws->norm, R, d, eps);
    if ((st = check("head norm2")) != LAYA_OK) return st;

    if ((st = linear(&w->linear1_w, w->linear1_b.ptr, ws->norm, ws->ff, R, ffn, d)) != LAYA_OK) return st;
    laya_relu(ws->ff, ws->ff_act, (size_t)R * (size_t)ffn);
    if ((st = check("head relu")) != LAYA_OK) return st;
    if ((st = linear(&w->linear2_w, w->linear2_b.ptr, ws->ff_act, ws->proj, R, d, ffn)) != LAYA_OK) return st;

    laya_residual_add(ws->h, ws->proj, ws->h, (size_t)R * d);
    if ((st = check("head ffn residual")) != LAYA_OK) return st;

    if (layer_index < 2)
        dump_dev("%s.head%d", ws->h, (size_t)((int64_t)R * d * 2), layer_index);
    return LAYA_OK;
}

/* ---- host post-processing (agent.py answer semantics) ------------- */

/*
 * temperature_by_options is indexed by qtype then option-count bucket:
 *   [choice:2, choice:3-5, choice:6-10, choice:11+,
 *    score:2,  score:3-5,  score:6-10,  score:11+,
 *    noul:2,   noul:3-5,   noul:6-10,   noul:11+]
 * A non-positive entry falls back to temperature[qtype], mirroring the Python
 * `temperature_by_options.get(...) or temperature[qtype]` when 0.0 is stored.
 */
static float temp_bucket_value(const laya_config *c, int qtype, int k) {
    int size_bucket = (k <= 2) ? 0 : (k <= 5) ? 1 : (k <= 10) ? 2 : 3;
    int idx = qtype * 4 + size_bucket;
    float v = 0.0f;
    if (idx >= 0 && idx < LAYA_TEMP_BUCKETS) v = c->temperature_by_options[idx];
    if (!(v > 0.0f)) v = (qtype >= 0 && qtype < 3) ? c->temperature[qtype] : 1.0f;
    return v;
}

/* agent.py confidence_from_probs: 1 - H(p)/log(k), clipped to [0,1]. */
static float confidence_from_probs(const float *p, int k) {
    if (k < 2) return 1.0f;
    double ent = 0.0;
    for (int i = 0; i < k; i++) {
        double pi = p[i];
        if (pi < 1e-12) pi = 1e-12;
        else if (pi > 1.0) pi = 1.0;
        ent += pi * log(pi);
    }
    double conf = 1.0 + ent / log((double)k);
    if (conf < 0.0) conf = 0.0;
    if (conf > 1.0) conf = 1.0;
    return (float)conf;
}

static void softmax_inplace(float *z, int k) {
    if (k <= 0) return;
    float mx = z[0];
    for (int i = 1; i < k; i++) if (z[i] > mx) mx = z[i];
    float sum = 0.0f;
    for (int i = 0; i < k; i++) { z[i] = expf(z[i] - mx); sum += z[i]; }
    if (sum > 0.0f) for (int i = 0; i < k; i++) z[i] /= sum;
}

/* ---- action head --------------------------------------------------- */

/*
 * act_logits = act_head(cat([h[:,0], top1, top1-top2, ent/log k, k/255])).
 * The four scalars come from the UNcalibrated softmax over the scorer logits,
 * matching common.py (only the reported answers use the calibrated ones).
 */
static laya_status action_head(laya_decision *dec, const laya_model *m,
                               laya_decision_result *out, int k, int pooled_row,
                               int seq) {
    laya_decision_ws *ws = &dec->ws;
    const laya_config *c = &m->cfg;
    const int d = c->hidden_size;
    const int dact = d + 4;
    laya_status st;

    float p[64];
    for (int i = 0; i < k; i++) p[i] = out->raw_logits[i];
    softmax_inplace(p, k);

    float a = -1.0f, b = -1.0f;
    for (int i = 0; i < k; i++) {
        if (p[i] > a) { b = a; a = p[i]; }
        else if (p[i] > b) { b = p[i]; }
    }
    float top1 = a;
    float top2 = (b < 0.0f) ? 0.0f : b;

    double ent = 0.0;
    for (int i = 0; i < k; i++) {
        double pi = p[i] < 1e-9 ? 1e-9 : p[i];
        ent += pi * log(pi);
    }
    double denom = (k >= 2) ? log((double)k) : 0.0;
    float ent_n = (denom > 0.0) ? (float)(-ent / denom) : 0.0f;
    float knorm = (float)k / 255.0f;

    /* [pooled(d) | feats(4)] concatenated on the host; the pooled row is the
     * first row of this question's block of the head output. */
    {
        uint16_t *vec = (uint16_t *)malloc((size_t)dact * 2);
        if (!vec) return dec_fail(LAYA_ERR_OOM, "act vec alloc");
        cudaError_t e = cudaMemcpy(vec, (uint8_t *)ws->h + (size_t)pooled_row * d * 2,
                                   (size_t)d * 2, cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) {
            free(vec);
            return dec_fail(LAYA_ERR_RUNTIME, "pooled copy: %s", cudaGetErrorString(e));
        }
        const float feats[4] = { top1, top1 - top2, ent_n, knorm };
        for (int i = 0; i < 4; i++) vec[d + i] = laya_f32_to_bf16(feats[i]);
        e = cudaMemcpy(ws->act_vec, vec, (size_t)dact * 2, cudaMemcpyHostToDevice);
        free(vec);
        if (e != cudaSuccess)
            return dec_fail(LAYA_ERR_RUNTIME, "act vec copy: %s", cudaGetErrorString(e));
    }
    (void)seq;

    /* Linear(d+4, 256) -> exact GELU -> Linear(256, n_act) */
    if ((st = linear(&m->decision.act0_w, m->decision.act0_b.ptr, ws->act_vec, ws->act1, 1, 256, dact)) != LAYA_OK)
        return st;
    laya_gelu(ws->act1, ws->act1_act, 256);
    if ((st = check("act gelu")) != LAYA_OK) return st;
    if ((st = linear(&m->decision.act2_w, m->decision.act2_b.ptr, ws->act1_act, ws->proj, 1, c->n_act, 256)) != LAYA_OK)
        return st;
    if ((st = check("act out")) != LAYA_OK) return st;

    {
        uint16_t raw[2] = { 0, 0 };
        float lg[2] = { 0.0f, 0.0f };
        int n = c->n_act < 2 ? c->n_act : 2;
        cudaError_t e = cudaMemcpy(raw, ws->proj, (size_t)n * 2,
                                   cudaMemcpyDeviceToHost);
        if (e != cudaSuccess)
            return dec_fail(LAYA_ERR_RUNTIME, "act logits copy: %s", cudaGetErrorString(e));
        laya_bf16_buf_to_f32(raw, lg, (size_t)n);
        for (int i = 0; i < n; i++) out->act_logits[i] = lg[i];
        softmax_inplace(lg, n);
        out->act_probability = lg[0];
    }
    return LAYA_OK;
}

/* ---- forward ------------------------------------------------------- */

laya_status laya_decision_forward_batch(laya_decision *dec, const laya_model *m,
                                        const void *enc_out_dev, int batch,
                                        int seq, const void *markers_dev,
                                        int kmax, const uint8_t *marker_mask,
                                        const int32_t *qtypes,
                                        const uint8_t *tok_valid,
                                        laya_decision_result *out) {
    if (!dec || !m || !enc_out_dev || !out)
        return dec_fail(LAYA_ERR_MISMATCH, "decision_forward: null argument");
    laya_status st = laya_decision_init(dec, m, batch, seq, batch * kmax);
    if (st != LAYA_OK) return st;
    if (seq > dec->ws.item_seq || batch > dec->ws.n_batch ||
        batch * kmax > dec->ws.n_markers_cap)
        return dec_fail(LAYA_ERR_RUNTIME, "shape exceeds decision workspace");

    const laya_config *c = &m->cfg;
    laya_decision_ws *ws = &dec->ws;
    const int d = c->hidden_size;
    const int S = seq, B = batch, R = B * S, K = kmax, M = B * kmax;

    for (int q = 0; q < B; q++) {
        if (!out) break;
        memset(&out[q], 0, sizeof(out[q]));
        out[q].qtype = qtypes ? qtypes[q] : 0;
        out[q].n_options = K;
        out[q].choice = -1;
        if (out[q].qtype < 0 || out[q].qtype >= c->n_types)
            return dec_fail(LAYA_ERR_MISMATCH, "qtype %d out of range", out[q].qtype);
    }
    if (!qtypes)
        return dec_fail(LAYA_ERR_MISMATCH, "batched forward needs per-question qtypes");

    /* Key-validity flags for the padding mask: same lifetime rule as the
     * encoder's -- a property of the request, not of the workspace. */
    if (tok_valid)
        cudaMemcpy(ws->valid, tok_valid, (size_t)R, cudaMemcpyHostToDevice);
    else
        cudaMemset(ws->valid, 1, (size_t)R);

    /* 1. type embedding: h = enc_out + type_emb(qtype)[None, :] per item */
    {
        int64_t idx[256];
        if (B > 256) return dec_fail(LAYA_ERR_RUNTIME, "batch %d too large", B);
        for (int q = 0; q < B; q++) idx[q] = out[q].qtype;
        cudaError_t e = cudaMemcpy(ws->type_idx_dev, idx, (size_t)B * sizeof(int64_t),
                                   cudaMemcpyHostToDevice);
        if (e != cudaSuccess)
            return dec_fail(LAYA_ERR_RUNTIME, "type idx copy: %s", cudaGetErrorString(e));
        laya_gather_rows(m->type_emb.ptr, ws->type_idx_dev, ws->type_vec, B, d,
                         m->type_emb.shape[0]);
        if ((st = check("type_emb gather")) != LAYA_OK) return st;
        laya_bcast_add_items(enc_out_dev, ws->type_vec, ws->h, B, S, d);
        if ((st = check("type_emb add")) != LAYA_OK) return st;
        dump_dev("%s.type", ws->h, (size_t)((int64_t)R * d * 2), 0);
    }

    /* 2. decision transformer head (stock TransformerEncoderLayer) */
    for (int i = 0; i < c->n_head_layers; i++) {
        if ((st = head_layer(dec, m, &m->head[i], i, B, S)) != LAYA_OK) return st;
    }

    /* 3. marker gather: m_hidden[b*K+j] = h[b*S + marker_pos[b][j]] */
    if (!markers_dev) return dec_fail(LAYA_ERR_MISMATCH, "marker gather needs positions");
    {
        int64_t *idx = (int64_t *)malloc((size_t)M * sizeof(int64_t));
        if (!idx) return dec_fail(LAYA_ERR_OOM, "marker idx alloc");
        cudaError_t e = cudaMemcpy(idx, markers_dev, (size_t)M * sizeof(int64_t),
                                   cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) {
            free(idx);
            return dec_fail(LAYA_ERR_RUNTIME, "marker idx read: %s", cudaGetErrorString(e));
        }
        for (int i = 0; i < M; i++) {
            if (idx[i] < 0 || idx[i] >= R) {
                int64_t p = idx[i];
                free(idx);
                return dec_fail(LAYA_ERR_MISMATCH, "marker %d at %lld outside [0,%d)",
                                i, (long long)p, R);
            }
        }
        e = cudaMemcpy(ws->m_idx_dev, idx, (size_t)M * sizeof(int64_t),
                       cudaMemcpyHostToDevice);
        free(idx);
        if (e != cudaSuccess)
            return dec_fail(LAYA_ERR_RUNTIME, "marker idx copy: %s", cudaGetErrorString(e));
        laya_gather_rows(ws->h, ws->m_idx_dev, ws->m_hidden, M, d, R);
        if ((st = check("marker gather")) != LAYA_OK) return st;
        dump_dev("%s.marker", ws->m_hidden, (size_t)((int64_t)M * d * 2), 0);
    }

    /* 4. scorer over one flattened batch of (question, option) rows. The FFN
     * scratch is sized R*4d >= M*4d, so the M-row scorer fits. */
    {
        const laya_decision_weights *dw = &m->decision;
        laya_layernorm(ws->m_hidden, dw->scorer0_w.ptr, dw->scorer0_b.ptr,
                       ws->m_hidden, M, d, c->layer_norm_eps);
        if ((st = check("scorer norm")) != LAYA_OK) return st;
        if ((st = linear(&dw->scorer1_w, dw->scorer1_b.ptr, ws->m_hidden, ws->ff, M, d, d)) != LAYA_OK) return st;
        laya_gelu(ws->ff, ws->ff_act, (size_t)M * (size_t)d);
        if ((st = check("scorer gelu")) != LAYA_OK) return st;
        if ((st = linear(&dw->scorer3_w, dw->scorer3_b.ptr, ws->ff_act, ws->proj, M, 1, d)) != LAYA_OK) return st;
        if ((st = check("scorer out")) != LAYA_OK) return st;
    }

    /* 5. read the logits once, then post-process every question on the host */
    {
        uint16_t *raw = (uint16_t *)malloc((size_t)M * 2);
        float *logits = (float *)malloc((size_t)M * sizeof(float));
        if (!raw || !logits) { free(raw); free(logits); return dec_fail(LAYA_ERR_OOM, "logits alloc"); }
        cudaError_t e = cudaMemcpy(raw, ws->proj, (size_t)M * 2, cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) {
            free(raw); free(logits);
            return dec_fail(LAYA_ERR_RUNTIME, "logits copy: %s", cudaGetErrorString(e));
        }
        laya_bf16_buf_to_f32(raw, logits, (size_t)M);
        free(raw);

        for (int q = 0; q < B; q++) {
            laya_decision_result *o = &out[q];
            const uint8_t *mm = marker_mask ? marker_mask + (size_t)q * K : NULL;

            for (int i = 0; i < K; i++) {
                int valid = mm ? (mm[i] != 0) : 1;
                o->valid[i] = valid;
                o->raw_logits[i] = valid ? logits[(size_t)q * K + i] : -1e4f;
            }

            float t_scale = temp_bucket_value(c, o->qtype, K);
            float tmin = t_scale < 1e-3f ? 1e-3f : t_scale;
            float z[64];
            int k = 0;
            for (int i = 0; i < K; i++) {
                if (!o->valid[i]) continue;
                z[k++] = o->raw_logits[i] / tmin;
            }
            if (k < 1) { free(logits); return dec_fail(LAYA_ERR_MISMATCH, "no valid options"); }
            softmax_inplace(z, k);

            int j = 0;
            for (int i = 0; i < K; i++) o->probs[i] = o->valid[i] ? z[j++] : 0.0f;
            o->confidence = confidence_from_probs(z, k);

            if (o->qtype == 0) {          /* choice: argmax over real options */
                int best = -1;
                float bv = -1e30f;
                for (int i = 0; i < K; i++)
                    if (o->valid[i] && o->probs[i] > bv) { bv = o->probs[i]; best = i; }
                o->choice = best;
            } else if (o->qtype == 1) {   /* score: expected option index */
                double s = 0.0;
                for (int i = 0; i < K; i++)
                    if (o->valid[i]) s += (double)i * o->probs[i];
                o->score = (float)s;
            } else {                      /* noul: p[1] and its confidence */
                o->noul = (K > 1) ? o->probs[1] : 0.0f;
                o->confidence = fmaxf(o->noul, 1.0f - o->noul);
            }
        }
        free(logits);
    }

    /* 6. action head, one question at a time (d+4 = 1028 wide: negligible) */
    for (int q = 0; q < B; q++) {
        int k = 0;
        for (int i = 0; i < K; i++) if (out[q].valid[i]) k++;
        if ((st = action_head(dec, m, &out[q], k, q * S, S)) != LAYA_OK) return st;
    }
    return LAYA_OK;
}

laya_status laya_decision_forward(laya_decision *dec, const laya_model *m,
                                  const void *enc_out_dev, int seq,
                                  const void *markers_dev, int n_markers,
                                  const uint8_t *marker_mask, int qtype,
                                  const uint8_t *tok_valid,
                                  laya_decision_result *out) {
    /* Single-question shim: marker positions arrive relative to the item. */
    laya_status st;
    int64_t rel[1024];
    int64_t abs[1024];
    if (n_markers > 1024) return dec_fail(LAYA_ERR_RUNTIME, "too many markers");
    cudaError_t e = cudaMemcpy(rel, markers_dev, (size_t)n_markers * sizeof(int64_t),
                               cudaMemcpyDeviceToHost);
    if (e != cudaSuccess)
        return dec_fail(LAYA_ERR_RUNTIME, "marker shim read: %s", cudaGetErrorString(e));
    for (int i = 0; i < n_markers; i++) abs[i] = rel[i];
    void *tmp = NULL;
    if (cudaMalloc(&tmp, (size_t)n_markers * sizeof(int64_t)) != cudaSuccess)
        return dec_fail(LAYA_ERR_OOM, "marker shim alloc");
    e = cudaMemcpy(tmp, abs, (size_t)n_markers * sizeof(int64_t), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) { cudaFree(tmp); return dec_fail(LAYA_ERR_RUNTIME, "marker shim copy"); }
    st = laya_decision_forward_batch(dec, m, enc_out_dev, 1, seq, tmp, n_markers,
                                     marker_mask, &qtype, tok_valid, out);
    cudaFree(tmp);
    return st;
}
