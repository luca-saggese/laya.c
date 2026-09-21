/*
 * ModernBERT encoder forward path for Laya.
 *
 * Direct translation of the torch module graph the original Laya checkpoint
 * instantiates (see _reference/laya/laya/common.py and
 * transformers/models/modernbert/modeling_modernbert.py):
 *
 *   embeddings   norm(tok_embeddings(ids))
 *   layer i      h = h + attn(attn_norm(h))
 *                h = h + mlp(mlp_norm(h))
 *   final        final_norm(h)
 *
 * The torch shapes per layer are
 *   Wqkv(h).view(S, 3, H, hd).unbind(-3).transpose(1,2)  -> q/k/v [H,S,hd]
 *   apply_rotary_pos_emb(..., unsqueeze_dim=1)
 *   softmax(q k^T * hd^-0.5 + mask, dim=-1, dtype=fp32).to(bf16)
 *   attn_output.transpose(1,2).reshape(S, H*hd) -> Wo
 *   Wi(h).chunk(2,-1) -> input, gate ; Wo(act(input) * gate)
 *
 * Every primitive is copied from o1.c:
 *   laya_gather_rows   embeddings lookup
 *   laya_layernorm     LayerNorm (bias disabled by this checkpoint)
 *   laya_linear        Wqkv/Wo/Wi/mlp_Wo via cuBLASLt bf16 tensor cores
 *   laya_rope_cos_sin  ModernBERT RoPE table (one per layer type)
 *   laya_apply_rotary  apply_rotary_pos_emb
 *   laya_qkv_split     the view/unbind/transpose
 *   laya_attention_ws  eager attention with caller-owned scratch
 *   laya_head_merge    attn_output.transpose(1,2).reshape(...)
 *   laya_glu_split     Wi(...).chunk(2,-1)
 *   laya_geglu         act(input) * gate
 *   laya_residual_add  the two residual adds
 */

#include "modernbert.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hd_cuda.h"

static char g_enc_err[512];

const char *laya_encoder_last_error(void) { return g_enc_err; }

static laya_status enc_fail(laya_status st, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_enc_err, sizeof(g_enc_err), fmt, ap);
    va_end(ap);
    return st;
}

static size_t align256(size_t n) { return (n + 255u) & ~(size_t)255u; }

/* ------------------------------------------------------------------ */
/* Step 4.1 -- weight binding validation                               */
/* ------------------------------------------------------------------ */

/* Every tensor the forward dereferences is shape-checked once at load time.
 * A wrong layout fails the load instead of producing silent garbage. */
static laya_status check_shape(const laya_tensor *t, int64_t d0, int64_t d1,
                               const char *what) {
    if (!t->ptr || t->rank == 0)
        return enc_fail(LAYA_ERR_MISSING, "%s: tensor absent", what);
    int64_t n0 = t->shape[0];
    int64_t n1 = t->rank >= 2 ? t->shape[1] : 1;
    if (n0 != d0 || n1 != d1)
        return enc_fail(LAYA_ERR_MISMATCH, "%s: expected [%lld,%lld] got [%lld,%lld]",
                        what, (long long)d0, (long long)d1, (long long)n0, (long long)n1);
    return LAYA_OK;
}

static laya_status check_vec(const laya_tensor *t, int64_t n, const char *what) {
    if (!t->ptr || t->rank == 0)
        return enc_fail(LAYA_ERR_MISSING, "%s: tensor absent", what);
    if (t->shape[0] != n)
        return enc_fail(LAYA_ERR_MISMATCH, "%s: expected [%lld] got [%lld]", what,
                        (long long)n, (long long)t->shape[0]);
    return LAYA_OK;
}

laya_status laya_encoder_check_weights(const laya_model *m) {
    const laya_config *c = &m->cfg;
    int d = c->hidden_size, inter = c->intermediate_size, V = c->vocab_size;
    laya_status st;
    char what[96];

    if ((st = check_shape(&m->tok_embeddings, V, d, "tok_embeddings")) != LAYA_OK) return st;
    if ((st = check_vec(&m->emb_norm, d, "embeddings.norm")) != LAYA_OK) return st;
    if ((st = check_vec(&m->final_norm, d, "final_norm")) != LAYA_OK) return st;

    for (int i = 0; i < c->n_layers; i++) {
        const laya_layer_weights *lw = &m->layers[i];
#define CHK(expr, ...)                                 \
        do {                                           \
            snprintf(what, sizeof(what), __VA_ARGS__); \
            if ((st = (expr)) != LAYA_OK) return st;   \
        } while (0)
        /* layer 0's attn_norm is nn.Identity(): no weight by design. */
        if (i != 0) CHK(check_vec(&lw->attn_norm, d, "%s"), "layers.%02d.attn_norm", i);
        CHK(check_vec(&lw->mlp_norm, d, "%s"), "layers.%02d.mlp_norm", i);
        CHK(check_shape(&lw->Wqkv, 3 * d, d, "%s"), "layers.%02d.attn.Wqkv", i);
        CHK(check_shape(&lw->Wo, d, d, "%s"), "layers.%02d.attn.Wo", i);
        CHK(check_shape(&lw->Wi, 2 * inter, d, "%s"), "layers.%02d.mlp.Wi", i);
        CHK(check_shape(&lw->mlp_Wo, d, inter, "%s"), "layers.%02d.mlp.Wo", i);
#undef CHK
    }
    return LAYA_OK;
}

/* ------------------------------------------------------------------ */
/* Step 4.4 -- persistent workspace                                    */
/* ------------------------------------------------------------------ */

/* A single cudaMalloc for the whole pass. Nothing in the per-layer loop
 * allocates, which is the o1.c residency policy and the §11 goal of zero
 * cudaMalloc/cudaFree in the steady-state forward. */
laya_status laya_encoder_init(laya_encoder *enc, const laya_model *m, int batch,
                              int seq) {
    if (seq <= 0 || batch <= 0)
        return enc_fail(LAYA_ERR_RUNTIME, "encoder_init: batch=%d seq=%d", batch, seq);
    if (enc->initialized && enc->ws.item_seq >= seq && enc->ws.rows >= batch * seq)
        return LAYA_OK;
    laya_encoder_free(enc);

    const laya_config *c = &m->cfg;
    int d = c->hidden_size, H = c->n_heads, hd = c->head_dim, inter = c->intermediate_size;
    size_t S = (size_t)seq;
    size_t B = (size_t)batch;
    size_t R = B * S;                             /* flattened rows */

    size_t bf = R * (size_t)d * 2;                /* one [R,d] bf16 activation   */
    /* Fused Wqkv writes [S,3d] per item and the MLP Wi output writes [S,2*inter]
     * into the same reused scratch slot, so the slot must be sized for the
     * larger of the two widths. 2*inter exceeds 3*d on this model. */
    size_t scratch = R * (size_t)(3 * d > 2 * inter ? 3 * d : 2 * inter) * 2;
    size_t hmaj = (size_t)H * R * (size_t)hd * 2; /* one [H,R,hd] buffer         */
    size_t mlp = R * (size_t)inter * 2;           /* one [R,I] buffer            */
    size_t mask = B * S * S * 2;
    size_t rope = S * (size_t)hd * 4;
    size_t f32scores = (size_t)H * R * S * 4;
    size_t vbytes = align256(R);                  /* [R] uint8 valid flags */

    size_t dump = 0;
    for (unsigned b = 0; b < 5; b++)
        if (enc->dump_mask & (1u << b)) dump += bf;

    size_t total = 0;
    total += bf * 4;                /* h, res, norm, attn_ctx  */
    total += scratch;               /* qkv / Wi scratch slot */
    total += hmaj * 6;              /* q, k, v, attn_out, q_rot, k_rot */
    total += mlp * 3 + bf;          /* gate, up, act, mlp_out  */
    total += rope * 4;              /* cos/sin per layer type  */
    total += mask * 2;              /* full + sliding          */
    total += f32scores;             /* attention scores        */
    total += (size_t)H * R * S * 2; /* attention probs         */
    total += vbytes;                /* valid-token flags       */
    total += dump;
    total = align256(total) + 256;

    void *base = NULL;
    cudaError_t e = cudaMalloc(&base, total);
    if (e != cudaSuccess)
        return enc_fail(LAYA_ERR_OOM, "encoder workspace cudaMalloc(%zu): %s", total,
                        cudaGetErrorString(e));

    laya_encoder_ws *ws = &enc->ws;
    memset(ws, 0, sizeof(*ws));
    ws->rows = (int)R;
    ws->item_seq = seq;
    ws->hidden = d;
    ws->n_heads = H;
    ws->head_dim = hd;
    ws->intermediate = inter;
    ws->bytes = (int64_t)total;

    uint8_t *p = (uint8_t *)base;
    ws->h = p;            p += bf;
    ws->res = p;          p += bf;
    ws->norm = p;         p += bf;
    ws->attn_ctx = p;     p += bf;
    ws->qkv = p;          p += scratch;
    ws->scratch_bytes = (int64_t)scratch;
    ws->q = p;            p += hmaj;
    ws->k = p;            p += hmaj;
    ws->v = p;            p += hmaj;
    ws->attn_out = p;     p += hmaj;
    ws->q_rot = p;        p += hmaj;
    ws->k_rot = p;        p += hmaj;
    ws->mlp_gate = p;     p += mlp;
    ws->mlp_up = p;       p += mlp;
    ws->mlp_act = p;      p += mlp;
    ws->mlp_out = p;      p += bf;
    ws->rope_cos = p;     p += rope;
    ws->rope_sin = p;     p += rope;
    ws->rope_cos2 = p;    p += rope;
    ws->rope_sin2 = p;    p += rope;
    ws->mask_full = p;    p += mask;
    ws->mask_sliding = p; p += mask;
    ws->scores = p;       p += f32scores;
    ws->probs = p;        p += (size_t)H * R * S * 2;
    ws->valid = p;        p += vbytes;
    if (enc->dump_mask & LAYA_DUMP_EMBEDDINGS) { ws->dump_emb = p; p += bf; }
    if (enc->dump_mask & LAYA_DUMP_LAYER0)     { ws->dump_l0 = p;  p += bf; }
    if (enc->dump_mask & LAYA_DUMP_LAYER1)     { ws->dump_l1 = p;  p += bf; }
    if (enc->dump_mask & LAYA_DUMP_LAYER15)    { ws->dump_l15 = p; p += bf; }
    if (enc->dump_mask & LAYA_DUMP_LAYER27)    { ws->dump_l27 = p; p += bf; }

    /* One-time allocation audit: every reusable slot, in allocation order,
     * with the capacity the layout actually reserved for it. */
    if (getenv("LAYA_DEBUG_PROBE")) {
        struct { const char *n; const void *p; size_t req; } tab[] = {
            {"h",         ws->h,        bf},
            {"res",       ws->res,      bf},
            {"norm",      ws->norm,     bf},
            {"attn_ctx",  ws->attn_ctx, bf},
            {"scratch",   ws->qkv,      scratch},
            {"q",         ws->q,        hmaj},
            {"k",         ws->k,        hmaj},
            {"v",         ws->v,        hmaj},
            {"attn_out",  ws->attn_out, hmaj},
            {"q_rot",     ws->q_rot,    hmaj},
            {"k_rot",     ws->k_rot,    hmaj},
            {"mlp_gate",  ws->mlp_gate, mlp},
            {"mlp_up",    ws->mlp_up,   mlp},
            {"mlp_act",   ws->mlp_act,  mlp},
            {"mlp_out",   ws->mlp_out,  bf},
            {"rope_cos",  ws->rope_cos, rope},
            {"rope_sin",  ws->rope_sin, rope},
            {"rope_cos2", ws->rope_cos2,rope},
            {"rope_sin2", ws->rope_sin2,rope},
            {"mask_full", ws->mask_full, mask},
            {"mask_slid", ws->mask_sliding, mask},
            {"scores",    ws->scores,   f32scores},
            {"probs",     ws->probs,    (size_t)H * R * S * 2},
            {"valid",     ws->valid,    vbytes},
        };
        const size_t ntab = sizeof(tab) / sizeof(tab[0]);
        fprintf(stderr, "%-10s %10s %10s  %s\n", "buffer", "capacity", "required", "status");
        for (size_t k = 0; k < ntab; k++) {
            size_t cap = (k + 1 < ntab)
                             ? (size_t)((const uint8_t *)tab[k + 1].p - (const uint8_t *)tab[k].p)
                             : (size_t)((const uint8_t *)base + total - (const uint8_t *)tab[k].p);
            fprintf(stderr, "%-10s %10zu %10zu  %s\n", tab[k].n, cap, tab[k].req,
                    tab[k].req <= cap ? "OK" : "OVERFLOW");
        }
    }

    /* Full-attention items get no mask at all from the oracle when a group is
     * a single unpadded item, but laya_attention_ws requires one: an all-zero
     * additive mask is exactly the identity of "every position attends to
     * every position" *within one item*. For B > 1 the mask must additionally
     * be block-diagonal, which is the [B,1,L,L] form transformers builds.
     * Both masks are rebuilt per request by the forward, which owns the
     * real-token flags; nothing to do here. */

    /* Two RoPE tables, one per layer type: the checkpoint uses a different
     * theta for global and sliding layers. */
    laya_rope_cos_sin((float *)ws->rope_cos, (float *)ws->rope_sin, seq, hd,
                      c->global_rope_theta);
    laya_rope_cos_sin((float *)ws->rope_cos2, (float *)ws->rope_sin2, seq, hd,
                      c->local_rope_theta);

    enc->initialized = 1;
    return LAYA_OK;
}

void laya_encoder_set_dump(laya_encoder *enc, unsigned mask) {
    if (!enc || enc->dump_mask == mask) return;
    enc->dump_mask = mask;
    if (enc->initialized) laya_encoder_free(enc);
    enc->dump_mask = mask;
}

void laya_encoder_free(laya_encoder *enc) {
    if (!enc) return;
    if (enc->initialized && enc->ws.h) cudaFree(enc->ws.h);
    unsigned mask = enc->dump_mask;
    memset(enc, 0, sizeof(*enc));
    enc->dump_mask = mask;
}

/* ------------------------------------------------------------------ */
/* Forward                                                             */
/* ------------------------------------------------------------------ */

static laya_status linear(const laya_tensor *w, const void *x, void *y,
                          int M, int N, int K) {
    /* o1.c laya_linear only supports the PyTorch [out, in] layout, which is
     * exactly what the GGUF holds, so no transposes appear anywhere. */
    laya_cuda_clear_error();
    laya_linear(x, w->ptr, NULL, y, M, N, K, 1);
    const char *err = laya_cuda_last_error();
    if (err && err[0]) return enc_fail(LAYA_ERR_RUNTIME, "linear %s: %s", w->name, err);
    return LAYA_OK;
}

static void dump_dev(const char *fmt, const void *dev, size_t nbytes) {
    const char *dbg = getenv("LAYA_DEBUG_PROBE");
    if (!dbg || !dbg[0]) return;
    cudaDeviceSynchronize();
    uint8_t *host = malloc(nbytes);
    if (!host) return;
    if (cudaMemcpy(host, dev, nbytes, cudaMemcpyDeviceToHost) != cudaSuccess) { free(host); return; }
    char path[512];
    snprintf(path, sizeof(path), fmt, dbg);
    FILE *f = fopen(path, "wb");
    if (f) { fwrite(host, 1, nbytes, f); fclose(f); }
    free(host);
}

laya_status laya_encoder_forward_batch(laya_encoder *enc, const laya_model *m,
                                       const void *ids_dev, int batch, int seq,
                                       const uint8_t *attn_dev, void *out_dev) {
    laya_status st = laya_encoder_init(enc, m, batch, seq);
    if (st != LAYA_OK) return st;
    if (seq > enc->ws.item_seq || batch * seq > enc->ws.rows)
        return enc_fail(LAYA_ERR_RUNTIME, "batch %d x seq %d exceeds workspace %d x %d",
                        batch, seq, enc->ws.rows / enc->ws.item_seq, enc->ws.item_seq);

    const laya_config *c = &m->cfg;
    laya_encoder_ws *ws = &enc->ws;
    const int d = c->hidden_size, H = c->n_heads, hd = c->head_dim;
    const int inter = c->intermediate_size, S = seq, B = batch;
    const int R = B * S;
    const float scale = 1.0f / sqrtf((float)hd);

    /* Padding mask: rebuilt per request, since the validity pattern is a
     * property of the request, not of the workspace. */
    uint8_t *vbuf = (uint8_t *)ws->valid;
    if (attn_dev)
        cudaMemcpy(vbuf, attn_dev, (size_t)R, cudaMemcpyHostToDevice);
    else
        cudaMemset(vbuf, 1, (size_t)R);
    laya_attn_mask_batch(ws->mask_full, vbuf, B, S, S, 0);
    laya_attn_mask_batch(ws->mask_sliding, vbuf, B, S, c->sliding_window, 1);

    /* ---- embeddings: norm(tok_embeddings(ids)) ---- */
    laya_gather_rows(m->tok_embeddings.ptr, ids_dev, ws->h, R, d, c->vocab_size);
    laya_layernorm(ws->h, m->emb_norm.ptr, NULL, ws->h, R, d, c->layer_norm_eps);
    if (ws->dump_emb) cudaMemcpyAsync(ws->dump_emb, ws->h, (size_t)R * d * 2,
                                      cudaMemcpyDeviceToDevice, 0);

    for (int i = 0; i < c->n_layers; i++) {
        const laya_layer_weights *lw = &m->layers[i];
        const int is_global = lw->is_global;
        const float *cos = (const float *)(is_global ? ws->rope_cos : ws->rope_cos2);
        const float *sin = (const float *)(is_global ? ws->rope_sin : ws->rope_sin2);

        /* ---------------- attention ---------------- */
        const void *attn_in = ws->h;
        if (i != 0) {
            laya_layernorm(ws->h, lw->attn_norm.ptr, NULL, ws->norm, R, d, c->layer_norm_eps);
            attn_in = ws->norm;
        }

        if ((st = linear(&lw->Wqkv, attn_in, ws->qkv, R, 3 * d, d)) != LAYA_OK) return st;

        /* Layout ops + attention are per item: the head-major buffers and the
         * [S,S] mask are indexed item-locally, so each item is passed as a
         * contiguous slice with its own mask block. */
        const void *mask = is_global ? ws->mask_full : ws->mask_sliding;
        for (int b = 0; b < B; b++) {
            const size_t off_qkv = (size_t)b * S * 3 * d * 2;
            const size_t off_h = (size_t)b * H * S * hd * 2;
            const size_t off_msk = (size_t)b * S * S * 2;
            const size_t off_sc = (size_t)b * H * S * S;
            const uint8_t *qkv_b = (const uint8_t *)ws->qkv + off_qkv;

            laya_qkv_split(qkv_b, (uint8_t *)ws->q + off_h,
                           (uint8_t *)ws->k + off_h, (uint8_t *)ws->v + off_h, S, H, hd);
            laya_apply_rotary_f32((uint8_t *)ws->q + off_h, cos, sin,
                                  (uint8_t *)ws->q_rot + off_h, H, S, hd);
            laya_apply_rotary_f32((uint8_t *)ws->k + off_h, cos, sin,
                                  (uint8_t *)ws->k_rot + off_h, H, S, hd);
            laya_attention_ws((uint8_t *)ws->q_rot + off_h,
                              (uint8_t *)ws->k_rot + off_h,
                              (uint8_t *)ws->v + off_h,
                              (const uint8_t *)mask + off_msk,
                              (float *)ws->scores + off_sc, ws->probs + off_sc,
                              (uint8_t *)ws->attn_out + off_h, H, S, hd, scale);
            laya_head_merge((uint8_t *)ws->attn_out + off_h,
                            (uint8_t *)ws->norm + (size_t)b * S * d * 2, S, H, hd);
        }

        if ((st = linear(&lw->Wo, ws->norm, ws->attn_ctx, R, d, d)) != LAYA_OK) return st;
        laya_residual_add(ws->h, ws->attn_ctx, ws->h, (size_t)R * d);
        if (i == 0) dump_dev("%s.l0.attnres", ws->h, (size_t)R * d * 2);

        /* ---------------- mlp (GeGLU) ---------------- */
        laya_layernorm(ws->h, lw->mlp_norm.ptr, NULL, ws->norm, R, d, c->layer_norm_eps);
        /* Wi writes [R, 2*inter] into the reused scratch that also backs the
         * [R, 3*d] qkv output; the arena slot is sized for the larger of the
         * two, so this is guaranteed in-bounds. */
        size_t wi_bytes = (size_t)R * (size_t)(2 * inter) * 2;
        if (wi_bytes > (size_t)ws->scratch_bytes)
            return enc_fail(LAYA_ERR_RUNTIME,
                            "Wi scratch overflow: need %zu have %lld", wi_bytes,
                            (long long)ws->scratch_bytes);
        if ((st = linear(&lw->Wi, ws->norm, ws->qkv, R, 2 * inter, d)) != LAYA_OK) return st;
        /* torch: input, gate = Wi(h).chunk(2, -1); Wo(act(input) * gate). */
        laya_glu_split(ws->qkv, ws->mlp_gate, ws->mlp_up, R, inter);
        laya_geglu(ws->mlp_gate, ws->mlp_up, ws->mlp_act, (size_t)R * inter);
        if ((st = linear(&lw->mlp_Wo, ws->mlp_act, ws->mlp_out, R, d, inter)) != LAYA_OK)
            return st;
        laya_residual_add(ws->h, ws->mlp_out, ws->h, (size_t)R * d);

        if (i == 0 && ws->dump_l0)
            cudaMemcpyAsync(ws->dump_l0, ws->h, (size_t)R * d * 2,
                            cudaMemcpyDeviceToDevice, 0);
        if (i == 1 && ws->dump_l1)
            cudaMemcpyAsync(ws->dump_l1, ws->h, (size_t)R * d * 2,
                            cudaMemcpyDeviceToDevice, 0);
        if (i == 15 && ws->dump_l15)
            cudaMemcpyAsync(ws->dump_l15, ws->h, (size_t)R * d * 2,
                            cudaMemcpyDeviceToDevice, 0);
        if (i == 27 && ws->dump_l27)
            cudaMemcpyAsync(ws->dump_l27, ws->h, (size_t)R * d * 2,
                            cudaMemcpyDeviceToDevice, 0);
    }

    laya_layernorm(ws->h, m->final_norm.ptr, NULL, out_dev, R, d, c->layer_norm_eps);
    return LAYA_OK;
}

laya_status laya_encoder_forward(laya_encoder *enc, const laya_model *m,
                                 const void *ids_dev, int seq, void *out_dev) {
    return laya_encoder_forward_batch(enc, m, ids_dev, 1, seq, NULL, out_dev);
}