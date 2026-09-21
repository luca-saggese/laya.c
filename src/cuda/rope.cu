/*
 * M1.2 reference CUDA primitives -- MRoPE and apply_rotary_pos_emb.
 *
 * The oracle builds cos/sin from a fused rotary embedding: interleaved mrope
 * frequency merge then cat(freqs, freqs). cos = cos(emb)*scaling.
 * See Qwen3VLTextRotaryEmbedding::forward and apply_interleaved_mrope, and
 * apply_rotary_pos_emb / rotate_half for the rotation itself.
 */

#include "cuda_internal.h"

#include <stdio.h>

/* ------------------------------------------------------------------ */
/* MRoPE cos/sin table                                                 */
/* ------------------------------------------------------------------ */

/*
 * Build matrix  E[s, k] for k in [0, half): freqs_inv[k] * position[s].
 * position_ids is [3, bs, seq]. The oracle transposes and then merges the
 * three T/H/W slices via apply_interleaved_mrope, producing emb[s, k].
 * emb = cat(freqs, freqs)  -> [s, dim];  cos = cos(emb)*scaling.
 *
 * For k in [0, half):     freqs        = [T,H,W] columns selected by section
 *                         interleaved  -> emb[s,k] = chosen freq * pos
 *                         cat          -> emb[s, half+k] = emb[s,k]
 */
__global__ void laya_mrope_cos_sin_kernel(const float *__restrict__ pos,  /* [3,bs,seq] */
                                        const int64_t *__restrict__ section, /* [3] */
                                        float *__restrict__ cosd, /* [bs,seq,dim] */
                                        float *__restrict__ sind, /* [bs,seq,dim] */
                                        int bs, int seq, int dim,
                                        float theta, float scaling,
                                        int n_section, int interleaved) {
    int s = blockIdx.x;          /* seq */
    int k = threadIdx.x;         /* [0, dim) */
    if (s >= seq || k >= dim) return;

    int half = dim / 2;
    int kk = (k < half) ? k : (k - half);       /* frequency index [0,half) */

    bool use_t = true;
    int sec = 0;
    if (interleaved) {
        /* Slot q = kk: base T; H overwrites q%3==1 && q<60; W overwrites q%3==2 && q<60. */
        int chunk = kk / 3;
        int slot = kk % 3;
        if (slot == 1 || slot == 2) {
            if (slot <= n_section - 1 && chunk < (int)section[slot]) { use_t = false; sec = slot; }
        }
    } else {
        /* chunked [TTT..HHH..WWW]: sec is the section kk falls into. */
        int acc = 0;
        sec = 0;
        for (int t = 0; t < n_section && t < 3; t++) {
            if (kk < acc + (int)section[t]) { sec = t; break; }
            acc += (int)section[t];
        }
    }

    /* Position from the selected row (T row index 0, H=1, W=2) at (bs, s). */
    int row = use_t ? 0 : sec;
    float p = pos[((size_t)row) * bs * seq + (size_t)(blockIdx.y) * seq + s];
    float base = expf(-logf(theta) * (2.0f * (float)kk) / (float)dim);
    float ang = base * p;
    float c = cosf(ang) * scaling;
    float sn = sinf(ang) * scaling;
    size_t base_idx = ((size_t)blockIdx.y * seq + s) * dim + k;
    cosd[base_idx] = c;
    sind[base_idx] = sn;
}

void laya_mrope_cos_sin(const float *pos_dev, int bs, int seq,
                      const int64_t *section_dev, int n_section,
                      int head_dim, float theta, float attention_scaling,
                      int interleaved, float *cos_dev, float *sin_dev) {
    if (!pos_dev || !section_dev || !cos_dev || !sin_dev || seq <= 0 || head_dim <= 0) {
        snprintf(laya_cuda_errbuf(), 512, "mrope: bad args");
        return;
    }
    (void)cos_dev; (void)sin_dev;
    laya_mrope_cos_sin_kernel<<<seq, 256>>>(
        pos_dev, section_dev, (float*)cos_dev, (float*)sin_dev,
        bs, seq, head_dim, theta, attention_scaling, n_section, interleaved);
}

/* ------------------------------------------------------------------ */
/* apply_rotary_pos_emb / rotate_half                                 */
/* ------------------------------------------------------------------ */

/* The rotation is exactly compute-gated by rotate_half (pairs) plus the
 * cosine/sine alignment to the oracle's rotate_half cat((-x2, x1), -1):
 *   out[0..half)   = x[0..half)*cos - x[half..)*sin
 *   out[half..)    = x[half..)*cos + x[0..half)*sin
 * The embedding (emb → cos/sin) repeats over the second half, so k and k+half
 * share the same angle.  */
__global__ void laya_apply_rotary_kernel(const uint16_t *__restrict__ x,
                                       const float *__restrict__ cosd,
                                       const float *__restrict__ sind,
                                       uint16_t *__restrict__ y,
                                       int heads, int seq, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; /* [H, S, D] */
    long total = (long)heads * seq;
    if (idx >= total * dim) return; /* guarded below via safe division */
    /* element index in [H*S*D] */
    long e = idx;
    int d = e % dim;
    long hs = e / dim;             /* head*seq + s */
    int s = (int)(hs % seq);
    long src = hs * dim + d;
    /* cos/sin shape [seq, dim] (broadcast over bs=1 and heads) */
    long cbase = (long)s * dim + d;

    int half = dim / 2;
    long halfe = (d < half) ? (d + half) : (d - half);
    long src_half = hs * dim + halfe;
    float x0 = laya_dev_bf16_to_f32(x[src]);
    float xh = laya_dev_bf16_to_f32(x[src_half]);
    float c = cosd[cbase];
    float sn = sind[cbase];
    /* Oracle: (q*cos) + (rotate_half(q)*sin) in bf16; torch rounds each
     * multiply to bf16 and the sum to bf16. Emulate that bf16 per-op rounding. */
    float v;
    if (d < half) {
        float a = laya_dev_bf16_round(x0 * c);
        float b = laya_dev_bf16_round(xh * sn);
        v = laya_dev_bf16_round(a - b);
    } else {
        float a = laya_dev_bf16_round(x0 * c);
        float b = laya_dev_bf16_round(xh * sn);
        v = laya_dev_bf16_round(a + b);
    }
    y[src] = laya_dev_f32_to_bf16(v);
}

void laya_apply_rotary(const void *x_dev, const float *cos_dev,
                     const float *sin_dev, void *y_dev,
                     int heads, int seq, int dim) {
    if (!x_dev || !cos_dev || !sin_dev || !y_dev || heads <= 0 || seq <= 0 || dim <= 0) {
        snprintf(laya_cuda_errbuf(), 512, "apply_rotary: bad args");
        return;
    }
    long total = (long)heads * seq * dim;
    long blocks = (total + 255) / 256;
    laya_apply_rotary_kernel<<<blocks, 256>>>(
        (const uint16_t *)x_dev, cos_dev, sin_dev, (uint16_t *)y_dev,
        heads, seq, dim);
}

/*
 * Vision-tower variant of apply_rotary. The oracle's
 * apply_rotary_pos_emb_vision upcasts q/k and cos/sin to FP32, applies the
 * rotation, then casts the result back to the original (bf16) dtype ONCE.
 * There is no per-op bf16 rounding (unlike the text path). Same indexing /
 * half-split layout as laya_apply_rotary_kernel.
 */
__global__ void laya_apply_rotary_f32_kernel(const uint16_t *__restrict__ x,
                                           const float *__restrict__ cosd,
                                           const float *__restrict__ sind,
                                           uint16_t *__restrict__ y,
                                           int heads, int seq, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)heads * seq;
    if (idx >= total * dim) return;
    long e = idx;
    int d = e % dim;
    long hs = e / dim;
    int s = (int)(hs % seq);
    long src = hs * dim + d;
    long cbase = (long)s * dim + d;

    int half = dim / 2;
    long halfe = (d < half) ? (d + half) : (d - half);
    long src_half = hs * dim + halfe;
    float x0 = laya_dev_bf16_to_f32(x[src]);
    float xh = laya_dev_bf16_to_f32(x[src_half]);
    float c = cosd[cbase];
    float sn = sind[cbase];
    float v = (d < half) ? (x0 * c - xh * sn) : (x0 * c + xh * sn);
    y[src] = laya_dev_f32_to_bf16(v);
}

void laya_apply_rotary_f32(const void *x_dev, const float *cos_dev,
                         const float *sin_dev, void *y_dev,
                         int heads, int seq, int dim) {
    if (!x_dev || !cos_dev || !sin_dev || !y_dev || heads <= 0 || seq <= 0 || dim <= 0) {
        snprintf(laya_cuda_errbuf(), 512, "apply_rotary_f32: bad args");
        return;
    }
    long total = (long)heads * seq * dim;
    long blocks = (total + 255) / 256;
    laya_apply_rotary_f32_kernel<<<blocks, 256>>>(
        (const uint16_t *)x_dev, cos_dev, sin_dev, (uint16_t *)y_dev,
        heads, seq, dim);
}