/*
 * Laya-specific CUDA primitives -- the gap left after copying the o1.c
 * primitive set (norm/act/residual/embed/attn/rope).
 *
 * Everything here is adapted from an existing o1.c kernel:
 *   - laya_layernorm      <- hd_vision_layernorm (vision_kernels.cu)
 *   - laya_gelu           <- hd_vision_gelu_exact (vision_kernels.cu)
 *   - laya_geglu          <- hd_swiglu (act.cu), activation swapped
 *   - laya_rope_cos_sin   <- hd_mrope_cos_sin (rope.cu), single-section
 *   - laya_attn_mask      <- the host mask fill in runtime/sequence.c
 *   - laya_gather_pos     <- hd_gather_rows (embed.cu), int32 indices
 *   - laya_softmax_f32 / laya_top2_entropy <- new, tiny decision reductions
 */

#include "cuda_internal.h"

#include <math.h>
#include <stdio.h>

/* ------------------------------------------------------------------ */
/* LayerNorm (ModernBERT: eps 1e-5, norm_bias=false)                   */
/* ------------------------------------------------------------------ */

__global__ void laya_layernorm_kernel(const uint16_t *__restrict__ x,
                                      const uint16_t *__restrict__ w,
                                      const uint16_t *__restrict__ b,
                                      uint16_t *__restrict__ y,
                                      int rows, int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const uint16_t *xr = x + (size_t)row * cols;
    uint16_t *yr = y + (size_t)row * cols;
    int t = threadIdx.x;
    int nt = blockDim.x;
    float mean = 0.0f, var = 0.0f;
    for (int i = t; i < cols; i += nt) {
        float v = laya_dev_bf16_to_f32(xr[i]);
        mean += v;
        var += v * v;
    }
    __shared__ float s_m[256], s_v[256];
    s_m[t] = mean; s_v[t] = var;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) {
        if (t < s) { s_m[t] += s_m[t + s]; s_v[t] += s_v[t + s]; }
        __syncthreads();
    }
    if (t == 0) {
        float inv = 1.0f / (float)cols;
        s_m[0] *= inv;
        s_v[0] = s_v[0] * inv - s_m[0] * s_m[0];
    }
    __syncthreads();
    float inv_std = rsqrtf(s_v[0] + eps);
    for (int i = t; i < cols; i += nt) {
        float v = (laya_dev_bf16_to_f32(xr[i]) - s_m[0]) * inv_std;
        float wv = laya_dev_bf16_to_f32(w[i]);
        float bv = b ? laya_dev_bf16_to_f32(b[i]) : 0.0f;
        yr[i] = laya_dev_f32_to_bf16(v * wv + bv);
    }
}

void laya_layernorm(const void *x, const void *w, const void *b,
                    void *y, int rows, int cols, float eps) {
    if (!x || !w || !y || rows <= 0 || cols <= 0) {
        snprintf(laya_cuda_errbuf(), 512, "layernorm: bad args");
        return;
    }
    laya_layernorm_kernel<<<rows, 256>>>(
        (const uint16_t *)x, (const uint16_t *)w, (const uint16_t *)b,
        (uint16_t *)y, rows, cols, eps);
}

/* fp32 variant: the oracle keeps nn.LayerNorm in fp32 under autocast, so the
   embedding norm and every block norm run here with an fp32 weight and an fp32
   residual stream. Same reduction order as the bf16 kernel. */
__global__ void laya_layernorm_f32_kernel(const float *__restrict__ x,
                                          const float *__restrict__ w,
                                          const float *__restrict__ b,
                                          float *__restrict__ y,
                                          int rows, int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float *xr = x + (size_t)row * cols;
    float *yr = y + (size_t)row * cols;
    int t = threadIdx.x;
    int nt = blockDim.x;
    float mean = 0.0f, var = 0.0f;
    for (int i = t; i < cols; i += nt) {
        float v = xr[i];
        mean += v;
        var += v * v;
    }
    __shared__ float s_m[256], s_v[256];
    s_m[t] = mean; s_v[t] = var;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) {
        if (t < s) { s_m[t] += s_m[t + s]; s_v[t] += s_v[t + s]; }
        __syncthreads();
    }
    if (t == 0) {
        float inv = 1.0f / (float)cols;
        s_m[0] *= inv;
        s_v[0] = s_v[0] * inv - s_m[0] * s_m[0];
    }
    __syncthreads();
    float inv_std = rsqrtf(s_v[0] + eps);
    for (int i = t; i < cols; i += nt) {
        float v = (xr[i] - s_m[0]) * inv_std;
        float wv = w[i];
        float bv = b ? b[i] : 0.0f;
        yr[i] = v * wv + bv;
    }
}

void laya_layernorm_f32(const void *x, const void *w, const void *b,
                        void *y, int rows, int cols, float eps) {
    if (!x || !w || !y || rows <= 0 || cols <= 0) {
        snprintf(laya_cuda_errbuf(), 512, "layernorm_f32: bad args");
        return;
    }
    laya_layernorm_f32_kernel<<<rows, 256>>>(
        (const float *)x, (const float *)w, (const float *)b,
        (float *)y, rows, cols, eps);
}

/* ------------------------------------------------------------------ */
/* GELU (exact erf form; ModernBERT hidden_activation = "gelu")        */
/* ------------------------------------------------------------------ */

__global__ void laya_gelu_kernel(const uint16_t *__restrict__ x,
                                 uint16_t *__restrict__ y, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = laya_dev_bf16_to_f32(x[i]);
    float e = erff(v * 0.7071067811865476f);
    y[i] = laya_dev_f32_to_bf16(0.5f * v * (1.0f + e));
}

void laya_gelu(const void *x, void *y, size_t n) {
    if (!x || !y || n == 0) {
        snprintf(laya_cuda_errbuf(), 512, "gelu: bad args");
        return;
    }
    laya_gelu_kernel<<<(n + 255) / 256, 256>>>((const uint16_t *)x, (uint16_t *)y, n);
}

/* ------------------------------------------------------------------ */
/* GeGLU: y = gelu(gate) * up  (ModernBERT MLP)                        */
/* ------------------------------------------------------------------ */

__global__ void laya_geglu_kernel(const uint16_t *__restrict__ gate,
                                  const uint16_t *__restrict__ up,
                                  uint16_t *__restrict__ y, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = laya_dev_bf16_to_f32(gate[i]);
    float u = laya_dev_bf16_to_f32(up[i]);
    float e = erff(g * 0.7071067811865476f);
    float gelu_g = laya_dev_bf16_round(0.5f * g * (1.0f + e));
    y[i] = laya_dev_f32_to_bf16(gelu_g * u);
}

void laya_geglu(const void *gate, const void *up, void *y, size_t n) {
    if (!gate || !up || !y || n == 0) {
        snprintf(laya_cuda_errbuf(), 512, "geglu: bad args");
        return;
    }
    laya_geglu_kernel<<<(n + 255) / 256, 256>>>(
        (const uint16_t *)gate, (const uint16_t *)up, (uint16_t *)y, n);
}

/* ------------------------------------------------------------------ */
/* RoPE cos/sin table (ModernBERT: single section, cat(freqs, freqs))  */
/* ------------------------------------------------------------------ */

__global__ void laya_rope_cos_sin_kernel(float *__restrict__ cosd,
                                         float *__restrict__ sind,
                                         int seq, int dim, float theta) {
    int s = blockIdx.x;
    int k = threadIdx.x;
    if (s >= seq || k >= dim) return;
    int half = dim / 2;
    int kk = (k < half) ? k : (k - half);
    float inv = expf(-logf(theta) * (2.0f * (float)kk) / (float)dim);
    float ang = inv * (float)s;
    cosd[(size_t)s * dim + k] = cosf(ang);
    sind[(size_t)s * dim + k] = sinf(ang);
}

void laya_rope_cos_sin(float *cos_dev, float *sin_dev, int seq, int dim, float theta) {
    if (!cos_dev || !sin_dev || seq <= 0 || dim <= 0) {
        snprintf(laya_cuda_errbuf(), 512, "rope_cos_sin: bad args");
        return;
    }
    laya_rope_cos_sin_kernel<<<seq, 256>>>(cos_dev, sin_dev, seq, dim, theta);
}

/* ------------------------------------------------------------------ */
/* Attention mask [1,1,S,S] bf16, additive (0.0 / bf16 min)            */
/* ------------------------------------------------------------------ */

#define LAYA_BF16_MIN_BITS 0xFF7Fu

__global__ void laya_attn_mask_kernel(uint16_t *__restrict__ mask,
                                      int seq, int window, int sliding) {
    int r = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= seq || c >= seq) return;
    /* transformers: abs(q_idx - kv_idx) <= sliding_window, symmetric. */
    int keep = 1;
    if (sliding) { int d = c - r; if (d < 0) d = -d; keep = d <= window; }
    mask[(size_t)r * seq + c] = keep ? 0x0000u : LAYA_BF16_MIN_BITS;
}

void laya_attn_mask(void *mask_dev, int seq, int window, int sliding) {
    if (!mask_dev || seq <= 0) {
        snprintf(laya_cuda_errbuf(), 512, "attn_mask: bad args");
        return;
    }
    dim3 blk(256);
    dim3 grd((seq + 255) / 256, seq);
    laya_attn_mask_kernel<<<grd, blk>>>((uint16_t *)mask_dev, seq, window, sliding);
}

/* Batched variant: `batch` independent items flattened to rows = batch*seq.
 * The mask is stored as `batch` contiguous [seq,seq] blocks so that a
 * per-item attention call can take (mask + b*seq*seq), which is exactly the
 * slice layout laya_attn_scores_ws_kernel expects. Cross-item attention is
 * always masked, giving the block-diagonal structure the oracle's
 * [B,1,L,L] mask has.
 *
 * `valid` is a device uint8 [batch*seq] flag (1 = real token, NULL = all
 * real). A padded key column is masked for every query, which is what
 * src_key_padding_mask / attention_mask does in torch. */
__global__ void laya_attn_mask_batch_kernel(uint16_t *__restrict__ mask,
                                            const uint8_t *__restrict__ valid,
                                            int batch, int seq,
                                            int window, int sliding) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)batch * (size_t)seq * (size_t)seq;
    if (idx >= total) return;
    int key = (int)(idx % (size_t)seq);
    int query = (int)((idx / (size_t)seq) % (size_t)seq);
    int b = (int)(idx / ((size_t)seq * (size_t)seq));
    int keep = valid ? valid[(size_t)b * seq + key] != 0 : 1;
    if (sliding) {
        int d = key - query;
        if (d < 0) d = -d;
        keep = keep && (d <= window);
    }
    mask[idx] = keep ? 0x0000u : LAYA_BF16_MIN_BITS;
}

void laya_attn_mask_batch(void *mask_dev, const void *valid_dev, int batch,
                          int seq, int window, int sliding) {
    if (!mask_dev || batch <= 0 || seq <= 0) {
        snprintf(laya_cuda_errbuf(), 512, "attn_mask_batch: bad args");
        return;
    }
    size_t total = (size_t)batch * (size_t)seq * (size_t)seq;
    int block = 256;
    size_t grid = (total + block - 1) / block;
    laya_attn_mask_batch_kernel<<<(unsigned)grid, block>>>(
        (uint16_t *)mask_dev, (const uint8_t *)valid_dev, batch, seq, window,
        sliding);
}

/* ------------------------------------------------------------------ */
/* Gather selected sequence positions: out[i] = h[idx[i]]              */
/* ------------------------------------------------------------------ */

__global__ void laya_gather_pos_kernel(const uint16_t *__restrict__ h,
                                       const int *__restrict__ idx,
                                       uint16_t *__restrict__ out,
                                       int M, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    int src = idx[row];
    if (src < 0) src = 0;
    const uint16_t *s = h + (size_t)src * cols;
    uint16_t *d = out + (size_t)row * cols;
    for (int c = threadIdx.y; c < cols; c += blockDim.y) d[c] = s[c];
}

void laya_gather_pos(const void *h_dev, const void *idx_dev, void *out_dev,
                     int M, int cols) {
    if (!h_dev || !idx_dev || !out_dev || M <= 0 || cols <= 0) {
        snprintf(laya_cuda_errbuf(), 512, "gather_pos: bad args");
        return;
    }
    dim3 block(64, 8);
    dim3 grid((M + block.x - 1) / block.x);
    laya_gather_pos_kernel<<<grid, block>>>(
        (const uint16_t *)h_dev, (const int *)idx_dev, (uint16_t *)out_dev, M, cols);
}

/* ------------------------------------------------------------------ */
/* Decision reductions: softmax / top-2 / entropy over K logits        */
/* ------------------------------------------------------------------ */

/* One block per row; K is small (<= 16). */
__global__ void laya_softmax_f32_kernel(const float *__restrict__ logits,
                                        float *__restrict__ probs,
                                        int rows, int K) {
    int r = blockIdx.x;
    if (r >= rows) return;
    const float *lr = logits + (size_t)r * K;
    float *pr = probs + (size_t)r * K;
    __shared__ float red[32];
    int t = threadIdx.x;
    float mx = -INFINITY;
    for (int i = t; i < K; i += blockDim.x) mx = fmaxf(mx, lr[i]);
    red[t] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) red[t] = fmaxf(red[t], red[t + s]);
        __syncthreads();
    }
    float m = red[0];
    __syncthreads();
    float sum = 0.0f;
    for (int i = t; i < K; i += blockDim.x) sum += expf(lr[i] - m);
    red[t] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) red[t] += red[t + s];
        __syncthreads();
    }
    float inv = 1.0f / red[0];
    for (int i = t; i < K; i += blockDim.x) pr[i] = expf(lr[i] - m) * inv;
}

void laya_softmax_f32(const float *logits_dev, float *probs_dev, int rows, int K) {
    if (!logits_dev || !probs_dev || rows <= 0 || K <= 0) {
        snprintf(laya_cuda_errbuf(), 512, "softmax_f32: bad args");
        return;
    }
    laya_softmax_f32_kernel<<<rows, 32>>>(logits_dev, probs_dev, rows, K);
}

/* feats[r] = [top1, top1-top2, entropy/log(k), k/255] */
__global__ void laya_top2_entropy_kernel(const float *__restrict__ probs,
                                         float *__restrict__ feats,
                                         int rows, int K) {
    int r = blockIdx.x;
    if (r >= rows) return;
    const float *pr = probs + (size_t)r * K;
    float t1 = -INFINITY, t2 = -INFINITY, ent = 0.0f;
    for (int i = 0; i < K; i++) {
        float p = pr[i];
        if (p > t1) { t2 = t1; t1 = p; }
        else if (p > t2) { t2 = p; }
        float pc = p < 1e-9f ? 1e-9f : p;
        ent -= p * logf(pc);
    }
    float k = (float)(K < 2 ? 2 : K);
    float *f = feats + (size_t)r * 4;
    f[0] = t1;
    f[1] = t1 - t2;
    f[2] = ent / logf(k);
    f[3] = k / 255.0f;
}

void laya_top2_entropy(const float *probs_dev, float *feats_dev, int rows, int K) {
    if (!probs_dev || !feats_dev || rows <= 0 || K <= 0) {
        snprintf(laya_cuda_errbuf(), 512, "top2_entropy: bad args");
        return;
    }
    laya_top2_entropy_kernel<<<rows, 1>>>(probs_dev, feats_dev, rows, K);
}
/* ------------------------------------------------------------------ */
/* Layout: fused qkv [S,3,H,hd] -> head-major q/k/v [H,S,hd]           */
/* ------------------------------------------------------------------ */

/* ModernBERT does qkv.view(S, 3, H, hd).unbind(-3).transpose(1,2), i.e. the
 * fused projection is stored [S, 3, H, hd] and each of q/k/v becomes
 * [H, S, hd] -- exactly what laya_attention_eager consumes. */
__global__ void laya_qkv_split_kernel(const uint16_t *__restrict__ qkv,
                                      uint16_t *__restrict__ q,
                                      uint16_t *__restrict__ k,
                                      uint16_t *__restrict__ v,
                                      int seq, int heads, int hd) {
    long total = (long)seq * heads * hd;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int e = (int)(idx % hd);
    long sh = idx / hd;
    int h = (int)(sh % heads);
    int s = (int)(sh / heads);
    const uint16_t *src = qkv + ((long)s * 3 * heads + (long)h) * hd + e;
    long dst = ((long)h * seq + s) * hd + e;
    q[dst] = src[0];
    k[dst] = src[(long)heads * hd];
    v[dst] = src[2L * heads * hd];
}

void laya_qkv_split(const void *qkv_dev, void *q_dev, void *k_dev, void *v_dev,
                    int seq, int heads, int hd) {
    if (!qkv_dev || !q_dev || !k_dev || !v_dev || seq <= 0 || heads <= 0 || hd <= 0) {
        snprintf(laya_cuda_errbuf(), 512, "qkv_split: bad args");
        return;
    }
    long total = (long)seq * heads * hd;
    laya_qkv_split_kernel<<<(total + 255) / 256, 256>>>(
        (const uint16_t *)qkv_dev, (uint16_t *)q_dev, (uint16_t *)k_dev,
        (uint16_t *)v_dev, seq, heads, hd);
}

/* ------------------------------------------------------------------ */
/* Layout: Wi output [S, 2*I] -> (input, gate) halves of [S, I]        */
/* ------------------------------------------------------------------ */

/* ModernBertMLP: input, gate = Wi(h).chunk(2, -1); act(input) * gate.
 * The first half is the activation input, the second half is the gate. */
__global__ void laya_glu_split_kernel(const uint16_t *__restrict__ fused,
                                      uint16_t *__restrict__ input,
                                      uint16_t *__restrict__ gate,
                                      int seq, int inter) {
    long total = (long)seq * inter;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int c = (int)(idx % inter);
    long s = idx / inter;
    const uint16_t *src = fused + (s * 2 * inter) + c;
    input[idx] = src[0];
    gate[idx] = src[inter];
}

void laya_glu_split(const void *fused_dev, void *input_dev, void *gate_dev,
                    int seq, int inter) {
    if (!fused_dev || !input_dev || !gate_dev || seq <= 0 || inter <= 0) {
        snprintf(laya_cuda_errbuf(), 512, "glu_split: bad args");
        return;
    }
    long total = (long)seq * inter;
    laya_glu_split_kernel<<<(total + 255) / 256, 256>>>(
        (const uint16_t *)fused_dev, (uint16_t *)input_dev, (uint16_t *)gate_dev,
        seq, inter);
}

/* ------------------------------------------------------------------ */
/* Attention with caller-provided fp32 score scratch                   */
/* ------------------------------------------------------------------ */

/* Same math as laya_attention_eager but with the fp32 score scratch and the
 * head-major output supplied by the caller, so the steady-state forward does
 * no allocation at all (the o1.c workspace policy). */
__global__ void laya_attn_scores_ws_kernel(const uint16_t *__restrict__ q,
                                           const uint16_t *__restrict__ k,
                                           const uint16_t *__restrict__ mask,
                                           float *__restrict__ scores,
                                           int heads, int seq, int d, float scaling) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)heads * seq * seq;
    if (idx >= total) return;
    int t = (int)(idx % seq);
    long hs = idx / seq;
    int s = (int)(hs % seq);
    int h = (int)(hs / seq);
    const uint16_t *qr = q + ((long)h * seq + s) * d;
    const uint16_t *kr = k + ((long)h * seq + t) * d;
    float acc = 0.0f;
    for (int i = 0; i < d; i++) acc += laya_dev_bf16_to_f32(qr[i]) * laya_dev_bf16_to_f32(kr[i]);
    acc *= scaling;
    acc += laya_dev_bf16_to_f32(mask[(size_t)s * seq + t]);
    scores[idx] = acc;
}

/* softmax over t in fp32, then round to bf16 once (torch: softmax(dtype=f32).to(q.dtype)). */
__global__ void laya_attn_softmax_ws_kernel(const float *__restrict__ scores,
                                            uint16_t *__restrict__ probs,
                                            int heads, int seq) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)heads * seq) return;
    int s = (int)(idx % seq);
    int h = (int)(idx / seq);
    const float *row = scores + ((long)h * seq + s) * seq;
    uint16_t *pr = probs + ((long)h * seq + s) * seq;
    float mx = -INFINITY;
    for (int t = 0; t < seq; t++) mx = fmaxf(mx, row[t]);
    float sum = 0.0f;
    for (int t = 0; t < seq; t++) sum += expf(row[t] - mx);
    float inv = 1.0f / sum;
    for (int t = 0; t < seq; t++) pr[t] = laya_dev_f32_to_bf16(expf(row[t] - mx) * inv);
}

/* out[h,s,:] = sum_t probs[h,s,t] * v[h,t,:] (no GQA: ModernBERT has no MQA). */
__global__ void laya_attn_out_ws_kernel(const uint16_t *__restrict__ probs,
                                        const uint16_t *__restrict__ v,
                                        uint16_t *__restrict__ out,
                                        int heads, int seq, int d) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)heads * seq * d) return;
    int e = (int)(idx % d);
    long hs = idx / d;
    int s = (int)(hs % seq);
    int h = (int)(hs / seq);
    const uint16_t *pr = probs + ((long)h * seq + s) * seq;
    const uint16_t *vr = v + (long)h * seq * d + e;
    float acc = 0.0f;
    for (int t = 0; t < seq; t++) acc += laya_dev_bf16_to_f32(pr[t]) * laya_dev_bf16_to_f32(vr[(long)t * d]);
    out[idx] = laya_dev_f32_to_bf16(acc);
}

void laya_attention_ws(const void *q_dev, const void *k_dev, const void *v_dev,
                       const void *mask_dev, float *scores_dev, void *probs_dev,
                       void *out_dev, int heads, int seq, int dim, float scaling) {
    if (!q_dev || !k_dev || !v_dev || !mask_dev || !scores_dev || !probs_dev || !out_dev ||
        heads <= 0 || seq <= 0 || dim <= 0) {
        snprintf(laya_cuda_errbuf(), 512, "attention_ws: bad args");
        return;
    }
    long total_s = (long)heads * seq * seq;
    laya_attn_scores_ws_kernel<<<(total_s + 255) / 256, 256>>>(
        (const uint16_t *)q_dev, (const uint16_t *)k_dev, (const uint16_t *)mask_dev,
        scores_dev, heads, seq, dim, scaling);
    long rows = (long)heads * seq;
    laya_attn_softmax_ws_kernel<<<(rows + 255) / 256, 256>>>(
        scores_dev, (uint16_t *)probs_dev, heads, seq);
    long total_o = (long)heads * seq * dim;
    laya_attn_out_ws_kernel<<<(total_o + 255) / 256, 256>>>(
        (const uint16_t *)probs_dev, (const uint16_t *)v_dev, (uint16_t *)out_dev,
        heads, seq, dim);
}
