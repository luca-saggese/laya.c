/*
 * M1.2 reference CUDA primitives -- eager attention and head layout ops.
 *
 * Eager attention (class D): q[heads,seq,d], k/v[kv_heads,seq,d],
 * mask[1,1,seq,seq]. Computes scores = q . k^T * scaling (+mask), softmax in
 * fp32 over the last dim, probs . v, then transposes to seq-major out
 * [seq, heads, d] (matching the oracle's eager_attention_forward transpose).
 *
 * head_split / head_merge (class A): byte-exact permutations.
 */

#include "cuda_internal.h"

#include <stdio.h>

/* ------------------------------------------------------------------ */
/* Head-major layout helpers                                           */
/* ------------------------------------------------------------------ */

/* q layout [heads, seq, d]; k/v layout [kv_heads, seq, d]. */

__global__ void laya_attn_scores_kernel(
    const uint16_t *__restrict__ q,      /* [heads, seq, d] */
    const uint16_t *__restrict__ k,      /* [kv_heads, seq, d] */
    const uint16_t *__restrict__ mask,   /* [1,1,seq,seq] */
    float *__restrict__ scores,          /* [heads, seq, seq] fp32 accum */
    int heads, int kv_heads, int seq, int d, float scaling) {
    /* One thread computes one scores[h, s, t]. */
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)heads * seq * seq;
    if (idx >= total) return;
    int t = idx % seq;
    long hs = idx / seq;
    int s = (int)(hs % seq);
    int h = (int)(hs / seq);

    /* KV head for GQA: repeat_interleave makes kv_head = h / groups. */
    int kh = h / (heads / kv_heads);

    const uint16_t *qr = q + ((long)h * seq + s) * d;
    const uint16_t *kr = k + ((long)kh * seq + t) * d;

    float acc = 0.0f;
    for (int i = 0; i < d; i += 4) {
        int lim = (i + 4 <= d) ? i + 4 : d;
        float qb[4], kb[4];
        for (int j = 0; j < lim - i; j++) {
            qb[j] = laya_dev_bf16_to_f32(qr[i + j]);
            kb[j] = laya_dev_bf16_to_f32(kr[i + j]);
        }
        for (int j = 0; j < lim - i; j++) acc += qb[j] * kb[j];
    }
    acc *= scaling;

    /* Mask: add mask[0,0,s,t] (0.0 allowed, -3.3895e38 masked). */
    float mv = laya_dev_bf16_to_f32(mask[(size_t)s * seq + t]);
    acc += mv;

    scores[idx] = acc;
}

__global__ void laya_attn_softmax_kernel(
    const float *__restrict__ scores,      /* [heads, seq, seq] fp32 */
    uint16_t *__restrict__ scores_out,     /* [heads, seq, seq] bf16 (pre-softmax) */
    uint16_t *__restrict__ probs,          /* [heads, seq, seq] bf16 (softmax) */
    int heads, int seq) {
    /* Softmax over t for each (h,s). */
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long rows = (long)heads * seq;
    if (idx >= rows) return;
    int s = (int)(idx % seq);
    int h = (int)(idx / seq);
    const float *row = scores + ((long)h * seq + s) * seq;
    uint16_t *sr = scores_out + ((long)h * seq + s) * seq;
    uint16_t *pr = probs + ((long)h * seq + s) * seq;

    for (int t = 0; t < seq; t++) sr[t] = laya_dev_f32_to_bf16(row[t]);

    float mx = -INFINITY;
    for (int t = 0; t < seq; t++) if (row[t] > mx) mx = row[t];
    float sum = 0.0f;
    for (int t = 0; t < seq; t++) sum += expf(row[t] - mx);
    float inv = 1.0f / sum;
    for (int t = 0; t < seq; t++) pr[t] = laya_dev_f32_to_bf16(expf(row[t] - mx) * inv);
}

/* attn_out[h, s, d] = sum_t probs[h,s,t] * v[kh, t, d]. Then the harness copies
 * to seq-major out[s, h, d]. We write attn_out already in [h, s, d] and let the
 * merge kernel do the transpose. */
__global__ void laya_attn_out_kernel(
    const uint16_t *__restrict__ probs,  /* [heads, seq, seq] */
    const uint16_t *__restrict__ v,      /* [kv_heads, seq, d] */
    uint16_t *__restrict__ out,          /* [heads, seq, d] (head-major) */
    int heads, int kv_heads, int seq, int d) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)heads * seq * d;
    if (idx >= total) return;
    int dd = idx % d;
    long hs = idx / d;
    int s = (int)(hs % seq);
    int h = (int)(hs / seq);
    int kh = h / (heads / kv_heads);   /* GQA group */

    const uint16_t *pr = probs + ((long)h * seq + s) * seq;
    const uint16_t *kv = v + ((long)kh * seq * d) + dd;  /* [kv_heads,seq,d] stride d */

    float acc = 0.0f;
    for (int t = 0; t < seq; t++) {
        acc += laya_dev_bf16_to_f32(pr[t]) * laya_dev_bf16_to_f32(kv[(long)t * d]);
    }
    out[idx] = laya_dev_f32_to_bf16(acc);
}

/* seq-major out [seq, heads, d] from head-major [heads, seq, d]. */
__global__ void laya_attn_transpose_kernel(
    const uint16_t *__restrict__ hm,   /* [heads, seq, d] */
    uint16_t *__restrict__ sm,         /* [seq, heads, d] */
    int heads, int seq, int d) {
    long total = (long)heads * seq * d;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int dd = idx % d;
    long hs = idx / d;
    int s = (int)(hs % seq);
    int h = (int)(hs / seq);
    sm[(long)s * heads * d + h * d + dd] = hm[(long)h * seq * d + s * d + dd];
}

void laya_attention_eager(const void *q_dev, const void *k_dev, const void *v_dev,
                        const void *mask_dev, void *scores_dev,
                        void *probs_dev, void *out_dev,
                        int heads, int kv_heads, int seq, int dim,
                        float scaling) {
    if (!q_dev || !k_dev || !v_dev || !mask_dev || !scores_dev ||
        !probs_dev || !out_dev || heads <= 0 || kv_heads <= 0 || seq <= 0 || dim <= 0) {
        snprintf(laya_cuda_errbuf(), 512, "attention_eager: bad args");
        return;
    }
    int groups = heads / kv_heads;
    /* scores as fp32 scratch (softmax in fp32), probs as bf16. */
    float *scratch = NULL;
    cudaError_t e = cudaMalloc(&scratch, (size_t)heads * seq * seq * sizeof(float));
    if (e != cudaSuccess) { snprintf(laya_cuda_errbuf(), 512, "attn scratch: %s", cudaGetErrorString(e)); return; }

    long total_s = (long)heads * seq * seq;
    long blocks_s = (total_s + 255) / 256;
    laya_attn_scores_kernel<<<blocks_s, 256>>>(
        (const uint16_t *)q_dev, (const uint16_t *)k_dev, (const uint16_t *)mask_dev,
        scratch, heads, kv_heads, seq, dim, scaling);

    long rows = (long)heads * seq;
    laya_attn_softmax_kernel<<<(rows + 255) / 256, 256>>>(
        scratch, (uint16_t *)scores_dev, (uint16_t *)probs_dev, heads, seq);

    long total_o = (long)heads * seq * dim;
    uint16_t *hm = NULL;
    e = cudaMalloc(&hm, (size_t)((long)heads * seq * dim) * sizeof(uint16_t));
    if (e != cudaSuccess) { cudaFree(scratch); snprintf(laya_cuda_errbuf(), 512, "attn hm: %s", cudaGetErrorString(e)); return; }
    laya_attn_out_kernel<<<(total_o + 255) / 256, 256>>>(
        (const uint16_t *)probs_dev, (const uint16_t *)v_dev, hm,
        heads, kv_heads, seq, dim);
    laya_attn_transpose_kernel<<<(total_o + 255) / 256, 256>>>(
        hm, (uint16_t *)out_dev, heads, seq, dim);

    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        snprintf(laya_cuda_errbuf(), 512, "attn sync: %s", cudaGetErrorString(e));
        cudaFree(hm); cudaFree(scratch);
        return;
    }
    cudaFree(hm); cudaFree(scratch);
    (void)groups;
}

/* ------------------------------------------------------------------ */
/* head_split / head_merge (class A, bit-exact)                        */
/* ------------------------------------------------------------------ */

/* in [1, seq, heads, d] -> out [heads, seq, d]. */
__global__ void laya_head_split_kernel(const uint16_t *__restrict__ in,
                                     uint16_t *__restrict__ out,
                                     int seq, int heads, int d) {
    long total = (long)seq * heads * d;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int dd = idx % d;
    long sh = idx / d;
    int s = (int)(sh % seq);
    int h = (int)(sh / seq);
    out[(long)h * seq * d + s * d + dd] = in[(long)s * heads * d + h * d + dd];
}

/* in [heads, seq, d] -> out [1, seq, heads, d]. */
__global__ void laya_head_merge_kernel(const uint16_t *__restrict__ in,
                                     uint16_t *__restrict__ out,
                                     int seq, int heads, int d) {
    long total = (long)seq * heads * d;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int dd = idx % d;
    long sh = idx / d;
    int s = (int)(sh % seq);
    int h = (int)(sh / seq);
    out[(long)s * heads * d + h * d + dd] = in[(long)h * seq * d + s * d + dd];
}

void laya_head_split(const void *in_dev, void *out_dev, int seq, int heads, int dim) {
    if (!in_dev || !out_dev) { snprintf(laya_cuda_errbuf(), 512, "head_split: bad args"); return; }
    long total = (long)seq * heads * dim;
    laya_head_split_kernel<<<(total + 255) / 256, 256>>>(
        (const uint16_t *)in_dev, (uint16_t *)out_dev, seq, heads, dim);
}

void laya_head_merge(const void *in_dev, void *out_dev, int seq, int heads, int dim) {
    if (!in_dev || !out_dev) { snprintf(laya_cuda_errbuf(), 512, "head_merge: bad args"); return; }
    long total = (long)seq * heads * dim;
    laya_head_merge_kernel<<<(total + 255) / 256, 256>>>(
        (const uint16_t *)in_dev, (uint16_t *)out_dev, seq, heads, dim);
}