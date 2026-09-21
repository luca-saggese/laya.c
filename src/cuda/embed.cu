/*
 * M1.4 embedding lookup primitives.
 *
 * Row-gather embedding kernel matching the oracle's
 * `nn.Embedding(vocab, hidden)` lookup for
 * model.language_model.embed_tokens. bf16 output rows indexed by int64
 * token ids. Caller provides device buffers; no allocation in forward.
 */

#include "cuda_internal.h"

#include <stdint.h>
#include <stdio.h>

__global__ void laya_gather_rows_kernel(
        const uint16_t *__restrict__ table, /* [rows, cols] bf16        */
        const int64_t *__restrict__ idx,    /* [M] int64 token ids       */
        uint16_t *__restrict__ out,         /* [M, cols] bf16            */
        int M, int cols, int64_t nrows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    int64_t id = idx[row];
    if (id < 0 || id >= nrows) id = 0;   /* padding index fallback */
    const uint16_t *src = table + id * cols;
    uint16_t *dst = out + row * cols;
    for (int c = threadIdx.y; c < cols; c += blockDim.y) {
        dst[c] = src[c];
    }
}

void laya_gather_rows(const void *table_dev, const void *idx_dev, void *out_dev,
                    int M, int cols, int64_t nrows) {
    if (!table_dev || !idx_dev || !out_dev || M <= 0 || cols <= 0) {
        snprintf(laya_cuda_errbuf(), 512, "gather_rows: bad args");
        return;
    }
    dim3 block(64, 8);
    dim3 grid((M + block.x - 1) / block.x);
    laya_gather_rows_kernel<<<grid, block>>>(
        (const uint16_t *)table_dev, (const int64_t *)idx_dev,
        (uint16_t *)out_dev, M, cols, nrows);
}

__global__ void laya_tms_condition_kernel(
        const int64_t *__restrict__ idx,  /* [M] token ids            */
        const uint16_t *__restrict__ emb, /* [M, H] bf16 text emb     */
        const uint16_t *__restrict__ t_emb, /* [H] bf16 timestep emb  */
        uint16_t *__restrict__ out,       /* [M, H] bf16              */
        int M, int H, int64_t tms_id) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    const uint16_t *src = (idx[row] == tms_id)
        ? t_emb : (emb + (size_t)row * H);
    uint16_t *dst = out + (size_t)row * H;
    for (int c = threadIdx.y; c < H; c += blockDim.y) {
        dst[c] = src[c];
    }
}

void laya_apply_tms_condition(const void *idx_dev, const void *emb_dev,
                            const void *t_emb_dev, void *out_dev,
                            int M, int H, int64_t tms_id) {
    if (!idx_dev || !emb_dev || !t_emb_dev || !out_dev || M <= 0 || H <= 0) {
        snprintf(laya_cuda_errbuf(), 512, "tms_condition: bad args");
        return;
    }
    dim3 block(64, 8);
    dim3 grid((M + block.x - 1) / block.x);
    laya_tms_condition_kernel<<<grid, block>>>(
        (const int64_t *)idx_dev, (const uint16_t *)emb_dev,
        (const uint16_t *)t_emb_dev, (uint16_t *)out_dev, M, H, tms_id);
}

/* ------------------------------------------------------------------ */
/* timestep scale (fp32) + f32->bf16 device convert used by the        */
/* t_embedder1 pipeline (oracle: t * 1000 then bf16 cast of freq)      */
/* ------------------------------------------------------------------ */

__global__ void laya_scale_f32_kernel(const float *__restrict__ in,
                                    float *__restrict__ out,
                                    float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * s;
}

void laya_scale_f32(const float *in_dev, float *out_dev, float s, int n) {
    if (!in_dev || !out_dev || n <= 0) {
        snprintf(laya_cuda_errbuf(), 512, "scale_f32: bad args");
        return;
    }
    laya_scale_f32_kernel<<<(n + 255) / 256, 256>>>(in_dev, out_dev, s, n);
}

__global__ void laya_f32_convert_bf16_kernel(const float *__restrict__ in,
                                           uint16_t *__restrict__ out,
                                           int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = laya_dev_f32_to_bf16(in[i]);
}

void laya_f32_convert_bf16(const float *in_dev, void *out_dev, int n) {
    if (!in_dev || !out_dev || n <= 0) {
        snprintf(laya_cuda_errbuf(), 512, "f32_convert_bf16: bad args");
        return;
    }
    laya_f32_convert_bf16_kernel<<<(n + 255) / 256, 256>>>(
        in_dev, (uint16_t *)out_dev, n);
}
