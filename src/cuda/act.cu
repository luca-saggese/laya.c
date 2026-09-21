/*
 * M1.2 reference CUDA primitives -- activation functions.
 *
 * SiLU:   y = x * sigmoid(x)
 * SwiGLU: y = silu(gate) * up
 * Elementwise bf16 in/out (contract B).
 */

#include "cuda_internal.h"

#include <stdio.h>

__global__ void laya_silu_kernel(const uint16_t *__restrict__ x,
                               uint16_t *__restrict__ y, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xv = laya_dev_bf16_to_f32(x[i]);
    float s = 1.0f / (1.0f + expf(-xv));
    y[i] = laya_dev_f32_to_bf16(xv * s);
}

__global__ void laya_swiglu_kernel(const uint16_t *__restrict__ gate,
                                 const uint16_t *__restrict__ up,
                                 uint16_t *__restrict__ y, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = laya_dev_bf16_to_f32(gate[i]);
    float u = laya_dev_bf16_to_f32(up[i]);
    /* Oracle: act_fn(gate) * up where act_fn = SiLU. Torch bf16 rounds each
     * op result to bf16; do the same: silu(g) as bf16, then bf16 product. */
    float sg = 1.0f / (1.0f + expf(-g));
    float silu_g = laya_dev_bf16_round(g * sg);
    y[i] = laya_dev_f32_to_bf16(silu_g * u);
}

void laya_silu(const void *x_dev, void *y_dev, size_t n) {
    if (!x_dev || !y_dev) { snprintf(laya_cuda_errbuf(), 512, "silu: bad args"); return; }
    if (n == 0) return;
    size_t threads = 256;
    size_t blocks = (n + threads - 1) / threads;
    laya_silu_kernel<<<blocks, threads>>>((const uint16_t *)x_dev, (uint16_t *)y_dev, n);
}

void laya_swiglu(const void *gate_dev, const void *up_dev, void *y_dev, size_t n) {
    if (!gate_dev || !up_dev || !y_dev) {
        snprintf(laya_cuda_errbuf(), 512, "swiglu: bad args");
        return;
    }
    if (n == 0) return;
    size_t threads = 256;
    size_t blocks = (n + threads - 1) / threads;
    laya_swiglu_kernel<<<blocks, threads>>>(
        (const uint16_t *)gate_dev, (const uint16_t *)up_dev, (uint16_t *)y_dev, n);
}
/* ------------------------------------------------------------------ */
/* Laya decision head: ReLU and type-embedding broadcast add.          */
/* ------------------------------------------------------------------ */

/* ReLU (class B): y = max(x, 0), bf16 elementwise. The Laya decision-head
 * TransformerEncoderLayer uses the PyTorch default activation (ReLU), NOT the
 * GELU used by the scorer / action head. */
__global__ void laya_relu_kernel(const uint16_t *__restrict__ x,
                                 uint16_t *__restrict__ y, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = laya_dev_bf16_to_f32(x[i]);
    y[i] = laya_dev_f32_to_bf16(v > 0.0f ? v : 0.0f);
}

void laya_relu(const void *x_dev, void *y_dev, size_t n) {
    if (!x_dev || !y_dev) { snprintf(laya_cuda_errbuf(), 512, "relu: bad args"); return; }
    if (n == 0) return;
    size_t threads = 256;
    size_t blocks = (n + threads - 1) / threads;
    laya_relu_kernel<<<blocks, threads>>>((const uint16_t *)x_dev, (uint16_t *)y_dev, n);
}

/* Broadcast add of one [cols] row across `rows` rows of a [rows, cols] tensor:
 * y[r, c] = x[r, c] + row[c]. Reproduces
 * `h + type_emb(qtype)[:, None, :]` for a single-item batch. */
__global__ void laya_bcast_add_kernel(const uint16_t *__restrict__ x,
                                      const uint16_t *__restrict__ row,
                                      uint16_t *__restrict__ y,
                                      int rows, int cols) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cols) return;
    float rv = laya_dev_bf16_to_f32(row[c]);
    for (int r = 0; r < rows; r++) {
        size_t i = (size_t)r * cols + c;
        y[i] = laya_dev_f32_to_bf16(laya_dev_bf16_to_f32(x[i]) + rv);
    }
}

void laya_bcast_add(const void *x_dev, const void *row_dev, void *y_dev,
                    int rows, int cols) {
    if (!x_dev || !row_dev || !y_dev || rows <= 0 || cols <= 0) {
        snprintf(laya_cuda_errbuf(), 512, "bcast_add: bad args");
        return;
    }
    int threads = 256;
    laya_bcast_add_kernel<<<(cols + threads - 1) / threads, threads>>>(
        (const uint16_t *)x_dev, (const uint16_t *)row_dev, (uint16_t *)y_dev,
        rows, cols);
}
