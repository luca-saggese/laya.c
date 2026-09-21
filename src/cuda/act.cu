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