/*
 * M1.3 reference CUDA primitive -- residual add.
 *
 * Computes y[i] = x[i] + a[i] elementwise in bf16. The oracle adds a
 * bf16 residual to a bf16 branch (`residual + hidden`); torch elementwise
 * bf16 add accumulates in fp32 and rounds the result to bf16, which we
 * emulate (class B).
 */

#include "cuda_internal.h"

#include <stdio.h>

__global__ void laya_residual_add_kernel(const uint16_t *__restrict__ x,
                                       const uint16_t *__restrict__ a,
                                       uint16_t *__restrict__ y,
                                       size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xv = laya_dev_bf16_to_f32(x[i]);
    float av = laya_dev_bf16_to_f32(a[i]);
    y[i] = laya_dev_f32_to_bf16(xv + av);
}

void laya_residual_add(const void *x_dev, const void *a_dev, void *y_dev, size_t n) {
    if (!x_dev || !a_dev || !y_dev) {
        snprintf(laya_cuda_errbuf(), 512, "residual_add: bad args");
        return;
    }
    if (n == 0) return;
    size_t threads = 256;
    size_t blocks = (n + threads - 1) / threads;
    laya_residual_add_kernel<<<blocks, threads>>>(
        (const uint16_t *)x_dev, (const uint16_t *)a_dev, (uint16_t *)y_dev, n);
}

/* fp32 residual stream: the oracle's autocast leaves the residual adds in
 * fp32, so the encoder carries h in fp32 and only rounds at the Linear edges. */
__global__ void laya_residual_add_f32_kernel(const float *__restrict__ x,
                                             const float *__restrict__ a,
                                             float *__restrict__ y, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = x[i] + a[i];
}

void laya_residual_add_f32(const void *x_dev, const void *a_dev, void *y_dev, size_t n) {
    if (!x_dev || !a_dev || !y_dev) {
        snprintf(laya_cuda_errbuf(), 512, "residual_add_f32: bad args");
        return;
    }
    if (n == 0) return;
    size_t threads = 256;
    size_t blocks = (n + threads - 1) / threads;
    laya_residual_add_f32_kernel<<<blocks, threads>>>(
        (const float *)x_dev, (const float *)a_dev, (float *)y_dev, n);
}