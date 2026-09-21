/*
 * M1.2 reference CUDA primitives -- RMSNorm.
 *
 * y[row, k] = w[k] * x * rsqrt(mean(x^2 over last dim) + eps)
 * Computed entirely in fp32; bf16 in/out (contract B).
 */

#include "cuda_internal.h"

#include <stdio.h>

__global__ void laya_rmsnorm_kernel(const uint16_t *__restrict__ x,
                                  const uint16_t *__restrict__ w,
                                  uint16_t *__restrict__ y,
                                  int rows, int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const uint16_t *xr = x + (size_t)row * cols;
    uint16_t *yr = y + (size_t)row * cols;

    int t = threadIdx.x;
    int nthreads = blockDim.x;

    /* Accumulate sum(x^2) over the row in fp32. */
    float sq = 0.0f;
    for (int i = t; i < cols; i += nthreads) {
        float xv = laya_dev_bf16_to_f32(xr[i]);
        sq += xv * xv;
    }
    __shared__ float s_sq[256];
    s_sq[t] = sq;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (t < s) s_sq[t] += s_sq[t + s];
        __syncthreads();
    }
    if (t == 0) s_sq[0] = rsqrtf(s_sq[0] / (float)cols + eps) ;
    __syncthreads();
    float inv_std = s_sq[0];

    for (int i = t; i < cols; i += nthreads) {
        float xv = laya_dev_bf16_to_f32(xr[i]);
        float wv = laya_dev_bf16_to_f32(w[i]);
        float v = wv * xv * inv_std;
        yr[i] = laya_dev_f32_to_bf16(v);
    }
}

void laya_rmsnorm(const void *x_dev, const void *w_dev, void *y_dev,
                int rows, int cols, float eps) {
    if (!x_dev || !w_dev || !y_dev || rows <= 0 || cols <= 0) {
        snprintf(laya_cuda_errbuf(), 512, "rmsnorm: bad args");
        return;
    }
    int nthreads = cols < 256 ? cols : 256;
    laya_rmsnorm_kernel<<<rows, nthreads>>>(
        (const uint16_t *)x_dev, (const uint16_t *)w_dev,
        (uint16_t *)y_dev, rows, cols, eps);
}