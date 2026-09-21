/*
 * M1.2 reference CUDA primitives -- shared support.
 *
 * BF16<->FP32 conversions and the shared module error string.
 */

#include "cuda_internal.h"

#include <stdio.h>
#include <string.h>

static char g_cuda_err[512] = "";

char *laya_cuda_errbuf(void) { return g_cuda_err; }

const char *laya_cuda_last_error(void) { return g_cuda_err; }
void laya_cuda_clear_error(void) { g_cuda_err[0] = '\0'; }

/* ---------- host BF16 helpers ---------- */

uint16_t laya_f32_to_bf16(float f) {
    uint32_t u;
    __builtin_memcpy(&u, &f, sizeof(u));
    uint32_t lsb = (u >> 16) & 1u;
    uint32_t rounding_bias = 0x7FFFu + lsb;
    u += rounding_bias;
    return (uint16_t)(u >> 16);
}

float laya_bf16_to_f32(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    float f;
    __builtin_memcpy(&f, &u, sizeof(f));
    return f;
}

void laya_bf16_buf_to_f32(const void *src, float *dst, size_t n) {
    const uint16_t *s = (const uint16_t *)src;
    for (size_t i = 0; i < n; i++) dst[i] = laya_bf16_to_f32(s[i]);
}

void laya_f32_buf_to_bf16(const float *src, void *dst, size_t n) {
    uint16_t *d = (uint16_t *)dst;
    for (size_t i = 0; i < n; i++) d[i] = laya_f32_to_bf16(src[i]);
}

/* ---------- device FP32<->BF16 elementwise casts ---------- */

__global__ void laya_dev_cast_f32_to_bf16_kernel(const float *__restrict__ in,
                                                 uint16_t *__restrict__ out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = laya_dev_f32_to_bf16(in[i]);
}

__global__ void laya_dev_cast_bf16_to_f32_kernel(const uint16_t *__restrict__ in,
                                                 float *__restrict__ out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = laya_dev_bf16_to_f32(in[i]);
}

void laya_cast_f32_to_bf16(const void *in_dev, void *out_dev, size_t n) {
    if (!in_dev || !out_dev || n == 0) return;
    laya_dev_cast_f32_to_bf16_kernel<<<(unsigned)((n + 255) / 256), 256>>>(
        (const float *)in_dev, (uint16_t *)out_dev, n);
}

void laya_cast_bf16_to_f32(const void *in_dev, void *out_dev, size_t n) {
    if (!in_dev || !out_dev || n == 0) return;
    laya_dev_cast_bf16_to_f32_kernel<<<(unsigned)((n + 255) / 256), 256>>>(
        (const uint16_t *)in_dev, (float *)out_dev, n);
}