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