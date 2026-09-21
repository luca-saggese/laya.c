#ifndef LAYA_CUDA_INTERNAL_H
#define LAYA_CUDA_INTERNAL_H

/*
 * Internal shared declarations for the M1.2 reference kernels.
 * Not part of the public ABI; consumed only by src/cuda/*.cu.
 */

#include "hd_cuda.h"

/* Shared error buffer (defined in support.cu). Must be inside extern "C"
 * so nvcc does not mangle the symbol (C callers link against it). */
#ifdef __cplusplus
extern "C" {
#endif
char *laya_cuda_errbuf(void);
#ifdef __cplusplus
}
#endif

/* BF16 <-> FP32 (host + device). */
__device__ __forceinline__ float laya_dev_bf16_to_f32(uint16_t b) {
    /* Reinterpret lower 16 bits into upper half of a fp32. */
    uint32_t u = ((uint32_t)b) << 16;
    float f;
    __builtin_memcpy(&f, &u, sizeof(f));
    return f;
}
__device__ __forceinline__ uint16_t laya_dev_f32_to_bf16(float f) {
    uint32_t u;
    __builtin_memcpy(&u, &f, sizeof(u));
    uint32_t lsb = (u >> 16) & 1u;
    uint32_t rounding_bias = 0x7FFFu + lsb;
    u += rounding_bias;
    return (uint16_t)(u >> 16);
}
/* bf16-round a float, keeping it as float (emulates torch bf16 per-op rounding). */
__device__ __forceinline__ float laya_dev_bf16_round(float f) {
    return laya_dev_bf16_to_f32(laya_dev_f32_to_bf16(f));
}

#endif /* LAYA_CUDA_INTERNAL_H */