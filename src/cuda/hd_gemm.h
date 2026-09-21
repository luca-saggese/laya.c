#ifndef LAYA_GEMM_H
#define LAYA_GEMM_H

/*
 * M2 production GEMM backend.
 *
 * laya_linear() dispatches to a persistent cuBLASLt runtime when one is
 * active (laya_gemm_runtime_init), otherwise falls back to the hand-written
 * reference kernel (laya_linear_reference). The runtime owns:
 *   - one cublasHandle_t / cublasLtHandle_t (created once, destroyed once)
 *   - a persistent workspace (no cudaMalloc/free in the forward)
 *   - a cached cuBLASLt matmul plan per (M,N,K) shape, with a tuned and
 *     cached algorithm selected once per shape
 *
 * The reference kernel is kept as the correctness/debug backend.
 */

#include <stddef.h>
#include <stdint.h>

#include "laya.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque persistent GEMM runtime. */
typedef struct laya_gemm_runtime laya_gemm_runtime;

/*
 * Creates the persistent cuBLAS/cuBLASLt handles and allocates the workspace.
 * Call ONCE at runtime init (e.g. inside laya_weights_to_device). Returns NULL
 * on failure (errbuf set via laya_cuda_errbuf).
 */
laya_gemm_runtime *laya_gemm_runtime_init(int device_id, size_t workspace_bytes);

/* Destroys the runtime and frees the workspace. Call ONCE at shutdown. */
void laya_gemm_runtime_destroy(laya_gemm_runtime *rt);

/* Returns the runtime's persistent workspace (may be NULL if 0 bytes). */
void *laya_gemm_workspace(laya_gemm_runtime *rt);
size_t laya_gemm_workspace_bytes(laya_gemm_runtime *rt);

/*
 * Reference linear (class C): y[M,N] = x[M,K] . w[N,K]^T, bf16 in/out,
 * fp32 accumulate. Hand-written tiled kernel. Kept for correctness/debug.
 */
void laya_linear_reference(const void *x_dev, const void *w_dev,
                         const void *bias_dev, void *y_dev,
                         int M, int N, int K, int transpose_w);

/*
 * Production linear: dispatches to the persistent cuBLAS/cuBLASLt backend
 * when the runtime is active, else falls back to laya_linear_reference.
 * Same contract as laya_linear_reference.
 */
void laya_linear(const void *x_dev, const void *w_dev, const void *bias_dev,
               void *y_dev, int M, int N, int K, int transpose_w);

/* Selects the backend used by laya_linear (default: production). */
void laya_gemm_set_backend(int use_production);

/*
 * Selects the production sub-backend: 0 = cuBLASLt (default, tuned+cached
 * per shape), 1 = cublasGemmEx (fallback/debug). Only meaningful when the
 * production backend is active.
 */
void laya_gemm_set_prod_backend(int backend);

#ifdef __cplusplus
}
#endif

#endif /* LAYA_GEMM_H */