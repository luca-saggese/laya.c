/*
 * M2 production GEMM backend.
 *
 * laya_linear() dispatches to a persistent cuBLASLt runtime when one is active
 * (laya_gemm_runtime_init), otherwise falls back to the hand-written reference
 * kernel (laya_linear_reference). The reference kernel is kept as the
 * correctness/debug backend (class C: bf16 in/out, fp32 accumulate).
 *
 * The production path is cuBLASLt with explicit row-major layouts, BF16
 * in/out, FP32 compute and tensor cores. For every unique (M,N,K) shape the
 * runtime selects and caches a cuBLASLt algorithm (tuned once with CUDA
 * events, validated against cublasGemmEx), so the forward hot path does no
 * handle creation, no heuristic search, no descriptor churn and no
 * cudaMalloc/free. cublasGemmEx is retained as the fallback/debug backend.
 */

#include "cuda_internal.h"
#include "hd_gemm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>
#include <time.h>
#include <cublas_v2.h>
#include <cublasLt.h>

/* ------------------------------------------------------------------ */
/* Reference tiled matmul y = x @ w^T (w stored [N,K])                 */
/* ------------------------------------------------------------------ */

#define LAYA_GEMM_TILE 16

__global__ void laya_gemm_wT_kernel(const uint16_t *__restrict__ x,   /* [M,K] */
                                  const uint16_t *__restrict__ w,   /* [N,K] */
                                  const uint16_t *__restrict__ bias,/* [N] */
                                  uint16_t *__restrict__ y,         /* [M,N] */
                                  int M, int N, int K) {
    __shared__ float xs[LAYA_GEMM_TILE][LAYA_GEMM_TILE];
    __shared__ float ws[LAYA_GEMM_TILE][LAYA_GEMM_TILE];

    int row = blockIdx.x * LAYA_GEMM_TILE + threadIdx.x; /* M dim */
    int col = blockIdx.y * LAYA_GEMM_TILE + threadIdx.y; /* N dim */
    int txi = threadIdx.x, tyi = threadIdx.y;

    float acc = 0.0f;

    for (int kk = 0; kk < K; kk += LAYA_GEMM_TILE) {
        /* Load x tile [LAYA_GEMM_TILE x LAYA_GEMM_TILE] at (row, kk). */
        int gx = kk + tyi;             /* K coord */
        xs[txi][tyi] = (row < M && gx < K) ? laya_dev_bf16_to_f32(x[(size_t)row * K + gx]) : 0.0f;
        /* Load w tile [LAYA_GEMM_TILE x LAYA_GEMM_TILE] at (col, kk) of stored [N,K]. */
        int gw = blockIdx.y * LAYA_GEMM_TILE + txi;  /* N coord */
        int gk = kk + tyi;                          /* K coord */
        ws[txi][tyi] = (gw < N && gk < K) ? laya_dev_bf16_to_f32(w[(size_t)gw * K + gk]) : 0.0f;
        __syncthreads();

        if (row < M && col < N) {
            for (int k = 0; k < LAYA_GEMM_TILE; k++) {
                acc += xs[txi][k] * ws[tyi][k];
            }
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        if (bias) acc += laya_dev_bf16_to_f32(bias[col]);
        y[(size_t)row * N + col] = laya_dev_f32_to_bf16(acc);
    }
}

void laya_linear_reference(const void *x_dev, const void *w_dev,
                         const void *bias_dev, void *y_dev,
                         int M, int N, int K, int transpose_w) {
    if (!x_dev || !w_dev || !y_dev || M <= 0 || N <= 0 || K <= 0) {
        snprintf(laya_cuda_errbuf(), 512, "linear: bad args");
        return;
    }
    if (!transpose_w) {
        snprintf(laya_cuda_errbuf(), 512, "linear: transpose_w=0 unsupported");
        return;
    }
    dim3 block(LAYA_GEMM_TILE, LAYA_GEMM_TILE);
    dim3 grid((M + LAYA_GEMM_TILE - 1) / LAYA_GEMM_TILE,
              (N + LAYA_GEMM_TILE - 1) / LAYA_GEMM_TILE);
    laya_gemm_wT_kernel<<<grid, block>>>(
        (const uint16_t *)x_dev, (const uint16_t *)w_dev,
        (const uint16_t *)bias_dev, (uint16_t *)y_dev, M, N, K);
}

/* ------------------------------------------------------------------ */
/* Persistent cuBLAS/cuBLASLt runtime                                  */
/* ------------------------------------------------------------------ */

/* One cached cuBLASLt plan per unique (M,N,K) production shape. The
 * descriptors are shape-only (they carry no data pointers), so they are
 * created once and reused for every block and every denoise step. */
#define HD_LT_MAX_PLANS 32

typedef struct {
    int M, N, K;
    int valid;
    float tune_ms;   /* one-time algorithm-selection cost */
    int tuned;
    cublasLtMatmulAlgo_t algo;
    size_t workspace_bytes;
    cublasLtMatrixLayout_t Adesc; /* X [M,K] row-major, ld=K */
    cublasLtMatrixLayout_t Bdesc; /* W [N,K] row-major, ld=K, op=T */
    cublasLtMatrixLayout_t Cdesc; /* Y [M,N] row-major, ld=N */
    cublasLtMatmulDesc_t opdesc;
} laya_lt_plan;

struct laya_gemm_runtime {
    cublasHandle_t cublas;
    cublasLtHandle_t lt;
    void *workspace;
    size_t workspace_bytes;
    int device_id;
    cublasLtMatmulPreference_t pref;
    laya_lt_plan plans[HD_LT_MAX_PLANS];
    int n_plans;
    int tune;  /* 1 = benchmark candidates with CUDA events, 0 = heuristic[0] */
    int debug; /* 1 = print the selected algorithm per shape */
};

/* Active runtime (NULL = reference backend). Set by laya_gemm_runtime_init. */
static laya_gemm_runtime *g_rt = NULL;
/* Backend selection: 1 = production (cuBLAS/cuBLASLt), 0 = reference. */
static int g_use_production = 1;
/* Production sub-backend: 0 = cuBLASLt (default), 1 = cublasGemmEx. */
static int g_prod_backend = 0;

laya_gemm_runtime *laya_gemm_runtime_init(int device_id, size_t workspace_bytes) {
    laya_gemm_runtime *rt = (laya_gemm_runtime *)calloc(1, sizeof(laya_gemm_runtime));
    if (!rt) { snprintf(laya_cuda_errbuf(), 512, "gemm rt oom"); return NULL; }
    rt->device_id = device_id;

    cudaError_t ce = cudaSetDevice(device_id);
    if (ce != cudaSuccess) {
        snprintf(laya_cuda_errbuf(), 512, "gemm rt cudaSetDevice: %s", cudaGetErrorString(ce));
        free(rt);
        return NULL;
    }
    cublasStatus_t cs = cublasCreate(&rt->cublas);
    if (cs != CUBLAS_STATUS_SUCCESS) {
        snprintf(laya_cuda_errbuf(), 512, "gemm rt cublasCreate failed");
        free(rt);
        return NULL;
    }
    cublasStatus_t ls = cublasLtCreate(&rt->lt);
    if (ls != CUBLAS_STATUS_SUCCESS) {
        snprintf(laya_cuda_errbuf(), 512, "gemm rt cublasLtCreate failed");
        cublasDestroy(rt->cublas);
        free(rt);
        return NULL;
    }
    if (workspace_bytes > 0) {
        ce = cudaMalloc(&rt->workspace, workspace_bytes);
        if (ce != cudaSuccess) {
            snprintf(laya_cuda_errbuf(), 512, "gemm rt workspace cudaMalloc: %s",
                     cudaGetErrorString(ce));
            cublasLtDestroy(rt->lt);
            cublasDestroy(rt->cublas);
            free(rt);
            return NULL;
        }
        rt->workspace_bytes = workspace_bytes;
    }
    cublasLtMatmulPreference_t pref = NULL;
    if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) {
        snprintf(laya_cuda_errbuf(), 512, "gemm rt preference create failed");
        if (rt->workspace) cudaFree(rt->workspace);
        cublasLtDestroy(rt->lt);
        cublasDestroy(rt->cublas);
        free(rt);
        return NULL;
    }
    cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &rt->workspace_bytes, sizeof(rt->workspace_bytes));
    rt->pref = pref;
    rt->n_plans = 0;
    /* Tuning is on by default. The heuristic's first candidate is ~1% faster
     * only because it skips the one-time selection, but it does not reproduce
     * the cublasGemmEx result (cos~0.88 vs the accepted ~0.99), whereas the
     * tuned selection is bit-exact. Steady-state throughput is identical, so
     * correctness wins: tune by default, O1_GEMM_TUNE=0 for heuristic[0].
     * O1_GEMM_DEBUG=1 prints the selected algorithm per shape. */
    const char *tune = getenv("O1_GEMM_TUNE");
    rt->tune = (tune && tune[0] == '0') ? 0 : 1;
    const char *dbg = getenv("O1_GEMM_DEBUG");
    rt->debug = (dbg && dbg[0] == '1') ? 1 : 0;
    /* O1_GEMM_BACKEND=ex forces the cublasGemmEx sub-backend (A/B testing). */
    const char *be = getenv("O1_GEMM_BACKEND");
    if (be && be[0] == 'e') g_prod_backend = 1;
    g_rt = rt;
    return rt;
}

void laya_gemm_runtime_destroy(laya_gemm_runtime *rt) {
    if (!rt) return;
    if (g_rt == rt) g_rt = NULL;
    for (int i = 0; i < rt->n_plans; i++) {
        laya_lt_plan *p = &rt->plans[i];
        if (p->opdesc) cublasLtMatmulDescDestroy(p->opdesc);
        if (p->Adesc) cublasLtMatrixLayoutDestroy(p->Adesc);
        if (p->Bdesc) cublasLtMatrixLayoutDestroy(p->Bdesc);
        if (p->Cdesc) cublasLtMatrixLayoutDestroy(p->Cdesc);
    }
    if (rt->pref) cublasLtMatmulPreferenceDestroy(rt->pref);
    if (rt->workspace) cudaFree(rt->workspace);
    if (rt->lt) cublasLtDestroy(rt->lt);
    if (rt->cublas) cublasDestroy(rt->cublas);
    free(rt);
}

void *laya_gemm_workspace(laya_gemm_runtime *rt) { return rt ? rt->workspace : NULL; }
size_t laya_gemm_workspace_bytes(laya_gemm_runtime *rt) { return rt ? rt->workspace_bytes : 0; }

void laya_gemm_set_backend(int use_production) { g_use_production = use_production; }

/* Production sub-backend: 0 = cuBLASLt (default), 1 = cublasGemmEx. */
void laya_gemm_set_prod_backend(int backend) { g_prod_backend = backend; }

__global__ void laya_gemm_bias_add_kernel(const uint16_t *__restrict__ bias,
                                        uint16_t *__restrict__ y,
                                        int M, int N) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)M * N;
    if (idx >= total) return;
    int n = (int)(idx % N);
    float b = laya_dev_bf16_to_f32(bias[n]);
    float v = laya_dev_bf16_to_f32(y[idx]);
    y[idx] = laya_dev_f32_to_bf16(v + b);
}

/* ------------------------------------------------------------------ */
/* cuBLASLt plan cache + algorithm selection                           */
/* ------------------------------------------------------------------ */
/*
 * The production GEMM contract is fixed:
 *   Y[M,N] = X[M,K] . W[N,K]^T   (row-major X, W, Y)
 *   X,W,Y: BF16   compute: FP32   epilogue: default (bias added separately)
 * so (M,N,K) uniquely identifies an execution configuration. There are only
 * a handful of unique shapes per generation, and they repeat across all 36
 * blocks and all 28 denoise steps, so the plan (descriptors + selected
 * algorithm + workspace) is created and tuned once and then reused.
 */

static laya_lt_plan *laya_lt_get_plan(laya_gemm_runtime *rt, int M, int N, int K) {
    for (int i = 0; i < rt->n_plans; i++) {
        laya_lt_plan *p = &rt->plans[i];
        if (p->valid && p->M == M && p->N == N && p->K == K) return p;
    }
    if (rt->n_plans >= HD_LT_MAX_PLANS) return NULL;

    laya_lt_plan *p = &rt->plans[rt->n_plans];
    memset(p, 0, sizeof(*p));
    p->M = M; p->N = N; p->K = K;

    int32_t row = CUBLASLT_ORDER_ROW;
    cublasStatus_t st;
    st = cublasLtMatrixLayoutCreate(&p->Adesc, CUDA_R_16BF, M, K, K);
    if (st == CUBLAS_STATUS_SUCCESS)
        st = cublasLtMatrixLayoutSetAttribute(p->Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                              &row, sizeof(row));
    if (st == CUBLAS_STATUS_SUCCESS)
        st = cublasLtMatrixLayoutCreate(&p->Bdesc, CUDA_R_16BF, N, K, K);
    if (st == CUBLAS_STATUS_SUCCESS)
        st = cublasLtMatrixLayoutSetAttribute(p->Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                              &row, sizeof(row));
    if (st == CUBLAS_STATUS_SUCCESS)
        st = cublasLtMatrixLayoutCreate(&p->Cdesc, CUDA_R_16BF, M, N, N);
    if (st == CUBLAS_STATUS_SUCCESS)
        st = cublasLtMatrixLayoutSetAttribute(p->Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                              &row, sizeof(row));
    if (st == CUBLAS_STATUS_SUCCESS)
        st = cublasLtMatmulDescCreate(&p->opdesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (st == CUBLAS_STATUS_SUCCESS) {
        cublasOperation_t opn = CUBLAS_OP_N, opt = CUBLAS_OP_T;
        st = cublasLtMatmulDescSetAttribute(p->opdesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                            &opn, sizeof(opn));
        if (st == CUBLAS_STATUS_SUCCESS)
            st = cublasLtMatmulDescSetAttribute(p->opdesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                                &opt, sizeof(opt));
    }
    if (st != CUBLAS_STATUS_SUCCESS) {
        if (p->opdesc) cublasLtMatmulDescDestroy(p->opdesc);
        if (p->Adesc) cublasLtMatrixLayoutDestroy(p->Adesc);
        if (p->Bdesc) cublasLtMatrixLayoutDestroy(p->Bdesc);
        if (p->Cdesc) cublasLtMatrixLayoutDestroy(p->Cdesc);
        memset(p, 0, sizeof(*p));
        return NULL;
    }
    p->valid = 1;
    rt->n_plans++;
    return p;
}

/* Runs one cuBLASLt matmul with the plan's cached algorithm. */
static cublasStatus_t laya_lt_run(laya_gemm_runtime *rt, laya_lt_plan *p,
                                const void *x, const void *w, void *y) {
    float alpha = 1.0f, beta = 0.0f;
    return cublasLtMatmul(rt->lt, p->opdesc, &alpha,
                          x, p->Adesc, w, p->Bdesc,
                          &beta, y, p->Cdesc, y, p->Cdesc,
                          &p->algo, rt->workspace, p->workspace_bytes, 0);
}

/* cublasGemmEx reference for the same contract (fallback + tuning oracle). */
static cublasStatus_t laya_gemmex_run(laya_gemm_runtime *rt,
                                    const void *x, const void *w, void *y,
                                    int M, int N, int K) {
    float alpha = 1.0f, beta = 0.0f;
    return cublasGemmEx(rt->cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                        &alpha, w, CUDA_R_16BF, K,
                                x, CUDA_R_16BF, K,
                        &beta,  y, CUDA_R_16BF, N,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

/*
 * Enumerates the supported cuBLASLt algorithms for this shape, validates each
 * against cublasGemmEx, benchmarks the valid ones with CUDA events and caches
 * the fastest. Runs once per unique shape (first call), never in steady state.
 */
static void laya_lt_tune(laya_gemm_runtime *rt, laya_lt_plan *p,
                       const void *x, const void *w, void *y) {
    struct timespec _ts0, _ts1;
    clock_gettime(CLOCK_MONOTONIC, &_ts0);
    cublasLtMatmulHeuristicResult_t results[16];
    int n = 0;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(
        rt->lt, p->opdesc, p->Adesc, p->Bdesc, p->Cdesc, p->Cdesc,
        rt->pref, 16, results, &n);
    if (st != CUBLAS_STATUS_SUCCESS || n <= 0) {
        snprintf(laya_cuda_errbuf(), 512, "cublasLt heuristic failed: %d", (int)st);
        p->valid = 0;
        return;
    }

    /* Oracle: cublasGemmEx result for this exact input. */
    size_t nbytes = (size_t)p->M * p->N * 2;
    uint16_t *ref = (uint16_t *)malloc(nbytes);
    uint16_t *got = (uint16_t *)malloc(nbytes);
    if (!ref || !got) { free(ref); free(got); p->valid = 0; return; }
    if (laya_gemmex_run(rt, x, w, y, p->M, p->N, p->K) != CUBLAS_STATUS_SUCCESS) {
        free(ref); free(got); p->valid = 0; return;
    }
    cudaMemcpy(ref, y, nbytes, cudaMemcpyDeviceToHost);

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    int best = -1;
    float best_ms = 0.0f;
    for (int i = 0; i < n; i++) {
        if (results[i].workspaceSize > rt->workspace_bytes) continue;
        p->algo = results[i].algo;
        p->workspace_bytes = results[i].workspaceSize;
        if (laya_lt_run(rt, p, x, w, y) != CUBLAS_STATUS_SUCCESS) continue;
        cudaMemcpy(got, y, nbytes, cudaMemcpyDeviceToHost);
        /* Reject any candidate that does not reproduce the cublasGemmEx
         * result bit-for-bit (same BF16 rounding, same accumulation order
         * class). This keeps the numerical contract identical. */
        int ok = 1;
        for (size_t j = 0; j < nbytes / 2; j++) {
            if (got[j] != ref[j]) { ok = 0; break; }
        }
        if (!ok) continue;

        float ms = 0.0f;
        for (int it = 0; it < 3; it++) {
            cudaEventRecord(e0, 0);
            laya_lt_run(rt, p, x, w, y);
            cudaEventRecord(e1, 0);
            cudaEventSynchronize(e1);
            float t = 0.0f;
            cudaEventElapsedTime(&t, e0, e1);
            if (it == 0 || t < ms) ms = t;
        }
        if (best < 0 || ms < best_ms) { best = i; best_ms = ms; }
    }

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    free(ref);
    free(got);

    if (best < 0) {
        snprintf(laya_cuda_errbuf(), 512, "cublasLt: no valid algorithm for %dx%dx%d",
                 p->M, p->N, p->K);
        p->valid = 0;
        return;
    }
    p->algo = results[best].algo;
    p->workspace_bytes = results[best].workspaceSize;
    p->tuned = 1;
    clock_gettime(CLOCK_MONOTONIC, &_ts1);
    p->tune_ms = (_ts1.tv_sec - _ts0.tv_sec) * 1000.0f +
                 (_ts1.tv_nsec - _ts0.tv_nsec) / 1e6f;
    if (rt->debug)
        fprintf(stderr, "[gemm] Lt tune %dx%dx%d: %.1f ms\n",
                p->M, p->N, p->K, p->tune_ms);
    if (rt->debug) {
        fprintf(stderr, "[gemm] Lt %dx%dx%d: %d candidates, best=%d ws=%zu %.3f ms\n",
                p->M, p->N, p->K, n, best, p->workspace_bytes, best_ms);
    }
}

static void laya_gemm_cublaslt(const void *x_dev, const void *w_dev,
                             const void *bias_dev, void *y_dev,
                             int M, int N, int K) {
    laya_lt_plan *p = laya_lt_get_plan(g_rt, M, N, K);
    if (!p) {
        /* Plan cache exhausted or descriptor creation failed: fall back. */
        if (laya_gemmex_run(g_rt, x_dev, w_dev, y_dev, M, N, K) != CUBLAS_STATUS_SUCCESS)
            snprintf(laya_cuda_errbuf(), 512, "cublasGemmEx fallback failed");
        goto bias;
    }
    if (!p->tuned) {
        if (g_rt->tune) {
            laya_lt_tune(g_rt, p, x_dev, w_dev, y_dev);
        } else {
            cublasLtMatmulHeuristicResult_t r;
            int got = 0;
            if (cublasLtMatmulAlgoGetHeuristic(g_rt->lt, p->opdesc, p->Adesc,
                                               p->Bdesc, p->Cdesc, p->Cdesc,
                                               g_rt->pref, 1, &r, &got) ==
                    CUBLAS_STATUS_SUCCESS && got > 0) {
                p->algo = r.algo;
                p->workspace_bytes = r.workspaceSize;
                p->tuned = 1;
            }
        }
    }
    if (!p->valid) {
        if (laya_gemmex_run(g_rt, x_dev, w_dev, y_dev, M, N, K) != CUBLAS_STATUS_SUCCESS)
            snprintf(laya_cuda_errbuf(), 512, "cublasGemmEx fallback failed");
        goto bias;
    }
    if (!p->tuned) {
        /* No algorithm was selected for this shape (heuristic miss or tuner
         * rejected every candidate): keep the cublasGemmEx fallback rather
         * than launching an unselected algorithm. */
        if (laya_gemmex_run(g_rt, x_dev, w_dev, y_dev, M, N, K) != CUBLAS_STATUS_SUCCESS)
            snprintf(laya_cuda_errbuf(), 512, "cublasGemmEx fallback failed");
        goto bias;
    }
    {
        cublasStatus_t rs = laya_lt_run(g_rt, p, x_dev, w_dev, y_dev);
        if (rs != CUBLAS_STATUS_SUCCESS) {
            snprintf(laya_cuda_errbuf(), 512,
                     "cublasLtMatmul failed: status=%d M=%d N=%d K=%d ws=%zu tune=%d",
                     (int)rs, p->M, p->N, p->K, p->workspace_bytes, p->tuned);
            return;
        }
    }
bias:
    if (bias_dev) {
        laya_gemm_bias_add_kernel<<<(M * N + 255) / 256, 256>>>(
            (const uint16_t *)bias_dev, (uint16_t *)y_dev, M, N);
    }
}

/* ------------------------------------------------------------------ */
/* cuBLAS GEMM (BF16 in/out, FP32 accumulate)                          */
/* ------------------------------------------------------------------ */

static void laya_gemm_cublas(const void *x_dev, const void *w_dev,
                           const void *bias_dev, void *y_dev,
                           int M, int N, int K) {
    /* y[M,N] = x[M,K] . w[N,K]^T, all row-major.
     *
     * In cuBLAS column-major terms the same memory is:
     *   W [N,K] row-major  == W^T [K,N] col-major, ld=K
     *   X [M,K] row-major  == X^T [K,M] col-major, ld=K
     *   Y [M,N] row-major  == Y^T [N,M] col-major, ld=N
     * and Y[m,n] = sum_k X[m,k]*W[n,k]  =>  Y^T[n,m] = sum_k W^T[k,n]*X^T[k,m]
     *   = sum_k A[k,n]*B[k,m] with A=W^T [K,N], B=X^T [K,M]
     *   => C[N,M] = op(A)[N,K] . op(B)[K,M] with op(A)=T (A=W^T), op(B)=T (B=X^T). */
    const __nv_bfloat16 *A = (const __nv_bfloat16 *)w_dev; /* [N,K] ld=K */
    const __nv_bfloat16 *B = (const __nv_bfloat16 *)x_dev; /* [M,K] ld=K */
    __nv_bfloat16 *C = (__nv_bfloat16 *)y_dev;             /* [M,N] result */
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasStatus_t cs = cublasGemmEx(g_rt->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                     N, M, K,
                                     &alpha, A, CUDA_R_16BF, K,
                                             B, CUDA_R_16BF, K,
                                     &beta,  C, CUDA_R_16BF, N,
                                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (cs != CUBLAS_STATUS_SUCCESS) {
        snprintf(laya_cuda_errbuf(), 512, "cublasGemmEx failed: %d", (int)cs);
        return;
    }
    if (bias_dev) {
        /* Bias add: y[M,N] += bias[N] broadcast over rows. */
        laya_gemm_bias_add_kernel<<<(M * N + 255) / 256, 256>>>(
            (const uint16_t *)bias_dev, (uint16_t *)y_dev, M, N);
    }
}

/* ------------------------------------------------------------------ */
/* Production linear dispatch                                          */
/* ------------------------------------------------------------------ */

void laya_linear(const void *x_dev, const void *w_dev, const void *bias_dev,
               void *y_dev, int M, int N, int K, int transpose_w) {
    if (!x_dev || !w_dev || !y_dev || M <= 0 || N <= 0 || K <= 0) {
        snprintf(laya_cuda_errbuf(), 512, "linear: bad args");
        return;
    }
    if (!transpose_w) {
        snprintf(laya_cuda_errbuf(), 512, "linear: transpose_w=0 unsupported");
        return;
    }
    if (g_use_production && g_rt && g_rt->cublas) {
        if (g_prod_backend == 0 && g_rt->lt) {
            laya_gemm_cublaslt(x_dev, w_dev, bias_dev, y_dev, M, N, K);
        } else {
            laya_gemm_cublas(x_dev, w_dev, bias_dev, y_dev, M, N, K);
        }
        return;
    }
    laya_linear_reference(x_dev, w_dev, bias_dev, y_dev, M, N, K, transpose_w);
}

/* ------------------------------------------------------------------ */
/* Timestep sinusoidal embedding (class B)                             */
/* ------------------------------------------------------------------ */

__global__ void laya_timestep_embed_kernel(const float *__restrict__ t,
                                         float *__restrict__ y,
                                         int N, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * dim) return;
    int n = idx / dim;
    int d = idx % dim;
    int half = dim / 2;
    int p = d < half ? d : d - half;
    float freq = expf(-logf(10000.0f) * (float)p / (float)half);
    float arg = t[n] * freq;
    float v = (d < half) ? cosf(arg) : sinf(arg);
    if (dim % 2 && d == dim - 1) v = 0.0f;
    y[idx] = v;
}

void laya_timestep_embed(const float *t_dev, float *y_dev, int N, int dim) {
    if (!t_dev || !y_dev || N <= 0 || dim <= 0) {
        snprintf(laya_cuda_errbuf(), 512, "timestep_embed: bad args");
        return;
    }
    laya_timestep_embed_kernel<<<(N * dim + 255) / 256, 256>>>(t_dev, y_dev, N, dim);
}
