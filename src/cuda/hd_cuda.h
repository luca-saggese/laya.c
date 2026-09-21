#ifndef LAYA_CUDA_PROJECT_H
#define HD_CUDA_PROJECT_H

/*
 * Public C ABI for the M1.2 reference CUDA transformer primitives.
 *
 * This is a *reference implementation* engine for validating the native
 * pipeline against frozen golden fixtures. Grade is BF16 compute with FP32
 * accumulation for GEMMs / softmax (see docs/M1_NUMERICAL_CONTRACT.md).
 * Kernels are hand-written CUDA; cuBLAS is NOT used for any primitive.
 *
 * Every primitive is a small kernel-launch wrapper. Pointers must be device
 * pointers in the current CUDA context (callers stage host data). Reference
 * numeric values are exposed as host helpers too, so the C harness can
 * cross-check against cuBLAS without invoking any model forward.
 */

#include <stddef.h>
#include <stdint.h>

#include "laya.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* BF16 host helpers                                                   */
/* ------------------------------------------------------------------ */

typedef uint16_t hd_bf16;

/* IEEE-754 fp32 <-> BF16 (round-to-nearest-even on the way down). */
float hd_bf16_to_f32(uint16_t b);
uint16_t hd_f32_to_bf16(float f);

/* Raw little-endian payload host readers used by the fixture harness. */
void hd_bf16_buf_to_f32(const void *src, float *dst, size_t n);
void hd_f32_buf_to_bf16(const float *src, void *dst, size_t n);

/* ------------------------------------------------------------------ */
/* Error reporting                                                     */
/* ------------------------------------------------------------------ */

const char *laya_cuda_last_error(void);
void hd_cuda_clear_error(void);

/* ------------------------------------------------------------------ */
/* Device-layout primitives (reference kernels, current stream)        */
/* ------------------------------------------------------------------ */

/*
 * RMSNorm (class B): y[row, k] = w[k] * x * rsqrt(mean(x^2, dim=-1) + eps),
 * fp32 internal, bf16 in/out. x is [rows, cols], weight length cols, reduced
 * over the last (contiguous) dim.
 */
void hd_rmsnorm(const void *x_dev, const void *w_dev, void *y_dev,
                int rows, int cols, float eps);

/* SiLU (class B): y = x * sigmoid(x), bf16 elementwise. */
void hd_silu(const void *x_dev, void *y_dev, size_t n);

/* SwiGLU (class B): y = silu(gate) * up, bf16 elementwise. */
void hd_swiglu(const void *gate_dev, const void *up_dev, void *y_dev, size_t n);

/* Residual add (class B): y[i] = x[i] + a[i] elementwise, bf16. */
void hd_residual_add(const void *x_dev, const void *a_dev, void *y_dev, size_t n);

/*
 * Linear / GEMM (class C): y[M,N] = x[M,K] x W[K,N]^T with fp32 accumulation.
 * transpose_w == 1: W stored [N,K] (out,in) -> compute x^T . W^T.
 * transpose_w == 0: W stored [K,N] (in,out) -> direct mac.
 * All operands are device pointers (bf16).
 */
void hd_linear(const void *x_dev, const void *w_dev, const void *bias_dev,
               void *y_dev, int M, int N, int K, int transpose_w);

/*
 * Timestep sinusoidal embedding (class B): produce the [N, dim] fp32 table.
 * t is [N] fp32. y is fp32.
 */
void hd_timestep_embed(const float *t_dev, float *y_dev, int N, int dim);

/*
 * MRoPE cos/sin table (class B): given fp32 position_ids [3, bs, seq],
 * fp32 inv_freq [dim/2], mrope_section, produce cos/sin [1, seq, dim]
 * (fp32 outputs; oracle casts to bf16 only at use). Returns fp32.
 */
void hd_mrope_cos_sin(const float *pos_dev, int bs, int seq,
                      const int64_t *section_dev, int n_section,
                      int head_dim, float theta, float attention_scaling,
                      int interleaved, float *cos_dev, float *sin_dev);

/*
 * apply_rotary_pos_emb (class B). q/k shape [heads, seq, dim] device bf16.
 * cos/sin [1, seq, dim] fp32. y same layout, bf16.
 */
void hd_apply_rotary(const void *x_dev, const float *cos_dev,
                     const float *sin_dev, void *y_dev,
                     int heads, int seq, int dim);

/*
 * Vision-tower rotary (apply_rotary_pos_emb_vision parity): identical layout
 * to hd_apply_rotary, but computes the rotation in FP32 (upcast q/k and
 * cos/sin) and casts the result to bf16 once at the end, with NO per-op bf16
 * rounding. Used only by the Qwen3-VL vision tower.
 */
void hd_apply_rotary_f32(const void *x_dev, const float *cos_dev,
                         const float *sin_dev, void *y_dev,
                         int heads, int seq, int dim);

/*
 * Eager attention (class D): q[heads,seq,dim], k/v[kv_heads,seq,dim],
 * mask[1,1,seq,seq]. Produces scores[heads,seq,seq], probs[heads,seq,seq],
 * out[seq,heads,dim] (seq-major). All bf16, softmax fp32.
 */
void hd_attention_eager(const void *q_dev, const void *k_dev, const void *v_dev,
                        const void *mask_dev, void *scores_dev,
                        void *probs_dev, void *out_dev,
                        int heads, int kv_heads, int seq, int dim,
                        float scaling);

/*
 * head_split / head_merge (class A, bit-exact). in [1,seq,H,D] bf16 ->
 * out [H,seq,D] bf16, and inverse. Byte-level permutation only.
 */
void hd_head_split(const void *in_dev, void *out_dev, int seq, int heads, int dim);
void hd_head_merge(const void *in_dev, void *out_dev, int seq, int heads, int dim);

/*
 * Row-gather embedding lookup (class A, byte-copy). table [nrows, cols]
 * bf16, idx [M] int64 -> out [M, cols] bf16. Out-of-range idx fall back to
 * padding row 0. Caller provides device buffers; no allocation.
 */
void hd_gather_rows(const void *table_dev, const void *idx_dev, void *out_dev,
                    int M, int cols, int64_t nrows);

/*
 * Timestep conditioning (class A copy): out[row] = t_emb when idx[row]==tms_id
 * else emb[row] (broadcast of the single [H] t_emb across matching rows).
 * Used to reproduce `torch.where(tms_mask, t_emb_expanded, inputs_embeds)`.
 */
void hd_apply_tms_condition(const void *idx_dev, const void *emb_dev,
                            const void *t_emb_dev, void *out_dev,
                            int M, int H, int64_t tms_id);

/* y[i] = x[i] * s elementwise fp32 (used to scale timestep by 1000). */
void hd_scale_f32(const float *in_dev, float *out_dev, float s, int n);
/* y = f32-to-bf16 device conversion (used for the freq table cast). */
void hd_f32_convert_bf16(const float *in_dev, void *out_dev, int n);

/* ------------------------------------------------------------------ */
/* M1.5 scheduler kernels (contract section 7, fp32 pointwise)         */
/* ------------------------------------------------------------------ */

/* out[i] = bf16_to_f32(in[i])  (upcast of the current sample z). */
void hd_sched_bf16_upcast(const void *in_dev, float *out_dev, int n);
/* denoised[i] = z[i] - model_output[i] * sigma  (fp32, torch order). */
void hd_sched_denoised(const float *z_dev, const float *mo_dev, float sigma,
                       float *denoised_dev, int n);
/* z_next[i] = (sigma_next*noise[i])*s_noise + (1.0f-sigma_next)*denoised[i]
 * fp32, torch left-to-right (sigma_next*noise)*s_noise. */
void hd_sched_z_next(const float *noise_dev, const float *denoised_dev,
                     float sigma_next, float s_noise, float *z_next_dev, int n);
/* model_output[i] = (z[i] - x_pred_masked[i]) / sigma  (fp32, folded
 * v_cond = (xp - z)/sigma; model_output = -v_guided). */
void hd_sched_vcond(const void *z_dev, const void *xp_dev, float sigma,
                    float *mo_dev, int n);

/* FlowMatch Euler: z_next = z + (sigma_next - sigma) * model_output. */
void hd_sched_flow_match_step(const void *z_dev, const float *mo_dev,
                              float sigma, float sigma_next,
                              void *z_next_dev, int n);

/* ------------------------------------------------------------------ */
/* CFG guidance combine (base "default" path)                          */
/* ------------------------------------------------------------------ */
/* model_output = -v_guided = (1-g)*mo_uncond + g*mo_cond  (fp32). */
void hd_sched_cfg_guided(const float *mo_cond, const float *mo_uncond, float g,
                         float *mo_guided, int n);

/* ------------------------------------------------------------------ */
/* FlowUniPC multistep kernels (base "default" path)                   */
/* ------------------------------------------------------------------ */
/* conv[j] = sample[j] - sigma_cur * model_output[j]  (fp32 x0 pred). */
void hd_unipc_convert(const float *sample_dev, const float *mo_dev,
                      float sigma_cur, float *conv_dev, int n);
/* UniC corrector step (order_c = 1 or 2); out = corrected sample. */
void hd_unipc_correct(const float *last_dev, const float *mo0_dev,
                      const float *mo_old_dev, const float *conv_dev,
                      float sig_t, float sig_s0, float alpha_t, float h_phi_1,
                      float B_h, float rhos_c0, float rhos_c1, float inv_rks0,
                      int order_c, float *out_dev, int n);
/* UniP predictor step (order_p = 1 or 2); out = prev_sample. */
void hd_unipc_predict(const float *sample_dev, const float *mo0_dev,
                      const float *mo_old_dev, float sig_t, float sig_s0,
                      float alpha_t, float h_phi_1, float B_h, float rhos_p,
                      float inv_rks0, int order_p, float *out_dev, int n);

#ifdef __cplusplus
}
#endif

#endif /* LAYA_CUDA_PROJECT_H */