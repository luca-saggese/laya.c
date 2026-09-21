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

typedef uint16_t laya_bf16;

/* IEEE-754 fp32 <-> BF16 (round-to-nearest-even on the way down). */
float laya_bf16_to_f32(uint16_t b);
uint16_t laya_f32_to_bf16(float f);

/* Raw little-endian payload host readers used by the fixture harness. */
void laya_bf16_buf_to_f32(const void *src, float *dst, size_t n);
void laya_f32_buf_to_bf16(const float *src, void *dst, size_t n);

/* ------------------------------------------------------------------ */
/* Error reporting                                                     */
/* ------------------------------------------------------------------ */

const char *laya_cuda_last_error(void);
void laya_cuda_clear_error(void);

/* ------------------------------------------------------------------ */
/* Device-layout primitives (reference kernels, current stream)        */
/* ------------------------------------------------------------------ */

/*
 * RMSNorm (class B): y[row, k] = w[k] * x * rsqrt(mean(x^2, dim=-1) + eps),
 * fp32 internal, bf16 in/out. x is [rows, cols], weight length cols, reduced
 * over the last (contiguous) dim.
 */
void laya_rmsnorm(const void *x_dev, const void *w_dev, void *y_dev,
                int rows, int cols, float eps);

/* SiLU (class B): y = x * sigmoid(x), bf16 elementwise. */
void laya_silu(const void *x_dev, void *y_dev, size_t n);

/* SwiGLU (class B): y = silu(gate) * up, bf16 elementwise. */
void laya_swiglu(const void *gate_dev, const void *up_dev, void *y_dev, size_t n);

/* Residual add (class B): y[i] = x[i] + a[i] elementwise, bf16. */
void laya_residual_add(const void *x_dev, const void *a_dev, void *y_dev, size_t n);

/*
 * Linear / GEMM (class C): y[M,N] = x[M,K] x W[K,N]^T with fp32 accumulation.
 * transpose_w == 1: W stored [N,K] (out,in) -> compute x^T . W^T.
 * transpose_w == 0: W stored [K,N] (in,out) -> direct mac.
 * All operands are device pointers (bf16).
 */
void laya_linear(const void *x_dev, const void *w_dev, const void *bias_dev,
               void *y_dev, int M, int N, int K, int transpose_w);

/*
 * Timestep sinusoidal embedding (class B): produce the [N, dim] fp32 table.
 * t is [N] fp32. y is fp32.
 */
void laya_timestep_embed(const float *t_dev, float *y_dev, int N, int dim);

/*
 * MRoPE cos/sin table (class B): given fp32 position_ids [3, bs, seq],
 * fp32 inv_freq [dim/2], mrope_section, produce cos/sin [1, seq, dim]
 * (fp32 outputs; oracle casts to bf16 only at use). Returns fp32.
 */
void laya_mrope_cos_sin(const float *pos_dev, int bs, int seq,
                      const int64_t *section_dev, int n_section,
                      int head_dim, float theta, float attention_scaling,
                      int interleaved, float *cos_dev, float *sin_dev);

/*
 * apply_rotary_pos_emb (class B). q/k shape [heads, seq, dim] device bf16.
 * cos/sin [1, seq, dim] fp32. y same layout, bf16.
 */
void laya_apply_rotary(const void *x_dev, const float *cos_dev,
                     const float *sin_dev, void *y_dev,
                     int heads, int seq, int dim);

/*
 * Vision-tower rotary (apply_rotary_pos_emb_vision parity): identical layout
 * to laya_apply_rotary, but computes the rotation in FP32 (upcast q/k and
 * cos/sin) and casts the result to bf16 once at the end, with NO per-op bf16
 * rounding. Used only by the Qwen3-VL vision tower.
 */
void laya_apply_rotary_f32(const void *x_dev, const float *cos_dev,
                         const float *sin_dev, void *y_dev,
                         int heads, int seq, int dim);

/*
 * Eager attention (class D): q[heads,seq,dim], k/v[kv_heads,seq,dim],
 * mask[1,1,seq,seq]. Produces scores[heads,seq,seq], probs[heads,seq,seq],
 * out[seq,heads,dim] (seq-major). All bf16, softmax fp32.
 */
void laya_attention_eager(const void *q_dev, const void *k_dev, const void *v_dev,
                        const void *mask_dev, void *scores_dev,
                        void *probs_dev, void *out_dev,
                        int heads, int kv_heads, int seq, int dim,
                        float scaling);

/*
 * head_split / head_merge (class A, bit-exact). in [1,seq,H,D] bf16 ->
 * out [H,seq,D] bf16, and inverse. Byte-level permutation only.
 */
void laya_head_split(const void *in_dev, void *out_dev, int seq, int heads, int dim);
void laya_head_merge(const void *in_dev, void *out_dev, int seq, int heads, int dim);

/*
 * Row-gather embedding lookup (class A, byte-copy). table [nrows, cols]
 * bf16, idx [M] int64 -> out [M, cols] bf16. Out-of-range idx fall back to
 * padding row 0. Caller provides device buffers; no allocation.
 */
void laya_gather_rows(const void *table_dev, const void *idx_dev, void *out_dev,
                    int M, int cols, int64_t nrows);

/*
 * Timestep conditioning (class A copy): out[row] = t_emb when idx[row]==tms_id
 * else emb[row] (broadcast of the single [H] t_emb across matching rows).
 * Used to reproduce `torch.where(tms_mask, t_emb_expanded, inputs_embeds)`.
 */
void laya_apply_tms_condition(const void *idx_dev, const void *emb_dev,
                            const void *t_emb_dev, void *out_dev,
                            int M, int H, int64_t tms_id);

/* y[i] = x[i] * s elementwise fp32 (used to scale timestep by 1000). */
void laya_scale_f32(const float *in_dev, float *out_dev, float s, int n);
/* y = f32-to-bf16 device conversion (used for the freq table cast). */
void laya_f32_convert_bf16(const float *in_dev, void *out_dev, int n);

/* ------------------------------------------------------------------ */
/* M1.5 scheduler kernels (contract section 7, fp32 pointwise)         */
/* ------------------------------------------------------------------ */

/* out[i] = bf16_to_f32(in[i])  (upcast of the current sample z). */
void laya_sched_bf16_upcast(const void *in_dev, float *out_dev, int n);
/* denoised[i] = z[i] - model_output[i] * sigma  (fp32, torch order). */
void laya_sched_denoised(const float *z_dev, const float *mo_dev, float sigma,
                       float *denoised_dev, int n);
/* z_next[i] = (sigma_next*noise[i])*s_noise + (1.0f-sigma_next)*denoised[i]
 * fp32, torch left-to-right (sigma_next*noise)*s_noise. */
void laya_sched_z_next(const float *noise_dev, const float *denoised_dev,
                     float sigma_next, float s_noise, float *z_next_dev, int n);
/* model_output[i] = (z[i] - x_pred_masked[i]) / sigma  (fp32, folded
 * v_cond = (xp - z)/sigma; model_output = -v_guided). */
void laya_sched_vcond(const void *z_dev, const void *xp_dev, float sigma,
                    float *mo_dev, int n);

/* FlowMatch Euler: z_next = z + (sigma_next - sigma) * model_output. */
void laya_sched_flow_match_step(const void *z_dev, const float *mo_dev,
                              float sigma, float sigma_next,
                              void *z_next_dev, int n);

/* ------------------------------------------------------------------ */
/* CFG guidance combine (base "default" path)                          */
/* ------------------------------------------------------------------ */
/* model_output = -v_guided = (1-g)*mo_uncond + g*mo_cond  (fp32). */
void laya_sched_cfg_guided(const float *mo_cond, const float *mo_uncond, float g,
                         float *mo_guided, int n);

/* ------------------------------------------------------------------ */
/* FlowUniPC multistep kernels (base "default" path)                   */
/* ------------------------------------------------------------------ */
/* conv[j] = sample[j] - sigma_cur * model_output[j]  (fp32 x0 pred). */
void laya_unipc_convert(const float *sample_dev, const float *mo_dev,
                      float sigma_cur, float *conv_dev, int n);
/* UniC corrector step (order_c = 1 or 2); out = corrected sample. */
void laya_unipc_correct(const float *last_dev, const float *mo0_dev,
                      const float *mo_old_dev, const float *conv_dev,
                      float sig_t, float sig_s0, float alpha_t, float h_phi_1,
                      float B_h, float rhos_c0, float rhos_c1, float inv_rks0,
                      int order_c, float *out_dev, int n);
/* UniP predictor step (order_p = 1 or 2); out = prev_sample. */
void laya_unipc_predict(const float *sample_dev, const float *mo0_dev,
                      const float *mo_old_dev, float sig_t, float sig_s0,
                      float alpha_t, float h_phi_1, float B_h, float rhos_p,
                      float inv_rks0, int order_p, float *out_dev, int n);

/* ------------------------------------------------------------------ */
/* Laya / ModernBERT primitives (src/cuda/laya_ops.cu)                 */
/* ------------------------------------------------------------------ */

/* LayerNorm over the last dim, bf16 in/out, fp32 internal. b may be NULL. */
void laya_layernorm(const void *x, const void *w, const void *b,
                    void *y, int rows, int cols, float eps);

/* Exact GELU: 0.5*x*(1+erf(x/sqrt(2))), bf16 elementwise. */
void laya_gelu(const void *x, void *y, size_t n);

/* GeGLU: y = gelu(gate) * up, bf16 elementwise (ModernBERT MLP). */
void laya_geglu(const void *gate, const void *up, void *y, size_t n);

/* RoPE cos/sin table [seq, dim] fp32 for a single-section (text) rope. */
void laya_rope_cos_sin(float *cos_dev, float *sin_dev, int seq, int dim, float theta);

/* Additive attention mask [1,1,seq,seq] bf16: 0.0 keep, bf16 min masked.
 * sliding != 0 applies the bidirectional window |c - r| <= window.
 * window must be config.sliding_window (64 for Laya), NOT the attention
 * module's +1 value: verified against transformers 5.12
 * create_bidirectional_sliding_window_mask. Full layers pass no mask at all
 * (create_bidirectional_mask returns None for a single unpadded item). */
void laya_attn_mask(void *mask_dev, int seq, int window, int sliding);

/* out[i] = h[idx[i]] for int32 indices; h [S, cols] bf16 -> out [M, cols]. */
void laya_gather_pos(const void *h_dev, const void *idx_dev, void *out_dev,
                     int M, int cols);

/* Row-wise softmax over K logits (fp32). */
void laya_softmax_f32(const float *logits_dev, float *probs_dev, int rows, int K);

/* feats[r] = [top1, top1-top2, entropy/log(k), k/255] from softmax probs. */
void laya_top2_entropy(const float *probs_dev, float *feats_dev, int rows, int K);

#ifdef __cplusplus
}
#endif

#endif /* LAYA_CUDA_PROJECT_H */