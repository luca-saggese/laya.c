# O1 reuse map — Laya native runtime

What Laya needs, where it already exists in `_reference/o1.c`, and what to do.
`_reference/` is a local, git-ignored checkout of https://github.com/luca-saggese/o1.c.

## Copied unchanged (or near-unchanged)

| Laya requirement | Reuse from `o1.c` | Action |
|---|---|---|
| JSON parsing (requests, config) | `src/io/json.c/.h` | copy unchanged (`hd_json_*`) |
| GGUF v3 reader | `src/io/gguf.c/.h` | copy; extend metadata keys for `laya.*` / `modernbert.*` |
| safetensors (converter only) | `src/io/safetensors.c/.h` | Python converter reads HF directly; keep C reader out of runtime |
| BF16 host/device helpers | `src/cuda/support.cu`, `src/cuda/cuda_internal.h` | copy unchanged |
| GEMM (cuBLAS/cuBLASLt, cached plans) | `src/cuda/gemm.cu/.h` | copy; keep `hd_linear` contract (`W` stored `[out,in]`) |
| Embedding row gather | `src/cuda/embed.cu` (`hd_gather_rows`) | copy; also reused for marker gather |
| Residual add | `src/cuda/residual.cu` | copy unchanged |
| RoPE | `src/cuda/rope.cu` | adapt: ModernBERT has one theta per layer type (160000 full / 10000 sliding) |
| cuDNN SDPA | `src/cuda/hd_cudnn_sdpa.cu`, `include/hd_cudnn_sdpa.h` | reuse as primary attention backend (masked single-graph plan) |
| Eager attention | `src/cuda/attn.cu` | fallback/diagnostic only |
| Timing | `src/runtime/o1_timing.c/.h` | copy unchanged |
| Error/status plumbing | `include/hidream.h` (`hd_status`, `hd_last_error`) | copy the enum; Laya gets its own header with the same shape |
| Resident arena load | `src/model/weights.c` `hd_weights_to_device_gguf` | copy the flow: 1 arena `cudaMalloc`, 4×128 MiB pinned stages, bulk H2D on a private stream, pointer bind |
| Makefile CUDA rules | `Makefile` (`-arch=sm_121`, object dirs, link lines) | copy patterns into `Makefile` `laya-spark` target |

## Adapted (copy then change)

| Laya requirement | Reuse from `o1.c` | Change |
|---|---|---|
| LayerNorm | `src/cuda/norm.cu` (`hd_rmsnorm`) | ModernBERT uses LayerNorm (mean+variance, eps=1e-5, no bias), not RMSNorm |
| Activation | `src/cuda/act.cu` (`hd_silu`, `hd_swiglu`) | ModernBERT MLP is GeGLU: `Wo(act(input) * gate)` with `act = gelu` |
| Tokenizer | `src/model/tokenizer.c/.h`, `tools/_gen_c_tables.py` | Laya tokenizer is GPT-2 byte-level BPE (like Qwen) but with mmBERT vocab + `[CLS]/[SEP]/[MASK]/[PAD]`; keep the pre-tokenizer/BPE machinery, regenerate tables |
| Sequence construction | none | direct C port of `laya/common.py:build_sequence` + `render_options` + `collate_items` |
| Model struct / weight binding | `src/model/model.c`, `src/model/weights.h` | Laya-specific `laya_model` with `src/laya/laya_weights.c` |

## Not copied

HiDream-only: `src/image/*`, `src/io/png*`, `src/model/scheduler.c`, `src/model/lora*`,
`src/model/vision.c`, `src/model/forward.c`, `src/model/block.c` (diffusion blocks),
`src/runtime/decode.c`, `src/runtime/generate.c`, `src/runtime/refiner.c`,
`src/runtime/preview.c`, `src/server/*`, `tools/hidream_convert.py` (pattern reused,
not the file), `src/cuda/sched.cu`, `src/cuda/vision_kernels.cu`.

## Oracle

Original Python Laya (`_reference/laya`, https://github.com/NandhaKishorM/laya) stays
the oracle: `laya/common.py` (sequence construction, decision head math) and
`laya/agent.py` (`system_one` end-to-end path). `tools/oracle_laya.py` wraps them;
Python is never required at inference runtime.
