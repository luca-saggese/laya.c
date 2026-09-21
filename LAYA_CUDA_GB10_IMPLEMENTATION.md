# Laya C/CUDA Inference Engine on NVIDIA GB10 / DGX Spark
## Implementation specification: adapt `o1.c`, do not rewrite it

> **Primary objective**
>
> Build a native C/CUDA inference engine for [Laya](https://github.com/NandhaKishorM/laya) targeting NVIDIA DGX Spark / GB10, by **reusing and adapting the existing `o1.c` codebase as aggressively as possible**.
>
> The implementation must prioritize shipping working code quickly. It must avoid speculative redesigns, unnecessary abstractions, premature optimization, excessive unit-test proliferation, and rewrites of code that already exists in `o1.c`.

---

## 0. Non-negotiable rules

These rules override stylistic preferences during implementation.

1. **Copy first, adapt second. Do not rewrite working infrastructure.**
   - Before implementing a subsystem, search `_reference/o1.c` for the closest existing implementation.
   - If a reusable implementation exists, copy it into the Laya runtime and make the smallest necessary changes.
   - Preserve data structures, naming, error handling, CUDA wrapper style, build structure, and memory-management patterns whenever possible.
   - New code is justified only when Laya requires an operation or architecture that `o1.c` genuinely does not implement.

2. **The reference repository must be local and ignored by Git.**
   - Clone `o1.c` into the project-local directory:
     ```bash
     mkdir -p _reference
     git clone https://github.com/luca-saggese/o1.c _reference/o1.c
     ```
   - Add the entire reference directory to `.gitignore`:
     ```gitignore
     /_reference/
     ```
   - Do not commit `_reference/o1.c`.

3. **Use the original Laya Python implementation as the oracle.**
   - The C runtime is correct only when it matches the original Python/PyTorch implementation closely enough to preserve decisions and probabilities.
   - Do not invent a new interpretation of the architecture.
   - When behavior is ambiguous, inspect the original Python implementation and Hugging Face model configuration.
   - Python is allowed for:
     - model conversion;
     - fixture generation;
     - oracle comparison;
     - diagnostics.
   - Python must **not** be required at inference runtime.

4. **Create a GGUF immediately.**
   - Do not make native runtime development depend on loading raw Hugging Face `safetensors` every startup.
   - One of the first milestones must convert the original Laya checkpoint to a native GGUF artifact.
   - The runtime must load that GGUF directly and upload model tensors to VRAM with minimal startup work.
   - Any tensor transposition, renaming, repacking, dtype normalization, or layout conversion that can be performed offline must happen during GGUF generation, not during every runtime startup.

5. **Testing must be sparse but strategically placed.**
   - Do not add a test for every helper, kernel, or minor function.
   - Do not run the full regression suite after every edit.
   - Use:
     - cheap build/smoke checks during implementation;
     - focused oracle comparison at meaningful architecture boundaries;
     - full end-to-end regression only at milestone boundaries where it provides real value.
   - When a parity failure occurs, temporarily add a narrow diagnostic probe, fix the problem, then remove or simplify that probe if it no longer protects an important contract.

6. **Commit immediately after every completed milestone step.**
   - Do not accumulate several unrelated steps into one commit.
   - Do not squash the implementation history.
   - Every commit must:
     - compile;
     - contain one coherent step;
     - have a descriptive imperative commit message;
     - leave the tree in a usable state.

7. **Do not refactor `o1.c` merely to make it prettier.**
   - The goal is a fast Laya engine, not a framework rewrite.
   - Avoid generic graph engines, tensor DSLs, operator registries, plugin systems, generalized model loaders, or other infrastructure unless absolutely required by Laya.

8. **Optimize only after end-to-end parity exists.**
   - First target: correct BF16 inference on GB10.
   - Then optimize allocations, kernel launches, attention, GEMM selection, batching, and CUDA Graphs.
   - Do not introduce quantization before the BF16 oracle path is validated.

---

# 1. Target scope

The finished project must support native inference for the Laya family:

- `convaiinnovations/laya`
- `convaiinnovations/laya-multilingual`
- `convaiinnovations/laya-typed-decisions`

The first complete path must be the English `convaiinnovations/laya` checkpoint. The other checkpoints are added only after the core architecture has passed end-to-end oracle comparison.

The runtime must implement the behavior used by Laya inference:

- state serialization;
- typed question formatting;
- tokenizer;
- `[CLS]`, `[SEP]`, `[MASK]` sequence construction;
- marker positions;
- attention mask;
- ModernBERT encoder;
- Laya type embedding;
- Laya decision-head Transformer layers;
- marker gather;
- scorer MLP;
- action head;
- calibrated output probabilities;
- `choice`;
- `score`;
- `noul`;
- optional router support after all checkpoints work.

No autoregressive generation is required.

No KV cache is required.

No sampling loop is required.

No causal decoding path is required.

---

# 2. Base architecture decision

Use `o1.c` as the implementation base.

`q38.c` is not the main codebase. It may be consulted only for GB10/Blackwell-specific build settings or proven CUDA techniques that do not already exist in `o1.c`.

The rationale is practical:

- Laya is a single-forward encoder model.
- `o1.c` already contains a modular native CUDA transformer implementation.
- `o1.c` already contains:
  - CUDA infrastructure;
  - BF16 helpers;
  - cuBLAS/cuBLASLt GEMM paths;
  - cuDNN SDPA;
  - embedding lookup;
  - residual operations;
  - normalization kernels;
  - activation kernels;
  - RoPE support;
  - JSON handling;
  - GGUF handling;
  - safetensors handling;
  - runtime timing;
  - model-loading patterns;
  - numerical fixture methodology;
  - Makefile patterns for native CUDA builds.

Do not port Laya onto the `q38.c` decode engine.

---

# 3. Repository bootstrap

## Step 0.1 — clone the reference

Execute:

```bash
mkdir -p _reference
git clone https://github.com/luca-saggese/o1.c _reference/o1.c
```

If `_reference/o1.c` already exists:

```bash
git -C _reference/o1.c fetch origin
git -C _reference/o1.c checkout main
git -C _reference/o1.c pull --ff-only
```

Immediately add:

```gitignore
/_reference/
```

### Commit

```text
chore: add ignored o1.c reference workspace
```

Do not continue until the commit exists.

---

## Step 0.2 — inventory reusable code

Create a short `docs/O1_REUSE_MAP.md`.

It must map Laya needs to existing `o1.c` files. Keep it concrete and short.

Expected starting map:

| Laya requirement | Reuse from `o1.c` | Action |
|---|---|---|
| JSON | `src/io/json.c/.h` | copy unchanged |
| GGUF | `src/io/gguf.c/.h` | copy, minimal rename |
| safetensors conversion support | `src/io/safetensors.c/.h` | copy only if converter/runtime needs it |
| BF16 helpers | `src/cuda/support.cu` | copy |
| GEMM | `src/cuda/gemm.cu/.h` | copy |
| embedding gather | `src/cuda/embed.cu` | copy |
| residual | `src/cuda/residual.cu` | copy |
| RoPE | `src/cuda/rope.cu` | adapt parameters/layout only |
| attention | `src/cuda/hd_cudnn_sdpa.cu` | reuse as primary SDPA backend |
| eager attention | `src/cuda/attn.cu` | retain only as fallback/diagnostic |
| normalization | existing norm / vision LayerNorm code | adapt, do not rewrite from scratch |
| GELU | existing activation / vision kernels | adapt exact variant required |
| timing | `src/runtime/o1_timing.c/.h` | copy |
| CUDA init | existing runtime CUDA setup | copy |
| Makefile CUDA rules | `Makefile` | copy patterns |
| tokenizer approach | `src/model/tokenizer.*`, `tools/_gen_c_tables.py` | reuse strategy, adapt tokenizer data |

This document is not a design exercise. Maximum target length: roughly 1–2 pages.

### Commit

```text
docs: map laya runtime onto reusable o1.c components
```

---

# 4. Immediate GGUF path

The GGUF is an early deliverable, not a cleanup task.

## Step 1.1 — create a Laya GGUF converter

Add:

```text
tools/laya_to_gguf.py
```

The converter must accept either:

```bash
python tools/laya_to_gguf.py \
    --model convaiinnovations/laya \
    --output models/laya-bf16.gguf
```

or a local Hugging Face snapshot:

```bash
python tools/laya_to_gguf.py \
    --model /path/to/local/laya \
    --output models/laya-bf16.gguf
```

The converter must:

1. load `rl_agent_config.json`;
2. load the encoder config;
3. load `model.safetensors`;
4. enumerate all required tensors;
5. verify required tensor names and shapes;
6. convert tensors to the runtime's expected layout;
7. store the result as BF16/F32 where appropriate;
8. write all static model metadata required by the C runtime;
9. avoid any transform at C runtime that can be done once here.

Prefer using or copying the GGUF-writing strategy already present in `_reference/o1.c` or its tooling. Do not introduce a new model container.

### Recommended metadata

Keep metadata sufficient to avoid external config files:

```text
general.architecture                   = "laya"
general.name                           = checkpoint name

laya.variant                           = "english" | "multilingual" | "typed-decisions"
laya.max_len
laya.head_max_len
laya.head_layers
laya.temperature.*
laya.temperature_by_options.*

modernbert.hidden_size
modernbert.num_hidden_layers
modernbert.num_attention_heads
modernbert.intermediate_size
modernbert.max_position_embeddings
modernbert.layer_norm_eps
modernbert.global_attn_every_n_layers
modernbert.local_attention
modernbert.global_rope_theta
modernbert.local_rope_theta
...
```

Do not add metadata speculatively. Only persist values needed to reproduce the Python model.

### Tensor layout policy

Use the layout expected by copied `o1.c` GEMM code.

If PyTorch stores a linear weight as `[out, in]` and the native kernel would otherwise transpose it on every load or every call:

- transform it during GGUF conversion;
- record the final shape;
- keep runtime code simple.

The converter is allowed to change tensor names into stable native names.

Prefer predictable names such as:

```text
encoder.embeddings.tok_embeddings.weight
encoder.embeddings.norm.weight
encoder.layers.00.attn.qkv.weight
encoder.layers.00.attn.out.weight
encoder.layers.00.mlp.wi.weight
encoder.layers.00.mlp.wo.weight
...
decision.type_emb.weight
decision.head.00.*
decision.head.01.*
decision.scorer.*
decision.act_head.*
```

Only fuse tensors offline when it clearly reduces runtime work and does not make oracle debugging harder.

### Commit

```text
feat: add bf16 laya gguf converter
```

---

## Step 1.2 — produce the first GGUF

Generate the English model immediately:

```bash
mkdir -p models
python tools/laya_to_gguf.py \
    --model convaiinnovations/laya \
    --output models/laya-bf16.gguf
```

Add generated large model files to `.gitignore`:

```gitignore
/models/*.gguf
```

Print a short converter summary:

```text
checkpoint:
tensor count:
weight bytes:
gguf bytes:
dtype summary:
```

Do not build a large converter test suite.

Run one converter sanity check:

- reopen the GGUF;
- enumerate tensors;
- verify tensor count/names/shapes against the converter manifest.

### Commit

```text
feat: generate native laya gguf model artifact
```

The `.gguf` itself should remain ignored unless the repository intentionally uses Git LFS.

---

# 5. Native model loader and fast VRAM residency

## Step 2.1 — copy the `o1.c` GGUF/runtime loader

Copy the relevant GGUF parsing and model-loading code from `_reference/o1.c`.

Do not start from an empty file.

Preserve the existing loader flow as much as possible:

```text
open GGUF
-> parse metadata
-> create tensor directory
-> calculate resident bytes
-> allocate GPU storage
-> upload tensors
-> bind native tensor pointers
-> keep model resident
```

Add Laya-specific model structures only where needed.

Suggested top-level structure:

```c
typedef struct {
    /* copied CUDA runtime handles from o1.c */
    /* cuBLAS / cuBLASLt */
    /* cuDNN / SDPA plan cache */
    /* stream(s) */
} laya_cuda_runtime;

typedef struct {
    /* GGUF mapping */
    /* resident GPU allocation */
    /* tensor bindings */
    /* model config */
    /* reusable workspaces */
    laya_cuda_runtime cuda;
} laya_model;
```

Avoid one `cudaMalloc` per tensor if the existing `o1.c` loader already has or can trivially support a contiguous resident arena.

Preferred startup behavior:

```text
1 x large cudaMalloc for resident tensor arena
few large workspace allocations
bulk sequential H2D copies
zero weight allocations during inference
```

If `o1.c` already has a different proven resident-loading strategy, use it instead of inventing a new one.

### Smoke check only

At this step test only:

```bash
./laya --model models/laya-bf16.gguf --inspect
```

Expected:

- GGUF opens;
- metadata is valid;
- all required tensors bind;
- VRAM allocation succeeds;
- process exits cleanly.

No inference test yet.

### Commit

```text
feat: load laya gguf into resident cuda memory
```

---

## Step 2.2 — GB10 build target

Copy the CUDA build structure from `o1.c` and use the DGX Spark architecture setting proven in `q38.c`.

Add:

```bash
make laya-spark
```

DGX Spark / GB10 target:

```text
sm_121
```

Do not restructure the whole Makefile.

Add only the target and source lists needed by the new binary/library.

Expected build products:

```text
build/laya
```

Optional later:

```text
build/laya-server
```

### Smoke check

Only:

```bash
make clean
make laya-spark
./build/laya --help
```

### Commit

```text
build: add dgx spark sm121 target for laya
```

---

# 6. Python oracle

The oracle must be implemented early, but kept small.

Add:

```text
tools/oracle_laya.py
```

It must use the **original Laya Python code**, not a hand-reimplemented PyTorch model.

It must support two modes.

## Oracle mode A — end-to-end

Input:

```json
{
  "state": {"body": "..."},
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "...",
      "criteria": {
        "a": "...",
        "b": "..."
      }
    }
  }
}
```

Output must include:

```json
{
  "input_ids": [],
  "attention_mask": [],
  "marker_pos": [],
  "qtype": [],
  "raw_logits": [],
  "act_logits": [],
  "answers": {}
}
```

Keep the raw values before NumPy rounding.

## Oracle mode B — focused diagnostic

Only for debugging parity failures.

Allow optionally dumping one or more of:

```text
embedding output
selected encoder layer output
final encoder output
decision-head output
marker-gather output
```

Do **not** make every intermediate dump part of the normal regression run.

### Commit

```text
test: add original python laya oracle
```

---

# 7. Tokenizer and exact input construction

The tokenizer is part of model correctness. It must match before debugging CUDA.

However, do not build a general tokenizer framework.

## Step 3.1 — adapt the `o1.c` tokenizer strategy

Start from:

```text
_reference/o1.c/src/model/tokenizer.c
_reference/o1.c/src/model/tokenizer.h
_reference/o1.c/tools/_gen_c_tables.py
```

Reuse as much implementation and table-generation machinery as compatible with the Laya tokenizer.

The generator must read the original tokenizer files and emit the static tables required by the C tokenizer.

If the Laya tokenizer algorithm differs from the Qwen tokenizer used by `o1.c`:

- keep the surrounding implementation pattern;
- replace only the algorithm-specific pieces;
- do not write a generic Hugging Face tokenizer runtime.

### Minimal test

Use a compact fixture containing approximately:

- 10 normal English strings;
- punctuation;
- UTF-8;
- JSON-like state;
- the Laya special tokens;
- one long truncation case.

Compare only:

```text
input_ids
marker_pos
attention length
```

Do not create hundreds of tokenizer unit tests.

### Commit

```text
feat: adapt o1 tokenizer path for laya inputs
```

---

## Step 3.2 — port `build_sequence`

Port the logic from original Laya `common.py` directly.

Preserve:

```text
[CLS]
<type> question: <instructions>
[SEP]
[MASK] option0
[MASK] option1
...
[SEP]
state
[SEP]
```

Preserve exactly:

- option rendering;
- option ordering;
- `head_max_len`;
- max length;
- option truncation;
- state truncation;
- marker positions;
- `noul` option text;
- choice/score rendering.

This code should look recognizably like a C translation of the original Python, not a redesigned prompt builder.

### Focused oracle comparison

Run one small fixture set through Python and C.

Stop when token arrays and marker positions are identical.

### Commit

```text
feat: match laya sequence construction with python oracle
```

---

# 8. CUDA primitive gap closure

Before implementing full ModernBERT, identify only the operations missing from copied `o1.c`.

Expected gaps are small.

Likely required operations:

- LayerNorm with exact Laya/ModernBERT epsilon behavior;
- exact GELU variant;
- GeGLU;
- attention-mask construction;
- local/sliding-attention mask;
- gather selected sequence positions;
- small softmax/top-2/entropy reductions for decision metadata.

## Rule

For every missing primitive:

1. search `_reference/o1.c`;
2. search the vision path;
3. search an existing kernel that can be adapted;
4. only then write a new CUDA kernel.

Do not create a standalone test per kernel.

Create one temporary primitive-smoke binary only if necessary to debug several primitives at once.

### Commit

```text
feat: add minimal cuda primitives required by modernbert
```

---

# 9. ModernBERT implementation

Implement the encoder as a direct model-specific forward path.

Do not create a generic graph executor.

Suggested files:

```text
src/laya/modernbert.c
src/laya/modernbert.h
src/laya/laya_weights.c
src/laya/laya_weights.h
```

## Step 4.1 — weight binding

Bind every encoder tensor from GGUF to typed model fields.

Validate shapes at startup.

Use static per-layer structures:

```c
typedef struct {
    /* norm */
    /* qkv */
    /* attention output */
    /* mlp */
} laya_modernbert_layer;
```

Avoid repeated name lookup during inference.

### Check

Loader-only smoke test.

### Commit

```text
feat: bind modernbert tensors from laya gguf
```

---

## Step 4.2 — embeddings

Implement by copying/adapting `o1.c` embedding and normalization paths.

Required flow must match the original ModernBERT checkpoint.

Use the oracle on one fixed request and compare the embedding output only once.

This is a **focused milestone-boundary comparison**, not a permanent full regression.

### Commit

```text
feat: implement modernbert embedding stage
```

---

## Step 4.3 — one ModernBERT layer

Implement one layer using copied `o1.c` GEMM, residual, RoPE, normalization, and SDPA infrastructure.

Start with the architecture exactly as configured by the original checkpoint.

Do not optimize local attention yet.

For initial parity:

- use cuDNN SDPA;
- use an explicit mask for full/sliding attention if that is the simplest path;
- make numerical behavior correct before special-casing local attention.

The layer implementation should contain no dynamic allocation in the steady-state forward path.

### Oracle comparison

Compare one representative layer:

- one global-attention layer;
- one sliding-attention layer.

Do not compare all layers individually.

If both layer types match, move on.

### Commit

```text
feat: implement modernbert global and sliding attention block
```

---

## Step 4.4 — full ModernBERT encoder

Run all layers using the same reusable workspace.

Allocate all scratch buffers during model/request setup.

No `cudaMalloc` / `cudaFree` inside the per-layer forward loop.

### Oracle comparison

For a small fixed fixture set compare:

```text
final encoder hidden state
```

Do not dump every layer unless final parity fails.

If it fails:

- use oracle diagnostic mode;
- binary-search the first bad layer;
- fix;
- return to final-only comparison.

### Commit

```text
feat: run complete modernbert encoder on cuda
```

---

# 10. Laya decision head

The decision head must be ported directly from original `laya/common.py`.

Do not fold it into the encoder until parity is established.

## Step 5.1 — type embedding and head Transformer layers

Implement:

```text
h = encoder(...)
h += type_emb[qtype]
2 x TransformerEncoderLayer
```

Important: reproduce the original PyTorch `TransformerEncoderLayer` semantics, including:

- pre-norm setting;
- activation;
- padding mask;
- residual order;
- epsilon;
- dropout disabled in inference.

Do not assume these blocks are identical to ModernBERT blocks.

Reuse the same GEMM / SDPA / normalization infrastructure wherever possible.

### Commit

```text
feat: add laya typed transformer decision head
```

---

## Step 5.2 — marker gather and scorer

Port directly:

```text
gather marker hidden states
-> LayerNorm
-> Linear
-> GELU
-> Linear(1)
-> masked logits
```

Use a simple gather kernel adapted from existing index/embedding logic if possible.

### Commit

```text
feat: add marker scorer and typed logits
```

---

## Step 5.3 — action head

Port the exact Python path:

```text
p = softmax(detached_logits)
k = number_of_valid_options
entropy
top2 probability
top1-top2 margin
k / 255
pooled = h[:, 0]
concat(pooled, features)
Linear
GELU
Linear
```

The statistical feature calculation can run on GPU or CPU initially.

Prefer the simplest correct implementation.

Do not optimize this tiny amount of work prematurely.

### Commit

```text
feat: add laya action head and confidence features
```

---

# 11. End-to-end Laya API

## Step 6.1 — native API

Expose a compact C API.

Example:

```c
typedef struct laya_model laya_model;
typedef struct laya_request laya_request;
typedef struct laya_result laya_result;

laya_model *laya_load(const char *gguf_path);

int laya_predict(
    laya_model *model,
    const laya_request *request,
    laya_result *result
);

void laya_result_free(laya_result *result);
void laya_free(laya_model *model);
```

Internally the model remains resident between requests.

The CLI can accept a JSON request.

Example:

```bash
./build/laya \
  --model models/laya-bf16.gguf \
  --input request.json
```

### Commit

```text
feat: expose resident laya inference api and cli
```

---

## Step 6.2 — Python oracle end-to-end parity

Create a compact regression corpus.

Target approximately 12–24 requests, not hundreds.

It should cover:

- choice with 2 options;
- choice with several options;
- score;
- noul;
- multiple questions batched in one call;
- short state;
- long/truncated state;
- JSON state;
- Unicode state;
- varied marker counts.

Compare:

1. tokenization;
2. raw logits;
3. final probabilities;
4. selected labels/scores;
5. action probability.

Use practical tolerances for BF16.

Primary requirement:

- discrete decisions match;
- probability drift stays within an explicitly defined tolerance;
- no NaN/Inf.

Only if this fails should intermediate diagnostics be enabled.

### Commit

```text
test: validate native laya inference against python oracle
```

This is the first meaningful full regression.

---

# 12. Performance pass on GB10

Performance work begins only after end-to-end parity.

## Step 7.1 — remove runtime allocations

Profile one normal request.

There must be:

```text
0 cudaMalloc/cudaFree in steady-state inference
```

Move all recurring buffers into model/request workspaces.

### Commit

```text
perf: make laya forward allocation free
```

---

## Step 7.2 — cache GEMM and SDPA plans

Reuse the production backend already present in `o1.c`.

Cache by relevant shape/dtype.

Typical recurring shapes should pay setup cost once.

Do not write a custom matmul kernel while cuBLAS/cuBLASLt is performing adequately.

### Commit

```text
perf: cache laya gemm and sdpa execution plans
```

---

## Step 7.3 — optimize sliding attention only if profiling justifies it

Initial implementation may use explicit-mask SDPA.

Profile first.

If local/sliding attention is a material bottleneck:

- adapt an existing attention kernel;
- or introduce a dedicated local-attention path.

Do not replace working cuDNN SDPA based on theoretical concerns.

### Commit

Only if implemented:

```text
perf: specialize modernbert sliding attention
```

---

## Step 7.4 — optional CUDA Graph capture

Only after shapes and buffers are stable.

Useful candidates:

- fixed model;
- bucketed sequence lengths;
- repeated batch/question shapes.

Do not complicate correctness paths to support CUDA Graphs.

### Commit

Only if useful:

```text
perf: capture stable laya forward paths with cuda graphs
```

---

# 13. Additional checkpoints

After English is correct and reasonably fast, support the other two models using the same runtime.

## Step 8.1 — typed-decisions

Generate:

```text
models/laya-typed-decisions-bf16.gguf
```

Reuse the existing engine.

Only add configuration/tensor differences actually required by the checkpoint.

Run a small end-to-end oracle comparison.

### Commit

```text
feat: support laya typed-decisions checkpoint
```

---

## Step 8.2 — multilingual

Generate:

```text
models/laya-multilingual-bf16.gguf
```

Do not fork the engine.

Parameterize hidden size, layer count, attention heads, context length, tokenizer data, and other checkpoint differences.

Tokenizer changes are allowed only where required by the multilingual checkpoint.

Run a compact multilingual oracle set.

### Commit

```text
feat: support laya multilingual checkpoint
```

---

# 14. Router

Port the original Python routing logic only after all target checkpoints load and infer correctly.

The router is CPU-side logic and should remain simple.

Implement only behavior used by the original Laya package:

- explicit model selection;
- task selection;
- script/language routing;
- typed-decisions workflow routing if enabled;
- resident-model selection.

Do not make the router responsible for model execution details.

If enough GB10 memory is available, allow all desired checkpoints to remain resident.

### Commit

```text
feat: add native laya checkpoint router
```

---

# 15. Test policy

This section is mandatory.

The implementation must **not** become test-driven to the point that writing tests dominates writing the engine.

Use four levels only.

## Level A — build/smoke

Run frequently because it is cheap.

Examples:

```bash
make laya-spark
./build/laya --help
./build/laya --model models/laya-bf16.gguf --inspect
```

Run after low-risk implementation steps.

---

## Level B — focused oracle check

Run only when crossing an important numerical boundary.

Planned focused checks:

1. tokenizer + sequence construction;
2. embeddings;
3. one global-attention encoder block;
4. one sliding-attention encoder block;
5. full encoder output.

These checks are not required after every later commit once the boundary is stable.

---

## Level C — end-to-end oracle regression

Run after:

- complete English end-to-end path;
- significant encoder numerical change;
- attention backend replacement;
- quantization/layout change;
- multilingual support;
- typed-decisions support;
- release candidate.

Do not run it after documentation changes, CLI formatting changes, build cleanup, or unrelated tiny fixes.

---

## Level D — performance benchmark

Run only when working on performance.

Measure at minimum:

```text
startup / model residency time
single-question latency
multi-question latency
GPU memory use
```

Keep correctness and performance checks separate.

---

# 16. Minimal permanent test set

The repository should end with a small durable test suite.

Suggested permanent tests:

```text
tests/
├── test_tokenizer_fixture.py or equivalent comparison harness
├── test_oracle_e2e.py
└── fixtures/
    ├── tokenizer_cases.json
    └── oracle_cases.json
```

Native test binaries may be retained only when they protect a fragile numerical contract or have proven useful for regression.

Do not create one test source file per CUDA primitive.

---

# 17. Commit discipline

Commit **immediately after each step**, before starting the next step.

Required characteristics:

- one coherent change;
- buildable tree;
- descriptive message;
- no unrelated formatting;
- no massive “implement everything” commit;
- no deferred mega-commit at the end;
- no squashing.

Expected implementation history should look approximately like:

```text
chore: add ignored o1.c reference workspace
docs: map laya runtime onto reusable o1.c components
feat: add bf16 laya gguf converter
feat: generate native laya gguf model artifact
feat: load laya gguf into resident cuda memory
build: add dgx spark sm121 target for laya
test: add original python laya oracle
feat: adapt o1 tokenizer path for laya inputs
feat: match laya sequence construction with python oracle
feat: add minimal cuda primitives required by modernbert
feat: bind modernbert tensors from laya gguf
feat: implement modernbert embedding stage
feat: implement modernbert global and sliding attention block
feat: run complete modernbert encoder on cuda
feat: add laya typed transformer decision head
feat: add marker scorer and typed logits
feat: add laya action head and confidence features
feat: expose resident laya inference api and cli
test: validate native laya inference against python oracle
perf: make laya forward allocation free
perf: cache laya gemm and sdpa execution plans
feat: support laya typed-decisions checkpoint
feat: support laya multilingual checkpoint
feat: add native laya checkpoint router
```

If a step requires multiple commits because it is genuinely too large, split by functional outcome, not by file.

---

# 18. Coding rules for maximum reuse

Before adding any non-trivial new file or kernel, answer internally:

```text
1. Does o1.c already have this?
2. Does o1.c have 80% of this?
3. Can I copy it and alter shapes/config?
4. Can conversion-time work eliminate runtime code?
5. Is this abstraction actually needed by Laya?
```

If answer 1, 2, or 3 is yes, reuse.

Preferred pattern:

```text
copy existing file
-> compile
-> change only model-specific assumptions
-> compare with oracle
-> commit
```

Avoid:

```text
read existing file
-> decide to design a cleaner abstraction
-> write a new framework
-> spend days restoring features already present
```

---

# 19. Runtime-performance rules

The initial high-performance target is BF16.

Do not quantize early.

For GB10:

- keep weights resident;
- prefer BF16 Tensor Core GEMMs;
- use cuBLAS/cuBLASLt;
- use cuDNN SDPA initially;
- reuse workspaces;
- avoid per-request CPU/GPU allocation churn;
- batch questions exactly as the original Laya runtime does;
- avoid CPU/GPU synchronization between layers;
- synchronize only where required for final result transfer or diagnostics.

The model is small enough that model capacity is not the immediate problem. Latency and launch overhead matter more than fitting weights into memory.

---

# 20. GGUF loading performance requirements

The GGUF path must be designed for startup efficiency.

Desired runtime startup:

```text
parse a small metadata section
-> allocate final GPU weight arena
-> read/copy already-converted tensor payloads
-> bind pointers
-> create persistent workspaces/plans
-> ready
```

Avoid at startup:

- PyTorch;
- safetensors parsing if GGUF exists;
- tensor transpose;
- dtype conversion of the full model;
- weight fusion;
- repeated temporary allocations;
- Python subprocesses.

All static conversion belongs in:

```text
tools/laya_to_gguf.py
```

---

# 21. Numerical correctness policy

The Python implementation is authoritative.

When there is a mismatch:

1. confirm input IDs and marker positions;
2. compare final encoder output;
3. if bad, locate the first bad layer;
4. compare only that layer;
5. identify the primitive causing drift;
6. fix it;
7. remove excessive diagnostic dumping;
8. rerun end-to-end oracle regression.

Do not react to a mismatch by creating broad speculative test coverage.

Do not change model semantics merely to improve numerical agreement on one fixture.

---

# 22. Error tolerance

Define explicit tolerances once BF16 behavior is observed.

Recommended acceptance hierarchy:

### Hard requirements

- same token IDs;
- same marker positions;
- same option ordering;
- same chosen `choice`;
- same `noul` side of 0.5 unless the oracle itself is essentially tied;
- same `score` interpretation;
- no NaN/Inf;
- deterministic result for fixed input in deterministic mode.

### Numeric requirements

Use measured BF16 drift to set limits for:

- logits;
- probabilities;
- action probability.

Do not demand bitwise equivalence when cuDNN/cuBLAS operation ordering differs from PyTorch.

Do not accept a large drift simply because the final class happens to match.

---

# 23. Files that should remain close to `o1.c`

Where feasible, keep copied infrastructure recognizable.

Examples:

```text
src/io/json.*
src/io/gguf.*
src/cuda/support.cu
src/cuda/gemm.*
src/cuda/embed.cu
src/cuda/residual.cu
src/cuda/rope.cu
src/cuda/*sdpa*
src/runtime/*timing*
```

Model-specific code should live under:

```text
src/laya/
```

Do not rename every `hd_*` symbol merely for aesthetic consistency if retaining it reduces churn.

Rename only when symbol collisions or semantic ambiguity justify it.

---

# 24. Files that should not be copied wholesale

Do not blindly import HiDream-only functionality such as:

- image preprocessing;
- image decoder;
- scheduler;
- diffusion loop;
- LoRA image path;
- preview generation;
- PNG/JPEG support;
- timestep conditioning;
- diffusion-specific blocks.

Only copy reusable pieces.

---

# 25. Definition of done — English checkpoint

English is complete when:

- `make laya-spark` succeeds on DGX Spark;
- `models/laya-bf16.gguf` is generated by the converter;
- runtime opens the GGUF without Python;
- weights become resident in VRAM;
- tokenizer matches Python fixtures;
- sequence construction matches Python;
- ModernBERT full forward executes;
- Laya decision head executes;
- `choice`, `score`, and `noul` execute;
- compact end-to-end oracle suite passes;
- steady-state forward performs no CUDA allocations;
- runtime can execute repeated requests without reloading the model.

Only then move to the other checkpoints.

---

# 26. Definition of done — complete Laya runtime

Project is complete when:

- English checkpoint works;
- multilingual checkpoint works;
- typed-decisions checkpoint works;
- router works;
- all supported models can be converted to GGUF;
- C/CUDA runtime has no Python dependency;
- original Python Laya remains the oracle;
- a compact permanent regression suite exists;
- DGX Spark / GB10 is the primary optimized CUDA target;
- the implementation visibly reuses `o1.c` rather than recreating its infrastructure.

---

# 27. Agent execution rule

When implementing this plan, do not stop to produce long speculative analyses.

For each step:

```text
inspect nearest o1.c implementation
-> copy/adapt it
-> build
-> run the minimum check required for that step
-> commit immediately
-> continue
```

If something fails:

```text
localize the failure
-> use the Python oracle if numerical
-> fix the narrow cause
-> rerun the focused check
-> commit
-> continue
```

Do not spend time designing hypothetical future abstractions.

Do not broaden the test suite unless an actual regression risk has been discovered.

The priority order is:

```text
1. working native code
2. Python-oracle parity
3. GGUF fast startup / residency
4. GB10 performance
5. additional checkpoints
6. cleanup
```

---

# 28. First commands to execute

The implementation should begin with these commands, in this order:

```bash
mkdir -p _reference
git clone https://github.com/luca-saggese/o1.c _reference/o1.c

printf '\n/_reference/\n/models/*.gguf\n' >> .gitignore

git add .gitignore
git commit -m "chore: add ignored o1.c reference workspace"
```

Then create the reuse map by inspecting:

```bash
_reference/o1.c/src/io/gguf.c
_reference/o1.c/src/io/gguf.h
_reference/o1.c/src/io/safetensors.c
_reference/o1.c/src/io/json.c
_reference/o1.c/src/cuda/gemm.cu
_reference/o1.c/src/cuda/gemm.h
_reference/o1.c/src/cuda/embed.cu
_reference/o1.c/src/cuda/residual.cu
_reference/o1.c/src/cuda/rope.cu
_reference/o1.c/src/cuda/hd_cudnn_sdpa.cu
_reference/o1.c/src/cuda/norm.cu
_reference/o1.c/src/cuda/vision_kernels.cu
_reference/o1.c/src/model/tokenizer.c
_reference/o1.c/tools/_gen_c_tables.py
_reference/o1.c/src/runtime/o1_timing.c
_reference/o1.c/Makefile
```

The next executable milestone is **not** ModernBERT.

It is:

```text
HF Laya checkpoint
-> laya_to_gguf.py
-> laya-bf16.gguf
-> native GGUF loader
-> resident weights on GB10
```

Only after that pipeline works should encoder implementation begin.
