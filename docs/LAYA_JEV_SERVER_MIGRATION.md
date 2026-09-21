Laya — TypeSafe Jev-Compatible Server Migration

Status: implementation specification for agent
Target: native laya C/CUDA runtime on NVIDIA GB10 / DGX Spark
Starting point: the existing hidream-server implementation from _reference/o1.c
Primary compatibility target: TypeSafe Jev native System One HTTP protocol
Primary endpoint: POST /v1/systemone
Core requirement: reuse the existing server transport/lifecycle code; do not write a new HTTP server.

0. Mission

Implement a new executable:

build/laya-server

that exposes the existing native Laya inference engine through a wire-compatible TypeSafe Jev/System One API.

The intended integration model is:

existing Jev client / SDK
        |
        | POST /v1/systemone
        v
    laya-server
        |
        | one resident local model
        | one request = many typed questions
        | questions evaluated in one batched forward
        v
typed answers + probabilities

The server must behave like a System One decision endpoint, not like an OpenAI chat/completions server.

The implementation goal is:

same request shape
same endpoint
same typed question primitives
same answer shapes
same top-level response shape
same multi-question request semantics

Model identity may remain truthful to the local implementation; protocol compatibility must not require pretending that the local model is TypeSafe Jev.

1. Non-negotiable reuse policy

Do not create a new socket server.

Do not introduce libcurl, civetweb, mongoose, Boost, Python, Node, FastAPI, or another HTTP framework.

The repository already has a production-oriented C server implementation in:

_reference/o1.c/src/server/o1_server.c

and an implementation document in:

_reference/o1.c/docs/O1_OPENAI_IMAGE_SERVER_IMPLEMENTATION.md

Port that implementation into Laya and strip the image-specific parts.

The migration principle is:

COPY
→ REMOVE HiDream-specific logic
→ REPLACE request/response model layer
→ KEEP transport and lifecycle

Do not start with an empty server.c.

2. Verified Jev protocol contract

The target is TypeSafe's native System One API:

POST /v1/systemone
Content-Type: application/json
Authorization: Bearer <token>

Native TypeSafe endpoint:

https://api.typesafe.ai/v1/systemone

The request body is:

{
  "model": "jev-latest",
  "state": "...",
  "questions": {
    "question_id": {
      "type": "choice | score | noul",
      "instructions": "...",
      "criteria": "..."
    }
  }
}

All questions in one request are evaluated against the same state.

They are independent decisions and are evaluated in parallel.

There is no text generation and no streaming protocol.

3. Request schema

3.1 Top-level object

Accepted fields:

model
state
questions

Required:

state
questions

model is optional at the wire level.

Unknown top-level fields should be skipped for forward compatibility rather than causing a parser crash.

state

Accept:

string
object
array

Reject:

number
boolean
null

Structured state must be rendered exactly like the Python oracle before tokenization.

Current Python reference:

def serialize_state(state):
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)

Therefore:

DO NOT feed the raw JSON slice directly to the tokenizer.

Whitespace in the incoming HTTP JSON must not change the semantic serialized state.

Structured state must be parsed and serialized according to the local Laya Python reference.

4. Question primitives

Support exactly:

choice
score
noul

No free-text answer type.

No generated strings.

No OpenAI JSON-schema response mode.

Question IDs are caller-controlled keys and must be returned unchanged.

They are identifiers only; the question ID itself is not model instructions.

5. choice

Request:

{
  "type": "choice",
  "instructions": "Which team should handle this?",
  "criteria": {
    "billing": "Payments and refunds",
    "technical": "Bugs and outages",
    "other": null
  }
}

Requirements:

criteria is an object
at least 2 options
up to 255 protocol-level options
option names are strings
descriptions may be string/object/array/null

The local model may have a stricter effective limit due to head_max_len.
If the sequence cannot represent all options, return a validation error.
Do not silently truncate away an option.

Response:

{
  "type": "choice",
  "choice": "billing",
  "probabilities": {
    "billing": 0.82,
    "technical": 0.13,
    "other": 0.05
  },
  "confidence": 0.71
}

Required response semantics:

choice
    one of the original criterion keys

probabilities
    object keyed by the original criterion keys

confidence
    the model confidence for this closed-set answer

Do not rename the keys.

Do not return numeric option indices to the client.

6. score

Request:

{
  "type": "score",
  "instructions": "How urgent is the claim?",
  "criteria": [
    "Routine",
    "Priority",
    "Urgent",
    "Immediate"
  ]
}

Requirements:

criteria is an ordered array
minimum 2 levels
protocol target maximum 10 levels
levels may be string/object/array/null if supported by local rendering

Response:

{
  "type": "score",
  "score": 2.68,
  "legend": {
    "0": "Routine",
    "1": "Priority",
    "2": "Urgent",
    "3": "Immediate"
  },
  "probabilities": {
    "0": 0.00,
    "1": 0.00,
    "2": 0.32,
    "3": 0.68
  },
  "confidence": 0.68
}

score is the probability-weighted expected rubric index:

score = Σ(index * probability[index])

It is not normalized to 0..1.

legend keys are JSON strings:

"0"
"1"
"2"
...

probabilities uses those same stringified numeric indices.

7. noul

Request:

{
  "type": "noul",
  "instructions": "Is this claim urgent?"
}

Optional criteria:

{
  "type": "noul",
  "instructions": "Is the vehicle drivable?",
  "criteria": {
    "true": "The vehicle can safely continue under its own power",
    "false": "The vehicle is immobilized or unsafe to drive"
  }
}

Only these criterion keys are valid:

true
false

Response:

{
  "type": "noul",
  "noul": 0.93
}

Critical compatibility rule:

Jev Noul has NO confidence field.

The local Laya Python implementation currently computes a confidence-like value for noul.

Do not expose it in the Jev-compatible wire response.

The noul number itself is:

P(true)

8. Do not expose Laya-specific internal answer fields

Current local inference may have internal fields such as:

action
act_probability
qtype
raw_logits
marker positions
calibration bucket

These are useful internally and for debugging.

They are not part of the Jev native response protocol.

The public /v1/systemone response must omit them.

In particular, do not emit:

{
  "action": {
    "act_probability": 0.9
  }
}

from the Jev-compatible endpoint.

If a future AIDA endpoint wants richer diagnostics, expose it through a separate explicitly non-Jev API.

Do not contaminate /v1/systemone.

9. Top-level success response

Return:

{
  "model": "laya-local",
  "answers": {
    "...": {}
  },
  "usage": {
    "input_tokens": 1234,
    "output_tokens": 0
  }
}

answers must use exactly the original request question IDs.

Recommended local usage semantics:

input_tokens
    total attention-mask token count actually processed across the question batch

output_tokens
    0

Do not invent generated-token counts.

10. Model identity and aliases

Do not falsely identify the local model as an actual TypeSafe model.

Recommended startup option:

--served-model-name laya-local

or a model-specific truthful identifier:

laya-english-bf16
aida-auto-it-v1

Request validation:

if model is absent:
    use resident model

if model matches served model:
    accept

if model matches an explicitly configured alias:
    accept

otherwise:
    return model-not-found / validation response

Optional compatibility alias:

--model-alias jev-latest

This may be useful when exercising clients whose default model is jev-latest.

It must be explicit configuration, not silent impersonation.

The response should still report the actual configured served model name.

11. Multiple questions are one inference request

This is fundamental.

Do not implement:

for each question:
    call model forward

The local Python behavior is:

question 0 ─┐
question 1 ─┤
question 2 ─┼→ collate → ONE model forward
...         │
question N ─┘

Native server behavior must be:

parse all questions
        ↓
build one Laya sequence per question
        ↓
collate into [B,S]
        ↓
ONE batched native inference
        ↓
serialize B typed answers

B is the number of questions in the request.

The state is conceptually shared but appears in each encoded decision sequence, matching the Python implementation.

Do not perform one HTTP-side inference call per question.

12. Exact local sequence semantics

The server must call the existing native Laya request-building path.

Do not duplicate sequence construction in the server.

The authoritative Python sequence is:

[CLS]
<question type> question: <instructions>
[SEP]
[MASK] option0
[MASK] option1
...
[SEP]
state
[SEP]

Use the already ported native implementation.

Important Python normalization semantics:

serialize_state(dict_or_list)
    -> json.dumps(..., ensure_ascii=False)

structured instructions
    -> json.dumps(...)

criterion values
    -> JSON rendering

noul options
    -> false then true

The server parser produces typed/raw request data.

The runtime owns model-specific rendering.

Do not merge those layers.

13. Source server migration map

Starting source:

_reference/o1.c/src/server/o1_server.c

Create:

src/server/laya_server.c

by copying the existing server and reducing it.

KEEP almost unchanged

process-global shutdown state
stop_signal_handler()
buf
xmalloc/xrealloc/xstrdup/xstrndup
JSON string/number/bool parser
bounded recursive JSON skip
json_raw_value
json_escape
wall_ms()
send_all()
CORS support
HTTP reason/helper layer
http_request
HTTP header parser
Content-Length handling
request size limits
socket timeouts
listen_on()
client thread lifecycle
bounded worker queue
queue mutex/condition variables
worker thread
graceful drain/shutdown
optional API-key checking pattern
CLI parsing style
startup logging style
resident-model startup gate

REMOVE completely

base64 image support
PNG encoding
multipart/form-data
image upload handling
reference images
layout conditions
HiDream generation request
scheduler settings
seed/noise/image size options
image response_format
OpenAI Images endpoints
/v1/images/generations
/v1/images/edits
HiDream profile dev/base selection
HiDream LoRA request parsing
image-specific errors

REPLACE

hd_generation_engine
    ->
resident Laya model/runtime

HiDream server_job fields
    ->
System One request + typed questions + answer body

job_run(): image generation
    ->
one batched Laya decision inference

OpenAI Images response serialization
    ->
Jev System One answer serialization

14. Resident model lifecycle

Preserve the best property of hidream-server:

load once
serve many

Required lifecycle:

process startup
    |
    +-> parse CLI
    +-> select CUDA device
    +-> load GGUF once
    +-> upload resident tensors once
    +-> initialize tokenizer once
    +-> initialize CUDA/cuBLAS/cuDNN once
    +-> initialize reusable workspaces/plans once
    |
    v
listen socket
    |
    +-> request
    +-> request
    +-> request
    ...
    |
shutdown
    |
    +-> drain queue
    +-> free runtime once

The socket must not begin listening until model initialization succeeds.

No model loading in job_run().

No GGUF parsing per request.

No CUDA weight upload per request.

15. Server runtime API

Do not rewrite the inference engine for the server.

Wrap the existing resident native engine with the narrowest API possible.

Use the current runtime structures if already suitable.

Conceptually:

typedef struct laya_engine laya_engine;

int laya_engine_open(
    laya_engine **out,
    const char *model_path,
    int device_id);

int laya_engine_system_one(
    laya_engine *engine,
    const laya_system_one_request *req,
    laya_system_one_response *resp);

void laya_engine_close(laya_engine *engine);

If equivalent functions already exist, use them.

Do not create duplicate forward implementations.

The server is orchestration and protocol translation only.

16. Suggested request data structures

Keep ownership explicit.

Conceptual structures:

typedef enum {
    LAYA_Q_CHOICE = 0,
    LAYA_Q_SCORE  = 1,
    LAYA_Q_NOUL   = 2
} laya_question_type;

typedef struct {
    char *id;

    laya_question_type type;

    /* JSON content rendered according to oracle semantics. */
    char *instructions_text;

    /* choice */
    char **choice_keys;
    char **choice_rendered;
    int n_choices;

    /* score */
    char **score_rendered;
    int n_levels;

    /* noul optional true/false criterion rendering */
    char *false_rendered;
    char *true_rendered;
} laya_wire_question;

typedef struct {
    char *requested_model;

    /* Already normalized according to serialize_state(). */
    char *state_text;

    laya_wire_question *questions;
    int n_questions;
} laya_wire_request;

Names may differ.

Do not use these exact structs if equivalent runtime structures already exist.

17. JSON parsing: reuse existing parser, extend only what is needed

The HiDream server already includes:

json_string
json_number
json_bool
json_skip_value
json_raw_value
json_escape

Reuse them.

Add only minimal helpers for:

questions object iteration
criteria object iteration
criteria array iteration
structured JSON content extraction/normalization

Do not introduce a second JSON parser unless the Laya repository already has a better parser that can be reused directly.

Prefer the implementation already copied from o1.c.

18. Structured JSON content

The Jev protocol permits structured content for state/instructions/criteria.

This is subtle.

The server must not merely store incoming raw JSON bytes and concatenate them into the model input.

For model parity, normalize according to Python behavior.

Examples:

Incoming:

{
  "state": {
     "claim": "collision",
     "drivable": false
  }
}

Model text should correspond to Python:

json.dumps(state, ensure_ascii=False)

Similarly:

instructions object/list
criterion object/list/number

must follow the local Python rendering functions.

Inspect and port exactly from:

_reference/laya/laya/common.py

especially:

serialize_state()
render_criterion()
render_options()

Do not invent different JSON spacing rules.

Tokenization parity depends on this.

19. Request validation

Implement validation before enqueueing GPU work.

Recommended limits:

request body default: 512 KiB
questions per request: 1..100
choice options: 2..255 at wire level
score levels: 2..10
JSON nesting: reuse existing bounded nesting limit

Then apply runtime/model-specific limits:

max_len
head_max_len
maximum marker count
tokenizer limits

If a question cannot be represented without losing required options:

reject it

Do not silently change its closed set.

Validation should be CPU-only.

Invalid requests must never enter the GPU queue.

20. HTTP endpoint routing

Required:

POST /v1/systemone
GET  /v1/models
OPTIONS *

Recommended local extension:

GET /health

/health is not part of the strict Jev evaluation protocol and should remain clearly auxiliary.

Remove image routes.

Unknown routes:

404

Wrong method:

405

Wrong request content type:

415

Only:

application/json

is required for /v1/systemone.

No multipart.

No streaming.

21. GET /v1/models

TypeSafe also exposes a model-list endpoint.

Implement it using the resident model identity.

Keep this simple.

Example local response should follow the shape expected by the client chosen for compatibility testing.

Do not spend significant time cloning every provider-specific descriptive field.

At minimum expose the configured local model and aliases in a stable JSON shape.

Before finalizing this endpoint, inspect the official TypeSafe SDK used for the smoke test and match the fields it actually reads.

POST /v1/systemone is higher priority than model listing.

22. Authentication

TypeSafe uses:

Authorization: Bearer <API_KEY>

For local deployment support:

--api-key <secret>

or environment equivalent.

Behavior:

api key configured
    -> require Authorization: Bearer exactly

no api key configured
    -> local server may allow unauthenticated requests

Do not send or log tokens.

Do not include authorization contents in error messages.

The authorization header must never reach model state.

23. Response header

Where easy, add:

x-typesafe-request-id

using a locally generated request ID.

The existing server already has random ID helpers.

Example:

req_<hex>

Do not block the main implementation on this header.

The JSON wire contract is more important.

24. Error behavior

Use HTTP status codes consistently.

Suggested mapping:

400
    malformed JSON / malformed basic request

401
    configured local API key missing/invalid

404
    endpoint or requested model not available

405
    method not allowed

413
    body too large

415
    unsupported Content-Type

422
    structurally valid JSON but invalid System One question schema

429
    bounded inference queue full

500
    internal serialization/runtime error

503
    shutting down / model unavailable

The official clients recognize both 400 and 422 as validation failures.

Do not preserve the OpenAI Images error envelope as a requirement.

Use a small JSON error object.

For example:

{
  "error": "Question 'route' choice criteria must contain at least two options."
}

If a live TypeSafe fixture/key is available, capture the current direct error body and align later.

Do not block first server bring-up on byte-identical provider error prose.

Successful request/response compatibility is the primary contract.

25. Queue and concurrency model

Reuse hidream-server's architecture:

HTTP connection thread
    |
    +-> read / parse / validate
    +-> enqueue job
    +-> wait
    +-> write response

single resident GPU worker
    |
    +-> dequeue
    +-> one batched System One inference
    +-> serialize result
    +-> signal connection thread

This is correct for first release.

Do not let arbitrary HTTP threads call the CUDA model concurrently.

Keep a bounded queue.

A full queue should return:

429

or an equivalent overload response.

26. Distinguish two forms of batching

This is critical.

Required immediately: question batching inside one request

one state
+
N questions
=
one native batched forward

This must work.

Not required initially: batching multiple HTTP requests together

Do not implement cross-request dynamic batching yet.

That is a later performance optimization.

First protocol release:

request A → one batched forward for A's questions
request B → next batched forward for B's questions

27. Worker implementation

job_run() should be extremely small.

Conceptually:

static void job_run(server_job *j) {
    /* request already parsed and validated */

    laya_system_one_response out = {0};

    int rc = laya_engine_system_one(
        g_engine,
        &j->request,
        &out);

    if (rc != 0) {
        /* translate runtime error */
        ...
        return;
    }

    j->body = serialize_jev_response(&out);
}

Do not put tokenizer logic, CUDA kernels, or model forward logic into laya_server.c.

28. Response serialization

Build JSON using the existing buf and json_escape code.

Do not depend on a third-party serializer.

Serialize answers in request question order for deterministic output.

Choice

"route": {
  "type": "choice",
  "choice": "billing",
  "probabilities": {
    "billing": 0.826055,
    "technical": 0.0994325,
    "other": 0.074513
  },
  "confidence": 0.0
}

Use the native confidence value computed by the Laya output layer.

Score

"urgency": {
  "type": "score",
  "score": 1.72,
  "legend": {
    "0": "routine",
    "1": "priority",
    "2": "urgent"
  },
  "probabilities": {
    "0": 0.05,
    "1": 0.18,
    "2": 0.77
  },
  "confidence": 0.0
}

Noul

"injury": {
  "type": "noul",
  "noul": 0.93
}

Again:

NO `confidence` on noul.
NO `action`.
NO `act_probability`.

29. Confidence semantics

For local Laya, use its existing confidence computation for:

choice
score

Do not invent a new server-side confidence function.

Do not replace it with top-1 probability.

Do not use internal action probability as Jev confidence.

For noul, omit confidence entirely.

30. Calibration

Preserve the existing Laya calibration path:

raw logits
→ temperature selected by qtype / option bucket
→ calibrated probabilities
→ response

The server must serialize the calibrated probabilities.

Do not accidentally expose uncalibrated softmax values.

The server should not implement calibration independently; call the existing result-building/runtime layer.

31. Server source should remain mostly copied transport code

Expected final server file composition:

~60-75% copied/adapted HTTP/queue/lifecycle code from o1_server.c
~15-25% System One request parser/validator
~10% Laya engine call + Jev response serializer

If the agent ends up writing a large new HTTP stack, stop.

If the server starts containing transformer math, stop.

If the server duplicates build_sequence, stop.

32. Suggested file layout

src/server/
    laya_server.c

src/laya/
    ... existing runtime ...

Optional only if it materially simplifies ownership:

src/server/
    jev_protocol.c
    jev_protocol.h

But do not split files merely for aesthetics during bring-up.

A single laya_server.c copied from o1_server.c is acceptable and likely faster.

33. Build integration

Copy the server build pattern from _reference/o1.c/Makefile.

Add:

make server

or:

make laya-server

producing:

build/laya-server

Use the exact same CUDA objects/runtime libraries as the existing native Laya executable.

Do not create a separate CUDA implementation for the server.

34. CLI

Minimum server CLI:

--model <gguf>
--host <ip>
--port <port>
--device <cuda-device>
--queue-depth <N>
--api-key <secret>
--cors
--served-model-name <name>
--model-alias <name>

Defaults may follow hidream-server where sensible.

Example:

./build/laya-server \
  --model models/laya-bf16.gguf \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name laya-local

Startup should clearly print:

model path
served model name
CUDA device
listen address
queue depth
auth enabled/disabled

Never print API-key contents.

35. Canonical smoke request

Use one request exercising all three primitives simultaneously.

{
  "model": "laya-local",
  "state": {
    "ticket": "I was charged twice and I cannot use my car after the accident."
  },
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "Payment or invoice issue",
        "claims": "Insurance claim handling",
        "other": "Neither category"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this situation?",
      "criteria": [
        "Routine",
        "Priority",
        "Urgent"
      ]
    },
    "vehicle_drivable": {
      "type": "noul",
      "instructions": "Is the vehicle still safely drivable?"
    }
  }
}

This must result in:

ONE request
ONE question batch
ONE model forward
THREE typed answers

36. Canonical response shape test

The smoke response must structurally look like:

{
  "model": "laya-local",
  "answers": {
    "team": {
      "type": "choice",
      "choice": "...",
      "probabilities": {
        "billing": 0.0,
        "claims": 0.0,
        "other": 0.0
      },
      "confidence": 0.0
    },
    "urgency": {
      "type": "score",
      "score": 0.0,
      "legend": {
        "0": "Routine",
        "1": "Priority",
        "2": "Urgent"
      },
      "probabilities": {
        "0": 0.0,
        "1": 0.0,
        "2": 0.0
      },
      "confidence": 0.0
    },
    "vehicle_drivable": {
      "type": "noul",
      "noul": 0.0
    }
  },
  "usage": {
    "input_tokens": 0,
    "output_tokens": 0
  }
}

Numbers above are placeholders only.

Validate structure and actual inference values separately.

37. Minimal test strategy

Do not build a huge test suite.

Use four focused layers.

Test A — parser without GPU

One small C or Python fixture covering:

state string
state object
choice
score
noul
structured criterion
unknown top-level field

Test B — invalid requests

Only a compact set:

invalid JSON
missing state
missing questions
unknown qtype
choice <2 options
score <2 levels
bad noul criteria key
too many questions
wrong Content-Type

Do not enumerate hundreds of cases.

Test C — live local curl smoke

Start resident server once.

Send one mixed-question request.

Verify:

HTTP 200
model
answers keys preserved
choice shape
score shape
noul shape
usage
no action field
no noul confidence

Test D — SDK compatibility smoke

This is important.

Point an official TypeSafe SDK, if available locally, at:

http://127.0.0.1:<port>

and perform one mixed System One call.

Goal:

client code changes only base URL/model

No custom adapter.

If the SDK cannot override base URL easily, use its low-level documented request types or curl fixture.

Do not vendor a large SDK dependency into the C project.

38. Optional differential fixture against TypeSafe

If a TypeSafe API key is already available to the developer, capture request/response shape for one request from:

https://api.typesafe.ai/v1/systemone

Use it only to confirm:

field names
field omission/presence
top-level model/answers/usage
noul has no confidence
score legend/probabilities shape

Do not try to match TypeSafe model numerical answers.

The local model is different.

Do not make external network access a required build/test dependency.

39. Performance instrumentation

Keep minimal timing fields internally:

parse_ms
queue_wait_ms
encode_ms
inference_ms
serialize_ms
total_ms

Do not add them to the Jev-compatible JSON response.

Log them server-side when verbose timing is enabled.

We want eventual measurements for:

1 question
4 questions
8 questions
16 questions
32 questions

but first finish wire compatibility.

40. Performance rule

Do not optimize HTTP before measuring.

Expected hot path:

HTTP parse
    cheap

sequence/tokenization
    CPU

batched model inference
    dominant

JSON serialization
    tiny

Keep the model resident.

Reuse CUDA/cuBLAS/cuDNN plans/workspaces.

Do not add per-request model allocation.

After baseline, likely performance work:

batched question path
persistent workspaces
remove synchronization
length buckets
cuBLASLt plan reuse
cuDNN plan reuse
possibly inter-request microbatching later

Cross-request microbatching is explicitly post-MVP.

41. Implementation milestones and commits

Use short milestones with immediate commits.

S1 — Server skeleton port

Copy/reduce o1_server.c.

Keep:

HTTP
JSON helpers
thread lifecycle
queue
shutdown
CLI

Remove image-specific code.

Build a server that starts with resident Laya model and exposes a health endpoint.

Commit:

feat: port resident http server to laya

S2 — Jev request parser

Implement:

state
model
questions
choice
score
noul
validation

No GPU changes.

Commit:

feat: parse typesafe system one requests

S3 — Native batched inference wiring

Convert parsed questions into the existing Laya batched request API.

Call one model forward.

Commit:

feat: serve batched laya decisions

S4 — Exact response serializer

Implement:

choice
score
noul
model
usage

Ensure:

noul has no confidence
no action field

Commit:

feat: add jev-compatible response format

S5 — Protocol smoke

Run:

mixed choice/score/noul request
invalid request sample
queue overload sample
SDK/base-url smoke if available

Commit:

test: validate system one wire compatibility

Do not combine the entire migration into one giant final commit.

42. Hard constraints for the agent

Do not:

write a new HTTP server
introduce a web framework
rewrite JSON handling
rewrite native inference
duplicate tokenizer logic
duplicate build_sequence
run one forward per question
expose Laya action fields
add confidence to Noul
expose uncalibrated probabilities
hot-load models per request
support image endpoints
support chat/completions
support text generation
implement streaming
build inter-request dynamic batching yet

43. Definition of done

The migration is complete when all of the following are true.

Build

build/laya-server exists

Startup

GGUF/model loaded once
CUDA resident runtime initialized once
server listens only after successful preload

Protocol

POST /v1/systemone works
GET /v1/models works sufficiently for selected client
state accepts string/object/array
choice works
score works
noul works
many questions work in one request

Response compatibility

question IDs preserved
choice fields match Jev
score fields match Jev
noul fields match Jev
Noul has no confidence
no Laya-only action fields
top-level model/answers/usage present

Runtime

one request with N questions
→ one collated question batch
→ one model forward

Lifecycle

model does not reload
GGUF does not reparse
weights do not re-upload

Reliability

malformed requests rejected before GPU
bounded body
bounded question count
bounded queue
graceful shutdown
no CUDA calls from concurrent client threads

44. Final agent instruction

Read first:

_reference/o1.c/src/server/o1_server.c
_reference/o1.c/docs/O1_OPENAI_IMAGE_SERVER_IMPLEMENTATION.md
_reference/laya/laya/common.py
_reference/laya/laya/agent.py

Then inspect the current native Laya engine API.

Do not design around remembered APIs from earlier commits.

Port the existing server.

Do the minimum work necessary to translate:

Jev wire request
    ↓
existing native Laya batched request
    ↓
existing native result
    ↓
Jev wire response

Do not stop after writing a plan.

Implement through the first live local /v1/systemone request.

If the server implementation grows large, check whether HiDream/server code or the existing Laya runtime already implements the functionality before adding new code.

45. Sources used to define the compatibility contract

Primary TypeSafe/public sources reviewed for this specification:

https://typesafe.ai/blog/introducing-system-one-models-and-jev
https://api.typesafe.ai/v1/systemone
https://github.com/TypeSafeAI/typesafe-router
https://github.com/TypeSafeAI/typesafe-playground

The TypeSafe public protocol establishes:

POST /v1/systemone
{model,state,questions}
choice / score / noul
parallel questions over one state
typed answers
choice/score probabilities + confidence
noul probability

Local behavioral oracle:

_reference/laya/laya/common.py
_reference/laya/laya/agent.py

Server transport/lifecycle donor:

_reference/o1.c/src/server/o1_server.c

The TypeSafe protocol is the public HTTP contract.

The Laya Python implementation is the model-semantic oracle.

The HiDream server is the transport/lifecycle implementation donor.

Keep those three responsibilities separate.
