#!/usr/bin/env python3
"""
laya_to_gguf.py -- Hugging Face Laya checkpoint -> GGUF v3 pack for the native runtime.

The writer/layout follows tools/hidream_convert.py from o1.c (same GGUF v3
subset: general.alignment = 256, BF16 payload, flat tensor table). Everything
that can be done offline is done here:

  * tensor rename -> stable native names (encoder.layers.00.attn.Wqkv.weight, ...)
  * F16/F32 -> BF16 normalization (the runtime contract is BF16-only)
  * static model metadata (laya.* + modernbert.*) so the C runtime needs no
    sidecar JSON and no config parsing of its own
  * temperature_by_options resolved to a flat bucket table (no runtime logic)

No transposes: PyTorch stores every Linear weight as [out, in], which is
exactly the layout hd_linear() (transpose_w = 1) expects.

Usage:
  python3 tools/laya_to_gguf.py --model convaiinnovations/laya \
      --output models/laya-bf16.gguf
  python3 tools/laya_to_gguf.py --model /path/to/local/laya \
      --output models/laya-bf16.gguf
  python3 tools/laya_to_gguf.py --model convaiinnovations/laya \
      --subfolder multilingual --variant multilingual \
      --output models/laya-multilingual-bf16.gguf
"""

import argparse
import json
import os
import struct
import sys

import numpy as np

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian
GGUF_VERSION = 3
ALIGNMENT = 256

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_BF16 = 30

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

QTYPE_NAMES = {0: "choice", 1: "score", 2: "noul"}


def f32_buf_to_bf16_np(raw):
    """Vectorized F32 bytes -> BF16 bytes (round-to-nearest-even)."""
    a = np.frombuffer(raw, dtype=np.float32)
    u = a.view(np.uint32).astype(np.uint64)
    lsb = (u >> 16) & 1
    u = u + 0x7FFF + lsb
    return (u >> 16).astype(np.uint16).tobytes()


def f16_buf_to_bf16_np(raw):
    """F16 bytes -> BF16 bytes via float32 (exact for normal values)."""
    a = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    u = a.view(np.uint32).astype(np.uint64)
    lsb = (u >> 16) & 1
    u = u + 0x7FFF + lsb
    return (u >> 16).astype(np.uint16).tobytes()


def align_up(v, a):
    return (v + a - 1) & ~(a - 1)


class Writer:
    def __init__(self, path):
        self.f = open(path, "wb")
        self.pos = 0

    def write(self, data):
        self.f.write(data)
        self.pos += len(data)

    def pad(self, alignment):
        pad = (alignment - (self.pos % alignment)) % alignment
        if pad:
            self.write(b"\x00" * pad)

    def u8(self, v):
        self.write(struct.pack("<B", v))

    def u32(self, v):
        self.write(struct.pack("<I", v))

    def u64(self, v):
        self.write(struct.pack("<Q", v))

    def f32(self, v):
        self.write(struct.pack("<f", v))

    def string(self, s):
        b = s.encode("utf-8")
        self.u64(len(b))
        self.write(b)

    def kv_string(self, key, value):
        self.string(key)
        self.u32(GGUF_TYPE_STRING)
        self.string(value)

    def kv_uint32(self, key, value):
        self.string(key)
        self.u32(GGUF_TYPE_UINT32)
        self.u32(value)

    def kv_f32(self, key, value):
        self.string(key)
        self.u32(GGUF_TYPE_FLOAT32)
        self.f32(value)

    def kv_f32_array(self, key, values):
        self.string(key)
        self.u32(GGUF_TYPE_ARRAY)
        self.u32(GGUF_TYPE_FLOAT32)
        self.u64(len(values))
        for v in values:
            self.f32(float(v))

    def kv_u32_array(self, key, values):
        self.string(key)
        self.u32(GGUF_TYPE_ARRAY)
        self.u32(GGUF_TYPE_UINT32)
        self.u64(len(values))
        for v in values:
            self.u32(int(v))

    def close(self):
        self.f.close()


# ------------------------------------------------------------------ #
# Checkpoint access                                                   #
# ------------------------------------------------------------------ #

def resolve_checkpoint(model, subfolder, token):
    """Returns the local directory holding the checkpoint files."""
    if os.path.exists(model):
        root = model
    else:
        if model.startswith(("/", "./", "../")):
            raise SystemExit("local model path not found: %r" % model)
        from huggingface_hub import snapshot_download

        root = snapshot_download(model, token=token or os.environ.get("HF_TOKEN"))
    if subfolder:
        root = os.path.join(root, subfolder)
        if not os.path.isdir(root):
            raise SystemExit("subfolder %r not found in %r" % (subfolder, model))
    return root


def load_tensors(model_dir):
    """{name: (numpy array, original dtype string)} from model.safetensors."""
    from safetensors import safe_open

    path = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(path):
        raise SystemExit("model.safetensors not found in %r" % model_dir)
    out = {}
    with safe_open(path, framework="np") as f:
        for name in f.keys():
            sl = f.get_slice(name)
            out[name] = (sl[:], sl.get_dtype())
    return out


def to_bf16_bytes(arr, dtype):
    """BF16 payload bytes for one tensor, normalizing F16/F32 -> BF16."""
    if dtype == "BF16":
        return np.ascontiguousarray(arr).tobytes()
    if dtype == "F16":
        return f16_buf_to_bf16_np(np.ascontiguousarray(arr).tobytes())
    if dtype == "F32":
        return f32_buf_to_bf16_np(np.ascontiguousarray(arr).tobytes())
    raise SystemExit("unsupported dtype %r" % dtype)


# ------------------------------------------------------------------ #
# Laya-specific layout                                                #
# ------------------------------------------------------------------ #

def layer_is_global(cfg, layer_idx):
    every = int(cfg.get("global_attn_every_n_layers", 3) or 0)
    layer_types = cfg.get("layer_types")
    if layer_types:
        return 1 if layer_types[layer_idx] == "full_attention" else 0
    if every <= 0:
        return 1
    return 1 if (layer_idx % every == 0) else 0


def temp_bucket(qtype, k):
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


def build_manifest(tensors, cfg, agent_cfg):
    """Returns (ordered [(native_name, source_name)], meta)."""
    n_layers = int(cfg["num_hidden_layers"])
    n_head_layers = int(agent_cfg.get("head_layers", 2))
    n_act = len(agent_cfg.get("act_costs", {})) + 1

    required = ["encoder.embeddings.tok_embeddings.weight",
                "encoder.embeddings.norm.weight",
                "encoder.final_norm.weight",
                "type_emb.weight",
                "scorer.0.weight", "scorer.0.bias",
                "scorer.1.weight", "scorer.1.bias",
                "scorer.3.weight", "scorer.3.bias",
                "act_head.0.weight", "act_head.0.bias",
                "act_head.2.weight", "act_head.2.bias",
                "temperature"]
    for i in range(n_layers):
        required += ["encoder.layers.%d.attn.Wqkv.weight" % i,
                     "encoder.layers.%d.attn.Wo.weight" % i,
                     "encoder.layers.%d.mlp.Wi.weight" % i,
                     "encoder.layers.%d.mlp.Wo.weight" % i,
                     "encoder.layers.%d.mlp_norm.weight" % i]
        if i > 0:
            required.append("encoder.layers.%d.attn_norm.weight" % i)
    for i in range(n_head_layers):
        for sub in ("norm1.weight", "norm1.bias", "linear1.weight", "linear1.bias",
                    "norm2.weight", "norm2.bias", "linear2.weight", "linear2.bias",
                    "self_attn.in_proj_weight", "self_attn.in_proj_bias",
                    "self_attn.out_proj.weight", "self_attn.out_proj.bias"):
            required.append("head.layers.%d.%s" % (i, sub))

    missing = [n for n in required if n not in tensors]
    if missing:
        raise SystemExit("checkpoint missing %d required tensors, e.g. %s"
                         % (len(missing), missing[:5]))

    entries = []

    entries.append(("encoder.embeddings.tok_embeddings.weight",
                    "encoder.embeddings.tok_embeddings.weight"))
    entries.append(("encoder.embeddings.norm.weight", "encoder.embeddings.norm.weight"))
    for i in range(n_layers):
        p = "encoder.layers.%02d" % i
        s = "encoder.layers.%d" % i
        if i > 0:
            entries.append((p + ".attn_norm.weight", s + ".attn_norm.weight"))
        entries.append((p + ".attn.Wqkv.weight", s + ".attn.Wqkv.weight"))
        entries.append((p + ".attn.Wo.weight", s + ".attn.Wo.weight"))
        entries.append((p + ".mlp_norm.weight", s + ".mlp_norm.weight"))
        entries.append((p + ".mlp.Wi.weight", s + ".mlp.Wi.weight"))
        entries.append((p + ".mlp.Wo.weight", s + ".mlp.Wo.weight"))
    entries.append(("encoder.final_norm.weight", "encoder.final_norm.weight"))
    entries.append(("decision.type_emb.weight", "type_emb.weight"))
    for i in range(n_head_layers):
        p = "decision.head.%02d" % i
        s = "head.layers.%d" % i
        for sub in ("norm1.weight", "norm1.bias", "linear1.weight", "linear1.bias",
                    "norm2.weight", "norm2.bias", "linear2.weight", "linear2.bias",
                    "self_attn.in_proj_weight", "self_attn.in_proj_bias",
                    "self_attn.out_proj.weight", "self_attn.out_proj.bias"):
            entries.append((p + "." + sub, s + "." + sub))
    for sub in ("0.weight", "0.bias", "1.weight", "1.bias", "3.weight", "3.bias"):
        entries.append(("decision.scorer." + sub, "scorer." + sub))
    for sub in ("0.weight", "0.bias", "2.weight", "2.bias"):
        entries.append(("decision.act_head." + sub, "act_head." + sub))

    temperature = [float(x) for x in agent_cfg.get("temperature", [1.0, 1.0, 1.0])]
    tbo = agent_cfg.get("temperature_by_options", {}) or {}
    # Resolve every bucket the runtime can ask for, so C needs no lookup logic.
    # Index = qtype * 4 + size_index, size_index 0..3 = "2" | "3-5" | "6-10" | "11+".
    buckets = []
    for qt in range(3):
        for size in ("2", "3-5", "6-10", "11+"):
            buckets.append(float(tbo.get("%s:%s" % (QTYPE_NAMES[qt], size), temperature[qt])))

    meta = {
        "n_layers": n_layers,
        "n_head_layers": n_head_layers,
        "n_act": n_act,
        "temperature": temperature,
        "buckets": buckets,
    }
    return entries, meta


def write_gguf(output, entries, tensors, cfg, agent_cfg, meta, variant, source_name):
    w = Writer(output)
    w.u32(GGUF_MAGIC)
    w.u32(GGUF_VERSION)
    w.u64(len(entries))

    rp = cfg.get("rope_parameters", {}) or {}
    kvs = [
        ("general.architecture", "laya"),
        ("general.name", source_name),
        ("laya.variant", variant),
        ("laya.source_format", "safetensors"),
        ("laya.layout_version", "1"),
        ("laya.dtype", "bf16"),
        ("laya.max_len", int(agent_cfg.get("max_len", 512))),
        ("laya.head_max_len", int(agent_cfg.get("head_max_len", 192))),
        ("laya.head_layers", meta["n_head_layers"]),
        ("laya.n_act", meta["n_act"]),
        ("modernbert.hidden_size", int(cfg["hidden_size"])),
        ("modernbert.num_hidden_layers", int(cfg["num_hidden_layers"])),
        ("modernbert.num_attention_heads", int(cfg["num_attention_heads"])),
        ("modernbert.intermediate_size", int(cfg["intermediate_size"])),
        ("modernbert.max_position_embeddings", int(cfg.get("max_position_embeddings", 8192))),
        ("modernbert.vocab_size", int(cfg["vocab_size"])),
        ("modernbert.layer_norm_eps", float(cfg.get("norm_eps", cfg.get("layer_norm_eps", 1e-5)))),
        ("modernbert.local_attention", int(cfg.get("local_attention", 128))),
        ("modernbert.global_attn_every_n_layers", int(cfg.get("global_attn_every_n_layers", 3))),
        ("modernbert.global_rope_theta",
         float((rp.get("full_attention") or {}).get("rope_theta", 160000.0))),
        ("modernbert.local_rope_theta",
         float((rp.get("sliding_attention") or {}).get("rope_theta", 10000.0))),
        ("modernbert.cls_token_id", int(cfg.get("cls_token_id", 50281))),
        ("modernbert.sep_token_id", int(cfg.get("sep_token_id", 50282))),
        ("modernbert.mask_token_id", int(cfg.get("mask_token_id", 50284))),
        ("modernbert.pad_token_id", int(cfg.get("pad_token_id", 50283))),
        ("general.alignment", ALIGNMENT),
    ]
    w.u64(len(kvs) + 3)  # + laya.temperature, laya.temperature_by_options, laya.layer_is_global
    for key, value in kvs:
        if key == "general.alignment":
            w.kv_uint32(key, ALIGNMENT)
        elif isinstance(value, str):
            w.kv_string(key, value)
        else:
            w.kv_uint32(key, int(value))

    w.kv_f32_array("laya.temperature", meta["temperature"])
    w.kv_f32_array("laya.temperature_by_options", meta["buckets"])
    w.kv_u32_array("laya.layer_is_global",
                   [layer_is_global(cfg, i) for i in range(meta["n_layers"])])

    # ---- tensor infos (offsets relative to tensor_data) ----
    infos = []
    offset = 0
    for native, src in entries:
        arr, dtype = tensors[src]
        nbytes = int(arr.size) * 2  # everything is BF16 in the pack
        offset = align_up(offset, ALIGNMENT)
        infos.append((native, src, arr, dtype, offset, nbytes))
        offset += nbytes
    payload_bytes = align_up(offset, ALIGNMENT)

    for native, src, arr, dtype, off, nbytes in infos:
        w.string(native)
        w.u32(len(arr.shape))
        for d in arr.shape:
            w.u64(int(d))
        w.u32(GGML_TYPE_BF16)
        w.u64(off)

    w.pad(ALIGNMENT)

    weight_bytes = 0
    for native, src, arr, dtype, off, nbytes in infos:
        w.write(to_bf16_bytes(arr, dtype))
        w.pad(ALIGNMENT)
        weight_bytes += nbytes
    w.close()
    return payload_bytes, weight_bytes


def skip_value(blob, pos, vtype):
    if vtype in (GGUF_TYPE_UINT8, GGUF_TYPE_INT8, GGUF_TYPE_BOOL):
        return pos + 1
    if vtype in (GGUF_TYPE_UINT16, GGUF_TYPE_INT16):
        return pos + 2
    if vtype in (GGUF_TYPE_UINT32, GGUF_TYPE_INT32, GGUF_TYPE_FLOAT32):
        return pos + 4
    if vtype in (GGUF_TYPE_UINT64, GGUF_TYPE_INT64, GGUF_TYPE_FLOAT64):
        return pos + 8
    if vtype == GGUF_TYPE_STRING:
        n = struct.unpack_from("<Q", blob, pos)[0]
        return pos + 8 + n
    if vtype == GGUF_TYPE_ARRAY:
        atype = struct.unpack_from("<I", blob, pos)[0]
        n = struct.unpack_from("<Q", blob, pos + 4)[0]
        pos += 12
        for _ in range(n):
            pos = skip_value(blob, pos, atype)
        return pos
    raise SystemExit("unknown metadata type %d" % vtype)


def sanity_check(path, entries, tensors):
    """Reopen the GGUF and verify tensor count/names/shapes against the manifest."""
    with open(path, "rb") as f:
        blob = f.read()

    magic, version = struct.unpack_from("<II", blob, 0)
    n_tensors, n_kv = struct.unpack_from("<QQ", blob, 8)
    assert magic == GGUF_MAGIC, "bad magic"
    assert version == GGUF_VERSION, "bad version"
    assert n_tensors == len(entries), "tensor count %d != %d" % (n_tensors, len(entries))

    pos = 24
    for _ in range(n_kv):
        klen = struct.unpack_from("<Q", blob, pos)[0]
        pos += 8 + klen
        vtype = struct.unpack_from("<I", blob, pos)[0]
        pos += 4
        pos = skip_value(blob, pos, vtype)

    found = {}
    for _ in range(n_tensors):
        nlen = struct.unpack_from("<Q", blob, pos)[0]
        pos += 8
        name = blob[pos:pos + nlen].decode("utf-8")
        pos += nlen
        nd = struct.unpack_from("<I", blob, pos)[0]
        pos += 4
        dims = struct.unpack_from("<%dQ" % nd, blob, pos)
        pos += 8 * nd
        ttype = struct.unpack_from("<I", blob, pos)[0]
        pos += 4
        pos += 8  # offset
        assert ttype == GGML_TYPE_BF16, "%s: unexpected type %d" % (name, ttype)
        found[name] = tuple(int(d) for d in dims)

    problems = []
    for native, src in entries:
        shape = tuple(int(d) for d in tensors[src][0].shape)
        if native not in found:
            problems.append("%s missing" % native)
        elif found[native] != shape:
            problems.append("%s shape %s != %s" % (native, found[native], shape))
    if len(found) != len(entries):
        problems.append("duplicate/extra tensor names")
    if problems:
        raise SystemExit("GGUF sanity check failed: %s" % problems[:5])
    print("[sanity] %d tensors verified (names + shapes), %d metadata keys"
          % (n_tensors, n_kv))


def main():
    ap = argparse.ArgumentParser(description="Laya checkpoint -> GGUF v3 converter")
    ap.add_argument("--model", default="convaiinnovations/laya",
                    help="HF repo id or local checkpoint directory")
    ap.add_argument("--output", required=True, help="output .gguf path")
    ap.add_argument("--subfolder", default=None,
                    help="checkpoint subfolder inside the repo (e.g. multilingual)")
    ap.add_argument("--variant", default="english",
                    help="english | multilingual | typed-decisions")
    ap.add_argument("--token", default=None, help="HF token for gated repos")
    ap.add_argument("--no-check", action="store_true", help="skip the reopen sanity check")
    args = ap.parse_args()

    model_dir = resolve_checkpoint(args.model, args.subfolder, args.token)
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        agent_cfg = json.load(f)
    enc_dir = os.path.join(model_dir, "encoder")
    cfg_path = os.path.join(enc_dir if os.path.isdir(enc_dir) else "", "config.json")
    if not os.path.exists(cfg_path):
        raise SystemExit("encoder config.json not found under %r" % model_dir)
    with open(cfg_path) as f:
        cfg = json.load(f)

    tensors = load_tensors(model_dir)
    entries, meta = build_manifest(tensors, cfg, agent_cfg)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    payload_bytes, weight_bytes = write_gguf(args.output, entries, tensors, cfg, agent_cfg,
                                             meta, args.variant,
                                             "%s/%s" % (args.model, args.subfolder or ""))
    dtypes = {}
    for _, src in entries:
        d = tensors[src][1]
        dtypes[d] = dtypes.get(d, 0) + 1

    print("checkpoint:      %s" % model_dir)
    print("variant:         %s" % args.variant)
    print("tensor count:    %d" % len(entries))
    print("weight bytes:    %.1f MB (bf16)" % (weight_bytes / 1e6))
    print("gguf bytes:      %.1f MB" % (os.path.getsize(args.output) / 1e6))
    print("dtype summary:   %s"
          % ", ".join("%s->bf16 x%d" % (k, v) for k, v in sorted(dtypes.items())))

    if not args.no_check:
        sanity_check(args.output, entries, tensors)


if __name__ == "__main__":
    main()
