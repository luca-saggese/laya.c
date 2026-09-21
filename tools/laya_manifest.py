#!/usr/bin/env python3
"""Generate models/laya-bf16.manifest.json -- the authoritative runtime manifest.

The manifest is the single source of truth for the Laya English checkpoint:
source revision, hashes, model geometry, special token ids, and one row per
tensor with source dtype/shape, GGUF dtype/shape, repack flag, native consumer
and the dtype that consumer expects.

Everything is read from local artifacts only (models/hf-laya/ and the GGUF).
No network access, no re-derivation from source code.

Usage:
    python3 tools/laya_manifest.py [--hf models/hf-laya] [--gguf models/laya-bf16.gguf]
                                   [--out models/laya-bf16.manifest.json]
"""

import argparse
import hashlib
import json
import os
import struct
import sys

# ------------------------------------------------------------------ #
# GGUF reader (metadata value types are a separate enum from ggml types)
# ------------------------------------------------------------------ #

META_TYPES = {
    0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4),
    5: ("i", 4), 6: ("f", 4), 7: ("?", 1), 8: ("str", 0), 9: ("arr", 0),
    10: ("Q", 8), 11: ("q", 8), 12: ("d", 8),
}

GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 30: "BF16",
}


class _Reader:
    def __init__(self, fh):
        self.fh = fh

    def raw(self, n):
        b = self.fh.read(n)
        if len(b) != n:
            raise EOFError("short read")
        return b

    def u32(self):
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.raw(8))[0]

    def string(self):
        return self.raw(self.u64()).decode("utf-8")

    def value(self, vtype):
        kind, size = META_TYPES[vtype]
        if kind == "str":
            return self.string()
        if kind == "arr":
            etype = self.u32()
            return [self.value(etype) for _ in range(self.u64())]
        return struct.unpack("<" + kind, self.raw(size))[0]


def read_gguf(path):
    with open(path, "rb") as fh:
        r = _Reader(fh)
        magic = r.raw(4)
        if magic != b"GGUF":
            raise SystemExit("%s: not a GGUF file (magic %r)" % (path, magic))
        version = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()
        kvs = {}
        for _ in range(n_kv):
            key = r.string()
            kvs[key] = r.value(r.u32())
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            n_dims = r.u32()
            dims = [r.u64() for _ in range(n_dims)]
            gtype = r.u32()
            offset = r.u64()
            tensors.append({
                "name": name,
                "dims": dims,
                "gguf_dtype": GGML_TYPES.get(gtype, "?%d" % gtype),
                "offset": offset,
            })
        data_offset = fh.tell()
    return {
        "version": version,
        "kvs": kvs,
        "tensors": tensors,
        "data_offset": data_offset,
        "alignment": kvs.get("general.alignment", 32),
    }


# ------------------------------------------------------------------ #
# Native consumer map
# ------------------------------------------------------------------ #

# Every model tensor in this checkpoint is consumed by a BF16 kernel. The
# runtime contract is therefore: all model tensors are BF16 in the GGUF.
# Only calibration scalars stay F32, and those live in GGUF metadata, not in
# the tensor table.
CONSUMERS = [
    ("encoder.embeddings.tok_embeddings.weight", "laya_gather_rows", "BF16"),
    ("encoder.embeddings.norm.weight", "laya_layernorm", "BF16"),
    ("encoder.final_norm.weight", "laya_layernorm", "BF16"),
    ("decision.type_emb.weight", "laya_gather_rows", "BF16"),
    ("decision.scorer.0.weight", "laya_layernorm", "BF16"),
    ("decision.scorer.0.bias", "laya_layernorm", "BF16"),
    ("decision.scorer.1.weight", "laya_linear", "BF16"),
    ("decision.scorer.1.bias", "laya_linear", "BF16"),
    ("decision.scorer.3.weight", "laya_linear", "BF16"),
    ("decision.scorer.3.bias", "laya_linear", "BF16"),
    ("decision.act_head.0.weight", "laya_linear", "BF16"),
    ("decision.act_head.0.bias", "laya_linear", "BF16"),
    ("decision.act_head.2.weight", "laya_linear", "BF16"),
    ("decision.act_head.2.bias", "laya_linear", "BF16"),
]

# Suffix rules, applied when no exact name matches.
CONSUMER_SUFFIXES = [
    ("attn_norm.weight", "laya_layernorm", "BF16"),
    ("mlp_norm.weight", "laya_layernorm", "BF16"),
    ("attn.Wqkv.weight", "laya_linear", "BF16"),
    ("attn.Wo.weight", "laya_linear", "BF16"),
    ("mlp.Wi.weight", "laya_linear", "BF16"),
    ("mlp.Wo.weight", "laya_linear", "BF16"),
    ("norm1.weight", "laya_layernorm", "BF16"),
    ("norm1.bias", "laya_layernorm", "BF16"),
    ("norm2.weight", "laya_layernorm", "BF16"),
    ("norm2.bias", "laya_layernorm", "BF16"),
    ("linear1.weight", "laya_linear", "BF16"),
    ("linear1.bias", "laya_linear", "BF16"),
    ("linear2.weight", "laya_linear", "BF16"),
    ("linear2.bias", "laya_linear", "BF16"),
    ("in_proj.weight", "laya_linear", "BF16"),
    ("in_proj.bias", "laya_linear", "BF16"),
    ("out_proj.weight", "laya_linear", "BF16"),
    ("out_proj.bias", "laya_linear", "BF16"),
    ("self_attn.in_proj_weight", "laya_linear", "BF16"),
    ("self_attn.in_proj_bias", "laya_linear", "BF16"),
]

_EXACT = {name: (consumer, dtype) for name, consumer, dtype in CONSUMERS}


def consumer_of(native_name):
    if native_name in _EXACT:
        return _EXACT[native_name]
    for suffix, consumer, dtype in CONSUMER_SUFFIXES:
        if native_name.endswith(suffix):
            return consumer, dtype
    return "unbound", "?"


# ------------------------------------------------------------------ #
# Source tensor map (native GGUF name -> HF safetensors name)
# ------------------------------------------------------------------ #

def hf_name_of(native):
    """Map a native GGUF tensor name back to its HF safetensors name."""
    if native.startswith("encoder."):
        # encoder.layers.07.attn.Wqkv.weight -> encoder.layers.7.attn.Wqkv.weight
        parts = native.split(".")
        if len(parts) > 2 and parts[1] == "layers":
            parts[2] = str(int(parts[2]))
            return ".".join(parts)
        return native
    if native.startswith("decision.head."):
        # decision.head.00.linear1.weight -> head.layers.0.linear1.weight
        parts = native.split(".")
        return ".".join(["head", "layers", str(int(parts[2]))] + parts[3:])
    if native.startswith("decision."):
        # decision.scorer.0.weight -> scorer.0.weight
        return native[len("decision."):]
    return native


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_safetensors_index(path):
    """{name: (dtype, shape)} from the safetensors header, without loading data."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n).decode("utf-8"))
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        out[name] = (info["dtype"], tuple(info["shape"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="models/hf-laya")
    ap.add_argument("--gguf", default="models/laya-bf16.gguf")
    ap.add_argument("--out", default="models/laya-bf16.manifest.json")
    ap.add_argument("--revision", default=None,
                    help="source model revision (default: read from HF cache refs)")
    args = ap.parse_args()

    st_path = os.path.join(args.hf, "model.safetensors")
    enc_cfg_path = os.path.join(args.hf, "encoder", "config.json")
    agent_cfg_path = os.path.join(args.hf, "rl_agent_config.json")
    tok_cfg_path = os.path.join(args.hf, "tokenizer", "tokenizer_config.json")
    tok_path = os.path.join(args.hf, "tokenizer", "tokenizer.json")

    for p in (st_path, enc_cfg_path, agent_cfg_path, tok_cfg_path, tok_path, args.gguf):
        if not os.path.exists(p):
            raise SystemExit("missing required artifact: %s" % p)

    enc_cfg = json.load(open(enc_cfg_path))
    agent_cfg = json.load(open(agent_cfg_path))
    tok_cfg = json.load(open(tok_cfg_path))
    tok = json.load(open(tok_path))
    st_index = read_safetensors_index(st_path)
    gguf = read_gguf(args.gguf)

    revision = args.revision
    if revision is None:
        # Best effort: the HF cache stores the resolved commit in refs/main.
        cand = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(args.hf)), "..", "..", "..", "refs", "main"))
        if os.path.exists(cand):
            revision = open(cand).read().strip()
    if revision is None:
        revision = "unknown"

    # ---- special token ids, from the tokenizer itself ----
    added = {t["content"]: t["id"] for t in tok.get("added_tokens", [])}
    special_ids = {
        "cls_token_id": enc_cfg.get("cls_token_id"),
        "sep_token_id": enc_cfg.get("sep_token_id"),
        "pad_token_id": enc_cfg.get("pad_token_id"),
        "mask_token_id": enc_cfg.get("mask_token_id"),
        "bos_token_id": enc_cfg.get("bos_token_id"),
        "eos_token_id": enc_cfg.get("eos_token_id"),
        "unk_token_id": added.get("[UNK]"),
    }

    # ---- tensor rows ----
    rows = []
    for t in gguf["tensors"]:
        native = t["name"]
        hf = hf_name_of(native)
        src_dtype, src_shape = st_index.get(hf, (None, None))
        consumer, expected = consumer_of(native)
        # GGUF dims are fastest-first; reverse into torch [out, in] order.
        gguf_shape = list(reversed(t["dims"]))
        rows.append({
            "native_name": native,
            "source_name": hf,
            "source_dtype": src_dtype,
            "source_shape": list(src_shape) if src_shape else None,
            "gguf_dtype": t["gguf_dtype"],
            "gguf_shape": gguf_shape,
            "transposed": bool(src_shape and list(src_shape) != gguf_shape),
            "native_consumer": consumer,
            "expected_native_dtype": expected,
            "dtype_ok": (src_dtype is not None and t["gguf_dtype"] == expected),
        })

    missing_source = [r["native_name"] for r in rows if r["source_dtype"] is None]
    dtype_mismatch = [r["native_name"] for r in rows if not r["dtype_ok"]]
    transposed = [r["native_name"] for r in rows if r["transposed"]]

    manifest = {
        "manifest_version": 1,
        "generated_by": "tools/laya_manifest.py",
        "source": {
            "model": "convaiinnovations/laya",
            "variant": "english",
            "revision": revision,
            "safetensors": st_path,
            "safetensors_sha256": sha256_file(st_path),
            "safetensors_bytes": os.path.getsize(st_path),
            "encoder_config": enc_cfg_path,
            "rl_agent_config": agent_cfg_path,
            "tokenizer": tok_path,
            "tokenizer_config": tok_cfg_path,
        },
        "artifact": {
            "path": args.gguf,
            "sha256": sha256_file(args.gguf),
            "bytes": os.path.getsize(args.gguf),
            "gguf_version": gguf["version"],
            "alignment": gguf["alignment"],
            "data_offset": gguf["data_offset"],
            "tensor_count": len(gguf["tensors"]),
        },
        "geometry": {
            "hidden_size": enc_cfg.get("hidden_size"),
            "vocab_size": enc_cfg.get("vocab_size"),
            "layer_count": enc_cfg.get("num_hidden_layers"),
            "attention_heads": enc_cfg.get("num_attention_heads"),
            "head_dim": (enc_cfg.get("hidden_size") // enc_cfg.get("num_attention_heads")
                         if enc_cfg.get("hidden_size") and enc_cfg.get("num_attention_heads") else None),
            "intermediate_size": enc_cfg.get("intermediate_size"),
            "max_position_embeddings": enc_cfg.get("max_position_embeddings"),
            "layer_norm_eps": enc_cfg.get("layer_norm_eps"),
            "local_attention": enc_cfg.get("local_attention"),
            "sliding_window": enc_cfg.get("sliding_window"),
            "global_attn_every_n_layers": enc_cfg.get("global_attn_every_n_layers"),
            "hidden_activation": enc_cfg.get("hidden_activation"),
            "classifier_activation": enc_cfg.get("classifier_activation"),
            "layer_types": enc_cfg.get("layer_types"),
            "max_len": agent_cfg.get("max_len"),
            "head_max_len": agent_cfg.get("head_max_len"),
            "head_layers": agent_cfg.get("head_layers"),
            "n_act": len(agent_cfg.get("act_costs", {}) or {}),
        },
        "special_token_ids": special_ids,
        "tokenizer": {
            "class": tok_cfg.get("tokenizer_class"),
            "model_max_length": tok_cfg.get("model_max_length"),
            "vocab_size": len(tok["model"]["vocab"]),
            "merges": len(tok["model"]["merges"]),
            "added_tokens": len(tok.get("added_tokens", [])),
            "normalizer": (tok.get("normalizer") or {}).get("type"),
            "pre_tokenizer": (tok.get("pre_tokenizer") or {}).get("type"),
            "decoder": (tok.get("decoder") or {}).get("type"),
        },
        "dtype_contract": {
            "rule": "every model tensor consumed by a BF16 kernel is stored BF16 in the GGUF",
            "model_tensor_dtype": "BF16",
            "calibration_dtype": "F32 (GGUF metadata, not tensor table)",
        },
        "dtype_summary": {
            "BF16": sum(1 for r in rows if r["gguf_dtype"] == "BF16"),
            "F32": sum(1 for r in rows if r["gguf_dtype"] == "F32"),
            "other": sum(1 for r in rows if r["gguf_dtype"] not in ("BF16", "F32")),
        },
        "validation": {
            "tensors_without_source": missing_source,
            "dtype_mismatches": dtype_mismatch,
            "transposed_tensors": transposed,
            "ok": not missing_source and not dtype_mismatch,
        },
        "tensors": rows,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")

    print("wrote %s" % args.out)
    print("  source revision : %s" % revision)
    print("  safetensors     : %s (%d bytes)" % (manifest["source"]["safetensors_sha256"][:16],
                                                 manifest["source"]["safetensors_bytes"]))
    print("  gguf            : %s (%d bytes)" % (manifest["artifact"]["sha256"][:16],
                                                 manifest["artifact"]["bytes"]))
    print("  tensors         : %d" % len(rows))
    print("  dtype summary   : %s" % ", ".join("%s x%d" % (k, v)
                                              for k, v in sorted(manifest["dtype_summary"].items())))
    print("  transposed      : %d" % len(transposed))
    print("  dtype mismatches: %d" % len(dtype_mismatch))
    for n in dtype_mismatch:
        r = next(x for x in rows if x["native_name"] == n)
        print("    %-48s source=%-5s gguf=%-5s expected=%s (%s)"
              % (n, r["source_dtype"], r["gguf_dtype"], r["expected_native_dtype"],
                 r["native_consumer"]))
    if missing_source:
        print("  MISSING SOURCE TENSORS: %s" % ", ".join(missing_source))
    return 0 if manifest["validation"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
