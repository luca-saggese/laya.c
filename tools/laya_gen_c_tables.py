#!/usr/bin/env python3
"""Generate src/tokenizer/laya_tokenizer_tables.h from the Laya tokenizer files.

Reuses the o1.c table-generation strategy (tools/_gen_c_tables.py): emit the
static tables the C tokenizer binary-searches at runtime.

Differences from o1.c's generator, all forced by the Laya tokenizer:

  * the vocabulary is mmBERT's byte-level BPE (50280 base entries, 50009
    merges) plus 116 added tokens, not Qwen2's 151643/151387;
  * the pre-tokenizer regex is the transformers ByteLevel pattern
    (no apostrophe alternatives - those come from the BPE merges);
  * the normalizer is NFC, so the canonical decomposition / combining-class /
    canonical-composition tables are embedded too;
  * `\\p{L}` / `\\p{N}` classes are expanded into code-point ranges.

Usage:
    python3 tools/laya_gen_c_tables.py --tokenizer DIR --out FILE
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# transformers ByteLevel pre-tokenizer regex (see
# transformers/models/gpt2/tokenization_gpt2.py). The ByteLevel component of
# the fast tokenizer uses exactly this pattern.
# ---------------------------------------------------------------------------
PRETOK_PATTERN = (
    r"""'s|'t|'re|'ve|'m|'ll|'d"""
    r"""| ?\p{L}+"""
    r"""| ?\p{N}+"""
    r"""| ?[^\s\p{L}\p{N}]+"""
    r"""|\s+(?!\S)"""
    r"""|\s+"""
)

GPT2_SPACE = "Ġ"


def byte_to_unicode() -> dict:
    """GPT-2 byte <-> printable unicode mapping."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def esc_c(s: str) -> str:
    out = '"'
    for byte in s.encode("utf-8"):
        if byte == 0x22:
            out += '\\"'
        elif byte == 0x5C:
            out += "\\\\"
        elif 0x20 <= byte < 0x7F:
            out += chr(byte)
        else:
            out += "\\%03o" % byte
    out += '"'
    return out


# ---------------------------------------------------------------------------
# Unicode tables (NFC + \p{L} / \p{N})
# ---------------------------------------------------------------------------


def ranges(pred):
    out = []
    start = None
    for cp in range(0x110000):
        ok = not (0xD800 <= cp <= 0xDFFF) and pred(cp)
        if ok and start is None:
            start = cp
        elif not ok and start is not None:
            out.append((start, cp - 1))
            start = None
    if start is not None:
        out.append((start, 0x10FFFF))
    return out


def build_unicode_tables():
    letters = ranges(lambda cp: unicodedata.category(chr(cp)).startswith("L"))
    numbers = ranges(lambda cp: unicodedata.category(chr(cp)).startswith("N"))

    # Unicode White_Space property (matches Rust regex `\s`); Python's
    # str.isspace() additionally (and wrongly) reports U+001C..U+001F.
    ws = []
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        if chr(cp).isspace() and not (0x1C <= cp <= 0x1F):
            ws.append(cp)

    ccc = []
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        c = unicodedata.combining(chr(cp))
        if c:
            ccc.append((cp, c))

    decomp = []
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        d = unicodedata.decomposition(chr(cp))
        if d and not d.startswith("<"):
            parts = [int(x, 16) for x in d.split()]
            if len(parts) == 1:
                # single-part canonical decomposition: rewrite cp -> parts[0]
                decomp.append((cp, parts[0], 0))
            elif len(parts) == 2:
                decomp.append((cp, parts[0], parts[1]))

    # Canonical composition: inverse of the two-part decompositions, minus the
    # composition exclusions. Derived by testing unicodedata.normalize.
    compose = []
    for cp, a, b in decomp:
        if b and unicodedata.normalize("NFC", chr(a) + chr(b)) == chr(cp):
            compose.append((a, b, cp))
    compose.sort(key=lambda t: (t[0], t[1]))

    return letters, numbers, ws, ccc, decomp, compose


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate laya tokenizer C tables")
    ap.add_argument("--tokenizer", required=True, help="directory holding tokenizer.json")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "src/tokenizer/laya_tokenizer_tables.h"))
    args = ap.parse_args()

    tok_path = os.path.join(args.tokenizer, "tokenizer.json")
    with open(tok_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    model = data["model"]
    if model["type"] != "BPE":
        sys.exit("unsupported tokenizer model: %s" % model["type"])
    if model.get("byte_fallback"):
        sys.exit("byte_fallback tokenizers are not supported")

    vocab = model["vocab"]  # piece -> id
    merges = [tuple(m) for m in model["merges"]]
    added = sorted(data["added_tokens"], key=lambda a: a["id"])

    # Pieces that are part of the base vocab.
    pieces = sorted(vocab.items(), key=lambda kv: kv[0])  # (piece, id) by piece
    merges_sorted = sorted(merges, key=lambda m: (m[0], m[1]))
    merges_rank = {m: i for i, m in enumerate(merges)}

    b2u = byte_to_unicode()
    byte_pieces = [b2u[b] for b in range(256)]

    letters, numbers, ws, ccc, decomp, compose = build_unicode_tables()

    L = []
    L.append("/* Generated by tools/laya_gen_c_tables.py -- DO NOT EDIT BY HAND.")
    L.append("   Source: %s" % tok_path)
    L.append("   Model:  mmBERT byte-level BPE, NFC normalizer")
    L.append("   %d base vocab entries, %d merges, %d added tokens */" % (len(pieces), len(merges_sorted), len(added)))
    L.append("#ifndef LAYA_TOKENIZER_TABLES_H")
    L.append("#define LAYA_TOKENIZER_TABLES_H")
    L.append("")
    L.append("/* Byte -> BPE piece string (UTF-8), GPT-2 byte-to-unicode mapping. */")
    L.append("static const char *const laya_tok_byte_piece[256] = {")
    for b in range(256):
        L.append("    %s%s" % (esc_c(byte_pieces[b]), "," if b < 255 else ""))
    L.append("};")
    L.append("")
    L.append("/* Base vocab: (id, piece). Sorted by piece (strcmp order). */")
    L.append("static const int laya_tok_vocab_count = %d;" % len(pieces))
    L.append("static const int laya_tok_vocab_id[%d] = {" % len(pieces))
    for i, (_, tid) in enumerate(pieces):
        L.append("    %d%s" % (tid, "," if i < len(pieces) - 1 else ""))
    L.append("};")
    L.append("static const char *const laya_tok_vocab_str[%d] = {" % len(pieces))
    for i, (p, _) in enumerate(pieces):
        L.append("    %s%s" % (esc_c(p), "," if i < len(pieces) - 1 else ""))
    L.append("};")
    L.append("")
    L.append("/* Merges: (left, right, rank). Sorted by (left, right) for bsearch. */")
    L.append("static const int laya_tok_merge_count = %d;" % len(merges_sorted))
    L.append("static const char *const laya_tok_merge_left[%d] = {" % len(merges_sorted))
    for i, (a, _) in enumerate(merges_sorted):
        L.append("    %s%s" % (esc_c(a), "," if i < len(merges_sorted) - 1 else ""))
    L.append("};")
    L.append("static const char *const laya_tok_merge_right[%d] = {" % len(merges_sorted))
    for i, (_, b) in enumerate(merges_sorted):
        L.append("    %s%s" % (esc_c(b), "," if i < len(merges_sorted) - 1 else ""))
    L.append("};")
    L.append("static const int laya_tok_merge_rank[%d] = {" % len(merges_sorted))
    for i, m in enumerate(merges_sorted):
        L.append("    %d%s" % (merges_rank[m], "," if i < len(merges_sorted) - 1 else ""))
    L.append("};")
    L.append("")
    L.append("/* Added tokens (content, id): matched longest-first at each position. */")
    L.append("static const int laya_tok_added_count = %d;" % len(added))
    L.append("static const char *const laya_tok_added_str[%d] = {" % len(added))
    for i, a in enumerate(added):
        L.append("    %s%s" % (esc_c(a["content"]), "," if i < len(added) - 1 else ""))
    L.append("};")
    L.append("static const int laya_tok_added_id[%d] = {" % len(added))
    for i, a in enumerate(added):
        L.append("    %d%s" % (a["id"], "," if i < len(added) - 1 else ""))
    L.append("};")
    L.append("")
    L.append("/* lstrip flags: the token swallows the whitespace that precedes it. */")
    L.append("static const unsigned char laya_tok_added_lstrip[%d] = {" % len(added))
    for i, a in enumerate(added):
        L.append("    %d%s" % (1 if a.get("lstrip") else 0, "," if i < len(added) - 1 else ""))
    L.append("};")
    L.append("")
    L.append("/* Code-point ranges for the pre-tokenizer character classes. */")

    def emit_ranges(name, rs):
        L.append("static const int %s_count = %d;" % (name, len(rs)))
        L.append("static const unsigned %s_lo[%d] = {" % (name, len(rs)))
        L.append("    " + ",".join("%d" % lo for lo, _ in rs))
        L.append("};")
        L.append("static const unsigned %s_hi[%d] = {" % (name, len(rs)))
        L.append("    " + ",".join("%d" % hi for _, hi in rs))
        L.append("};")

    emit_ranges("laya_tok_letter_range", letters)
    emit_ranges("laya_tok_number_range", numbers)
    L.append("static const int laya_tok_ws_count = %d;" % len(ws))
    L.append("static const unsigned laya_tok_ws_cp[%d] = {" % len(ws))
    L.append("    " + ",".join("%d" % c for c in ws))
    L.append("};")
    L.append("")
    L.append("/* NFC: canonical combining class. */")
    L.append("static const int laya_tok_ccc_count = %d;" % len(ccc))
    L.append("static const unsigned laya_tok_ccc_cp[%d] = {" % len(ccc))
    L.append("    " + ",".join("%d" % cp for cp, _ in ccc))
    L.append("};")
    L.append("static const unsigned char laya_tok_ccc_val[%d] = {" % len(ccc))
    L.append("    " + ",".join("%d" % v for _, v in ccc))
    L.append("};")
    L.append("")
    L.append("/* NFC: canonical decompositions (a,0 means cp -> a). */")
    L.append("static const int laya_tok_decomp_count = %d;" % len(decomp))
    L.append("static const unsigned laya_tok_decomp_cp[%d] = {" % len(decomp))
    L.append("    " + ",".join("%d" % cp for cp, _, _ in decomp))
    L.append("};")
    L.append("static const unsigned laya_tok_decomp_a[%d] = {" % len(decomp))
    L.append("    " + ",".join("%d" % a for _, a, _ in decomp))
    L.append("};")
    L.append("static const unsigned laya_tok_decomp_b[%d] = {" % len(decomp))
    L.append("    " + ",".join("%d" % b for _, _, b in decomp))
    L.append("};")
    L.append("")
    L.append("/* NFC: canonical compositions, sorted by (a, b). */")
    L.append("static const int laya_tok_compose_count = %d;" % len(compose))
    L.append("static const unsigned laya_tok_compose_a[%d] = {" % len(compose))
    L.append("    " + ",".join("%d" % a for a, _, _ in compose))
    L.append("};")
    L.append("static const unsigned laya_tok_compose_b[%d] = {" % len(compose))
    L.append("    " + ",".join("%d" % b for _, b, _ in compose))
    L.append("};")
    L.append("static const unsigned laya_tok_compose_c[%d] = {" % len(compose))
    L.append("    " + ",".join("%d" % c for _, _, c in compose))
    L.append("};")
    L.append("")
    L.append("#endif")
    L.append("")

    text = "\n".join(L)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print("wrote %s (%.1f MB)" % (args.out, len(text) / 1e6))
    print("vocab=%d merges=%d added=%d letters=%d numbers=%d ws=%d ccc=%d decomp=%d compose=%d"
          % (len(pieces), len(merges_sorted), len(added), len(letters), len(numbers), len(ws), len(ccc), len(decomp), len(compose)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
