# Deferred work

Items deliberately postponed to keep the first end-to-end run moving.
Each entry records what is deferred, why, and how to close it.

## Tokenizer (Step 3.1)

### D1. Three rare-script pre-tokenization boundary mismatches

**Status:** deferred (2026-09-21)

`src/tokenizer/laya_tokenizer.c` matches the reference `tokenizers` pipeline on:

- curated fixture `build/tok_fixture.json`: **58/58**
- ASCII / added-token fuzz (1500 cases): **1500/1500**
- broad Unicode fuzz `/tmp/tok_fuzz2.json` (6073 cases): **6070/6073**

The 3 remaining mismatches involve Greek Extended (U+1FAE, U+1FAF) and two
mixed-symbol strings. NFC normalization is **not** the cause: the C normalizer
and Python `unicodedata.normalize("NFC", ...)` agree on these code points
(`U+1FAE -> U+1FAE`, `U+1FFB -> U+038F`, ...). The divergence is in the
ByteLevel pre-tokenizer regex boundary handling for these inputs.

**Impact on Laya:** none observed. The Laya request corpus
(`tools/oracle_requests.json`) and the sequence templates use plain Latin text,
digits, `[MASK]`/`[SEP]`/`[CLS]` and the option renderings. Token ids for the
oracle corpus already match.

**How to close:** re-run the prefix bisection
(`python3 /tmp/bisect2.py`, kept out of the repo) to obtain the minimal failing
prefix, then compare `laya_pretokenize` span choices against
`Tokenizer.pre_tokenizer.pre_tokenize_str` for that prefix.

### D2. Tokenizer tables are generated, not vendored

`src/tokenizer/laya_tokenizer_tables.h` (3.6 MB) is produced by
`tools/laya_gen_c_tables.py` from `tokenizer/tokenizer.json`. Regenerate with:

```bash
python3 tools/laya_gen_c_tables.py \
  --tokenizer <hf-snapshot>/tokenizer \
  --out src/tokenizer/laya_tokenizer_tables.h
```

The generated header is committed because the engine must build without a
Python/HF dependency at compile time.
