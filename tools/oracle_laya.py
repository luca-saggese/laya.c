#!/usr/bin/env python3
"""Laya Python oracle: authoritative reference for the native C/CUDA engine.

This tool does NOT reimplement Laya. It imports the original Laya Python
package (``_reference/laya``) and calls it, so the numbers it prints are the
numbers the reference implementation produces.

Two modes:

  A. end-to-end   -- full request -> ids / markers / logits / act_logits / answers
  B. diagnostic   -- dump selected intermediates for parity debugging

Examples
--------
    python3 tools/oracle_laya.py --mode e2e --request req.json
    python3 tools/oracle_laya.py --mode e2e --corpus tools/oracle_requests.json
    python3 tools/oracle_laya.py --mode dump --request req.json \
        --dump embeddings,final,layer:0,head,gather

The JSON layout in mode A is:

    {
      "model": "...",
      "items": [
        {
          "input_ids": [...],
          "attention_mask": [...],
          "marker_pos": [...],
          "marker_mask": [...],
          "qtype": 0,
          "raw_logits": [...],      # per-item, before temperature
          "act_logits": [...],      # per-item, before softmax
          "answers": {...}          # the public Agent answer, rounded
        }
      ],
      "answers": {...}              # top-level answers for a single request
    }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_LAYA = os.path.join(REPO_ROOT, "_reference", "laya")


def _import_laya():
    if not os.path.isdir(REFERENCE_LAYA):
        sys.exit(
            "error: %s is missing. Clone the reference workspace first:\n"
            "  git clone --depth 1 https://github.com/luca-saggese/laya _reference/laya"
            % REFERENCE_LAYA
        )
    if REFERENCE_LAYA not in sys.path:
        sys.path.insert(0, REFERENCE_LAYA)
    import laya  # noqa: E402

    return laya


# --------------------------------------------------------------------------
# Request normalisation: the oracle accepts the documented JSON shape and
# hands it to the original Agent unchanged.
# --------------------------------------------------------------------------


def load_request(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_corpus(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        return [data]
    return data


def _select_device(requested: Optional[str]) -> Optional[str]:
    import torch

    if requested == "cpu":
        return "cpu"
    if requested is None and not torch.cuda.is_available():
        return "cpu"
    return requested


def run_e2e(agent, request: Dict[str, Any]) -> Dict[str, Any]:
    """Run one request through the original Laya code and return raw values."""
    import torch

    from laya.common import QTYPES, build_sequence, collate_items, render_options

    state = request["state"]
    questions = request["questions"]
    ids = list(questions.keys())

    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)

    items = []
    for qid in ids:
        q = agent._to_internal(questions[qid])
        seq, markers = build_sequence(agent.tok, state, q, max_len, head_max_len)
        if len(markers) != len(render_options(q)):
            raise ValueError("question %r options exceed head_max_len=%d" % (qid, head_max_len))
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})

    batch = collate_items([items], agent.tok.pad_token_id)
    dev = agent.device
    with torch.no_grad():
        logits, act = agent.model(
            batch["input_ids"].to(dev),
            batch["attention_mask"].to(dev),
            batch["marker_pos"].to(dev),
            batch["marker_mask"].to(dev),
            batch["qtype"].to(dev),
        )
    logits = logits.float().cpu()
    act_logits = act.float().cpu()

    out_items = []
    for r, qid in enumerate(ids):
        k = len(items[r]["markers"])
        out_items.append(
            {
                "question_id": qid,
                "input_ids": batch["input_ids"][r].tolist(),
                "attention_mask": batch["attention_mask"][r].tolist(),
                "marker_pos": batch["marker_pos"][r].tolist(),
                "marker_mask": batch["marker_mask"][r].tolist(),
                "qtype": int(batch["qtype"][r]),
                "raw_logits": logits[r, :k].tolist(),
                "act_logits": act_logits[r].tolist(),
            }
        )

    answers = agent.system_one(state, questions)
    for it, qid in zip(out_items, ids):
        it["answers"] = answers["answers"][qid]

    return {
        "model": answers.get("model", "laya-rl-agent"),
        "usage": answers.get("usage"),
        "items": out_items,
        "answers": answers["answers"],
    }


def run_dump(agent, request: Dict[str, Any], dumps: List[str]) -> Dict[str, Any]:
    """Dump selected intermediates. Only used while debugging parity failures."""
    import torch

    from laya.common import QTYPES, build_sequence, collate_items

    state = request["state"]
    questions = request["questions"]
    ids = list(questions.keys())
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)

    items = []
    for qid in ids:
        q = agent._to_internal(questions[qid])
        seq, markers = build_sequence(agent.tok, state, q, max_len, head_max_len)
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})

    batch = collate_items([items], agent.tok.pad_token_id)
    dev = agent.device
    model = agent.model
    encoder = model.encoder

    wanted_layers = sorted(
        int(d.split(":", 1)[1]) for d in dumps if d.startswith("layer:")
    )

    captured: Dict[str, Any] = {}
    hooks = []
    if "embeddings" in dumps or wanted_layers:
        def embed_hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if "embeddings" in dumps:
                captured["embeddings"] = h.detach().float().cpu()
            for li in wanted_layers:
                captured.setdefault("layers", {})[li] = None

        hooks.append(encoder.embeddings.register_forward_hook(embed_hook))

    for li in wanted_layers:
        def layer_hook(_mod, _inp, out, li=li):
            h = out[0] if isinstance(out, tuple) else out
            captured.setdefault("layers", {})[li] = h.detach().float().cpu()

        hooks.append(encoder.layers[li].register_forward_hook(layer_hook))

    with torch.no_grad():
        logits, act = model(
            batch["input_ids"].to(dev),
            batch["attention_mask"].to(dev),
            batch["marker_pos"].to(dev),
            batch["marker_mask"].to(dev),
            batch["qtype"].to(dev),
        )
    for h in hooks:
        h.remove()

    result: Dict[str, Any] = {"items": []}
    if "embeddings" in captured:
        result["embeddings"] = captured["embeddings"].tolist()
    if captured.get("layers"):
        result["layers"] = {str(k): v.tolist() for k, v in captured["layers"].items() if v is not None}

    if "final" in dumps:
        with torch.no_grad():
            h = encoder(
                input_ids=batch["input_ids"].to(dev),
                attention_mask=batch["attention_mask"].to(dev),
            ).last_hidden_state
        result["final"] = h.detach().float().cpu().tolist()

    if "head" in dumps or "gather" in dumps:
        h = encoder(
            input_ids=batch["input_ids"].to(dev),
            attention_mask=batch["attention_mask"].to(dev),
        ).last_hidden_state
        h = h + model.type_emb(batch["qtype"].to(dev))[:, None, :]
        if model.head is not None:
            pad = ~batch["attention_mask"].to(dev).bool()
            for layer in model.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        if "head" in dumps:
            result["head"] = h.detach().float().cpu().tolist()
        idx = batch["marker_pos"].clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1)).to(dev)
        m = torch.gather(h, 1, idx)
        if "gather" in dumps:
            result["gather"] = m.detach().float().cpu().tolist()

    for r, qid in enumerate(ids):
        result["items"].append(
            {
                "question_id": qid,
                "input_ids": batch["input_ids"][r].tolist(),
                "marker_pos": batch["marker_pos"][r].tolist(),
                "raw_logits": logits[r].float().cpu().tolist(),
                "act_logits": act[r].float().cpu().tolist(),
            }
        )
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Laya Python oracle")
    ap.add_argument("--mode", choices=("e2e", "dump"), default="e2e")
    ap.add_argument("--request", help="single request JSON")
    ap.add_argument("--corpus", help="list of request JSON objects")
    ap.add_argument("--model", default="convaiinnovations/laya")
    ap.add_argument("--subfolder", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dump", default="", help="comma separated: embeddings,layer:N,final,head,gather")
    ap.add_argument("--out", default=None, help="write JSON here instead of stdout")
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args()

    if not args.request and not args.corpus:
        ap.error("--request or --corpus is required")

    laya = _import_laya()
    device = _select_device(args.device)
    agent = laya.Agent(args.model, device=device, subfolder=args.subfolder)

    requests = [load_request(args.request)] if args.request else load_corpus(args.corpus)

    if args.mode == "e2e":
        if len(requests) == 1:
            payload = run_e2e(agent, requests[0])
        else:
            payload = {"runs": [run_e2e(agent, r) for r in requests]}
    else:
        dumps = [d.strip() for d in args.dump.split(",") if d.strip()]
        if not dumps:
            ap.error("--mode dump requires --dump")
        if len(requests) == 1:
            payload = run_dump(agent, requests[0], dumps)
        else:
            payload = {"runs": [run_dump(agent, r, dumps) for r in requests]}

    text = json.dumps(payload, indent=2 if args.pretty else None, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
