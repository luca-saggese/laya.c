#!/usr/bin/env python3
"""Laya decision-head oracle: exact Python/PyTorch reference values for one request.

Companion to ``tools/oracle_laya.py`` (which only dumps encoder taps). This tool
dumps the *decision head* intermediates so a native C/CUDA implementation can be
diffed against them with ``json.load``.

It does not reimplement anything: it imports the original Laya package from
``_reference/laya`` and reproduces ``common.py::DecisionModel.forward`` step by
step, then verifies that the manual re-run matches ``DecisionModel.forward``.

    python3 tools/oracle_head.py --request build/req0.json \
        --out build/oracle_head.json --model models/hf-laya
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_LAYA = os.path.join(REPO_ROOT, "_reference", "laya")


def _import_laya():
    if not os.path.isdir(REFERENCE_LAYA):
        sys.exit(f"error: {REFERENCE_LAYA} is missing")
    if REFERENCE_LAYA not in sys.path:
        sys.path.insert(0, REFERENCE_LAYA)
    import laya  # noqa: E402

    return laya


def _build_batch(agent, request: Dict[str, Any]):
    """Same normalisation as tools/oracle_laya.py --mode e2e / Agent.system_one."""
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

    return ids, items, collate_items([items], agent.tok.pad_token_id)


def run(agent, request: Dict[str, Any], row: int) -> Dict[str, Any]:
    import torch

    qids, items, batch = _build_batch(agent, request)
    model = agent.model
    dev = agent.device
    use_amp = dev.type == "cuda"
    amp = {"device_type": dev.type, "dtype": agent.dtype, "enabled": use_amp}

    input_ids = batch["input_ids"].to(dev)
    attention_mask = batch["attention_mask"].to(dev)
    marker_pos = batch["marker_pos"].to(dev)
    marker_mask = batch["marker_mask"].to(dev)
    qtype = batch["qtype"].to(dev)

    # --- reference call: the exact DecisionModel.forward path -----------------
    with torch.no_grad(), torch.autocast(**amp):
        ref_logits, ref_act = model(input_ids, attention_mask, marker_pos, marker_mask, qtype)

    # --- manual reproduction of common.py::DecisionModel.forward --------------
    # Everything stays inside autocast: act_head is reached under autocast in
    # forward() too (pooled/feats are fp32 there but the Linear casts to bf16).
    with torch.no_grad(), torch.autocast(**amp):
        h = model.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        encoder_final = h.detach().float().cpu()

        h = h + model.type_emb(qtype)[:, None, :]
        after_type_embedding = h.detach().float().cpu()

        head_outputs: List[Any] = []
        if model.head is not None:
            pad = ~attention_mask.bool()
            for layer in model.head.layers:
                h = layer(h, src_key_padding_mask=pad)
                head_outputs.append(h.detach().float().cpu())

        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        marker_hidden = m.detach().float().cpu()

        raw_logits = model.scorer(m).squeeze(-1).float()
        masked_logits = raw_logits.masked_fill(~marker_mask, -1e4)

        p = torch.softmax(masked_logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = p.topk(2, -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        probabilities = p.detach().float().cpu()
        feats_cpu = feats.detach().float().cpu()
        pooled = h[:, 0].float()
        act_logits = model.act_head(torch.cat([pooled, feats], -1))

    max_abs_logits = float((ref_logits.float() - masked_logits.float()).abs().max())
    max_abs_act = float((ref_act.float() - act_logits.float()).abs().max())
    if max_abs_logits >= 1e-4 or max_abs_act >= 1e-4:
        raise SystemExit(
            "verification FAILED: manual re-run diverges (logits %.3e, act %.3e)"
            % (max_abs_logits, max_abs_act)
        )

    # --- public Agent answer (verbatim), same forward, rounded ----------------
    answers = agent.system_one(request["state"], request["questions"])

    qid = qids[row]
    out: Dict[str, Any] = {
        "request": None,  # filled by caller
        "question_id": qid,
        "qtype": int(batch["qtype"][row]),
        "input_ids": batch["input_ids"][row].tolist(),
        "attention_mask": batch["attention_mask"][row].tolist(),
        "marker_pos": batch["marker_pos"][row].tolist(),
        "marker_mask": [int(v) for v in batch["marker_mask"][row].tolist()],
        "encoder_final": encoder_final[row].tolist(),
        "after_type_embedding": after_type_embedding[row].tolist(),
        "head_layer0_output": head_outputs[0][row].tolist(),
        "head_layer1_output": head_outputs[1][row].tolist(),
        "marker_hidden": marker_hidden[row].tolist(),
        "raw_logits": raw_logits[row].float().cpu().tolist(),
        "masked_logits": masked_logits[row].float().cpu().tolist(),
        "probabilities": probabilities[row].tolist(),
        "action_features": feats_cpu[row].tolist(),
        "act_logits": act_logits[row].float().cpu().tolist(),
        "act_probability": answers["answers"][qid]["action"]["act_probability"],
        "answers": answers["answers"],
        "verification": {
            "max_abs_raw_logits_vs_forward": max_abs_logits,
            "max_abs_act_logits_vs_forward": max_abs_act,
            "amp_dtype": str(agent.dtype),
            "device": str(agent.device),
        },
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Laya decision-head oracle dump")
    ap.add_argument("--request", default="build/req0.json")
    ap.add_argument("--model", default="models/hf-laya")
    ap.add_argument("--subfolder", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="build/oracle_head.json")
    ap.add_argument("--row", type=int, default=0)
    args = ap.parse_args()

    laya = _import_laya()
    agent = laya.Agent(args.model, device=args.device, subfolder=args.subfolder)

    with open(args.request, "r", encoding="utf-8") as fh:
        request = json.load(fh)

    payload = run(agent, request, args.row)
    payload["request"] = os.path.relpath(os.path.abspath(args.request), REPO_ROOT)

    text = json.dumps(payload, separators=(",", ":"))
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(
        "wrote %s (S=%d, M=%d, max_abs logits=%.3e, act=%.3e, act_probability=%s)"
        % (
            args.out,
            len(payload["input_ids"]),
            len(payload["marker_pos"]),
            payload["verification"]["max_abs_raw_logits_vs_forward"],
            payload["verification"]["max_abs_act_logits_vs_forward"],
            payload["act_probability"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
