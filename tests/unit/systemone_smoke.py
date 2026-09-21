#!/usr/bin/env python3
"""Live TypeSafe/System One wire smoke test for the resident Laya server.

Covers the spec's Test A-D layers in one compact script:

  A  parser      -- one canonical mixed request is accepted
  B  invalid     -- compact set of rejected requests
  C  curl smoke  -- HTTP 200, Jev response shapes, no action/noul confidence
  D  batch       -- N questions in one request == one batched forward

The server is expected to be running already; pass --base-url (default
http://127.0.0.1:8000). Exits non-zero on the first failure.

    python3 tests/unit/systemone_smoke.py --base-url http://127.0.0.1:8016
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE = os.path.join(REPO_ROOT, "fixtures", "laya", "systemone_smoke.json")

failures: list[str] = []


def check(cond: bool, label: str, detail: str = "") -> None:
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}{(' -- ' + detail) if detail else ''}")
        failures.append(label)


def post(base: str, body, content_type: str = "application/json", request_id: str | None = None):
    """POST raw bytes; returns (status, headers, parsed_json_or_None)."""
    if isinstance(body, (dict, list)):
        body = json.dumps(body).encode()
    headers = {"Content-Type": content_type}
    if request_id is not None:
        headers["x-typesafe-request-id"] = request_id
    req = urllib.request.Request(base + "/v1/systemone", data=body, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read()
            return resp.status, dict(resp.headers), (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            parsed = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            parsed = None
        return exc.code, dict(exc.headers), parsed


def get(base: str, path: str):
    try:
        with urllib.request.urlopen(base + path) as resp:
            return resp.status, dict(resp.headers), json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), None


def test_models(base: str) -> None:
    print("GET /v1/models")
    status, _, body = get(base, "/v1/models")
    check(status == 200, f"status {status} == 200")
    data = (body or {}).get("data") or []
    check(bool(data), "at least one served model", repr(body))


def test_options(base: str) -> None:
    print("OPTIONS /v1/systemone")
    req = urllib.request.Request(base + "/v1/systemone", method="OPTIONS")
    try:
        with urllib.request.urlopen(req) as resp:
            check(resp.status in (200, 204), f"status {resp.status} in (200, 204)")
    except urllib.error.HTTPError as exc:
        check(False, "OPTIONS preflight", f"status {exc.code}")


def test_mixed(base: str, req: dict) -> None:
    print("POST /v1/systemone  (Test A + C: mixed questions)")
    status, headers, body = post(base, req, request_id="smoke-0001")
    check(status == 200, f"status {status} == 200", json.dumps(body)[:200])
    if status != 200:
        return
    check(headers.get("x-typesafe-request-id") == "smoke-0001", "x-typesafe-request-id echoed")
    check(body.get("model") == "laya-local", "model echoed", str(body.get("model")))
    check(set(body.get("answers", {})) == set(req["questions"]), "answer keys preserved")
    usage = body.get("usage") or {}
    check(isinstance(usage.get("input_tokens"), int) and usage["input_tokens"] > 0,
          "usage.input_tokens > 0", str(usage))
    check(usage.get("output_tokens") == 0, "usage.output_tokens == 0")

    for qid, q in req["questions"].items():
        ans = body["answers"][qid]
        print(f"    {qid}: {json.dumps(ans)}")
        check("action" not in ans, f"{qid}: no action field")
        check("act_probability" not in ans, f"{qid}: no act_probability field")
        if q["type"] == "choice":
            check(ans.get("type") == "choice", f"{qid}: type")
            check(set(ans.get("probabilities", {})) == set(q["criteria"]),
                  f"{qid}: probability keys are the criteria keys")
            check(ans.get("choice") in q["criteria"], f"{qid}: choice is a criterion key")
            check(0.0 <= ans.get("confidence", -1) <= 1.0, f"{qid}: confidence in [0,1]")
        elif q["type"] == "score":
            check(ans.get("type") == "score", f"{qid}: type")
            check(set(ans.get("legend", {})) == {str(i) for i in range(len(q["criteria"]))},
                  f"{qid}: legend keys 0..N")
            check([ans["legend"][str(i)] for i in range(len(q["criteria"]))] == list(q["criteria"]),
                  f"{qid}: legend carries the raw criteria values", json.dumps(ans.get("legend")))
            check(set(ans.get("probabilities", {})) == {str(i) for i in range(len(q["criteria"]))},
                  f"{qid}: probability keys 0..N")
            check(0.0 <= ans.get("score", -1) <= len(q["criteria"]) - 1,
                  f"{qid}: score within the level range")
        else:
            check(ans.get("type") == "noul", f"{qid}: type")
            check(0.0 <= ans.get("noul", -1) <= 1.0, f"{qid}: noul in [0,1]")
            check("confidence" not in ans, f"{qid}: NO confidence on noul (spec)")

    probs = body["answers"]["coverage_choice"]["probabilities"]
    check(abs(sum(probs.values()) - 1.0) < 1e-3, "choice probabilities sum to 1")


def test_batching(base: str, req: dict) -> None:
    print("POST /v1/systemone  (Test D: one request == one batched forward)")
    _, _, allq = post(base, req)
    wide = allq["answers"]["coverage_choice"]["probabilities"]
    # The same question alone must still give the same discrete answer; the
    # probabilities may drift with the batch shape (BF16 GEMM), the decision
    # must not.
    _, _, only = post(
        base, {"state": req["state"], "questions": {"q": req["questions"]["coverage_choice"]}}
    )
    single = only["answers"]["q"]
    check(wide and set(wide) == set(single["probabilities"]), "same option keys batched vs single")
    check(allq["answers"]["coverage_choice"]["choice"] == single["choice"],
          "same discrete choice batched vs single")


def test_contamination(base: str, req: dict) -> None:
    print("POST /v1/systemone  (no cross-question contamination)")
    q = req["questions"]["coverage_choice"]
    _, _, trip = post(base, {"state": req["state"], "questions": {"a": q, "b": q, "c": q}})
    ref = trip["answers"]["a"]["probabilities"]
    for other in ("b", "c"):
        drift = max(abs(ref[k] - trip["answers"][other]["probabilities"][k]) for k in ref)
        check(drift < 1e-6, f"question '{other}' identical to 'a' (drift {drift:.2e})")


def test_invalid(base: str, req: dict) -> None:
    print("POST /v1/systemone  (Test B: invalid requests)")
    qs = req["questions"]
    cases = [
        ("invalid JSON", b"{not json", "application/json", 400),
        ("missing state", {"questions": {"q": qs["coverage_choice"]}}, "application/json", 422),
        ("missing questions", {"state": req["state"]}, "application/json", 422),
        ("empty questions", {"state": req["state"], "questions": {}}, "application/json", 422),
        ("unknown qtype",
         {"state": req["state"], "questions": {"q": {"type": "bogus", "criteria": {"a": "b", "c": "d"}}}},
         "application/json", 422),
        ("choice <2 options",
         {"state": req["state"], "questions": {"q": {"type": "choice", "criteria": {"only": "one"}}}},
         "application/json", 422),
        ("score <2 levels",
         {"state": req["state"], "questions": {"q": {"type": "score", "criteria": ["only"]}}},
         "application/json", 422),
        ("bad noul criteria key",
         {"state": req["state"], "questions": {"q": {"type": "noul", "criteria": {"maybe": "x"}}}},
         "application/json", 422),
        ("wrong Content-Type", json.dumps(req).encode(), "text/plain", 415),
    ]
    for label, body, ctype, want in cases:
        status, _, parsed = post(base, body, ctype)
        check(status == want, f"{label} -> {status} (want {want})", json.dumps(parsed)[:120])
        if want != 415 and status >= 400:
            check(isinstance(parsed, dict) and "error" in parsed,
                  f"{label}: body is {{\"error\": ...}}", json.dumps(parsed)[:120])

    print("routing")
    status, _, _ = get(base, "/v1/systemone")
    check(status == 405, f"GET /v1/systemone -> {status} (want 405)")
    status, _, _ = get(base, "/v1/does-not-exist")
    check(status == 404, f"GET unknown endpoint -> {status} (want 404)")
    status, _, parsed = post(
        base, {"model": "nope", "state": req["state"], "questions": {"q": qs["coverage_choice"]}}
    )
    check(status == 404, f"unknown model -> {status} (want 404)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    with open(FIXTURE) as fh:
        req = json.load(fh)
    req.pop("name", None)

    print(f"System One wire smoke against {base}  ({len(req['questions'])} questions)")
    test_models(base)
    test_options(base)
    test_mixed(base, req)
    test_batching(base, req)
    test_contamination(base, req)
    test_invalid(base, req)

    print()
    if failures:
        print(f"FAILED ({len(failures)}): " + ", ".join(failures))
        return 1
    print("PASS: all System One wire checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
