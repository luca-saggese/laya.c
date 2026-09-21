#!/usr/bin/env python3
"""
Synthetic insurance dataset generator for Laya via OpenRouter.

Pipeline per example:
  target schedule -> scenario planner -> claim writer -> independent critic
  -> repair/retry -> global dedupe -> stratified splits

The writer never sees Laya's internal target labels. Python is needed only for
synthetic-data generation; the produced JSONL is runtime-agnostic.

Quick start:
  export OPENROUTER_API_KEY='sk-or-v1-...'
  python generate_laya_claims_openrouter.py --init-config insurance.json
  python generate_laya_claims_openrouter.py --config insurance.json \
      --output-dir data/claims --count 1000
"""
from __future__ import annotations

import argparse, concurrent.futures as cf, copy, datetime as dt, hashlib, json
import math, os, random, re, sys, threading, time, unicodedata
import urllib.error, urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

VERSION = "1.0.0"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_CONFIG: Dict[str, Any] = {
  "dataset": {
    "name": "laya-insurance-auto-it-synthetic-v1",
    "schema_id": "claims-auto-it-v1",
    "language": "it", "locale": "it-IT", "jurisdiction": "Italia",
    "product": "assicurazione auto RCA / garanzie accessorie",
    "seed": 42017, "count": 1000,
    "splits": {"train": 0.82, "validation": 0.13, "demo": 0.05},
    "stratify_on": "claim_type",
    "include_questions_in_records": True,
    "keep_generation_facts": True
  },
  "openrouter": {
    "api_url": OPENROUTER_URL, "timeout_seconds": 180,
    "max_http_retries": 6, "retry_base_seconds": 2.0,
    "http_referer": "", "x_title": "Laya Synthetic Claims Generator",
    "structured_output_fallback": True
  },
  "models": {
    "planner": {"model": "openai/gpt-5.4", "temperature": 0.8, "max_tokens": 3500, "extra": {}},
    "writer": {"model": "~anthropic/claude-sonnet-latest", "temperature": 1.0, "max_tokens": 3000, "extra": {}},
    "critic": {"model": "~google/gemini-pro-latest", "temperature": 0.15, "max_tokens": 2500, "extra": {}}
  },
  "generation": {
    "workers": 3, "max_plan_attempts": 3, "max_writer_repairs": 2,
    "min_realism_score": 85, "min_label_alignment_score": 95,
    "min_information_fidelity_score": 95, "min_style_quality_score": 80,
    "reject_label_leakage": True, "near_duplicate_hamming_distance": 6,
    "replacement_rounds": 8, "request_jitter_seconds": [0.0, 0.35]
  },
  "controls": {
    "claim_type": {
      "collision": .22, "parking_damage": .14, "theft": .08,
      "attempted_theft": .06, "weather": .10, "vandalism": .08,
      "glass": .10, "animal_collision": .05, "fire": .04,
      "road_hazard": .06, "other": .07
    },
    "difficulty": {"easy": .25, "medium": .45, "hard": .25, "borderline": .05},
    "channel": {"web_form": .30, "email": .20, "call_center_transcript": .20,
                "mobile_app": .15, "agent_note": .10, "chat": .05},
    "style": {"clear_formal": .15, "normal_customer": .30, "colloquial": .15,
              "telegraphic": .10, "noisy_typos": .10, "long_with_distractors": .10,
              "indirect_implicit": .10}
  },
  "target_distributions": {
    "urgency": {"0": .66, "1": .27, "2": .07},
    "injury_reported": {"false": .88, "true": .12},
    "third_party_involved": {"false": .46, "true": .54},
    "vehicle_drivable": {"false": .24, "true": .76},
    "documents_complete": {"0": .13, "1": .44, "2": .43},
    "needs_human_review": {"false": .77, "true": .23}
  },
  "questions": {
    "claim_type": {
      "type": "choice", "instructions": "Classifica la tipologia principale del sinistro descritto.",
      "criteria": {
        "collision": "collisione o urto mentre il veicolo era in movimento",
        "parking_damage": "veicolo trovato danneggiato mentre era parcheggiato o in sosta",
        "theft": "furto riuscito del veicolo o di componenti rilevanti",
        "attempted_theft": "tentativo di furto non riuscito con eventuali danni",
        "weather": "danno causato principalmente da grandine, vento, allagamento o altro evento atmosferico",
        "vandalism": "danno intenzionale provocato da terzi senza finalità di furto",
        "glass": "rottura o danneggiamento di parabrezza, lunotto o altri cristalli come evento principale",
        "animal_collision": "urto con animale come causa principale",
        "fire": "incendio o principio d'incendio come causa principale",
        "road_hazard": "danno causato principalmente da buca, detrito, ostacolo o dissesto stradale",
        "other": "evento non riconducibile in modo affidabile alle categorie precedenti"
      }
    },
    "urgency": {"type": "score", "instructions": "Valuta la priorità operativa del caso.",
                "criteria": ["gestione ordinaria", "gestione prioritaria a breve", "attenzione umana immediata"]},
    "injury_reported": {"type": "noul", "instructions": "Nel sinistro vengono riportate lesioni o possibili lesioni a persone?"},
    "third_party_involved": {"type": "noul", "instructions": "È coinvolto almeno un terzo identificato o chiaramente presente nella dinamica?"},
    "vehicle_drivable": {"type": "noul", "instructions": "Il veicolo assicurato risulta utilizzabile e in grado di marciare?"},
    "documents_complete": {"type": "score", "instructions": "Quanto è completa la documentazione per avviare la gestione ordinaria?",
                           "criteria": ["insufficiente", "parziale", "sufficiente"]},
    "needs_human_review": {"type": "noul", "instructions": "Il caso richiede revisione umana per ambiguità, contraddizioni, severità o eccezioni?"}
  }
}

PLANNER_SYSTEM = """You design FICTIONAL insurance training cases. Create canonical facts and exact ground truth, not customer prose. Respect requested targets exactly. Make cases realistic and internally coherent for the given jurisdiction/product. Difficulty may add distractors or indirect evidence but must not make the target unknowable. Use no real people, real plates, real claim IDs, or proprietary insurer forms. Never put internal taxonomy names into customer-facing facts. Output only structured JSON."""
WRITER_SYSTEM = """You write realistic FICTIONAL insurance claim records from canonical facts. You do not receive classifier labels. Preserve all material facts and missing information; invent no material facts. Never mention labels, decision IDs, targets, prompts, datasets or synthetic generation. Match requested channel/style. Hard cases may be indirect, messy or distracting but still defensible by a careful human. Output only structured JSON."""
CRITIC_SYSTEM = """You are an independent strict auditor of a synthetic insurance dataset. Verify that the GENERATED STATE ITSELF supports every intended target, not merely the hidden canonical facts. Reject unsupported labels, invented/omitted material facts, label leakage, implausible cases and mechanically templated prose. Ambiguity may make a case difficult but not unknowable. Return concise repair instructions. Output only structured JSON."""


def canonical(obj): return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
def pretty(obj): return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)
def now(): return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
def stable_seed(seed, *parts):
    h = hashlib.sha256("|".join(map(str, (seed,)+parts)).encode()).digest()
    return int.from_bytes(h[:4], "big") & 0x7fffffff

def deep_merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, Mapping) and isinstance(out.get(k), Mapping) else copy.deepcopy(v)
    return out

def load_config(path):
    if not path: return copy.deepcopy(DEFAULT_CONFIG)
    text = Path(path).read_text(encoding="utf-8")
    if str(path).lower().endswith((".yaml", ".yml")):
        try: import yaml
        except ImportError: raise SystemExit("YAML requires PyYAML; use JSON or pip install pyyaml")
        user = yaml.safe_load(text) or {}
    else: user = json.loads(text)
    return deep_merge(DEFAULT_CONFIG, user)

def dump_json(path, obj): Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":"))+"\n")
def append_jsonl(path, row, lock=None):
    line = json.dumps(row, ensure_ascii=False, separators=(",", ":"))+"\n"
    if lock:
        with lock:
            with Path(path).open("a", encoding="utf-8") as f: f.write(line)
    else:
        with Path(path).open("a", encoding="utf-8") as f: f.write(line)

def weighted_schedule(weights, n, rng):
    vals = [(str(k), max(0., float(v))) for k,v in weights.items() if float(v)>0]
    total = sum(v for _,v in vals); raw = [(k, n*v/total) for k,v in vals]
    counts = {k:int(math.floor(x)) for k,x in raw}
    rem = n-sum(counts.values())
    frac = sorted([(x-math.floor(x), rng.random(), k) for k,x in raw], reverse=True)
    for _,_,k in frac[:rem]: counts[k]+=1
    out=[]
    for k,_ in vals: out += [k]*counts[k]
    rng.shuffle(out); return out

def normalize_target(v, q):
    if q["type"]=="noul": return str(v).lower()=="true" if not isinstance(v,bool) else v
    if q["type"]=="score": return int(v)
    return str(v)

def legal_values(q):
    if q["type"]=="choice": return list(q["criteria"].keys()) if isinstance(q["criteria"], Mapping) else list(q["criteria"])
    if q["type"]=="score": return list(range(len(q["criteria"])))
    return [False, True]

def value_schema(q):
    if q["type"]=="choice": return {"type":"string","enum":legal_values(q)}
    if q["type"]=="score": return {"type":"integer","minimum":0,"maximum":len(q["criteria"])-1}
    return {"type":"boolean"}

def planner_schema(questions):
    props={k:value_schema(v) for k,v in questions.items()}
    arr={"type":"array","items":{"type":"string"}}
    fact_props={k:copy.deepcopy(arr) for k in ["chronology","people_and_roles","vehicles_and_objects","damage_and_condition","injury_facts","documents_present","documents_missing","policy_context","intended_ambiguities","contradictions","irrelevant_distractors","must_be_explicit","may_be_implicit"]}
    fact_props["event_summary"]={"type":"string"}
    return {"type":"object","additionalProperties":False,"properties":{
      "ground_truth":{"type":"object","additionalProperties":False,"properties":props,"required":list(props)},
      "facts":{"type":"object","additionalProperties":False,"properties":fact_props,"required":list(fact_props)},
      "scenario_quality_notes":{"type":"string"}},"required":["ground_truth","facts","scenario_quality_notes"]}

WRITER_SCHEMA={"type":"object","additionalProperties":False,"properties":{
 "state":{"type":"object","additionalProperties":False,"properties":{
  "channel":{"type":"string"},"subject":{"type":"string"},"message":{"type":"string"},
  "claimant_statement":{"type":"string"},"agent_notes":{"type":"string"},
  "form":{"type":"object","additionalProperties":False,"properties":{
   "event_date_text":{"type":"string"},"event_location_text":{"type":"string"},
   "vehicle_status_text":{"type":"string"},"third_party_text":{"type":"string"},
   "injury_text":{"type":"string"},"police_or_authority_text":{"type":"string"},
   "attachments_text":{"type":"string"}},"required":["event_date_text","event_location_text","vehicle_status_text","third_party_text","injury_text","police_or_authority_text","attachments_text"]}},
  "required":["channel","subject","message","claimant_statement","agent_notes","form"]},
 "writer_notes":{"type":"string"}},"required":["state","writer_notes"]}

CRITIC_SCHEMA={"type":"object","additionalProperties":False,"properties":{
 "accept":{"type":"boolean"},"realism_score":{"type":"integer","minimum":0,"maximum":100},
 "label_alignment_score":{"type":"integer","minimum":0,"maximum":100},
 "information_fidelity_score":{"type":"integer","minimum":0,"maximum":100},
 "style_quality_score":{"type":"integer","minimum":0,"maximum":100},
 "label_leakage":{"type":"boolean"},"material_facts_invented":{"type":"boolean"},
 "material_facts_omitted":{"type":"boolean"},"issues":{"type":"array","items":{"type":"string"}},
 "repair_instructions":{"type":"array","items":{"type":"string"}},
 "decision_checks":{"type":"array","items":{"type":"object","additionalProperties":False,"properties":{
  "decision_id":{"type":"string"},"supported_by_state":{"type":"boolean"},"comment":{"type":"string"}},
  "required":["decision_id","supported_by_state","comment"]}}},
 "required":["accept","realism_score","label_alignment_score","information_fidelity_score","style_quality_score","label_leakage","material_facts_invented","material_facts_omitted","issues","repair_instructions","decision_checks"]}

class OpenRouter:
    def __init__(self,cfg):
        self.cfg=cfg; self.key=os.getenv("OPENROUTER_API_KEY")
        if not self.key: raise SystemExit("Set OPENROUTER_API_KEY")
    def call(self, model_cfg, system, user, name, schema, seed):
        base={"model":model_cfg["model"],"messages":[{"role":"system","content":system},{"role":"user","content":user}],"seed":seed}
        for k in ("temperature","max_tokens"):
            if model_cfg.get(k) is not None: base[k]=model_cfg[k]
        for k,v in (model_cfg.get("extra") or {}).items():
            if k not in {"model","messages","response_format"}: base[k]=v
        formats=[{"type":"json_schema","json_schema":{"name":name[:64],"strict":True,"schema":schema}}]
        if self.cfg.get("structured_output_fallback",True): formats.append({"type":"json_object"})
        last=None
        for fmt in formats:
          for attempt in range(int(self.cfg.get("max_http_retries",6))):
            payload=dict(base); payload["response_format"]=fmt
            headers={"Authorization":f"Bearer {self.key}","Content-Type":"application/json","User-Agent":f"laya-dataset/{VERSION}"}
            if self.cfg.get("http_referer"): headers["HTTP-Referer"]=self.cfg["http_referer"]
            if self.cfg.get("x_title"): headers["X-Title"]=self.cfg["x_title"]
            req=urllib.request.Request(self.cfg.get("api_url",OPENROUTER_URL),data=json.dumps(payload).encode(),headers=headers,method="POST")
            try:
                with urllib.request.urlopen(req,timeout=int(self.cfg.get("timeout_seconds",180))) as r: env=json.loads(r.read())
                msg=env["choices"][0]["message"].get("content","")
                if isinstance(msg,list): msg="\n".join(x.get("text","") if isinstance(x,dict) else str(x) for x in msg)
                msg=str(msg).strip(); msg=re.sub(r"^```(?:json)?\s*|\s*```$","",msg,flags=re.I)
                try: data=json.loads(msg)
                except json.JSONDecodeError:
                    a,b=msg.find("{"),msg.rfind("}"); data=json.loads(msg[a:b+1])
                if not isinstance(data,dict): raise ValueError("response is not object")
                return data, env.get("model"), env.get("usage") or {}
            except urllib.error.HTTPError as e:
                body=e.read().decode("utf-8","replace") if hasattr(e,"read") else ""; last=RuntimeError(f"HTTP {e.code}: {body[:1000]}")
                if e.code in (400,404,422): break
                if e.code not in (408,409,429,500,502,503,504): raise last
                ra=e.headers.get("Retry-After"); delay=float(ra) if ra and ra.replace('.','',1).isdigit() else float(self.cfg.get("retry_base_seconds",2))*(2**attempt)
                time.sleep(min(delay,60))
            except Exception as e:
                last=e; time.sleep(min(float(self.cfg.get("retry_base_seconds",2))*(2**attempt),60))
        raise RuntimeError(f"OpenRouter failed for {model_cfg['model']}: {last}")

def build_blueprints(cfg,n):
    rng=random.Random(int(cfg["dataset"]["seed"])); qs=cfg["questions"]
    schedules={k:weighted_schedule(cfg["controls"][k],n,rng) for k in ("difficulty","channel","style")}
    primary=cfg["dataset"]["stratify_on"]
    if primary in cfg["controls"]: schedules[primary]=weighted_schedule(cfg["controls"][primary],n,rng)
    for qid,dist in cfg.get("target_distributions",{}).items():
        if qid in qs: schedules[qid]=[normalize_target(x,qs[qid]) for x in weighted_schedule(dist,n,rng)]
    out=[]
    for i in range(n):
        targets={qid:schedules[qid][i] for qid in qs if qid in schedules}
        out.append({"ordinal":i,"seed":stable_seed(cfg["dataset"]["seed"],"example",i),
                    "controls":{k:schedules[k][i] for k in ("difficulty","channel","style")},"targets":targets})
    rng.shuffle(out); return out

def validate_gt(gt,requested,questions):
    issues=[]
    if set(gt)!=set(questions): issues.append("ground_truth keys != questions")
    for qid,q in questions.items():
        if qid in gt and gt[qid] not in legal_values(q): issues.append(f"{qid}: illegal {gt[qid]!r}")
        if qid in requested and gt.get(qid)!=requested[qid]: issues.append(f"{qid}: changed requested target {requested[qid]!r} -> {gt.get(qid)!r}")
    return issues

def critic_ok(r,g):
    return bool(r.get("accept")) and int(r.get("realism_score",0))>=g["min_realism_score"] and int(r.get("label_alignment_score",0))>=g["min_label_alignment_score"] and int(r.get("information_fidelity_score",0))>=g["min_information_fidelity_score"] and int(r.get("style_quality_score",0))>=g["min_style_quality_score"] and not r.get("material_facts_invented") and not r.get("material_facts_omitted") and not (g.get("reject_label_leakage",True) and r.get("label_leakage")) and all(x.get("supported_by_state") for x in r.get("decision_checks",[]))

def plan_prompt(cfg,bp):
    d=cfg["dataset"]
    return f"""Create ONE canonical synthetic claim.\nContext: language={d['language']}, locale={d['locale']}, jurisdiction={d['jurisdiction']}, product={d['product']}.\nControls:\n{pretty(bp['controls'])}\nRequested targets:\n{pretty(bp['targets'])}\nDecision schema:\n{pretty(cfg['questions'])}\nGround truth MUST exactly match requested targets. For questions without requested targets, choose the most coherent value from the canonical facts. Return facts rich enough for another writer to realize the case without seeing labels."""

def writer_prompt(cfg,facts,controls,prior=None,repair=None):
    d=cfg["dataset"]; extra=""
    if prior is not None: extra=f"\nPrevious state:\n{pretty(prior)}\nReviewer repair instructions:\n{pretty(repair or [])}\nRewrite coherently from canonical facts."
    return f"""Write ONE fictional insurance record. Language={d['language']} locale={d['locale']} jurisdiction={d['jurisdiction']} product={d['product']}. Channel={controls['channel']} style={controls['style']} difficulty={controls['difficulty']}.\nCanonical facts:\n{pretty(facts)}{extra}\nstate.channel must equal {controls['channel']!r}. Use empty strings for inapplicable form/agent fields. Do not expose internal labels."""

def critic_prompt(cfg,gt,facts,state,controls):
    return f"""Audit one candidate.\nControls:\n{pretty(controls)}\nDecision schema:\n{pretty(cfg['questions'])}\nIntended ground truth:\n{pretty(gt)}\nCanonical facts:\n{pretty(facts)}\nGenerated state:\n{pretty(state)}\nCheck every decision. supported_by_state means a careful human can defend that exact target from the state alone."""

class Generator:
    def __init__(self,cfg): self.cfg=cfg; self.orc=OpenRouter(cfg["openrouter"]); self.ps=planner_schema(cfg["questions"])
    def generate(self,bp):
        c=self.cfg; g=c["generation"]; m=c["models"]; rng=random.Random(bp["seed"]); rejected=[]
        def jitter():
            lo,hi=g.get("request_jitter_seconds",[0,0]); time.sleep(rng.uniform(float(lo),float(hi))) if float(hi)>0 else None
        for pa in range(int(g["max_plan_attempts"])):
            jitter(); plan,pm,_=self.orc.call(m["planner"],PLANNER_SYSTEM,plan_prompt(c,bp),"laya_claim_plan",self.ps,stable_seed(bp["seed"],"plan",pa))
            gt,facts=plan.get("ground_truth",{}),plan.get("facts",{})
            issues=validate_gt(gt,bp["targets"],c["questions"])
            if issues: rejected.append({"stage":"planner","issues":issues}); continue
            prior=repair=None
            for wa in range(int(g["max_writer_repairs"])+1):
                jitter(); wr,wm,_=self.orc.call(m["writer"],WRITER_SYSTEM,writer_prompt(c,facts,bp["controls"],prior,repair),"laya_claim_record",WRITER_SCHEMA,stable_seed(bp["seed"],"writer",pa,wa)); state=wr.get("state",{})
                jitter(); rev,cm,_=self.orc.call(m["critic"],CRITIC_SYSTEM,critic_prompt(c,gt,facts,state,bp["controls"]),"laya_claim_review",CRITIC_SCHEMA,stable_seed(bp["seed"],"critic",pa,wa))
                if critic_ok(rev,g):
                    rid=hashlib.sha256(f"{c['dataset']['name']}|{bp['seed']}|{canonical(state)}".encode()).hexdigest()[:20]
                    rec={"id":f"syn-{rid}","schema_id":c["dataset"]["schema_id"],"source":"synthetic_openrouter","state":state,"targets":gt,
                         "metadata":{"synthetic":True,"controls":bp["controls"],"seed":bp["seed"],"planner_model":pm or m["planner"]["model"],"writer_model":wm or m["writer"]["model"],"critic_model":cm or m["critic"]["model"],"critic":rev,"generated_at":now(),"generator_version":VERSION}}
                    if c["dataset"].get("include_questions_in_records",True): rec["questions"]=c["questions"]
                    if c["dataset"].get("keep_generation_facts",True): rec["generation"]={"canonical_facts":facts,"scenario_quality_notes":plan.get("scenario_quality_notes","")}
                    return rec,rejected
                rejected.append({"stage":"critic","review":rev}); prior=state; repair=rev.get("repair_instructions",[])
        raise RuntimeError("quality gate exhausted")

WORD_RE=re.compile(r"\w+",re.UNICODE)
def state_text(x):
    if isinstance(x,str): return x
    if isinstance(x,Mapping): return " ".join(state_text(v) for v in x.values())
    if isinstance(x,Sequence) and not isinstance(x,(str,bytes)): return " ".join(state_text(v) for v in x)
    return ""
def simhash(text):
    toks=WORD_RE.findall(unicodedata.normalize("NFKC",text).lower()); shingles=toks if len(toks)<3 else [" ".join(toks[i:i+3]) for i in range(len(toks)-2)]; vec=[0]*64
    for s in shingles:
        h=int.from_bytes(hashlib.blake2b(s.encode(),digest_size=8).digest(),"big")
        for b in range(64): vec[b]+=1 if (h>>b)&1 else -1
    return sum((1<<b) for b,v in enumerate(vec) if v>=0)
class LSH:
    def __init__(self,d): self.d=d; self.h=[]; self.b=defaultdict(list)
    def keys(self,h):
        for i in range(4): yield (i,(h>>(16*i))&0xffff)
    def near(self,h):
        cand=set(); [cand.update(self.b[k]) for k in self.keys(h)]
        for i in cand:
            if (h^self.h[i]).bit_count()<=self.d: return i
        return None
    def add(self,h):
        i=len(self.h); self.h.append(h)
        for k in self.keys(h): self.b[k].append(i)

def dedupe(rows,maxdist):
    out=[]; dup=[]; exact=set(); lsh=LSH(maxdist)
    for r in rows:
        text=" ".join(WORD_RE.findall(unicodedata.normalize("NFKC",state_text(r["state"])).lower())); dg=hashlib.sha256(text.encode()).hexdigest(); sh=simhash(text)
        if dg in exact: dup.append((r,"exact")); continue
        j=lsh.near(sh)
        if j is not None: dup.append((r,f"near:{out[j]['id']}")); continue
        exact.add(dg); lsh.add(sh); out.append(r)
    return out,dup

def split_rows(rows,cfg):
    sc=cfg["dataset"]["splits"]; names=list(sc); ws=[float(sc[n]) for n in names]; s=sum(ws); ws=[x/s for x in ws]; key=cfg["dataset"]["stratify_on"]; groups=defaultdict(list)
    for r in rows: groups[str(r["targets"].get(key))].append(r)
    rng=random.Random(stable_seed(cfg["dataset"]["seed"],"splits")); out={n:[] for n in names}
    for items in groups.values():
        rng.shuffle(items); n=len(items); raw=[n*w for w in ws]; cnt=[int(math.floor(x)) for x in raw]; rem=n-sum(cnt); order=sorted(range(len(names)),key=lambda i:(raw[i]-cnt[i],rng.random()),reverse=True)
        for i in order[:rem]: cnt[i]+=1
        p=0
        for name,c in zip(names,cnt):
            chunk=items[p:p+c]; p+=c
            for r in chunk: r["split"]=name
            out[name]+=chunk
    for x in out.values(): rng.shuffle(x)
    return out

def report(rows,cfg):
    dist={qid:dict(Counter(str(r["targets"].get(qid)) for r in rows)) for qid in cfg["questions"]}
    for k in ("difficulty","channel","style"): dist["control:"+k]=dict(Counter(str(r["metadata"]["controls"].get(k)) for r in rows))
    scores={}
    for k in ("realism_score","label_alignment_score","information_fidelity_score","style_quality_score"):
        v=[float(r["metadata"]["critic"].get(k,0)) for r in rows]; scores[k]={"min":min(v),"mean":sum(v)/len(v),"max":max(v)} if v else {}
    return {"count":len(rows),"dataset":cfg["dataset"]["name"],"schema_id":cfg["dataset"]["schema_id"],"distributions":dist,"critic_scores":scores,"note":"Synthetic benchmark only: validate production performance on held-out real customer data."}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",type=Path); ap.add_argument("--output-dir",type=Path,default=Path("data/insurance_synthetic")); ap.add_argument("--count",type=int); ap.add_argument("--workers",type=int); ap.add_argument("--init-config",type=Path); ap.add_argument("--overwrite",action="store_true"); a=ap.parse_args()
    if a.init_config: a.init_config.parent.mkdir(parents=True,exist_ok=True); dump_json(a.init_config,DEFAULT_CONFIG); print(a.init_config); return
    cfg=load_config(a.config); n=a.count or int(cfg["dataset"]["count"]); workers=a.workers or int(cfg["generation"]["workers"]); out=a.output_dir; out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()) and not a.overwrite: raise SystemExit("output-dir is not empty; use --overwrite or another directory")
    if a.overwrite:
        for p in out.iterdir():
            if p.is_file(): p.unlink()
    dump_json(out/"effective_config.json",cfg); dump_json(out/"questions.json",{"schema_id":cfg["dataset"]["schema_id"],"questions":cfg["questions"]})
    rejected=out/"rejected.work.jsonl"; lock=threading.Lock(); gen=Generator(cfg); blue=build_blueprints(cfg,n)
    def job(bp):
        try: return bp,*gen.generate(bp),None
        except Exception as e: return bp,None,[],f"{type(e).__name__}: {e}"
    rows=[]; failed=[]
    with cf.ThreadPoolExecutor(max_workers=max(1,workers)) as pool:
        futs=[pool.submit(job,b) for b in blue]
        for i,f in enumerate(cf.as_completed(futs),1):
            bp,rec,rejs,err=f.result()
            for x in rejs: append_jsonl(rejected,{"blueprint":bp,"detail":x,"at":now()},lock)
            if rec is None: failed.append(bp); append_jsonl(rejected,{"blueprint":bp,"error":err,"at":now()},lock); print(f"[{i}/{n}] reject {err}",file=sys.stderr)
            else: rows.append(rec); print(f"[{i}/{n}] {rec['id']} {rec['targets']}")
    rows,dups=dedupe(rows,int(cfg["generation"]["near_duplicate_hamming_distance"]))
    for r,why in dups: append_jsonl(rejected,{"stage":"dedupe","record_id":r["id"],"reason":why},lock)
    needs=failed+[next((b for b in blue if b["seed"]==r["metadata"]["seed"]),blue[0]) for r,_ in dups]
    roundno=0
    while len(rows)<n and roundno<int(cfg["generation"]["replacement_rounds"]):
        roundno+=1; need=n-len(rows); jobs=[]
        for i in range(need):
            b=copy.deepcopy((needs or blue)[i%len(needs or blue)]); b["seed"]=stable_seed(cfg["dataset"]["seed"],"replacement",roundno,i,b["seed"]); jobs.append(b)
        new=[]
        with cf.ThreadPoolExecutor(max_workers=max(1,workers)) as pool:
            for f in cf.as_completed([pool.submit(job,b) for b in jobs]):
                bp,rec,rejs,err=f.result()
                for x in rejs: append_jsonl(rejected,{"blueprint":bp,"detail":x,"at":now()},lock)
                if rec: new.append(rec)
                else: append_jsonl(rejected,{"blueprint":bp,"error":err,"at":now()},lock)
        rows,dups=dedupe(rows+new,int(cfg["generation"]["near_duplicate_hamming_distance"]))
    if len(rows)<n: raise SystemExit(f"Only {len(rows)}/{n} unique accepted cases; inspect {rejected}")
    rows=rows[:n]; splits=split_rows(rows,cfg); write_jsonl(out/"all.jsonl",rows)
    for name,data in splits.items():
        write_jsonl(out/f"{name}.jsonl",data)
        write_jsonl(out/f"{name}.laya.jsonl",({"id":r["id"],"schema_id":r["schema_id"],"state":r["state"],"questions":cfg["questions"],"targets":r["targets"],"split":name} for r in data))
    rep=report(rows,cfg); rep["splits"]={k:len(v) for k,v in splits.items()}; dump_json(out/"report.json",rep)
    dump_json(out/"generation_manifest.json",{"version":VERSION,"completed_at":now(),"count":len(rows),"config_sha256":hashlib.sha256(canonical(cfg).encode()).hexdigest(),"models":cfg["models"]})
    print("DONE",out); print(json.dumps(rep["splits"],indent=2))

if __name__=="__main__": main()
