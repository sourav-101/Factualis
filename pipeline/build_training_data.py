from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from factualis.llm_client import chat, get_client
from factualis.policy_library import Retriever, load_corpus
from factualis.prompts import build_messages, format_context
from factualis.answer_format import ABSTAIN_TEMPLATE, ANSWERABLE, FinComplianceOutput
from factualis.verifier import is_claim, parse, quote_grounded, value_grounded

rng = random.Random(13)


def _question_from_record(rec: dict) -> str | None:
    if "messages" in rec:
        users = [m["content"] for m in rec["messages"] if m.get("role") == "user"]
        q = users[-1] if users else None
    elif "conversations" in rec:
        users = [m.get("value") for m in rec["conversations"] if m.get("from") in ("human", "user")]
        q = users[-1] if users else None
    else:
        q = rec.get("instruction") or rec.get("question") or rec.get("prompt")
    if not q:
        return None
    q = re.split(r"(?i)question\s*:", q)[-1].strip().split("\n\n")[0].strip()
    return q if 10 < len(q) < 600 else None


def load_questions(path: Path | None, expected: str) -> list[dict]:
    if not path or not path.exists():
        return []
    files = [path] if path.is_file() else sorted(path.rglob("*.jsonl"))
    out = []
    for f in files:
        if f.name in ("rejected.jsonl",):
            continue
        for line in f.open(encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            q = _question_from_record(rec)
            if q:
                out.append({"question": q, "expected": rec.get("expected", expected), "doc_id": rec.get("doc_id"), "source": str(f)})
    seen, uniq = set(), []
    for r in out:
        k = r["question"].lower()
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    return uniq


def gate(out: FinComplianceOutput | None, context: str, expected: str) -> str | None:
    if out is None:
        return "schema_invalid"
    if len(out.summary_answer.strip()) < 20:
        return "empty_summary"
    if out.epistemic_status == "ABSTAIN_NOT_IN_POLICY" and (out.verbatim_citations or out.extracted_rules):
        return "abstain_with_claims"
    if expected == "trap" and out.epistemic_status != "ABSTAIN_NOT_IN_POLICY":
        return "trap_not_abstained"
    if expected == "answerable" and out.epistemic_status not in ANSWERABLE:
        return "answerable_abstained"
    if expected == "answerable" and out.epistemic_status == "PARTIALLY_GROUNDED":
        return "answerable_marked_partial"
    if expected == "partial" and out.epistemic_status != "PARTIALLY_GROUNDED":
        return "partial_mislabelled"
    if out.epistemic_status in ANSWERABLE and not out.verbatim_citations:
        return "answer_without_citation"
    for c in out.verbatim_citations:
        if not quote_grounded(c.exact_quote, context):
            return "fabricated_quote"
    for r in out.extracted_rules:
        if not value_grounded(r.exact_value, context):
            return "unsupported_value"
    return None


FOREIGN_THRESHOLDS = ["USD 10,000", "Rs. 10,00,000", "USD 3,000", "EUR 15,000", "Rs. 2,00,000"]


def neg_threshold(o: FinComplianceOutput) -> FinComplianceOutput | None:
    if not o.extracted_rules:
        return None
    r = o.model_copy(deep=True)
    i = rng.randrange(len(r.extracted_rules))
    old = r.extracted_rules[i].exact_value
    new = rng.choice([t for t in FOREIGN_THRESHOLDS if t != old])
    r.extracted_rules[i].exact_value = new
    r.summary_answer = r.summary_answer.replace(old, new) if old in r.summary_answer else r.summary_answer
    return r


def neg_drop_condition(o: FinComplianceOutput) -> FinComplianceOutput | None:
    if not o.extracted_rules:
        return None
    r = o.model_copy(deep=True)
    approvals = [x for x in r.extracted_rules if x.rule_category == "APPROVAL_AUTHORITY"]
    if approvals and len(r.extracted_rules) > 1:
        r.extracted_rules = [x for x in r.extracted_rules if x not in approvals]
    else:
        r.extracted_rules[0].trigger_condition = "Applies in all cases without further conditions."
    return r


def neg_paraphrase_quote(o: FinComplianceOutput) -> FinComplianceOutput | None:
    if not o.verbatim_citations:
        return None
    r = o.model_copy(deep=True)
    c = r.verbatim_citations[0]
    words = c.exact_quote.split()
    if len(words) < 6:
        return None
    swaps = {"shall": "must", "must": "should", "obtained": "collected", "required": "needed", "and above": "or more", "within": "in"}
    q = c.exact_quote
    for a, b in swaps.items():
        q = re.sub(rf"\b{a}\b", b, q)
    if q == c.exact_quote:
        q = "As per policy, " + " ".join(words[: len(words) // 2]) + " " + " ".join(words[len(words) // 2 + 1 :])
    c.exact_quote = q
    return r


def neg_over_refusal(_: FinComplianceOutput) -> FinComplianceOutput:
    return ABSTAIN_TEMPLATE.model_copy(deep=True)


def neg_trap_hallucination(donor: FinComplianceOutput) -> FinComplianceOutput:
    r = donor.model_copy(deep=True)
    r.epistemic_status = "FULLY_GROUNDED"
    r.unstated_regulatory_gaps = []
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/policy_clauses.jsonl")
    ap.add_argument("--qa", type=Path, default=Path("output/curator_qa"))
    ap.add_argument("--traps", type=Path, default=Path("output/curator_traps"))
    ap.add_argument("--seeds", type=Path, default=Path("data/training_seed_questions.jsonl"),
                    help="hand-written / Gemini questions, {question, expected: answerable|trap|partial}")
    ap.add_argument("--eval", type=Path, default=Path("data/test_questions.jsonl"), help="excluded from training")
    ap.add_argument("--out", type=Path, default=Path("output/fincomp"))
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--n-partial", type=int, default=120)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--holdout-doc", action="append", default=[],
                    help="doc_id kept out of training entirely (never retrieved, no seeds); eval reports it separately")
    a = ap.parse_args()

    chunks_all = load_corpus(Path(a.corpus))
    retriever = Retriever([c for c in chunks_all if c["doc_id"] not in a.holdout_doc])
    seeds = [q for q in load_questions(a.seeds, "answerable") if q.get("doc_id") not in a.holdout_doc]
    qs = load_questions(a.qa, "answerable") + load_questions(a.traps, "trap") + seeds

    ans = [q for q in qs if q["expected"] == "answerable"]
    traps = [q for q in qs if q["expected"] == "trap"]
    for _ in range(min(a.n_partial, len(qs) // 10, len(ans), len(traps))):
        x, y = rng.choice(ans), rng.choice(traps)
        qs.append({"question": f"{x['question'].rstrip('?')}? Also, {y['question'][0].lower()}{y['question'][1:]}",
                   "expected": "partial", "source": "synthetic_partial"})

    held_out = set()
    if a.eval.exists():
        held_out = {json.loads(l)["question"].lower() for l in a.eval.open(encoding="utf-8") if l.strip()}
    qs = [q for q in qs if q["question"].lower() not in held_out]
    rng.shuffle(qs)
    if a.limit:
        qs = qs[: a.limit]
    print(f"{len(qs)} questions ({sum(q['expected'] == 'answerable' for q in qs)} answerable, "
          f"{sum(q['expected'] == 'trap' for q in qs)} trap, {sum(q['expected'] == 'partial' for q in qs)} partial)")

    client = get_client()

    cache_path = a.out.parent / "label_cache.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if cache_path.exists():
        for line in cache_path.open(encoding="utf-8"):
            try:
                row = json.loads(line)
                cache[row["key"]] = row["raw"]
            except (ValueError, KeyError):
                pass
    print(f"  {len(cache)} cached teacher answers in {cache_path}")
    cache_lock = threading.Lock()
    done_count = [0]

    def label(q: dict) -> dict:
        chunks = retriever.search(q["question"], a.k)
        msgs = build_messages(q["question"], chunks)
        ctx = format_context(chunks)
        key = hashlib.sha1(json.dumps(msgs, ensure_ascii=False).encode()).hexdigest()
        raw = cache.get(key)
        for attempt in range(4 if raw is None else 0):
            try:
                raw = chat(msgs, client=client)
                with cache_lock, cache_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "raw": raw}, ensure_ascii=False) + "\n")
                break
            except Exception as e:
                if attempt == 3:
                    return {**q, "messages": msgs, "context": ctx, "raw": repr(e), "out": None, "reason": "request_failed"}
                time.sleep(10 * (attempt + 1))
        done_count[0] += 1
        if done_count[0] % 50 == 0:
            print(f"  labelled {done_count[0]}/{len(qs)}", flush=True)
        out = parse(raw)
        if out is not None:
            out.extracted_rules = [r for r in out.extracted_rules if is_claim(r.exact_value)]
        return {**q, "messages": msgs, "context": ctx, "raw": raw, "out": out, "reason": gate(out, ctx, q["expected"])}

    with ThreadPoolExecutor(a.workers) as ex:
        labelled = list(ex.map(label, qs))

    kept = [r for r in labelled if r["reason"] is None]
    rejected = [r for r in labelled if r["reason"] is not None]
    a.out.mkdir(parents=True, exist_ok=True)

    sft, dpo = [], []
    answered = [r["out"] for r in kept if r["out"].epistemic_status in ANSWERABLE]
    for r in kept:
        chosen = r["out"].model_dump_json()
        sft.append({"messages": r["messages"] + [{"role": "assistant", "content": chosen}], "expected": r["expected"]})
        if r["expected"] == "trap":
            if answered:
                negs = [("trap_hallucination", neg_trap_hallucination(rng.choice(answered)))]
            else:
                negs = []
        else:
            negs = [(fn.__name__[4:], fn(r["out"])) for fn in (neg_threshold, neg_drop_condition, neg_paraphrase_quote, neg_over_refusal)]
            negs = [(m, n) for m, n in negs if n is not None]
            negs = rng.sample(negs, min(2, len(negs)))
        for mode, neg in negs:
            dpo.append({"prompt": r["messages"], "chosen": chosen, "rejected": neg.model_dump_json(), "mode": mode})

    def write(name: str, rows: list[dict]) -> None:
        with (a.out / name).open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def split(rows: list[dict], name: str) -> None:
        cut = int(len(rows) * 0.95)
        write(f"{name}_train.jsonl", rows[:cut])
        write(f"{name}_val.jsonl", rows[cut:])

    split(sft, "sft")
    split(dpo, "dpo")
    write("rejected.jsonl", [{"question": r["question"], "expected": r["expected"], "reason": r["reason"], "raw": r["raw"]} for r in rejected])

    funnel = {
        "questions": len(labelled),
        "kept": len(kept),
        "rejected_by_reason": {k: sum(r["reason"] == k for r in rejected) for k in sorted({r["reason"] for r in rejected})},
        "sft_samples": len(sft),
        "dpo_pairs": len(dpo),
        "dpo_by_mode": {m: sum(d["mode"] == m for d in dpo) for m in sorted({d["mode"] for d in dpo})},
    }
    (a.out / "funnel.json").write_text(json.dumps(funnel, indent=2))
    print(json.dumps(funnel, indent=2))


if __name__ == "__main__":
    main()
