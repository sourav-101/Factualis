from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from factualis.llm_client import chat, get_client
from factualis.policy_library import Retriever, load_corpus
from factualis.prompts import build_messages, format_context
from factualis.verifier import aggregate, normalize, score_sample

RESULTS = Path("output/eval")


def _retrieval_hit(g: dict, context: str) -> bool | None:
    quotes = [e["exact_quote"] for e in g.get("evidence", [])]
    return all(normalize(q) in normalize(context) for q in quotes) if quotes else None


def run(name: str, gold_path: Path, corpus: Path, constrained: bool, k: int, holdout: list[str]) -> None:
    retriever = Retriever(load_corpus(corpus))
    gold = [json.loads(l) for l in gold_path.open(encoding="utf-8") if l.strip()]
    client = get_client()
    contexts = {g["id"]: retriever.search(g["question"], k) for g in gold}

    def predict(g: dict) -> tuple[str, float]:
        t = time.time()
        raw = chat(build_messages(g["question"], contexts[g["id"]]), constrained=constrained, client=client)
        return raw, time.time() - t

    with ThreadPoolExecutor(8) as ex:
        preds = list(ex.map(predict, gold))

    ctx_text = {g["id"]: format_context(contexts[g["id"]]) for g in gold}
    scores = [score_sample(raw, ctx_text[g["id"]], g["gold_status"]) for g, (raw, _) in zip(gold, preds)]
    metrics = aggregate(scores)
    lat = sorted(t for _, t in preds)
    metrics["latency_p50_s"] = lat[len(lat) // 2]
    hit_by_item = [_retrieval_hit(g, ctx_text[g["id"]]) for g in gold]
    hits = [h for h in hit_by_item if h is not None]
    metrics["retrieval_evidence_recall"] = sum(hits) / len(hits) if hits else None
    metrics["evidence_retrieved"] = aggregate([s for s, h in zip(scores, hit_by_item) if h])
    if holdout:
        idx = [i for i, g in enumerate(gold) if any(e["doc_id"] in holdout for e in g.get("evidence", []))]
        metrics["holdout_docs"] = holdout
        metrics["holdout"] = aggregate([scores[i] for i in idx])

    try:
        import auditkit as ak

        by_q = {g["question"]: raw for g, (raw, _) in zip(gold, preds)}
        samples = [ak.Sample(input=g["question"], target=g["gold_status"]) for g in gold]
        ak_result = ak.evaluate(samples, model=lambda prompts: [by_q.get(p, "") for p in prompts])
        metrics["auditkit_fingerprint"] = getattr(ak_result, "fingerprint", None)
        metrics["auditkit_headline"] = str(getattr(ak_result, "headline", ""))
    except Exception as e:
        metrics["auditkit_error"] = repr(e)
        print(f"AuditKIT step failed ({e!r}); deterministic metrics are still saved.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(metrics, indent=2, default=str))
    with (RESULTS / f"{name}_predictions.jsonl").open("w", encoding="utf-8") as f:
        for g, (raw, t), s in zip(gold, preds, scores):
            f.write(json.dumps({**g, "raw": raw, "pred_status": s.pred_status, "errors": s.errors, "latency_s": t}, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, default=str))


def report() -> None:
    cols = ["usable_correct_rate", "summary_present_rate", "status_consistency_rate", "hallucination_rate", "trap_answered_rate", "over_refusal_rate", "quote_exact_match_rate",
            "unsupported_claims_per_answer", "abstention_f1", "status_accuracy", "schema_valid_rate",
            "over_refusal_when_retrieved", "holdout_hallucination_rate", "holdout_status_accuracy",
            "retrieval_evidence_recall", "latency_p50_s"]
    rows = [(p.stem, json.loads(p.read_text())) for p in sorted(RESULTS.glob("*.json")) if not p.stem.startswith("radix")]
    for _, m in rows:
        for key in ("hallucination_rate", "status_accuracy"):
            m[f"holdout_{key}"] = (m.get("holdout") or {}).get(key)
        m["over_refusal_when_retrieved"] = (m.get("evidence_retrieved") or {}).get("over_refusal_rate")
    fmt = lambda v: "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
    always = {"schema_valid_rate", "retrieval_evidence_recall", "latency_p50_s"}

    def cell(m: dict, c: str) -> str:
        if c not in always and (m.get("schema_valid_rate") or 0) < 0.5:
            return "-"
        return fmt(m.get(c))

    lines = ["| config | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    lines += [f"| {n} | " + " | ".join(cell(m, c) for c in cols) + " |" for n, m in rows]
    table = "\n".join(lines)
    (RESULTS / "report.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--gold", type=Path, default=Path("data/test_questions.jsonl"))
    ap.add_argument("--corpus", type=Path, default=Path("data/policy_clauses.jsonl"))
    ap.add_argument("--unconstrained", action="store_true", help="disable XGrammar (shows what the schema buys you)")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--results-dir", type=Path, default=RESULTS, help="where per-config results and report.md go")
    ap.add_argument("--holdout-doc", action="append", default=[], help="doc_id excluded from training (reported separately)")
    a = ap.parse_args()
    RESULTS = a.results_dir
    report() if a.report else run(a.name, a.gold, a.corpus, not a.unconstrained, a.k, a.holdout_doc)
