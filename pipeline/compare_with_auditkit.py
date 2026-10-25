from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import auditkit as ak
from auditkit.metrics.rag import LexicalGroundedness

from factualis.policy_library import Retriever, load_corpus
from factualis.prompts import format_context
from factualis.verifier import parse, score_sample

METRICS = ["valid_json", "summary_present", "status_correct", "no_unsupported_claim", "quotes_verbatim",
           "usable_correct", "is_json", "summary_groundedness"]
_lexical = LexicalGroundedness()


def _check(sample) -> object:
    md = sample.metadata
    if "_score" not in md:
        md["_score"] = score_sample(sample.actual_output or "", md["context"], md["gold_status"])
    return md["_score"]


def _scorer(name):
    def deco(fn):
        return ak.scorer(fn, direction=ak.Direction.MAXIMIZE, name=name)
    return deco


@_scorer("valid_json")
def valid_json(sample, output):
    return float(_check(sample).schema_valid)


@_scorer("summary_present")
def summary_present(sample, output):
    return float(_check(sample).summary_ok)


@_scorer("status_correct")
def status_correct(sample, output):
    s = _check(sample)
    return float(s.pred_status == s.gold_status)


@_scorer("no_unsupported_claim")
def no_unsupported_claim(sample, output):
    s = _check(sample)
    return float(s.schema_valid and not s.has_hallucination)


@_scorer("quotes_verbatim")
def quotes_verbatim(sample, output):
    s = _check(sample)
    return s.n_quotes_grounded / s.n_quotes if s.n_quotes else float(s.schema_valid)


@_scorer("usable_correct")
def usable_correct(sample, output):
    s = _check(sample)
    return float(s.schema_valid and s.summary_ok and s.consistent and s.pred_status == s.gold_status and not s.has_hallucination)


@_scorer("summary_groundedness")
def summary_groundedness(sample, output):
    out = parse(output or "")
    if out is None:
        return 0.0
    text = " ".join([out.summary_answer] + [c.exact_quote for c in out.verbatim_citations])
    sc = _lexical.score(sample, text)
    return float(sc.value) if hasattr(sc, "value") else 0.0


def load_samples(pred_path: Path, retriever: Retriever) -> list:
    samples = []
    for line in pred_path.open(encoding="utf-8"):
        r = json.loads(line)
        chunks = retriever.search(r["question"], 6)
        samples.append(ak.Sample(
            input=r["question"], target=r["gold_status"], id=r["id"],
            retrieval_context=[c["text"] for c in chunks], actual_output=r["raw"],
            metadata={"gold_status": r["gold_status"], "context": format_context(chunks)},
        ))
    return samples


def per_sample(run, metric: str) -> dict[str, float]:
    out = {}
    for p in run.predictions:
        for s in p.metadata.get("scores", []):
            if s.get("name") == metric and s.get("value") is not None:
                out[p.sample_id] = float(s["value"])
    return out


def sign_flip_p(a: dict, b: dict, n_resamples: int = 10000, seed: int = 13) -> tuple[float, float, int]:
    ids = sorted(a.keys() & b.keys())
    diffs = [b[i] - a[i] for i in ids]
    n = len(diffs)
    observed = sum(diffs) / n if n else 0.0
    rng = random.Random(seed)
    hits = sum(abs(sum(d if rng.random() < 0.5 else -d for d in diffs) / n) >= abs(observed) - 1e-12 for _ in range(n_resamples))
    return observed, (hits + 1) / (n_resamples + 1), n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results/final_evaluation"))
    ap.add_argument("--baseline", default="base")
    ap.add_argument("--candidates", nargs="+", default=["finetuned"])
    ap.add_argument("--corpus", type=Path, default=Path("data/policy_clauses.jsonl"))
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out_md = a.out or a.results_dir / "auditkit_comparison.md"

    retriever = Retriever(load_corpus(a.corpus))
    scorers = [valid_json, summary_present, status_correct, no_unsupported_claim, quotes_verbatim, usable_correct,
               "is_json", summary_groundedness]
    runs = {}
    for name in [a.baseline, *a.candidates]:
        path = a.results_dir / f"{name}_predictions.jsonl"
        if not path.exists():
            print(f"skip {name}: {path} not found")
            continue
        runs[name] = ak.evaluate(load_samples(path, retriever), model="precomputed", scorers=scorers, experiment_name=name)
        print(f"{name}: fingerprint {runs[name].fingerprint}")

    lines = ["# AuditKIT evaluation", "",
             "Saved outputs graded offline with AuditKIT (`model=\"precomputed\"`). Each run's sha256 fingerprint covers "
             "model, data, scorers and config. 94 hand-verified questions.", "",
             "| config | fingerprint | " + " | ".join(METRICS) + " |", "|" + "---|" * (len(METRICS) + 2)]
    means = {}
    for name, run in runs.items():
        means[name] = {}
        for m in METRICS:
            vals = list(per_sample(run, m).values())
            means[name][m] = sum(vals) / len(vals) if vals else None
        lines.append(f"| {name} | `{str(run.fingerprint)[:12]}` | " + " | ".join(
            "-" if means[name][m] is None else f"{means[name][m]:.3f}" for m in METRICS) + " |")

    base = runs.get(a.baseline)
    for cand in a.candidates:
        if base is None or cand not in runs:
            continue
        cmp = ak.compare(base, runs[cand])
        lines += ["", f"## {cand} vs {a.baseline}", "",
                  "| metric | baseline | candidate | delta | p (sign-flip, 10k) | significant at 0.05 |", "|---|---|---|---|---|---|"]
        for m in METRICS:
            bs, cs = per_sample(base, m), per_sample(runs[cand], m)
            if not bs or not cs:
                continue
            delta, p, n = sign_flip_p(bs, cs)
            lines.append(f"| {m} | {means[a.baseline][m]:.3f} | {means[cand][m]:.3f} | {delta:+.3f} | {p:.4f} | {'yes' if p < 0.05 else 'no'} |")
        try:
            lines += ["", "AuditKIT comparison summary:", "", "```", str(cmp.summary()).strip()[:3000], "```"]
        except Exception as e:
            lines += ["", f"(AuditKIT summary unavailable: {e!r})"]

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_md.with_suffix(".json")).write_text(json.dumps({"fingerprints": {k: str(v.fingerprint) for k, v in runs.items()},
                                                         "means": means}, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
