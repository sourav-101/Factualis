from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from .answer_format import ANSWERABLE, FinComplianceOutput

_QUOTE_MAP = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "})


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(_QUOTE_MAP)).strip().lower()


def parse(raw: str) -> FinComplianceOutput | None:
    try:
        return FinComplianceOutput.model_validate_json(raw)
    except (ValidationError, ValueError):
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return None
        try:
            return FinComplianceOutput.model_validate(json.loads(m.group(0)))
        except (ValidationError, ValueError):
            return None


def squash(text: str) -> str:
    return re.sub(r"\s+", "", normalize(text))


def quote_grounded(quote: str, context: str) -> bool:
    q = squash(quote)
    return len(q) >= 8 and q in squash(context)


NON_CLAIM_RE = re.compile(r"^\s*(not (specified|stated|mentioned|provided|defined|covered)|n/?a|none|unknown|unspecified)\b", re.I)


def is_claim(value: str) -> bool:
    return bool(value.strip()) and not NON_CLAIM_RE.match(value)


def value_grounded(value: str, context: str) -> bool:
    nv, nc = squash(value), squash(context)
    if nv and nv in nc:
        return True
    nums = re.findall(r"\d[\d,]*(?:\.\d+)?", value)
    ctx_nums = {n.replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", context)}
    return bool(nums) and all(n.replace(",", "") in ctx_nums for n in nums)


@dataclass
class SampleScore:
    schema_valid: bool
    pred_status: str | None
    gold_status: str
    n_quotes: int = 0
    n_quotes_grounded: int = 0
    n_values: int = 0
    n_values_grounded: int = 0
    summary_ok: bool = False
    consistent: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def unsupported_claims(self) -> int:
        return (self.n_quotes - self.n_quotes_grounded) + (self.n_values - self.n_values_grounded)

    @property
    def has_hallucination(self) -> bool:
        trap_answered = self.gold_status == "ABSTAIN_NOT_IN_POLICY" and self.pred_status in ANSWERABLE
        return trap_answered or self.unsupported_claims > 0


def score_sample(raw: str, context: str, gold_status: str) -> SampleScore:
    out = parse(raw)
    if out is None:
        return SampleScore(schema_valid=False, pred_status=None, gold_status=gold_status, errors=["schema_invalid"])
    s = SampleScore(schema_valid=True, pred_status=out.epistemic_status, gold_status=gold_status)
    s.summary_ok = len(out.summary_answer.strip()) >= 20
    if not s.summary_ok:
        s.errors.append("empty_summary")
    if out.epistemic_status == "ABSTAIN_NOT_IN_POLICY":
        s.consistent = not out.verbatim_citations and not out.extracted_rules
    else:
        s.consistent = bool(out.verbatim_citations)
    if not s.consistent:
        s.errors.append("status_content_mismatch")
    for c in out.verbatim_citations:
        s.n_quotes += 1
        if quote_grounded(c.exact_quote, context):
            s.n_quotes_grounded += 1
        else:
            s.errors.append(f"fabricated_quote: {c.exact_quote[:60]!r}")
    for r in out.extracted_rules:
        if not is_claim(r.exact_value):
            continue
        s.n_values += 1
        if value_grounded(r.exact_value, context):
            s.n_values_grounded += 1
        else:
            s.errors.append(f"unsupported_value: {r.parameter_name}={r.exact_value!r}")
    return s


def aggregate(scores: list[SampleScore]) -> dict:
    n = len(scores) or 1
    abstain_pred = [s.pred_status == "ABSTAIN_NOT_IN_POLICY" for s in scores]
    abstain_gold = [s.gold_status == "ABSTAIN_NOT_IN_POLICY" for s in scores]
    tp = sum(p and g for p, g in zip(abstain_pred, abstain_gold))
    fp = sum(p and not g for p, g in zip(abstain_pred, abstain_gold))
    fn = sum(g and not p for p, g in zip(abstain_pred, abstain_gold))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    answerable = [s for s in scores if s.gold_status != "ABSTAIN_NOT_IN_POLICY"]
    traps = [s for s in scores if s.gold_status == "ABSTAIN_NOT_IN_POLICY"]
    quotes = sum(s.n_quotes for s in scores)
    return {
        "n": len(scores),
        "schema_valid_rate": sum(s.schema_valid for s in scores) / n,
        "hallucination_rate": sum(s.has_hallucination for s in scores) / n,
        "trap_answered_rate": (sum(s.pred_status in ANSWERABLE for s in traps) / len(traps)) if traps else None,
        "over_refusal_rate": (sum(s.pred_status == "ABSTAIN_NOT_IN_POLICY" for s in answerable) / len(answerable)) if answerable else None,
        "quote_exact_match_rate": (sum(s.n_quotes_grounded for s in scores) / quotes) if quotes else None,
        "unsupported_claims_per_answer": sum(s.unsupported_claims for s in scores) / n,
        "status_accuracy": sum(s.pred_status == s.gold_status for s in scores) / n,
        "summary_present_rate": sum(s.summary_ok for s in scores) / n,
        "status_consistency_rate": sum(s.consistent for s in scores) / n,
        "usable_correct_rate": sum(s.schema_valid and s.summary_ok and s.consistent and s.pred_status == s.gold_status
                                   and not s.has_hallucination for s in scores) / n,
        "abstention_precision": prec,
        "abstention_recall": rec,
        "abstention_f1": (2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
    }
