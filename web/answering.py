from __future__ import annotations

import json
from pathlib import Path

from openai import OpenAI

from factualis.answer_format import response_format
from factualis.policy_library import Retriever
from factualis.prompts import build_messages
from factualis.verifier import is_claim, parse, quote_grounded, value_grounded

EXAMPLES = [
    ("Large transfer by a new customer", "A customer who opened her account two months ago wants to RTGS Rs. 9,00,000. What review is needed before release?"),
    ("Crypto transfer", "What AML threshold applies to crypto staking rewards sent to a VASP?"),
    ("Transfer to a high-risk country", "A customer wants to send USD 5,000 to a country on the FATF increased monitoring list. Can the branch just process it?"),
    ("Employee share trading", "Can an AVP in Retail buy Rs. 3 lakh of the bank's shares during the quarterly trading window closure?"),
]

STATUS = {
    "FULLY_GROUNDED": ("Answered from policy", "The policy documents answer this question.", "ok"),
    "PARTIALLY_GROUNDED": ("Partly answered", "Some parts of the question are not covered by the policy.", "warn"),
    "ABSTAIN_NOT_IN_POLICY": ("No answer in policy", "The policy documents do not cover this question.", "none"),
    "REGULATORY_CONFLICT": ("Policy rules conflict", "Two clauses give different rules for this case.", "conflict"),
}

DOC_NAMES = {
    "fb_aml_wire_sop": "Wire Transfer & Sanctions SOP",
    "fb_cdd_risk_sop": "Customer Due Diligence SOP",
    "fb_personal_dealing_policy": "Employee Personal Dealing Policy",
    "rbi_kyc_master_direction": "RBI KYC Master Direction",
    "pml_maintenance_of_records_rules": "PML (Maintenance of Records) Rules",
    "sebi_pit_regulations": "SEBI Insider Trading Regulations",
}


def doc_name(doc_id: str) -> str:
    return DOC_NAMES.get(doc_id, doc_id.replace("_", " "))


def ask(client: OpenAI, question: str, chunks: list[dict], constrained: bool) -> str:
    kwargs = {"response_format": response_format()} if constrained else {}
    r = client.chat.completions.create(model="default", messages=build_messages(question, chunks), temperature=0.0, max_tokens=1536, **kwargs)
    return r.choices[0].message.content or ""


def verify(raw: str, context: str):
    out = parse(raw)
    if out is None:
        return None, [], []
    quotes = [(c, quote_grounded(c.exact_quote, context)) for c in out.verbatim_citations]
    values = [(r, value_grounded(r.exact_value, context)) for r in out.extracted_rules if is_claim(r.exact_value)]
    return out, quotes, values


def precompute_examples(port: str, out: Path, retriever: Retriever) -> None:
    client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="EMPTY")
    cache = {}
    for _, q in EXAMPLES:
        chunks = retriever.search(q, 6)
        cache[q] = {str(c): ask(client, q, chunks, c) for c in (True, False)}
        print("cached:", q[:70], flush=True)
    out.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
