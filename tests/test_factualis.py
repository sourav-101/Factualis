from factualis.policy_library import Retriever, split_clauses
from factualis.prompts import format_context
from factualis.answer_format import FinComplianceOutput, response_format
from factualis.verifier import aggregate, score_sample
from pipeline.build_training_data import gate, neg_drop_condition, neg_over_refusal, neg_paraphrase_quote, neg_threshold

DOC = """Master Direction on KYC
3. Definitions apply as follows for the purposes of this direction.
4.2 Permanent Account Number shall be obtained for any cash transaction of Rs. 50,000 and above.
4.3 Transactions above Rs. 10,00,000 require approval of the Principal Officer within 7 days.
9.1 Enhanced due diligence shall be applied to Politically Exposed Persons at all times.
"""

CHUNKS = split_clauses("rbi_kyc", DOC)
CTX = format_context(CHUNKS)

GOOD = FinComplianceOutput(
    epistemic_status="FULLY_GROUNDED",
    summary_answer="Transactions above Rs. 10,00,000 need Principal Officer approval within 7 days.",
    extracted_rules=[
        {"rule_category": "MONETARY_THRESHOLD", "parameter_name": "approval threshold", "exact_value": "Rs. 10,00,000",
         "currency_or_unit": "INR", "trigger_condition": "transaction above threshold"},
        {"rule_category": "APPROVAL_AUTHORITY", "parameter_name": "approver", "exact_value": "Principal Officer",
         "currency_or_unit": None, "trigger_condition": "transaction above Rs. 10,00,000"},
    ],
    verbatim_citations=[{"doc_id": "rbi_kyc", "clause_or_section": "4.3",
                         "exact_quote": "Transactions above Rs. 10,00,000 require approval of the Principal Officer within 7 days."}],
    unstated_regulatory_gaps=[],
)


def test_grammar_forces_every_field():
    s = response_format()["json_schema"]["schema"]
    assert set(s["required"]) == set(s["properties"])
    rule = s["$defs"]["ExtractedRule"]
    assert set(rule["required"]) == set(rule["properties"])


def test_chunker_finds_clauses():
    assert [c["clause"] for c in CHUNKS] == ["preamble", "3", "4.2", "4.3", "9.1"] or {"4.2", "4.3", "9.1"} <= {c["clause"] for c in CHUNKS}


def test_retriever_ranks_relevant_clause():
    top = Retriever(CHUNKS).search("politically exposed persons due diligence", k=1)
    assert top[0]["clause"] == "9.1"


def test_schema_is_xgrammar_ready():
    rf = response_format()
    assert rf["type"] == "json_schema" and "epistemic_status" in rf["json_schema"]["schema"]["properties"]


def test_good_answer_scores_clean():
    s = score_sample(GOOD.model_dump_json(), CTX, "FULLY_GROUNDED")
    assert s.schema_valid and not s.has_hallucination and s.errors == []
    assert gate(GOOD, CTX, "answerable") is None


def test_every_hard_negative_fails_the_gate():
    assert gate(neg_threshold(GOOD), CTX, "answerable") == "unsupported_value"
    assert gate(neg_paraphrase_quote(GOOD), CTX, "answerable") == "fabricated_quote"
    assert gate(neg_over_refusal(GOOD), CTX, "answerable") == "answerable_abstained"
    dropped = neg_drop_condition(GOOD)
    assert dropped.model_dump() != GOOD.model_dump()


def test_trap_answered_is_hallucination_and_metrics():
    trap = score_sample(GOOD.model_dump_json(), CTX, "ABSTAIN_NOT_IN_POLICY")
    abstain = score_sample(neg_over_refusal(GOOD).model_dump_json(), CTX, "ABSTAIN_NOT_IN_POLICY")
    junk = score_sample("Sure! The limit is $10,000.", CTX, "FULLY_GROUNDED")
    assert trap.has_hallucination and not abstain.has_hallucination and not junk.schema_valid
    m = aggregate([trap, abstain, junk])
    assert m["abstention_precision"] == 1.0 and m["abstention_recall"] == 0.5 and m["trap_answered_rate"] == 0.5


def test_parse_recovers_fenced_json():
    s = score_sample("```json\n" + GOOD.model_dump_json() + "\n```", CTX, "FULLY_GROUNDED")
    assert s.schema_valid


def test_chunker_ignores_cross_references_and_table_numbers():
    text = ("3. Maintenance of records.\n(1) Every reporting entity shall keep records as defined in\n"
            "section 2 of the Aadhaar Act for the period specified.\n10.00 25.00\n"
            "4. HEADING ONLY\n4.1 A real clause that is long enough to be kept as a chunk here.\n")
    cs = split_clauses("doc", text)
    assert [c["clause"] for c in cs] == ["3", "4.1"]
    assert cs[1]["text"].startswith("4. HEADING ONLY\n4.1")


def test_gold_check_rejects_paraphrased_quote(tmp_path):
    import json

    import pytest

    from pipeline.test_set_tools import check

    corpus = tmp_path / "chunks.jsonl"
    corpus.write_text("\n".join(json.dumps(c) for c in CHUNKS), encoding="utf-8")
    gold = tmp_path / "gold.jsonl"
    row = {"id": "x", "question": "PAN threshold?", "gold_status": "FULLY_GROUNDED",
           "evidence": [{"doc_id": "rbi_kyc", "clause": "4.2", "exact_quote": "PAN must be collected for cash deals of Rs. 50,000 or more"}]}
    gold.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(SystemExit):
        check(gold, corpus, seeds=False, against=None)


def test_v2_summary_first_and_required_in_grammar():
    s = response_format()["json_schema"]["schema"]
    assert list(s["properties"])[0] == "summary_answer" and s["properties"]["summary_answer"]["minLength"] == 20
    assert "compliance_decision" not in s["properties"]


def test_usability_metrics_catch_v1_failure():
    import json
    v1_like = json.dumps({"epistemic_status": "FULLY_GROUNDED", "compliance_decision": "INSUFFICIENT_POLICY_BASIS", "summary_answer": "",
                          "extracted_rules": [], "verbatim_citations": GOOD.model_dump()["verbatim_citations"], "unstated_regulatory_gaps": []})
    s = score_sample(v1_like, CTX, "FULLY_GROUNDED")
    assert s.schema_valid and not s.summary_ok and "empty_summary" in s.errors
    m = aggregate([s, score_sample(GOOD.model_dump_json(), CTX, "FULLY_GROUNDED")])
    assert m["summary_present_rate"] == 0.5 and m["usable_correct_rate"] == 0.5
