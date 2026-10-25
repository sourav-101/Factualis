from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

EpistemicStatus = Literal[
    "FULLY_GROUNDED",
    "PARTIALLY_GROUNDED",
    "ABSTAIN_NOT_IN_POLICY",
    "REGULATORY_CONFLICT",
]

RuleCategory = Literal[
    "MONETARY_THRESHOLD",
    "REPORTING_SLA",
    "KYC_DOCUMENT_REQ",
    "APPROVAL_AUTHORITY",
    "SANCTION_RESTRICTION",
    "PENALTY",
]


class ExtractedRule(BaseModel):
    rule_category: RuleCategory
    parameter_name: str
    exact_value: str = Field(description="Value exactly as written in the policy, e.g. 'Rs. 50,000' or '7 days'.")
    currency_or_unit: Optional[str]
    trigger_condition: str


class Citation(BaseModel):
    doc_id: str
    clause_or_section: str
    exact_quote: str = Field(description="Must be an exact substring of the provided policy context.")


class FinComplianceOutput(BaseModel):

    summary_answer: str = Field(json_schema_extra={"minLength": 20})
    epistemic_status: EpistemicStatus
    extracted_rules: List[ExtractedRule] = Field(max_length=6)
    verbatim_citations: List[Citation] = Field(max_length=6)
    unstated_regulatory_gaps: List[str] = Field(max_length=5)


ANSWERABLE = {"FULLY_GROUNDED", "PARTIALLY_GROUNDED", "REGULATORY_CONFLICT"}


def json_schema() -> dict:
    return FinComplianceOutput.model_json_schema()


def response_format() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {"name": "FinComplianceOutput", "schema": json_schema()},
    }


ABSTAIN_TEMPLATE = FinComplianceOutput(
    epistemic_status="ABSTAIN_NOT_IN_POLICY",
    summary_answer="The provided policy documents do not address this question, so no grounded answer can be given.",
    extracted_rules=[],
    verbatim_citations=[],
    unstated_regulatory_gaps=[],
)
