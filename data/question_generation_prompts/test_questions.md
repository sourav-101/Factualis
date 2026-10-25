# Gemini prompt 1: gold evaluation set (`data/test_questions.jsonl`)

**Attach:** `data/question_generation_prompts/policy_context.txt`, created by `python -m pipeline.test_set_tools export`.
Attach this file and NOT the original PDFs. The quotes have to match our extracted text character for character.

**Also paste:** the questions already in `data/test_questions.jsonl`, so Gemini doesn't repeat them.

**After:** save the output as `data/generated_test_questions.jsonl`, run
`python -m pipeline.test_set_tools check data/generated_test_questions.jsonl`, fix or drop the failing rows, review
the generated `.review.md` sheet by hand, then append the good rows to `data/test_questions.jsonl`.

---

```text
You are a principal financial-compliance auditor building a HELD-OUT TEST SET for a model that must answer
ONLY from a fixed policy library and must abstain when the library is silent.

The attached file is the ENTIRE policy library of "Factualis Bank Limited" (a fictional bank). It is split into
chunks, each headed by a line of the form:
[doc_id: <doc_id> | clause: <clause>]
The library contains these doc_ids:
- fb_aml_wire_sop            (internal SOP: wire transfers & sanctions screening)
- fb_cdd_risk_sop            (internal SOP: customer due diligence & risk categorisation)
- fb_personal_dealing_policy (internal policy: employee personal dealing / insider trading code)
- pml_maintenance_of_records_rules  (Prevention of Money-laundering (Maintenance of Records) Rules, 2005)
- sebi_pit_regulations       (SEBI (Prohibition of Insider Trading) Regulations, 2015, amended up to 12 March 2025)
- rbi_kyc_master_direction   (RBI Master Direction - Know Your Customer Direction, 2016)  [delete this line if the file has no such doc_id]

IMPORTANT: the internal SOPs deliberately use thresholds that DIFFER from common international rules
(e.g. not the US FinCEN USD 10,000 rule, not a 5-year record retention). Treat ONLY the attached text as true.

Generate 70 NEW test items as JSON Lines (one JSON object per line, no markdown fences, no commentary), with this mix:
- 25 "FULLY_GROUNDED": answer is fully stated in the library. Prefer questions needing an exact amount, time
  limit, approver role, list of documents, or penalty. At least 10 must come from the two government documents.
  At least 4 must require reading an ANNEXURE or a definition clause.
- 20 "ABSTAIN_NOT_IN_POLICY": near-miss traps that sound like they belong in this library but are NOT answered
  by it (e.g. topics the SOPs explicitly exclude in their scope clauses, adjacent customer types, crypto/VDA,
  correspondent banking, NRI or minor accounts, thresholds for instruments not covered, numbers a model would
  guess from general knowledge). Do NOT make traps that are absurd or obviously off-topic.
- 15 "PARTIALLY_GROUNDED": two-part questions where part A is answered by the library and part B is not.
- 10 "REGULATORY_CONFLICT": a general rule in one clause is modified or overridden by a specific exception or a
  stricter rule in another clause or another document (e.g. general thresholds vs PEP / FATF clauses, SOP vs
  regulation, an exemption vs the rule it carves out). The question must make BOTH clauses relevant.

Every line must have exactly these keys:
{"id": "gm001", "question": "...", "gold_status": "<one of the four labels>",
 "gold_decision": "<PERMITTED | PROHIBITED | REQUIRES_ENHANCED_DUE_DILIGENCE | REQUIRES_ESCALATION | INSUFFICIENT_POLICY_BASIS>",
 "evidence": [{"doc_id": "...", "clause": "...", "exact_quote": "..."}],
 "missing": "<what the library does not answer; empty string for FULLY_GROUNDED and REGULATORY_CONFLICT>",
 "trap_type": "<for ABSTAIN only: scope_exclusion | adjacent_entity | general_knowledge_number | unsupported_instrument | invented_exception; else empty>"}

Hard rules:
1. "exact_quote" must be copied CHARACTER FOR CHARACTER from the chunk text under the cited header,
   including odd spacing and footnote numbers. 8 to 40 words. Never paraphrase, never join text from two chunks.
2. "doc_id" and "clause" must be copied exactly from the chunk header line.
3. FULLY_GROUNDED and REGULATORY_CONFLICT items need 1-3 evidence quotes (CONFLICT needs at least 2, one per side).
   PARTIALLY_GROUNDED items need evidence for part A only. ABSTAIN items must have "evidence": [].
4. Before writing an ABSTAIN item, search the whole library for the topic. If any chunk answers it, do not use it.
   The internal SOPs exclude some topics (e.g. correspondent banking, NRI accounts) that the RBI Master Direction
   DOES cover, so an SOP scope exclusion alone does not make a trap.
4b. If an internal SOP and a regulation give DIFFERENT values for the same parameter (e.g. KYC updation periods,
   record retention years), the item is REGULATORY_CONFLICT, never FULLY_GROUNDED. Cite both sides, plus the
   SOP's clause 1.4 ("stricter requirement ... shall prevail") when it decides the answer.
5. Write questions the way a busy compliance officer or branch employee would ask them, with realistic
   scenarios ("A customer opened her account 3 months ago and wants to RTGS Rs. 9 lakh..."). Vary phrasing;
   avoid copying clause wording into the question.
6. Ids gm001..gm070. Output only the 70 JSON lines.
```
