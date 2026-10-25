# Gemini prompt 3: RBI KYC seed questions (append to `data/training_seed_questions.jsonl`)

Prompt 02 was written before the RBI Master Direction joined the library, so its seeds skip RBI completely.
This prompt fills that gap.

**Use a NEW Gemini chat.** Attach `data/question_generation_prompts/policy_context.txt`.
**After:** save the output as `data/rbi_seed_questions.jsonl`, then run:
```
.venv\Scripts\python.exe -m pipeline.test_set_tools check data/rbi_seed_questions.jsonl --seeds --against data/test_questions.jsonl
.venv\Scripts\python.exe -m pipeline.test_set_tools merge data/rbi_seed_questions.jsonl data/training_seed_questions.jsonl
```

---

```text
You are writing TRAINING questions for a compliance assistant at "Factualis Bank Limited" (fictional). It must
answer only from the attached policy library and abstain when the library is silent. Chunks are headed
[doc_id: ... | clause: ...]. Treat only the attached text as true.

Focus ONLY on doc_id rbi_kyc_master_direction (RBI Master Direction - KYC, 2016, updated Aug 2025), and on how
it interacts with the bank's internal SOPs (fb_cdd_risk_sop, fb_aml_wire_sop).

Produce 70 questions as JSON Lines (no fences, no commentary), each exactly:
{"question": "...", "expected": "<answerable | trap | partial>", "doc_id": "rbi_kyc_master_direction"}

Mix:
- 40 "answerable": spread across the whole direction - definitions, customer acceptance policy, CDD for
  individuals / proprietorships / companies / partnerships / trusts, V-CIP, Aadhaar e-KYC limits, small accounts,
  periodic updation, PEPs, foreign students, wire transfers, correspondent banking, CKYCR, UAPA / WMD freezing,
  record keeping, FIU-IND reporting, FATCA/CRS. About 12 should be questions where RBI and a Factualis SOP give
  DIFFERENT values for the same thing (periodicity, retention, approvals) - the model must notice both.
- 20 "trap": plausible questions RBI does NOT answer (product pricing, interest rates, specific SWIFT fields,
  card limits, crypto, foreign regulator rules, invented exemptions). Search the text first; drop any it answers.
- 10 "partial": two-part questions where one part is in RBI and the other is not anywhere in the library.

Vary the voice (branch officer, relationship manager, auditor, new joiner) and include realistic customer
scenarios with amounts and dates. Do not copy clause wording verbatim. Output only the 70 JSON lines.
```
