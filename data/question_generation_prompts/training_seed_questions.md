# Gemini prompt 2: training seed questions (`data/training_seed_questions.jsonl`)

**Use a NEW Gemini chat**, so it can't see and copy the gold test questions.
**Attach:** `data/question_generation_prompts/policy_context.txt`.

**After:** save the output as `data/training_seed_questions.jsonl`, then run
`python -m pipeline.test_set_tools check data/training_seed_questions.jsonl --seeds --against data/test_questions.jsonl`.
Remove anything it flags as a train/test leak.

These questions are mixed with CuratorKIT's generated ones in `pipeline/build_training_data.py`. The SGLang teacher
writes the answers, and the deterministic verifier filters them. So Gemini only writes questions here, not answers.

---

```text
You are writing TRAINING questions for a compliance assistant that must answer only from the attached policy
library of "Factualis Bank Limited" (fictional) and abstain when the library is silent. Chunks are headed
[doc_id: ... | clause: ...]. Treat only the attached text as true; its thresholds intentionally differ from
general international practice.

Produce 180 questions as JSON Lines (no fences, no commentary), each exactly:
{"question": "...", "expected": "<answerable | trap | partial>", "doc_id": "<main doc the question targets, or empty for traps>"}

Mix:
- 100 "answerable": cover EVERY clause of the three fb_* documents at least once, and at least 35 questions on
  pml_maintenance_of_records_rules and sebi_pit_regulations. Include: exact amounts, time limits, approvers,
  document lists, definitions, annexure tables, multi-clause questions, and scenario questions with concrete
  customer details. (RBI Master Direction seeds come from prompt 03.) About 20 should be phrased as leading questions that contain a WRONG assumption the policy
  corrects (e.g. "Since it's under USD 20,000 we don't need HFCC approval, right?").
- 50 "trap": plausible questions the library does NOT answer (scope exclusions, adjacent customer types,
  crypto/VDA, correspondent banking, NRI/minor/trust accounts, foreign regulators, invented exceptions,
  general-knowledge numbers like USD 10,000 CTR rules). Search the library before writing each one; drop any the
  library actually answers.
- 30 "partial": two-part questions where one part is answered and the other is not.

Vary the voice (branch officer, relationship manager, treasury dealer, new joiner, auditor), length and
formality. Include some informal phrasing and a few typos. Do not reuse clause wording verbatim.
Output only the 180 JSON lines.
```
