# Results

Both folders hold the raw model outputs for the 94 test questions and the AuditKIT comparison between the base model and the fine-tuned (SFT + DPO) model.

| Folder | What it is |
|---|---|
| `initial_evaluation/` | The first evaluation, with the original answer format. This is the run where fine-tuning made the model worse. |
| `final_evaluation/` | The evaluation with the corrected answer format, reported in the paper. `metrics_table.md` has every metric for both models, and `*_metrics.json` has the same numbers per model. |

In each folder, `base_predictions.jsonl` and `finetuned_predictions.jsonl` have one line per question: the question, its expected status and evidence, the model's raw answer, the verifier's findings and how long the answer took. `auditkit_comparison.md` has the AuditKIT scores, run fingerprints and paired significance tests, and `auditkit_comparison.json` has the fingerprints and averages.

To grade a folder again on any machine (no GPU needed):

```bash
python -m pipeline.compare_with_auditkit --results-dir results/final_evaluation --out results/final_evaluation/regraded.md
```

The final evaluation re-grades to the same numbers and the same fingerprints.
