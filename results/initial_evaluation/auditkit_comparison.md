# AuditKIT evaluation

Saved outputs graded offline with AuditKIT (`model="precomputed"`). Each run's sha256 fingerprint covers model, data, scorers and config. 94 hand-verified questions.

| config | fingerprint | valid_json | summary_present | status_correct | no_unsupported_claim | quotes_verbatim | usable_correct | is_json | summary_groundedness |
|---|---|---|---|---|---|---|---|---|---|
| B_base_xgrammar | `d8804b7ffa2b` | 1.000 | 0.723 | 0.500 | 0.851 | 0.908 | 0.202 | 1.000 | 0.679 |
| D_aligned_xgrammar | `15f6da2115f2` | 0.915 | 0.000 | 0.500 | 0.670 | 0.798 | 0.000 | 0.915 | 0.904 |

## D_aligned_xgrammar vs B_base_xgrammar

| metric | baseline | candidate | delta | p (sign-flip, 10k) | significant at 0.05 |
|---|---|---|---|---|---|
| valid_json | 1.000 | 0.915 | -0.085 | 0.0095 | yes |
| summary_present | 0.723 | 0.000 | -0.723 | 0.0001 | yes |
| status_correct | 0.500 | 0.500 | +0.000 | 1.0000 | no |
| no_unsupported_claim | 0.851 | 0.670 | -0.181 | 0.0010 | yes |
| quotes_verbatim | 0.908 | 0.798 | -0.110 | 0.0142 | yes |
| usable_correct | 0.202 | 0.000 | -0.202 | 0.0001 | yes |
| is_json | 1.000 | 0.915 | -0.085 | 0.0095 | yes |
| summary_groundedness | 0.679 | 0.904 | +0.224 | 0.0001 | yes |

AuditKIT comparison summary:

```
Comparison: d8804b7ffa2b77e7 (baseline) -> 15f6da2115f23a2b (candidate)
  per-metric (baseline -> candidate | absolute delta | relative %):
    is_json: 1.0000 -> 0.9149  delta=-0.0851 (-8.5% rel)  (n=94->94, std=0.0000->0.2790)
    no_unsupported_claim: 0.8511 -> 0.6702  delta=-0.1809 (-21.2% rel)  (n=94->94, std=0.3560->0.4701)
    quotes_verbatim: 0.9078 -> 0.7979  delta=-0.1099 (-12.1% rel)  (n=94->94, std=0.2810->0.4016)
    status_correct: 0.5000 -> 0.5000  delta=+0.0000 (+0.0% rel)  (n=94->94, std=0.5000->0.5000)
    summary_groundedness: 0.6793 -> 0.9036  delta=+0.2243 (+33.0% rel)  (n=94->94, std=0.3790->0.2941)
    summary_present: 0.7234 -> 0.0000  delta=-0.7234 (-100.0% rel)  (n=94->94, std=0.4473->0.0000)
    usable_correct: 0.2021 -> 0.0000  delta=-0.2021 (-100.0% rel)  (n=94->94, std=0.4016->0.0000)
    valid_json: 1.0000 -> 0.9149  delta=-0.0851 (-8.5% rel)  (n=94->94, std=0.0000->0.2790)
  samples: 8 went correct->wrong, 0 went wrong->correct
  model: precomputed -> precomputed (no measurable size)
```
