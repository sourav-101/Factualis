# AuditKIT evaluation

Saved outputs graded offline with AuditKIT (`model="precomputed"`). Each run's sha256 fingerprint covers model, data, scorers and config. 94 hand-verified questions.

| config | fingerprint | valid_json | summary_present | status_correct | no_unsupported_claim | quotes_verbatim | usable_correct | is_json | summary_groundedness |
|---|---|---|---|---|---|---|---|---|---|
| B2_base_v2 | `4e16c7a18cd5` | 0.989 | 0.989 | 0.521 | 0.777 | 0.846 | 0.415 | 0.989 | 0.851 |
| D2_aligned_v2 | `df64935d0b7c` | 0.979 | 0.979 | 0.574 | 0.777 | 0.878 | 0.415 | 0.979 | 0.885 |

## D2_aligned_v2 vs B2_base_v2

| metric | baseline | candidate | delta | p (sign-flip, 10k) | significant at 0.05 |
|---|---|---|---|---|---|
| valid_json | 0.989 | 0.979 | -0.011 | 1.0000 | no |
| summary_present | 0.989 | 0.979 | -0.011 | 1.0000 | no |
| status_correct | 0.521 | 0.574 | +0.053 | 0.1838 | no |
| no_unsupported_claim | 0.777 | 0.777 | +0.000 | 1.0000 | no |
| quotes_verbatim | 0.846 | 0.878 | +0.032 | 0.5705 | no |
| usable_correct | 0.415 | 0.415 | +0.000 | 1.0000 | no |
| is_json | 0.989 | 0.979 | -0.011 | 1.0000 | no |
| summary_groundedness | 0.851 | 0.885 | +0.034 | 0.1011 | no |

AuditKIT comparison summary:

```
Comparison: 4e16c7a18cd5affe (baseline) -> df64935d0b7c14d7 (candidate)
  per-metric (baseline -> candidate | absolute delta | relative %):
    is_json: 0.9894 -> 0.9787  delta=-0.0106 (-1.1% rel)  (n=94->94, std=0.1026->0.1443)
    no_unsupported_claim: 0.7766 -> 0.7766  delta=+0.0000 (+0.0% rel)  (n=94->94, std=0.4165->0.4165)
    quotes_verbatim: 0.8457 -> 0.8777  delta=+0.0319 (+3.8% rel)  (n=94->94, std=0.3500->0.3236)
    status_correct: 0.5213 -> 0.5745  delta=+0.0532 (+10.2% rel)  (n=94->94, std=0.4995->0.4944)
    summary_groundedness: 0.8506 -> 0.8849  delta=+0.0343 (+4.0% rel)  (n=94->94, std=0.1838->0.1741)
    summary_present: 0.9894 -> 0.9787  delta=-0.0106 (-1.1% rel)  (n=94->94, std=0.1026->0.1443)
    usable_correct: 0.4149 -> 0.4149  delta=+0.0000 (+0.0% rel)  (n=94->94, std=0.4927->0.4927)
    valid_json: 0.9894 -> 0.9787  delta=-0.0106 (-1.1% rel)  (n=94->94, std=0.1026->0.1443)
  samples: 2 went correct->wrong, 1 went wrong->correct
  model: precomputed -> precomputed (no measurable size)
```
