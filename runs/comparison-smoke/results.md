# Results: comparison-smoke

- **Status:** Run marked complete. Scripted smoke-test data is not model evidence.
- **Backend:** scripted_smoke_only
- **Cases:** 12
- **Worker model:** qwen3:1.7b-q4_K_M
- **Supervisor model:** qwen3:8b-q4_K_M

## End-to-end results

| Metric | Workers only | Generic review | Custom review |
| --- | --- | --- | --- |
| Exact final answers | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| Final hallucinations | 100.0% (36/36) | 100.0% (24/24) | 100.0% (24/24) |
| Final case errors | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| Final answer coverage | 100.0% (30/30) | 63.3% (19/30); 95% CI 56.5–66.7% | 63.3% (19/30); 95% CI 56.5–66.7% |
| Worker hallucinations | 100.0% (36/36) | 100.0% (36/36) | 100.0% (36/36) |
| Worker failures | 0.0% (0/36) | 0.0% (0/36) | 0.0% (0/36) |
| Reviewer misses | N/A | 33.3% (12/36) | 33.3% (12/36) |
| Worker errors escaping review | N/A | 33.3% (12/36) | 33.3% (12/36) |
| Verified reviewer repairs | N/A | 0.0% (0/36) | 0.0% (0/36) |
| Supervisor-originated hallucinations | undefined (0/0) | 100.0% (12/12) | 100.0% (12/12) |
| Synthesis failures | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |

## Reviewer replay (same saved inputs)

| Metric | Generic review | Custom review |
| --- | --- | --- |
| Reviewer catch recall | 66.7% (24/36) | 66.7% (24/36) |
| Reviewer misses | 33.3% (12/36) | 33.3% (12/36) |
| Reviewer false alarms | undefined (0/0) | undefined (0/0) |
| Verified reviewer repairs | 0.0% (0/36) | 0.0% (0/36) |
| Reviewer-originated hallucinations | 100.0% (12/12) | 100.0% (12/12) |
| Reviewer failures | 0.0% (0/36) | 0.0% (0/36) |

## Paired differences

| Metric | Comparison | Delta |
| --- | --- | --- |
| Final hallucinations | Generic review - Workers only | +0.0 pp (24/24 vs 36/36) |
| Final case errors | Generic review - Workers only | +0.0 pp (12/12 vs 12/12) |
| Exact final answers | Generic review - Workers only | +0.0 pp (0/12 vs 0/12) |
| Final hallucinations | Custom review - Generic review | +0.0 pp (24/24 vs 24/24) |
| Final case errors | Custom review - Generic review | +0.0 pp (12/12 vs 12/12) |
| Exact final answers | Custom review - Generic review | +0.0 pp (0/12 vs 0/12) |
| Final hallucinations | Custom review - Workers only | +0.0 pp (24/24 vs 36/36) |
| Final case errors | Custom review - Workers only | +0.0 pp (12/12 vs 12/12) |
| Exact final answers | Custom review - Workers only | +0.0 pp (0/12 vs 0/12) |
| Reviewer misses | Custom review - Generic review | +0.0 pp (12/36 vs 12/36) |
| Reviewer false alarms | Custom review - Generic review | undefined (0/0 vs 0/0) |

Rates show numerator/denominator. Confidence intervals are 95% case-clustered bootstrap intervals when available.
End-to-end results and reviewer replay are separate experiments. Detailed case-type and worker-field breakdowns remain in `summary.csv`.
