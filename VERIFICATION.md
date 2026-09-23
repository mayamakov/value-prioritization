# Verification report — pipeline output vs. manuscript

Regenerated after completion of all failed-call re-queries (all 16 models now
100% genuine responses; see `code/repair/` and Supplementary Results 1).
Every value below was re-derived from the shipped pipeline on a clean checkout.

## Panel and design
| Claim | Pipeline | Status |
|---|---|---|
| Synthetic cohort of 1,000 adults | 1000 | OK |
| 10 physicians, 16 LLMs (26 raters) | 10 / 16 / 26 | OK |
| 300 common pairs labelled by every rater | 300 | OK |
| 224 pairs with >=7-physician majority | 224 | OK |

## Agreement (mean pairwise, 300 common pairs)
| Claim | Pipeline | Status |
|---|---|---|
| Physician-physician 70% | 0.6971 | OK |
| LLM-LLM 85% | 0.8477 | OK |
| Physician-LLM 55% | 0.5478 | OK |

## Signatures and distances
| Claim | Pipeline | Status |
|---|---|---|
| Physician centroid: utility +0.676, priority-to-need -1.174 | +0.676 / -1.174 | OK |
| 14 of 16 LLMs outside the physician range (0.76-2.60) | 14 of 16; within: Gemini 3.1 Pro (2.362), Gemini 3.5 Flash (2.498) | OK |
| Unique risk contribution (LLMs), lambda=2 | 0.3765 ("~0.38"; lambda grid 0.369-0.377) | OK |
| Axis regression, LLM priority-to-need | +3.13 [2.02, 4.24] | OK |
| Run-to-run agreement | median 0.95, range 0.84-0.99 (lowest: Gemini 2.5 Pro 0.838) | OK |

## Per-rater distance from the physician centroid (primary, Supplementary Table S7)
| Model | Distance |
|---|---|
| Gemini 3.1 Pro | 2.362 |
| Gemini 3.5 Flash | 2.498 |
| DeepSeek V3.2 | 2.868 |
| Llama-4 Maverick | 2.917 |
| GPT-5.5 | 2.938 |
| Qwen3.5 27B | 3.156 |
| Gemini 2.5 Pro | 3.288 |
| GPT-5.4 | 3.523 |
| GPT-5 | 3.550 |
| Mistral Medium 3.5 | 3.600 |
| Llama 3.3 70B | 3.603 |
| Claude Opus 4.7 | 3.609 |
| GPT-5-mini | 3.668 |
| Claude Opus 4.8 | 3.727 |
| Claude Opus 4.5 | 3.755 |
| GPT-4o-mini | 3.852 |

## SHAP reproducibility policy
The SHAP artifact shipped in this repository is the canonical artifact analysed in
the paper (raw-vs-mapped Pearson r = 0.849, Spearman rho = 0.884 are properties of
this artifact and are exactly reproduced by the default `run_all.py`, which reads
it). The original SHAP generation predated per-rater seeding and was
processing-order dependent; per-rater seeding was added subsequently for future
runs. Regenerating with `--with-shap` therefore yields equivalent but not
byte-identical SHAP values; no pairwise-based result depends on this stage.

## Rater-label permutation tests (Supplementary Table S3)
Script: `code/perm_test_agreement.py` (10^6 label permutations, seed 2026; resolution floor 1e-6).
| Contrast | Observed difference | p |
|---|---|---|
| Within-physician vs. physician-LLM (69.7% vs. 54.8%) | +14.9 pp | 0.003 |
| Within-LLM vs. physician-LLM (84.8% vs. 54.8%) | +30.0 pp | <1e-5 |
| Pooled within- vs. between-group (80.7% vs. 54.8%) | +25.9 pp | <1e-5 |
| Physicians agreeing more with own group | 9 of 10 | 0.021 (two-sided binomial) |
| LLMs agreeing more with own group | 16 of 16 | 3.1e-5 (two-sided binomial) |

## Shipped Stage-1 intermediate
`results/ethical_value_profiles_per_doctor.csv` holds the SHAP-derived per-rater
signature profiles produced by Stage 1 (`stage1_ethical_value_mapping.py`). It is
shipped with the repository because the SHAP stage is skipped by default, and it is a
required input for `stage2_top_and_pairs.py`, `normalize_tables.py`, and
`make_figS2.py`. It contains SHAP-based signatures and is NOT the source of the
common-pairs signature values in Supplementary Table S7 (those are shipped separately
as `results/value_signatures_distances_TableS7.csv`).

## Full-pipeline check (2026-09-23)
`run_all.py` executed end-to-end on a clean checkout of this commit: stages A, C, D,
all figR figures, and stage E completed; `perm_test_agreement.py` reproduced
Supplementary Table S3 exactly (within-physician 0.6971, within-LLM 0.8477, between
0.5478; +14.9 pp p=0.0034; +30.0 pp and +25.9 pp with 0/10^6 exceedances; 9/10 and
16/16 sign tests). Run-to-run median 0.947 (range 0.838-0.987); raw-vs-mapped
Pearson r=0.849, Spearman rho=0.884.
