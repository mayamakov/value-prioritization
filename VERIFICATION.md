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

## Per-rater distance from the physician centroid (primary, Table S2)
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
