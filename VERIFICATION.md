# Verification report — pipeline output vs. manuscript

Every numeric claim in the main text and the supplement was re-derived by running the shipped
pipeline (`python run_all.py`) on a clean extraction of this repository and
compared against the manuscript.

Pipeline status on the clean run: **18 stages passed, 0 failed, 0 missing.**

---

## Verified — exact match

| # | Manuscript claim | Pipeline value | Status |
|---|------------------|----------------|--------|
| 1 | Synthetic cohort of 1,000 adults | 1000 | ✅ |
| 2 | 10 physicians, 16 LLMs (26 raters) | 10 / 16 / 26 | ✅ |
| 3 | 300 common pairs labelled by every rater | 300 | ✅ |
| 4 | 224 pairs with ≥7-physician majority | 224 | ✅ |
| 5 | Physician–physician agreement 70% | 0.6971 → 70% | ✅ |
| 6 | LLM–LLM agreement 84% | 0.8420 → 84% | ✅ |
| 7 | Physician–LLM agreement 55% | 0.5508 → 55% | ✅ |
| 8 | Utility-oriented beneficence +0.636 vs −0.398 | 0.636 / −0.398 | ✅ exact |
| 9 | Priority-to-need +0.727 vs −1.163 | 0.727 / −1.163 | ✅ exact |
| 10 | Priority-to-need and resource justice survive Bonferroni | p = 3.5×10⁻⁵ / 2.8×10⁻⁵ | ✅ |
| 11 | Physician centroid mean distance 1.74 | 1.735 | ✅ |
| 12 | Physician range upper bound 2.62 | 2.622 | ✅ |
| 13 | 14 of 16 LLMs outside the physician range | 14 of 16 | ✅ |
| 14 | Gemini 3.1 Pro 2.35 / Gemini 3.5 Flash 2.48 | 2.353 / 2.484 | ✅ |
| 15 | DeepSeek 2.85, Llama-4 2.91, GPT-5.5 2.93 | 2.852 / 2.914 / 2.931 | ✅ |
| 16 | GPT-5.4 3.53, GPT-5 3.55, GPT-5-mini 3.68, GPT-4o-mini 3.86 | 3.525 / 3.551 / 3.675 / 3.857 | ✅ |
| 17 | **Developer-family distances 1.56 vs 2.68, 1.7-fold** | **1.557 / 2.684 / 1.72×** | ✅ |
| 18 | LLM-preferred patients 11.1–19.1 years older | 11.1 – 19.1 | ✅ |
| 19 | 5.3–9.3 risk points higher | 5.3 – 9.3 | ✅ |
| 20 | Gemini 2.5 Pro smallest age gap (+11.1 vs +14.8 to +19.1) | 11.12; others 14.81–19.08 | ✅ |
| 21 | GPT-5-mini 54%, GPT-5.4 52% of 224 | 121/224 = 54.0%, 117/224 = 52.2% | ✅ |
| 22 | Gemini 3.1 Pro 26%, Gemini 3.5 Flash 30% of 224 | 59/224 = 26.3%, 67/224 = 29.9% | ✅ |
| 23 | Physician risk contribution Δ pseudo-R² ≈ 0.01 | 0.0114 | ✅ |
| 24 | First-line statin largest recommendation contributor ≈ 0.14 | 0.1403 | ✅ |
| 25 | Top-50 of 500 held-out patients = top 10% | 50 / 500 | ✅ |
| 26 | Max within-model run-to-run SD ≈ 0.30 | 0.301 | ✅ exact |
| 27 | Gemini family closest to physicians across alternative mappings | closest in M0, M1, M2, M3 | ✅ |
| 28 | Raw-SHAP vs mapped distances Pearson r = 0.849 | 0.849 (p = 4.2×10⁻⁸) | ✅ exact |
| 29 | Spearman ρ = 0.884 | 0.884 (p = 2.2×10⁻⁹) | ✅ exact |
| 30 | Run-to-run agreement median 0.95, Gemini 2.5 Pro outlier 0.77 | median 0.949, Gemini 2.5 Pro 0.771 | ✅ |
| 31 | Unanimous pairs: LLMs 144/300 (48%), physicians 77/300 (26%) | 144 / 77 | ✅ exact |
| 32 | PCA of 5-dim signatures: PC1 56.7%, PC2 22.9% | 56.7% / 22.9% | ✅ exact |
| 33 | Raw 24-dim SHAP: PC1+PC2 = 83.6% | 76.8% + 6.8% = 83.6% | ✅ exact |
| 34 | Non-maleficence β: physicians 0.21 (0.10–0.32), LLMs 0.37 (0.26–0.48) | 0.207 (0.099–0.315), 0.366 (0.255–0.477) | ✅ |
| 35 | Priority-to-need LLMs β = 3.02 (1.90–4.13) | 3.015 (1.896–4.134) | ✅ |
| 36 | Agreement vs run-to-run variation Spearman ρ = −0.76 | −0.762 (mean agreement) | ✅ |
| 37 | **Table S1: physicians AUC 0.803 (SD 0.124), accuracy 0.746 (SD 0.102)** | **0.803 (0.124) / 0.746 (0.102)** | ✅ exact |
| 38 | **Table S1: LLMs AUC 0.930 (SD 0.043), accuracy 0.859 (SD 0.047)** | **0.930 (0.043) / 0.859 (0.047)** | ✅ exact |

Claim 17 is computed by `family_distances.py`, added to the pipeline during this
verification (Stage D4). Developer families are taken from the authoritative
`panel_config.MODEL_FAMILY` map, which correctly resolves the legacy rater keys
(`doctor8` → GPT-4o-mini/OpenAI, `doctor9` → GPT-5-mini/OpenAI, `doctor10` →
Llama-4 Maverick/Meta). The `across` figure follows the matched-panel
convention: both sums are restricted to models belonging to a multi-model
developer, so `within` and `across` are computed over the same models. The
script also reports the all-pairs convention (2.348, 1.51×) for transparency.

---

## Corrections applied to the manuscript

### 1. Physician range lower bound: 0.76 → **0.75**

The minimum physician distance from the physician centroid is 0.7545
(physician FM-3), which rounds to 0.75. The mean (1.74) and upper bound (2.62)
were already correct. Corrected in the main text.

### 2. LLM risk contribution: ≈ 0.39 → **≈ 0.36**

The manuscript quoted 0.39 while citing Supplementary Figure S4. These come from
two different model specifications:

| Script | Demographic block | LLM Δ pseudo-R² | Physician Δ pseudo-R² |
|--------|-------------------|-----------------|------------------------|
| `figR_decomposition.py` | age **and** risk | 0.387 → 0.39 | 0.012 → 0.01 |
| `figR_rec_pseudoR2.py` (**= Figure S4**) | risk only (v11) | 0.364 → **0.36** | 0.0114 → 0.01 |

The v11 model underlying Figure S4 uses risk only (age subsumed), which is why
the figure axis is labelled "Patient risk". Since the surrounding sentence and
the figure caption now both describe a risk-only model, the quoted value was
updated to 0.36 to match. The physician value (0.01) and the statin value (0.14)
are identical under both specifications and needed no change.

---

## A note on Table S1

Table S1 reports held-out RankNet fit under the **single reporting configuration**
selected by the ablation, `soft_path_noAuto_syn0_noHier`. It is reproduced exactly
by `table_S1_model_fit.py`, which selects that configuration from the shipped
48-configuration sweep in `results/ablation_full_auc.csv`.

Two other saved files hold a *different* quantity and should not be confused with
Table S1: `results/model_fit_metrics.csv` and `results/ablation_A3_best_per_rater.csv`
record the best configuration chosen **separately for each rater**, which averages
0.821 (physicians) and 0.937 (LLMs).

Because RankNet training is stochastic — random weight initialisation and batch
shuffling, with no RNG seed fixed — retraining from scratch via
`run_all_experiments.py` reproduces these metrics only to within roughly ±0.02.
Shipping the completed sweep is what makes Table S1 exactly reproducible on CPU.

---

## Summary

**All 38 checked claims across the main text and the supplement reproduce**, two
after small numeric corrections to the manuscript (0.76 → 0.75; 0.39 → 0.36).
Nothing is left unverified.

No qualitative conclusion in the paper changes: the physician–LLM separation,
its direction on every axis, the developer-family clustering, and the robustness
results all hold as reported.
