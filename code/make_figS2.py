#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figS2.py  -  standalone (re)builder for Supplementary Figure 2
                  (three-method comparison: SHAP-based / Top-50 / Common-pairs)

ROOT CAUSE THIS HANDLES
  generate_all_figures_FINAL.py::figure_S2() self-skips ('! Skip S2') whenever
  the SHAP_ columns in ethical_profiles_combined_top30.csv are empty. Those
  columns are populated by stage2_top_and_pairs.py, which merges them in from
  results/ethical_value_profiles_per_doctor.csv. If that merge predates the SHAP
  file, SHAP_ stays empty and S2 is silently skipped.

WHAT THIS SCRIPT DOES DIFFERENTLY
  * SHAP panel  : read straight from ethical_value_profiles_per_doctor.csv
                  (the authoritative Stage-1 output), NOT the combined SHAP_.
  * Top-50 panel: from analysis1_topN_profiles.csv, and VALIDATE it has all
                  len(LLM_ORDER) LLMs (fails loudly if it is the stale 6-LLM file).
  * Common-pairs: from combined Pairs_.
  * Legend fixed: 'LLMs (n=<count>)' instead of the hardcoded 'all 6'.

IF THE TOP-N FILE IS STALE (only 6 LLMs)  ->  regenerate it (CPU only, no GPU):
      python stage2_top_and_pairs.py
  That re-reads physicians_results for all raters and also repopulates SHAP_.

REQUIRES
  results/ethical_value_profiles_per_doctor.csv   (SHAP-based signatures)
  results/analysis1_topN_profiles.csv             (Top-50, all LLMs)
  results/ethical_profiles_combined_top30.csv     (Pairs_)
  panel_config.py

USAGE
  python make_figS2.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = 'results'
FIG_DIR     = os.path.join('results', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

def R(f):
    p = os.path.join(RESULTS_DIR, f)
    return p if os.path.exists(p) else f

DIMS  = ['Beneficence-Immediate', 'Beneficence-LongTerm', 'Non-maleficence', 'Autonomy', 'Justice']
SHORT = ['Beneficence-\nutility', 'Beneficence-\nneed', 'Non-mal', 'Patient\nempower', 'Resource\njustice']  # v10 display, aligned to DIMS order
HUMAN_COL = '#888888'

from panel_config import MODELS as LLM_ORDER, HUMANS, DISPLAY as DISP  # noqa: E402

def fail(msg):
    print('\n[make_figS2] CANNOT BUILD S2:\n  ' + msg.replace('\n', '\n  ') + '\n')
    sys.exit(1)

# ---------- SHAP-based signatures (authoritative Stage-1 output) ----------
shap_path = R('ethical_value_profiles_per_doctor.csv')
if not os.path.exists(shap_path):
    fail("ethical_value_profiles_per_doctor.csv not found (Stage-1 SHAP output).")
shap_df = pd.read_csv(shap_path).set_index('doctor')
if shap_df[DIMS].isna().all().all():
    fail("ethical_value_profiles_per_doctor.csv has empty value columns.\n"
         "Re-run Stage-1 (GPU) to compute SHAP, then stage2_top_and_pairs.py.")
miss = [r for r in (list(HUMANS) + list(LLM_ORDER)) if r not in shap_df.index]
if miss:
    fail("raters missing from SHAP profiles: " + ", ".join(miss))

# ---------- Top-50 (validate it is the full 16-LLM file) ----------
topn_path = R('analysis1_topN_profiles.csv')
if not os.path.exists(topn_path):
    fail("analysis1_topN_profiles.csv not found.\n"
         "Generate it (CPU only): python stage2_top_and_pairs.py")
topn = pd.read_csv(topn_path)
t50 = topn[topn['top_N'] == 50].copy()
if t50.empty:
    fail("analysis1_topN_profiles.csv has no top_N == 50 rows.")
llm_in_t50 = [d for d in t50['doctor'].unique() if d in set(LLM_ORDER)]
if len(llm_in_t50) < len(LLM_ORDER):
    fail(f"Top-N file is STALE: only {len(llm_in_t50)} of {len(LLM_ORDER)} LLMs at top_N=50.\n"
         f"Missing: {sorted(set(LLM_ORDER) - set(llm_in_t50))}\n"
         f"Regenerate (CPU only, reads all 16 from physicians_results):\n"
         f"    python stage2_top_and_pairs.py")
t50_grp = np.where(t50['entity_type'].astype(str).str.startswith('human'), 'human', 'llm')

# ---------- Common-pairs ----------
comb = pd.read_csv(R('ethical_profiles_combined_top30.csv')).set_index('doctor')
pairs_cols = [f'Pairs_{d}' for d in DIMS]
if comb[pairs_cols].loc[list(LLM_ORDER)].isna().all().all():
    fail("Pairs_ columns empty in ethical_profiles_combined_top30.csv.")

# ---------- plot ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
x = np.arange(len(DIMS)); w = 0.38

def panel(ax, title, hum, llm):
    ax.bar(x - w / 2, hum, w, label='Physicians', color=HUMAN_COL, edgecolor='black', lw=0.5)
    ax.bar(x + w / 2, llm, w, label=f'LLMs (n={len(LLM_ORDER)})',
           color='#4285f4', edgecolor='black', lw=0.5)
    ax.axhline(0, color='black', lw=0.6); ax.set_xticks(x)
    ax.set_xticklabels(SHORT, rotation=30, ha='right', size=9)
    ax.set_title(title, weight='bold'); ax.legend(fontsize=8)

panel(axes[0], 'SHAP-based',
      shap_df.loc[list(HUMANS), DIMS].mean().values,
      shap_df.loc[list(LLM_ORDER), DIMS].mean().values)
panel(axes[1], 'Top 10% patients',
      t50[t50_grp == 'human'][DIMS].mean().values,
      t50[t50_grp == 'llm'][DIMS].mean().values)
panel(axes[2], 'Common pairs',
      comb.loc[list(HUMANS), pairs_cols].mean().values,
      comb.loc[list(LLM_ORDER), pairs_cols].mean().values)
axes[0].set_ylabel('Mean operational weight')
plt.suptitle('Three-method comparison: physician vs LLM operational signatures',
             weight='bold', y=1.02)
plt.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(FIG_DIR, f'figure_S2_three_methods.{ext}'),
                dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"[make_figS2] OK -> {FIG_DIR}/figure_S2_three_methods.png/.pdf  "
      f"(physicians n={len(HUMANS)}, LLMs n={len(LLM_ORDER)})")