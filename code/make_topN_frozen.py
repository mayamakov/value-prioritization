#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_topN_frozen.py  -  Top-N (20/30/50) profiles under the FROZEN v9 mapping.

This is byte-for-byte the same logic as stage2_top_and_pairs.py's Analysis-1
(rank each rater's patients by SHAP score, take top-N, average their patient
values), EXCEPT it imports patient_to_values from the single source of truth
(value_mapping_rec_intrinsic) instead of stage2's stale local /5-split,
rec-weight-Justice copy.

It OVERWRITES results/analysis1_topN_profiles.csv with the frozen-v9 version,
which is what figure_S2 / make_figS2.py consume for the Top-50 panel.

Sanity target (from the manuscript): physician Top-50 Beneficence-LongTerm
should land around +0.91 .. +0.95, NOT stage2's stale ~0.835.

Run from the project root (same place you run stage2):
    python make_topN_frozen.py
"""
import os
import pickle
import numpy as np
import pandas as pd

from data_loading import init_data
from llm_helpers import ALL_DOCTORS, LEGACY_LLMS, NEW_LLMS
# ---- SINGLE SOURCE OF TRUTH (frozen v9): price-Justice, /8 split ----
from value_mapping_rec_intrinsic import patient_to_values, ETHICAL_DIMENSIONS as DIMS
import value_mapping_rec_intrinsic as _vm_mod
_vm_mod.load_support_weights('results/rec_support_weights.csv')   # v11 support weights (item 1)

TOP_N_VALUES = [20, 30, 50]
OUT_DIR = 'results'


def get_entity_type(key):
    if key in LEGACY_LLMS:
        return 'llm_legacy'
    if key in NEW_LLMS:
        return 'llm_new'
    return 'human'


print("=" * 72)
print("make_topN_frozen  -  Top-N profiles under FROZEN v9 mapping")
print("=" * 72)

# guard: make sure we really imported the frozen mapping
import value_mapping_rec_intrinsic as _M
assert hasattr(_M, 'priority_to_need'), "NOT the v10 mapping (no priority_to_need)"
assert _M.REC_INTRINSIC_MAPPING['rec11']['JU'] == 0.0, "NOT frozen (rec11 JU != 0)"
print("v10 mapping confirmed: priority=raw-risk, price-Justice")

print("\nLoading patient data (init_data)...")
(patient_df, df_train, df_train_scaled,
 df_test, df_test_scaled, *_) = init_data()
patient_lookup = patient_df.set_index('patient_num')
test_patient_ids = df_test_scaled['patient_num'].tolist()
print(f"  patients={len(patient_lookup)}  test_patients={len(test_patient_ids)}")

shap_pkl = os.path.join(OUT_DIR, 'ethical_shap_per_patient.pkl')
with open(shap_pkl, 'rb') as f:
    stage1_shap = pickle.load(f)
print(f"  entities with SHAP: {len(stage1_shap)}")

rows = []
for key in ALL_DOCTORS:
    if key not in stage1_shap:
        continue
    sp = stage1_shap[key]['shap_per_patient']
    order = np.argsort(-sp.sum(axis=1))          # same ranking as stage2
    for N in TOP_N_VALUES:
        ids = [test_patient_ids[i] for i in order[:N]]
        acc = {d: 0.0 for d in DIMS}
        for pid in ids:
            if pid not in patient_lookup.index:
                continue
            v = patient_to_values(patient_lookup.loc[pid])   # FROZEN values
            for d in DIMS:
                acc[d] += v[d]
        rows.append({'doctor': key,
                     'entity_type': get_entity_type(key),
                     'top_N': N,
                     **{d: acc[d] / N for d in DIMS}})

df = pd.DataFrame(rows)
out_path = os.path.join(OUT_DIR, 'analysis1_topN_profiles.csv')
df.to_csv(out_path, index=False)
print(f"\n\u2713 Saved -> {out_path}  (FROZEN v9, {df['doctor'].nunique()} raters)")

# ---- sanity vs manuscript ----
h50 = df[(df.top_N == 50) & (df.entity_type == 'human')]['Beneficence-LongTerm']
l50 = df[(df.top_N == 50) & (df.entity_type != 'human')]['Beneficence-LongTerm']
print("\nSANITY (Top-50 Beneficence-LongTerm):")
print(f"  physicians mean = {h50.mean():.3f}   (manuscript +0.91 .. +0.95)")
print(f"  LLMs       mean = {l50.mean():.3f}   (manuscript ~ +0.14)")
print("=" * 72)
