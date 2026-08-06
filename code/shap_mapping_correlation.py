#!/usr/bin/env python3
"""
shap_mapping_correlation.py — does the value mapping create the separation?

Compares, for all 26 raters, the distance from the physician centroid measured
two ways: in the raw 24-dimensional SHAP space (before any ethical projection),
and in the mapped 5-dimensional value space. A strong correlation means the
physician-LLM structure is already present in the raw attributions and is not an
artefact of the Beauchamp-Childress mapping.

Outputs results/shap_mapping_correlation.csv and prints Pearson / Spearman.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from panel_config import HUMANS, DISPLAY

HERE = os.path.dirname(os.path.abspath(__file__)) or "."
RESULTS = os.path.join(HERE, "results")
if not os.path.isdir(RESULTS):
    RESULTS = os.path.join(os.path.dirname(HERE), "results")

DIMS = ['Beneficence-Immediate', 'Beneficence-LongTerm',
        'Non-maleficence', 'Autonomy', 'Justice']

RAW = os.path.join(RESULTS, 'raw_shap_clustering_distances.csv')
MAPPED = os.path.join(RESULTS, 'ethical_value_profiles_per_doctor.csv')


def main():
    for p in (RAW, MAPPED):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} — run raw_shap_clustering_FIXED.py and stage A first")

    raw = pd.read_csv(RAW).set_index('rater')
    prof = pd.read_csv(MAPPED)
    key = 'doctor' if 'doctor' in prof.columns else prof.columns[0]
    prof = prof.set_index(key)

    hum = [h for h in HUMANS if h in prof.index]
    centroid = prof.loc[hum, DIMS].mean().to_numpy(float)

    rows = []
    for r in raw.index:
        if r not in prof.index:
            continue
        rows.append({
            'rater': r,
            'label': DISPLAY.get(r, r),
            'entity_type': 'human' if r in HUMANS else 'llm',
            'distance_raw_shap': float(raw.loc[r, 'distance_raw_shap']),
            'distance_mapped': float(np.linalg.norm(
                prof.loc[r, DIMS].to_numpy(float) - centroid)),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, 'shap_mapping_correlation.csv'), index=False)

    pr, pp = pearsonr(df.distance_raw_shap, df.distance_mapped)
    sr, sp = spearmanr(df.distance_raw_shap, df.distance_mapped)

    print("=" * 70)
    print("Raw 24-dim SHAP distance vs mapped 5-dim distance from physician centroid")
    print(f"  raters : {len(df)} ({(df.entity_type=='human').sum()} physicians, "
          f"{(df.entity_type=='llm').sum()} LLMs)")
    print("=" * 70)
    print(f"  Pearson  r = {pr:.3f}   (p = {pp:.2g})")
    print(f"  Spearman r = {sr:.3f}   (p = {sp:.2g})")
    print("=" * 70)
    print("  wrote results/shap_mapping_correlation.csv")


if __name__ == '__main__':
    main()
