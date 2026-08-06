"""
============================================================================
STAGE 2: TWO COMPLEMENTARY ETHICAL-VALUE ANALYSES
============================================================================
Analysis 1: TOP-N PATIENTS (N = 20, 30, 50)
  For each entity, sum its SHAP per patient -> patient score.
  Rank patients, take top N, compute ethical-value profile based on:
    - which recs they have
    - their age & risk (for Beneficence-Immediate/LongTerm split)

  CHANGE FROM PREVIOUS: now also computes Top-50 (sensitivity analysis).
  Top-50 is used as the primary supporting analysis in Figure S2.

Analysis 2: COMMON-PAIRS (300 pairs shared across all entities)
  For each common pair (A, B), each entity tells us who "won".
  The differences between A and B (which recs differ? age? risk?)
  translate to ethical-value contributions.

Both analyses use the SAME REC_ETHICAL_MAPPING from Stage 1, so the
three analyses (Top-N, Common-pairs, SHAP-based) are comparable.

Output:
  - results/analysis1_topN_profiles.csv          (N = 20, 30, 50)
  - results/analysis2_common_pairs_profiles.csv
============================================================================
"""
import os
import pickle
import numpy as np
import pandas as pd

from data_loading import init_data
from llm_helpers import (ALL_DOCTORS, ALL_HUMAN_DOCTORS, LEGACY_LLMS,
                          NEW_LLMS, load_ranked_pairs)

# Ethical framework - SAME as Stage 1
ETHICAL_DIMENSIONS = [
    'Beneficence-Immediate',
    'Beneficence-LongTerm',
    'Non-maleficence',
    'Autonomy',
    'Justice',
]

REC_ETHICAL_MAPPING = {
    'rec1':  {'Beneficence': 0.0, 'Non-maleficence': 0.3, 'Autonomy': 0.0, 'Justice': 0.3},
    'rec2':  {'Beneficence': 0.0, 'Non-maleficence': 1.0, 'Autonomy': 0.0, 'Justice': 0.0},
    'rec3':  {'Beneficence': 0.0, 'Non-maleficence': 0.3, 'Autonomy': 0.0, 'Justice': 0.3},
    'rec4':  {'Beneficence': 0.3, 'Non-maleficence': 0.3, 'Autonomy': 0.0, 'Justice': 0.0},
    'rec5':  {'Beneficence': 0.3, 'Non-maleficence': 0.3, 'Autonomy': 0.0, 'Justice': 0.0},
    'rec6':  {'Beneficence': 0.5, 'Non-maleficence': 0.0, 'Autonomy': 0.0, 'Justice': -0.5},
    'rec8':  {'Beneficence': 0.0, 'Non-maleficence': 0.3, 'Autonomy': 0.0, 'Justice': 0.5},
    'rec10': {'Beneficence': 1.0, 'Non-maleficence': 0.0, 'Autonomy': 0.0, 'Justice': 0.5},
    'rec11': {'Beneficence': 1.0, 'Non-maleficence': 0.0, 'Autonomy': 0.0, 'Justice': -1.0},
    'rec12': {'Beneficence': 1.0, 'Non-maleficence': 0.3, 'Autonomy': 0.0, 'Justice': 0.0},
    'rec13': {'Beneficence': 0.3, 'Non-maleficence': 1.0, 'Autonomy': 0.0, 'Justice': 0.0},
    'rec16': {'Beneficence': 0.3, 'Non-maleficence': 0.0, 'Autonomy': 0.0, 'Justice': 0.0},
    'rec17': {'Beneficence': 0.3, 'Non-maleficence': 0.0, 'Autonomy': 0.0, 'Justice': 0.0},
    'rec18': {'Beneficence': 0.0, 'Non-maleficence': 0.0, 'Autonomy': 1.0, 'Justice': 0.0},
    'rec19': {'Beneficence': 0.0, 'Non-maleficence': 0.0, 'Autonomy': 1.0, 'Justice': 0.5},
    'rec21': {'Beneficence': 0.0, 'Non-maleficence': 0.0, 'Autonomy': 0.0, 'Justice': 0.0},
}

AGE_THRESHOLD  = 65
RISK_THRESHOLD = 20

# CHANGE: added 50 to the list
TOP_N_VALUES = [20, 30, 50]

# CHANGE: which N is the primary supporting analysis in figures
PRIMARY_TOP_N = 50

OUT_DIR = 'results'


def beneficence_split(age, risk):
    """For one patient, returns (weight_immediate, weight_longterm)."""
    age_imm = 1.0 / (1.0 + np.exp(-(age - AGE_THRESHOLD) / 5.0))
    age_lt  = 1.0 - age_imm
    risk_boost = 1.0 / (1.0 + np.exp(-(risk - RISK_THRESHOLD) / 5.0))
    w_imm = age_imm * (1.0 + risk_boost) / 2.0 + risk_boost * 0.5
    w_lt  = age_lt  * (1.0 - 0.5 * risk_boost)
    total = w_imm + w_lt + 1e-9
    return w_imm / total, w_lt / total


def get_entity_type(key):
    if key in LEGACY_LLMS:
        return 'llm_legacy'
    if key in NEW_LLMS:
        return 'llm_new'
    return 'human'



# === v10 OVERRIDE: use the canonical mapping, not the local v9 copy ===
from value_mapping_rec_intrinsic import patient_to_values as _v10_ptv
import value_mapping_rec_intrinsic as _vm_mod
_vm_mod.load_support_weights('results/rec_support_weights.csv')   # v11 support weights (item 1)
def patient_to_values(patient_row, weight=1.0):
    return _v10_ptv(patient_row)
# === /v10 OVERRIDE ===
def _DISABLED_local_patient_to_values(patient_row, weight=1.0):
    """
    Given a patient row (from patient_df), return value contributions
    based on which recs are active and patient demographics.

    patient_row must have: age, risk, rec1..rec21
    Returns dict of {dim_name -> value}
    """
    values = {d: 0.0 for d in ETHICAL_DIMENSIONS}

    age  = float(patient_row.get('age', 60))
    risk = float(patient_row.get('risk', 15))
    w_imm, w_lt = beneficence_split(age, risk)

    for rec, mapping in REC_ETHICAL_MAPPING.items():
        if rec not in patient_row.index:
            continue
        if patient_row[rec] < 0.5:
            continue
        # Beneficence split by age/risk
        if mapping['Beneficence'] != 0:
            base = mapping['Beneficence'] * weight
            values['Beneficence-Immediate'] += base * w_imm
            values['Beneficence-LongTerm']  += base * w_lt
        if mapping['Non-maleficence'] != 0:
            values['Non-maleficence'] += mapping['Non-maleficence'] * weight
        if mapping['Autonomy'] != 0:
            values['Autonomy'] += mapping['Autonomy'] * weight
        if mapping['Justice'] != 0:
            values['Justice'] += mapping['Justice'] * weight

    return values


# ============================================================================
# LOAD DATA
# ============================================================================
print("=" * 80)
print("STAGE 2: TOP-N + COMMON-PAIRS ANALYSES")
print(f"  Top-N values: {TOP_N_VALUES} (primary supporting: Top-{PRIMARY_TOP_N})")
print("=" * 80)

print("\nLoading patient data...")
(patient_df, df_train, df_train_scaled,
 df_test, df_test_scaled, *_) = init_data()
patient_lookup = patient_df.set_index('patient_num')
print(f"  Patients: {len(patient_lookup)}")

# Load SHAP per patient from Stage 1
shap_pkl = os.path.join(OUT_DIR, 'ethical_shap_per_patient.pkl')
print(f"\nLoading Stage-1 SHAP from {shap_pkl}...")
with open(shap_pkl, 'rb') as f:
    stage1_shap = pickle.load(f)
print(f"  Entities with SHAP: {len(stage1_shap)}")

test_patient_ids = df_test_scaled['patient_num'].tolist()
print(f"  Test patients (eval set used for SHAP): {len(test_patient_ids)}")

# ============================================================================
# ANALYSIS 1: TOP-N PATIENTS
# ============================================================================
print("\n" + "=" * 80)
print(f"ANALYSIS 1: TOP-N PATIENTS (N = {TOP_N_VALUES})")
print("=" * 80)

results_top = []
for entity_key in ALL_DOCTORS:
    if entity_key not in stage1_shap:
        continue
    shap_per_patient = stage1_shap[entity_key]['shap_per_patient']
    patient_scores = shap_per_patient.sum(axis=1)
    order = np.argsort(-patient_scores)

    for N in TOP_N_VALUES:
        top_idx = order[:N]
        top_patient_ids = [test_patient_ids[i] for i in top_idx]

        profile_sum = {d: 0.0 for d in ETHICAL_DIMENSIONS}
        for pid in top_patient_ids:
            if pid not in patient_lookup.index:
                continue
            patient_row = patient_lookup.loc[pid]
            vals = patient_to_values(patient_row, weight=1.0)
            for d in ETHICAL_DIMENSIONS:
                profile_sum[d] += vals[d]
        profile_mean = {d: v / N for d, v in profile_sum.items()}

        results_top.append({
            'doctor':      entity_key,
            'entity_type': get_entity_type(entity_key),
            'top_N':       N,
            **profile_mean,
        })

df_top = pd.DataFrame(results_top)
top_path = os.path.join(OUT_DIR, 'analysis1_topN_profiles.csv')
df_top.to_csv(top_path, index=False)
print(f"\n✓ Saved -> {top_path}")

for N in TOP_N_VALUES:
    print(f"\n--- Top-{N} profiles ---")
    sub = df_top[df_top['top_N'] == N].copy()
    print(sub[['doctor', 'entity_type'] + ETHICAL_DIMENSIONS].round(3).to_string(index=False))

print("\n--- Mean by entity_type, by N ---")
mean_by = df_top.groupby(['top_N', 'entity_type'])[ETHICAL_DIMENSIONS].mean().round(3)
print(mean_by.to_string())

# ============================================================================
# ANALYSIS 2: COMMON PAIRS (300 pairs shared by ALL entities)
# ============================================================================
print("\n\n" + "=" * 80)
print("ANALYSIS 2: COMMON PAIRS (300 shared pairs)")
print("=" * 80)

all_pairs_per_entity = {}
for key in ALL_DOCTORS:
    train, test = load_ranked_pairs(key, 'physicians_results')
    pair_sets = {'train': set(), 'test': set()}
    pair_dir  = {}
    for stage, pairs in [('train', train), ('test', test)]:
        for (a, b), conf in pairs:
            unordered = tuple(sorted([a, b]))
            pair_sets[stage].add(unordered)
            pair_dir[unordered] = a
    all_pairs_per_entity[key] = {
        'train_set': pair_sets['train'],
        'test_set':  pair_sets['test'],
        'pair_dir':  pair_dir,
    }

common_train = set.intersection(*[d['train_set'] for d in all_pairs_per_entity.values()])
common_test  = set.intersection(*[d['test_set']  for d in all_pairs_per_entity.values()])
common_all   = list(common_train) + list(common_test)
print(f"\nCommon pairs: train={len(common_train)}, test={len(common_test)}, total={len(common_all)}")

results_pairs = []
for entity_key in ALL_DOCTORS:
    pair_dir = all_pairs_per_entity[entity_key]['pair_dir']
    profile_sum = {d: 0.0 for d in ETHICAL_DIMENSIONS}
    n_used = 0

    for (a, b) in common_all:
        if (a, b) not in pair_dir:
            continue
        winner = pair_dir[(a, b)]
        loser  = b if winner == a else a

        if winner not in patient_lookup.index or loser not in patient_lookup.index:
            continue

        win_row  = patient_lookup.loc[winner]
        lose_row = patient_lookup.loc[loser]

        win_vals  = patient_to_values(win_row,  weight=1.0)
        lose_vals = patient_to_values(lose_row, weight=1.0)

        for d in ETHICAL_DIMENSIONS:
            profile_sum[d] += (win_vals[d] - lose_vals[d])
        n_used += 1

    profile_mean = ({d: v / max(n_used, 1) for d, v in profile_sum.items()}
                    if n_used > 0 else
                    {d: 0.0 for d in ETHICAL_DIMENSIONS})

    results_pairs.append({
        'doctor':      entity_key,
        'entity_type': get_entity_type(entity_key),
        'n_pairs':     n_used,
        **profile_mean,
    })

df_pairs = pd.DataFrame(results_pairs)
pairs_path = os.path.join(OUT_DIR, 'analysis2_common_pairs_profiles.csv')
df_pairs.to_csv(pairs_path, index=False)
print(f"\n✓ Saved -> {pairs_path}")

print("\n--- Common-pairs differential profiles ---")
print(df_pairs[['doctor', 'entity_type', 'n_pairs'] + ETHICAL_DIMENSIONS].round(3).to_string(index=False))

print("\n--- Mean by entity_type ---")
mean_pair = df_pairs.groupby('entity_type')[ETHICAL_DIMENSIONS].mean().round(3)
print(mean_pair.to_string())

humans = df_pairs[df_pairs['entity_type'] == 'human']
llms   = df_pairs[df_pairs['entity_type'].str.startswith('llm')]
if not humans.empty and not llms.empty:
    print("\n--- HUMAN-LLM divergence (common-pairs analysis) ---")
    print(f"  {'Dimension':<25} {'humans':>10} {'LLMs':>10} {'diff':>10}")
    for d in ETHICAL_DIMENSIONS:
        h = humans[d].mean()
        l = llms[d].mean()
        print(f"  {d:<25} {h:>+10.3f} {l:>+10.3f} {l - h:>+10.3f}")

# ============================================================================
# COMBINED COMPARISON: SHAP-based vs Top-N (primary) vs Common-pairs
# ============================================================================
print("\n\n" + "=" * 80)
print(f"CROSS-ANALYSIS COMPARISON (Top-{PRIMARY_TOP_N} as primary supporting)")
print("=" * 80)

shap_path = os.path.join(OUT_DIR, 'ethical_value_profiles_per_doctor.csv')
df_shap = pd.read_csv(shap_path)
df_shap_simple = df_shap[['doctor'] + ETHICAL_DIMENSIONS].copy()
df_shap_simple.columns = ['doctor'] + [f"SHAP_{d}" for d in ETHICAL_DIMENSIONS]

# CHANGE: Use Top-50 as primary supporting analysis (was Top-30)
df_topN = df_top[df_top['top_N'] == PRIMARY_TOP_N][['doctor'] + ETHICAL_DIMENSIONS].copy()
df_topN.columns = ['doctor'] + [f"Top{PRIMARY_TOP_N}_{d}" for d in ETHICAL_DIMENSIONS]

df_cp = df_pairs[['doctor'] + ETHICAL_DIMENSIONS].copy()
df_cp.columns = ['doctor'] + [f"Pairs_{d}" for d in ETHICAL_DIMENSIONS]

df_all = df_shap_simple.merge(df_topN, on='doctor').merge(df_cp, on='doctor')

print(f"\n--- CORRELATION BETWEEN ANALYSES (per ethical dimension, across doctors) ---")
print(f"  {'Dimension':<25} {f'SHAP-Top{PRIMARY_TOP_N}':>12} {'SHAP-Pairs':>12} {f'Top{PRIMARY_TOP_N}-Pairs':>13}")
for d in ETHICAL_DIMENSIONS:
    c1 = df_all[f"SHAP_{d}"].corr(df_all[f"Top{PRIMARY_TOP_N}_{d}"])
    c2 = df_all[f"SHAP_{d}"].corr(df_all[f"Pairs_{d}"])
    c3 = df_all[f"Top{PRIMARY_TOP_N}_{d}"].corr(df_all[f"Pairs_{d}"])
    print(f"  {d:<25} {c1:>+12.3f} {c2:>+12.3f} {c3:>+13.3f}")

combined_path = os.path.join(OUT_DIR, 'ethical_profiles_combined.csv')
df_all.to_csv(combined_path, index=False)
print(f"\n✓ Saved combined comparison -> {combined_path}")

# Also save the Top-30 combined for backward compatibility / appendix
df_top30 = df_top[df_top['top_N'] == 30][['doctor'] + ETHICAL_DIMENSIONS].copy()
df_top30.columns = ['doctor'] + [f"Top30_{d}" for d in ETHICAL_DIMENSIONS]
df_all_top30 = df_shap_simple.merge(df_top30, on='doctor').merge(df_cp, on='doctor')
combined_top30_path = os.path.join(OUT_DIR, 'ethical_profiles_combined_top30.csv')
df_all_top30.to_csv(combined_top30_path, index=False)
print(f"✓ Also saved Top-30 version -> {combined_top30_path}")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
