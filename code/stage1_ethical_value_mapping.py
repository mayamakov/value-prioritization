"""
============================================================================
STAGE 1: ETHICAL VALUE PROFILE PER DOCTOR
============================================================================
Maps SHAP attributions to 5 ethical-value dimensions based on the
Beauchamp & Childress framework, with Beneficence split into Immediate vs
LongTerm depending on patient age and risk.

5 dimensions:
  1. Beneficence-Immediate    (treat the patient NOW - older, higher risk)
  2. Beneficence-LongTerm     (preventive care - younger, lower risk)
  3. Non-maleficence          ("first do no harm")
  4. Autonomy                 (respect patient choice / lifestyle)
  5. Justice                  (fair allocation of resources)

Two thresholds:
  - Age:  65   (soft sigmoid)
  - Risk: 20   (soft sigmoid)

Config: soft_path_noAuto_syn0_noHier

Output:
  - results/ethical_value_profiles_per_doctor.csv
  - results/ethical_shap_per_patient.pkl

Paste into a new notebook cell and run. ~20-30 minutes total.
============================================================================
"""
import os
import sys
import subprocess
import numpy as np
import pandas as pd
import torch
import warnings
import pickle
warnings.filterwarnings('ignore')

try:
    import shap
except ImportError:
    print("Installing shap...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "shap"])
    import shap

from data_loading import init_data
from utils import FEATURE_COLS
from llm_helpers import (ALL_DOCTORS, ALL_HUMAN_DOCTORS, ALL_LLMS,
                          LEGACY_LLMS, NEW_LLMS,
                          load_ranked_pairs, get_group)
from run_recipe_ablation_v2 import train_one_recipe

# ============================================================================
# ETHICAL VALUE MAPPING
# ============================================================================
ETHICAL_DIMENSIONS = [
    'Beneficence-Immediate',
    'Beneficence-LongTerm',
    'Non-maleficence',
    'Autonomy',
    'Justice',
]

# For each rec, weight of its SHAP contribution to each *base* dimension.
# Beneficence is the only one that gets split later by age/risk.
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

# Thresholds
AGE_THRESHOLD  = 65
RISK_THRESHOLD = 20


def beneficence_split(age, risk):
    """
    For a single patient, return (weight_immediate, weight_longterm)
    based on age (sigmoid around 65) and risk (sigmoid around 20).

      - older patient  -> more Immediate
      - younger patient -> more LongTerm
      - high risk      -> boost Immediate
    """
    # base age weight - sigmoid: ~0 if young, ~1 if old
    age_immediate = 1.0 / (1.0 + np.exp(-(age - AGE_THRESHOLD) / 5.0))
    age_longterm  = 1.0 - age_immediate

    # risk modifier: high risk -> add more Immediate
    # sigmoid on risk: ~0 if low risk, ~1 if high risk
    risk_boost = 1.0 / (1.0 + np.exp(-(risk - RISK_THRESHOLD) / 5.0))

    # final weights - risk increases Immediate at the expense of LongTerm
    w_immediate = age_immediate * (1.0 + risk_boost) / 2.0 + risk_boost * 0.5
    w_longterm  = age_longterm  * (1.0 - 0.5 * risk_boost)

    # Normalize so weights sum to ~1 (loose constraint)
    total = w_immediate + w_longterm + 1e-9
    return w_immediate / total, w_longterm / total


# === v10 OVERRIDE: use the canonical SHAP aggregation (Priority = raw risk) ===
from value_mapping_rec_intrinsic import aggregate_shap_to_ethical_values as _v10_agg
def aggregate_shap_to_ethical_values(shap_matrix, feature_cols, patient_df_orig):
    return _v10_agg(shap_matrix, feature_cols, patient_df_orig)
# === /v10 OVERRIDE ===
def _DISABLED_local_aggregate(shap_matrix, feature_cols, patient_df_orig):
    """
    shap_matrix: (n_patients, n_features) SHAP for one model.
    Returns:     (n_patients, 5)     ethical value attributions.
    """
    n_patients = shap_matrix.shape[0]
    out = np.zeros((n_patients, len(ETHICAL_DIMENSIONS)))

    # Map dimension name to column index
    DIM = {d: i for i, d in enumerate(ETHICAL_DIMENSIONS)}

    # Get patient age & risk arrays for split computation
    ages  = patient_df_orig['age'].values  if 'age'  in patient_df_orig else np.full(n_patients, 60.0)
    risks = patient_df_orig['risk'].values if 'risk' in patient_df_orig else np.full(n_patients, 15.0)

    for fi, feat in enumerate(feature_cols):
        if feat not in REC_ETHICAL_MAPPING:
            continue
        mapping = REC_ETHICAL_MAPPING[feat]

        # Only patients who actually HAVE this rec contribute.
        if feat in patient_df_orig.columns:
            has_rec = (patient_df_orig[feat].values >= 0.5).astype(float)
        else:
            has_rec = np.ones(n_patients, dtype=float)

        rec_contribution = shap_matrix[:, fi] * has_rec

        # Beneficence: split into Immediate / LongTerm per patient
        if mapping['Beneficence'] != 0:
            for pi in range(n_patients):
                w_imm, w_lt = beneficence_split(ages[pi], risks[pi])
                base = rec_contribution[pi] * mapping['Beneficence']
                out[pi, DIM['Beneficence-Immediate']] += base * w_imm
                out[pi, DIM['Beneficence-LongTerm']]  += base * w_lt

        # Non-maleficence (no split)
        if mapping['Non-maleficence'] != 0:
            out[:, DIM['Non-maleficence']] += rec_contribution * mapping['Non-maleficence']

        # Autonomy (no split)
        if mapping['Autonomy'] != 0:
            out[:, DIM['Autonomy']] += rec_contribution * mapping['Autonomy']

        # Justice (no split, but can be negative)
        if mapping['Justice'] != 0:
            out[:, DIM['Justice']] += rec_contribution * mapping['Justice']

    return out


# ============================================================================
# MAIN
# ============================================================================
NUM_EPOCHS    = 300
SEED          = 42
N_BACKGROUND  = 100
RESULTS_DIR   = 'physicians_results'
OUT_DIR       = 'results'

print("=" * 80)
print("STAGE 1: ETHICAL VALUE PROFILE PER DOCTOR")
print(f"Config: soft_path_noAuto_syn0_noHier")
print(f"Ethical dimensions: {ETHICAL_DIMENSIONS}")
print(f"Age threshold: {AGE_THRESHOLD} (soft sigmoid)")
print(f"Risk threshold: {RISK_THRESHOLD} (soft sigmoid)")
print("=" * 80)

print("\nLoading data...")
(patient_df, df_train, df_train_scaled,
 df_test, df_test_scaled,
 overall_ranking, overall_pairs,
 all_train_pairs_full, all_test_pairs_full,
 train_rankings, test_rankings,
 train_aut_pairs, train_rec_prior, *_) = init_data()

feature_cols = FEATURE_COLS
df_test_orig = df_test.reset_index(drop=True)

# Background and eval setup
np.random.seed(SEED)
bg_idx = np.random.choice(len(df_train_scaled),
                          size=min(N_BACKGROUND, len(df_train_scaled)),
                          replace=False)
background_tensor = torch.tensor(
    df_train_scaled.iloc[bg_idx][feature_cols].values, dtype=torch.float32)
eval_X_tensor = torch.tensor(
    df_test_scaled[feature_cols].values, dtype=torch.float32)

print(f"  Features: {len(feature_cols)}")
print(f"  Background patients: {len(bg_idx)}")
print(f"  Eval patients (test): {len(df_test_scaled)}")
print(f"  Total entities to process: {len(ALL_DOCTORS)}")
print(f"     - {len(ALL_HUMAN_DOCTORS)} human doctors")
print(f"     - {len(LEGACY_LLMS)} legacy LLMs: {LEGACY_LLMS}")
print(f"     - {len(NEW_LLMS)} new LLMs: {NEW_LLMS}")


def get_entity_type(key):
    """Return 'human', 'llm_legacy', or 'llm_new'."""
    if key in LEGACY_LLMS:
        return 'llm_legacy'
    if key in NEW_LLMS:
        return 'llm_new'
    return 'human'


def safe_get_group(key):
    """get_group() may not work for LLMs - return 'llm_legacy'/'llm_new' instead."""
    try:
        return get_group(key)
    except Exception:
        return get_entity_type(key)


# ----------------------------------------------------------------------------
all_profiles  = []
all_shap_data = {}
all_ranknet_scores = []   # NEW: RankNet score per patient per rater

for i, doctor_key in enumerate(ALL_DOCTORS, 1):
    entity_type = get_entity_type(doctor_key)
    print(f"\n[{i}/{len(ALL_DOCTORS)}] {doctor_key}  ({entity_type})")
    try:
        train_pairs, test_pairs = load_ranked_pairs(doctor_key, RESULTS_DIR)
    except Exception as e:
        print(f"   skip: could not load pairs ({e})")
        continue
    if not train_pairs:
        print(f"   skip: no train pairs")
        continue

    print(f"   training ({len(train_pairs)} pairs)...")
    model = train_one_recipe(
        train_pairs, feature_cols, df_train_scaled,
        soft_labels=True, weight_by_confidence=True,
        num_epochs=NUM_EPOCHS, seed=SEED,
    )
    model.eval()

    # ---- NEW: extract RankNet score per patient (for Figure 5) ----
    try:
        import torch as _torch
        with _torch.no_grad():
            _rn = model(eval_X_tensor).squeeze().detach().numpy()
        _pids = df_test_orig['patient_num'].values if 'patient_num' in df_test_orig.columns else df_test_orig.index.values
        for _pid, _sc in zip(_pids, _rn):
            all_ranknet_scores.append({'doctor': doctor_key, 'entity_type': entity_type,
                                       'patient_num': int(_pid), 'ranknet_score': float(_sc)})
    except Exception as _e:
        print(f"   [warn] could not extract RankNet scores: {_e}")

    print(f"   computing SHAP on {len(eval_X_tensor)} test patients...")
    try:
        explainer = shap.GradientExplainer(model, background_tensor)
        shap_vals = explainer.shap_values(eval_X_tensor)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        if hasattr(shap_vals, 'detach'):
            shap_vals = shap_vals.detach().numpy()
        shap_vals = np.asarray(shap_vals)
        if shap_vals.ndim == 3 and shap_vals.shape[-1] == 1:
            shap_vals = shap_vals.squeeze(-1)
    except Exception as e:
        print(f"   SHAP failed: {e}")
        print(f"   fallback: gradient-based attribution...")
        try:
            eval_X_grad = eval_X_tensor.clone().requires_grad_(True)
            scores = model(eval_X_grad).squeeze()
            shap_list = []
            for j in range(len(eval_X_grad)):
                model.zero_grad()
                if eval_X_grad.grad is not None:
                    eval_X_grad.grad.zero_()
                scores[j].backward(retain_graph=(j < len(eval_X_grad) - 1))
                shap_list.append(eval_X_grad.grad[j].detach().numpy().copy())
            shap_vals = np.array(shap_list) * eval_X_tensor.detach().numpy()
        except Exception as e2:
            print(f"   fallback failed: {e2}")
            continue

    # ---- Aggregate to ethical values ----
    ethical_shap = aggregate_shap_to_ethical_values(
        shap_vals, feature_cols, df_test_orig)
    # mean per dimension
    mean_per_dim = ethical_shap.mean(axis=0)

    profile = {
        'doctor':       doctor_key,
        'entity_type':  entity_type,
        'group':        safe_get_group(doctor_key),
    }
    for dim_name, val in zip(ETHICAL_DIMENSIONS, mean_per_dim):
        profile[dim_name] = val
    all_profiles.append(profile)

    all_shap_data[doctor_key] = {
        'shap_per_patient':         shap_vals,
        'ethical_shap_per_patient': ethical_shap,
        'feature_cols':             feature_cols,
    }

    print(f"   profile: " + "  ".join(
        f"{dim[:13]}={val:+.3f}"
        for dim, val in zip(ETHICAL_DIMENSIONS, mean_per_dim)))

# ----------------------------------------------------------------------------
df_profiles = pd.DataFrame(all_profiles)
os.makedirs(OUT_DIR, exist_ok=True)

csv_path = os.path.join(OUT_DIR, 'ethical_value_profiles_per_doctor.csv')
df_profiles.to_csv(csv_path, index=False)
print(f"\n✓ Saved -> {csv_path}")

import pandas as _pd
_rn_path = os.path.join(OUT_DIR, 'ranknet_scores_per_rater.csv')
_pd.DataFrame(all_ranknet_scores).to_csv(_rn_path, index=False)
print(f"\u2713 Saved RankNet scores -> {_rn_path}  ({len(all_ranknet_scores)} rows)")

pkl_path = os.path.join(OUT_DIR, 'ethical_shap_per_patient.pkl')
with open(pkl_path, 'wb') as f:
    pickle.dump(all_shap_data, f)
print(f"✓ Saved -> {pkl_path}")

# ----------------------------------------------------------------------------
# DISPLAY
# ----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("ETHICAL VALUE PROFILES")
print("=" * 80)
if df_profiles.empty:
    print("  (no profiles - all SHAP failed)")
else:
    print(df_profiles.round(3).to_string(index=False))

    # Summary by entity type (human vs LLM)
    print("\n" + "=" * 80)
    print("MEAN BY ENTITY TYPE (human vs LLM)")
    print("=" * 80)
    entity_summary = df_profiles.groupby('entity_type')[ETHICAL_DIMENSIONS].mean().round(3)
    print(entity_summary.to_string())

    # Summary by group (specialty)
    print("\n" + "=" * 80)
    print("MEAN BY GROUP (specialty / LLM source)")
    print("=" * 80)
    group_summary = df_profiles.groupby('group')[ETHICAL_DIMENSIONS].mean().round(3)
    print(group_summary.to_string())

    # Most prominent ethical value per doctor
    print("\n" + "=" * 80)
    print("DOMINANT ETHICAL VALUE PER DOCTOR")
    print("=" * 80)
    for _, row in df_profiles.iterrows():
        dim_vals = {d: row[d] for d in ETHICAL_DIMENSIONS}
        sorted_dims = sorted(dim_vals.items(), key=lambda x: -abs(x[1]))
        print(f"  {row['doctor']:<20} ({row['entity_type']:<12}, {row['group']:<15})")
        print(f"     " + ", ".join(f"{d}={v:+.3f}" for d, v in sorted_dims))

    # Variability split by entity type
    print("\n" + "=" * 80)
    print("VARIABILITY: humans alone vs LLMs alone")
    print("=" * 80)
    print(f"  {'Dimension':<25} {'SD humans':>12} {'SD llms':>10}")
    humans_df = df_profiles[df_profiles['entity_type'] == 'human']
    llms_df   = df_profiles[df_profiles['entity_type'].str.startswith('llm')]
    for dim in ETHICAL_DIMENSIONS:
        sd_h = humans_df[dim].std() if not humans_df.empty else float('nan')
        sd_l = llms_df[dim].std()   if not llms_df.empty   else float('nan')
        print(f"  {dim:<25} {sd_h:>12.3f} {sd_l:>10.3f}")

    # Distance: LLM mean vs humans mean
    if not llms_df.empty and not humans_df.empty:
        print("\n" + "=" * 80)
        print("HUMAN-LLM DIVERGENCE (per dimension)")
        print("=" * 80)
        print(f"  {'Dimension':<25} {'humans':>10} {'LLMs':>10} {'diff':>10}")
        for dim in ETHICAL_DIMENSIONS:
            h_mean = humans_df[dim].mean()
            l_mean = llms_df[dim].mean()
            diff = l_mean - h_mean
            print(f"  {dim:<25} {h_mean:>+10.3f} {l_mean:>+10.3f} {diff:>+10.3f}")

print("\n" + "=" * 80)
print("DONE - ready for Stage 2 (spider plots) and Analysis B (differential)")
print("=" * 80)