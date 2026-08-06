"""
==========================================================================
run_recipe_ablation_v2.py   (v3 with auto-detected hierarchy pairs)
==========================================================================
Enhanced recipe ablation that:
  1) Adds a new CONFIG axis: include_hierarchy_pairs (on/off).
     When ON, the code AUTO-IDENTIFIES recommendations that have a
     "structural problem" in the dataset (high prevalence + rarely-alone
     OR isolated co-occurrence pattern) and adds synthetic hierarchy
     pairs that force the model to learn the correct clinical ordering.

  2) Auto-detection uses TWO INDEPENDENT criteria:
        INFLATES:  prev * (1 - pct_alone) > 0.08
                   - rec is common AND rarely appears alone
                   - this rec saturates training pairs
        ISOLATED:  pct_alone < 20% AND n >= 30
                   - rec is rarely seen in isolation
                   - model cannot learn its true effect

     Union of the two flag-sets is the list to "break".

  3) For each flagged rec, creates pairs against the IMMEDIATE LEVEL
     ABOVE in the clinical hierarchy:
        rec at level 1 -> partner = rec5 (representative level-2)
        rec at level 2 -> partner = rec10 (representative level-3)
        rec at level 3 -> skip (already top)
     With 3 anchor patients per pair (configurable).

  4) Records all the same rich metrics as v2 (SHAP quality, clinical
     rules, interactions) + a composite score.

The auto-detection is fully data-driven: change the dataset, and the
problematic recs will be re-identified.
==========================================================================
"""
from __future__ import annotations
import os, sys, json, warnings, time
from datetime import datetime
from pathlib import Path
from collections import Counter
from itertools import combinations, product

import numpy as np
import pandas as pd
import torch
import shap as _shap  # import once at top to avoid '_to_dlpack' issue

warnings.filterwarnings("ignore")
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from data_loading import init_data
from top7_clinical_metric import compute_top7_clinical
from utils import (
    FEATURE_COLS, PATH_FEATURE_NAMES, PATH_MAPPING,
    ACTIVE_RECS, ALL_RECS, align_feature_columns,
)
from llm_helpers import ALL_HUMAN_DOCTORS, LEGACY_LLMS, load_ranked_pairs, get_group
from ranknet import RankNet
from metrics import pairwise_accuracy, calc_auc

# ==========================================================================
# Embedded helpers (formerly in run_recipe_ablation.py - V1)
# Inlined here to make V2 standalone (V1 may not exist in user's setup)
# ==========================================================================
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch.nn as nn

class FlexListwiseDataset(Dataset):
    def __init__(self, ranked_pairs, patient_df, feature_cols,
                 pairs_per_batch=32, soft_labels=True):
        self.ranked_pairs   = ranked_pairs
        self.feature_cols   = feature_cols
        self.pairs_per_batch = pairs_per_batch
        self.soft_labels    = soft_labels

        df_aligned = patient_df.copy()
        # Make sure every feature_col is present (zero-fill if missing)
        for c in feature_cols:
            if c not in df_aligned.columns:
                df_aligned[c] = 0.0

        self._feat_lookup = {}
        for _, row in df_aligned.iterrows():
            pid = int(row['patient_num'])
            self._feat_lookup[pid] = row[feature_cols].values.astype(np.float32)

        self._batches = [
            ranked_pairs[i:i + pairs_per_batch]
            for i in range(0, len(ranked_pairs), pairs_per_batch)
        ]

    def _label_for(self, conf):
        if self.soft_labels:
            c = int(round(float(conf)))
            c = max(1, min(6, c))
            return 0.55 + (c - 1) * (1.00 - 0.55) / 5.0
        else:
            return 1.0

    def __len__(self): return len(self._batches)

    def __getitem__(self, idx):
        chunk = self._batches[idx]
        unique = []
        loc = {}
        for (a, b), _ in chunk:
            a, b = int(a), int(b)
            if a not in loc:
                loc[a] = len(unique); unique.append(a)
            if b not in loc:
                loc[b] = len(unique); unique.append(b)
        feats = np.stack([self._feat_lookup[p] for p in unique])
        pair_idx = []
        conf_list = []
        labels = []
        for (a, b), conf in chunk:
            a, b = int(a), int(b)
            pair_idx.append([loc[a], loc[b]])
            conf_list.append(float(conf))
            labels.append(self._label_for(conf))
        return {
            'patient_features': torch.tensor(feats,    dtype=torch.float32),
            'pair_indices':     torch.tensor(pair_idx, dtype=torch.long),
            'confidences':      torch.tensor(conf_list, dtype=torch.float32),
            'labels':           torch.tensor(labels,    dtype=torch.float32),
        }


# ==========================================================================
# Train RankNet with a given recipe
# ==========================================================================

def train_one_recipe(
    train_pairs, feature_cols, df_train_for_model,
    soft_labels=True,
    weight_by_confidence=False,
    num_epochs=300, hidden_dim=128, lr=1e-3, seed=42,
):
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    dataset = FlexListwiseDataset(
        train_pairs, df_train_for_model, feature_cols,
        pairs_per_batch=32, soft_labels=soft_labels,
    )
    model = RankNet(input_dim=len(feature_cols), hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    idx_all = list(range(len(dataset)))
    if len(idx_all) < 2:
        return model
    train_idx, val_idx = train_test_split(idx_all, test_size=0.2, random_state=seed)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2,
    )
    best_val, no_imp = float('inf'), 0
    for epoch in range(num_epochs):
        model.train()
        np.random.shuffle(train_idx)
        for b in train_idx:
            batch = dataset[b]
            feats = batch['patient_features']
            pidx  = batch['pair_indices']
            labels = batch['labels']
            confs  = batch['confidences']
            scores = model(feats).squeeze(-1)
            diff = scores[pidx[:, 0]] - scores[pidx[:, 1]]
            loss_vec = criterion(diff, labels)
            if weight_by_confidence:
                w = confs.detach()
                loss = (loss_vec * w).mean()
            else:
                loss = loss_vec.mean()
            loss = loss + 0.001 * (scores ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # val
        model.eval()
        vloss = 0.0; n = 0
        with torch.no_grad():
            for b in val_idx:
                batch = dataset[b]
                feats  = batch['patient_features']
                pidx   = batch['pair_indices']
                labels = batch['labels']
                scores = model(feats).squeeze(-1)
                diff = scores[pidx[:, 0]] - scores[pidx[:, 1]]
                lv = criterion(diff, labels)
                vloss += lv.mean().item() * labels.size(0)
                n += labels.size(0)
        avg_val = vloss / max(1, n)
        scheduler.step(avg_val)

        if best_val - avg_val > 1e-5:
            best_val, no_imp = avg_val, 0
        else:
            no_imp += 1
            if no_imp >= 5:
                break

    return model

def score_patients(model, patient_df, feature_cols):
    df = patient_df.copy()
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    model.eval()
    pids = df['patient_num'].values
    X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    with torch.no_grad():
        s = model(X).squeeze().numpy()
    return dict(zip(pids, s))


# ==========================================================================
# Synthetic pair builder, parameterised by cap (returns also extended df)
# ==========================================================================

def build_synth(df_train, df_train_scaled, cap, target=None, seed=42):
    """cap=0 means no synthetic pairs at all; returns ([], df_train_scaled)."""
    if cap == 0:
        return [], df_train_scaled
    if target is None:
        target = max(15 * cap, 1)  # roughly 15 active recs

    df_train_scaled_idx = df_train_scaled.set_index('patient_num')
    df_train_orig_idx   = df_train.set_index('patient_num')

    young  = df_train_orig_idx.index[df_train_orig_idx['age'] < 45].tolist()
    middle = df_train_orig_idx.index[(df_train_orig_idx['age'] >= 45) & (df_train_orig_idx['age'] < 70)].tolist()
    older  = df_train_orig_idx.index[df_train_orig_idx['age'] >= 70].tolist()
    rng = np.random.default_rng(seed)
    base_pids = (
        rng.choice(young,  size=min(33, len(young)),  replace=False).tolist() +
        rng.choice(middle, size=min(34, len(middle)), replace=False).tolist() +
        rng.choice(older,  size=min(33, len(older)),  replace=False).tolist()
    )

    real_combos = set()
    for pid in df_train_orig_idx.index:
        p_recs = [r for r in ACTIVE_RECS if df_train_orig_idx.loc[pid, r] == 1]
        if len(p_recs) >= 2:
            for combo in combinations(sorted(p_recs), 2):
                real_combos.add(combo)
    extra_combos = []
    for partner in ['rec1','rec2','rec5','rec10','rec18']:
        extra_combos.append(tuple(sorted(['rec13', partner])))
        extra_combos.append(tuple(sorted(['rec17', partner])))
    for partner in ['rec1','rec5','rec19']:
        extra_combos.append(tuple(sorted(['rec8',  partner])))
        extra_combos.append(tuple(sorted(['rec11', partner])))
    extra_combos = sorted(set(extra_combos))
    extra_combos = [c for c in extra_combos if c not in real_combos]
    all_combos = sorted(real_combos) + extra_combos

    def make_synth(base_row, recs_to_set, next_pid):
        new_row = base_row.copy()
        new_row['patient_num'] = next_pid
        for r in ALL_RECS:
            if r in new_row.index: new_row[r] = 0.0
        for p in PATH_FEATURE_NAMES:
            if p in new_row.index: new_row[p] = 0.0
        for r in recs_to_set: new_row[r] = 1.0
        paths_active = set()
        for r in recs_to_set:
            for path_name, recs_in_path in PATH_MAPPING.items():
                if r in recs_in_path: paths_active.add(path_name)
        for p in paths_active: new_row[p] = 1.0
        return new_row

    synth_pts, pool = [], []
    next_pid = int(df_train_scaled['patient_num'].max()) + 100000
    for base_pid in base_pids:
        base_row = df_train_scaled_idx.loc[base_pid].copy()
        base_row['patient_num'] = base_pid
        empty = make_synth(base_row, [], next_pid); empty_pid = next_pid
        synth_pts.append(empty); next_pid += 1
        for rec in ACTIVE_RECS:
            rp = make_synth(base_row, [rec], next_pid); rec_pid = next_pid
            synth_pts.append(rp); next_pid += 1
            pool.append((((int(rec_pid), int(empty_pid)), 6), [rec]))
    for base_pid in base_pids:
        base_row = df_train_scaled_idx.loc[base_pid].copy()
        base_row['patient_num'] = base_pid
        for (a, b) in all_combos:
            both = make_synth(base_row, [a, b], next_pid); pid_both = next_pid
            synth_pts.append(both); next_pid += 1
            pa = make_synth(base_row, [a], next_pid); pid_a = next_pid
            synth_pts.append(pa); next_pid += 1
            pb = make_synth(base_row, [b], next_pid); pid_b = next_pid
            synth_pts.append(pb); next_pid += 1
            pool.append((((int(pid_both), int(pid_a)), 6), [b]))
            pool.append((((int(pid_both), int(pid_b)), 6), [a]))

    rng2 = np.random.default_rng(seed)
    idx = list(range(len(pool))); rng2.shuffle(idx)
    selected = []
    counts = Counter()
    for i in idx:
        if len(selected) >= target: break
        pair, recs_used = pool[i]
        if any(counts[r] + 1 > cap for r in recs_used): continue
        selected.append(pair)
        for r in recs_used: counts[r] += 1

    df_synth = pd.DataFrame(synth_pts)
    df_ext = pd.concat([df_train_scaled, df_synth], ignore_index=True)
    return selected, df_ext


# ==========================================================================
# 24-config iterator
# ==========================================================================
CONFIGS = list(product(
    [False, True],    # soft_labels: hard, soft
    [False, True],    # include_paths
    [True, False],    # include_auto_pairs (the legacy behavior was True)
    [0, 5, 10],       # synth cap
))
# Each config -> (soft, path, auto, syn_cap)

def config_name(soft, path, auto, syn_cap):
    parts = [
        'soft' if soft else 'hard',
        'path' if path else 'noPath',
        'auto' if auto else 'noAuto',
        f'syn{syn_cap}',
    ]
    return '_'.join(parts)


# ==========================================================================
# Main
# ==========================================================================

# Alias for compat with old code
_orig_config_name = config_name



# ==========================================================================
# Clinical hierarchy (matches the value_mapping content)
# ==========================================================================
HIERARCHY_LEVELS = {
    3: ['rec10', 'rec11', 'rec12', 'rec13',         # treatments
        'rec16', 'rec17'],                           # clinical consultations
    2: ['rec2', 'rec3', 'rec5', 'rec6'],             # advanced diagnostics
    1: ['rec1', 'rec4', 'rec8',                      # basic lab + measurement
        'rec18', 'rec19', 'rec21'],                  # lifestyle (incl. dietitian) + records
}
REC_TO_LEVEL = {r: lvl for lvl, recs in HIERARCHY_LEVELS.items() for r in recs}

# Representative partner per level (chosen for clinical meaningfulness)
LEVEL_REPRESENTATIVE = {
    2: 'rec5',    # diagnostic imaging - representative of level 2
    3: 'rec10',   # first-line treatment - representative of level 3
}

# Auto-detection thresholds (defaults; can be overridden)
INFLATES_THRESHOLD = 0.08   # prev * (1 - pct_alone) > this -> inflates
ISOLATED_THRESHOLD = 0.80   # (1 - pct_alone) > this AND n>=30 -> isolated

# Top-5 categorization for SHAP quality
TREATMENTS_AND_CONSULTS = ['rec10', 'rec11', 'rec12', 'rec13',
                           'rec16', 'rec17']  # rec18 reclassified as lifestyle

# Clinical rules for SHAP hierarchy checks (carried over from v2)
CLINICAL_RULES = {
    'rule1_replace_below_advanced':
        {'lhs': 'rec13', 'rhs': ['rec11', 'rec12']},
    'rule2_consult_below_treatment':
        {'lhs': 'rec16', 'rhs': ['rec10', 'rec11']},
    'rule3_lab_below_treatment':
        {'lhs': 'rec1',  'rhs': ['rec11', 'rec12']},
}


# ==========================================================================
# Auto-detect problematic recs
# ==========================================================================
def auto_detect_problematic_recs(df, active_recs=ACTIVE_RECS,
                                   inflates_thresh=INFLATES_THRESHOLD,
                                   isolated_thresh=ISOLATED_THRESHOLD,
                                   isolated_min_n=30,
                                   verbose=True):
    """
    Identify recommendations whose dataset structure makes them
    structurally problematic for learning isolated effects.

    Returns: dict with 'flagged' list, plus detail per rec.
    """
    rows = []
    for r in active_recs:
        if r not in df.columns:
            continue
        n = int(df[r].sum())
        if n == 0:
            continue
        prevalence = n / len(df)
        other_cols = [x for x in active_recs if x != r and x in df.columns]
        only_r = int(((df[r] == 1) & (df[other_cols].sum(axis=1) == 0)).sum())
        pct_alone = only_r / n if n > 0 else 1.0

        inflates_score = prevalence * (1 - pct_alone)
        isolated_score = (1 - pct_alone) if n >= isolated_min_n else 0.0

        inflates = inflates_score >= inflates_thresh
        isolated = isolated_score >= isolated_thresh
        flagged = inflates or isolated

        rows.append({
            'rec': r,
            'n': n,
            'prevalence': prevalence,
            'pct_alone': pct_alone,
            'inflates_score': inflates_score,
            'isolated_score': isolated_score,
            'inflates': inflates,
            'isolated': isolated,
            'flagged': flagged,
        })

    df_det = pd.DataFrame(rows)
    flagged = sorted(df_det[df_det['flagged']]['rec'].tolist())

    if verbose:
        print("\n" + "=" * 80)
        print("AUTO-DETECTING problematic recommendations")
        print("=" * 80)
        print(f"{'rec':<6} {'n':>4} {'prev':>6} {'alone':>7} {'INFL':>7} {'ISO':>7}   flags")
        print("-" * 80)
        df_show = df_det.sort_values('inflates_score', ascending=False)
        for _, row in df_show.iterrows():
            flag_str = []
            if row['inflates']: flag_str.append('INFLATES')
            if row['isolated']: flag_str.append('ISOLATED')
            flag_str = ' | '.join(flag_str) if flag_str else 'fine'
            print(f"{row['rec']:<6} {row['n']:>4} {row['prevalence']*100:5.1f}% "
                  f"{row['pct_alone']*100:6.1f}% {row['inflates_score']:>7.3f} "
                  f"{row['isolated_score']:>7.3f}   {flag_str}")
        print(f"\n=> {len(flagged)} recs to break: {flagged}")
    return {'flagged': flagged, 'detail': df_det}


# ==========================================================================
# Build hierarchy pairs (synthetic data)
# ==========================================================================
def _make_synthetic_patient(base_row, recs_to_set, next_pid):
    """Create a synthetic patient row with only the given recs."""
    new_row = base_row.copy()
    new_row['patient_num'] = next_pid
    for r in ALL_RECS:
        if r in new_row.index:
            new_row[r] = 0.0
    for p in PATH_FEATURE_NAMES:
        if p in new_row.index:
            new_row[p] = 0.0
    for r in recs_to_set:
        new_row[r] = 1.0
    paths_active = set()
    for r in recs_to_set:
        for path_name, recs_in_path in PATH_MAPPING.items():
            if r in recs_in_path:
                paths_active.add(path_name)
    for p in paths_active:
        new_row[p] = 1.0
    return new_row


def build_hierarchy_pairs(df_train_scaled,
                          flagged_recs,
                          n_anchors=3,
                          confidence=4,
                          seed=42,
                          verbose=True):
    """
    For each flagged rec, create a synthetic pair against its
    representative partner at the immediate higher hierarchy level.

    Returns: (list of pairs, DataFrame of synthetic patients to add to train).
    """
    rng = np.random.default_rng(seed)
    df_idx = df_train_scaled.set_index('patient_num')
    all_pids = df_train_scaled['patient_num'].values

    # Select anchors
    n_anchors_eff = min(n_anchors, len(all_pids))
    anchor_pids = rng.choice(all_pids, size=n_anchors_eff, replace=False)

    # Plan: for each flagged rec, find the partner one level up
    plan = []
    for r in flagged_recs:
        lvl = REC_TO_LEVEL.get(r)
        if lvl is None:
            continue
        next_lvl = lvl + 1
        partner = LEVEL_REPRESENTATIVE.get(next_lvl)
        if partner is None:
            continue  # rec is at top level, nothing to break
        plan.append((r, partner, lvl, next_lvl))

    if verbose:
        print("\n" + "=" * 60)
        print(f"Hierarchy pair plan (anchors={n_anchors_eff}):")
        print("=" * 60)
        for r, p, lvl, nl in plan:
            print(f"  {r:<6} (level {lvl}) < {p:<6} (level {nl})")

    if not plan:
        return [], pd.DataFrame()

    synth_rows = []
    synth_pairs = []
    next_pid = int(df_train_scaled['patient_num'].max()) + 300000

    for anchor_pid in anchor_pids:
        base_row = df_idx.loc[anchor_pid].copy()
        base_row['patient_num'] = anchor_pid
        for (rec_lo, rec_hi, _, _) in plan:
            # Build patient with only rec_lo
            p_lo = _make_synthetic_patient(base_row, [rec_lo], next_pid)
            pid_lo = next_pid; synth_rows.append(p_lo); next_pid += 1
            # Build patient with only rec_hi
            p_hi = _make_synthetic_patient(base_row, [rec_hi], next_pid)
            pid_hi = next_pid; synth_rows.append(p_hi); next_pid += 1
            # Pair: hi wins over lo
            synth_pairs.append(((int(pid_hi), int(pid_lo)), confidence))

    synth_df = pd.DataFrame(synth_rows)
    if verbose:
        print(f"=> Created {len(synth_pairs)} hierarchy pairs "
              f"({len(synth_rows)} new patients)")
    return synth_pairs, synth_df


# ==========================================================================
# New CONFIGS list with hierarchy axis
# ==========================================================================
CONFIGS_V3 = list(product(
    [False, True],     # soft_labels
    [False, True],     # include_paths
    [True, False],     # include_auto_pairs
    [0, 5, 10],        # syn_cap (A+B synth pairs)
    [False, True],     # include_hierarchy_pairs
))
# Each config tuple: (soft, path, auto, syn_cap, hier)


def config_name_v3(soft, path, auto, syn_cap, hier):
    parts = [
        'soft' if soft else 'hard',
        'path' if path else 'noPath',
        'auto' if auto else 'noAuto',
        f'syn{syn_cap}',
        'hier' if hier else 'noHier',
    ]
    return '_'.join(parts)


# ==========================================================================
# SHAP quality (carried from v2)
# ==========================================================================
def compute_normalized_shap_per_rec(model, df_test_scaled, fcols, patient_df,
                                     n_background=128):
    """For each active rec, compute mean(|SHAP|) over patients who have the rec."""
    df_test_aligned = align_feature_columns(df_test_scaled.copy())
    test_pids = df_test_aligned['patient_num'].values
    X_test = df_test_aligned[fcols].to_numpy(np.float32)

    rng_bg = np.random.default_rng(42)
    bg_idx = rng_bg.choice(len(X_test), size=min(n_background, len(X_test)),
                            replace=False)
    bg_t = torch.tensor(X_test[bg_idx], dtype=torch.float32)
    test_t = torch.tensor(X_test, dtype=torch.float32)

    explainer = _shap.GradientExplainer(model, bg_t)
    phi = np.array(explainer.shap_values(test_t))
    if phi.ndim == 3 and phi.shape[2] == 1:
        phi = phi[..., 0]

    df_shap = pd.DataFrame(phi, index=test_pids, columns=fcols)
    ptd = patient_df.set_index('patient_num')

    out = {}
    for rec in ACTIVE_RECS:
        if rec not in df_shap.columns:
            continue
        common = df_shap.index.intersection(ptd.index)
        active_mask = ptd.loc[common, rec] >= 0.5
        if active_mask.sum() == 0:
            continue
        path_for_rec = None
        for path_name, recs_in_path in PATH_MAPPING.items():
            if rec in recs_in_path and path_name in df_shap.columns:
                path_for_rec = path_name
        rec_shap = df_shap.loc[common, rec].loc[active_mask]
        if path_for_rec is not None:
            rec_shap = rec_shap + df_shap.loc[common, path_for_rec].loc[active_mask]
        out[rec] = {
            'mean_abs_shap': float(rec_shap.abs().mean()),
            'mean_signed_shap': float(rec_shap.mean()),
            'n_active': int(active_mask.sum()),
        }
    return out


def compute_shap_quality_metrics(rec_shap_dict):
    """Derive quality metrics from normalized SHAP per rec."""
    if not rec_shap_dict:
        return {}
    n_pos = sum(1 for v in rec_shap_dict.values() if v['mean_signed_shap'] > 0)
    ranked = sorted(rec_shap_dict.items(),
                    key=lambda x: x[1]['mean_abs_shap'], reverse=True)
    rec_to_rank = {r: i+1 for i, (r, _) in enumerate(ranked)}

    top5 = [r for r, _ in ranked[:5]]
    n_treat_consult_top5 = sum(1 for r in top5 if r in TREATMENTS_AND_CONSULTS)
    rec1_rank = rec_to_rank.get('rec1', None)
    rec1_in_top5 = 'rec1' in top5

    rules_results = {}
    for rule_name, rule in CLINICAL_RULES.items():
        lhs = rule['lhs']
        rhs_list = rule['rhs']
        if lhs not in rec_shap_dict:
            rules_results[rule_name] = None
            continue
        all_rhs_present = all(r in rec_shap_dict for r in rhs_list)
        if not all_rhs_present:
            rules_results[rule_name] = None
            continue
        lhs_imp = rec_shap_dict[lhs]['mean_abs_shap']
        passed = all(
            lhs_imp < rec_shap_dict[r]['mean_abs_shap']
            for r in rhs_list
        )
        rules_results[rule_name] = bool(passed)

    n_rules_passed = sum(1 for v in rules_results.values() if v is True)

    return {
        'n_pos_normalized':       n_pos,
        'top5_recs':              ','.join(top5),
        'n_treat_consult_top5':   n_treat_consult_top5,
        'rec1_rank':              rec1_rank,
        'rec1_in_top5':           rec1_in_top5,
        'n_rules_passed':         n_rules_passed,
        **rules_results,
    }


# ==========================================================================
# Interaction metric (carried from v2)
# ==========================================================================
def compute_interaction_metric(model, df_test_scaled, fcols,
                                df_train_for_combos, max_combos=15,
                                max_anchors_per_combo=8, seed=42):
    df_idx = df_train_for_combos.set_index('patient_num')
    real_combos = set()
    for pid in df_idx.index:
        p_recs = [r for r in ACTIVE_RECS
                  if r in df_idx.columns and df_idx.loc[pid, r] == 1]
        if len(p_recs) >= 2:
            for combo in combinations(sorted(p_recs), 2):
                real_combos.add(combo)
    real_combos = sorted(real_combos)
    if max_combos and len(real_combos) > max_combos:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(real_combos), size=max_combos, replace=False)
        real_combos = [real_combos[i] for i in idx]

    df_test_aligned = align_feature_columns(df_test_scaled.copy())
    test_idx = df_test_aligned.set_index('patient_num')

    interactions = []
    for (rec_a, rec_b) in real_combos:
        if rec_a not in test_idx.columns or rec_b not in test_idx.columns:
            continue
        mask = (test_idx[rec_a] == 1) | (test_idx[rec_b] == 1)
        anchor_pids = test_idx[mask].index.tolist()
        if not anchor_pids:
            continue
        if len(anchor_pids) > max_anchors_per_combo:
            rng = np.random.default_rng(seed)
            anchor_pids = rng.choice(anchor_pids,
                                     size=max_anchors_per_combo, replace=False)
        for pid in anchor_pids:
            base_row = test_idx.loc[pid].copy()
            f_vals = {}
            for a_on, b_on in product([False, True], [False, True]):
                v = base_row.copy()
                v[rec_a] = 1.0 if a_on else 0.0
                v[rec_b] = 1.0 if b_on else 0.0
                for path_name, recs_in_path in PATH_MAPPING.items():
                    if path_name not in fcols: continue
                    active_in_path = [r for r in recs_in_path
                                       if r in v.index and v[r] >= 0.5]
                    v[path_name] = 1.0 if active_in_path else 0.0
                x = torch.tensor(v[fcols].values.astype(np.float32)).unsqueeze(0)
                with torch.no_grad():
                    f_vals[(a_on, b_on)] = float(model(x).squeeze())
            interaction = (f_vals[(True, True)] - f_vals[(True, False)]
                          - f_vals[(False, True)] + f_vals[(False, False)])
            interactions.append(interaction)

    if not interactions:
        return {'pct_positive_interactions': None,
                'n_positive_interactions': 0,
                'n_interactions_total': 0,
                'mean_interaction': None}
    interactions = np.array(interactions)
    n_pos = int((interactions > 0.01).sum())
    return {
        'pct_positive_interactions': float(n_pos / len(interactions)),
        'n_positive_interactions':   n_pos,
        'n_interactions_total':       len(interactions),
        'mean_interaction':           float(interactions.mean()),
    }


# ==========================================================================
# Main
# ==========================================================================
class _LightSkip(Exception):
    """Raised in light mode to skip the expensive SHAP/top7/interaction metrics."""
    pass


def _light_postprocess(df_full, out_dir):
    """AUC-only post-processing: refill ablation tables A1/A3 and the
    per-rater best-config held-out AUC (the manuscript's selection criterion)."""
    out_dir = Path(out_dir)
    PHYS_GROUPS = ('family_medicine', 'public_health')
    df_full = df_full.copy()
    df_full['is_phys'] = df_full['group'].isin(PHYS_GROUPS)

    # ---- Table A1: top configs by mean held-out AUC ----
    a1 = (df_full.groupby('config')
                 .apply(lambda g: pd.Series({
                     'mean_auc_phys': g.loc[g.is_phys, 'test_auc'].mean(),
                     'mean_acc_phys': g.loc[g.is_phys, 'test_accuracy'].mean(),
                     'mean_auc_all':  g['test_auc'].mean(),
                     'n_raters':      g['doctor'].nunique(),
                 }))
                 .reset_index()
                 .sort_values('mean_auc_phys', ascending=False))
    a1.head(12).round(4).to_csv(out_dir / 'ablation_A1_top_configs.csv', index=False)

    # ---- Table A3: best config per rater, chosen by held-out AUC ----
    idx = df_full.groupby('doctor')['test_auc'].idxmax()
    a3 = (df_full.loc[idx, ['doctor', 'group', 'config', 'test_auc', 'test_accuracy']]
                 .sort_values(['group', 'test_auc'], ascending=[True, False])
                 .reset_index(drop=True))
    a3.round(4).to_csv(out_dir / 'ablation_A3_best_per_rater.csv', index=False)
    a3.round(4).to_csv(out_dir / 'model_fit_metrics.csv', index=False)

    phys = a3[a3['group'].isin(PHYS_GROUPS)]
    mods = a3[~a3['group'].isin(PHYS_GROUPS)]
    print("\n" + "=" * 72)
    print("LIGHT ABLATION  -  best held-out AUC per rater (48-config sweep)")
    print("=" * 72)
    print(a3.to_string(index=False))
    print(f"\n  physicians mean best AUC = {phys['test_auc'].mean():.3f}")
    print(f"  models     mean best AUC = {mods['test_auc'].mean():.3f}")
    print(f"\n  A1 top config (mean physician AUC): {a1.iloc[0]['config']}  "
          f"({a1.iloc[0]['mean_auc_phys']:.3f})")
    print("  Saved: ablation_A1_top_configs.csv, ablation_A3_best_per_rater.csv, model_fit_metrics.csv")
    return df_full, a1, a3


def main(out_dir='results', doctors_to_run=None, num_epochs=300, seed=42,
         configs_to_run=None, hier_anchors=3, light=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    if doctors_to_run is None:
        doctors_to_run = ALL_HUMAN_DOCTORS
    if configs_to_run is None:
        configs_to_run = CONFIGS_V3

    print("=" * 72)
    print(f"RECIPE ABLATION v3 - {len(configs_to_run)} configs x {len(doctors_to_run)} doctors")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    print("\n[1/5] init_data ...")
    (
        patient_df, df_train, df_train_scaled, df_test, df_test_scaled,
        overall_ranking, overall_pairs,
        all_train_pairs_full, all_test_pairs_full, train_rankings, test_rankings,
        train_aut_pairs, train_rec_prior,
        exclude_pairs, train_graph, test_graph, patient_df_scaled,
    ) = init_data()

    print("\n[2/5] Auto-detect problematic recs from training data...")
    detection = auto_detect_problematic_recs(df_train, verbose=True)
    flagged_recs = detection['flagged']

    print("\n[3/5] Build hierarchy pairs once (will be added to syn pool per config)...")
    hier_pairs, hier_synth_df = build_hierarchy_pairs(
        df_train_scaled, flagged_recs,
        n_anchors=hier_anchors, confidence=4, seed=seed, verbose=True,
    )
    # df extended with hierarchy patients
    df_train_with_hier = pd.concat([df_train_scaled, hier_synth_df],
                                    ignore_index=True)

    print("\n[4/5] Building (A+B) synth pools for each cap...")
    synth_cache = {}
    for cap in [0, 5, 10]:
        sel, ext = build_synth(df_train, df_train_scaled, cap=cap, seed=seed)
        synth_cache[cap] = (sel, ext)
        print(f"  cap={cap}: {len(sel)} A+B pairs")

    fcols_with    = FEATURE_COLS
    fcols_no_path = [c for c in FEATURE_COLS if not c.startswith('path_')]

    results_dir = SCRIPT_DIR / 'physicians_results'
    rows = []

    print(f"\n[5/5] Running {len(configs_to_run)} configs x {len(doctors_to_run)} doctors")
    t_start = time.time()
    n_total = len(configs_to_run) * len(doctors_to_run)
    done_count = 0

    for di, doctor_key in enumerate(doctors_to_run, 1):
        train_pairs, test_pairs = load_ranked_pairs(doctor_key, results_dir)
        if not train_pairs or not test_pairs:
            continue
        print(f"\n--- {di}/{len(doctors_to_run)}: {doctor_key} ---")

        for ci, (soft, path, auto, syn_cap, hier) in enumerate(configs_to_run, 1):
            name = config_name_v3(soft, path, auto, syn_cap, hier)
            fcols = fcols_with if path else fcols_no_path
            synth_pairs, df_ext_synth = synth_cache[syn_cap]

            combined = list(train_pairs)
            if syn_cap > 0:
                combined += list(synth_pairs)
            if auto:
                combined += list(train_aut_pairs) + list(train_rec_prior)
            if hier:
                combined += list(hier_pairs)

            # Choose the training df: if either synth A+B or hier are present,
            # we need the extended df. Merge them properly.
            if hier and syn_cap > 0:
                # Need both: extend the A+B df with the hier patients
                df_train_for_this = pd.concat([df_ext_synth, hier_synth_df],
                                              ignore_index=True)
            elif hier:
                df_train_for_this = df_train_with_hier
            elif syn_cap > 0:
                df_train_for_this = df_ext_synth
            else:
                df_train_for_this = df_train_scaled

            row = {
                'doctor':        doctor_key,
                'group':         get_group(doctor_key),
                'config':        name,
                'soft_labels':   soft,
                'include_paths': path,
                'include_auto':  auto,
                'syn_cap':       syn_cap,
                'include_hier':  hier,
                'n_hier_pairs':  len(hier_pairs) if hier else 0,
            }
            try:
                # 1) Train
                model = train_one_recipe(
                    combined, fcols, df_train_for_this,
                    soft_labels=soft, weight_by_confidence=(not soft),
                    num_epochs=num_epochs, seed=seed,
                )
                # 2) Standard metrics
                test_sd = score_patients(model, df_test_scaled, fcols)
                row['test_accuracy'] = pairwise_accuracy(test_sd, test_pairs)
                row['test_auc']      = calc_auc(test_sd, test_pairs)

                if light:
                    raise _LightSkip()

                # 3) SHAP quality
                rec_shap = compute_normalized_shap_per_rec(
                    model, df_test_scaled, fcols, patient_df,
                    n_background=128,
                )
                quality = compute_shap_quality_metrics(rec_shap)
                row.update(quality)

                # 4) Interaction metric
                interactions = compute_interaction_metric(
                    model, df_test_scaled, fcols, df_train,
                    max_combos=15, max_anchors_per_combo=8, seed=seed,
                )
                row.update(interactions)

                # 5) top7_clinical metric (reuse SHAP from step 3 if possible)
                # We need the FULL SHAP DataFrame; recompute it here efficiently
                try:
                    df_test_aligned = align_feature_columns(df_test_scaled.copy())
                    X_test_arr = df_test_aligned[fcols].to_numpy(np.float32)
                    rng_t7 = np.random.default_rng(seed)
                    bg_idx_t7 = rng_t7.choice(len(X_test_arr),
                                              size=min(128, len(X_test_arr)),
                                              replace=False)
                    bg_t7 = torch.tensor(X_test_arr[bg_idx_t7], dtype=torch.float32)
                    expl_t7 = _shap.GradientExplainer(model, bg_t7)
                    phi_t7 = np.array(expl_t7.shap_values(
                        torch.tensor(X_test_arr, dtype=torch.float32)))
                    if phi_t7.ndim == 3 and phi_t7.shape[2] == 1:
                        phi_t7 = phi_t7[..., 0]
                    df_shap_t7 = pd.DataFrame(phi_t7,
                                              index=df_test_aligned['patient_num'].values,
                                              columns=fcols)
                    top7_result = compute_top7_clinical(df_shap_t7, patient_df, min_n=10)
                    row['top7_clinical'] = top7_result['top7_clinical']
                    row['n_clinical_in_top7'] = top7_result['n_clinical_in_top7']
                    row['top7_missing'] = ','.join(top7_result['missing'])
                    row['top7_intruders'] = ','.join(top7_result['intruders'])
                except Exception as e_t7:
                    print(f"      WARN top7 calc failed: {e_t7}")
                    row['top7_clinical'] = None
                    row['n_clinical_in_top7'] = None

            except _LightSkip:
                pass
            except Exception as e:
                print(f"  [{ci:02d}] {name}: ERROR {e}")
                rows.append(row)
                done_count += 1
                continue

            done_count += 1
            elapsed = time.time() - t_start
            eta = elapsed / done_count * (n_total - done_count) if done_count else 0
            print(f"  [{ci:02d}/{len(configs_to_run)}] {name:<35}  "
                  f"auc={row.get('test_auc', 0):.3f}  "
                  f"top7={row.get('top7_clinical', 0):.2f}  "
                  f"n_pos={row.get('n_pos_normalized', 0)}/16  "
                  f"rules={row.get('n_rules_passed', '-')}/3  "
                  f"(eta {eta/60:.0f}m)")
            rows.append(row)

    # Save
    df_full = pd.DataFrame(rows)
    _fname = 'ablation_full_auc.csv' if light else 'ablation_full_v3.csv'
    df_full.to_csv(out_dir / _fname, index=False)
    print(f"\n\u2713 Saved {out_dir / _fname} ({len(df_full)} rows)")

    if light:
        return _light_postprocess(df_full, out_dir)

    # Aggregate
    agg = df_full.groupby('config').agg({
        'test_accuracy':              'mean',
        'test_auc':                   'mean',
        'top7_clinical':              'mean',
        'n_clinical_in_top7':         'mean',
        'n_pos_normalized':           'mean',
        'n_treat_consult_top5':       'mean',
        'rec1_rank':                  'mean',
        'rec1_in_top5':               'mean',
        'n_rules_passed':             'mean',
        'pct_positive_interactions':  'mean',
        'doctor':                     'count',
    }).rename(columns={'doctor': 'n_doctors'}).reset_index()

    # Composite (Maya's weights: AUC + top7 dominant, then clinical logic)
    # 0.30 AUC + 0.30 top7_clinical + 0.20 n_pos + 0.15 rules + 0.05 rec1_rank
    agg['composite'] = (
        0.30 * agg['test_auc'].fillna(0) +
        0.30 * agg['top7_clinical'].fillna(0) +
        0.20 * (agg['n_pos_normalized'].fillna(0) / 16.0) +
        0.15 * (agg['n_rules_passed'].fillna(0) / 3.0) +
        0.05 * (agg['rec1_rank'].fillna(0) / 16.0)
    )
    agg = agg.sort_values('composite', ascending=False)
    agg.to_csv(out_dir / 'ablation_summary_v3.csv', index=False)
    print(f"\u2713 Saved {out_dir / 'ablation_summary_v3.csv'}")

    # Print top
    print("\n" + "=" * 72)
    print("Top 10 configs by composite score:")
    print("=" * 72)
    show_cols = ['config', 'test_auc', 'top7_clinical',
                 'n_pos_normalized', 'n_rules_passed',
                 'rec1_rank', 'pct_positive_interactions',
                 'composite']
    print(agg[show_cols].head(10).round(3).to_string(index=False))

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return df_full, agg


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--doctors', nargs='+', default=None)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--out', default='results')
    p.add_argument('--hier_anchors', type=int, default=3)
    args = p.parse_args()
    main(out_dir=args.out, doctors_to_run=args.doctors,
         num_epochs=args.epochs, hier_anchors=args.hier_anchors)