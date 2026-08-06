"""
==========================================================================
RUN ALL EXPERIMENTS — single script that:
  1. Loads data (with the new path features)
  2. Builds the 75 synthetic pairs (cap=5 per active rec)
  3. For each doctor with complete rankings:
       a. Loads the doctor's 4-iteration ranked train pairs
          (concatenated from iter_0..iter_3)
       b. Loads the doctor's test pairs (parts A + B)
       c. Trains a RankNet with the NEW recipe
          (soft labels + path features + no auto pairs + 75 synth pairs)
       d. Computes metrics on train and test
  4. Saves a summary CSV with all metrics across all doctors.

How to run:
  cd <this folder>
  python run_all_experiments.py

Output:
  - results/all_doctors_metrics.csv       <-- the main summary table
  - results/<doctor>_iter_3_metrics.json   <-- per-doctor detailed metrics
  - results/<doctor>_model_final.pth       <-- the trained model
==========================================================================
"""
import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

# ---- make sure we can import the local modules ----
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import (
    FEATURE_COLS, PATH_FEATURE_NAMES, PATH_MAPPING,
    ACTIVE_RECS, ALL_RECS, add_path_features,
)
from data_loading import init_data
from ranknet import run_single_ranknet_experiment

# ==========================================================================
# Doctors with complete rankings (4 train-iter files + 2 test parts)
# ==========================================================================
# Doctors are now defined in llm_helpers.py.
# By default we run all human doctors (10) plus the 3 legacy LLMs.
# ph_5 is included (her test set is a single _test_ranked.json file,
# which is handled by llm_helpers.load_ranked_pairs).
from llm_helpers import (
    ALL_HUMAN_DOCTORS, LEGACY_LLMS, NEW_LLMS,
    load_ranked_pairs as _llm_load_ranked,
    get_group as _llm_get_group,
)

DOCTORS_FULL = ALL_HUMAN_DOCTORS + LEGACY_LLMS + NEW_LLMS
DOCTORS_PARTIAL = []  # excluded: partial data (test set but only 1 train iteration)

# ==========================================================================
# Helpers for loading ranked pair files
# ==========================================================================
def _norm_pair_item(item):
    """Normalises a ranked-pair item to ((a, b), conf).
    Handles legacy formats: [[a,b], conf], [a, b, conf], etc."""
    try:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            inner, conf = item
            if isinstance(inner, (list, tuple)) and len(inner) == 2:
                a, b = inner
                return ((int(a), int(b)), float(conf))
            elif not isinstance(inner, (list, tuple)):
                # legacy: [a, b]
                a, b = item
                return ((int(a), int(b)), 6.0)
        if isinstance(item, (list, tuple)) and len(item) == 3:
            a, b, conf = item
            return ((int(a), int(b)), float(conf))
    except (TypeError, ValueError):
        return None
    return None


def load_doctor_ranked_pairs(doctor_key, results_dir):
    """Wrapper around llm_helpers.load_ranked_pairs to keep this script
    backward-compatible. Handles LLMs (single test file) and ph_5."""
    return _llm_load_ranked(doctor_key, results_dir)


def _legacy_load_doctor_ranked_pairs(doctor_key, results_dir):
    """Loads and concatenates all 4 train-iter ranked files and 2 test parts
    for a doctor. Returns (train_pairs, test_pairs) as lists of
    ((a, b), confidence).
    """
    doc_dir = Path(results_dir) / doctor_key

    # Train pairs: iter 0..3
    train_pairs = []
    for it in range(4):
        path = doc_dir / f"{doctor_key}_train_iter_{it}_ranked.json"
        if not path.exists():
            print(f"  WARNING: missing {path.name}")
            continue
        with open(path) as f:
            data = json.load(f)
        for item in data:
            np_item = _norm_pair_item(item)
            if np_item is not None:
                train_pairs.append(np_item)

    # Test pairs: A + B
    test_pairs = []
    for part in ['A', 'B']:
        path = doc_dir / f"{doctor_key}_test_part_{part}_ranked.json"
        if not path.exists():
            print(f"  WARNING: missing {path.name}")
            continue
        with open(path) as f:
            data = json.load(f)
        for item in data:
            np_item = _norm_pair_item(item)
            if np_item is not None:
                test_pairs.append(np_item)

    return train_pairs, test_pairs


# ==========================================================================
# Build the 75 synthetic pairs (adapted from the new-code recipe)
# ==========================================================================
def build_synthetic_pairs(df_train, df_train_scaled, cap=5, target=75, seed=42):
    """
    Returns (selected_synth_pairs, df_train_scaled_extended).
    Pairs are in ((a, b), conf=6) format. Extended df includes the synthetic
    patients so the model can look them up by patient_num.
    """
    df_train_scaled_idx = df_train_scaled.set_index('patient_num')
    df_train_orig_idx   = df_train.set_index('patient_num')

    # ---- 100 anchors balanced across age strata ----
    young  = df_train_orig_idx.index[df_train_orig_idx['age'] < 45].tolist()
    middle = df_train_orig_idx.index[(df_train_orig_idx['age'] >= 45) & (df_train_orig_idx['age'] < 70)].tolist()
    older  = df_train_orig_idx.index[df_train_orig_idx['age'] >= 70].tolist()

    rng = np.random.default_rng(seed)
    sel_young  = rng.choice(young,  size=min(33, len(young)),  replace=False).tolist() if young else []
    sel_middle = rng.choice(middle, size=min(34, len(middle)), replace=False).tolist() if middle else []
    sel_older  = rng.choice(older,  size=min(33, len(older)),  replace=False).tolist() if older else []
    base_pids = sel_young + sel_middle + sel_older
    print(f"  Anchors: {len(base_pids)} ({len(sel_young)}/{len(sel_middle)}/{len(sel_older)} young/mid/old)")

    # ---- Real combos from train data ----
    real_combos = set()
    for pid in df_train_orig_idx.index:
        p_recs = [r for r in ACTIVE_RECS if df_train_orig_idx.loc[pid, r] == 1]
        if len(p_recs) >= 2:
            for combo in combinations(sorted(p_recs), 2):
                real_combos.add(combo)

    # ---- Additional controlled combos for under-represented recs ----
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
    print(f"  Combos pool: {len(all_combos)} ({len(real_combos)} real + {len(extra_combos)} extra)")

    # ---- Helper to build a synthetic patient row ----
    def make_synth(base_row, recs_to_set, next_pid):
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
        # activate paths
        paths_active = set()
        for r in recs_to_set:
            for path_name, recs_in_path in PATH_MAPPING.items():
                if r in recs_in_path:
                    paths_active.add(path_name)
        for p in paths_active:
            new_row[p] = 1.0
        return new_row

    # ---- Build candidate pool ----
    synthetic_patients = []
    candidate_pool = []  # list of ((pair, conf=6), [recs_exercised_by_this_pair])
    next_pid = int(df_train_scaled['patient_num'].max()) + 100000

    # Type A: rec X alone > empty
    for base_pid in base_pids:
        base_row = df_train_scaled_idx.loc[base_pid].copy()
        base_row['patient_num'] = base_pid
        empty = make_synth(base_row, [], next_pid); empty_pid = next_pid
        synthetic_patients.append(empty); next_pid += 1
        for rec in ACTIVE_RECS:
            rp = make_synth(base_row, [rec], next_pid); rec_pid = next_pid
            synthetic_patients.append(rp); next_pid += 1
            candidate_pool.append((((int(rec_pid), int(empty_pid)), 6), [rec]))

    # Type B: rec X + rec Y > rec X (and > rec Y)
    for base_pid in base_pids:
        base_row = df_train_scaled_idx.loc[base_pid].copy()
        base_row['patient_num'] = base_pid
        for (rec_a, rec_b) in all_combos:
            p_both = make_synth(base_row, [rec_a, rec_b], next_pid); pid_both = next_pid
            synthetic_patients.append(p_both); next_pid += 1
            p_a = make_synth(base_row, [rec_a], next_pid); pid_a = next_pid
            synthetic_patients.append(p_a); next_pid += 1
            p_b = make_synth(base_row, [rec_b], next_pid); pid_b = next_pid
            synthetic_patients.append(p_b); next_pid += 1
            candidate_pool.append((((int(pid_both), int(pid_a)), 6), [rec_b]))
            candidate_pool.append((((int(pid_both), int(pid_b)), 6), [rec_a]))

    print(f"  Candidate pool: {len(candidate_pool)} pairs from {len(synthetic_patients)} synthetic patients")

    # ---- Greedy-select with cap ----
    rng2 = np.random.default_rng(seed)
    indices = list(range(len(candidate_pool)))
    rng2.shuffle(indices)
    selected = []
    counts = Counter()
    for idx in indices:
        if len(selected) >= target:
            break
        pair, trained_recs = candidate_pool[idx]
        if any(counts[r] + 1 > cap for r in trained_recs):
            continue
        selected.append(pair)
        for r in trained_recs:
            counts[r] += 1

    print(f"  Selected {len(selected)} synthetic pairs (cap={cap})")
    print(f"  Per-rec coverage: {dict(sorted(counts.items()))}")

    # ---- Extend df_train_scaled ----
    df_synth = pd.DataFrame(synthetic_patients)
    df_extended = pd.concat([df_train_scaled, df_synth], ignore_index=True)
    return selected, df_extended


# ==========================================================================
# Run experiment for one doctor
# ==========================================================================
def run_doctor(doctor_key, train_pairs, test_pairs,
               selected_synth_pairs,
               df_train_scaled_extended, df_test_scaled,
               seed=42, num_epochs=500, hidden_dim=128):
    """Trains the RankNet model and computes pairwise metrics."""
    # The synthetic pairs are added to the doctor's training pairs
    combined_train = list(train_pairs) + list(selected_synth_pairs)
    print(f"  Training on {len(train_pairs)} doctor pairs + "
          f"{len(selected_synth_pairs)} synth pairs = {len(combined_train)} total")
    print(f"  Test pairs: {len(test_pairs)}")

    # We need to package the pairs into the dict format that
    # run_single_ranknet_experiment expects.
    # Train metrics are computed against the doctor's own train pairs only
    # (not the synthetic ones), to reflect doctor agreement.
    all_train_pairs = {doctor_key: list(train_pairs)}
    all_test_pairs  = {doctor_key: list(test_pairs)}
    # Empty rankings dicts because we use llm=True (skip ndcg/tau/map/rbo)
    train_rankings = {}
    test_rankings  = {}

    model, metrics = run_single_ranknet_experiment(
        doctor_key=doctor_key,
        chosen_train_pairs=combined_train,
        all_train_pairs=all_train_pairs,
        all_test_pairs=all_test_pairs,
        df_train_scaled=df_train_scaled_extended,
        df_test_scaled=df_test_scaled,
        train_rankings=train_rankings,
        test_rankings=test_rankings,
        seed=seed,
        hidden_dim=hidden_dim,
        lr=1e-3,
        num_epochs=num_epochs,
        print_flag=False,
        recs_only=True,
        llm=True,           # skip ndcg/tau/map/rbo (no rankings)
        weighted_metics=False,
    )
    return model, metrics


# ==========================================================================
# Main
# ==========================================================================
def main(out_dir='results', doctors_to_run=None, num_epochs=500):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    # ---- Defaults ----
    if doctors_to_run is None:
        doctors_to_run = DOCTORS_FULL

    print("=" * 72)
    print("RUN ALL EXPERIMENTS")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Doctors: {doctors_to_run}")
    print(f"Output dir: {out_dir}")
    print("=" * 72)

    # ---- Step 1: Load data (with the new path features) ----
    print("\n[1/4] Loading data via init_data() ...")
    (
        patient_df,
        df_train, df_train_scaled,
        df_test,  df_test_scaled,
        overall_ranking, overall_pairs,
        all_train_pairs, all_test_pairs,
        train_rankings, test_rankings,
        train_aut_pairs, train_rec_prior,
        exclude_pairs, train_graph, test_graph,
        patient_df_scaled,
    ) = init_data()

    n_paths = sum(1 for c in df_train_scaled.columns if c.startswith('path_'))
    print(f"  df_train: {df_train.shape}, df_train_scaled: {df_train_scaled.shape}")
    print(f"  df_test: {df_test.shape}, df_test_scaled: {df_test_scaled.shape}")
    print(f"  Path features in df_train_scaled: {n_paths} (expected 6)")
    print(f"  len(FEATURE_COLS): {len(FEATURE_COLS)} (expected 29)")
    assert n_paths == 6, f"Path features missing! got {n_paths}, expected 6"
    assert len(FEATURE_COLS) == 29, f"FEATURE_COLS wrong! got {len(FEATURE_COLS)}, expected 29"

    # ---- Step 2: Build the 75 synthetic pairs ----
    print("\n[2/4] Building synthetic pairs ...")
    selected_synth_pairs, df_train_scaled_extended = build_synthetic_pairs(
        df_train, df_train_scaled, cap=5, target=75, seed=42,
    )

    # ---- Step 3: Loop over doctors ----
    print("\n[3/4] Training one model per doctor ...")
    results_rows = []
    results_dir = SCRIPT_DIR / 'physicians_results'

    for i, doctor_key in enumerate(doctors_to_run, 1):
        print(f"\n--- Doctor {i}/{len(doctors_to_run)}: {doctor_key} ---")

        train_pairs, test_pairs = load_doctor_ranked_pairs(doctor_key, results_dir)
        if len(train_pairs) == 0 or len(test_pairs) == 0:
            print(f"  SKIPPING {doctor_key}: train={len(train_pairs)}, test={len(test_pairs)}")
            continue

        try:
            model, metrics = run_doctor(
                doctor_key=doctor_key,
                train_pairs=train_pairs,
                test_pairs=test_pairs,
                selected_synth_pairs=selected_synth_pairs,
                df_train_scaled_extended=df_train_scaled_extended,
                df_test_scaled=df_test_scaled,
                seed=42,
                num_epochs=num_epochs,
                hidden_dim=128,
            )
        except Exception as e:
            print(f"  ERROR training {doctor_key}: {e}")
            import traceback; traceback.print_exc()
            continue

        print(f"  Train acc: {metrics.get('train_accuracy'):.3f}  | "
              f"Test acc: {metrics.get('test_accuracy'):.3f}")
        print(f"  Train AUC: {metrics.get('train_auc'):.3f}  | "
              f"Test AUC:  {metrics.get('test_auc'):.3f}")

        # Save per-doctor outputs
        torch.save(model.state_dict(), out_dir / f"{doctor_key}_model_final.pth")
        with open(out_dir / f"{doctor_key}_metrics.json", 'w') as f:
            json.dump({
                'doctor': doctor_key,
                'n_train_pairs_doctor': len(train_pairs),
                'n_synth_pairs': len(selected_synth_pairs),
                'n_test_pairs': len(test_pairs),
                'metrics': metrics,
            }, f, indent=2, default=str)

        results_rows.append({
            'doctor': doctor_key,
            'n_train_pairs_doctor': len(train_pairs),
            'n_synth_pairs': len(selected_synth_pairs),
            'n_test_pairs': len(test_pairs),
            'train_accuracy': metrics.get('train_accuracy'),
            'test_accuracy':  metrics.get('test_accuracy'),
            'train_auc':      metrics.get('train_auc'),
            'test_auc':       metrics.get('test_auc'),
        })

    # ---- Step 4: Save summary CSV ----
    print("\n[4/4] Saving summary ...")
    if results_rows:
        df_sum = pd.DataFrame(results_rows)
        out_csv = out_dir / 'all_doctors_metrics.csv'
        df_sum.to_csv(out_csv, index=False)
        print(f"\nSaved: {out_csv}")
        print("\n" + "=" * 72)
        print("SUMMARY")
        print("=" * 72)
        print(df_sum.to_string(index=False))
        print()
        print(f"Mean train accuracy: {df_sum['train_accuracy'].mean():.3f}")
        print(f"Mean test accuracy:  {df_sum['test_accuracy'].mean():.3f}")
        print(f"Mean train AUC:      {df_sum['train_auc'].mean():.3f}")
        print(f"Mean test AUC:       {df_sum['test_auc'].mean():.3f}")
    else:
        print("WARNING: no doctors were successfully processed!")

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    # Default: run all doctors with full rankings, 500 epochs.
    # Override with command-line args if you want to run a subset or fewer epochs.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--doctors', nargs='+', default=None,
                        help='Specific doctor keys to run (default: all 9 full).')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Max training epochs (default: 500).')
    parser.add_argument('--out', default='results',
                        help='Output directory (default: results).')
    args = parser.parse_args()

    main(out_dir=args.out,
         doctors_to_run=args.doctors,
         num_epochs=args.epochs)
