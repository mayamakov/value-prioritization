#!/usr/bin/env python3
"""
run_to_run_agreement.py — stability of each LLM across independent rollouts

Each LLM was asked to rank the same common pairs in three independent rollouts
(the *_seed1 / *_seed2 / *_seed3 folders under data/physicians_results). This
script recovers each rollout's pairwise choices, compares every pair of rollouts
for the same model, and reports the fraction of common pairs on which the two
rollouts picked the same patient.

This quantifies run-to-run stochasticity: how much of a model's operational
value signature is a stable property of the model, and how much is sampling
noise. It regenerates results/multi_seed_consistency.csv for all 16 LLMs.

Outputs
-------
    results/multi_seed_consistency.csv   one row per (model, rollout_a, rollout_b)
    printed summary: per-model mean agreement, median across models, range
"""
import os
import json
import itertools
from pathlib import Path

import pandas as pd

from panel_config import LLMS, DISPLAY, LEGACYSET

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
if not RESULTS.is_dir():
    RESULTS = HERE.parent / "results"
PR = HERE / "physicians_results"
if not PR.is_dir():                       # running outside run_all.py
    PR = HERE.parent / "data" / "physicians_results"

N_SEEDS = 3


def _norm(entry):
    """Pull the ordered patient pair out of one ranked-JSON record."""
    if isinstance(entry, dict):
        for k in ("ranking", "ranked", "order", "result"):
            if k in entry and isinstance(entry[k], (list, tuple)):
                entry = entry[k]
                break
        else:
            return None
    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        try:
            return [int(entry[0]), int(entry[1])]
        except (TypeError, ValueError):
            return None
    return None


def _load(path):
    if not path.exists():
        return []
    try:
        raw = json.load(open(path))
    except (json.JSONDecodeError, OSError):
        return []
    out = []
    for e in raw:
        p = _norm(e)
        if p:
            out.append(p)
    return out


def choices(folder, key, seed):
    """sorted-pair -> winning patient id, for one rollout folder.

    Rollout folders store a single file, {key}_seed{N}_common_pairs.json,
    holding a list of [[winner, loser], score] records over the common pairs.
    """
    f = folder / f"{key}_seed{seed}_common_pairs.json"
    if not f.exists():                       # tolerate the ranked-json layout too
        prs = []
        for i in range(4):
            prs += _load(folder / f"{key}_train_iter_{i}_ranked.json")
        for part in "AB":
            prs += _load(folder / f"{key}_test_part_{part}_ranked.json")
        prs += _load(folder / f"{key}_test_ranked.json")
        return {tuple(sorted(p)): p[0] for p in prs}
    try:
        raw = json.load(open(f))
    except (json.JSONDecodeError, OSError):
        return {}
    out = {}
    for rec in raw:
        pair = rec[0] if isinstance(rec, (list, tuple)) and rec else None
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            try:
                a, b = int(pair[0]), int(pair[1])
            except (TypeError, ValueError):
                continue
            out[tuple(sorted((a, b)))] = a      # first element is the prioritized patient
    return out


def main():
    RESULTS.mkdir(exist_ok=True)
    rows = []

    for key in LLMS:
        label = DISPLAY.get(key, key)
        rollouts = {}
        for s in range(1, N_SEEDS + 1):
            folder = PR / f"{key}_seed{s}"
            if folder.is_dir():
                c = choices(folder, key, s)
                if c:
                    rollouts[s] = c
        if len(rollouts) < 2:
            print(f"  [skip] {label}: {len(rollouts)} rollout(s) found")
            continue
        for a, b in itertools.combinations(sorted(rollouts), 2):
            ca, cb = rollouts[a], rollouts[b]
            shared = set(ca) & set(cb)
            if not shared:
                continue
            same = sum(1 for p in shared if ca[p] == cb[p])
            rows.append({
                "model": key,
                "label": label,
                "seed_a": a,
                "seed_b": b,
                "n_pairs": len(shared),
                "agreement": same / len(shared),
            })

    if not rows:
        raise SystemExit("no rollout pairs found — check data/physicians_results/*_seed*")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "multi_seed_consistency.csv", index=False)

    per = df.groupby("label")["agreement"].mean().sort_values()

    print("=" * 70)
    print("Run-to-run agreement across independent rollouts")
    print(f"  models with >=2 rollouts : {per.size}")
    print(f"  rollout pairs compared   : {len(df)}")
    print("=" * 70)
    print(per.round(3).to_string())
    print("-" * 70)
    print(f"  median across all models : {per.median():.3f}")
    print(f"  range                    : {per.min():.3f} - {per.max():.3f}")
    # the manuscript separates the single outlier from the rest
    outlier = per.idxmin()
    rest = per.drop(outlier)
    print(f"  excluding {outlier}: median {rest.median():.3f}, "
          f"range {rest.min():.3f}-{rest.max():.3f}; {outlier} = {per.min():.3f}")
    print("=" * 70)
    print("  wrote results/multi_seed_consistency.csv")


if __name__ == "__main__":
    main()
