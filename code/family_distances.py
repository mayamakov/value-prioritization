#!/usr/bin/env python3
"""
family_distances.py — developer-family similarity of operational value signatures

Computes the claim reported in the Results section: models from the same
developer have more similar operational value signatures than models from
different developers.

For every pair of LLMs we take the Euclidean distance between their normalized
five-dimensional value signatures (results/analysis2_common_pairs_profiles.csv),
then split those pairs into within-developer and across-developer sets.

Two conventions for the across-developer set are reported, because they answer
slightly different questions:

  A. all pairs
     Every cross-developer pair counts, including pairs involving developers
     represented by a single model (DeepSeek, Qwen, Mistral).

  B. multi-model developers only
     Restricted to developers contributing more than one model, so the same
     set of models appears on both sides of the comparison. This is the
     like-for-like contrast: a single-model developer can never contribute a
     within-developer pair, so including it only on the across side inflates
     the denominator with models that cannot be balanced.

Convention B is the one that supports the reported fold-separation. Both are
printed so the choice is explicit and checkable.

Outputs
-------
    results/family_distances.csv          per-pair distances with labels
    results/family_distances_summary.csv  the summary statistics
    (both printed to stdout)
"""
import os
import itertools
import numpy as np
import pandas as pd

from panel_config import LLMS, MODEL_FAMILY, DISPLAY

HERE = os.path.dirname(os.path.abspath(__file__)) or "."
RESULTS = os.path.join(HERE, "results")

DIMS = ['Beneficence-Immediate', 'Beneficence-LongTerm',
        'Non-maleficence', 'Autonomy', 'Justice']

PROFILE = os.path.join(RESULTS, 'analysis2_common_pairs_profiles.csv')


def main():
    if not os.path.exists(PROFILE):
        raise SystemExit(f"missing {PROFILE} — run stage A first (rebuild_pair_csvs.py, "
                         "stage2_top_and_pairs.py) and normalize_tables.py")

    df = pd.read_csv(PROFILE)
    key = 'doctor' if 'doctor' in df.columns else df.columns[0]
    df = df.set_index(key)

    models = [k for k in LLMS if k in df.index]
    if len(models) < 2:
        raise SystemExit("fewer than two LLMs found in the profile table")

    vec = {k: df.loc[k, DIMS].to_numpy(dtype=float) for k in models}
    fam = {k: MODEL_FAMILY[k] for k in models}

    n_per_family = pd.Series(list(fam.values())).value_counts()
    multi = set(n_per_family[n_per_family > 1].index)

    rows = []
    for a, b in itertools.combinations(models, 2):
        rows.append({
            'model_a': DISPLAY.get(a, a),
            'model_b': DISPLAY.get(b, b),
            'family_a': fam[a],
            'family_b': fam[b],
            'same_developer': fam[a] == fam[b],
            'both_multi_model': fam[a] in multi and fam[b] in multi,
            'distance': float(np.linalg.norm(vec[a] - vec[b])),
        })
    pairs = pd.DataFrame(rows)
    pairs.to_csv(os.path.join(RESULTS, 'family_distances.csv'), index=False)

    within = pairs.loc[pairs.same_developer, 'distance']
    across_all = pairs.loc[~pairs.same_developer, 'distance']
    across_multi = pairs.loc[~pairs.same_developer & pairs.both_multi_model, 'distance']

    summary = pd.DataFrame([
        {'set': 'within-developer', 'n_pairs': len(within),
         'mean_distance': within.mean(), 'sd': within.std()},
        {'set': 'across-developer (A: all pairs)', 'n_pairs': len(across_all),
         'mean_distance': across_all.mean(), 'sd': across_all.std()},
        {'set': 'across-developer (B: multi-model developers only)', 'n_pairs': len(across_multi),
         'mean_distance': across_multi.mean(), 'sd': across_multi.std()},
    ])
    summary.to_csv(os.path.join(RESULTS, 'family_distances_summary.csv'), index=False)

    print('=' * 74)
    print('Developer-family distances between operational value signatures')
    print('  source :', os.path.basename(PROFILE), '(normalized, 5 axes)')
    print(f'  models : {len(models)} across {n_per_family.size} developers '
          f'({", ".join(sorted(multi))} contribute >1 model)')
    print('=' * 74)
    print(summary.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print('-' * 74)
    print(f'  fold separation, convention A : {across_all.mean() / within.mean():.2f}x')
    print(f'  fold separation, convention B : {across_multi.mean() / within.mean():.2f}x')
    print('-' * 74)
    print('  Reported in the manuscript: mean pairwise Euclidean distance')
    print(f'    within-developer  {within.mean():.2f}')
    print(f'    across-developer  {across_multi.mean():.2f}   (convention B)')
    print(f'    separation        {across_multi.mean() / within.mean():.1f}-fold')
    print('=' * 74)
    print('  wrote results/family_distances.csv and results/family_distances_summary.csv')


if __name__ == '__main__':
    main()
