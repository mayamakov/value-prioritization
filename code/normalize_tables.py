#!/usr/bin/env python3
"""
normalize_tables.py — Stage C of the pipeline.

Z-scores every profile table per axis across the 26 raters (within group where
grouped), backing up each raw table to *_raw.csv first. Extracted verbatim from
the original regenerate_everything_v10.py normalize block so the numbers match
the manuscript exactly.
"""
import os
import pandas as pd
import paper_normalize_v10 as _norm

HERE = os.path.dirname(os.path.abspath(__file__)) or "."
RESULTS = os.path.join(HERE, "results")
DIMS = ['Beneficence-Immediate', 'Beneficence-LongTerm',
        'Non-maleficence', 'Autonomy', 'Justice']


def _backup(path, df):
    raw = path.replace('.csv', '_raw.csv')
    if not os.path.exists(raw):
        df.to_csv(raw, index=False)


def normalize_self(filename, cols, groupby=None):
    path = os.path.join(RESULTS, filename)
    if not os.path.exists(path):
        print(f'  [NORM-MISS] {filename}'); return
    df = pd.read_csv(path); cols = [c for c in cols if c in df.columns]
    if not cols:
        print(f'  [NORM-SKIP] {filename}'); return
    _backup(path, df)
    if groupby and groupby in df.columns:
        for _, idx in df.groupby(groupby).groups.items():
            df.loc[idx, cols] = _norm.apply_params(
                df.loc[idx], _norm.fit_params(df.loc[idx], cols), cols)[cols].values
    else:
        df = _norm.apply_params(df, _norm.fit_params(df, cols), cols)
    df.to_csv(path, index=False)
    print(f'  [NORM-OK] {filename} ({len(cols)} cols)')


def normalize_seed_with_cp():
    cp = os.path.join(RESULTS, 'analysis2_common_pairs_profiles.csv')
    sd = os.path.join(RESULTS, 'seed_profile_stability.csv')
    if not (os.path.exists(cp) and os.path.exists(sd)):
        print('  [NORM-MISS] seed (needs common-pairs)'); return
    raw_cp = cp.replace('.csv', '_raw.csv')
    cpdf = pd.read_csv(raw_cp if os.path.exists(raw_cp) else cp)
    params = _norm.fit_params(cpdf, DIMS)
    phys = _norm.centroid(_norm.apply_params(cpdf, params, DIMS), 'doctor', _norm.HUMANS, DIMS)
    sddf = pd.read_csv(sd); _backup(sd, sddf)
    sddf = _norm.apply_params(sddf, params, DIMS)
    if 'distance_from_physician_centroid' in sddf.columns:
        sddf['distance_from_physician_centroid'] = _norm.distances(sddf, phys, DIMS)
    sddf.to_csv(sd, index=False)
    print('  [NORM-OK] seed (cp params + physician centroid)')


def main():
    # fresh-raw guard: remove stale *_raw.csv so we never read yesterday's data
    import glob
    for f in glob.glob(os.path.join(RESULTS, '*_raw.csv')):
        try: os.remove(f)
        except OSError: pass
    normalize_self('ethical_value_profiles_per_doctor.csv', DIMS)
    normalize_self('analysis2_common_pairs_profiles.csv',   DIMS)
    normalize_self('analysis1_topN_profiles.csv',           DIMS, groupby='top_N')
    normalize_self('ethical_profiles_combined_top30.csv',   ['Pairs_' + d for d in DIMS])
    normalize_self('ethical_profiles_combined_top30.csv',   ['Top30_' + d for d in DIMS])
    normalize_seed_with_cp()


if __name__ == "__main__":
    main()
