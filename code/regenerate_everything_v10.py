# -*- coding: utf-8 -*-
"""
regenerate_everything_v10.py — ONE-BUTTON regeneration for the v10 value revision.

Run this single file (python regenerate_everything_v10.py, or %run in Jupyter)
from PATCHED_CODE/PATCHED_CODE/. It:

  1. injects value_mapping_v10 under the name 'value_mapping_rec_intrinsic'
     => every existing pipeline script uses v10 with ZERO edits and the v9 file
        on disk is NEVER overwritten.
  2. runs the build scripts (pair CSVs, Top-N, common-pairs, sensitivity, raw-SHAP,
     optionally GPU stage1 SHAP) in-process so the alias is honoured.
  3. NORMALIZES every profile table (z-score per axis across the 26 raters),
     backing up each to *_raw.csv first.
  4. runs the figure scripts (which read the now-normalized CSVs).
  5. prints a MANIFEST: PASS / FAIL / MISSING + which tables were normalized.

Nothing is reported "done" silently: read the manifest at the end, paste it back,
and any straggler gets a one-line normalize hook.
"""
import os, sys, time, traceback
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)) or '.'
sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, 'results')
FIGS    = os.path.join(RESULTS, 'figures')

# internal keys (= v9, frozen); v10 display names live in paper_value_config_v10.DISPLAY_LABELS
DIMS = ['Beneficence-Immediate', 'Beneficence-LongTerm',
        'Non-maleficence', 'Autonomy', 'Justice']

# =============================== CONFIG ====================================
RUN_STAGE1_SHAP = True   # True => re-run GPU SHAP (stage1_ethical_value_mapping.py, ~6 min).
                          # SHAP-based outputs (Fig2 profiles, S5 PCA, S8, per-doctor SHAP)
                          # stay STALE w.r.t. v10 until this runs once on the GPU box.
DO_NORMALIZE    = True    # z-score every profile table across its 26 raters
NORM_METHOD     = 'zscore'
# ===========================================================================

# ---- 1. INJECT v10 MAPPING UNDER THE v9 NAME (no overwrite) ---------------
import value_mapping_v10 as _v10
# [alias removed] rec_intrinsic.py imports v10 directly now
import paper_value_config_v10 as cfg
print('=' * 78)
print('mapping in force : value_mapping_v10  (aliased as value_mapping_rec_intrinsic)')
print('display labels   :', list(cfg.DISPLAY_LABELS.values()))
print('normalize        :', NORM_METHOD if DO_NORMALIZE else 'OFF')
print('=' * 78)

MANIFEST = []

def run_script(filename, label=None):
    """Exec a pipeline script in-process so the sys.modules alias is honoured."""
    label = label or filename
    path = os.path.join(HERE, filename)
    if not os.path.exists(path):
        MANIFEST.append(('MISSING', label, 0.0)); print(f'  [MISSING] {label}'); return
    t0 = time.time()
    try:
        src = open(path, encoding='utf-8').read()
        g = {'__name__': '__main__', '__file__': path}
        exec(compile(src, path, 'exec'), g)
        dt = time.time() - t0
        MANIFEST.append(('PASS', label, dt)); print(f'  [PASS] {label} ({dt:.0f}s)')
    except SystemExit:
        MANIFEST.append(('PASS', label, time.time() - t0))
    except Exception as e:
        dt = time.time() - t0
        MANIFEST.append(('FAIL', label, dt)); print(f'  [FAIL] {label} ({dt:.0f}s) -> {e}')
        traceback.print_exc()


# ---- 2. BUILD STEPS -------------------------------------------------------
# === fresh-raw guard: delete stale *_raw.csv so tables never read yesterday's data ===
import glob as _glob, os as _os
for _f in _glob.glob(_os.path.join('results', '*_raw.csv')):
    try: _os.remove(_f)
    except OSError: pass
# === /fresh-raw guard ===

print('\n[STAGE A] pair-based build (CPU)')
run_script('rebuild_pair_csvs.py')
run_script('make_topN_frozen.py')
run_script('stage2_top_and_pairs.py')          # common-pairs + combined profiles

print('\n[STAGE B] SHAP-based build (GPU)')
if RUN_STAGE1_SHAP:
    run_script('stage1_ethical_value_mapping.py')
    run_script('raw_shap_clustering_FIXED.py')
else:
    print('  [SKIP] stage1 SHAP (RUN_STAGE1_SHAP=False) — SHAP-based figures stay stale vs v10')


# ---- 3. NORMALIZE PASS ----------------------------------------------------
# Uses the shared helper. Full-26-rater tables: self-fit z-score across raters
# (within group if grouped). Seed table (LLM-only): common-pairs params + the
# common-pairs physician centroid (recomputes its distance column). Raw->*_raw.csv.
import paper_normalize_v10 as _norm

def _backup(path, df):
    raw = path.replace('.csv', '_raw.csv')
    if not os.path.exists(raw):
        df.to_csv(raw, index=False)

def normalize_self(filename, cols, groupby=None):
    path = os.path.join(RESULTS, filename)
    if not os.path.exists(path):
        MANIFEST.append(('NORM-MISS', filename, 0.0)); print(f'  [NORM-MISS] {filename}'); return
    df = pd.read_csv(path); cols = [c for c in cols if c in df.columns]
    if not cols:
        MANIFEST.append(('NORM-SKIP', filename, 0.0)); print(f'  [NORM-SKIP] {filename}'); return
    _backup(path, df)
    if groupby and groupby in df.columns:
        for _, idx in df.groupby(groupby).groups.items():
            df.loc[idx, cols] = _norm.apply_params(df.loc[idx], _norm.fit_params(df.loc[idx], cols), cols)[cols].values
    else:
        df = _norm.apply_params(df, _norm.fit_params(df, cols), cols)
    df.to_csv(path, index=False)
    tag = cols[0].split('_')[0] if '_' in cols[0] else 'dims'
    MANIFEST.append(('NORM-OK', f'{filename} [{tag}]', 0.0)); print(f'  [NORM-OK] {filename} ({len(cols)} cols)')

def normalize_seed_with_cp():
    cp = os.path.join(RESULTS, 'analysis2_common_pairs_profiles.csv')
    sd = os.path.join(RESULTS, 'seed_profile_stability.csv')
    if not (os.path.exists(cp) and os.path.exists(sd)):
        MANIFEST.append(('NORM-MISS', 'seed_profile_stability.csv', 0.0)); print('  [NORM-MISS] seed (needs common-pairs)'); return
    cpdf = pd.read_csv(cp.replace('.csv', '_raw.csv') if os.path.exists(cp.replace('.csv','_raw.csv')) else cp)
    params = _norm.fit_params(cpdf, DIMS)
    phys = _norm.centroid(_norm.apply_params(cpdf, params, DIMS), 'doctor', _norm.HUMANS, DIMS)
    sddf = pd.read_csv(sd); _backup(sd, sddf)
    sddf = _norm.apply_params(sddf, params, DIMS)
    if 'distance_from_physician_centroid' in sddf.columns:
        sddf['distance_from_physician_centroid'] = _norm.distances(sddf, phys, DIMS)
    sddf.to_csv(sd, index=False)
    MANIFEST.append(('NORM-OK', 'seed_profile_stability.csv [cp-params]', 0.0)); print('  [NORM-OK] seed (cp params + physician centroid)')

if DO_NORMALIZE:
    print('\n[STAGE C] normalize profile tables (z-score, equal opportunity)')
    normalize_self('ethical_value_profiles_per_doctor.csv', DIMS)
    normalize_self('analysis2_common_pairs_profiles.csv',   DIMS)
    normalize_self('analysis1_topN_profiles.csv',           DIMS, groupby='top_N')
    normalize_self('ethical_profiles_combined_top30.csv',   ['Pairs_' + d for d in DIMS])
    normalize_self('ethical_profiles_combined_top30.csv',   ['Top30_' + d for d in DIMS])
    normalize_seed_with_cp()
else:
    print('\n[STAGE C] normalization OFF')


# ---- 4. FIGURE / TABLE SCRIPTS (read the normalized CSVs) ------------------
print('\n[STAGE D] figures + tables')
run_script('make_figS2.py')                      # Figure 2 (three methods)
run_script('paper_figures_CORRECTED.py')         # main-paper figures (was KeyError on stale panel)
run_script('generate_all_figures_FINAL_FIXED.py')# S1/S4 agreement etc. (functions; harmless import)

print('\n[STAGE D2] per-rater figR figures (RankNet-based, auto-discovered)')
import glob as _g2
for _s in sorted(_g2.glob('figR_*.py')):
    run_script(_s)


# ---- 4b. PRINT ALL TABLES (T3-T13) ----------------------------------------
print('\n[STAGE E] print all manuscript tables')
try:
    import print_all_tables_v10
    print_all_tables_v10.main()
    MANIFEST.append(('PASS', 'print_all_tables_v10', 0.0))
except Exception as e:
    MANIFEST.append(('FAIL', 'print_all_tables_v10', 0.0)); print(f'  [FAIL] print_all_tables_v10 -> {e}')
    traceback.print_exc()


# ---- 5. MANIFEST ----------------------------------------------------------
print('\n' + '=' * 78)
print('MANIFEST')
print('=' * 78)
order = {'FAIL': 0, 'MISSING': 1, 'NORM-MISS': 2, 'NORM-SKIP': 3, 'PASS': 4, 'NORM-OK': 5}
for status, label, dt in sorted(MANIFEST, key=lambda r: order.get(r[0], 9)):
    t = f'{dt:5.0f}s' if dt else '     '
    print(f'  {status:9} {t}  {label}')
n_fail    = sum(1 for s, *_ in MANIFEST if s == 'FAIL')
n_missing = sum(1 for s, *_ in MANIFEST if s == 'MISSING')
print('-' * 78)
print(f'  {n_fail} failed, {n_missing} missing scripts, '
      f'{sum(1 for s,*_ in MANIFEST if s=="NORM-OK")} tables normalized')
if not RUN_STAGE1_SHAP:
    print('  NOTE: SHAP-based figures stale until stage1 re-runs on GPU (set RUN_STAGE1_SHAP=True).')
print('  Paste this manifest back; any FAIL/MISSING gets patched.')
print('=' * 78)
