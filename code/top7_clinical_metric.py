"""
==========================================================================
top7_clinical_metric.py
==========================================================================
Computes a new diagnostic metric:

    top7_clinical = fraction of clinically-important recs found in the
                    TOP 7 of the SHAP-by-signed-value ranking
                    (using COMBINED rec+path SHAP, normalized over patients
                    who got the rec)

Clinically-important recs (7 total):
  rec6  - Advanced imaging (CTA / perfusion)
  rec10 - First-line treatment (low-dose statin)
  rec11 - Advanced treatment (PCSK9 / high-dose)
  rec12 - Treatment upgrade
  rec13 - Treatment replacement
  rec16 - Specialist consultation (lipidologist)
  rec17 - Other consultation (hepatology / cardiology)

EXCLUDED (by user request):
  rec18 - Dietitian (RECLASSIFIED as lifestyle - moved to path_lifestyle)
  rec5  - Routine imaging (Doppler) — not as time-critical as CTA

The metric ranges 0 to 1:
  0/7 = none of the important recs are in the top 7
  7/7 = all 7 are in the top 7 (ideal — clinically aligned model)

Two helper outputs:
  - top7_clinical:        the raw fraction
  - missing_from_top7:    which clinical recs failed to make top 7
  - intruders_in_top7:    which non-clinical recs took their place
==========================================================================
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()
sys.path.insert(0, str(SCRIPT_DIR))

from utils import PATH_MAPPING, ACTIVE_RECS


# ==========================================================================
# Configuration: which recs are clinically important
# ==========================================================================
CLINICAL_TOP_RECS = {
    'rec6',   # Advanced imaging (CTA)
    'rec10',  # First-line tx
    'rec11',  # Advanced tx (PCSK9)
    'rec12',  # Treatment upgrade
    'rec13',  # Treatment replacement
    'rec16',  # Specialist consult
    'rec17',  # Other consult
}
N_CLINICAL = len(CLINICAL_TOP_RECS)  # 7


# Map rec -> path_name (for combined SHAP)
REC_TO_PATH = {}
for path_name, recs in PATH_MAPPING.items():
    for r in recs:
        REC_TO_PATH[r] = path_name


def compute_top7_clinical(df_shap, patient_df, min_n=10, verbose=False):
    """
    Computes the top7_clinical metric from a SHAP DataFrame.

    Args:
      df_shap:     DataFrame of SHAP values (index=patient_num, cols=features)
      patient_df:  DataFrame with the original rec columns (to identify
                   who has each rec)
      min_n:       Minimum patients with the rec for the rec to be ranked
                   (rare recs with n<min_n are skipped, not penalized)

    Returns:
      dict with:
        top7_clinical:     fraction of clinical recs in top 7 (0-1)
        n_clinical_found:  count of clinical recs in top 7 (0-7)
        missing:           clinical recs NOT in top 7
        intruders:         non-clinical recs in top 7
        top7_full:         full list of top 7 recs by combined SHAP
        rec_combined:      dict {rec: combined_shap}
    """
    ptd = patient_df.set_index('patient_num')
    common = df_shap.index.intersection(ptd.index)

    rec_combined = {}
    for r in ACTIVE_RECS:
        if r not in df_shap.columns:
            continue
        active_mask = ptd.loc[common, r] >= 0.5
        n = int(active_mask.sum())
        if n < min_n:
            continue  # skip rare recs (noise)
        rec_signed = float(df_shap.loc[common, r].loc[active_mask].mean())
        # Add path contribution if path feature is present
        path_name = REC_TO_PATH.get(r)
        path_signed = 0.0
        if path_name and path_name in df_shap.columns:
            path_signed = float(df_shap.loc[common, path_name].loc[active_mask].mean())
        rec_combined[r] = rec_signed + path_signed

    # Rank by combined SHAP descending (highest first)
    ranked = sorted(rec_combined.items(), key=lambda x: x[1], reverse=True)
    top7 = [r for r, _ in ranked[:7]]

    # Score: how many of the clinical 7 are in the top 7
    clinical_in_top7 = [r for r in top7 if r in CLINICAL_TOP_RECS]
    missing = [r for r in CLINICAL_TOP_RECS
               if r in rec_combined and r not in top7]
    intruders = [r for r in top7 if r not in CLINICAL_TOP_RECS]
    # Skipped clinical recs (filtered by min_n)
    skipped_clinical = [r for r in CLINICAL_TOP_RECS if r not in rec_combined]

    # Effective denominator: 7 minus the ones we couldn't rank at all
    n_eligible = N_CLINICAL - len(skipped_clinical)
    score = len(clinical_in_top7) / N_CLINICAL  # use raw 7 as the spec

    result = {
        'top7_clinical':       score,
        'n_clinical_in_top7':  len(clinical_in_top7),
        'n_eligible':          n_eligible,
        'top7_recs':           top7,
        'missing':             missing,
        'intruders':           intruders,
        'skipped_clinical':    skipped_clinical,
        'rec_combined':        rec_combined,
    }

    if verbose:
        print(f"  top7_clinical = {score:.3f}  ({len(clinical_in_top7)}/7)")
        print(f"  Top 7 (by combined SHAP):  {top7}")
        print(f"  Clinical recs in top 7:    {clinical_in_top7}")
        print(f"  Missing from top 7:        {missing}")
        print(f"  Intruders in top 7:        {intruders}")
        if skipped_clinical:
            print(f"  Skipped (n<{min_n}):       {skipped_clinical}")

    return result


# ==========================================================================
# Helper to apply to a directory of SHAP pkl files
# ==========================================================================
def apply_to_all_doctors(results_dir='results',
                          patient_df_path='synthetic_data (7).xlsx',
                          shap_suffix='_per_feature_shap.pkl',
                          min_n=10):
    """Apply the metric to every <doctor>_per_feature_shap.pkl in results/."""
    results_dir = Path(results_dir)
    patient_df = pd.read_excel(patient_df_path)

    rows = []
    pkl_files = sorted(results_dir.glob(f'*{shap_suffix}'))
    print(f"Found {len(pkl_files)} SHAP pickle files")
    for p in pkl_files:
        doctor = p.stem.replace(shap_suffix.replace('.pkl', ''), '')
        df_shap = pd.read_pickle(p)
        result = compute_top7_clinical(df_shap, patient_df, min_n=min_n)
        rows.append({
            'doctor':              doctor,
            'top7_clinical':       result['top7_clinical'],
            'n_clinical_in_top7':  result['n_clinical_in_top7'],
            'top7_recs':           ','.join(result['top7_recs']),
            'missing':             ','.join(result['missing']),
            'intruders':           ','.join(result['intruders']),
        })

    df_summary = pd.DataFrame(rows).sort_values('top7_clinical', ascending=False)
    out_path = results_dir / 'top7_clinical_per_doctor.csv'
    df_summary.to_csv(out_path, index=False)
    print(f"\n\u2713 Saved {out_path}")
    return df_summary


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--results', default='results')
    p.add_argument('--patient_df', default='synthetic_data (7).xlsx')
    p.add_argument('--suffix', default='_per_feature_shap.pkl',
                   help='Suffix of the SHAP pkl files (e.g. _per_feature_shap_winning.pkl)')
    p.add_argument('--min_n', type=int, default=10)
    args = p.parse_args()
    df = apply_to_all_doctors(results_dir=args.results,
                               patient_df_path=args.patient_df,
                               shap_suffix=args.suffix,
                               min_n=args.min_n)
    print("\n" + "=" * 70)
    print("Summary (sorted by top7_clinical):")
    print("=" * 70)
    print(df.to_string(index=False))
