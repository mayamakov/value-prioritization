# -*- coding: utf-8 -*-
"""
value_mapping_v10.py — REVISED value mapping (v10, June 2026).

Replaces the v9 single-Beneficence-split scheme. Beneficence is now SPECIFIED
into two INDEPENDENT operational forms (Beauchamp–Childress umbrella):

  internal key            display label                    source
  --------------------    ------------------------------   ----------------------------
  Beneficence-Immediate   Utility-oriented beneficence     rec-intrinsic U (flat)
  Beneficence-LongTerm    Priority-to-need beneficence      patient raw 10-year CV risk
  Non-maleficence         Non-maleficence                   rec-intrinsic NM   (unchanged)
  Autonomy                Patient empowerment               rec-intrinsic EMP  (renamed only)
  Justice                 Resource justice                  price-log JU       (unchanged)

INTERNAL KEYS ARE DELIBERATELY KEPT IDENTICAL TO v9 so every existing downstream
script keeps working with ZERO edits. Only DISPLAY labels change, applied at
render time via DISPLAY_LABELS.

regenerate_everything_v10.py injects this module under the name
'value_mapping_rec_intrinsic' -> v9 file on disk is never overwritten.
"""
import math
import numpy as np
from paper_value_config_v10 import (
    REC_TABLE, DISPLAY_LABELS,
    JUSTICE_SCALE, NORMALIZE, compute_justice,
)

# internal keys frozen (= v9); see DISPLAY_LABELS for the v10 names
ETHICAL_DIMENSIONS = [
    'Beneficence-Immediate',   # = Utility-oriented beneficence
    'Beneficence-LongTerm',    # = Priority-to-need beneficence
    'Non-maleficence',
    'Autonomy',                # = Patient empowerment
    'Justice',                 # = Resource justice
]

_JU = compute_justice(REC_TABLE, scale=JUSTICE_SCALE)

# Frozen mapping consumed by the rest of the pipeline.
REC_INTRINSIC_MAPPING = {
    rec: dict(U=m['U'], NM=m['NM'], EMP=m['EMP'],
              JU=round(_JU[rec], 4), PRICE=m['PRICE'], note=m.get('note', ''))
    for rec, m in REC_TABLE.items()
}

# Priority-to-need configuration (patient-level)
from paper_value_config_v10 import PRIORITY_SOURCE, RISK_MAX, SUPPORT_WEIGHT, SUPPORT_K

# === v11 recommendation-support weighting (item 1) =========================
# REC_SUPPORT_WEIGHT: dict rec -> weight (>0; >1 for rare recs, <1 for common).
# None => every weight is 1.0 (no support normalization). Populated at run time
# from the common-pairs support (rebuild_pair_csvs.py computes + saves
# results/rec_support_weights.csv; downstream scripts call load_support_weights()).
REC_SUPPORT_WEIGHT = None


def _w(rec):
    """Support weight for a recommendation (1.0 if weighting is off/unset)."""
    if not SUPPORT_WEIGHT or REC_SUPPORT_WEIGHT is None:
        return 1.0
    return float(REC_SUPPORT_WEIGHT.get(rec, 1.0))


def compute_support_weights(common_pairs, coh, recs=None, K=None):
    """Support shrinkage (item 1 — empirical-Bayes style, the 'best estimate given
    limited data' answer to 'what would the effect be with more patients').
    n_rec = #common pairs whose two patients differ on rec.
    w_rec = n_rec / (n_rec + K)  -> a rec measured on FEW pairs is pulled toward 0
    (we trust it less); a well-supported rec keeps ~full weight. K = SUPPORT_K,
    else median(positive n_rec) (a transparent, PRE-SPECIFIED reference; do NOT
    tune K to obtain a p-value). Applied INSIDE the value mapping, so it changes
    axis composition non-uniformly across raters.
    Preserves Priority-to-need (raw risk, no rec components) and Resource justice;
    stabilises the sparse axes instead of amplifying their noise.
    common_pairs: iterable of (a, b) patient-id tuples; coh: DataFrame indexed by patient id."""
    import numpy as _np
    recs = recs or [r for r in REC_INTRINSIC_MAPPING if r in coh.columns]
    n_rec = {}
    for r in recs:
        n_rec[r] = int(sum(1 for (a, b) in common_pairs
                           if (coh.loc[a, r] >= 0.5) != (coh.loc[b, r] >= 0.5)))
    if K is None:
        K = SUPPORT_K
    if K is None:
        pos = [v for v in n_rec.values() if v > 0]
        K = float(_np.median(pos)) if pos else 1.0
    return {r: (n_rec[r] / (n_rec[r] + K) if (n_rec[r] + K) > 0 else 0.0) for r in recs}, n_rec, float(K)


def set_support_weights(weights):
    """Install the support-weight dict on this module (used by every patient_to_values call)."""
    global REC_SUPPORT_WEIGHT
    REC_SUPPORT_WEIGHT = dict(weights) if weights is not None else None


def load_support_weights(path='results/rec_support_weights.csv'):
    """Load support weights saved by rebuild_pair_csvs.py; no-op if file is absent."""
    import os as _os, pandas as _pd
    if not _os.path.exists(path):
        return False
    df = _pd.read_csv(path)
    set_support_weights(dict(zip(df['rec'], df['weight'])))
    return True


def priority_to_need(patient_row):
    """Priority-to-need = the patient's raw 10-year CV risk (age is subsumed in it).
    NOT derived from the recommendation. One value per patient."""
    risk = float(patient_row.get('risk', 0.0))
    if PRIORITY_SOURCE == 'risk_norm':
        return min(risk / RISK_MAX, 1.0)
    return risk


def patient_to_values(patient_row, mapping=REC_INTRINSIC_MAPPING):
    """Utility/NM/EMP/Justice rec-intrinsic; Priority-to-need = raw 10-year risk."""
    v = {d: 0.0 for d in ETHICAL_DIMENSIONS}
    for rec, m in mapping.items():
        if rec not in patient_row.index or patient_row[rec] < 0.5:
            continue
        w = _w(rec)                              # v11 support weight (1.0 if off)
        v['Beneficence-Immediate'] += w * m['U']
        v['Non-maleficence']       += w * m['NM']
        v['Autonomy']              += w * m['EMP']
        v['Justice']               += w * m['JU']
    v['Beneficence-LongTerm'] = priority_to_need(patient_row)
    return v


def aggregate_shap_to_ethical_values(shap_matrix, feature_cols, patient_df_orig):
    """Rec-intrinsic SHAP aggregation: (n,n_features) -> (n,5).
    Utility/NM/EMP/Justice are rec-weighted SHAP contributions. Priority-to-need
    is the patient's raw 10-year risk (patient-level), broadcast per row."""
    n = shap_matrix.shape[0]
    out = np.zeros((n, len(ETHICAL_DIMENSIONS)))
    DIM = {d: i for i, d in enumerate(ETHICAL_DIMENSIONS)}
    for fi, feat in enumerate(feature_cols):
        if feat not in REC_INTRINSIC_MAPPING:
            continue
        m = REC_INTRINSIC_MAPPING[feat]
        has = (patient_df_orig[feat].values >= 0.5).astype(float) if feat in patient_df_orig.columns else np.ones(n)
        contrib = shap_matrix[:, fi] * has * _w(feat)   # v11 support weight
        out[:, DIM['Beneficence-Immediate']] += contrib * m['U']
        out[:, DIM['Non-maleficence']]       += contrib * m['NM']
        out[:, DIM['Autonomy']]              += contrib * m['EMP']
        out[:, DIM['Justice']]               += contrib * m['JU']
    # Priority-to-need (SHAP) = SHAP attribution of the risk feature, per rater.
    if 'risk' in feature_cols:
        ri = list(feature_cols).index('risk')
        out[:, DIM['Beneficence-LongTerm']] = np.abs(shap_matrix[:, ri])
    elif 'risk' in patient_df_orig.columns:
        out[:, DIM['Beneficence-LongTerm']] = patient_df_orig['risk'].values.astype(float)
    return out


def normalize_profiles(df, dims=None, method=NORMALIZE):
    """Per-axis normalization across the rows present (the 26 raters), so every
    dimension gets EQUAL OPPORTUNITY in distances / pentagon. Returns a COPY.
    Call this on any profile table (rows=raters, cols include `dims`) BEFORE
    computing Euclidean distances or drawing the pentagon."""
    dims = dims or ETHICAL_DIMENSIONS
    out = df.copy()
    for d in dims:
        if d not in out.columns:
            continue
        col = out[d].astype(float)
        if method == 'zscore':
            sd = col.std(ddof=0)
            out[d] = (col - col.mean()) / sd if sd > 0 else 0.0
        elif method == 'minmax':
            lo, hi = col.min(), col.max()
            out[d] = (col - lo) / (hi - lo) if hi > lo else 0.0
        # 'none' -> unchanged
    return out


def make_pricebased_justice(mapping, rec_price, max_price):
    """Compatibility shim (Justice is already price-based here)."""
    out = {k: dict(v) for k, v in mapping.items()}
    for rec in out:
        if rec in rec_price:
            out[rec]['JU'] = 1.0 - (rec_price[rec] / max_price)
    return out


if __name__ == '__main__':
    print("value_mapping_v10  |  internal keys frozen, display relabeled")
    print("display labels:")
    for k, lbl in DISPLAY_LABELS.items():
        print(f"    {k:24} -> {lbl}")
    print(f"\n{'rec':6}{'U':>6}{'NM':>5}{'EMP':>5}{'Price':>8}{'Justice':>9}  note")
    for rec, m in REC_INTRINSIC_MAPPING.items():
        print(f"{rec:6}{m['U']:>6}{m['NM']:>5}{m['EMP']:>5}{m['PRICE']:>8}{m['JU']:>9.2f}  {m['note'][:38]}")
    print(f"\nPriority-to-need = patient raw 10-year risk (source={PRIORITY_SOURCE}); not per-rec.")