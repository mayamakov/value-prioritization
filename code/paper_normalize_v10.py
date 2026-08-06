# -*- coding: utf-8 -*-
"""
paper_normalize_v10.py — single normalization helper for the v10 revision.

Per-axis normalization across the raters present, so every dimension gets equal
opportunity in distances / pentagon. Used by BOTH the orchestrator (Stage C, to
normalize the CSVs that figure scripts read) and print_all_tables_v10.py.

Design:
  * full-26-rater value tables  -> self-fit z-score across their 26 raters.
  * seed table (LLM-only)        -> borrow the common-pairs params + the
                                    common-pairs physician centroid (same
                                    signature definition), so distances stay
                                    referenced to physicians.
  * raw-SHAP / mapping-sensitivity tables are NOT normalized (controls).
"""
import numpy as np
import pandas as pd
from paper_value_config_v10 import NORMALIZE

DIMS = ['Beneficence-Immediate', 'Beneficence-LongTerm',
        'Non-maleficence', 'Autonomy', 'Justice']

HUMANS = ['fm_1', 'fm_2', 'fm_3', 'fm_4', 'fm_5',
          'ph_1', 'ph_2', 'ph_3', 'ph_4', 'ph_5']


def fit_params(df, cols=DIMS, method=None):
    """Return {col: (method, a, b)} fit across the rows of df."""
    method = method or NORMALIZE
    params = {}
    for c in cols:
        if c not in df.columns:
            continue
        x = df[c].astype(float)
        if method == 'zscore':
            params[c] = ('zscore', float(x.mean()), float(x.std(ddof=0)))
        elif method == 'minmax':
            params[c] = ('minmax', float(x.min()), float(x.max()))
        else:
            params[c] = ('none', 0.0, 1.0)
    return params


def apply_params(df, params, cols=DIMS):
    """Apply previously-fit params; returns a COPY."""
    out = df.copy()
    for c in cols:
        if c not in out.columns or c not in params:
            continue
        m, a, b = params[c]
        x = out[c].astype(float)
        if m == 'zscore':
            out[c] = (x - a) / b if b > 0 else 0.0
        elif m == 'minmax':
            out[c] = (x - a) / (b - a) if b > a else 0.0
    return out


def normalize_self(df, cols=DIMS, method=None):
    """Self-fit + apply across the rows present."""
    return apply_params(df, fit_params(df, cols, method), cols)


def centroid(df, rater_col, raters, cols=DIMS):
    sub = df[df[rater_col].isin(raters)]
    return sub[cols].astype(float).mean().values


def distances(df, centroid_vec, cols=DIMS):
    M = df[cols].astype(float).values
    return np.sqrt(((M - centroid_vec) ** 2).sum(axis=1))
