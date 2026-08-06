# -*- coding: utf-8 -*-
"""paper_value_config_v10.py — single source of truth for the v10 value revision."""
import math

DISPLAY_LABELS = {
    'Beneficence-Immediate': 'Utility-oriented beneficence',
    'Beneficence-LongTerm':  'Priority-to-need beneficence',
    'Non-maleficence':       'Non-maleficence',
    'Autonomy':              'Patient empowerment',
    'Justice':               'Resource justice',
}

AGE_CENTER  = 65.0
RISK_CENTER = 20.0
SLOPE       = 8.0

NORMALIZE        = 'zscore'
PENTAGON_DISPLAY = 'minmax'
JUSTICE_SCALE    = 'log'

# --- v11 recommendation-support shrinkage (item 1) -------------------------
# A recommendation measured on FEW common pairs (e.g. rec3 ~7/300, the rec13+
# rec17 that carry Non-maleficence) gives a noisy per-rec estimate. We shrink it
# toward 0 by its support (empirical-Bayes style: trust low-support recs less):
#   w_rec = n_rec / (n_rec + K) ,  n_rec = #common pairs whose two patients differ
#   on the rec, K = SUPPORT_K (None => median n_rec). PRE-SPECIFIED; do NOT tune K
#   to a p-value. Applied INSIDE the value mapping (non-uniform across raters).
# This STABILISES the sparse axes (does not amplify their noise) and preserves
# Priority-to-need (raw risk) and Resource justice. The inverse-support variant
# (1/n) was rejected: it amplifies sparse-rec noise and destroys Resource justice.
SUPPORT_WEIGHT = False    # OFF: do not weight by support. Sparse-axis effects are
                          # shown via the pooled CI regression (figR_axis_regression_CI.py),
                          # not by reweighting the signature (which either penalises or
                          # amplifies under-represented features). See cross-analysis defence.
SUPPORT_K      = None     # None => median(n_rec); pre-specified, not tuned

REC_TABLE = {
 'rec1':  dict(U=0.25, NM=0.0, EMP=0.0, PRICE=128,   note='Basic lab panel (LDL/HDL/A1C)'),
 'rec2':  dict(U=0.25, NM=0.0, EMP=0.0, PRICE=200,   note='Advanced lab panel (ApoB/Lpa)'),
 'rec3':  dict(U=0.50, NM=0.0, EMP=0.0, PRICE=132,   note='Pathophysiology investigation'),
 'rec4':  dict(U=0.25, NM=0.0, EMP=0.0, PRICE=68,    note='Routine LDL monitoring'),
 'rec5':  dict(U=0.25, NM=0.0, EMP=0.0, PRICE=255,   note='Diagnostic imaging (carotid Doppler)'),
 'rec6':  dict(U=0.50, NM=0.0, EMP=0.0, PRICE=4013,  note='Advanced imaging (CTA/perfusion)'),
 'rec8':  dict(U=0.25, NM=0.0, EMP=0.0, PRICE=111,   note='BP/BMI measurement'),
 'rec10': dict(U=1.00, NM=0.0, EMP=0.0, PRICE=392,   note='First-line treatment (low-dose statin)'),
 'rec11': dict(U=1.00, NM=0.0, EMP=0.0, PRICE=25527, note='Advanced treatment (med/high statin/PCSK9)'),
 'rec12': dict(U=0.75, NM=0.0, EMP=0.0, PRICE=652,   note='Treatment upgrade'),
 'rec13': dict(U=0.75, NM=1.0, EMP=0.0, PRICE=2630,  note='Treatment replacement (contraindication)'),
 'rec16': dict(U=0.50, NM=0.0, EMP=0.0, PRICE=364,   note='Specialist consultation (Lipidologist)'),
 'rec17': dict(U=0.25, NM=1.0, EMP=0.0, PRICE=364,   note='Other consultation (Hepatology)'),
 'rec18': dict(U=0.25, NM=0.0, EMP=1.0, PRICE=191,   note='Nutritional consultation (Dietitian)'),
 'rec19': dict(U=0.25, NM=0.0, EMP=1.0, PRICE=0,     note='Lifestyle improvement (exercise/diet)'),
 'rec21': dict(U=0.00, NM=0.0, EMP=0.0, PRICE=0,     note='Curate medical record'),
}

PRIORITY_SOURCE = 'risk'
RISK_MAX = 40.0

def compute_justice(table, scale=JUSTICE_SCALE):
    pos = [m['PRICE'] for m in table.values() if m['PRICE'] > 0]
    mn, mx = float(min(pos)), float(max(pos))
    out = {}
    for rec, m in table.items():
        p = float(m['PRICE'])
        if p <= 0:
            out[rec] = 1.0
        elif scale == 'linear':
            out[rec] = 1.0 - (p - mn) / (mx - mn)
        else:
            out[rec] = 1.0 - math.log(p / mn) / math.log(mx / mn)
    return out
