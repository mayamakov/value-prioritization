# -*- coding: utf-8 -*-
"""
stat_review_sensitivity.py — sensitivity & uncertainty analyses requested by the
pre-submission statistical review (Aug 2026).

Covers:
  [A] CRITICAL 1 (remainder): ridge-penalty (lambda) and predictor-scaling
      sensitivity for the pooled pairwise choice model (risk + 16 recs).
  [B] CRITICAL 2: two-way (rater x pair) cluster bootstrap for the pooled
      per-axis choice regression — CIs that respect BOTH dependence dimensions
      and the small number of rater clusters.
  [C] CRITICAL 2: developer-family-balanced sensitivity for the per-axis
      physician-vs-LLM Mann-Whitney comparison (family means + all
      one-model-per-family panels).
  [D] MAJOR 4: joint pair bootstrap (300 common pairs resampled once per
      replicate, shared by all 26 raters) propagated through signatures ->
      z-scoring -> physician centroid -> Euclidean distances. Percentile CIs
      for every distance, the outside-range count, and closest-model stability.
  [E] MAJOR 4: leave-one-axis-out distances and rankings.
  [F] MAJOR 4: sparse-axis sensitivity — exclude or support-weight the
      Non-maleficence axis (and report per-axis pair support).

All analyses reuse the pipeline's own loaders / value mapping / normalization
convention (fixed canonical pair orientation a<b; per-axis z-score across the
26 raters present). Outputs -> results/stat_review/ (CSVs) + console report.

Run:  cd code && python stat_review_sensitivity.py [--fast]
      (--fast uses B=200 bootstrap replicates instead of 1000)
"""
import argparse, json, os, sys, itertools
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or '.')

from value_mapping_rec_intrinsic import patient_to_values, ETHICAL_DIMENSIONS as DIMS
import value_mapping_rec_intrinsic as _vm
_vm.set_support_weights(None)          # raw clinical signal, as in the pipeline
from panel_config import (HUMANS, LLMS, LEGACYSET, PANEL, DISPLAY,
                          MODELS_BY_FAMILY, MODEL_FAMILY)

DISP_AXIS = {'Beneficence-Immediate': 'Beneficence-utility',
             'Beneficence-LongTerm':  'Beneficence-need',
             'Non-maleficence':       'Non-maleficence',
             'Autonomy':              'Patient empowerment',
             'Justice':               'Resource justice'}

RECS = ['rec1','rec2','rec3','rec4','rec5','rec6','rec8','rec10','rec11',
        'rec12','rec13','rec16','rec17','rec18','rec19','rec21']

RNG = np.random.default_rng(20260811)

# --------------------------------------------------------------------------
# Data loading (pipeline conventions)
# --------------------------------------------------------------------------
def _n(it):
    if isinstance(it,(list,tuple)) and len(it)==2 and isinstance(it[0],(list,tuple)):
        return (int(it[0][0]), int(it[0][1]))
    if isinstance(it,(list,tuple)) and len(it)>=2:
        return (int(it[0]), int(it[1]))

def _l(p):
    p = Path(p)
    return [x for x in (_n(z) for z in json.load(open(p))) if x] if p.exists() else []

def load_choices(key, base='physicians_results'):
    d = Path(base)/key; pr = []
    for i in range(4):
        pr += _l(d/f'{key}_train_iter_{i}_ranked.json')
    if key in LEGACYSET:
        pr += _l(d/f'{key}_test_ranked.json')
    else:
        got = False
        for pt in 'AB':
            t = _l(d/f'{key}_test_part_{pt}_ranked.json'); pr += t; got = got or bool(t)
        if not got:
            pr += _l(d/f'{key}_test_ranked.json')
    return {tuple(sorted(p)): p[0] for p in pr}      # canonical (a<b) -> winner

def find_data(name):
    for base in ['.', '..', '../data', 'data']:
        p = Path(base)/name
        if p.exists():
            return p
    raise FileNotFoundError(name)

print('Loading data ...')
coh = pd.read_excel(find_data('synthetic_data.xlsx')).set_index('patient_num')
base_dir = 'physicians_results' if Path('physicians_results').exists() else '../data/physicians_results'
C = {k: load_choices(k, base_dir) for k in PANEL}
common = sorted(set.intersection(*[set(C[k]) for k in PANEL]))
common = [u for u in common if u[0] in coh.index and u[1] in coh.index]
NP_, NR = len(common), len(PANEL)
print(f'  raters={NR} ({len(HUMANS)} physicians, {len(LLMS)} LLMs), common pairs={NP_}')

pids = sorted({p for u in common for p in u})
V = {pid: np.array([patient_to_values(coh.loc[pid])[d] for d in DIMS], float) for pid in pids}

# fixed canonical orientation: DELTA[j] = V[a]-V[b] for pair j=(a,b), a<b
DELTA = np.array([V[a]-V[b] for (a,b) in common])                    # (P,5)
# per-rater choice sign: +1 if rater chose a (the canonical first patient)
SGN = np.array([[1.0 if C[k][(a,b)]==a else -1.0 for (a,b) in common]
                for k in PANEL])                                     # (R,P)
Y = (SGN + 1) / 2                                                    # y=1 iff chose a
# signature tensor: T[r,j,:] = V[winner]-V[loser] = sgn * DELTA
T = SGN[:, :, None] * DELTA[None, :, :]                              # (R,P,5)

H_IDX = np.array([PANEL.index(k) for k in HUMANS])
L_IDX = np.array([PANEL.index(k) for k in LLMS])

# regression design (risk + 16 recs), fixed orientation
d_risk = np.array([float(coh.loc[a,'risk'] - coh.loc[b,'risk']) for (a,b) in common])
d_recs = np.array([[float(coh.loc[a,r] >= 0.5) - float(coh.loc[b,r] >= 0.5)
                    for r in RECS] for (a,b) in common])             # (P,16)
XFULL = np.column_stack([d_risk, d_recs])                            # (P,17)

_res = Path('results') if Path('results').is_dir() else \
       (Path('../results') if Path('../results').is_dir() else Path('results'))
OUT = _res / 'stat_review'; OUT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Fitting helpers
# --------------------------------------------------------------------------
def ridge_logit(X, y, lam=2.0, penalize_intercept=False, iters=300, tol=1e-11):
    """IRLS ridge logistic with intercept (unpenalized unless stated)."""
    Xd = np.column_stack([np.ones(len(X)), X])
    P = np.eye(Xd.shape[1]) * lam
    if not penalize_intercept:
        P[0, 0] = 0.0
    b = np.zeros(Xd.shape[1])
    for _ in range(iters):
        p = 1/(1+np.exp(-(Xd@b))); W = p*(1-p)
        try:
            step = np.linalg.solve((Xd.T*W)@Xd + P + 1e-10*np.eye(Xd.shape[1]),
                                   Xd.T@(y-p) - P@b)
        except np.linalg.LinAlgError:
            break
        b += step
        if np.max(np.abs(step)) < tol:
            break
    return b

def mcfadden(X, y, lam):
    b = ridge_logit(X, y, lam)
    Xd = np.column_stack([np.ones(len(X)), X])
    p = np.clip(1/(1+np.exp(-(Xd@b))), 1e-12, 1-1e-12)
    ll = np.sum(y*np.log(p) + (1-y)*np.log(1-p))
    yb = y.mean()
    ll0 = np.sum(y*np.log(yb) + (1-y)*np.log(1-yb))
    return 1 - ll/ll0

def logit_nointercept(X, y, iters=100, tol=1e-9):
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1/(1+np.exp(-(X@beta))); W = p*(1-p)
        try:
            step = np.linalg.solve((X*W[:,None]).T@X + 1e-8*np.eye(X.shape[1]),
                                   X.T@(y-p))
        except np.linalg.LinAlgError:
            break
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    return beta

def build_pooled(idx_raters, idx_pairs, Xmat):
    """Stack (rater x pair) rows for the given rater/pair index samples."""
    X = np.tile(Xmat[idx_pairs], (len(idx_raters), 1))
    y = Y[np.ix_(idx_raters, idx_pairs)].ravel()
    return X, y

def signatures_from_pairs(idx_pairs):
    return T[:, idx_pairs, :].mean(axis=1)                           # (R,5)

def zdist(sig):
    """z-score per axis across the 26 raters -> physician centroid -> distances."""
    mu = sig.mean(0); sd = sig.std(0); sd[sd == 0] = 1.0
    Z = (sig - mu) / sd
    cent = Z[H_IDX].mean(0)
    return np.sqrt(((Z - cent)**2).sum(1)), Z

# ==========================================================================
print('\n' + '='*78)
print('[A] CRITICAL 1 — ridge-penalty (lambda) and scaling sensitivity')
print('='*78)
LAMBDAS = [0.01, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
rowsA = []
for scaling in ['raw', 'standardized']:
    if scaling == 'raw':
        Xs = XFULL.copy()
    else:
        sd = XFULL.std(0); sd[sd == 0] = 1.0
        Xs = (XFULL - XFULL.mean(0)) / sd
    for grp, idx in [('Physicians', H_IDX), ('LLMs', L_IDX)]:
        Xg, yg = build_pooled(idx, np.arange(NP_), Xs)
        for lam in LAMBDAS:
            full = mcfadden(Xg, yg, lam)
            no_risk = mcfadden(Xg[:, 1:], yg, lam)          # drop risk column
            no_recs = mcfadden(Xg[:, :1], yg, lam)          # risk only
            rowsA.append(dict(scaling=scaling, group=grp, lam=lam,
                              full_R2=full,
                              d_risk=full - no_risk,
                              d_recs=full - no_recs))
A = pd.DataFrame(rowsA)
A.to_csv(OUT/'A_penalty_scaling_sensitivity.csv', index=False)
for scaling in ['raw', 'standardized']:
    print(f'\n  scaling = {scaling}   (dRisk / dRecs = unique drop-one blocks)')
    print('  lam     Phys full  dRisk  dRecs   |  LLM full  dRisk  dRecs   risk>recs(LLM)')
    for lam in LAMBDAS:
        p = A[(A.scaling==scaling)&(A.group=='Physicians')&(A.lam==lam)].iloc[0]
        l = A[(A.scaling==scaling)&(A.group=='LLMs')&(A.lam==lam)].iloc[0]
        print(f'  {lam:5.2f}   {p.full_R2:.3f}     {p.d_risk:.3f}  {p.d_recs:.3f}   |'
              f'  {l.full_R2:.3f}    {l.d_risk:.3f}  {l.d_recs:.3f}   '
              f'{"YES" if l.d_risk > l.d_recs else "no"}')

# ==========================================================================
print('\n' + '='*78)
print('[B] CRITICAL 2 — two-way (rater x pair) cluster bootstrap, per-axis regression')
print('='*78)
parser = argparse.ArgumentParser(); parser.add_argument('--fast', action='store_true')
ARGS, _ = parser.parse_known_args()
B = 200 if ARGS.fast else 1000
print(f'  B = {B} replicates; per replicate: raters AND pairs resampled with replacement')

def axis_beta(idx_raters, idx_pairs):
    D = DELTA[idx_pairs]
    mu, sd = D.mean(0), D.std(0); sd[sd == 0] = 1.0
    Xz = (D - mu) / sd
    X = np.tile(Xz, (len(idx_raters), 1))
    y = Y[np.ix_(idx_raters, idx_pairs)].ravel()
    return logit_nointercept(X, y)

full_pairs = np.arange(NP_)
beta_pt = {'Physicians': axis_beta(H_IDX, full_pairs),
           'LLMs':       axis_beta(L_IDX, full_pairs)}
boot = {g: np.empty((B, 5)) for g in beta_pt}
for b in range(B):
    ip = RNG.integers(0, NP_, NP_)
    ih = H_IDX[RNG.integers(0, len(H_IDX), len(H_IDX))]
    il = L_IDX[RNG.integers(0, len(L_IDX), len(L_IDX))]
    boot['Physicians'][b] = axis_beta(ih, ip)
    boot['LLMs'][b]       = axis_beta(il, ip)

rowsB = []
for g in ['Physicians', 'LLMs']:
    lo = np.percentile(boot[g], 2.5, axis=0); hi = np.percentile(boot[g], 97.5, axis=0)
    for i, d in enumerate(DIMS):
        sig = (lo[i] > 0) or (hi[i] < 0)
        rowsB.append(dict(group=g, axis=DISP_AXIS[d], beta=beta_pt[g][i],
                          lo=lo[i], hi=hi[i], excl_zero=sig))
Bdf = pd.DataFrame(rowsB)
Bdf.to_csv(OUT/'B_twoway_bootstrap_axis_regression.csv', index=False)
for g in ['Physicians', 'LLMs']:
    print(f'\n  {g}: standardized per-axis coefficient  [two-way 95% bootstrap CI]')
    for _, r in Bdf[Bdf.group==g].iterrows():
        print(f'    {r.axis:22s} {r.beta:+.3f}  [{r.lo:+.3f}, {r.hi:+.3f}]'
              f'  {"*" if r.excl_zero else " "}')

# ==========================================================================
print('\n' + '='*78)
print('[C] CRITICAL 2 — developer-family-balanced Mann-Whitney sensitivity')
print('='*78)
dist0, Z0 = zdist(signatures_from_pairs(full_pairs))
Zdf = pd.DataFrame(Z0, index=PANEL, columns=DIMS)
rowsC = []
fam_keys = list(MODELS_BY_FAMILY)
combos = list(itertools.product(*[MODELS_BY_FAMILY[f] for f in fam_keys]))
print(f'  one-per-family panels: {len(combos)} combinations of 7 models')
for d in DIMS:
    hvals = Zdf.loc[HUMANS, d].values
    # (i) original 16-model comparison
    p_all = mannwhitneyu(hvals, Zdf.loc[LLMS, d].values, alternative='two-sided').pvalue
    # (ii) developer means (7 obs)
    fam_means = np.array([Zdf.loc[MODELS_BY_FAMILY[f], d].mean() for f in fam_keys])
    p_fam = mannwhitneyu(hvals, fam_means, alternative='two-sided').pvalue
    # (iii) all one-model-per-family panels
    ps = np.array([mannwhitneyu(hvals, Zdf.loc[list(cb), d].values,
                                alternative='two-sided').pvalue for cb in combos])
    rowsC.append(dict(axis=DISP_AXIS[d], p_16models=p_all, p_family_means=p_fam,
                      p_1perfam_min=ps.min(), p_1perfam_median=np.median(ps),
                      p_1perfam_max=ps.max(),
                      frac_sig_bonf=float(np.mean(ps < 0.05/len(DIMS)))))
Cdf = pd.DataFrame(rowsC)
Cdf.to_csv(OUT/'C_family_balanced_mannwhitney.csv', index=False)
print('\n  axis                    p(16 LLMs)  p(7 fam means)  p range over 90 panels   frac<Bonf')
for _, r in Cdf.iterrows():
    print(f'  {r.axis:22s} {r.p_16models:10.2e}  {r.p_family_means:12.2e}  '
          f'[{r.p_1perfam_min:.1e}, {r.p_1perfam_max:.1e}]   {r.frac_sig_bonf:5.0%}')

# ==========================================================================
print('\n' + '='*78)
print(f'[D] MAJOR 4 — joint pair bootstrap (B={B}) through signatures -> distances')
print('='*78)
phys_lo0, phys_hi0 = dist0[H_IDX].min(), dist0[H_IDX].max()
bootD = np.empty((B, NR)); n_out = np.empty(B, int); closest = []
in_range_count = np.zeros(len(LLMS))
for b in range(B):
    ip = RNG.integers(0, NP_, NP_)
    dist_b, _ = zdist(signatures_from_pairs(ip))
    bootD[b] = dist_b
    lo, hi = dist_b[H_IDX].min(), dist_b[H_IDX].max()
    ins = (dist_b[L_IDX] >= lo) & (dist_b[L_IDX] <= hi)
    n_out[b] = int((~ins).sum())
    in_range_count += ins
    closest.append(LLMS[int(np.argmin(dist_b[L_IDX]))])
closest = pd.Series(closest)
rowsD = []
for i, k in enumerate(LLMS):
    ci = np.percentile(bootD[:, L_IDX[i]], [2.5, 97.5])
    rowsD.append(dict(model=DISPLAY.get(k, k), dist=dist0[L_IDX[i]],
                      ci_lo=ci[0], ci_hi=ci[1],
                      p_within_range=in_range_count[i]/B,
                      p_closest=float((closest == k).mean())))
Ddf = pd.DataFrame(rowsD).sort_values('dist')
Ddf.to_csv(OUT/'D_pair_bootstrap_distances.csv', index=False)
print(f'\n  physician range (point): {phys_lo0:.3f} – {phys_hi0:.3f}')
print(f'  LLMs outside range: point={int((~((dist0[L_IDX]>=phys_lo0)&(dist0[L_IDX]<=phys_hi0))).sum())}/16 ;'
      f'  bootstrap 95% CI = [{int(np.percentile(n_out,2.5))}, {int(np.percentile(n_out,97.5))}],'
      f'  median = {int(np.median(n_out))}')
print('\n  model                    dist    95% CI            P(within range)  P(closest)')
for _, r in Ddf.iterrows():
    print(f'  {r.model:22s} {r.dist:.3f}  [{r.ci_lo:.3f}, {r.ci_hi:.3f}]   '
          f'{r.p_within_range:8.0%}        {r.p_closest:6.0%}')

# ==========================================================================
print('\n' + '='*78)
print('[E] MAJOR 4 — leave-one-axis-out distances')
print('='*78)
rowsE = []
for drop_i, d in enumerate(DIMS):
    keep = [i for i in range(5) if i != drop_i]
    sig = signatures_from_pairs(full_pairs)
    mu = sig.mean(0); sd = sig.std(0); sd[sd == 0] = 1
    Z = ((sig - mu) / sd)[:, keep]
    cent = Z[H_IDX].mean(0)
    dd = np.sqrt(((Z - cent)**2).sum(1))
    lo, hi = dd[H_IDX].min(), dd[H_IDX].max()
    ins = (dd[L_IDX] >= lo) & (dd[L_IDX] <= hi)
    cl = DISPLAY.get(LLMS[int(np.argmin(dd[L_IDX]))])
    inside_models = ', '.join(DISPLAY.get(LLMS[i]) for i in np.where(ins)[0]) or '—'
    rowsE.append(dict(dropped_axis=DISP_AXIS[d], n_outside=int((~ins).sum()),
                      closest=cl, inside=inside_models))
    print(f'  drop {DISP_AXIS[d]:22s} outside={int((~ins).sum()):2d}/16  closest={cl:18s} inside: {inside_models}')
pd.DataFrame(rowsE).to_csv(OUT/'E_leave_one_axis_out.csv', index=False)

# ==========================================================================
print('\n' + '='*78)
print('[F] MAJOR 4 — sparse-axis support & support-weighted distance')
print('='*78)
support = (np.abs(DELTA) > 1e-12).sum(0)
print('  informative common pairs per axis (delta != 0):')
for i, d in enumerate(DIMS):
    print(f'    {DISP_AXIS[d]:22s} {support[i]:4d} / {NP_}')
w = np.sqrt(support / NP_); w = w / w.sum() * 5.0        # mean weight 1
sig = signatures_from_pairs(full_pairs)
mu = sig.mean(0); sd = sig.std(0); sd[sd == 0] = 1
Zw = ((sig - mu) / sd) * np.sqrt(w)
cent = Zw[H_IDX].mean(0)
dw = np.sqrt(((Zw - cent)**2).sum(1))
lo, hi = dw[H_IDX].min(), dw[H_IDX].max()
ins = (dw[L_IDX] >= lo) & (dw[L_IDX] <= hi)
order_w = [DISPLAY.get(LLMS[i]) for i in np.argsort(dw[L_IDX])]
rowsF = [dict(model=DISPLAY.get(k), dist_weighted=dw[L_IDX[i]],
              within=bool(ins[i])) for i, k in enumerate(LLMS)]
pd.DataFrame(rowsF).sort_values('dist_weighted').to_csv(OUT/'F_support_weighted_distance.csv', index=False)
print(f'\n  support-weighted (w ∝ sqrt(support)): outside={int((~ins).sum())}/16, '
      f'closest={order_w[0]}, then {order_w[1]}')
print('  top-4 order:', ' > '.join(order_w[:4]))

print('\nAll outputs saved to results/stat_review/*.csv')
