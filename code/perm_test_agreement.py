# -*- coding: utf-8 -*-
"""
perm_test_agreement.py — significance of within- vs between-group pairwise agreement
(physicians vs LLMs) on the 300 common pairs, via a rater-label permutation test.

Reproduces the manuscript numbers:
  within-physician 0.6971, within-LLM 0.8477, physician-LLM 0.5478
and tests:
  T1 = within-physician − between   (obs 0.1493, p ≈ 3.4e-3)
  T2 = within-LLM      − between   (obs 0.2999, p = 1e-6 floor at 1e6 perms)
  T3 = pooled within   − between   (obs 0.2588, p = 1e-6 floor at 1e6 perms)
Complementary per-rater view: 9/10 physicians and 16/16 LLMs agree more with
their own group (two-sided binomial p = 0.021 and 3.1e-5).

Run from the repo root:  python code/perm_test_agreement.py
"""
import json, itertools, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from panel_config import HUMANS, MODELS, PANEL, LEGACYSET

PR = Path(__file__).resolve().parent.parent / 'data' / 'physicians_results'
if not PR.exists():
    PR = Path('physicians_results')  # legacy layout

NPERM = 1_000_000
SEED = 2026

def _n(it):
    if isinstance(it, (list, tuple)) and len(it) == 2 and isinstance(it[0], (list, tuple)):
        return (int(it[0][0]), int(it[0][1]))
    if isinstance(it, (list, tuple)) and len(it) >= 2:
        return (int(it[0]), int(it[1]))
    return None

def _l(p):
    p = Path(p)
    return [x for x in (_n(z) for z in json.load(open(p))) if x] if p.exists() else []

def choices(k):
    d = PR / k
    pr = []
    for i in range(4):
        pr += _l(d / f'{k}_train_iter_{i}_ranked.json')
    if k in LEGACYSET:
        pr += _l(d / f'{k}_test_ranked.json')
    else:
        got = False
        for pt in 'AB':
            t = _l(d / f'{k}_test_part_{pt}_ranked.json'); pr += t; got = got or bool(t)
        if not got:
            pr += _l(d / f'{k}_test_ranked.json')
    return {tuple(sorted(p)): p[0] for p in pr}

def main():
    C = {k: choices(k) for k in PANEL}
    present = [k for k in PANEL if C[k]]
    common = sorted(set.intersection(*[set(C[k]) for k in present]))
    print(f'raters={len(present)}  common pairs={len(common)}')

    idx = {k: i for i, k in enumerate(present)}
    n = len(present)
    M = np.zeros((n, n))
    for a, b in itertools.combinations(present, 2):
        v = float(np.mean([C[a][u] == C[b][u] for u in common]))
        M[idx[a], idx[b]] = M[idx[b], idx[a]] = v

    H = np.array([idx[k] for k in HUMANS]); L = np.array([idx[k] for k in MODELS])

    def three(h, l):
        hh = M[np.ix_(h, h)][np.triu_indices(len(h), 1)]
        ll = M[np.ix_(l, l)][np.triu_indices(len(l), 1)]
        hl = M[np.ix_(h, l)].mean()
        return hh.mean() - hl, ll.mean() - hl, np.concatenate([hh, ll]).mean() - hl

    hh = M[np.ix_(H, H)][np.triu_indices(10, 1)].mean()
    ll = M[np.ix_(L, L)][np.triu_indices(16, 1)].mean()
    hl = M[np.ix_(H, L)].mean()
    print(f'within-physician={hh:.4f}  within-LLM={ll:.4f}  between={hl:.4f}')

    T = three(H, L)
    rng = np.random.default_rng(SEED)
    exc = np.zeros(3, dtype=int)
    ai = np.arange(n)
    for _ in range(NPERM):
        p = rng.permutation(ai)
        t = three(p[:10], p[10:])
        exc += [t[i] >= T[i] for i in range(3)]
    for name, t, c in zip(['T1 physician', 'T2 LLM', 'T3 pooled'], T, exc):
        print(f'{name}: obs={t:.4f}  p={(c + 1) / (NPERM + 1):.2e}  (exceedances={c}/{NPERM})')

    wins = sum(np.mean([M[i, j] for j in H if j != i]) > np.mean([M[i, j] for j in L]) for i in H)
    winsL = sum(np.mean([M[i, j] for j in L if j != i]) > np.mean([M[i, j] for j in H]) for i in L)
    print(f'per-physician own-group preference: {wins}/10;  per-LLM: {winsL}/16')

if __name__ == '__main__':
    main()
