# -*- coding: utf-8 -*-
"""
print_all_tables_v10.py — print every manuscript table (T3-T13) under the v10
mapping + normalization, ready to copy into the paper.

Transcribed from Maya's working Jupyter snippets, adapted to:
  * v10 display axis names (paper_value_config_v10.DISPLAY_LABELS)
  * in-memory normalization (reads *_raw.csv if present, else the live file,
    then normalizes itself -> never double-normalizes)
  * T11 raw-SHAP and T12 mapping-sensitivity kept in native scale (controls).

Each table is wrapped so one failure does not kill the rest. Run standalone
(python print_all_tables_v10.py) or it is called by regenerate_everything_v10.py.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from paper_value_config_v10 import DISPLAY_LABELS
import paper_normalize_v10 as norm
from paper_normalize_v10 import DIMS, HUMANS

R = os.path.join(os.path.dirname(os.path.abspath(__file__)) or '.', 'results') + os.sep
LBL = [DISPLAY_LABELS[d] for d in DIMS]          # short headers in display order
SHORT = {d: DISPLAY_LABELS[d].split()[0][:8] for d in DIMS}

KEY2NAME = {
 'doctor8': 'GPT-4o-mini', 'doctor9': 'GPT-5-mini', 'doctor10': 'Llama-4 Maverick',
 'gpt_5': 'GPT-5', 'gpt_5_4': 'GPT-5.4', 'gpt_5_5': 'GPT-5.5',
 'claude_opus_4_5': 'Claude Opus 4.5', 'claude_opus_4_7': 'Claude Opus 4.7',
 'claude_opus_4_8': 'Claude Opus 4.8', 'gemini_2_5_pro': 'Gemini 2.5 Pro',
 'gemini_3_1_pro': 'Gemini 3.1 Pro', 'gemini_3_5_flash': 'Gemini 3.5 Flash',
 'llama_3_3_70b': 'Llama 3.3 70B', 'deepseek_v3_2': 'DeepSeek V3.2',
 'qwen3_5_27b': 'Qwen3.5 27B', 'mistral_medium_3_5': 'Mistral Medium 3.5',
}


def _read_raw(name):
    """Prefer the *_raw.csv backup (true pre-normalization values); else live file."""
    raw = R + name.replace('.csv', '_raw.csv')
    p = raw if os.path.exists(raw) else R + name
    return pd.read_csv(p)


def _hdr(num, title, note=''):
    print('\n' + '=' * 84)
    print(f'TABLE {num} — {title}' + (f'   [{note}]' if note else ''))
    print('=' * 84)


def _safe(fn, num, title):
    try:
        fn()
    except Exception as e:
        print(f'\n[TABLE {num} FAILED] {title}: {e}')


# ------------------------------------------------------------------ T13 / signatures
def t13_signatures():
    _hdr('13', 'Normalized operational value signatures (common pairs, 26 raters)',
         'z-score per axis')
    df = _read_raw('ethical_profiles_combined_top30.csv')
    pc = ['Pairs_' + d for d in DIMS]
    df = df.rename(columns={c: c.replace('Pairs_', '') for c in pc})
    n = norm.normalize_self(df, DIMS)
    if 'entity_type' not in n.columns:
        n['entity_type'] = n['doctor'].apply(lambda k: 'human' if k in HUMANS else 'llm')
    n['name'] = n['doctor'].map(lambda k: KEY2NAME.get(k, k))
    print(f"{'rater':22}{'type':10}" + ''.join(f'{l[:10]:>12}' for l in LBL))
    for _, r in n.iterrows():
        print(f"{r['name']:22}{r['entity_type']:10}" +
              ''.join(f"{r[d]:>12.3f}" for d in DIMS))


# ------------------------------------------------------------------ T3 Mann-Whitney
def t3_mannwhitney():
    _hdr('3', 'Human vs LLM per axis (Mann-Whitney, normalized common pairs)')
    df = _read_raw('ethical_profiles_combined_top30.csv')
    pc = ['Pairs_' + d for d in DIMS]
    df = df.rename(columns={c: c.replace('Pairs_', '') for c in pc})
    n = norm.normalize_self(df, DIMS); n['is_h'] = n.doctor.isin(HUMANS)
    print(f"{'Axis':30}{'humans':>9}{'LLMs':>9}{'U':>8}{'p':>11}")
    for d in DIMS:
        h = n[n.is_h][d]; l = n[~n.is_h][d]
        U, p = mannwhitneyu(h, l, alternative='two-sided')
        print(f"{DISPLAY_LABELS[d]:30}{h.mean():>9.3f}{l.mean():>9.3f}{U:>8.1f}{p:>11.2g}")


# ------------------------------------------------------------------ T4 distance+signature
def t4_distance():
    _hdr('4', 'Per-rater distance from human centroid + signature (normalized)')
    df = _read_raw('ethical_profiles_combined_top30.csv')
    pc = ['Pairs_' + d for d in DIMS]
    n = norm.normalize_self(df.rename(columns={c: c.replace('Pairs_', '') for c in pc}), DIMS)
    n['is_h'] = n.doctor.isin(HUMANS)
    cent = n[n.is_h][DIMS].mean().values
    n['dist'] = norm.distances(n, cent, DIMS)
    n['name'] = n.doctor.map(lambda k: KEY2NAME.get(k, k))
    print(f"{'Rater':22}{'Dist':>7}" + ''.join(f'{l[:8]:>10}' for l in LBL))
    print(f"{'Human centroid':22}{'--':>7}" + ''.join(f"{v:>10.3f}" for v in cent))
    for _, r in n[~n.is_h].sort_values('dist').iterrows():
        print(f"{r['name']:22}{r['dist']:>7.3f}" + ''.join(f"{r[d]:>10.3f}" for d in DIMS))


# ------------------------------------------------------------------ T5 leave-one-out
def t5_loo():
    _hdr('5', 'Leave-one-out physician distances (normalized)')
    df = _read_raw('ethical_profiles_combined_top30.csv')
    pc = ['Pairs_' + d for d in DIMS]
    n = norm.normalize_self(df.rename(columns={c: c.replace('Pairs_', '') for c in pc}), DIMS)
    H = n[n.doctor.isin(HUMANS)]
    loo = []
    for _, r in H.iterrows():
        others = H[H.doctor != r.doctor][DIMS].mean().values
        d = float(np.sqrt(((r[DIMS].values - others) ** 2).sum())); loo.append(d)
        print(f"  {r.doctor:18} {d:.3f}")
    print(f"  LOO physician range: {min(loo):.3f} to {max(loo):.3f}")


# ------------------------------------------------------------------ T6 top-N means
def t6_topN():
    _hdr('6', 'Top-N value means, human vs LLM (normalized within each N)')
    df = _read_raw('analysis1_topN_profiles.csv'); df['is_h'] = df.doctor.isin(HUMANS)
    for N in sorted(df.top_N.unique()):
        sub = norm.normalize_self(df[df.top_N == N].copy(), DIMS)
        sub['is_h'] = sub.doctor.isin(HUMANS)
        h = sub[sub.is_h][DIMS].mean(); l = sub[~sub.is_h][DIMS].mean()
        print(f"  Top-{N}: " + " | ".join(f"{SHORT[d]} H{h[d]:+.2f}/L{l[d]:+.2f}" for d in DIMS))


# ------------------------------------------------------------------ T7 discordant per-LLM
def t7_discordant():
    _hdr('7', 'Per-LLM discordant pairs (age/risk + top LLM-picked recs)', 'not value axes; raw')
    s = pd.read_csv(R + 'battleground_per_llm_summary.csv')
    from collections import Counter
    bl = pd.read_csv(R + 'battleground_per_llm.csv') if os.path.exists(R + 'battleground_per_llm.csv') else None
    print(f"{'LLM':20}{'n':>5}{'age_diff':>10}{'risk_diff':>10}  top LLM-picked recs")
    for _, r in s.iterrows():
        top = ''
        if bl is not None:
            c = Counter()
            for v in bl[bl.llm == r['llm']].llm_picked_only_recs.dropna():
                for x in str(v).split(','): c[x.strip()] += 1
            top = ", ".join(f"{k}({n})" for k, n in c.most_common(3))
        print(f"{r['llm']:20}{int(r['n_battleground']):>5}{r['mean_age_diff']:>10.1f}{r['mean_risk_diff']:>10.1f}  {top}")


# ------------------------------------------------------------------ T9 winner agreement
def t9_agreement():
    _hdr('9', 'Run-to-run winner agreement across reruns', 'agreement; raw')
    m = pd.read_csv(R + 'multi_seed_consistency.csv')
    mean = m[m.seed_a == 'MEAN']
    for _, r in mean.iterrows():
        print(f"  {KEY2NAME.get(r['model'], r['model']):20} mean agreement = {r['agreement']:.3f}")


# ------------------------------------------------------------------ T10 seed stability
def t10_seed():
    _hdr('10', 'Rerun signature stability per model (normalized; cp params + physician centroid)')
    seed = _read_raw('seed_profile_stability.csv')
    cp = _read_raw('analysis2_common_pairs_profiles.csv')
    params = norm.fit_params(cp, DIMS)
    cpn = norm.apply_params(cp, params, DIMS)
    phys_cent = norm.centroid(cpn, 'doctor', HUMANS, DIMS)
    sn = norm.apply_params(seed, params, DIMS)
    sn['dist'] = norm.distances(sn, phys_cent, DIMS)
    print(f"{'Model':20}{'dist mean(SD)':>16}" + ''.join(f'{SHORT[d]:>10}' for d in DIMS))
    for llm, g in sn.groupby('llm'):
        ds = f"{g['dist'].mean():.3f}({g['dist'].std(ddof=1):.3f})"
        sig = " ".join(f"{g[d].mean():+.2f}" for d in DIMS)
        print(f"{llm:20}{ds:>16}  {sig}")


# ------------------------------------------------------------------ T11 raw-SHAP (RAW)
def t11_rawshap():
    _hdr('11', 'Raw-SHAP distances from human centroid', 'RAW by design — robustness control')
    d = pd.read_csv(R + 'raw_shap_clustering_distances.csv').sort_values('distance_raw_shap')
    print(f"{'rater':22}{'family':12}{'raw-SHAP dist':>14}")
    for _, r in d.iterrows():
        print(f"{r['label']:22}{str(r['family']):12}{r['distance_raw_shap']:>14.4f}")


# ------------------------------------------------------------------ T12 mapping sensitivity (RAW)
def t12_sensitivity():
    _hdr('12', 'Mapping sensitivity M0-M3 (LLM distances)', 'RAW mapping-robustness control')
    ms = pd.read_csv(R + 'mapping_sensitivity_distances.csv')
    piv = ms[ms.entity_type == 'llm'].pivot_table(index='label', columns='mapping', values='distance')
    print(piv.round(3).to_string())


def main():
    print('#' * 84)
    print('# ALL MANUSCRIPT TABLES — v10 mapping + normalization')
    print('# axis names:', ' | '.join(LBL))
    print('#' * 84)
    _safe(t13_signatures, '13', 'signatures')
    _safe(t3_mannwhitney, '3', 'Mann-Whitney')
    _safe(t4_distance,    '4', 'distance+signature')
    _safe(t5_loo,         '5', 'leave-one-out')
    _safe(t6_topN,        '6', 'top-N means')
    _safe(t7_discordant,  '7', 'discordant per-LLM')
    _safe(t9_agreement,   '9', 'winner agreement')
    _safe(t10_seed,       '10', 'seed stability')
    _safe(t11_rawshap,    '11', 'raw-SHAP')
    _safe(t12_sensitivity,'12', 'mapping sensitivity')
    print('\n' + '#' * 84)
    print('# DONE. T11/T12 intentionally raw. Paste back anything that printed [FAILED].')
    print('#' * 84)


if __name__ == '__main__':
    main()
