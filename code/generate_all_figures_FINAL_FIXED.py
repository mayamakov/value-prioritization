"""
generate_all_figures_FINAL.py
=============================
Produces every manuscript figure that can be built from the result CSVs,
each saved with the EXACT manuscript filename (PNG + PDF).

Covers: Figure 2, Figure 3, S1, S2, S3, S4, S5, S6, S7a, S7b, S9, S9b.

NOT covered here (need raw artifacts; run their dedicated scripts):
  - Figure S6b : raw_shap is fine, but demographics needs patient_df + physicians_results/*.json
                 -> run figure_4_rank_by_demographics.py  (saves figS6b_priority_demographics)
  - Figure S8  : needs results/ethical_shap_per_patient.pkl
                 -> run raw_shap_clustering.py             (saves figS8_raw_shap_pca)
  - Figure 1   : study-design schematic, built by hand (not script-generated)

All numbering matches the Supplement legends exactly. No old/duplicate names.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.decomposition import PCA

# ==========================================================================
# CONFIG  — adjust these two paths to your environment if needed
# ==========================================================================
RESULTS_DIR = 'results'                 # where the result CSVs live
FIG_DIR     = 'results/figures'         # where figures are written
os.makedirs(FIG_DIR, exist_ok=True)

def R(fname):
    """Resolve a result file path (tries results/ then current dir)."""
    p = os.path.join(RESULTS_DIR, fname)
    return p if os.path.exists(p) else fname

# ==========================================================================
# Shared style / metadata
# ==========================================================================
# DIMS = the CSV column keys. DO NOT rename these: the result files
# (Pairs_*, SHAP_*, top-N, baseline) are still keyed by these strings.
# Their v10 MEANING is the new mapping -- 'Beneficence-Immediate' carries
# utility-oriented beneficence, 'Beneficence-LongTerm' carries priority-to-need.
DIMS  = ['Beneficence-Immediate', 'Beneficence-LongTerm', 'Non-maleficence', 'Autonomy', 'Justice']
# Short display labels (what the reader sees), aligned positionally to DIMS:
SHORT = ['Beneficence-\nutility', 'Beneficence-\nneed', 'Non-mal', 'Patient\nempower', 'Resource\njustice']
# Full display names for titles, keyed by the DIMS column name:
DISP_DIM = {
    'Beneficence-Immediate': 'Beneficence-utility',
    'Beneficence-LongTerm':  'Beneficence-need',
    'Non-maleficence':       'Non-maleficence',
    'Autonomy':              'Patient empowerment',
    'Justice':               'Resource justice',
}
from panel_config import (MODELS as LLM_ORDER, HUMANS, DISPLAY as DISP,
                          MODEL_FAMILY as FAM, FAMILY_COLOR as FAM_COL)
HUMAN_COL = '#888888'

# ---- Anonymized physician labels (FM-1..FM-5, PH-1..PH-5) from the keymap ----
# Set LABEL_STYLE='long' to get 'Family Medicine 1' / 'Public Health 1' instead.
LABEL_STYLE = 'short'

def _build_anon():
    try:
        km = pd.read_csv(R('rater_keymap.csv'))
        d = dict(zip(km['rater_key'], km['display_name']))
    except Exception:
        d = {'fm_1': 'FM-1', 'fm_2': 'FM-2', 'fm_3': 'FM-3',
             'fm_4': 'FM-4', 'fm_5': 'FM-5', 'ph_1': 'PH-1',
             'ph_3': 'PH-3', 'ph_4': 'PH-4', 'ph_2': 'PH-2', 'ph_5': 'PH-5'}
    if LABEL_STYLE == 'long':
        d = {k: v.replace('FM-', 'Family Medicine ').replace('PH-', 'Public Health ')
             for k, v in d.items()}
    return d

ANON = _build_anon()

def hlabel(rater_key):
    """Anonymized display label for any rater (physicians -> FM-/PH-, LLMs -> model name)."""
    if rater_key in ANON:
        return ANON[rater_key]
    return DISP.get(rater_key, rater_key)

def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {name}.png/.pdf')

# Common-pairs per-rater profiles (Pairs_* columns of the combined file)
_comb  = pd.read_csv(R('ethical_profiles_combined_top30.csv')).set_index('doctor')
PAIRS  = _comb[[f'Pairs_{d}' for d in DIMS]].copy(); PAIRS.columns = DIMS

# ==========================================================================
# Figure 2 — per-LLM operational profiles (spider)
# ==========================================================================
def figure_2():
    DV = ['Beneficence-LongTerm', 'Non-maleficence', 'Autonomy', 'Justice', 'Beneficence-Immediate']
    DL = ['Beneficence-\nneed', 'Non-mal', 'Patient\nempower', 'Resource\njustice', 'Beneficence-\nutility']
    import math as _m
    TITLES = {k: f"{DISP[k]}\n({FAM[k]})" for k in LLM_ORDER}
    # --- per-axis min-max normalization across ALL raters (display only) ---
    # each ethical axis is on a different native scale (Priority=raw risk ~5-6,
    # others ~0.1); normalize each axis to [0,1] so the pentagon is balanced.
    _allr = list(HUMANS) + list(LLM_ORDER)
    _sub = PAIRS.loc[_allr, DV]
    _lo = _sub.min(); _rng = (_sub.max() - _sub.min()).replace(0, 1.0)
    PAIRS_N = (PAIRS[DV] - _lo) / _rng     # normalized copy for plotting
    hc = PAIRS_N.loc[HUMANS].mean().values
    _nc=6; _nr=_m.ceil(len(LLM_ORDER)/_nc)   # 6 columns x 3 rows per Noa's request
    fig, axes = plt.subplots(_nr, _nc, figsize=(4.8*_nc, 4.2*_nr), subplot_kw=dict(projection='polar')); axes = axes.flatten()
    ang = np.linspace(0, 2 * np.pi, len(DV), endpoint=False).tolist(); ang += ang[:1]
    rmax = 1.0    # all axes normalized to [0,1]
    hv = np.clip(hc, 0, 1)
    for ax, k in zip(axes, LLM_ORDER):
        hcl = list(hv) + [hv[0]]
        ax.plot(ang, hcl, color=HUMAN_COL, lw=1.8, label='Physician mean'); ax.fill(ang, hcl, color=HUMAN_COL, alpha=0.18)
        v = np.clip(PAIRS_N.loc[k].values, 0, 1); c = FAM_COL[FAM[k]]; vl = list(v) + [v[0]]
        ax.plot(ang, vl, color=c, lw=2.5, label=DISP[k]); ax.fill(ang, vl, color=c, alpha=0.30)
        ax.set_xticks(ang[:-1]); ax.set_xticklabels(DL, size=9); ax.set_ylim(0, rmax)
        ax.set_yticks([rmax * .33, rmax * .66, rmax]); ax.set_yticklabels(['', '', ''])
        ax.grid(alpha=0.4); ax.set_title(TITLES[k], size=11, weight='bold', pad=15)
        ax.legend(loc='upper right', fontsize=8, frameon=True, bbox_to_anchor=(1.20, 1.10))
    for _ax in axes[len(LLM_ORDER):]: _ax.set_visible(False)
    # title removed from image (lives in the Word caption instead, per Noa's comment)
    plt.tight_layout(); save(fig, 'figure_2_per_llm_profiles')

# ==========================================================================
# Figure 3 — battleground pairs (2-panel)
# ==========================================================================
def figure_3():
    bg = pd.read_csv(R('battleground_per_llm_summary.csv'))
    bg['disp'] = bg['llm_key'].map(DISP); bg['col'] = bg['llm_key'].map(FAM)
    pc = pd.read_csv(R('pair_consensus.csv'))
    denom = int((pc['majority_size_human'] >= 7).sum())   # physician-majority pairs
    bg['prop'] = bg['n_battleground'] / denom * 100.0
    bg = bg.sort_values('n_battleground', ascending=False)   # fewest discordant at TOP of barh
    cols = [FAM_COL[c] for c in bg['col']]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
    a1.barh(bg['disp'], bg['prop'], color=cols, edgecolor='black', lw=0.6)
    for i, p in enumerate(bg['prop']):
        a1.text(p + 0.6, i, f'{p:.0f}%', va='center', size=9)
    a1.set_xlabel(f'Physician-majority pairs opposed (%)  (of {denom} pairs, \u22657/10 physician agreement)')
    a1.set_title('A. Discordant pairs per LLM', weight='bold', loc='left')
    x = np.arange(len(bg)); w = 0.38
    fam_cols = [FAM_COL[c] for c in bg['col']]   # same family colors as panel A
    a2.bar(x - w / 2, bg['mean_age_diff'], w,
           color=fam_cols, edgecolor='black', lw=0.5)
    a2.bar(x + w / 2, bg['mean_risk_diff'], w,
           color=fam_cols, edgecolor='black', lw=0.5, hatch='///')
    a2.axhline(0, color='black', lw=0.8)
    a2.set_xticks(x); a2.set_xticklabels(bg['disp'], rotation=30, ha='right', size=9)
    a2.set_ylabel('LLM-preferred minus physician-preferred\n(positive = LLM preferred older / higher-risk patients)')
    from matplotlib.patches import Patch
    a2.legend(handles=[
        Patch(facecolor='#cccccc', edgecolor='black', label='Mean age difference (yr)'),
        Patch(facecolor='#cccccc', edgecolor='black', hatch='///', label='Mean 10-yr CV risk difference (pts)')],
        fontsize=9)
    a2.set_title('B. Patient characteristics in discordant pairs', weight='bold', loc='left')
    plt.tight_layout(); save(fig, 'figure_3_battleground')

# ==========================================================================
# Figure S1 — pairwise agreement matrix
# ==========================================================================
def figure_S1():
    ag = pd.read_csv(R('agreement_pairwise.csv'), index_col=0)
    # fix any stale model label in the axis names
    ag = ag.rename(index={'GPT-4-mini': 'GPT-4o-mini'}, columns={'GPT-4-mini': 'GPT-4o-mini'})
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(ag.values, cmap='RdYlGn', vmin=0.3, vmax=1.0)
    ax.set_xticks(range(len(ag.columns))); ax.set_xticklabels(ag.columns, rotation=90, size=8)
    ax.set_yticks(range(len(ag.index))); ax.set_yticklabels(ag.index, size=8)
    for i in range(len(ag.index)):
        for j in range(len(ag.columns)):
            ax.text(j, i, f'{ag.values[i, j]:.2f}', ha='center', va='center', size=6,
                    color='black' if 0.45 < ag.values[i, j] < 0.92 else 'white')
    plt.colorbar(im, label='Pairwise winner agreement')
    # title removed from image (lives in the Word caption instead, per Noa's comment)
    plt.tight_layout(); save(fig, 'figure_S1_agreement_matrix')

# ==========================================================================
# Figure S2 — three-method comparison
# ==========================================================================
def figure_S2():
    if _comb[[f'SHAP_{d}' for d in DIMS]].notna().any().any() is np.bool_(False) or \
       _comb[[f'SHAP_{d}' for d in DIMS]].isna().all().all():
        print('  ! Skip S2 - SHAP_ columns empty (needs the SHAP/training pipeline)'); return
    # Panels: SHAP-based & Common-pairs from the combined file; Top 10% from analysis1_topN.
    topn = pd.read_csv(R('analysis1_topN_profiles.csv'))
    t50 = topn[topn['top_N'] == 50].copy()
    t50_grp = np.where(t50['entity_type'] == 'human', 'human', 'llm')
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(DIMS)); w = 0.38

    def panel(ax, title, hum, llm):
        ax.bar(x - w / 2, hum, w, label='Physicians', color=HUMAN_COL, edgecolor='black', lw=0.5)
        ax.bar(x + w / 2, llm, w, label='LLMs (n=16)', color='#4285f4', edgecolor='black', lw=0.5)
        ax.axhline(0, color='black', lw=0.6); ax.set_xticks(x)
        ax.set_xticklabels(SHORT, rotation=30, ha='right', size=9)
        ax.set_title(title, weight='bold'); ax.legend(fontsize=8)

    # (i) SHAP-based
    cols = [f'SHAP_{d}' for d in DIMS]
    panel(axes[0], 'A. SHAP-based',
          _comb.loc[HUMANS, cols].mean().values, _comb.loc[LLM_ORDER, cols].mean().values)
    # (ii) Top 10% patients (top_N=50 of 500 test patients)
    panel(axes[1], 'B. Top 10% patients',
          t50[t50_grp == 'human'][DIMS].mean().values, t50[t50_grp == 'llm'][DIMS].mean().values)
    # (iii) Common pairs
    cols = [f'Pairs_{d}' for d in DIMS]
    panel(axes[2], 'C. Common pairs',
          _comb.loc[HUMANS, cols].mean().values, _comb.loc[LLM_ORDER, cols].mean().values)

    axes[0].set_ylabel('Mean operational weight')
    plt.suptitle('Three-method comparison: physician vs LLM operational signatures', weight='bold', y=1.02)
    plt.tight_layout(); save(fig, 'figure_S2_three_methods')

# ==========================================================================
# Figure S3 — consensus distribution across the 300 common pairs
# (reads results/pair_consensus.csv with majority_size_all/human/llm columns)
# ==========================================================================
def figure_S3():
    con_path = R('pair_consensus.csv')
    if not os.path.exists(con_path):
        print('  ! Skip S3 - pair_consensus.csv not found'); return
    df_con = pd.read_csv(con_path)
    _na=len(HUMANS)+len(LLM_ORDER)
    panels = [(f'A. All {_na} raters', _na, 'majority_size_all', '#444444'),
              (f'B. {len(HUMANS)} physicians only', len(HUMANS), 'majority_size_human', '#1f77b4'),
              (f'C. {len(LLM_ORDER)} LLMs only', len(LLM_ORDER), 'majority_size_llm', '#cc0066')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, total, col, color) in zip(axes, panels):
        if col not in df_con.columns:
            ax.text(0.5, 0.5, f'Column "{col}" missing', ha='center', va='center',
                    transform=ax.transAxes, color='red'); ax.set_title(title, weight='bold'); continue
        values = df_con[col].dropna().astype(int)
        min_maj = int(np.ceil(total / 2)); x = list(range(min_maj, total + 1))
        counts = [int((values == m).sum()) for m in x]
        bars = ax.bar(x, counts, color=color, alpha=0.75, edgecolor='black', lw=0.5)
        bars[-1].set_alpha(1.0); bars[-1].set_edgecolor('#cc0066'); bars[-1].set_linewidth(2.0)
        for xi, cnt in zip(x, counts):
            if cnt > 0:
                ax.text(xi, cnt + max(counts) * 0.02, str(cnt), ha='center', va='bottom', size=9)
        ax.set_xlabel(f'Majority size (out of {total})')
        if ax is axes[0]:
            ax.set_ylabel('Number of common pairs')
        ax.set_title(title, weight='bold'); ax.set_xticks(x); ax.grid(axis='y', alpha=0.3)
        unan = counts[-1]; tot = sum(counts); pct = 100 * unan / tot if tot else 0
        ax.text(0.95, 0.95, f'Unanimous: {unan}/{tot}\n({pct:.0f}%)', transform=ax.transAxes,
                va='top', ha='right', size=10, weight='bold', color='#cc0066',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.95))
    fig.suptitle('Consensus distribution across the 300 common pairs', weight='bold', y=1.03)
    plt.tight_layout(); save(fig, 'figure_S3_consensus_distribution')

# ==========================================================================
# Figure S4 — full common-pairs operational-signature heatmap
# ==========================================================================
def figure_S4():
    order = HUMANS + LLM_ORDER
    M = PAIRS.loc[order, DIMS].values
    rownames = [hlabel(r) for r in HUMANS] + [DISP[k] for k in LLM_ORDER]
    fig, ax = plt.subplots(figsize=(8, 9))
    norm = TwoSlopeNorm(vmin=M.min(), vcenter=0, vmax=M.max())
    im = ax.imshow(M, cmap='RdBu_r', norm=norm, aspect='auto')
    ax.set_xticks(range(len(DIMS))); ax.set_xticklabels(SHORT, rotation=30, ha='right', size=9)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(rownames, size=8)
    ax.axhline(9.5, color='black', lw=1.5)
    for i in range(len(order)):
        for j in range(len(DIMS)):
            ax.text(j, i, f'{M[i, j]:+.2f}', ha='center', va='center', size=6, color='black')
    plt.colorbar(im, label='Common-pairs operational weight')
    pass  # ax.set_title('Operational value-signature heatmap (all 26 raters)', weight='bold')
    plt.tight_layout(); save(fig, 'figure_S4_heatmap_common_pairs')

# ==========================================================================
# Figure S5 — PCA of the five-dimensional common-pairs signatures
# ==========================================================================
def figure_S5():
    order = [r for r in (HUMANS + LLM_ORDER) if not PAIRS.loc[r, DIMS].isna().any()]
    X = PAIRS.loc[order, DIMS].values
    p = PCA(n_components=2).fit(X); Y = p.transform(X); ev = p.explained_variance_ratio_ * 100
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, r in enumerate(order):
        if r in HUMANS:
            ax.scatter(Y[i, 0], Y[i, 1], c=HUMAN_COL, s=90, edgecolor='black', zorder=3)
        else:
            ax.scatter(Y[i, 0], Y[i, 1], c=FAM_COL[FAM[r]], s=160, marker='D', edgecolor='black', zorder=4)
            ax.annotate(DISP[r], (Y[i, 0], Y[i, 1]), fontsize=8, xytext=(6, 4), textcoords='offset points')
    hc = Y[[order.index(h) for h in HUMANS]].mean(axis=0)
    ax.scatter(hc[0], hc[1], marker='*', s=420, c='#333333', edgecolor='black', zorder=5, label='Physician centroid')
    ax.set_xlabel(f'PC1 ({ev[0]:.1f}%)'); ax.set_ylabel(f'PC2 ({ev[1]:.1f}%)')
    ax.set_title('PCA of five-dimensional common-pairs signatures', weight='bold')
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout(); save(fig, 'figure_S5_pca_map')

# ==========================================================================
# Figure S6 — OpenAI scaling trajectory
# ==========================================================================
def figure_S6():
    openai = [k for k in LLM_ORDER if FAM[k]=='OpenAI']
    fig, ax = plt.subplots(figsize=(10, 6)); x = np.arange(len(DIMS))
    for k in openai:
        ax.plot(x, PAIRS.loc[k, DIMS].values, marker='o', lw=2, label=DISP[k])
    ax.plot(x, PAIRS.loc[HUMANS, DIMS].mean().values, marker='s', lw=2, ls='--',
            color=HUMAN_COL, label='Physician mean')
    ax.axhline(0, color='black', lw=0.6); ax.set_xticks(x); ax.set_xticklabels(SHORT, size=10)
    ax.set_ylabel('Common-pairs operational weight')
    ax.set_title('OpenAI models across operational dimensions\n'
                 '(value signatures are highly consistent within the developer family)', weight='bold')
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout(); save(fig, 'figure_S6_openai_trajectory')

# ==========================================================================
# Figure S7a / S7b — seed stability
# ==========================================================================
def figure_S7():
    ss = pd.read_csv(R('seed_profile_stability.csv'))
    name2fam = {DISP[k]: FAM[k] for k in LLM_ORDER}
    # every model present in the CSV, in panel_config order
    present = [DISP[k] for k in LLM_ORDER if DISP[k] in set(ss.llm)]
    DV = ['Beneficence-LongTerm', 'Non-maleficence', 'Autonomy', 'Justice', 'Beneficence-Immediate']
    DL = ['Beneficence-\nneed', 'Non-mal', 'Patient\nempower', 'Resource\njustice', 'Beneficence-\nutility']
    ang = np.linspace(0, 2 * np.pi, len(DV), endpoint=False).tolist(); ang += ang[:1]
    # --- per-axis min-max normalization across physician centroid + all seeds ---
    # each ethical axis has a different native scale (Priority=raw risk ~5-6,
    # others ~0.1); normalize each axis to [0,1] so the pentagon stays balanced.
    _phys_raw = PAIRS.loc[HUMANS, DV].mean().values.astype(float)
    _seed_raw = ss[DV].values.astype(float)
    _stack = np.vstack([_phys_raw[None, :], _seed_raw])
    _lo = _stack.min(axis=0); _hi = _stack.max(axis=0)
    _rng = np.where((_hi - _lo) == 0, 1.0, (_hi - _lo))
    def _norm_row(arr):
        return np.clip((np.asarray(arr, dtype=float) - _lo) / _rng, 0, 1)
    phys = _norm_row(_phys_raw)
    rmax = 1.0
    # S7a: one polar panel per model, dynamic grid (4 columns)
    n = len(present); ncol = 4; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.5 * nrow),
                             subplot_kw=dict(projection='polar'))
    axes = np.atleast_1d(axes).ravel()
    for ax, m in zip(axes, present):
        pcl = list(phys) + [phys[0]]
        ax.plot(ang, pcl, color=HUMAN_COL, lw=1.6, ls='--', label='Physician')
        ax.fill(ang, pcl, color=HUMAN_COL, alpha=0.12)
        base = FAM_COL[name2fam[m]]
        for _, r in ss[ss.llm == m].iterrows():
            v = _norm_row(r[DV].values); vl = list(v) + [v[0]]
            ax.plot(ang, vl, lw=1.4, alpha=0.85, color=base, label=f"run {int(r['seed'])}")
            ax.fill(ang, vl, alpha=0.08, color=base)
        ax.set_xticks(ang[:-1]); ax.set_xticklabels(DL, size=6.5); ax.set_ylim(0, rmax)
        ax.set_yticklabels([]); ax.set_title(m, weight='bold', size=9, pad=8)
    for ax in axes[n:]:
        ax.axis('off')
    plt.suptitle('Operational-signature stability across reruns (vs physician centroid)', weight='bold', y=1.005)
    plt.tight_layout(); save(fig, 'figure_S7a_seed_stability_spider')

    # S7b: one bar per LLM-seed, grouped & colored by family
    fig, ax = plt.subplots(figsize=(max(9, 0.30 * len(ss) + 2), 5.5))
    pos = 0; centers = []
    for m in present:
        g = ss[ss.llm == m].sort_values('seed'); base = FAM_COL[name2fam[m]]
        start = pos
        for _, r in g.iterrows():
            ax.bar(pos, r['distance_from_physician_centroid'], 0.8,
                   color=base, edgecolor='black', lw=0.4)
            pos += 1
        centers.append((start + (len(g) - 1) / 2, m)); pos += 0.8
    ax.set_xticks([c for c, _ in centers])
    ax.set_xticklabels([m for _, m in centers], rotation=45, ha='right', size=7.5)
    ax.set_ylabel('Distance from physician centroid')
    ax.set_title('Per-run distance from physician centroid', weight='bold')
    ax.grid(alpha=0.3, axis='y'); plt.tight_layout(); save(fig, 'figure_S7b_seed_distances')

# ==========================================================================
# Figure S9 — sensitivity to alternative Beauchamp-Childress mappings
# ==========================================================================
def figure_S9():
    md = pd.read_csv(R('mapping_sensitivity_distances.csv'))
    llm = md[md.entity_type == 'llm'].copy(); hum = md[md.entity_type == 'human']
    maps = ['M0_original', 'M1_statin_harm', 'M2_pcsk9_neutral', 'M3_perturbed']
    mlab = ['M0\n(original)', 'M1\n(statin harm)', 'M2\n(PCSK9 neutral)', 'M3\n(perturbed)']
    fig, ax = plt.subplots(figsize=(10, 6))
    lo = [hum[hum.mapping == m]['distance'].min() for m in maps]
    hi = [hum[hum.mapping == m]['distance'].max() for m in maps]
    ax.fill_between(range(4), lo, hi, color=HUMAN_COL, alpha=0.18, label='Physician range')
    for k in LLM_ORDER:
        _v=[llm[(llm.mapping == m) & (llm.rater == k)]['distance'].values for m in maps]
        if any(len(x)==0 for x in _v): continue
        ys=[x[0] for x in _v]
        ax.plot(range(4), ys, marker='o', lw=2, color=FAM_COL[FAM[k]], label=DISP[k])
    ax.set_xticks(range(4)); ax.set_xticklabels(mlab, size=9)
    ax.set_ylabel('Distance from physician centroid')
    ax.set_title('Sensitivity to alternative Beauchamp-Childress mappings\n'
                 '(Gemini-family models remain closest under every mapping)', weight='bold')
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3); plt.tight_layout()
    save(fig, 'figure_S9_mapping_sensitivity')

# ==========================================================================
# Figure S9b — random-rater null distribution
# ==========================================================================
def figure_S9b():
    rb = pd.read_csv(R('sensitivity_random_baseline.csv'))
    # tolerate a baseline CSV keyed by the new full names by mapping back to DIMS keys
    rb = rb.rename(columns={v: k for k, v in DISP_DIM.items()})
    # SCALE FIX: sensitivity_random_baseline.csv is written RAW (pre-normalization)
    # by rebuild_pair_csvs.py, whereas the module-level PAIRS table is z-scored by
    # the orchestrator (Stage C). Overlaying normalized means on a raw histogram is
    # meaningless. So here we overlay the RAW observed means, read from the *_raw.csv
    # backup the orchestrator leaves behind; on the raw scale the coin-flip null sits
    # near zero, and physician/LLM means fall on opposite sides on priority-to-need.
    raw_comb = R('ethical_profiles_combined_top30_raw.csv')
    if os.path.exists(raw_comb):
        _src = pd.read_csv(raw_comb).set_index('doctor')
    else:
        _src = pd.read_csv(R('ethical_profiles_combined_top30.csv')).set_index('doctor')
        print('  ! figure_S9b: *_raw.csv not found; overlay means may be on a different '
              'scale than the raw null. Re-run the orchestrator so Stage C writes the backup.')

    def _raw_mean(group, d):
        col = f'Pairs_{d}'
        rows = [r for r in group if r in _src.index]
        return float(_src.loc[rows, col].mean())

    fig, axes = plt.subplots(2, 3, figsize=(16, 9)); axes = axes.flatten()
    for ax, d in zip(axes, DIMS):   # all five dimensions
        ax.hist(rb[d], bins=40, color='#bbbbbb', edgecolor='white')
        lo, hi = np.percentile(rb[d], 2.5), np.percentile(rb[d], 97.5)
        ax.axvline(lo, color='red', ls='--', lw=1.4)
        ax.axvline(hi, color='red', ls='--', lw=1.4, label='95% null interval')
        hmean = _raw_mean(HUMANS, d); lmean = _raw_mean(LLM_ORDER, d)
        ax.axvline(hmean, color=HUMAN_COL, lw=2.5, label=f'Physician mean ({hmean:+.2f})')
        ax.axvline(lmean, color='#4285f4', lw=2.5, label=f'LLM mean ({lmean:+.2f})')
        ax.set_title(f'{DISP_DIM.get(d, d)}\n95% null = [{lo:+.3f}, {hi:+.3f}]', weight='bold', size=10)
        ax.legend(fontsize=7)
        ax.set_xlabel('Operational weight (raw)'); ax.set_ylabel('Random raters (n=500)')
    axes[-1].axis('off')   # 6th cell unused (5 dimensions)
    plt.suptitle('Random-rater null distribution (five-dimensional operational signature)',
                 weight='bold', y=1.00)
    plt.tight_layout(); save(fig, 'figure_S9b_random_baseline')

# ==========================================================================
if __name__ == '__main__':
    print('Generating manuscript figures ->', FIG_DIR)
    for fn in (figure_2, figure_3, figure_S1, figure_S2, figure_S3, figure_S4,
               figure_S5, figure_S6, figure_S7, figure_S9, figure_S9b):
        try:
            fn()
        except Exception as e:
            print(f'  !! {fn.__name__} failed: {e}')
    print('\nDone.')
    print('Remember to also run (they need raw artifacts):')
    print('   python raw_shap_clustering.py          -> figure_S8_raw_shap_pca')
    print('   %run figure_4_rank_by_demographics.py  -> figure_S6b_priority_demographics')