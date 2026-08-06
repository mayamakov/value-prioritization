"""
============================================================================
paper_figures_CORRECTED.py   (v10 — aligned to the revised value mapping)
============================================================================
Publication-grade figures for the Nature Medicine submission.

v10 changes vs the original:
  * Axis names come from paper_value_config_v10.DISPLAY_LABELS
    (Utility-oriented / Priority-to-need / Patient empowerment / Resource justice).
  * group_of / colours are built DYNAMICALLY from the panel, so all 16 LLMs are
    covered (was hard-coded to 6 -> KeyError 'Claude Opus 4.5').
  * Every group/colour lookup uses .get(...) with a neutral fallback -> the
    script can no longer crash on an unmapped rater.
  * LLM grids (Fig 5 spiders) size themselves to however many LLMs are present.
  * Reads the (already normalized) profile CSVs produced by Stage C.

Style: 300 DPI PNG + vector PDF, DejaVu Sans, family colours, panel labels.
Outputs to results/paper_figures/
============================================================================
"""

# ===== ANONYMIZATION =====
_PHYS_NAME_MAP = {
    'ph_1': 'Public Health 1',
    'ph_2': 'Public Health 2',
    'ph_3': 'Public Health 3',
    'ph_4': 'Public Health 4',
    'ph_5': 'Public Health 5',
    'fm_1': 'Family Medicine 1',
    'fm_2': 'Family Medicine 2',
    'fm_3': 'Family Medicine 3',
    'fm_4': 'Family Medicine 4',
    'fm_5': 'Family Medicine 5',
}
def ANONYMIZE(name):
    """Map real physician keys to anonymous codes; pass LLM keys through."""
    return _PHYS_NAME_MAP.get(name, name)
# =========================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# ---- v10 axis names (single source of truth) -------------------------------
try:
    from paper_value_config_v10 import DISPLAY_LABELS
except Exception:
    DISPLAY_LABELS = {
        'Beneficence-Immediate': 'Beneficence-utility',
        'Beneficence-LongTerm':  'Beneficence-need',
        'Non-maleficence':       'Non-maleficence',
        'Autonomy':              'Patient empowerment',
        'Justice':               'Resource justice',
    }

# ============================================================================
# STYLE
# ============================================================================
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         9,
    'axes.titlesize':    11,
    'axes.labelsize':    10,
    'axes.titleweight':  'bold',
    'axes.linewidth':    0.8,
    'axes.edgecolor':    '#333333',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'xtick.color':       '#333333',
    'ytick.color':       '#333333',
    'legend.fontsize':   8,
    'legend.frameon':    False,
    'figure.dpi':        100,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.facecolor': 'white',
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
})

SINGLE_COL = 3.5
DOUBLE_COL = 7.2
DOUBLE_COL_TALL = 7.2

# ============================================================================
# CONSTANTS  (internal keys frozen = v9; DISPLAY names = v10)
# ============================================================================
ETHICAL_DIMENSIONS = [
    'Beneficence-Immediate',   # -> Utility-oriented beneficence
    'Beneficence-LongTerm',    # -> Priority-to-need beneficence
    'Non-maleficence',
    'Autonomy',                # -> Patient empowerment
    'Justice',                 # -> Resource justice
]
# Short + 2-line labels DERIVED from the v10 display names (no stale 'Bene-Imm')
def _short(lbl):
    w = lbl.split()
    return w[0][:8] if len(w) == 1 else (w[0][:4] + '.' if w[0].lower() != 'non-maleficence' else 'Non-mal')
DIM_LABELS_SHORT = ['Beneficence-utility', 'Beneficence-need', 'Non-mal', 'Empower', 'Justice']
DIM_LABELS_2LINE = [
    'Beneficence-\nutility',
    'Beneficence-\nneed',
    'Non-\nmaleficence',
    'Patient\nempowerment',
    'Resource\njustice',
]
DIM_LABELS_FULL = [DISPLAY_LABELS[d] for d in ETHICAL_DIMENSIONS]

# ---- panel (all 16 LLMs) ---------------------------------------------------
import panel_config as _pc
_MARK = {'OpenAI': 'o', 'Anthropic': 's', 'Google': '^', 'Meta': 'D',
         'DeepSeek': 'P', 'Qwen': 'X', 'Mistral': '*'}
LLM_META = {
    k: dict(label=_pc.display(k), family=_pc.family(k),
            size=('mini' if k in _pc.SMALL_TIER else 'frontier'),
            marker=_MARK.get(_pc.family(k), 'o'),
            size_pt=(80 if k in _pc.SMALL_TIER else 150))
    for k in _pc.MODELS
}

FAMILY_COLORS = {
    'OpenAI':    '#10a37f',
    'Anthropic': '#d97706',
    'Google':    '#4285f4',
    'Meta':      '#1f3a93',
    'DeepSeek':  '#6d28d9',
    'Qwen':      '#db2777',
    'Mistral':   '#ea580c',
    'Humans':    '#444444',
}
NEUTRAL = '#333333'
def fam_color(fam):
    return FAMILY_COLORS.get(fam, NEUTRAL)
HUMAN_DOT_COLOR = '#7f7f7f'

HUMAN_DOCTORS = [
    'fm_1', 'fm_2', 'fm_3', 'fm_4', 'fm_5',
    'ph_1', 'ph_2', 'ph_3', 'ph_4', 'ph_5',
]
LLM_KEYS = list(LLM_META.keys())
LEGACY_LLMS = ('doctor8', 'doctor9', 'doctor10')

# ---- group map built DYNAMICALLY -> covers ALL 16 LLMs ----------------------
group_order = ['family_medicine', 'public_health', 'llm_legacy', 'llm_new']
group_of = {
    'fm_1': 'family_medicine', 'fm_2': 'family_medicine',
    'fm_3': 'family_medicine', 'fm_4': 'family_medicine',
    'fm_5': 'family_medicine',
    'ph_1': 'public_health', 'ph_2': 'public_health',
    'ph_3': 'public_health', 'ph_4': 'public_health',
    'ph_5': 'public_health',
}
for _k in LLM_KEYS:
    group_of[_k] = 'llm_legacy' if _k in LEGACY_LLMS else 'llm_new'

GROUP_COLORS = {
    'family_medicine': '#1f77b4',
    'public_health':   '#2ca02c',
    'llm_legacy':      '#d62728',
    'llm_new':         '#ff7f0e',
}
def group_color(entity):
    return GROUP_COLORS.get(group_of.get(entity, 'llm_new'), NEUTRAL)

RESULTS_DIR = 'results'
OUT_DIR = os.path.join(RESULTS_DIR, 'paper_figures')
os.makedirs(OUT_DIR, exist_ok=True)


def save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, f'{name}.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, f'{name}.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {name}.png + .pdf")


def panel_label(ax, label, x=-0.18, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='left')


def rater_label(k):
    if k in LLM_META:
        return LLM_META[k]['label']
    return ANONYMIZE(k)


# ============================================================================
# LOAD DATA
# ============================================================================
print("=" * 70)
print("PAPER FIGURES (v10) — building publication-grade panels")
print("=" * 70)

df_profiles = pd.read_csv(os.path.join(RESULTS_DIR, 'analysis2_common_pairs_profiles.csv'))
df_agree    = pd.read_csv(os.path.join(RESULTS_DIR, 'agreement_pairwise.csv'), index_col=0)

def _opt(name):
    p = os.path.join(RESULTS_DIR, name)
    return pd.read_csv(p) if os.path.exists(p) else None

df_bg     = _opt('battleground_pairs.csv')
df_bg_rel = _opt('battleground_pairs_relaxed.csv')

df_humans = df_profiles[df_profiles['doctor'].isin(HUMAN_DOCTORS)]
human_mean = df_humans[ETHICAL_DIMENSIONS].mean()
human_std  = df_humans[ETHICAL_DIMENSIONS].std()


# ============================================================================
# FIGURE 1 — Study design schematic (placeholder)
# ============================================================================
print("\nFigure 1: Study design schematic")
fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4.0))
ax.text(0.5, 0.5,
        "Figure 1 — Study Design Schematic\n\n"
        "(Vector editor: 10 physicians + 16 LLMs from 7 families →\n"
        "shared 100 seed pairs + 3 AL rounds + 200 test pairs →\n"
        "RankNet training → SHAP attribution →\n"
        "5-dimensional value profile)",
        ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle='round,pad=1', facecolor='#f5f5f5', edgecolor='#999'))
ax.set_xticks([]); ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
save(fig, 'figure_01_study_design')


# ============================================================================
# FIGURE 2 — Distance from human centroid (headline bar chart)
# ============================================================================
print("\nFigure 2: Distance from human centroid")

distances = []
for _, row in df_profiles.iterrows():
    entity = row['doctor']
    profile = row[ETHICAL_DIMENSIONS].values.astype(float)
    dist = euclidean(profile, human_mean.values)
    if entity in HUMAN_DOCTORS:
        distances.append({'entity': entity, 'label': ANONYMIZE(entity),
                          'family': 'Humans', 'type': 'human', 'distance': dist})
    elif entity in LLM_META:
        m = LLM_META[entity]
        distances.append({'entity': entity, 'label': m['label'],
                          'family': m['family'], 'type': 'llm', 'distance': dist})
df_dist = pd.DataFrame(distances)

humans_sorted = df_dist[df_dist['type'] == 'human'].sort_values('distance')
llms_sorted   = df_dist[df_dist['type'] == 'llm'].sort_values('distance')
df_plot = pd.concat([humans_sorted, llms_sorted]).reset_index(drop=True)
df_plot.to_csv(os.path.join(OUT_DIR, 'figure_02_distances.csv'), index=False)

fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4.8))
x_pos = np.arange(len(df_plot))
colors = [fam_color(f) for f in df_plot['family']]
bars = ax.bar(x_pos, df_plot['distance'], color=colors,
              edgecolor='black', linewidth=0.6, width=0.78)
for bar, dist in zip(bars, df_plot['distance']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
            f'{dist:.2f}', ha='center', va='bottom', fontsize=7)

n_humans = int((df_plot['type'] == 'human').sum())
ax.axvline(n_humans - 0.5, color='black', linestyle=':', linewidth=1.2, alpha=0.6)
human_max_dist = humans_sorted['distance'].max()
ax.axhspan(0, human_max_dist, color=HUMAN_DOT_COLOR, alpha=0.08, zorder=0)

ax.set_xticks(x_pos)
ax.set_xticklabels([ANONYMIZE(l) for l in df_plot['label']], rotation=45, ha='right', fontsize=7.5)
ax.set_ylabel('Distance from human centroid\n(Euclidean, 5-D value space, normalized)', fontsize=10)
ax.set_ylim(0, max(df_plot['distance']) * 1.15)

ax.text(n_humans / 2 - 0.5, ax.get_ylim()[1] * 0.92, 'Human physicians',
        ha='center', fontsize=10, fontweight='bold', color='#222')
ax.text(n_humans + (len(df_plot) - n_humans) / 2 - 0.5, ax.get_ylim()[1] * 0.92,
        f'LLMs ({len(llms_sorted)} from {df_plot[df_plot.type=="llm"].family.nunique()} families)',
        ha='center', fontsize=10, fontweight='bold', color='#222')

fams_present = [f for f in ['Humans','OpenAI','Anthropic','Google','Meta','DeepSeek','Qwen','Mistral']
                if f in df_plot['family'].values]
handles = [mpatches.Patch(color=fam_color(f), label=f) for f in fams_present]
ax.legend(handles=handles, loc='upper left', fontsize=8, ncol=4, bbox_to_anchor=(0, 1.01))
ax.grid(axis='y', alpha=0.25, linewidth=0.5); ax.set_axisbelow(True)
save(fig, 'figure_02_distance_from_humans')


# ============================================================================
# FIGURE 3 — PCA family map + loadings
# ============================================================================
print("\nFigure 3: PCA family map + loadings")

all_entities = HUMAN_DOCTORS + LLM_KEYS
entities_in_data = [e for e in all_entities if e in df_profiles['doctor'].values]
X = df_profiles.set_index('doctor').loc[entities_in_data, ETHICAL_DIMENSIONS].values.astype(float)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
loadings = pca.components_

fig = plt.figure(figsize=(DOUBLE_COL, 4.4))
gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 0.05, 1], wspace=0.05)
ax = fig.add_subplot(gs[0, 0])
ax_load = fig.add_subplot(gs[0, 2])

human_idx = [i for i, e in enumerate(entities_in_data) if e in HUMAN_DOCTORS]
ax.scatter(X_pca[human_idx, 0], X_pca[human_idx, 1],
           c=HUMAN_DOT_COLOR, s=70, alpha=0.55, edgecolor='black', linewidth=0.6,
           zorder=2, label='Human physicians (n=10)')
hc = X_pca[human_idx].mean(axis=0)
ax.scatter(*hc, marker='*', s=320, c=HUMAN_DOT_COLOR, edgecolor='black',
           linewidth=1.2, zorder=5, label='Human centroid')

plotted_fams = set()
for i, ent in enumerate(entities_in_data):
    if ent not in LLM_META:
        continue
    m = LLM_META[ent]; fam = m['family']
    fam_in_legend = fam if fam not in plotted_fams else None
    plotted_fams.add(fam)
    ax.scatter(X_pca[i, 0], X_pca[i, 1], c=fam_color(fam), s=m['size_pt'],
               marker=m['marker'], edgecolor='black', linewidth=1.0,
               zorder=4, label=fam_in_legend)
    ax.annotate(m['label'], (X_pca[i, 0], X_pca[i, 1]),
                xytext=(7, 3), textcoords='offset points', fontsize=7, fontweight='bold')

ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.0%} variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.0%} variance)')
ax.legend(loc='lower left', fontsize=7, framealpha=0.95, ncol=2, columnspacing=0.7)
ax.grid(alpha=0.2, linewidth=0.5); ax.set_axisbelow(True)
panel_label(ax, 'A', x=-0.13)

y_pos = np.arange(len(ETHICAL_DIMENSIONS))
width = 0.38
ax_load.barh(y_pos - width / 2, loadings[0], width, color='#4a4a4a', label='PC1')
ax_load.barh(y_pos + width / 2, loadings[1], width, color='#bbbbbb', label='PC2')
ax_load.set_yticks(y_pos)
ax_load.set_yticklabels(DIM_LABELS_2LINE, fontsize=8)
ax_load.set_xlabel('Loading')
ax_load.axvline(0, color='black', linewidth=0.6)
ax_load.legend(loc='lower right', fontsize=8)
ax_load.invert_yaxis()
ax_load.grid(axis='x', alpha=0.25, linewidth=0.5); ax_load.set_axisbelow(True)
panel_label(ax_load, 'B', x=-0.35)
save(fig, 'figure_03_pca_map')


# ============================================================================
# FIGURE 4 — OpenAI evolution trajectory (mini → frontier)
# ============================================================================
print("\nFigure 4: OpenAI evolution trajectory")

# all OpenAI models present, ordered mini -> frontier
_openai_order = ['doctor8', 'doctor9', 'gpt_5', 'gpt_5_4', 'gpt_5_5']
openai_in_data = [m for m in _openai_order if m in df_profiles['doctor'].values]
openai_labels = [LLM_META[m]['label'] if m in LLM_META else m for m in openai_in_data]

openai_profiles = df_profiles.set_index('doctor').loc[openai_in_data, ETHICAL_DIMENSIONS]
x_pos = np.arange(len(openai_in_data))

fig, axes = plt.subplots(1, 5, figsize=(DOUBLE_COL, 2.7), sharey=False)
for ax, dim, dim_label in zip(axes, ETHICAL_DIMENSIONS, DIM_LABELS_2LINE):
    vals = openai_profiles[dim].values.astype(float)
    ax.plot(x_pos, vals, '-o', color=fam_color('OpenAI'), linewidth=1.8,
            markersize=7, markeredgecolor='black', markeredgewidth=0.6, zorder=4)
    h_mean = human_mean[dim]
    ax.axhspan(h_mean - human_std[dim], h_mean + human_std[dim],
               color=HUMAN_DOT_COLOR, alpha=0.18, zorder=1)
    ax.axhline(h_mean, color=HUMAN_DOT_COLOR, linewidth=1.2, linestyle='--', alpha=0.85, zorder=2)
    ax.axhline(0, color='black', linewidth=0.4, alpha=0.3, zorder=0)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(openai_labels, rotation=45, ha='right', fontsize=6.5)
    ax.set_title(dim_label, fontsize=9, pad=4)
    ax.grid(alpha=0.25, axis='y', linewidth=0.5); ax.set_axisbelow(True)

axes[0].set_ylabel('Value contribution', fontsize=9.5)
legend_lines = [
    Line2D([0], [0], color=fam_color('OpenAI'), marker='o', markersize=7,
           markeredgecolor='black', markeredgewidth=0.6, linewidth=1.8, label='OpenAI models'),
    Line2D([0], [0], color=HUMAN_DOT_COLOR, linestyle='--', linewidth=1.2, label='Human mean'),
    mpatches.Patch(color=HUMAN_DOT_COLOR, alpha=0.18, label='Human ±1 SD'),
]
axes[-1].legend(handles=legend_lines, loc='lower right', fontsize=6.5, framealpha=0.95)
save(fig, 'figure_04_openai_trajectory')


# ============================================================================
# FIGURE 5 — Per-LLM profiles vs humans (spiders, auto-sized grid for all LLMs)
# ============================================================================
print("\nFigure 5: Per-LLM spider plots vs humans")

llms_in_data = [m for m in LLM_KEYS if m in df_profiles['doctor'].values]
# order by distance to humans (closest first)
def _dist(m):
    return euclidean(df_profiles[df_profiles.doctor == m][ETHICAL_DIMENSIONS].iloc[0].values.astype(float),
                     human_mean.values)
display_order = sorted(llms_in_data, key=_dist)

n_llms = len(display_order)
n_cols = 4 if n_llms > 6 else 3
n_rows = int(np.ceil(n_llms / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(DOUBLE_COL, 1.9 * n_rows),
                         subplot_kw=dict(projection='polar'))
axes_flat = np.atleast_1d(axes).flatten()

angles = np.linspace(0, 2 * np.pi, len(DIM_LABELS_SHORT), endpoint=False).tolist()
angles_closed = angles + angles[:1]

# shared min-max scale across all profiles (normalized values can be negative)
all_vals = []
for m in display_order:
    all_vals.extend(df_profiles[df_profiles['doctor'] == m][ETHICAL_DIMENSIONS].iloc[0].values.astype(float))
all_vals.extend(human_mean.values)
vmin, vmax = float(min(all_vals)), float(max(all_vals))
rng = vmax - vmin if vmax > vmin else 1.0

for idx, llm in enumerate(display_order):
    ax = axes_flat[idx]
    meta = LLM_META[llm]
    llm_prof = df_profiles[df_profiles['doctor'] == llm][ETHICAL_DIMENSIONS].iloc[0].values.astype(float)
    llm_norm   = (llm_prof - vmin) / rng
    human_norm = (human_mean.values - vmin) / rng
    llm_c   = list(llm_norm)   + [llm_norm[0]]
    human_c = list(human_norm) + [human_norm[0]]

    ax.plot(angles_closed, human_c, '-', color=HUMAN_DOT_COLOR, linewidth=1.4, alpha=0.8)
    ax.fill(angles_closed, human_c, color=HUMAN_DOT_COLOR, alpha=0.15)
    color = fam_color(meta['family'])
    ax.plot(angles_closed, llm_c, '-', color=color, linewidth=1.9)
    ax.fill(angles_closed, llm_c, color=color, alpha=0.32)

    ax.set_xticks(angles)
    ax.set_xticklabels(DIM_LABELS_SHORT, size=6.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75]); ax.set_yticklabels(['', '', ''], size=6)
    ax.grid(alpha=0.3, linewidth=0.4)
    dist = euclidean(llm_prof, human_mean.values)
    ax.set_title(f"{meta['label']}\n{meta['family']} · d={dist:.2f}", size=8.5, pad=10, fontweight='bold')

for j in range(len(display_order), len(axes_flat)):
    axes_flat[j].set_visible(False)

legend_lines = [
    Line2D([0], [0], color=HUMAN_DOT_COLOR, linewidth=2, label='Human mean profile'),
    Line2D([0], [0], color='black', linewidth=2, label='LLM profile (colour by family)'),
]
fig.legend(handles=legend_lines, loc='lower center', ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()
save(fig, 'figure_05_per_llm_profiles')


# ============================================================================
# FIGURE 6 — Battleground analysis
# ============================================================================
if df_bg is not None and df_bg_rel is not None:
    print("\nFigure 6: Battleground analysis")
    fig = plt.figure(figsize=(DOUBLE_COL, 3.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.6], wspace=0.35)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    strict_age  = df_bg['age_diff'].values
    relax_age   = df_bg_rel['age_diff'].values
    strict_risk = df_bg['risk_diff'].values
    relax_risk  = df_bg_rel['risk_diff'].values
    positions = [1, 2, 4, 5]
    data = [strict_age, relax_age, strict_risk, relax_risk]
    colors_box = ['#7f3a8a', '#a87fb5', '#a96632', '#c89572']

    bp = ax_a.boxplot(data, positions=positions, widths=0.7, patch_artist=True,
                      showfliers=False, medianprops=dict(color='black', linewidth=1.2))
    for patch, c in zip(bp['boxes'], colors_box):
        patch.set_facecolor(c); patch.set_alpha(0.75); patch.set_edgecolor('black')
    rng_local = np.random.default_rng(42)
    for pos, vals, c in zip(positions, data, colors_box):
        jitter = rng_local.normal(0, 0.06, size=len(vals))
        ax_a.scatter(np.full_like(vals, pos, dtype=float) + jitter, vals,
                     s=14, color=c, alpha=0.55, edgecolor='black', linewidth=0.3)
    ax_a.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    ax_a.set_xticks([1.5, 4.5])
    ax_a.set_xticklabels(['Age difference\n(years)', 'Risk difference\n(points)'], fontsize=9)
    ax_a.set_ylabel('Human-preferred − LLM-preferred', fontsize=9.5)
    for pos, lbl in zip([1, 2, 4, 5],
                        [f'strict\n(n={len(df_bg)})', f'relaxed\n(n={len(df_bg_rel)})',
                         f'strict\n(n={len(df_bg)})', f'relaxed\n(n={len(df_bg_rel)})']):
        ax_a.text(pos, ax_a.get_ylim()[0] - 5, lbl, ha='center', fontsize=6.5, color='#444')
    ax_a.grid(axis='y', alpha=0.25, linewidth=0.5); ax_a.set_axisbelow(True)
    panel_label(ax_a, 'A', x=-0.18)

    from collections import defaultdict
    human_recs = defaultdict(int); llm_recs = defaultdict(int)
    for _, row in df_bg.iterrows():
        if isinstance(row['human_picked_only_recs'], str):
            for r in row['human_picked_only_recs'].split(','):
                if r: human_recs[r] += 1
        if isinstance(row['llm_picked_only_recs'], str):
            for r in row['llm_picked_only_recs'].split(','):
                if r: llm_recs[r] += 1
    REC_NAMES = {
        'rec1': 'Basic labs', 'rec2': 'Advanced labs', 'rec3': 'Etiology workup',
        'rec4': 'Routine monitor', 'rec5': 'Carotid imaging', 'rec6': 'Advanced imaging',
        'rec8': 'BP/BMI', 'rec10': 'First-line statin', 'rec11': 'Adv. treatment',
        'rec12': 'Treatment ↑', 'rec13': 'Treatment swap', 'rec16': 'Lipidologist',
        'rec17': 'Other consult', 'rec18': 'Dietitian', 'rec19': 'Lifestyle', 'rec21': 'Records',
    }
    all_top = list(set(human_recs) | set(llm_recs))
    all_top = sorted(all_top, key=lambda r: -(max(human_recs.get(r, 0), llm_recs.get(r, 0))))[:8]
    n_bg = len(df_bg)
    human_freqs = [human_recs.get(r, 0) / n_bg * 100 for r in all_top]
    llm_freqs   = [llm_recs.get(r, 0) / n_bg * 100 for r in all_top]
    y_pos = np.arange(len(all_top)); width = 0.4
    ax_b.barh(y_pos - width / 2, human_freqs, width, color='#1f77b4',
              edgecolor='black', linewidth=0.4, label='in human-preferred')
    ax_b.barh(y_pos + width / 2, llm_freqs, width, color='#d62728',
              edgecolor='black', linewidth=0.4, label='in LLM-preferred')
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels([f"{r}: {REC_NAMES.get(r, r)}" for r in all_top], fontsize=8.5)
    ax_b.invert_yaxis()
    ax_b.set_xlabel('Frequency in battleground pairs (%)', fontsize=9.5)
    ax_b.legend(loc='lower right', fontsize=8)
    ax_b.grid(axis='x', alpha=0.25, linewidth=0.5); ax_b.set_axisbelow(True)
    panel_label(ax_b, 'B', x=-0.4)
    save(fig, 'figure_06_battleground')
else:
    print("\nFigure 6: skipped (battleground CSVs absent)")


# ============================================================================
# SUPPLEMENTARY FIGURES
# ============================================================================
print("\n--- Supplementary figures ---")

# S1 — Pairwise agreement matrix (all 26 raters, safe colour lookup)
print("\nFigure S1: pairwise agreement matrix")
ordered_entities = sorted(df_agree.columns,
                          key=lambda k: (group_order.index(group_of.get(k, 'llm_new')), k))
M = df_agree.loc[ordered_entities, ordered_entities].values

fig, ax = plt.subplots(figsize=(DOUBLE_COL, 6.8))
im = ax.imshow(M, cmap='RdYlGn', vmin=0.3, vmax=1.0, aspect='equal')
ax.set_xticks(np.arange(len(ordered_entities)))
ax.set_yticks(np.arange(len(ordered_entities)))
display_names = [rater_label(e) for e in ordered_entities]
ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=7)
ax.set_yticklabels(display_names, fontsize=7)

for tick, e in zip(ax.get_xticklabels(), ordered_entities):
    tick.set_color(group_color(e)); tick.set_fontweight('bold')
for tick, e in zip(ax.get_yticklabels(), ordered_entities):
    tick.set_color(group_color(e)); tick.set_fontweight('bold')

for i in range(len(ordered_entities)):
    for j in range(len(ordered_entities)):
        v = M[i, j]
        color = 'white' if v < 0.55 or v > 0.88 else 'black'
        ax.text(j, i, f'{v:.2f}', ha='center', va='center', color=color, fontsize=5.5)

prev = None
for i, e in enumerate(ordered_entities):
    g = group_of.get(e, 'llm_new')
    if prev and g != prev:
        ax.axhline(i - 0.5, color='black', linewidth=1.5)
        ax.axvline(i - 0.5, color='black', linewidth=1.5)
    prev = g

plt.colorbar(im, ax=ax, shrink=0.7, label='Agreement (300 common pairs)')
# title removed from image (lives in the Word caption instead, per Noa's comment)
handles = [mpatches.Patch(color=GROUP_COLORS[g],
                          label={'family_medicine': 'Family Medicine',
                                 'public_health': 'Public Health',
                                 'llm_legacy': 'LLM (Legacy)',
                                 'llm_new': 'LLM (New)'}[g]) for g in group_order]
ax.legend(handles=handles, bbox_to_anchor=(1.25, 1.0), loc='upper left', fontsize=8.5)
save(fig, 'figure_S1_agreement_matrix')


# S2 — Three-method comparison (group means, v10 axis labels)
print("\nFigure S2: three-method analysis comparison")
df_shap  = _opt('ethical_value_profiles_per_doctor.csv')
df_top30 = _opt('analysis1_topN_profiles.csv')
df_pairs = df_profiles
if df_top30 is not None:
    df_top30 = df_top30[df_top30['top_N'] == 30].copy()

methods = []
if df_shap is not None:  methods.append(('SHAP-based (Stage 1)', df_shap))
if df_top30 is not None: methods.append(('Top-30 patients (Stage 2A)', df_top30))
methods.append(('Common-pairs (Stage 2B)', df_pairs))

fig, axes = plt.subplots(1, len(methods), figsize=(DOUBLE_COL, 3.0), sharey=False)
axes = np.atleast_1d(axes)
for ax, (title, df_m) in zip(axes, methods):
    df_m = df_m.copy()
    if 'entity_type' not in df_m.columns:
        df_m['entity_type'] = df_m['doctor'].apply(lambda k: 'human' if k in HUMAN_DOCTORS else 'llm')
    means_h = df_m[df_m['entity_type'] == 'human'][ETHICAL_DIMENSIONS].mean()
    means_l = df_m[df_m['entity_type'].astype(str).str.startswith('llm')][ETHICAL_DIMENSIONS].mean()
    x = np.arange(len(ETHICAL_DIMENSIONS)); width = 0.38
    ax.bar(x - width / 2, means_h, width, color='#1f77b4', edgecolor='black', linewidth=0.4, label='Humans')
    ax.bar(x + width / 2, means_l, width, color='#d62728', edgecolor='black', linewidth=0.4, label='LLMs')
    ax.axhline(0, color='black', linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(DIM_LABELS_SHORT, rotation=35, ha='right', fontsize=7.5)
    ax.set_title(title, fontsize=9)
    ax.grid(axis='y', alpha=0.25, linewidth=0.4); ax.set_axisbelow(True)
axes[0].set_ylabel('Mean value contribution', fontsize=9)
axes[-1].legend(loc='upper right', fontsize=8)
plt.tight_layout()
save(fig, 'figure_S2_three_methods')


# S3 — Consensus distribution
df_cons = _opt('pair_consensus.csv')
if df_cons is not None and 'majority_size_all' in df_cons.columns:
    print("\nFigure S3: consensus distribution")
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.8))
    specs = [('majority_size_all', '#444', 'All raters', axes[0]),
             ('majority_size_human', '#1f77b4', 'Humans (n=10)', axes[1]),
             ('majority_size_llm', '#d62728', 'LLMs', axes[2])]
    for col, c, ttl, ax in specs:
        if col in df_cons.columns:
            counts = df_cons[col].value_counts().sort_index()
            ax.bar(counts.index, counts.values, color=c, edgecolor='black', linewidth=0.4)
        ax.set_xlabel('Majority size', fontsize=9); ax.set_title(ttl, fontsize=9.5)
        ax.grid(axis='y', alpha=0.25, linewidth=0.4); ax.set_axisbelow(True)
    axes[0].set_ylabel('Number of pairs', fontsize=9)
    plt.tight_layout()
    save(fig, 'figure_S3_consensus')
else:
    print("\nFigure S3: skipped (pair_consensus.csv absent)")


# S4 — Profile heatmap (all raters present, v10 axis labels, safe colours)
print("\nFigure S4: value profile heatmap")
order_for_heatmap = [k for k in (HUMAN_DOCTORS + sorted(
    [m for m in LLM_KEYS if m in df_profiles['doctor'].values], key=_dist))
    if k in df_profiles['doctor'].values]
M = df_profiles.set_index('doctor').loc[order_for_heatmap, ETHICAL_DIMENSIONS].values.astype(float)
labels_y = [rater_label(k) for k in order_for_heatmap]
group_y  = [group_of.get(k, 'llm_new') for k in order_for_heatmap]

fig, ax = plt.subplots(figsize=(SINGLE_COL * 2.0, 0.32 * len(order_for_heatmap) + 1.5))
vlim = float(np.nanmax(np.abs(M))) or 1.0
im = ax.imshow(M, cmap='RdBu_r', vmin=-vlim, vmax=vlim, aspect='auto')
ax.set_xticks(np.arange(len(ETHICAL_DIMENSIONS)))
ax.set_xticklabels(DIM_LABELS_2LINE, fontsize=8)
ax.set_yticks(np.arange(len(order_for_heatmap)))
ax.set_yticklabels(labels_y, fontsize=7.5)
for tick, g in zip(ax.get_yticklabels(), group_y):
    tick.set_color(GROUP_COLORS.get(g, NEUTRAL)); tick.set_fontweight('bold')
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        ax.text(j, i, f'{v:+.2f}', ha='center', va='center',
                color='black' if abs(v) < vlim * 0.6 else 'white', fontsize=6.5)
prev = None
for i, g in enumerate(group_y):
    if prev and g != prev:
        ax.axhline(i - 0.5, color='black', linewidth=1.2)
    prev = g
plt.colorbar(im, ax=ax, shrink=0.6, label='Common-pairs contribution (normalized)')
ax.set_title('Value profiles, by rater', fontsize=10)
save(fig, 'figure_S4_profile_heatmap')


print("\n" + "=" * 70)
print(f"DONE. All figures in {OUT_DIR}/")
print("=" * 70)