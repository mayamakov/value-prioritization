"""
==========================================================================
raw_shap_clustering.py
==========================================================================
Robustness analysis: Does the family-vs-size clustering pattern persist
without the Beauchamp-Childress mapping?

This addresses Reviewer #2's critique #1 and #7:
  "The ethical dimensions are not empirically measured but analytically
   imposed. Discovery and interpretation are entangled."

LOGIC:
  - We computed SHAP values per rater per patient (no B-C mapping yet).
  - We then aggregated them into a 5-dim B-C signature via our mapping.
  - We then ran PCA on the 5-dim signatures → found the family clustering.

  Question: does the same clustering appear if we skip the B-C mapping
  and run PCA directly on the raw SHAP values?

  If YES → the clustering is an empirical phenomenon, not an artifact of
           our mapping. The B-C decomposition is interpretation, not
           discovery. This neutralizes the circularity critique.

  If NO  → the clustering depends on the mapping. We must reframe.

INPUT:
  - results/ethical_shap_per_patient.pkl  (created by Stage 1)

OUTPUT:
  - results/raw_shap_clustering_distances.csv
  - results/figures/figS8_raw_shap_pca.png/.pdf
==========================================================================
"""

# ===== ANONYMIZATION (canonical mapping from rater_keymap.csv) =====
# NOTE: numbering MUST match rater_keymap.csv exactly so the same physician
# carries the same code across all figures (S1, S4, S8, ...).
_PHYS_NAME_MAP = {
    'fm_1': 'FM-1',
    'fm_2': 'FM-2',
    'fm_3': 'FM-3',
    'fm_4': 'FM-4',
    'fm_5': 'FM-5',
    'ph_1': 'PH-1',
    'ph_3': 'PH-3',
    'ph_4': 'PH-4',
    'ph_2': 'PH-2',
    'ph_5': 'PH-5',
}
def ANONYMIZE(name):
    """Map real physician keys to anonymous FM-/PH- codes (per rater_keymap.csv).
    Returns the input unchanged if it is not a physician (e.g., an LLM key)."""
    return _PHYS_NAME_MAP.get(name, name)
# =================================================================

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# ==========================================================================
# CONFIG
# ==========================================================================
INPUT_PKL = 'results/ethical_shap_per_patient.pkl'
OUT_DIR = 'results'
FIG_DIR = os.path.join(OUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

HUMAN_DOCTORS = [
    'fm_1', 'fm_2', 'fm_3', 'fm_4', 'fm_5',
    'ph_1', 'ph_2', 'ph_3', 'ph_4', 'ph_5',
]

# --- All 16 models, built dynamically from panel_config (no hard-coding) ---
from panel_config import MODELS, DISPLAY, MODEL_FAMILY, FAMILY_COLOR, SMALL_TIER
_FAM_MARKER = {'OpenAI': 'o', 'Anthropic': 's', 'Google': '^', 'Meta': 'D',
               'DeepSeek': 'v', 'Qwen': 'P', 'Mistral': 'X'}
LLM_META = {
    k: dict(label=DISPLAY[k],
            family=MODEL_FAMILY[k],
            marker=_FAM_MARKER.get(MODEL_FAMILY[k], 'o'),
            size_pt=(90 if k in SMALL_TIER else 200))
    for k in MODELS
}

FAMILY_COLORS = dict(FAMILY_COLOR)          # OpenAI, Anthropic, Google, Meta, DeepSeek, Qwen, Mistral
FAMILY_COLORS.setdefault('OpenAI',    '#10a37f')
FAMILY_COLORS.setdefault('Anthropic', '#d97706')
FAMILY_COLORS.setdefault('Google',    '#4285f4')
FAMILY_COLORS['Humans'] = '#444444'


# ==========================================================================
# LOAD DATA
# ==========================================================================
print("=" * 75)
print("RAW SHAP CLUSTERING ANALYSIS")
print("=" * 75)
print(f"\nLoading SHAP data from {INPUT_PKL}...")

with open(INPUT_PKL, 'rb') as f:
    shap_data = pickle.load(f)

print(f"  Loaded SHAP data for {len(shap_data)} raters")
for k in list(shap_data.keys())[:3]:
    s = shap_data[k]
    print(f"  Sample [{k}]: SHAP shape = {s['shap_per_patient'].shape}, features = {len(s['feature_cols'])}")

# Get feature names (should be consistent across raters)
first_rater = next(iter(shap_data.keys()))
feature_cols = shap_data[first_rater]['feature_cols']
print(f"  Features: {feature_cols}")
N_FEATURES = len(feature_cols)


# ==========================================================================
# AGGREGATE: per-rater raw SHAP signature (mean across patients per feature)
# ==========================================================================
print("\n" + "=" * 75)
print("STEP 1: Compute raw SHAP signature per rater (mean across patients)")
print("=" * 75)

raters = []
raw_signatures = []
for rater_key, data in shap_data.items():
    shap_mat = data['shap_per_patient']  # (n_patients, n_features)
    if shap_mat is None or shap_mat.shape[0] == 0:
        print(f"  SKIP {rater_key} - empty SHAP")
        continue
    # Mean across patients → 1 vector of length n_features per rater
    mean_per_feature = shap_mat.mean(axis=0)
    raters.append(rater_key)
    raw_signatures.append(mean_per_feature)

X_raw = np.array(raw_signatures)
print(f"  Built raw-SHAP matrix: {X_raw.shape}")
print(f"  Raters: {raters}")


# ==========================================================================
# PCA ON RAW SHAP
# ==========================================================================
print("\n" + "=" * 75)
print("STEP 2: PCA on raw SHAP (no B-C mapping applied)")
print("=" * 75)

pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_raw)
print(f"\n  Explained variance ratios:")
for i, r in enumerate(pca.explained_variance_ratio_):
    print(f"    PC{i+1}: {r:.1%}")
print(f"  Cumulative (PC1+PC2): {sum(pca.explained_variance_ratio_[:2]):.1%}")


# ==========================================================================
# COMPUTE DISTANCES FROM HUMAN CENTROID (in raw SHAP space)
# ==========================================================================
print("\n" + "=" * 75)
print("STEP 3: Distances from human centroid (raw SHAP, no mapping)")
print("=" * 75)

human_indices = [i for i, r in enumerate(raters) if r in HUMAN_DOCTORS]
llm_indices = [i for i, r in enumerate(raters) if r in LLM_META]

if not human_indices:
    raise RuntimeError("No human raters found in SHAP data!")

human_centroid_raw = X_raw[human_indices].mean(axis=0)

distance_rows = []
for i, rater in enumerate(raters):
    dist = euclidean(X_raw[i], human_centroid_raw)
    entity_type = 'human' if rater in HUMAN_DOCTORS else 'llm'
    label = LLM_META[rater]['label'] if rater in LLM_META else rater
    family = LLM_META[rater]['family'] if rater in LLM_META else 'Humans'
    distance_rows.append({
        'rater':       rater,
        'label':       ANONYMIZE(label),
        'family':      family,
        'entity_type': entity_type,
        'distance_raw_shap': dist,
    })

df_dist_raw = pd.DataFrame(distance_rows).sort_values('distance_raw_shap')
print("\n  Distance ranking (raw SHAP):")
print(df_dist_raw.round(4).to_string(index=False))


# ==========================================================================
# COMPARE WITH MAPPED DISTANCES (the original 5-dim B-C distances)
# ==========================================================================
print("\n" + "=" * 75)
print("STEP 4: Correlation between raw-SHAP distances and B-C mapped distances")
print("=" * 75)

mapped_csv = os.path.join(OUT_DIR, 'llm_distance_from_humans.csv')
if not os.path.exists(mapped_csv):
    print(f"  WARN: {mapped_csv} not found - skipping correlation check.")
    df_compare = df_dist_raw
else:
    df_mapped = pd.read_csv(mapped_csv)
    # Merge by rater label (or fallback to 'entity' column if present)
    join_col = 'entity' if 'entity' in df_mapped.columns else 'rater'
    df_mapped = df_mapped.rename(columns={join_col: 'rater', 'distance': 'distance_mapped'})
    df_compare = df_dist_raw.merge(df_mapped[['rater', 'distance_mapped']], on='rater', how='left')
    print("\n  Joined table:")
    print(df_compare.round(4).to_string(index=False))

    # Pearson + Spearman correlation between raw-SHAP distances and mapped distances
    valid = df_compare.dropna(subset=['distance_raw_shap', 'distance_mapped'])
    if len(valid) >= 3:
        pear_r, pear_p = pearsonr(valid['distance_raw_shap'], valid['distance_mapped'])
        spear_r, spear_p = spearmanr(valid['distance_raw_shap'], valid['distance_mapped'])
        print(f"\n  Correlation between raw-SHAP and B-C-mapped distances:")
        print(f"    Pearson r  = {pear_r:.3f}  (p = {pear_p:.4f})")
        print(f"    Spearman ρ = {spear_r:.3f}  (p = {spear_p:.4f})")
        print(f"    n = {len(valid)} raters")
        print()
        if pear_r > 0.8:
            print(f"  >>> STRONG agreement: raw-SHAP distances track B-C-mapped distances closely.")
            print(f"  >>> The B-C mapping is interpretation, not discovery. Clustering is robust.")
        elif pear_r > 0.5:
            print(f"  >>> MODERATE agreement: B-C mapping reinforces but does not create the clustering.")
        else:
            print(f"  >>> WEAK agreement: B-C mapping may be doing significant work.")

df_compare.to_csv(os.path.join(OUT_DIR, 'raw_shap_clustering_distances.csv'), index=False)
print(f"\n  Saved -> {os.path.join(OUT_DIR, 'raw_shap_clustering_distances.csv')}")


# ==========================================================================
# WITHIN-FAMILY vs ACROSS-FAMILY DISPERSION (raw SHAP)
# ==========================================================================
print("\n" + "=" * 75)
print("STEP 5: Within-OpenAI vs across-family dispersion (raw SHAP)")
print("=" * 75)

llm_df = df_dist_raw[df_dist_raw['entity_type'] == 'llm'].copy()
openai_df = llm_df[llm_df['family'] == 'OpenAI']
all_llms_df = llm_df

if len(openai_df) >= 2 and len(all_llms_df) >= 3:
    within_openai_sd = openai_df['distance_raw_shap'].std()
    across_family_sd = all_llms_df['distance_raw_shap'].std()
    ratio = across_family_sd / within_openai_sd if within_openai_sd > 0 else float('inf')
    print(f"\n  Within-OpenAI SD (raw SHAP):    {within_openai_sd:.4f}")
    print(f"  Across-family SD (raw SHAP):    {across_family_sd:.4f}")
    print(f"  Across/within ratio (raw SHAP): {ratio:.1f}")
    print()
    print(f"  For comparison, mapped 5-dim space gave ~57× ratio.")
    print(f"  If raw-SHAP ratio is also large → clustering is empirical, not mapping-driven.")


# ==========================================================================
# FIGURE: PCA scatter (raw SHAP) - mirror of Figure 3 but in raw space
# ==========================================================================
print("\n" + "=" * 75)
print("STEP 6: Producing Figure S8 (raw-SHAP PCA)")
print("=" * 75)

fig, ax = plt.subplots(figsize=(11, 8))

# Humans first
ax.scatter(X_pca[human_indices, 0], X_pca[human_indices, 1],
           c=FAMILY_COLORS['Humans'], s=140, alpha=0.5,
           edgecolor='black', linewidth=1,
           label=f'Participating physicians (n={len(human_indices)})', zorder=2)
human_centroid_pca = X_pca[human_indices].mean(axis=0)
ax.scatter(human_centroid_pca[0], human_centroid_pca[1], marker='*', s=700,
           c=FAMILY_COLORS['Humans'], edgecolor='black', linewidth=2,
           label='Physician centroid', zorder=5)

plotted_families = set()
for i, rater in enumerate(raters):
    if rater not in LLM_META:
        continue
    meta = LLM_META[rater]
    family = meta['family']
    color = FAMILY_COLORS[family]
    ax.scatter(X_pca[i, 0], X_pca[i, 1],
               c=color, s=meta['size_pt'],
               marker=meta['marker'],
               edgecolor='black', linewidth=1.5, zorder=4,
               label=(family if family not in plotted_families else None))
    plotted_families.add(family)
    ax.annotate(meta['label'],
                (X_pca[i, 0], X_pca[i, 1]),
                xytext=(10, 5), textcoords='offset points',
                fontsize=11, weight='bold')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} of variance)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} of variance)', fontsize=12)
ax.set_title('Raw-SHAP clustering (no Beauchamp-Childress mapping applied)\n'
             'PCA on 24-dimensional mean-SHAP-per-feature signature',
             fontsize=13, weight='bold', pad=15)
ax.grid(alpha=0.3)
ax.legend(loc='best', fontsize=10, framealpha=0.95)
ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)
ax.axvline(0, color='black', linewidth=0.5, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'figS8_raw_shap_pca.png'),
            dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(FIG_DIR, 'figS8_raw_shap_pca.pdf'),
            bbox_inches='tight')
plt.close(fig)
print(f"  Saved -> {FIG_DIR}/figS8_raw_shap_pca.png/.pdf")


# ==========================================================================
# SUMMARY
# ==========================================================================
print("\n" + "=" * 75)
print("INTERPRETATION GUIDE")
print("=" * 75)
print()
print("Look at the OUTPUT above and decide:")
print()
print("  ┌─────────────────────────────────────────────────────────────────┐")
print("  │ IF Pearson r between raw-SHAP and mapped distances > 0.8:       │")
print("  │   → STRONG result. Clustering is empirical, not mapping-driven. │")
print("  │   → This is the 'best case' for the manuscript.                 │")
print("  ├─────────────────────────────────────────────────────────────────┤")
print("  │ IF Pearson r is 0.5-0.8:                                         │")
print("  │   → MODERATE. Both raw and mapped views agree on direction.     │")
print("  │   → Still good - mapping reinforces but does not create.        │")
print("  ├─────────────────────────────────────────────────────────────────┤")
print("  │ IF Pearson r < 0.5:                                             │")
print("  │   → WEAK. Mapping does heavy lifting.                           │")
print("  │   → Must reframe the manuscript to acknowledge this.            │")
print("  └─────────────────────────────────────────────────────────────────┘")
print()
print("Also note the within-OpenAI SD vs across-family SD ratio in raw space.")
print("If it's still large (e.g., >10×), the family-vs-size finding survives.")
print()
print("=" * 75)
print(f"DONE. Outputs in {OUT_DIR}/ and {FIG_DIR}/")
print("=" * 75)