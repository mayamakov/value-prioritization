# -*- coding: utf-8 -*-
"""
rebuild_pair_csvs.py
Recompute the PAIR-DERIVED result CSVs (no training, no SHAP) for the FULL panel
straight from physicians_results/*.json + the current value mapping, so the
existing main/supplementary figures reflect all 16 models and the latest _MAP.

Writes into results/:
  rater_keymap.csv
  ethical_profiles_combined_top30.csv   (Pairs_<dim> filled; SHAP_/Top30_ = NaN)
  battleground_per_llm_summary.csv
  agreement_pairwise.csv
  pair_consensus.csv
  seed_profile_stability.csv            (if *_seed* folders exist)

NOT produced here (need the GPU/SHAP pipeline):
  SHAP_/Top30_ columns, analysis1_topN_profiles.csv  -> figure_S2
  ethical_shap_per_patient.pkl                        -> figure_S8 (raw-SHAP PCA)
  mapping_sensitivity_distances.csv / sensitivity_random_baseline.csv -> S9/S9b
"""
import os, json, glob
from pathlib import Path
import numpy as np
import pandas as pd

from panel_config import (HUMANS, MODELS, LEGACYSET, PANEL, display, family,
                          MODEL_FAMILY, FAMILY, PUBLIC)
from value_mapping_rec_intrinsic import patient_to_values, ETHICAL_DIMENSIONS as DIMS

RES = Path('results'); RES.mkdir(exist_ok=True)
PR  = Path('physicians_results')
coh = pd.read_excel('synthetic_data (7).xlsx').set_index('patient_num')

# ---- pair loaders (same convention as figR: [[winner,loser],conf]) ---------
def _n(it):
    if isinstance(it,(list,tuple)) and len(it)==2 and isinstance(it[0],(list,tuple)):
        return (int(it[0][0]), int(it[0][1]))
    if isinstance(it,(list,tuple)) and len(it)>=2:
        return (int(it[0]), int(it[1]))
    return None
def _l(p):
    p=Path(p)
    return [x for x in (_n(z) for z in json.load(open(p))) if x] if p.exists() else []
def ch(k, folder=None):
    d = Path(folder) if folder else PR/k
    pr=[]
    for i in range(4): pr += _l(d/f'{k}_train_iter_{i}_ranked.json')
    if k in LEGACYSET:
        pr += _l(d/f'{k}_test_ranked.json')
    else:
        got=False
        for pt in 'AB':
            t=_l(d/f'{k}_test_part_{pt}_ranked.json'); pr+=t; got=got or bool(t)
        if not got: pr += _l(d/f'{k}_test_ranked.json')
    return {tuple(sorted(p)): p[0] for p in pr}          # sorted-pair -> winner pid

C = {k: ch(k) for k in PANEL}
present = [k for k in PANEL if C[k]]
common = set.intersection(*[set(C[k]) for k in present])
common = [u for u in common if u[0] in coh.index and u[1] in coh.index]
print(f"raters present={len(present)}/{len(PANEL)}  common pairs={len(common)}")

# ---- per-patient value vectors (depends on mapping; recomputed live) --------
pids = sorted({p for u in common for p in u})

# === v11: recommendation-support weighting (item 1) ========================
# Compute n_rec from the common pairs, install the weights on the mapping module
# (so every patient_to_values below is support-weighted), and SAVE them so
# stage2 / make_topN / tables use the IDENTICAL weights.
import value_mapping_rec_intrinsic as _vm
from paper_value_config_v10 import SUPPORT_WEIGHT as _SW
if _SW:
    try:
        _wts, _nrec, _K = _vm.compute_support_weights(common, coh)
        _vm.set_support_weights(_wts)
        pd.DataFrame([{'rec': r, 'n_pairs_differ': _nrec[r], 'weight': _wts[r]}
                     for r in _wts]).to_csv(RES/'rec_support_weights.csv', index=False)
        print(f"support weighting ON (K={_K:.1f}); "
              f"min/max w = {min(_wts.values()):.2f}/{max(_wts.values()):.2f}  "
              f"-> results/rec_support_weights.csv")
    except Exception as _e:
        print(f"[support-weight] skipped ({_e}); using unweighted values")
else:
    _vm.set_support_weights(None)
    print("support weighting OFF (SUPPORT_WEIGHT=False) -> unweighted signatures")

V = {pid: patient_to_values(coh.loc[pid]) for pid in pids}     # dict dim->score (support-weighted)

def signature(choice_map):
    """Common-pairs differential value signature: mean(V[winner]-V[loser])."""
    acc = {d: 0.0 for d in DIMS}; n=0
    for u in common:
        w = choice_map.get(u);
        if w is None: continue
        a,b=u; l = b if w==a else a; n+=1
        for d in DIMS: acc[d]+= V[w][d]-V[l][d]
    return {d: (acc[d]/n if n else 0.0) for d in DIMS}

# anonymized labels: physicians FM-/PH- (by panel order), models -> display
_FM={k:f'FM-{i+1}' for i,k in enumerate(FAMILY)}
_PH={k:f'PH-{i+1}' for i,k in enumerate(PUBLIC)}
def anon(k): return _FM.get(k) or _PH.get(k) or display(k)

# ===== 1) rater_keymap.csv ==================================================
def grp(k):
    if k in FAMILY: return 'family_medicine'
    if k in PUBLIC: return 'public_health'
    return 'llm'
def fam(k):
    return MODEL_FAMILY.get(k, 'Physician' if k in HUMANS else 'Other')
pd.DataFrame([{'display_name':anon(k),'rater_key':k,'group':grp(k),'family':fam(k)}
              for k in PANEL]).to_csv(RES/'rater_keymap.csv', index=False)

# ===== 2) ethical_profiles_combined_top30.csv (Pairs_ filled) ===============
rows=[]
for k in PANEL:
    sig = signature(C[k]) if C[k] else {d: np.nan for d in DIMS}
    row={'doctor':k}
    for d in DIMS:
        row[f'SHAP_{d}']  = np.nan          # needs training (figure_S2 only)
        row[f'Top30_{d}'] = np.nan          # needs training
        row[f'Pairs_{d}'] = sig[d]
    rows.append(row)
pd.DataFrame(rows).to_csv(RES/'ethical_profiles_combined_top30.csv', index=False)

# ===== 3) battleground_per_llm_summary.csv =================================
#  discordant pair = >=7/10 physicians chose opposite to the LLM
bg=[]
for k in MODELS:
    if not C[k]: continue
    n_b=0; ad=[]; rd=[]
    for u in common:
        a,b=u
        ha=sum(1 for h in HUMANS if C[h].get(u)==a)        # physicians for a
        hb=len(HUMANS)-ha
        phys=a if ha>=hb else b; maj=max(ha,hb)
        llm=C[k].get(u)
        if llm is None: continue
        opp = ha if llm==b else hb                          # physicians opposing LLM
        if opp>=7:
            n_b+=1
            ad.append(coh.loc[llm,'age']-coh.loc[phys,'age'])
            rd.append(coh.loc[llm,'risk']-coh.loc[phys,'risk'])
    bg.append({'llm':display(k),'llm_key':k,'family':MODEL_FAMILY[k],
               'n_battleground':n_b,
               'mean_age_diff':float(np.mean(ad)) if ad else 0.0,
               'mean_risk_diff':float(np.mean(rd)) if rd else 0.0,
               'top_human_recs':'','top_llm_recs':''})
pd.DataFrame(bg).to_csv(RES/'battleground_per_llm_summary.csv', index=False)

# ===== 4) agreement_pairwise.csv (display-name square matrix) ===============
labels=[anon(k) for k in PANEL]
M=np.full((len(PANEL),len(PANEL)), np.nan)
for i,ki in enumerate(PANEL):
    for j,kj in enumerate(PANEL):
        if not C[ki] or not C[kj]: continue
        agree=sum(1 for u in common if C[ki].get(u)==C[kj].get(u))
        M[i,j]=agree/len(common) if common else np.nan
pd.DataFrame(M, index=labels, columns=labels).to_csv(RES/'agreement_pairwise.csv')

# ===== 5) pair_consensus.csv ===============================================
def maj_size(keys,u):
    a,b=u; ca=sum(1 for k in keys if C[k].get(u)==a)
    return max(ca, len(keys)-ca)
hum=[k for k in HUMANS if C[k]]; llm=[k for k in MODELS if C[k]]; allk=hum+llm
pc=[{'pair':f'{a}_{b}',
     'majority_size_all':maj_size(allk,(a,b)),
     'majority_size_human':maj_size(hum,(a,b)),
     'majority_size_llm':maj_size(llm,(a,b))} for (a,b) in common]
pd.DataFrame(pc).to_csv(RES/'pair_consensus.csv', index=False)

# ===== 6) seed_profile_stability.csv (if seed folders exist) ===============
phys_centroid=np.array([signature({u:C[h][u] for u in common})  # not used; below simpler
                        for h in []]) if False else None
hum_sig=np.array([[signature(C[h])[d] for d in DIMS] for h in hum])
phys_mean=hum_sig.mean(axis=0)
srows=[]
for k in MODELS:
    for sd in (1,2,3):
        folder=PR/f'{k}_seed{sd}'
        f=folder/f'{k}_seed{sd}_common_pairs.json'
        if not f.exists(): continue
        pairs=_l(f); cm={tuple(sorted(p)):p[0] for p in pairs}
        cm={u:cm[u] for u in common if u in cm}
        if not cm: continue
        sig=signature(cm); vec=np.array([sig[d] for d in DIMS])
        srows.append({'llm':display(k),'family':MODEL_FAMILY[k],'seed':sd,
                      'n_pairs':len(cm),
                      'distance_from_physician_centroid':float(np.linalg.norm(vec-phys_mean)),
                      **{d:sig[d] for d in DIMS}})
if srows:
    pd.DataFrame(srows).to_csv(RES/'seed_profile_stability.csv', index=False)
    print(f"seed_profile_stability.csv: {len(srows)} rows")
else:
    print("no seed folders found -> seed_profile_stability.csv not written (figure_S7 will skip)")


# ===== 7) mapping-sensitivity + random baseline (training-free) =============
#  figure_S9  : distance from physician centroid under 4 mapping variants
#  figure_S9b : 500 random-rater null distribution (per dimension)
from value_mapping_rec_intrinsic import REC_INTRINSIC_MAPPING as _BASE
import copy as _copy

def _variant(name):
    # variant definitions matched to the canonical mapping_sensitivity.py (manuscript),
    # adapted to the v10 6-field mapping (U, P, NM, EMP, JU are explicit fields here).
    m=_copy.deepcopy(_BASE)
    if name=='M1_statin_harm':       # statin/PCSK9/upgrade Non-maleficence
        m['rec10']['NM']=0.3; m['rec11']['NM']=0.3; m['rec12']['NM']=0.5
    elif name=='M2_pcsk9_neutral':   # PCSK9 (rec11) & CTA (rec6) Justice -> neutral
        m['rec11']['JU']=0.0; m['rec6']['JU']=0.0
    elif name=='M3_perturbed':       # every non-zero weight x (1 +/- 20%)
        rng=np.random.default_rng(42)
        for r in m:
            for d in ('U','P','NM','EMP','JU'):
                if d in m[r] and m[r][d]!=0:
                    m[r][d]=float(np.clip(m[r][d]*(1+rng.uniform(-0.2,0.2)),0,1))
    return m
_VARIANTS={'M0_original':_BASE,'M1_statin_harm':_variant('M1_statin_harm'),
           'M2_pcsk9_neutral':_variant('M2_pcsk9_neutral'),'M3_perturbed':_variant('M3_perturbed')}
_MLAB={'M0_original':'M0 (original)','M1_statin_harm':'M1 (statin harm)',
       'M2_pcsk9_neutral':'M2 (PCSK9 neutral)','M3_perturbed':'M3 (perturbed)'}

def _sig_vec(Vv, cmap):
    acc={d:0.0 for d in DIMS}; n=0
    for u in common:
        w=cmap.get(u)
        if w is None: continue
        a,b=u; l=b if w==a else a; n+=1
        for d in DIMS: acc[d]+=Vv[w][d]-Vv[l][d]
    return np.array([acc[d]/n if n else 0.0 for d in DIMS])

_ms=[]
for _mn,_mp in _VARIANTS.items():
    _Vv={pid:patient_to_values(coh.loc[pid], _mp) for pid in pids}
    _sig={k:_sig_vec(_Vv,C[k]) for k in present}
    # z-score per axis across all raters (equal opportunity), like T4/main analysis
    _M=np.array([_sig[k] for k in present])
    _mu=_M.mean(axis=0); _sd=_M.std(axis=0); _sd[_sd==0]=1.0
    _sig={k:(_sig[k]-_mu)/_sd for k in present}
    _cent=np.mean([_sig[h] for h in hum],axis=0)
    for k in present:
        _ms.append({'mapping':_mn,'mapping_label':_MLAB[_mn],'rater':k,'label':anon(k),
                    'family':fam(k),'entity_type':'human' if k in HUMANS else 'llm',
                    'distance':float(np.linalg.norm(_sig[k]-_cent))})
pd.DataFrame(_ms).to_csv(RES/'mapping_sensitivity_distances.csv', index=False)

_rng=np.random.default_rng(1); _rb=[]
for _ in range(500):
    acc={d:0.0 for d in DIMS}
    for u in common:
        a,b=u; w,l=(a,b) if _rng.random()<0.5 else (b,a)
        for d in DIMS: acc[d]+=V[w][d]-V[l][d]
    _rb.append({d:acc[d]/len(common) for d in DIMS})
pd.DataFrame(_rb).to_csv(RES/'sensitivity_random_baseline.csv', index=False)
print("mapping_sensitivity_distances.csv + sensitivity_random_baseline.csv written")

print("rebuilt pair-based CSVs in results/ :",
      "rater_keymap, ethical_profiles_combined_top30, battleground_per_llm_summary,",
      "agreement_pairwise, pair_consensus" + (", seed_profile_stability" if srows else ""))