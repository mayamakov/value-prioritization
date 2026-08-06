# -*- coding: utf-8 -*-
"""
Figure 5 (v11 redesign, two panels, one line / bar per rater).
  Panel A: RankNet ranking score (z within rater) vs patient 10-YEAR CV RISK bin.
  Panel B: mean RankNet score of patients WHO HAVE each TREATMENT recommendation
           (rec10/11/12/13), family-medicine vs public-health physicians.
Story: models climb with patient RISK (Panel A); physicians' prioritisation is
organised around the actionable TREATMENT recommendation (Panel B).
Colours: family-medicine = light blue (FAM_C), public-health = navy (PUB_C),
models = brown line (Panel A only).
"""
import json, numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from panel_config import (FAMILY, PUBLIC, MODELS, LEGACYSET, PANEL, display,
                          model_color, FAM_C, PUB_C, FAMILY_COLOR, MODELS_BY_FAMILY)
MCOL={k:model_color(k) for k in MODELS}
coh=pd.read_excel('synthetic_data (7).xlsx').set_index('patient_num')

# --- RankNet score per patient (z within rater) ---
_RN = pd.read_csv('results/ranknet_scores_per_rater.csv')
_RN['z'] = _RN.groupby('doctor')['ranknet_score'].transform(
    lambda s: (s - s.mean())/(s.std(ddof=0) if s.std(ddof=0)>0 else 1.0))
_RN_BY = {k: dict(zip(g['patient_num'], g['z'])) for k, g in _RN.groupby('doctor')}
def zscores(k):
    return {p: v for p, v in _RN_BY.get(k, {}).items() if p in coh.index}
WR={k:zscores(k) for k in PANEL}

# ---------- Panel A: RISK bins ----------
RISK_BINS=[(0,10,'<10'),(10,20,'10\u201320'),(20,30,'20\u201330'),(30,40,'30\u201340'),(40,1e9,'40+')]
def risk_curve(wr):
    ys=[]
    for lo,hi,_ in RISK_BINS:
        v=[wr[p] for p in wr if lo<=coh.loc[p,'risk']<hi]
        ys.append(np.mean(v) if v else np.nan)
    return ys

# ---------- Panel B: treatment recommendations ----------
TREAT=[('rec10','First-line\nstatin'),('rec11','Advanced\ntreatment'),
       ('rec12','Treatment\nupgrade'),('rec13','Replacement\n(contraindication)')]
def rec_mean(wr, rec):
    v=[wr[p] for p in wr if coh.loc[p,rec]>=0.5]
    return np.mean(v) if v else np.nan
def group_rec_means(grp, rec):
    vals=[rec_mean(WR[k],rec) for k in grp]
    vals=[x for x in vals if not np.isnan(x)]
    return (np.mean(vals) if vals else np.nan,
            np.std(vals)/np.sqrt(len(vals)) if len(vals)>1 else 0.0)

fig,(axA,axB)=plt.subplots(1,2,figsize=(15,8.5),gridspec_kw={'width_ratios':[1.05,1]})

# Panel A
xa=np.arange(len(RISK_BINS))
def plot_rater(ax,xs,ys,k):
    if k in FAMILY: c,m,lw,z=FAM_C,'o',1.4,2
    elif k in PUBLIC: c,m,lw,z=PUB_C,'o',1.4,2
    else: c,m,lw,z=MCOL[k],'s',1.7,3
    ax.plot(xs,ys,color=c,lw=lw,marker=m,ms=4,alpha=.85,zorder=z)
for k in PANEL: plot_rater(axA,xa,risk_curve(WR[k]),k)
for grp,c,mk in [(FAMILY,FAM_C,'o'),(PUBLIC,PUB_C,'o'),(MODELS,'#5a3210','s')]:
    mean=[np.nanmean([risk_curve(WR[k])[i] for k in grp]) for i in range(len(RISK_BINS))]
    axA.plot(xa,mean,color=c,lw=4,zorder=6,marker=mk,ms=7)
axA.set_xticks(xa); axA.set_xticklabels([b[2] for b in RISK_BINS])
axA.set_xlabel('Patient 10-year cardiovascular risk (%)',fontsize=10.5)
axA.set_ylabel('RankNet ranking score (z, within rater)',fontsize=10.5)
axA.set_title('A. Prioritisation by patient RISK',fontsize=12,fontweight='bold')

# Panel B: per-recommendation mean RankNet (z), grouped by ethical axis (Supp Fig 5 structure)
RECS=['rec1','rec2','rec3','rec4','rec5','rec6','rec8','rec10','rec11','rec12','rec13','rec16','rec17','rec18','rec19','rec21']
NAME={'rec1':'Basic lab panel','rec2':'Advanced lab panel','rec3':'Pathophysiology labs','rec4':'Routine lab monitoring',
      'rec5':'Diagnostic imaging','rec6':'Advanced imaging (CTA)','rec8':'Take BP/BMI measurement','rec10':'Initiate first-line statin',
      'rec11':'Initiate advanced tx','rec12':'Treatment upgrade','rec13':'Treatment replacement (CI)','rec16':'Specialist consult',
      'rec17':'Other consult (hepatology)','rec18':'Nutritional consult','rec19':'Lifestyle improvement','rec21':'Curate medical record'}
AXIS_OF={**{r:'Utility' for r in ['rec1','rec2','rec3','rec4','rec5','rec6','rec8','rec10','rec11','rec12','rec16','rec21']},
         **{r:'Non-maleficence' for r in ['rec13','rec17']},
         **{r:'Patient empowerment' for r in ['rec18','rec19']}}
AXIS_ORDER=['Utility','Non-maleficence','Patient empowerment']
AXIS_FULL={'Utility':'Beneficence-utility','Non-maleficence':'Non-maleficence','Patient empowerment':'Patient empowerment'}
PHYS_ALL=list(FAMILY)+list(PUBLIC)
def rec_gmean(grp,rec):
    v=[rec_mean(WR[k],rec) for k in grp]; v=[x for x in v if not np.isnan(x)]
    return np.mean(v) if v else np.nan
rows=[]
for ax_key in AXIS_ORDER:
    recs_in=sorted([r for r in RECS if AXIS_OF.get(r)==ax_key],
                   key=lambda r: abs(rec_gmean(PHYS_ALL,r))+abs(rec_gmean(MODELS,r)))
    for r in recs_in: rows.append(('rec',r,ax_key))
    rows.append(('gap',ax_key,ax_key))
if rows and rows[-1][0]=='gap': rows.pop()
yb=np.arange(len(rows)); off=0.2; PHYS_C='#333333'; LLM_C='#d97706'; ylabels=[]
for i,(kind,key,ax_key) in enumerate(rows):
    if kind=='rec':
        axB.barh(yb[i]+off, rec_gmean(PHYS_ALL,key), 0.38, color=PHYS_C, zorder=3)
        axB.barh(yb[i]-off, rec_gmean(MODELS,key),   0.38, color=LLM_C,  zorder=3); ylabels.append(NAME[key])
    else: ylabels.append('')
axB.set_yticks(yb); axB.set_yticklabels(ylabels,fontsize=8.5); axB.invert_yaxis()
axB.axvline(0,color='black',lw=0.6)
xmax=max(abs(rec_gmean(g,r)) for g in [PHYS_ALL,MODELS] for r in RECS)
# Draw each axis-group label at the vertical centre of its own group of bars,
# so every group is labelled (previously the last group's label was lost with its gap row).
for ax_key in AXIS_ORDER:
    idxs=[i for i,(kind,key,ak) in enumerate(rows) if kind=='rec' and ak==ax_key]
    if not idxs: continue
    y_centre=(yb[idxs[0]]+yb[idxs[-1]])/2.0
    axB.text(xmax*0.98, y_centre, AXIS_FULL[ax_key], fontsize=8,
             color='#444444', style='italic', va='center', ha='right')
for i,(kind,key,ax_key) in enumerate(rows):
    if kind=='gap':
        axB.axhline(yb[i],color='#ddd',lw=0.8)
axB.set_xlabel('Mean RankNet score (z) of patients with the recommendation',fontsize=9.5)
axB.set_title('B. Prioritisation by recommendation (grouped by ethical axis)',fontsize=12,fontweight='bold')
axB.legend(handles=[Patch(fc=PHYS_C,label='Physicians'),Patch(fc=LLM_C,label='LLMs (n=16)')],
           fontsize=8.5,frameon=False,loc='lower right')

for ax in (axA,axB):
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    ax.grid(axis='y',ls=':',alpha=.3); ax.set_axisbelow(True)
leg=[Line2D([],[],color=FAM_C,lw=3,marker='o',label='Family-medicine physicians'),
     Line2D([],[],color=PUB_C,lw=3,marker='o',label='Public-health physicians')]+\
    [Line2D([],[],color=FAMILY_COLOR[fam],lw=3,marker='s',label=fam) for fam in MODELS_BY_FAMILY]
axA.legend(handles=leg,loc='upper left',fontsize=8.5,frameon=False)
# main title removed from image (lives in the Word caption instead, per Noa's comment)
plt.tight_layout()
import os; os.makedirs('results/figures',exist_ok=True)
for ext in ['png','pdf']: fig.savefig(f'results/figures/figR_priority_by_age_and_rec.{ext}',dpi=200,bbox_inches='tight')
# numeric sanity
print('risk slope (40+ minus <10), mean z:')
for grp,nm in [(FAMILY,'family'),(PUBLIC,'public'),(MODELS,'models')]:
    r=np.nanmean([risk_curve(WR[k])[-1]-risk_curve(WR[k])[0] for k in grp])
    print(f'  {nm:8} risk(40+ - <10)={r:+.2f}')
print('per-rec RankNet means (Phys / LLM), grouped by axis:')
for rec in RECS:
    print(f'  {rec:6} {NAME[rec][:26]:26} Phys={rec_gmean(PHYS_ALL,rec):+.2f}  LLM={rec_gmean(MODELS,rec):+.2f}  [{AXIS_OF[rec]}]')
print('saved results/figures/figR_priority_by_age_and_rec.png/.pdf')
