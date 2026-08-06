# -*- coding: utf-8 -*-
"""
Per-rater pseudo-R^2 decomposition, slope-chart style (one line per rater).
For EACH rater we fit a pairwise logit (300 common pairs, ridge) and split the
explained choice (McFadden) into two buckets via drop-in pseudo-R^2:
   demographics = {d_age, d_risk}   vs   recommendations = {all d_rec}.
Each rater is a line connecting its demographics-share to its recommendations-share.
Models = brown shades; family-medicine + public-health physicians = blue shades.
"""
import json, numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from panel_config import (FAMILY, PUBLIC, MODELS, LEGACYSET, PANEL, display,
                          model_color, FAM_C, PUB_C, FAMILY_COLOR, MODELS_BY_FAMILY)
PHYS=FAMILY+PUBLIC
RECS=['rec1','rec2','rec3','rec4','rec5','rec6','rec8','rec10','rec11','rec12','rec13','rec16','rec17','rec18','rec19','rec21']

def _n(it):
    if isinstance(it,(list,tuple)) and len(it)==2 and isinstance(it[0],(list,tuple)): return (int(it[0][0]),int(it[0][1]))
    if isinstance(it,(list,tuple)) and len(it)>=2: return (int(it[0]),int(it[1]))
def _l(p):
    p=Path(p); return [x for x in (_n(z) for z in json.load(open(p))) if x] if p.exists() else []
def ch(k):
    d=Path('physicians_results')/k; pr=[]
    for i in range(4): pr+=_l(d/f'{k}_train_iter_{i}_ranked.json')
    if k in LEGACYSET: pr+=_l(d/f'{k}_test_ranked.json')
    else:
        g=False
        for pt in 'AB':
            t=_l(d/f'{k}_test_part_{pt}_ranked.json'); pr+=t; g=g or bool(t)
        if not g: pr+=_l(d/f'{k}_test_ranked.json')
    return {tuple(sorted(p)):p[0] for p in pr}
coh=pd.read_excel('synthetic_data (7).xlsx').set_index('patient_num')
C={k:ch(k) for k in PANEL}
common=set.intersection(*[set(C[k]) for k in PANEL]); common=[u for u in common if u[0] in coh.index and u[1] in coh.index]
DEMO=['d_age','d_risk']; RECF=['d_'+r for r in RECS]; FEAT=DEMO+RECF

def design(k):
    cc=C[k]; rows=[]
    for u in common:
        a,b=u; r={'d_age':float(coh.loc[a,'age']-coh.loc[b,'age']),'d_risk':float(coh.loc[a,'risk']-coh.loc[b,'risk'])}
        for rec in RECS: r['d_'+rec]=float(coh.loc[a,rec]>=0.5)-float(coh.loc[b,rec]>=0.5)
        r['y']=1.0 if cc[u]==a else 0.0; rows.append(r)
    return pd.DataFrame(rows)
def r2(D,feats,lam=3.0):
    X=np.column_stack([np.ones(len(D))]+[D[f].values for f in feats]); y=D.y.values
    P=np.eye(X.shape[1])*lam; P[0,0]=0; b=np.zeros(X.shape[1])
    for _ in range(300):
        p=1/(1+np.exp(-(X@b))); W=p*(1-p)
        step=np.linalg.solve((X.T*W)@X+P,X.T@(y-p)-P@b); b=b+step
        if np.max(np.abs(step))<1e-11: break
    p=np.clip(1/(1+np.exp(-(X@b))),1e-12,1-1e-12)
    ll=np.sum(y*np.log(p)+(1-y)*np.log(1-p)); yb=y.mean(); ll0=np.sum(y*np.log(yb)+(1-y)*np.log(1-yb))
    return 1-ll/ll0
def shares(k):
    D=design(k); full=r2(D,FEAT)
    ud=full-r2(D,RECF); ur=full-r2(D,DEMO); tot=ud+ur
    if tot<=0: return 0.5,0.5
    return ud/tot, ur/tot   # demo_share, recs_share

S={k:shares(k) for k in PANEL}
print(f'{"rater":18}{"demo%":>7}{"recs%":>7}')
for k in PANEL: print(f'{display(k):18}{S[k][0]*100:7.0f}{S[k][1]*100:7.0f}')

# ---- slope chart ----
import matplotlib as mpl
fig,ax=plt.subplots(figsize=(8.8,7.2))
x=[0,1]
for k in FAMILY:
    ax.plot(x,[S[k][0],S[k][1]],color=FAM_C,lw=1.5,alpha=.8,zorder=2,marker='o',ms=3)
for k in PUBLIC:
    ax.plot(x,[S[k][0],S[k][1]],color=PUB_C,lw=1.5,alpha=.8,zorder=2,marker='o',ms=3)
for k in MODELS:
    ax.plot(x,[S[k][0],S[k][1]],color=model_color(k),lw=1.6,alpha=.9,zorder=3,marker='s',ms=3)
# group means (thick)
fmean=[np.mean([S[k][0] for k in FAMILY]),np.mean([S[k][1] for k in FAMILY])]
pmean=[np.mean([S[k][0] for k in PUBLIC]),np.mean([S[k][1] for k in PUBLIC])]
mm=[np.mean([S[k][0] for k in MODELS]),np.mean([S[k][1] for k in MODELS])]
ax.plot(x,fmean,color=FAM_C,lw=4,zorder=5,marker='o',ms=7)
ax.plot(x,pmean,color=PUB_C,lw=4,zorder=5,marker='o',ms=7)
ax.plot(x,mm,color='#444444',lw=4,zorder=5,marker='s',ms=7)  # all-models mean
# model labels on right
for k in MODELS:
    ax.annotate(display(k),(1.02,S[k][1]),fontsize=6.5,color=model_color(k),va='center',ha='left')
ax.set_xlim(-0.18,1.5); ax.set_ylim(0,1)
ax.set_xticks([0,1]); ax.set_xticklabels(['Demographics\n(age + risk)','Recommendations'],fontsize=11)
ax.set_ylabel('Share of that rater\u2019s explained choice (McFadden pseudo-R\u00B2)',fontsize=10.5)
ax.set_title('What drives each rater\u2019s choices \u2014 one line per physician and model\n'
             'demographics vs recommendations (per-rater decomposition)',fontsize=12.5,fontweight='bold')
ax.axhline(0.5,color='#bbb',ls=':',lw=.8)
leg=[Line2D([],[],color=FAM_C,lw=3,marker='o',label='Family-medicine physicians'),
     Line2D([],[],color=PUB_C,lw=3,marker='o',label='Public-health physicians')]+\
    [Line2D([],[],color=FAMILY_COLOR[fam],lw=3,marker='s',label=fam) for fam in MODELS_BY_FAMILY]
ax.legend(handles=leg,loc='center left',fontsize=8,frameon=False,ncol=1)
for s in ['top','right']: ax.spines[s].set_visible(False)
ax.grid(axis='y',ls=':',alpha=.3); ax.set_axisbelow(True)
plt.tight_layout()
import os; os.makedirs('results/figures',exist_ok=True)
for ext in ['png','pdf']: fig.savefig(f'results/figures/figR_decomposition_per_rater.{ext}',dpi=200,bbox_inches='tight')
print('saved results/figures/figR_decomposition_per_rater.png/.pdf')
