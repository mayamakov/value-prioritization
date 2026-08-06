# -*- coding: utf-8 -*-
"""
LINE version of the per-predictor drop-in pseudo-R^2 figure.
x-axis = predictors (Age+risk block, then each recommendation);
y-axis = unique drop-in McFadden pseudo-R^2 (ridge logit, per entity).
One line per LANGUAGE MODEL (brown shades), one line for FAMILY-MEDICINE
physicians (light blue), one line for PUBLIC-HEALTH physicians (navy).
Each line shows what drives that rater/group: models peak on Age+risk,
physicians peak on the treatment recommendations.
"""
import json, numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt, matplotlib as mpl
from matplotlib.lines import Line2D

from panel_config import (FAMILY, PUBLIC, MODELS, LEGACYSET, PANEL, display as _disp,
                          model_color, FAM_C, PUB_C, FAMILY_COLOR, MODELS_BY_FAMILY)
MNAME={k:_disp(k) for k in MODELS}
RECS=['rec1','rec2','rec3','rec4','rec5','rec6','rec8','rec10','rec11','rec12','rec13','rec16','rec17','rec18','rec19','rec21']
RNAME={'rec1':'Basic labs','rec2':'Advanced labs','rec3':'Pathophys labs','rec4':'Monitoring','rec5':'Imaging (Doppler)',
       'rec6':'Adv. imaging (CTA)','rec8':'BP/BMI','rec10':'First-line statin','rec11':'Advanced tx (PCSK9)','rec12':'Tx upgrade',
       'rec13':'Tx replacement','rec16':'Specialist consult','rec17':'Hepatology consult','rec18':'Dietitian','rec19':'Lifestyle','rec21':'Curate records'}
DEMO=['d_age','d_risk']; RECF=['d_'+r for r in RECS]

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

def build(raters):
    rows=[]
    for k in raters:
        cc=C[k]
        for u in common:
            a,b=u; r={'d_age':float(coh.loc[a,'age']-coh.loc[b,'age']),'d_risk':float(coh.loc[a,'risk']-coh.loc[b,'risk'])}
            for rec in RECS: r['d_'+rec]=float(coh.loc[a,rec]>=0.5)-float(coh.loc[b,rec]>=0.5)
            r['y']=1.0 if cc[u]==a else 0.0; rows.append(r)
    return pd.DataFrame(rows)
def r2(D,feats,lam=2.0):
    X=np.column_stack([np.ones(len(D))]+[D[f].values for f in feats]); y=D.y.values
    P=np.eye(X.shape[1])*lam; P[0,0]=0; b=np.zeros(X.shape[1])
    for _ in range(300):
        p=1/(1+np.exp(-(X@b))); W=p*(1-p)
        step=np.linalg.solve((X.T*W)@X+P,X.T@(y-p)-P@b); b=b+step
        if np.max(np.abs(step))<1e-11: break
    p=np.clip(1/(1+np.exp(-(X@b))),1e-12,1-1e-12)
    ll=np.sum(y*np.log(p)+(1-y)*np.log(1-p)); yb=y.mean(); ll0=np.sum(y*np.log(yb)+(1-y)*np.log(1-yb))
    return 1-ll/ll0
def profile(raters):
    """drop-in dR2 for Age+risk block and each recommendation."""
    D=build(raters); full=r2(D,DEMO+RECF); ALL=DEMO+RECF
    demo=full-r2(D,RECF)
    rec={r: full-r2(D,[g for g in ALL if g!='d_'+r]) for r in RECS}
    return demo,rec

# compute profiles
prof={}
for k in MODELS: prof[k]=profile([k])
prof['FAMILY']=profile(FAMILY); prof['PUBLIC']=profile(PUBLIC)

# x order: DEMO first, then recs by physician (family+public) importance
recscore={r: max(prof['FAMILY'][1][r],prof['PUBLIC'][1][r]) for r in RECS}
rec_order=sorted(RECS,key=lambda r:-recscore[r])
xlabels=['Age + risk']+[RNAME[r] for r in rec_order]
xx=np.arange(len(xlabels))
def series(key):
    demo,rec=prof[key]; return [demo]+[rec[r] for r in rec_order]

fig,ax=plt.subplots(figsize=(12.5,6.8))
for k in MODELS:
    ax.plot(xx,series(k),color=model_color(k),lw=1.4,marker='s',ms=3,alpha=.9,zorder=3,label=MNAME[k])
ax.plot(xx,series('FAMILY'),color=FAM_C,lw=3.2,marker='o',ms=6,zorder=5,label='Family-medicine physicians')
ax.plot(xx,series('PUBLIC'),color=PUB_C,lw=3.2,marker='o',ms=6,zorder=5,label='Public-health physicians')
ax.axvline(0.5,color='#bbb',ls=':',lw=.9)
ax.text(0.0,ax.get_ylim()[1],'demographics',fontsize=8.5,color='#777',ha='center',va='bottom')
ax.set_xticks(xx); ax.set_xticklabels(xlabels,rotation=45,ha='right',fontsize=9)
ax.set_ylabel('Unique contribution to explained choice\n(drop-in McFadden pseudo-R\u00B2)',fontsize=10.5)
ax.set_title('What drives each rater\u2019s choices \u2014 one line per model, plus family-medicine and public-health physicians',
             fontsize=12.5,fontweight='bold')
ax.grid(axis='y',ls=':',alpha=.4); ax.set_axisbelow(True)
for s in ['top','right']: ax.spines[s].set_visible(False)
# legend: groups first then models
handles=[Line2D([],[],color=FAM_C,lw=3,marker='o',label='Family-medicine physicians'),
         Line2D([],[],color=PUB_C,lw=3,marker='o',label='Public-health physicians')]+\
        [Line2D([],[],color=FAMILY_COLOR[fam],lw=3,marker='s',label=fam) for fam in MODELS_BY_FAMILY]
ax.legend(handles=handles,loc='upper right',fontsize=8.5,frameon=False,ncol=1)
plt.tight_layout()
import os; os.makedirs('results/figures',exist_ok=True)
for ext in ['png','pdf']: fig.savefig(f'results/figures/figR_rec_pseudoR2_lines.{ext}',dpi=200,bbox_inches='tight')
print(f'{"entity":20}{"Age+risk":>9}{"statin":>8}{"top rec":>22}')
for key in ['FAMILY','PUBLIC']+MODELS:
    demo,rec=prof[key]; top=max(rec,key=rec.get)
    nm=key if key in ('FAMILY','PUBLIC') else MNAME[key]
    print(f'{nm:20}{demo:9.3f}{rec["rec10"]:8.3f}   {RNAME[top]:>19}')
print('saved results/figures/figR_rec_pseudoR2_lines.png/.pdf')
