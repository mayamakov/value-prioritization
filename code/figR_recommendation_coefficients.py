# -*- coding: utf-8 -*-
"""
Figure: per-recommendation effect on patient selection, net of age & risk.
For physicians and for LLMs separately we fit a pairwise logistic model
  P(rater chose patient a over b) ~ d_age + d_risk + sum_i d_rec_i
on all 300 common pairs (pooled within group; SE clustered by pair; light L2).
The coefficient of d_rec_i = how strongly that group is drawn to a patient
BECAUSE they carry recommendation i, holding age, risk and every OTHER
recommendation fixed. The physician-vs-LLM gap per recommendation is the
age-adjusted contribution of that recommendation to the divergence.
"""
import json, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from panel_config import HUMANS, LLMS, LEGACYSET as LEGACY, PANEL
RECS=['rec1','rec2','rec3','rec4','rec5','rec6','rec8','rec10','rec11','rec12','rec13','rec16','rec17','rec18','rec19','rec21']
NAME={'rec1':'Basic lab panel','rec2':'Advanced lab panel','rec3':'Pathophysiology labs','rec4':'Routine lab monitoring',
      'rec5':'Diagnostic imaging','rec6':'Advanced imaging (CTA)','rec8':'Take BP/BMI measurement','rec10':'Initiate first-line statin',
      'rec11':'Initiate advanced tx','rec12':'Treatment upgrade','rec13':'Treatment replacement (CI)','rec16':'Specialist consult',
      'rec17':'Other consult (hepato/cardio)','rec18':'Nutritional consult','rec19':'Lifestyle improvement','rec21':'Curate medical record'}
CAT={'rec10':'Treatment','rec11':'Treatment','rec12':'Treatment','rec13':'Treatment','rec16':'Consultation','rec17':'Consultation',
     'rec18':'Lifestyle/Prevention','rec19':'Lifestyle/Prevention','rec1':'Lab/Screening','rec2':'Lab/Screening','rec3':'Lab/Screening',
     'rec4':'Lab/Screening','rec5':'Imaging','rec6':'Imaging','rec8':'Lab/Screening','rec21':'Other'}
CMAP={'Treatment':'#b3261e','Imaging':'#8a5cf6','Lab/Screening':'#1f77b4','Consultation':'#0a7d6b','Lifestyle/Prevention':'#2e8b3d','Other':'#888888'}

def _norm(it):
    if isinstance(it,(list,tuple)) and len(it)==2 and isinstance(it[0],(list,tuple)): return (int(it[0][0]),int(it[0][1]))
    if isinstance(it,(list,tuple)) and len(it)>=2: return (int(it[0]),int(it[1]))
def _load(p):
    p=Path(p); return [pr for pr in (_norm(x) for x in json.load(open(p))) if pr] if p.exists() else []
def ch(k):
    d=Path('physicians_results')/k; prs=[]
    for i in range(4): prs+=_load(d/f'{k}_train_iter_{i}_ranked.json')
    if k in LEGACY: prs+=_load(d/f'{k}_test_ranked.json')
    else:
        g=False
        for pt in 'AB':
            t=_load(d/f'{k}_test_part_{pt}_ranked.json'); prs+=t; g=g or bool(t)
        if not g: prs+=_load(d/f'{k}_test_ranked.json')
    return {tuple(sorted(p)):p[0] for p in prs}
coh=pd.read_excel('synthetic_data (7).xlsx').set_index('patient_num')
C={k:ch(k) for k in PANEL}
common=set.intersection(*[set(C[k]) for k in PANEL]); common=[u for u in common if u[0] in coh.index and u[1] in coh.index]

FEAT=['d_age','d_risk']+['d_'+r for r in RECS]
def design(raters):
    rows=[]; grp=[]
    for k in raters:
        cc=C[k]
        for u in common:
            a,b=u; r={'d_age':float(coh.loc[a,'age']-coh.loc[b,'age']),'d_risk':float(coh.loc[a,'risk']-coh.loc[b,'risk'])}
            for rec in RECS: r['d_'+rec]=float(coh.loc[a,rec]>=0.5)-float(coh.loc[b,rec]>=0.5)
            r['y']=1.0 if cc[u]==a else 0.0; r['pair']=f'{a}_{b}'
            rows.append(r); grp.append(k)
    return pd.DataFrame(rows)

# penalised IRLS (small ridge on slopes, intercept unpenalised) + cluster-robust SE
def fit(D, lam=2.0):
    X=np.column_stack([np.ones(len(D))]+[D[c].values for c in FEAT]); y=D.y.values; groups=D.pair.values
    P=np.eye(X.shape[1])*lam; P[0,0]=0.0
    b=np.zeros(X.shape[1])
    for _ in range(200):
        p=1/(1+np.exp(-(X@b))); W=p*(1-p)
        H=(X.T*W)@X+P; g=X.T@(y-p)-P@b
        step=np.linalg.solve(H,g); b=b+step
        if np.max(np.abs(step))<1e-10: break
    p=1/(1+np.exp(-(X@b))); W=p*(1-p); Hinv=np.linalg.inv((X.T*W)@X+P)
    u=(y-p)[:,None]*X - (P@b)[None,:]/len(np.unique(groups))
    meat=sum(np.outer(u[groups==gg].sum(0),u[groups==gg].sum(0)) for gg in np.unique(groups))
    G=len(np.unique(groups)); N,K=X.shape
    se=np.sqrt(np.diag(Hinv@(((G/(G-1))*((N-1)/(N-K)))*meat)@Hinv))
    names=['const']+FEAT
    return {n:(b[i],se[i]) for i,n in enumerate(names)}

fp=fit(design(HUMANS)); fl=fit(design(LLMS))
gap={r: fp['d_'+r][0]-fl['d_'+r][0] for r in RECS}
order=sorted(RECS,key=lambda r:gap[r])     # LLM-favoured bottom, physician-favoured top

fig,ax=plt.subplots(figsize=(9.4,7.8))
ax.axvline(0,color='#444',lw=.8)
y=np.arange(len(order)); off=0.16; z=1.96
PHYS='#333333'; LLM='#d97706'
for i,r in enumerate(order):
    bp,sp=fp['d_'+r]; bl,sl=fl['d_'+r]
    ax.errorbar(bp,y[i]+off,xerr=z*sp,fmt='o',ms=6,color=PHYS,elinewidth=1.4,capsize=2,zorder=3)
    ax.errorbar(bl,y[i]-off,xerr=z*sl,fmt='s',ms=6,color=LLM,elinewidth=1.4,capsize=2,zorder=3)
ax.set_yticks(y); ax.set_yticklabels([NAME[r] for r in order],fontsize=10)
for t,r in zip(ax.get_yticklabels(),order): t.set_color(CMAP[CAT[r]])
ax.set_ylim(-.6,len(order)-.4)
ax.set_xlabel('Logistic coefficient (log-odds of selecting the patient who has the recommendation)\nadjusted for \u0394age, \u0394risk and all other recommendations  \u2014  95% CI',fontsize=9.5)
ax.set_title('Per-recommendation effect on patient selection, net of age\nPhysicians vs LLMs (all 300 common pairs)',fontsize=12,fontweight='bold')
h=[plt.Line2D([],[],marker='o',ls='',ms=7,color=PHYS,label='Physicians (n=10)'),
   plt.Line2D([],[],marker='s',ls='',ms=7,color=LLM,label='LLMs (n=8)')]
l1=ax.legend(handles=h,loc='lower right',fontsize=9,frameon=False,title='Rater group'); ax.add_artist(l1)
ax.legend(handles=[Patch(fc=CMAP[c],label=c) for c in ['Treatment','Imaging','Lab/Screening','Consultation','Lifestyle/Prevention']],
          loc='upper left',fontsize=8,frameon=False,title='Recommendation category')
ax.grid(axis='x',ls=':',alpha=.4); ax.set_axisbelow(True)
for s in ['top','right']: ax.spines[s].set_visible(False)
plt.tight_layout()
import os; os.makedirs('results/figures',exist_ok=True)
for ext in ['png','pdf']: fig.savefig(f'results/figures/figR_recommendation_coefficients.{ext}',dpi=200,bbox_inches='tight')
print('physician-favoured (top gap):',[(r,round(gap[r],2)) for r in order[::-1][:4]])
print('LLM-favoured (bottom gap):   ',[(r,round(gap[r],2)) for r in order[:4]])
print('saved results/figures/figR_recommendation_coefficients.png/.pdf')
