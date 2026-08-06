# -*- coding: utf-8 -*-
"""
Figure: what drives each group's choices -- patient demographics (age+risk)
vs the recommendation profile? For physicians and for LLMs separately we fit
pairwise logistic choice models and decompose explained variance (McFadden
pseudo-R^2) into two blocks:
    DEMO  = {d_age, d_risk}
    RECS  = {d_rec_i for all recommendations}
We report each block's UNIQUE contribution (drop-in-pseudo-R^2 when that block
is removed from the full model) and the SHARE of the full model's pseudo-R^2
attributable to each block. SE via raters as the resampling unit (jackknife).
"""
import json, numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from panel_config import HUMANS, LLMS, LEGACYSET as LEGACY, PANEL
RECS=['rec1','rec2','rec3','rec4','rec5','rec6','rec8','rec10','rec11','rec12','rec13','rec16','rec17','rec18','rec19','rec21']

def _n(it):
    if isinstance(it,(list,tuple)) and len(it)==2 and isinstance(it[0],(list,tuple)): return (int(it[0][0]),int(it[0][1]))
    if isinstance(it,(list,tuple)) and len(it)>=2: return (int(it[0]),int(it[1]))
def _l(p):
    p=Path(p); return [x for x in (_n(z) for z in json.load(open(p))) if x] if p.exists() else []
def ch(k):
    d=Path('physicians_results')/k; pr=[]
    for i in range(4): pr+=_l(d/f'{k}_train_iter_{i}_ranked.json')
    if k in LEGACY: pr+=_l(d/f'{k}_test_ranked.json')
    else:
        g=False
        for pt in 'AB':
            t=_l(d/f'{k}_test_part_{pt}_ranked.json'); pr+=t; g=g or bool(t)
        if not g: pr+=_l(d/f'{k}_test_ranked.json')
    return {tuple(sorted(p)):p[0] for p in pr}
import glob as _glob, os as _os
def _find_synth():
    try:
        _here = _os.path.dirname(_os.path.abspath(__file__))
    except NameError:
        _here = _os.getcwd()
    for _base in ['.', _here, _os.path.expanduser('~'),
                  _os.path.join(_os.path.expanduser('~'), 'PATCHED_CODE', 'PATCHED_CODE')]:
        _hits = sorted(_glob.glob(_os.path.join(_base, 'synthetic_data*.xlsx')))
        if _hits: return _hits[0]
    raise FileNotFoundError("synthetic_data*.xlsx not found (CWD / script dir / home / PATCHED_CODE)")
coh=pd.read_excel(_find_synth()).set_index('patient_num')
C={k:ch(k) for k in PANEL}
common=set.intersection(*[set(C[k]) for k in PANEL]); common=[u for u in common if u[0] in coh.index and u[1] in coh.index]

DEMO=['d_age','d_risk']; RECF=['d_'+r for r in RECS]
def build(raters):
    rows=[]
    for k in raters:
        cc=C[k]
        for u in common:
            a,b=u; r={'d_age':float(coh.loc[a,'age']-coh.loc[b,'age']),'d_risk':float(coh.loc[a,'risk']-coh.loc[b,'risk'])}
            for rec in RECS: r['d_'+rec]=float(coh.loc[a,rec]>=0.5)-float(coh.loc[b,rec]>=0.5)
            r['y']=1.0 if cc[u]==a else 0.0
            rows.append(r)
    return pd.DataFrame(rows)

def fit_ll(D,feats,lam=2.0):
    """penalised logistic; return log-likelihood and null LL for McFadden R2."""
    X=np.column_stack([np.ones(len(D))]+[D[f].values for f in feats]); y=D.y.values
    P=np.eye(X.shape[1])*lam; P[0,0]=0
    b=np.zeros(X.shape[1])
    for _ in range(300):
        p=1/(1+np.exp(-(X@b))); W=p*(1-p)
        step=np.linalg.solve((X.T*W)@X+P, X.T@(y-p)-P@b); b=b+step
        if np.max(np.abs(step))<1e-11: break
    p=np.clip(1/(1+np.exp(-(X@b))),1e-12,1-1e-12)
    ll=np.sum(y*np.log(p)+(1-y)*np.log(1-p))
    ybar=y.mean(); ll0=np.sum(y*np.log(ybar)+(1-y)*np.log(1-ybar))
    return ll, ll0

def decompose(D):
    ll_full,ll0=fit_ll(D,DEMO+RECF)
    ll_demo,_  =fit_ll(D,DEMO)            # demo only
    ll_recs,_  =fit_ll(D,RECF)            # recs only
    R2_full=1-ll_full/ll0
    # unique contributions = drop in McFadden R2 when a block is removed
    uniq_demo=(ll_full-ll_recs)/(-ll0)    # = R2_full - R2_recsOnly
    uniq_recs=(ll_full-ll_demo)/(-ll0)    # = R2_full - R2_demoOnly
    return dict(R2_full=R2_full,
                R2_demo_only=1-ll_demo/ll0, R2_recs_only=1-ll_recs/ll0,
                uniq_demo=uniq_demo, uniq_recs=uniq_recs)

def group_stats(raters):
    full=decompose(build(raters))
    # jackknife over raters for SE on the shares
    shares=[]
    for k in raters:
        d=decompose(build([r for r in raters if r!=k]))
        tot=d['uniq_demo']+d['uniq_recs']+1e-12
        shares.append(d['uniq_recs']/(d['uniq_demo']+d['uniq_recs']))
    shares=np.array(shares); n=len(raters)
    jack_mean=shares.mean(); jack_se=np.sqrt((n-1)/n*np.sum((shares-shares.mean())**2))
    tot=full['uniq_demo']+full['uniq_recs']
    return full, full['uniq_recs']/tot, jack_se

(fp, rec_share_h, se_h) = group_stats(HUMANS)
(fl, rec_share_l, se_l) = group_stats(LLMS)

print("== McFadden pseudo-R2 decomposition ==")
for nm,f,rs in [('Physicians',fp,rec_share_h),('LLMs',fl,rec_share_l)]:
    print(f"{nm}: R2_full={f['R2_full']:.3f} | unique DEMO={f['uniq_demo']:.3f}  unique RECS={f['uniq_recs']:.3f} "
          f"| recs share={rs*100:.0f}%  demo share={(1-rs)*100:.0f}%")

# ---- figure: stacked unique-contribution bars + recommendation-share with jackknife CI ----
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12.5,5.2),gridspec_kw={'width_ratios':[1.15,1]})
DEMOC='#4c78a8'; RECC='#e4572e'
groups=['Physicians','LLMs']; F=[fp,fl]
x=np.arange(2); w=.5
demo=[f['uniq_demo'] for f in F]; recs=[f['uniq_recs'] for f in F]
ax1.bar(x,demo,w,color=DEMOC,label='Age + risk (demographics)')
ax1.bar(x,recs,w,bottom=demo,color=RECC,label='Recommendation profile')
for i,f in enumerate(F):
    ax1.text(i,demo[i]/2,f'{demo[i]:.3f}',ha='center',va='center',color='white',fontsize=10,fontweight='bold')
    ax1.text(i,demo[i]+recs[i]/2,f'{recs[i]:.3f}',ha='center',va='center',color='white',fontsize=10,fontweight='bold')
ax1.set_xticks(x); ax1.set_xticklabels(groups,fontsize=11)
ax1.set_ylabel('Unique contribution to explained choice\n(McFadden pseudo-R\u00B2)',fontsize=10)
ax1.set_title('What explains each group\u2019s choices?',fontsize=12,fontweight='bold')
ax1.legend(fontsize=9,frameon=False,loc='upper center'); 
for s in ['top','right']: ax1.spines[s].set_visible(False)
ax1.grid(axis='y',ls=':',alpha=.4); ax1.set_axisbelow(True)

share=[rec_share_h*100,rec_share_l*100]; se=[se_h*100,se_l*100]
ax2.barh(x,share,.5,color=RECC,xerr=[1.96*s for s in se],capsize=4,error_kw=dict(ecolor='#333',lw=1.2))
for i,(sh,d) in enumerate(zip(share,[100-share[0],100-share[1]])):
    ax2.text(sh-2,x[i],f'{sh:.0f}%',ha='right',va='center',color='white',fontsize=11,fontweight='bold')
    ax2.text(101,x[i],f'(demographics {d:.0f}%)',ha='left',va='center',color=DEMOC,fontsize=9)
ax2.axvline(50,color='#888',ls='--',lw=.9)
ax2.set_xlim(0,140); ax2.set_yticks(x); ax2.set_yticklabels(groups,fontsize=11)
ax2.set_xlabel('Share of explained choice driven by the RECOMMENDATIONS\n(vs demographics) \u2014 95% jackknife CI over raters',fontsize=9.5)
ax2.set_title('Recommendations vs demographics',fontsize=12,fontweight='bold')
for s in ['top','right']: ax2.spines[s].set_visible(False)
ax2.grid(axis='x',ls=':',alpha=.4); ax2.set_axisbelow(True)
plt.tight_layout()
import os; os.makedirs('results/figures',exist_ok=True)
for ext in ['png','pdf']: fig.savefig(f'results/figures/figR_decomposition.{ext}',dpi=200,bbox_inches='tight')
print('saved results/figures/figR_decomposition.png/.pdf')