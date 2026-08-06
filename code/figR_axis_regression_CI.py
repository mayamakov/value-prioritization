# -*- coding: utf-8 -*-
"""
Pooled choice-regression with cluster-robust 95% CIs per ethical axis (item B).
Rationale: the 26-rater Mann-Whitney is low-powered for SPARSE axes (Non-
maleficence lives on ~26/300 pairs). Pooling ALL pair-level choices and
clustering SEs by rater gives the appropriate-power view: each axis gets a
standardized coefficient + 95% CI for physicians and for LLMs separately.
Conditional-style logit on pair differences: for canonical pair (a<b),
  y = 1 if the rater chose a,   X = V[a] - V[b]   (per axis, z-standardised),
fit with NO intercept; cluster-robust sandwich SE by rater.
Self-contained (numpy only) — no statsmodels dependency.
Outputs: results/axis_regression_CI.csv  + results/figures/figR_axis_regression_CI.png/.pdf
"""
import json, numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from value_mapping_rec_intrinsic import patient_to_values, ETHICAL_DIMENSIONS as DIMS
from panel_config import HUMANS, LEGACYSET, PANEL, FAM_C, PUB_C
import value_mapping_rec_intrinsic as _vm
# unweighted axis values for this analysis (raw clinical signal); comment next line to use support weights
_vm.set_support_weights(None)

DISP = {'Beneficence-Immediate':'Beneficence-utility','Beneficence-LongTerm':'Beneficence-need',
        'Non-maleficence':'Non-maleficence','Autonomy':'Patient empowerment','Justice':'Resource justice'}
coh = pd.read_excel('synthetic_data (7).xlsx').set_index('patient_num')

# ---- common pairs (same loader convention as the pipeline) ----
PR = Path('physicians_results')
def _n(it):
    if isinstance(it,(list,tuple)) and len(it)==2 and isinstance(it[0],(list,tuple)): return (int(it[0][0]),int(it[0][1]))
    if isinstance(it,(list,tuple)) and len(it)>=2: return (int(it[0]),int(it[1]))
def _l(p):
    p=Path(p); return [x for x in (_n(z) for z in json.load(open(p))) if x] if p.exists() else []
def ch(k):
    d=PR/k; pr=[]
    for i in range(4): pr+=_l(d/f'{k}_train_iter_{i}_ranked.json')
    if k in LEGACYSET: pr+=_l(d/f'{k}_test_ranked.json')
    else:
        g=False
        for pt in 'AB':
            t=_l(d/f'{k}_test_part_{pt}_ranked.json'); pr+=t; g=g or bool(t)
        if not g: pr+=_l(d/f'{k}_test_ranked.json')
    return {tuple(sorted(p)):p[0] for p in pr}
C={k:ch(k) for k in PANEL}; present=[k for k in PANEL if C[k]]
common=sorted(set.intersection(*[set(C[k]) for k in present]))
common=[u for u in common if u[0] in coh.index and u[1] in coh.index]
pids=sorted({p for u in common for p in u})
V={pid:np.array([patient_to_values(coh.loc[pid])[d] for d in DIMS],float) for pid in pids}

# ---- design: delta = V[a]-V[b] per axis (a<b canonical), standardised ----
DELTA=np.array([V[a]-V[b] for (a,b) in common])            # (300, 5)
mu, sd = DELTA.mean(0), DELTA.std(0); sd[sd==0]=1.0

def logit_irls(X, y, iters=100, tol=1e-9):
    n,p = X.shape; beta=np.zeros(p)
    for _ in range(iters):
        eta=X@beta; mupr=1/(1+np.exp(-eta)); W=mupr*(1-mupr)
        WX=X*W[:,None]; H=X.T@WX
        g=X.T@(y-mupr)
        try: step=np.linalg.solve(H+1e-8*np.eye(p), g)
        except np.linalg.LinAlgError: break
        beta+=step
        if np.max(np.abs(step))<tol: break
    return beta
def cluster_robust_cov(X, y, beta, groups):
    eta=X@beta; mupr=1/(1+np.exp(-eta)); W=mupr*(1-mupr)
    bread=np.linalg.inv((X*W[:,None]).T@X + 1e-8*np.eye(X.shape[1]))
    meat=np.zeros((X.shape[1],X.shape[1])); r=(y-mupr)[:,None]*X
    for gid in np.unique(groups):
        sg=r[groups==gid].sum(0)[:,None]; meat+=sg@sg.T
    return bread@meat@bread

def fit_group(raters, label):
    Xrows=[]; yrows=[]; grp=[]
    for ri,k in enumerate(raters):
        cm=C[k]
        for j,(a,b) in enumerate(common):
            w=cm.get((a,b))
            if w is None: continue
            Xrows.append((DELTA[j]-mu)/sd); yrows.append(1.0 if w==a else 0.0); grp.append(ri)
    X=np.array(Xrows); y=np.array(yrows); groups=np.array(grp)
    beta=logit_irls(X,y); cov=cluster_robust_cov(X,y,beta,groups); se=np.sqrt(np.diag(cov))
    from math import erf,sqrt
    rows=[]
    for i,d in enumerate(DIMS):
        z=beta[i]/se[i] if se[i]>0 else 0.0
        p=2*(1-0.5*(1+erf(abs(z)/sqrt(2))))
        rows.append(dict(group=label, axis=DISP[d], beta=beta[i], lo=beta[i]-1.96*se[i],
                         hi=beta[i]+1.96*se[i], se=se[i], z=z, p=p,
                         n_obs=len(y), n_raters=len(raters)))
    return rows

PHYS=[k for k in PANEL if k in HUMANS]; LLM=[k for k in PANEL if k not in HUMANS]
res=fit_group(PHYS,'Physicians')+fit_group(LLM,'LLMs')
df=pd.DataFrame(res); df.to_csv('results/axis_regression_CI.csv',index=False)
pd.set_option('display.width',160)
print("Pooled choice-regression — standardised coefficient (per-SD) with cluster-robust 95% CI")
print(df[['group','axis','beta','lo','hi','p','n_obs']].to_string(index=False,
      float_format=lambda x:f'{x:+.3f}'))

# ---- forest plot ----
fig,ax=plt.subplots(figsize=(8.8,6.2))
order=[DISP[d] for d in DIMS]
ypos=np.arange(len(order))[::-1]
for grp,color,off in [('Physicians',PUB_C,+0.16),('LLMs','#d97706',-0.16)]:
    g=df[df.group==grp].set_index('axis').loc[order]
    ax.errorbar(g['beta'], ypos+off, xerr=[g['beta']-g['lo'], g['hi']-g['beta']],
                fmt='o', color=color, capsize=3, lw=1.6, ms=6, label=grp)
ax.axvline(0,color='#888',lw=1,ls='--')
ax.set_yticks(ypos); ax.set_yticklabels(order,fontsize=10)
ax.set_xlabel('Standardised effect on choice (per-SD logit coefficient)  —  95% CI, rater-clustered',fontsize=9.5)
ax.set_title('Pooled choice-regression with confidence intervals\n(appropriate-power view for sparse axes)',
             fontsize=12,fontweight='bold')
ax.legend(frameon=False,fontsize=10,loc='lower right')
for s in ['top','right']: ax.spines[s].set_visible(False)
ax.grid(axis='x',ls=':',alpha=.4); ax.set_axisbelow(True)
plt.tight_layout()
import os; os.makedirs('results/figures',exist_ok=True)
for ext in ['png','pdf']: fig.savefig(f'results/figures/figR_axis_regression_CI.{ext}',dpi=200,bbox_inches='tight')
print('\nsaved results/figures/figR_axis_regression_CI.png/.pdf + results/axis_regression_CI.csv')
