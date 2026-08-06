# -*- coding: utf-8 -*-
"""
Figure: per-recommendation UNIQUE contribution to explained choice (drop-in
McFadden pseudo-R^2), for physicians vs LLMs. For each recommendation i we
refit the full pairwise choice model WITHOUT d_rec_i and record how much
pseudo-R^2 drops; that drop is recommendation i's unique explanatory share,
already adjusted for age, risk and every other recommendation. This shows
WHICH recommendations carry the 'recommendations' bucket of the decomposition.
"""
import json, numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from panel_config import HUMANS, LLMS, LEGACYSET as LEGACY, PANEL
RECS=['rec1','rec2','rec3','rec4','rec5','rec6','rec8','rec10','rec11','rec12','rec13','rec16','rec17','rec18','rec19','rec21']
NAME={'rec1':'Basic lab panel','rec2':'Advanced lab panel','rec3':'Pathophysiology labs','rec4':'Routine lab monitoring',
      'rec5':'Diagnostic imaging','rec6':'Advanced imaging (CTA)','rec8':'Take BP/BMI measurement','rec10':'Initiate first-line statin',
      'rec11':'Initiate advanced tx','rec12':'Treatment upgrade','rec13':'Treatment replacement (CI)','rec16':'Specialist consult',
      'rec17':'Other consult (hepatology)','rec18':'Nutritional consult','rec19':'Lifestyle improvement','rec21':'Curate medical record'}
CAT={'rec10':'Treatment','rec11':'Treatment','rec12':'Treatment','rec13':'Treatment','rec16':'Consultation','rec17':'Consultation',
     'rec18':'Lifestyle/Prevention','rec19':'Lifestyle/Prevention','rec1':'Lab/Screening','rec2':'Lab/Screening','rec3':'Lab/Screening',
     'rec4':'Lab/Screening','rec5':'Imaging','rec6':'Imaging','rec8':'Lab/Screening','rec21':'Other'}
CMAP={'Treatment':'#b3261e','Imaging':'#8a5cf6','Lab/Screening':'#1f77b4','Consultation':'#0a7d6b','Lifestyle/Prevention':'#2e8b3d','Other':'#888888'}
DEMO=['d_risk']; RECF=['d_'+r for r in RECS]   # v11: risk only = priority-to-need (age subsumed)

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
def r2(D,feats,lam=2.0):
    X=np.column_stack([np.ones(len(D))]+[D[f].values for f in feats]); y=D.y.values
    P=np.eye(X.shape[1])*lam; P[0,0]=0; b=np.zeros(X.shape[1])
    for _ in range(300):
        p=1/(1+np.exp(-(X@b))); W=p*(1-p)
        step=np.linalg.solve((X.T*W)@X+P, X.T@(y-p)-P@b); b=b+step
        if np.max(np.abs(step))<1e-11: break
    p=np.clip(1/(1+np.exp(-(X@b))),1e-12,1-1e-12)
    ll=np.sum(y*np.log(p)+(1-y)*np.log(1-p)); yb=y.mean(); ll0=np.sum(y*np.log(yb)+(1-y)*np.log(1-yb))
    return 1-ll/ll0
def dropin(raters):
    # unique drop-in pseudo-R2 for EVERY predictor: the 2 demographics + all recs
    D=build(raters); full=r2(D,DEMO+RECF); ALL=DEMO+RECF
    return {f: full - r2(D,[g for g in ALL if g!=f]) for f in ALL}

def dropin_recs(raters):
    D=build(raters); full=r2(D,DEMO+RECF); ALL=DEMO+RECF
    rec={f: full - r2(D,[g for g in ALL if g!=f]) for f in RECF}
    demo_block = full - r2(D,RECF)        # remove age AND risk together (they are collinear)
    return rec, demo_block

dh, demoH = dropin_recs(HUMANS); dl, demoL = dropin_recs(LLMS)

# ---- group recommendations by ETHICAL AXIS (v11) ----
AXIS_OF = {
 'rec1':'Utility','rec2':'Utility','rec3':'Utility','rec4':'Utility','rec5':'Utility',
 'rec6':'Utility','rec8':'Utility','rec10':'Utility','rec11':'Utility','rec12':'Utility',
 'rec16':'Utility','rec21':'Utility',
 'rec13':'Non-maleficence','rec17':'Non-maleficence',
 'rec18':'Patient empowerment','rec19':'Patient empowerment',
}
AXIS_ORDER = ['Utility','Non-maleficence','Patient empowerment']   # bottom -> up; risk block on top
AXIS_FULL  = {'Utility':'Beneficence-utility','Non-maleficence':'Non-maleficence',
              'Patient empowerment':'Patient empowerment'}
AXIS_COLOR = {'Utility':'#1f77b4','Non-maleficence':'#d97706','Patient empowerment':'#2e8b3d',
              'Priority-to-need':'#444444'}
def lab(rec): return NAME[rec]

rows=[]   # ('rec',recname,axis) | ('gap',axis,axis) | ('risk','RISK','Priority-to-need')
for ax_key in AXIS_ORDER:
    recs_in=sorted([r for r in RECS if AXIS_OF.get(r)==ax_key],
                   key=lambda r: max(dh['d_'+r], dl['d_'+r]))
    for r in recs_in: rows.append(('rec', r, ax_key))
    rows.append(('gap', ax_key, ax_key))
rows.append(('risk','RISK','Priority-to-need'))

y=np.arange(len(rows)); off=0.18
fig,ax=plt.subplots(figsize=(9.8,9.2))
PHYS='#333333'; LLM='#d97706'; ylabels=[]
for i,(kind,key,ax_key) in enumerate(rows):
    if kind=='rec':
        ax.barh(y[i]+off, dh['d_'+key], 0.34, color=PHYS, zorder=3)
        ax.barh(y[i]-off, dl['d_'+key], 0.34, color=LLM,  zorder=3); ylabels.append(lab(key))
    elif kind=='risk':
        ax.barh(y[i]+off, demoH, 0.34, color=PHYS, zorder=3)
        ax.barh(y[i]-off, demoL, 0.34, color=LLM,  zorder=3); ylabels.append('Patient risk')
    else:
        ylabels.append('')
ax.set_yticks(y); ax.set_yticklabels(ylabels,fontsize=10)
for t,(kind,key,ax_key) in zip(ax.get_yticklabels(),rows):
    if kind=='risk': t.set_fontweight('bold')
xmax=max([dh['d_'+r] for r in RECS]+[dl['d_'+r] for r in RECS]+[demoH,demoL])
for i,(kind,key,ax_key) in enumerate(rows):
    if kind=='gap':
        ax.axhline(y[i],color='#ddd',lw=0.8)
        ax.text(xmax*0.99,y[i],AXIS_FULL[key],fontsize=8.5,color='#444444',style='italic',va='center',ha='right')
ax.axhline(y[-1]-0.5,color='#bbb',lw=1.0)
ax.set_xlabel('Unique contribution to explained choice (drop-in McFadden pseudo-R\u00B2)\n'
              'recommendations grouped by ethical axis; each adjusted for all others; '
              'top block = patient risk (priority-to-need)',fontsize=9.0)
ax.set_title('What drives each group\u2019s choices, by ethical axis',fontsize=12.5,fontweight='bold')
h=[Patch(fc=PHYS,label='Physicians'),Patch(fc=LLM,label='LLMs')]
l1=ax.legend(handles=h,loc='lower right',fontsize=10,frameon=False,title='Rater group'); ax.add_artist(l1)
ax.grid(axis='x',ls=':',alpha=.4); ax.set_axisbelow(True)
for s in ['top','right']: ax.spines[s].set_visible(False)
ax.text(0.0,1.005,'Resource justice is price-based and distributed across all recommendations (not a single-rec group).',
        transform=ax.transAxes,fontsize=7.2,color='#999',va='bottom')
plt.tight_layout()
import os; os.makedirs('results/figures',exist_ok=True)
for ext in ['png','pdf']: fig.savefig(f'results/figures/figR_rec_pseudoR2.{ext}',dpi=200,bbox_inches='tight')
print('predictor                     phys_dR2   llm_dR2')
print(f'{"Patient risk (priority)":28s}  {demoH:+.4f}   {demoL:+.4f}')
for ax_key in AXIS_ORDER:
    for r in sorted([r for r in RECS if AXIS_OF.get(r)==ax_key], key=lambda r:-max(dh["d_"+r],dl["d_"+r])):
        print(f'{lab(r):28s}  {dh["d_"+r]:+.4f}   {dl["d_"+r]:+.4f}   [{ax_key}]')
print('saved results/figures/figR_rec_pseudoR2.png/.pdf')
