# -*- coding: utf-8 -*-
"""
Figure: per-recommendation prioritization differential on AGE+RISK-MATCHED pairs.
For each recommendation r, bar = mean over raters in the group of
  ( has_r(chosen) - has_r(not-chosen) )  averaged over matched pairs.
Positive  -> the group tends to PICK the patient who has recommendation r.
Restricting to matched pairs (|dage|<=5, |drisk|<=5) neutralises age, so the
contrast reflects recommendation prioritisation, not age.
"""
import json, numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RESULTS_DIR='physicians_results'; COHORT='synthetic_data (7).xlsx'
from panel_config import HUMANS, LLMS, LEGACYSET as LEGACY, PANEL
MATCH_AGE,MATCH_RISK=5,5

RECS=['rec10','rec12','rec11','rec13','rec16','rec17','rec4','rec2','rec8',
      'rec1','rec3','rec5','rec6','rec18','rec19','rec21']
NAME={'rec1':'Basic lab panel','rec2':'Advanced lab panel','rec3':'Pathophysiology labs',
      'rec4':'Routine lab monitoring','rec5':'Diagnostic imaging','rec6':'Advanced imaging (CTA)',
      'rec8':'Take BP/BMI measurement','rec10':'Initiate first-line statin','rec11':'Initiate advanced tx',
      'rec12':'Treatment upgrade','rec13':'Treatment replacement (CI)','rec16':'Specialist consult',
      'rec17':'Other consult (hepato/cardio)','rec18':'Nutritional consult','rec19':'Lifestyle improvement',
      'rec21':'Curate medical record'}
CAT={'rec10':'Treatment','rec11':'Treatment','rec12':'Treatment','rec13':'Treatment',
     'rec16':'Consultation','rec17':'Consultation','rec18':'Lifestyle/Prevention','rec19':'Lifestyle/Prevention',
     'rec1':'Lab/Screening','rec2':'Lab/Screening','rec3':'Lab/Screening','rec4':'Lab/Screening',
     'rec5':'Imaging','rec6':'Imaging','rec8':'Lab/Screening','rec21':'Other'}

def _norm(it):
    if isinstance(it,(list,tuple)) and len(it)==2 and isinstance(it[0],(list,tuple)): return (int(it[0][0]),int(it[0][1]))
    if isinstance(it,(list,tuple)) and len(it)>=2: return (int(it[0]),int(it[1]))
def _load(p):
    p=Path(p); return [pr for pr in (_norm(x) for x in json.load(open(p))) if pr] if p.exists() else []
def choices(k):
    d=Path(RESULTS_DIR)/k; prs=[]
    for i in range(4): prs+=_load(d/f"{k}_train_iter_{i}_ranked.json")
    if k in LEGACY: prs+=_load(d/f"{k}_test_ranked.json")
    else:
        got=False
        for pt in 'AB':
            t=_load(d/f"{k}_test_part_{pt}_ranked.json"); prs+=t; got=got or bool(t)
        if not got: prs+=_load(d/f"{k}_test_ranked.json")
    return {tuple(sorted(p)):p[0] for p in prs}

coh=pd.read_excel(COHORT).set_index('patient_num')
CH={k:choices(k) for k in PANEL}
common=set.intersection(*[set(CH[k]) for k in PANEL]); common=[u for u in common if u[0] in coh.index and u[1] in coh.index]
matched=[u for u in common if abs(coh.loc[u[0],'age']-coh.loc[u[1],'age'])<=MATCH_AGE and abs(coh.loc[u[0],'risk']-coh.loc[u[1],'risk'])<=MATCH_RISK]

def diff_per_rater(k,rec):
    ch=CH[k]; vals=[]
    for u in matched:
        w=ch[u]; l=u[1] if w==u[0] else u[0]
        vals.append(float(coh.loc[w,rec]>=0.5)-float(coh.loc[l,rec]>=0.5))
    return np.mean(vals)

stats={}
for rec in RECS:
    h=np.array([diff_per_rater(k,rec) for k in HUMANS]); l=np.array([diff_per_rater(k,rec) for k in LLMS])
    stats[rec]=dict(h=h.mean(),he=h.std(ddof=1)/np.sqrt(len(h)),l=l.mean(),le=l.std(ddof=1)/np.sqrt(len(l)))

order=sorted(RECS,key=lambda r: stats[r]['h']-stats[r]['l'])  # LLM-favoured at bottom -> physician-favoured at top
y=np.arange(len(order)); bw=0.38
CMAP={'Treatment':'#b3261e','Imaging':'#8a5cf6','Lab/Screening':'#1f77b4','Consultation':'#0a7d6b',
      'Lifestyle/Prevention':'#2e8b3d','Other':'#888888'}

fig,ax=plt.subplots(figsize=(9.2,7.6))
ax.axvline(0,color='#444',lw=.8,zorder=1)
PHYS='#444444'; LLM='#d97706'
for i,rec in enumerate(order):
    s=stats[rec]
    ax.barh(y[i]+bw/2,s['h'],bw,xerr=s['he'],color=PHYS,zorder=3,error_kw=dict(lw=.8,ecolor='#999'))
    ax.barh(y[i]-bw/2,s['l'],bw,xerr=s['le'],color=LLM,zorder=3,error_kw=dict(lw=.8,ecolor='#d9a86a'))
ax.set_yticks(y)
ax.set_yticklabels([NAME[r] for r in order],fontsize=10)
for tick,rec in zip(ax.get_yticklabels(),order): tick.set_color(CMAP[CAT[rec]])
ax.set_xlabel('Prioritisation differential on age+risk-matched pairs\n(chosen minus not-chosen; positive = group picks the patient WHO HAS this recommendation)',fontsize=10)
ax.set_title('Effect of each recommendation on prioritisation, net of age\n(age+risk-matched common pairs, n=%d)'%len(matched),fontsize=12,fontweight='bold')
leg1=[Patch(fc=PHYS,label='Physicians (n=10)'),Patch(fc=LLM,label='LLMs (n=8)')]
leg2=[Patch(fc=CMAP[c],label=c) for c in ['Treatment','Imaging','Lab/Screening','Consultation','Lifestyle/Prevention']]
l1=ax.legend(handles=leg1,loc='lower right',fontsize=9,frameon=False,title='Rater group')
ax.add_artist(l1)
ax.legend(handles=leg2,loc='upper left',fontsize=8,frameon=False,title='Recommendation category')
ax.grid(axis='x',ls=':',alpha=.4); ax.set_axisbelow(True)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
plt.tight_layout()
import os; os.makedirs('results/figures',exist_ok=True)
for ext in ['png','pdf']:
    fig.savefig(f'results/figures/figR_recommendation_effect_agematched.{ext}',dpi=200,bbox_inches='tight')
print('matched pairs:',len(matched))
print('top physician-favoured:',[ (r,round(stats[r]['h']-stats[r]['l'],3)) for r in order[::-1][:3]])
print('top LLM-favoured     :',[ (r,round(stats[r]['h']-stats[r]['l'],3)) for r in order[:3]])
print('saved results/figures/figR_recommendation_effect_agematched.png/.pdf')
