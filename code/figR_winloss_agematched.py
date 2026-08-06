# -*- coding: utf-8 -*-
"""
Figure: on the 61 age+risk-matched common pairs, for each recommendation,
how often each group CHOSE the patient who has it (win, green) vs the patient
who does not (loss, red) -- counted only on pairs that are DISCORDANT for that
recommendation (one patient has it, the other doesn't; the only informative
ones). The number of discordant matched-pairs (k) is shown for each row, so it
is obvious which recommendations have enough data to be comparable.
Decisions are counted at the rater x pair level (10 physicians, 8 LLMs).
"""
import json, numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from panel_config import HUMANS, LLMS, LEGACYSET as LEGACY, PANEL
RECS=['rec1','rec2','rec3','rec4','rec5','rec6','rec8','rec10','rec11','rec12','rec13','rec16','rec17','rec18','rec19','rec21']
NAME={'rec1':'Basic lab panel','rec2':'Advanced lab panel','rec3':'Pathophysiology labs','rec4':'Routine lab monitoring',
      'rec5':'Diagnostic imaging','rec6':'Advanced imaging (CTA)','rec8':'Take BP/BMI measurement','rec10':'Initiate first-line statin',
      'rec11':'Initiate advanced tx','rec12':'Treatment upgrade','rec13':'Treatment replacement (CI)','rec16':'Specialist consult',
      'rec17':'Other consult (hepato/cardio)','rec18':'Nutritional consult','rec19':'Lifestyle improvement','rec21':'Curate medical record'}

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
coh=pd.read_excel('synthetic_data (7).xlsx').set_index('patient_num')
C={k:ch(k) for k in PANEL}
common=set.intersection(*[set(C[k]) for k in PANEL]); common=[u for u in common if u[0] in coh.index and u[1] in coh.index]
matched=[u for u in common if abs(coh.loc[u[0],'age']-coh.loc[u[1],'age'])<=5 and abs(coh.loc[u[0],'risk']-coh.loc[u[1],'risk'])<=5]

def has(pid,r): return coh.loc[pid,r]>=0.5
def winloss(raters,r):
    disc=[u for u in matched if has(u[0],r)!=has(u[1],r)]
    wins=losses=0
    for k in raters:
        cc=C[k]
        for u in disc:
            chosen=cc[u]; other=u[1] if chosen==u[0] else u[0]
            if has(chosen,r) and not has(other,r): wins+=1
            else: losses+=1
    return wins,losses,len(disc)

rows=[]
for r in RECS:
    hw,hl,k=winloss(HUMANS,r); lw,ll,_=winloss(LLMS,r)
    rows.append((r,k,hw,hl,lw,ll))
rows.sort(key=lambda x:x[1])               # fewest discordant pairs at bottom
recs=[x[0] for x in rows]; ks=[x[1] for x in rows]; y=np.arange(len(recs))

fig,(axH,axL)=plt.subplots(1,2,figsize=(13.5,7.6),sharey=True)
WIN='#2e8b3d'; LOSS='#c0392b'
for ax,(wi,li,title,col) in zip([axH,axL],[(2,3,'Physicians (n=10)',axH),(4,5,'LLMs (n=8)',axL)]):
    for i,row in enumerate(rows):
        w=row[wi]; l=row[li]
        ax.barh(y[i], w, color=WIN, zorder=3)              # wins to the right
        ax.barh(y[i], -l, color=LOSS, zorder=3)            # losses to the left
        tot=w+l
        if tot>0:
            ax.text(w+0.5, y[i], f'{w}', va='center', ha='left', fontsize=8, color=WIN)
            ax.text(-l-0.5, y[i], f'{l}', va='center', ha='right', fontsize=8, color=LOSS)
    ax.axvline(0,color='#333',lw=.9)
    ax.set_title(title,fontsize=12,fontweight='bold')
    ax.set_xlabel('\u2190 chose patient WITHOUT it   (decisions)   chose patient WITH it \u2192',fontsize=9)
    for s in ['top','right','left']: ax.spines[s].set_visible(False)
    ax.grid(axis='x',ls=':',alpha=.35); ax.set_axisbelow(True)
axH.set_yticks(y)
axH.set_yticklabels([f'{NAME[r]}   (k={k})' for r,k in zip(recs,ks)],fontsize=9.5)
# grey-out labels for low-information recs
for t,k in zip(axH.get_yticklabels(),ks):
    if k<10: t.set_color('#aaaaaa')
m=max(max(r[2]+r[3],r[4]+r[5]) for r in rows)
axH.set_xlim(-m*1.15,m*1.15); axL.set_xlim(-m*1.15,m*1.15)
fig.suptitle('Wins vs losses per recommendation on age+risk-matched pairs (k = discordant matched-pairs)\n'
             'grey rows have <10 discordant matched-pairs \u2014 too few to compare',
             fontsize=12.5,fontweight='bold',y=1.02)
plt.tight_layout()
import os; os.makedirs('results/figures',exist_ok=True)
for ext in ['png','pdf']: fig.savefig(f'results/figures/figR_winloss_agematched.{ext}',dpi=200,bbox_inches='tight')
print('rec   k  | phys W/L | llm W/L')
for r,k,hw,hl,lw,ll in sorted(rows,key=lambda x:-x[1]):
    print(f'{r:6s} {k:2d} |  {hw:3d}/{hl:<3d} |  {lw:3d}/{ll:<3d}')
print('saved results/figures/figR_winloss_agematched.png/.pdf')
