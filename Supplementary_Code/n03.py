# =============================================================================
# PoCT - Toxoplasmosis Diagnosis Device (Dec, 2026)
# Dodit Suprianto
# Email: Dodit.suprianto@polinema.ac.id
# Affiliation:
#   - Department of Electrical Engineering, Politeknik Negeri Malang, Malang, Indonesia
#   - AIDS, Toxoplasma, Opportunistic Disease and Malaria (ATOM) Research Group,
#     Faculty of Medicine, Universitas Brawijaya, Malang, Indonesia
# =============================================================================
import os
from pathlib import Path
HERE = Path(__file__).resolve().parent
# Folder with the published ML/CNN results (tables/, cache/). Override with the environment variable DOTBLOT_RESULTS.
RESULTS = Path(os.environ.get('DOTBLOT_RESULTS', HERE.parent/'Dot_Blot_Model_ML_CNN'/'Results'))
# All outputs of these scripts are written here (never into Results/). Override with DOTBLOT_SUPP_OUT.
OUT = Path(os.environ.get('DOTBLOT_SUPP_OUT', HERE/'output')); OUT.mkdir(parents=True, exist_ok=True)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, pickle, pandas as pd
import pandas as pd
T11 = pd.read_csv(RESULTS/'tables'/'T11_permutation.csv')               # label-permutation results of the notebook
MS  = pd.read_csv(OUT/'ALL_membrane_sensitivity.csv')                    # output of extra_membrane_sensitivity.py
plt.rcParams.update({'font.size':9})
SP=pickle.load(open(OUT/'sel_perm.pkl','rb')); nm=SP['nullmax']; r=SP['res']
fig,ax=plt.subplots(1,3,figsize=(12,3.7))
t=T11.set_index('Unnamed: 0')
for i,p in enumerate(['ALL','MIX']):
    ax[0].plot([i,i],[0.3,t.loc[p,'null_q95']],color='#9aa5b1',lw=10,solid_capstyle='butt',alpha=.6,label='permutation null (mean→95th pct)' if i==0 else None)
    ax[0].plot([i-.15,i+.15],[t.loc[p,'null_mean']]*2,color='#44546a',lw=2)
    ax[0].scatter([i],[t.loc[p,'obs']],color='#d1495b',s=60,zorder=5,label='observed' if i==0 else None); ax[0].text(i+.2,t.loc[p,'obs'],f"p = {t.loc[p,'p']:.3f}",va='center')
ax[0].axhline(.5,color='k',ls=':',lw=1); ax[0].set_xticks([0,1]); ax[0].set_xticklabels(['ALL (46 sera)','MIX (22 sera)']); ax[0].set_xlim(-.5,1.9); ax[0].set_ylim(.3,1); ax[0].set_ylabel('Serum-level AUC (LR, COMBINED)'); ax[0].set_title('(a) Permutation test'); ax[0].legend(fontsize=7,loc='lower right')
ax[1].hist(nm,bins=22,color='#9aa5b1'); ax[1].axvline(r['obs_max'],color='#d1495b',lw=2); ax[1].set_xlabel('Maximum AUC over 7 feature sets'); ax[1].set_ylabel('Permutations'); ax[1].set_title(f"(b) Selection-aware null (MIX)\nobs. max = {r['obs_max']:.3f}; p = {r['p_max']:.3f}")
sel=[('COMBINED_Q1-Q6','MLP','COMBINED+MLP'),('CTRL_Shape','ET','Shape+ET'),('Q4_Texture','ET','Q4 texture+ET'),('Q5_Histogram','kNN','Q5 hist.+kNN')]
ms=MS.set_index(['fset','model']); x=np.arange(len(sel)); w=.27
for k,(c,l,col) in enumerate([('M1pos_detect','M1 positives (24)','#d1495b'),('M23pos_detect','M2/M3 positives (8)','#f2a541'),('M23neg_spec','M2/M3 negatives (14)','#3b6fb6')]):
    ax[2].bar(x+(k-1)*w,[ms.loc[(a,b),c] for a,b,_ in sel],w,label=l,color=col)
ax[2].set_xticks(x); ax[2].set_xticklabels([s[2] for s in sel],rotation=20,fontsize=8); ax[2].set_ylabel('Proportion correct (thr. 0.5)'); ax[2].set_title('(c) ALL models by membrane group'); ax[2].legend(fontsize=7,loc='upper right',ncol=1); ax[2].set_ylim(0,1.25)
plt.tight_layout(); plt.savefig(OUT/'N03_perm_membrane.png',dpi=220)
