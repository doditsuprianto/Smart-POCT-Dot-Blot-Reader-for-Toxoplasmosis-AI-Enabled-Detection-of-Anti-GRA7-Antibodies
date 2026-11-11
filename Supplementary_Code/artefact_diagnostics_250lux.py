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
RESULTS = Path(os.environ.get('DOTBLOT_RESULTS', HERE.parent/'Dot-Blot Model ML CNN'/'Results'))
# All outputs of these scripts are written here (never into Results/). Override with DOTBLOT_SUPP_OUT.
OUT = Path(os.environ.get('DOTBLOT_SUPP_OUT', HERE/'output')); OUT.mkdir(parents=True, exist_ok=True)
import pandas as pd, numpy as np, json, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
LUX=250
R=str(RESULTS/'tables')+'/'
T03=pd.read_csv(R+'T03_qc_flags.csv'); T04=pd.read_csv(R+'T04_features_raw.csv')
S={}
v=T03[T03.lux==LUX]   # all spots (pre-QC), as in the notebook's diagnostic table
for nm,d in [('ALL',v),('MIX',v[v.membrane_grp=='M23'])]:
    S[nm]=dict(n=len(d),AUC_crop=roc_auc_score(d.y,d.side_h),AUC_gray=roc_auc_score(d.dropna(subset=['g_mean']).y,d.dropna(subset=['g_mean']).g_mean),n_gray=int(d.g_mean.notna().sum()))
c=v.groupby(['membrane_grp','label']).side_h.mean(); S['side']={f'{a}_{b}':x for (a,b),x in c.items()}
S['gray_by_mem_lux']=T03.groupby(['lux','membrane']).g_mean.mean().unstack().round(2).to_dict()
# silhouette & PCA (COMBINED, valid crops)
D=T04.merge(T03[['image_id','membrane','membrane_grp','lux','y','label']],on='image_id')
F=[c for c in T04.columns if c.split('__')[0] in ('Q1','Q2','Q3','Q4','Q5','Q6')]
d=D[D.lux==LUX].reset_index(drop=True); X=d[F].replace([np.inf,-np.inf],np.nan); X=X.fillna(X.median())
Za=StandardScaler().fit_transform(X)
out=[]
for m,g in X.groupby(d.membrane):
    med=g.median(); s=((g-med).abs().median()*1.4826).replace(0,1); out.append(((g-med)/s))
Zm=StandardScaler().fit_transform(pd.concat(out).loc[X.index].clip(-10,10))
S['sil']={}
pcs={}
for k,Z in (('abs',Za),('mem',Zm)):
    S['sil'][k]=dict(membrane=silhouette_score(Z,d.membrane),cls=silhouette_score(Z,d.y))
    p=PCA(2,random_state=0); pcs[k]=(p.fit_transform(Z),p.explained_variance_ratio_)
S['pca_var']={k:[float(x) for x in pcs[k][1]] for k in pcs}
json.dump(S,open(OUT/'stats250.json','w'),indent=1,default=float); print(json.dumps(S,indent=1,default=float)[:1500])
# ---- Figure A: artefact diagnostics (3 panels)
sns.set_theme(style='whitegrid',font_scale=0.85)
fig,ax=plt.subplots(1,3,figsize=(15,4.4))
sns.boxplot(data=v,x='membrane',y='side_h',hue='label',hue_order=['Positive','Negative'],palette=['#4c72b0','#c44e52'],ax=ax[0])
ax[0].set_ylabel('Crop side (px)'); ax[0].set_xlabel('Membrane'); ax[0].set_title(f'(a) Crop size - {LUX} lux'); ax[0].legend(title='Class',loc='lower left')
sns.boxplot(data=v,x='membrane',y='g_mean',hue='label',hue_order=['Positive','Negative'],palette=['#4c72b0','#c44e52'],ax=ax[1])
ax[1].set_ylabel('Mean grey level'); ax[1].set_xlabel('Membrane'); ax[1].set_title(f'(b) Mean grey level - {LUX} lux'); ax[1].legend(title='Class',loc='lower left')
g=T03.groupby(['lux','membrane']).g_mean.mean().reset_index()
for l,dd in g.groupby('lux'): ax[2].plot(dd.membrane,dd.g_mean,marker='o',label=f'{l} lux')
ax[2].set_xticks([1,2,3]); ax[2].set_xlabel('Membrane'); ax[2].set_ylabel('Mean grey level'); ax[2].set_title('(c) Mean grey level per membrane (batch effect)'); ax[2].legend(fontsize=8)
sns.despine(); plt.tight_layout(); plt.savefig(OUT/'N05_artefact_250.png',dpi=200); plt.close()
# ---- Figure B: PCA
fig,ax=plt.subplots(2,2,figsize=(10,8))
mcol={1:'#e41a1c',2:'#377eb8',3:'#4daf4a'}; ccol={'Positive':'#3a6db5','Negative':'#c8455b'}
for i,k in enumerate(('abs','mem')):
    P,ev=pcs[k]
    for m in (1,2,3):
        s=(d.membrane==m).values; ax[i,0].scatter(P[s,0],P[s,1],s=14,c=mcol[m],label=f'M{m}',edgecolor='w',lw=.3)
    for cl in ('Positive','Negative'):
        s=(d.label==cl).values; ax[i,1].scatter(P[s,0],P[s,1],s=14,c=ccol[cl],label=cl,edgecolor='w',lw=.3)
    ax[i,0].set_title(f'PCA [{k}] - colour: membrane'); ax[i,1].set_title(f'PCA [{k}] - colour: class')
    for j in (0,1):
        ax[i,j].set_xlabel(f'PC1 ({ev[0]*100:.1f}% variance)'); ax[i,j].set_ylabel(f'PC2 ({ev[1]*100:.1f}% variance)'); ax[i,j].legend(fontsize=8,loc='best')
sns.despine(); plt.tight_layout(); plt.savefig(OUT/'N06_pca_250.png',dpi=200); plt.close()
