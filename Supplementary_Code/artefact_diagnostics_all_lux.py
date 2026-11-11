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
import pandas as pd, numpy as np, json, warnings; warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler
R=str(RESULTS/'tables')+'/'
T03=pd.read_csv(R+'T03_qc_flags.csv'); T04=pd.read_csv(R+'T04_features_raw.csv')
D=T04.merge(T03[['image_id','membrane','lux','y']],on='image_id')
F=[c for c in T04.columns if c.split('__')[0] in ('Q1','Q2','Q3','Q4','Q5','Q6')]
rows=[]
for lux,v in T03.groupby('lux'):
    r={'lux':lux}
    for nm,d in (('ALL',v),('MIX',v[v.membrane_grp=='M23'])):
        g=d.dropna(subset=['g_mean']); r[f'crop_{nm}']=roc_auc_score(d.y,d.side_h); r[f'gray_{nm}']=roc_auc_score(g.y,g.g_mean)
    c=v.groupby(['membrane_grp','label']).side_h.mean(); r['M1pos']=c[('M1','Positive')]; r['M23pos']=c[('M23','Positive')]; r['M23neg']=c[('M23','Negative')]
    gm=v.groupby('membrane').g_mean.mean(); r['g1'],r['g2'],r['g3']=gm[1],gm[2],gm[3]
    d=D[D.lux==lux].reset_index(drop=True); X=d[F].replace([np.inf,-np.inf],np.nan); X=X.fillna(X.median())
    Za=StandardScaler().fit_transform(X); out=[]
    for m,g in X.groupby(d.membrane):
        med=g.median(); s=((g-med).abs().median()*1.4826).replace(0,1); out.append((g-med)/s)
    Zm=StandardScaler().fit_transform(pd.concat(out).loc[X.index].clip(-10,10))
    r['sil_abs_mem']=silhouette_score(Za,d.membrane); r['sil_abs_cls']=silhouette_score(Za,d.y)
    r['sil_nrm_mem']=silhouette_score(Zm,d.membrane); r['sil_nrm_cls']=silhouette_score(Zm,d.y)
    rows.append(r)
M=pd.DataFrame(rows); M.to_csv(OUT/'multi_lux.csv',index=False); print(M.round(3).to_string())
