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
import pickle,pandas as pd,numpy as np
from scipy import stats
R=str(RESULTS)+'/'
o=pickle.load(open(R+'cache/oofs_ml.pkl','rb'))
def auc(y,s):
    y=np.asarray(y);s=np.asarray(s,float);n1=(y==1).sum();n0=len(y)-n1
    if n1==0 or n0==0:return np.nan
    r=stats.rankdata(s);return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
T01=pd.read_csv(R+'tables/T01_image_index.csv')[['image_id','membrane_grp']].drop_duplicates('image_id').set_index('image_id').membrane_grp
T09=pd.read_csv(R+'tables/T09_ml_benchmark_all.csv')
rows=[]
for (prot,fs,mod),df in o.items():
    if prot!='ALL': continue
    df=df.copy(); df['mg']=df.image_id.map(T01)
    cons=df.groupby('sample_id').agg(y=('y','first'),sc=('score','mean'),mg=('mg','first'))
    m23=cons[cons.mg=='M23']; m1=cons[cons.mg=='M1']
    rows.append(dict(fset=fs,model=mod,AUC_all=auc(cons.y,cons.sc),AUC_M23only=auc(m23.y,m23.sc),
       M1pos_detect=(m1.sc>=.5).mean(),M23pos_detect=(m23[m23.y==1].sc>=.5).mean(),M23neg_spec=(m23[m23.y==0].sc<.5).mean(),
       M1_mean_score=m1.sc.mean(),M23pos_mean=m23[m23.y==1].sc.mean(),M23neg_mean=m23[m23.y==0].sc.mean()))
r=pd.DataFrame(rows)
top=T09[T09.protocol=='ALL'].sort_values('sample_AUC',ascending=False).head(12)[['fset','model','sample_AUC']]
x=top.merge(r,on=['fset','model']); print(x.round(3).to_string())
r.to_csv(OUT/'ALL_membrane_sensitivity.csv',index=False)
print(r.AUC_M23only.describe())
