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
# NOTE: SVC(probability=True) is not seeded (Platt scaling uses an internal random split), so AUC values of the near-chance MIX protocol vary by about +/-0.04
# between runs; the ALL protocol reproduces exactly. The seed is left unchanged so that the published numbers keep their provenance.
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedGroupKFold, train_test_split
from sklearn.metrics import roc_auc_score
R=str(RESULTS/'tables')+'/'
T03=pd.read_csv(R+'T03_qc_flags.csv'); v=T03[(T03.valid)&(T03.lux==250)].copy()
v['circ']=4*np.pi*v.area/v.perimeter**2; v['eqd']=2*v.R; v['aspect']=1.0
feats=['circ','perimeter','area','eqd']
def cohen(a,b): return (a.mean()-b.mean())/np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
out=[]
for nm,d in [('ALL',v),('MIX',v[v.membrane_grp=='M23'])]:
    for f in ['area','perimeter','eqd','circ','solidity','extent']:
        a=d[d.y==1][f]; b=d[d.y==0][f]; p=stats.mannwhitneyu(a,b).pvalue
        out.append(dict(protocol=nm,feature=f,n_pos=len(a),n_neg=len(b),mean_pos=a.mean(),mean_neg=b.mean(),cohen_d=cohen(a,b),AUC=roc_auc_score(d.y,d[f]),p=p))
U=pd.DataFrame(out); print(U.round(4).to_string(index=False)); U.to_csv(OUT/'geo250_univariate.csv',index=False)
grid={'svc__C':[0.1,1,10,100],'svc__gamma':[0.001,0.01,0.1,1,'scale']}
res=[]
for nm,d in [('ALL',v),('MIX',v[v.membrane_grp=='M23'])]:
    X=d[feats].values; y=d.y.values; g=d.sample_id.values
    # old-style: stratified spot-level 70:30, 20 repeats
    aucs=[];accs=[]
    for s in range(20):
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.3,stratify=y,random_state=s)
        gs=GridSearchCV(make_pipeline(StandardScaler(),SVC(probability=True)),grid,cv=5,scoring='accuracy').fit(Xtr,ytr)
        p=gs.predict_proba(Xte)[:,1]; aucs.append(roc_auc_score(yte,p)); accs.append((gs.predict(Xte)==yte).mean())
    # group-wise 5-fold
    ga=[];gc=[]
    for s in range(5):
        for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=s).split(X,y,g):
            gs=GridSearchCV(make_pipeline(StandardScaler(),SVC(probability=True)),grid,cv=3,scoring='accuracy').fit(X[tr],y[tr])
            p=gs.predict_proba(X[te])[:,1]
            if len(np.unique(y[te]))>1: ga.append(roc_auc_score(y[te],p)); gc.append((gs.predict(X[te])==y[te]).mean())
    res.append(dict(protocol=nm,n_spots=len(d),spot_split_AUC_mean=np.mean(aucs),spot_split_AUC_sd=np.std(aucs),spot_split_ACC_mean=np.mean(accs),group_cv_AUC_mean=np.mean(ga),group_cv_AUC_sd=np.std(ga),group_cv_ACC_mean=np.mean(gc)))
    print(res[-1])
pd.DataFrame(res).to_csv(OUT/'geo250_svm.csv',index=False)
