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
import pandas as pd, numpy as np, pickle, warnings, os
warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
R=str(RESULTS)+'/'
T01=pd.read_csv(R+'tables/T01_image_index.csv'); T04=pd.read_csv(R+'tables/T04_features_raw.csv')
d=T01.merge(T04,on='image_id'); d['mem']=d.membrane_grp
d=d[d.lux==250].copy()
fams={q:[c for c in T04.columns if c.startswith(q+'__')] for q in ['Q1','Q2','Q3','Q4','Q5','Q6']}
fams['COMB']=sum(fams.values(),[])
# robust z per membrane x lux
def rz(df,cols):
    o=df.copy()
    for _,idx in df.groupby(['membrane']).groups.items():
        s=df.loc[idx,cols]; m=s.median(); mad=(s-m).abs().median()*1.4826
        o.loc[idx,cols]=((s-m)/mad.replace(0,np.nan))
    o[cols]=o[cols].replace([np.inf,-np.inf],np.nan).clip(-8,8); return o
allc=fams['COMB']; d=rz(d,allc)
def auc(y,s):
    y=np.asarray(y);s=np.asarray(s,float);n1=(y==1).sum();n0=len(y)-n1
    r=stats.rankdata(s);return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
def corrfilt(X,thr=.95):
    C=np.abs(np.nan_to_num(np.corrcoef(X,rowvar=False)));n=X.shape[1];keep=np.ones(n,bool)
    for i in range(n):
        if keep[i]: keep[(C[i]>thr)&(np.arange(n)>i)]=False
    return keep&(X.std(0)>1e-12)
def cv_scores(df,cols,ymap,reps=3,seed=42,C=0.1):
    sids=np.array(sorted(df.sample_id.unique())); ys=np.array([ymap[s] for s in sids]); X=df[cols].values.astype(float); sid=df.sample_id.values
    yy=np.array([ymap[s] for s in sid]); agg=[]
    for rep in range(reps):
        sc=np.full(len(df),np.nan)
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed+rep).split(sids,ys):
            mtr=np.isin(sid,sids[tr]); mte=np.isin(sid,sids[te])
            imp=SimpleImputer(strategy='median').fit(X[mtr]); Xt=imp.transform(X[mtr]); k=corrfilt(Xt)
            ss=StandardScaler().fit(Xt[:,k]); m=LogisticRegression(C=C,class_weight='balanced',max_iter=2000).fit(ss.transform(Xt[:,k]),yy[mtr])
            sc[mte]=m.predict_proba(ss.transform(imp.transform(X[mte])[:,k]))[:,1]
        agg.append(sc)
    s=pd.DataFrame({'sid':sid,'sc':np.mean(agg,0)}).groupby('sid').sc.mean()
    return auc([ymap[i] for i in s.index],s.values)
# MIX
mix=d[d.mem=='M23'].copy(); ymap=mix.groupby('sample_id').y.first().to_dict()
sets={k:v for k,v in fams.items()}
sets['CTRL_Shape']=['CS__area','CS__R','CS__perimeter','CS__circularity','CS__eccentricity','CS__solidity','CS__extent','CS__crop_side']
# position control
mix['colnum']=mix.col.astype(float); mix['rownum']=mix.row.map(lambda r:'ABCDEFGH'.index(r)).astype(float); mix['blockc']=mix.col.astype(float)
sets['CTRL_Position']=['colnum','rownum']
mix=mix.dropna(subset=['CS__area']) if False else mix
obs={k:cv_scores(mix,v,ymap) for k,v in sets.items()}
print('observed MIX LR AUC:',{k:round(v,3) for k,v in obs.items()})
sel=[k for k in sets if not k.startswith('CTRL')]
obs_max=max(obs[k] for k in sel); print('observed max (Q1-Q6,COMB):',obs_max)
rng=np.random.default_rng(7); sids=np.array(sorted(ymap)); yv=np.array([ymap[s] for s in sids]); NP=int(os.environ.get('NP',300))
nullmax=[];nullcomb=[];nullpos=[]
for p in range(NP):
    yp=rng.permutation(yv); pm=dict(zip(sids,yp)); mix2=mix.copy(); mix2['y']=mix2.sample_id.map(pm)
    a={k:cv_scores(mix2,sets[k],pm,reps=1,seed=1000+p) for k in sel+['CTRL_Position','CTRL_Shape']}
    nullmax.append(max(a[k] for k in sel)); nullcomb.append(a['COMB']); nullpos.append(a['CTRL_Position'])
nullmax=np.array(nullmax)
res=dict(obs=obs,obs_max=obs_max,p_max=(1+(nullmax>=obs_max).sum())/(NP+1),null_max_mean=nullmax.mean(),null_max_q95=np.quantile(nullmax,.95),
 p_comb=(1+(np.array(nullcomb)>=obs['COMB']).sum())/(NP+1),NP=NP)
print(res); pickle.dump(dict(res=res,nullmax=nullmax),open(OUT/'sel_perm.pkl','wb'))
