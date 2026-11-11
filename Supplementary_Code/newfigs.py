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
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.patches as mp, pandas as pd, numpy as np
R=str(RESULTS)+'/'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
# ---- Fig workflow
fig,ax=plt.subplots(figsize=(11,6.2)); ax.axis('off'); ax.set_xlim(0,100); ax.set_ylim(0,60)
def box(x,y,w,h,t,fc='#eef2f7',ec='#44546a',fs=8.5,b=False):
    ax.add_patch(mp.FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.3,rounding_size=1.2',fc=fc,ec=ec,lw=1.2))
    ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs,fontweight='bold' if b else 'normal',wrap=True)
def arr(x1,y1,x2,y2): ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='-|>',color='#44546a',lw=1.3))
box(1,44,19,13,'Grayscale spot crops\n46 sera · 272 spots\n5 illuminances\n1,360 images',fc='#dbe7f3',b=True)
box(24,44,17,13,'Quality control\n(acquisition failures only)\n1,296 valid images',fc='#dbe7f3')
box(45,50,20,8,'Protocol ALL\nM1+M2+M3 · 46 sera',fc='#fde9d9')
box(45,40,20,8,'Protocol MIX\nM2+M3 · 22 sera (confound-free)',fc='#fde9d9')
box(69,44,30,13,'Artefact / batch diagnostics\ncrop size, membrane, well position\nper-membrane robust normalisation',fc='#ffe9a8')
arr(20,50.5,24,50.5); arr(41,50.5,45,54); arr(41,50.5,45,44); arr(65,54,69,52); arr(65,44,69,48)
box(1,20,45,16,'PIPELINE A — quantification → ML\nQ1 densitometry · Q2 relative OD · Q3 radial profile\nQ4 texture · Q5 OD histogram · Q6 spectrum · COMBINED (79) · 5-lux fusion (395)\nLR · LDA · SVM · kNN · RF · ET · XGBoost · LightGBM · MLP',fc='#e2f0d9',fs=7.4)
box(54,20,45,16,'PIPELINE B — end-to-end CNN\n96×96 OD input masked to ROI, scaled by radius\nSimpleCNN · ResNet18 · MobileNetV3-S · EfficientNet-B0 · DenseNet121\nGrad-CAM rim analysis',fc='#e2f0d9',fs=7.8)
arr(30,43,24,36.5); arr(65,43,76,36.5)
box(1,2,97,13,'Common evaluation: sample-level stratified 5-fold CV (identical folds) · nested hyper-parameter tuning · sample-level AUC (primary)\ncluster bootstrap CI · paired ΔAUC · McNemar · label-permutation tests (incl. selection-aware) · dose–response · family ablation\nArtefact controls: CTRL_Shape (crop geometry) · CTRL_Batch (membrane dummies) · CTRL_MaskOnly CNN (silhouette) · CTRL_Position (well coordinates)',fc='#f2dcdb',fs=8.2)
arr(23,19.5,23,15.5); arr(76,19.5,76,15.5)
plt.tight_layout(); plt.savefig(OUT/'N01_workflow.png',dpi=220); plt.close()
# ---- Fig composition
T01=pd.read_csv(R+'tables/T01_image_index.csv'); T03=pd.read_csv(R+'tables/T03_qc_flags.csv')
fig,ax=plt.subplots(1,3,figsize=(11,3.4))
sp=T01.drop_duplicates('spot_id').groupby(['membrane','label']).size().unstack(fill_value=0)
sp.plot.bar(ax=ax[0],color={'Negative':'#3b6fb6','Positive':'#d1495b'},rot=0); ax[0].set_xticklabels(['M1','M2','M3']); ax[0].set_xlabel('Membrane'); ax[0].set_ylabel('Number of spots'); ax[0].set_title('(a) Spots per membrane and class'); ax[0].legend(title='')
se=T01.drop_duplicates('sample_id').assign(g=lambda d:d.membrane_grp.map({'M1':'M1','M23':'M2+M3'})).groupby(['g','clia_status']).size().unstack(fill_value=0)
se[['Positive','Intermediate','Negative']].plot.bar(stacked=True,ax=ax[1],color={'Positive':'#d1495b','Intermediate':'#f2a541','Negative':'#3b6fb6'},rot=0); ax[1].set_xlabel('Membrane group'); ax[1].set_ylabel('Number of sera'); ax[1].set_title('(b) Sera per membrane group'); ax[1].legend(title='CLIA',fontsize=8)
v=T03[T03.valid]; 
ig=T01.drop_duplicates('sample_id'); ig=ig[(ig.y==1)&(ig.clia_status=='Positive')]
ax[2].hist(ig.log10_IgG,bins=14,color='#d1495b',alpha=.85); ax[2].set_xlabel('log10 IgG (CLIA, units/mL; ">650" set to 650)'); ax[2].set_ylabel('Number of sera'); ax[2].set_title('(c) IgG titre of CLIA-positive sera')
plt.tight_layout(); plt.savefig(OUT/'N02_composition.png',dpi=220); plt.close()
