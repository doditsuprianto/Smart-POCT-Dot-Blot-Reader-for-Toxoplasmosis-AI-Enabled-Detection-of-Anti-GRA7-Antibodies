# =============================================================================
# PoCT - Toxoplasmosis Diagnosis Device (Dec, 2026)
# Dodit Suprianto
# Email: Dodit.suprianto@polinema.ac.id
# Affiliation:
#   - Department of Electrical Engineering, Politeknik Negeri Malang, Malang, Indonesia
#   - AIDS, Toxoplasma, Opportunistic Disease and Malaria (ATOM) Research Group,
#     Faculty of Medicine, Universitas Brawijaya, Malang, Indonesia
# =============================================================================
"""
train_diagnosis_model.py
========================
Builds the classifier used by the diagnosis application.

    python train_diagnosis_model.py          (defaults: ../Dot_Blot_Model_ML_CNN/Dataset -> model/dotblot_q4_mlp.joblib)

Design (chosen from the classification study, section "ML models"):
  * input      : grayscale crop of one spot at 250 lux (1 image = 1 decision)
  * features   : Q4 rotation-invariant texture of the inner ROI (26 features, shape-free)
  * model      : MLP (hyper-parameters tuned by serum-grouped CV), ensemble of seeds
  * training   : Membranes 2+3 ("MIX": both classes on the same membrane -> no batch confound)
  * validation : repeated serum-grouped 5-fold CV with nested tuning (no leakage),
                 label-permutation test and shape-only artefact control
  * decision   : dual threshold taken from out-of-fold scores (POSITIVE / NEGATIVE / INDETERMINATE)

NOTE: the model is trained on raw (absolute) optical-density features, because a single new
spot has no membrane to be normalised against.  The cross-validated performance of exactly this
deployed pipeline is stored in the bundle and shown by the application.
"""
import argparse, glob, json, math, os, re, sys, time, warnings, datetime, platform
import numpy as np, pandas as pd
from PIL import Image

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joblib, sklearn
import dotblot_features as F
from dotblot_classifier import CorrFilter, DiagnosisModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# (sample_no, code, CLIA status, membrane group, row block, column block) - sample map of the study.
# 'Intermediate' (S30) is treated as positive (conservative), as in the study.
SAMPLES = [
    (1,'N06','Positive','M1','A-B','1-2'),
    (2,'N09','Positive','M1','A-B','3-4'),
    (3,'41','Positive','M1','C-D','1-2'),
    (4,'N04','Positive','M1','C-D','3-4'),
    (5,'65','Positive','M1','A-B','5-6'),
    (6,'N27','Positive','M1','A-B','7-8'),
    (7,'43','Positive','M1','C-D','5-6'),
    (8,'N14','Positive','M1','C-D','7-8'),
    (9,'N13','Positive','M1','A-B','9-10'),
    (10,'N07','Positive','M1','A-B','11-12'),
    (11,'N28','Positive','M1','C-D','9-10'),
    (12,'44','Positive','M1','C-D','11-12'),
    (13,'37','Positive','M1','E-F','1-2'),
    (14,'N11','Positive','M1','E-F','3-4'),
    (15,'67','Positive','M1','G-H','1-2'),
    (16,'61','Positive','M1','G-H','3-4'),
    (17,'51','Positive','M1','E-F','5-6'),
    (18,'38','Positive','M1','E-F','7-8'),
    (19,'52','Positive','M1','G-H','5-6'),
    (20,'N24','Positive','M1','G-H','7-8'),
    (21,'45','Positive','M1','E-F','9-10'),
    (22,'N22','Positive','M1','E-F','11-12'),
    (23,'N10','Positive','M1','G-H','9-10'),
    (24,'N08','Positive','M1','G-H','11-12'),
    (25,'31','Positive','M23','A-B','1-2'),
    (26,'34','Positive','M23','A-B','3-4'),
    (27,'50','Positive','M23','C-D','1-2'),
    (28,'N19','Positive','M23','C-D','3-4'),
    (29,'58','Positive','M23','A-B','5-6'),
    (30,'N02','Intermediate','M23','A-B','7-8'),
    (31,'59','Positive','M23','C-D','5-6'),
    (32,'62','Positive','M23','C-D','7-8'),
    (33,'N20','Negative','M23','A-B','9-10'),
    (34,'30','Negative','M23','A-B','11-12'),
    (35,'54','Negative','M23','C-D','9-10'),
    (36,'47','Negative','M23','C-D','11-12'),
    (37,'49','Negative','M23','E-F','1-2'),
    (38,'42','Negative','M23','E-F','3-4'),
    (39,'53','Negative','M23','G-H','1-2'),
    (40,'N29','Negative','M23','G-H','3-4'),
    (41,'N18','Negative','M23','E-F','5-6'),
    (42,'N25','Negative','M23','E-F','7-8'),
    (43,'32','Negative','M23','G-H','5-6'),
    (44,'N01','Negative','M23','G-H','7-8'),
    (45,'N23','Negative','M23','E-F','9-10'),
    (46,'36','Negative','M23','E-F','11-12')
]
ROW_BLOCK = {"A": "A-B", "B": "A-B", "C": "C-D", "D": "C-D", "E": "E-F", "F": "E-F", "G": "G-H", "H": "G-H"}
COL_BLOCK = {c: f"{2 * ((c - 1) // 2) + 1}-{2 * ((c - 1) // 2) + 2}" for c in range(1, 13)}
PAT = re.compile(r"^(\d+)_([A-H])_(Positive|Negative)_(\d+)\.png$")
SEED = 42
GRID = [((32,), a) for a in (1e-3, 1e-2, 1e-1, 1.0)] + [((64, 32), a) for a in (1e-3, 1e-2, 1e-1, 1.0)]


def build_table(dataset, lux):
    meta = pd.DataFrame(SAMPLES, columns=["sample_no", "code", "status", "mgrp", "rb", "cb"])
    meta["sample_id"] = meta.sample_no.map(lambda n: f"S{n:02d}")
    meta["y"] = (meta.status != "Negative").astype(int)
    rows = []
    for d in sorted(glob.glob(os.path.join(dataset, "Grayscale_*"))):
        mem = int(re.search(r"(\d+)$", d).group(1))
        for f in sorted(os.listdir(d)):
            m = PAT.match(f)
            if not m or int(m[4]) != lux:
                continue
            col, row = int(m[1]), m[2]
            mg = "M1" if mem == 1 else "M23"
            s = meta[(meta.mgrp == mg) & (meta.rb == ROW_BLOCK[row]) & (meta.cb == COL_BLOCK[col])]
            assert len(s) == 1, f
            a = np.array(Image.open(os.path.join(d, f)).convert("L"))
            an = F.analyze_spot(a)
            g = an.geom
            rec = dict(file=f, membrane=mem, spot_id=f"M{mem}_{col}{row}", sample_id=s.sample_id.iloc[0],
                       y=int(s.y.iloc[0]), label_file=m[3], valid=an.ok, reason=an.reason)
            assert (m[3] == "Positive") == bool(rec["y"]), f"label mismatch {f}"
            if an.ok:
                rec.update(an.features)
                rec.update({"CS__area": g["area"], "CS__R": g["R"], "CS__perimeter": g["perimeter"],
                            "CS__circularity": 4 * math.pi * g["area"] / max(g["perimeter"] ** 2, 1e-6),
                            "CS__eccentricity": g["eccentricity"], "CS__solidity": g["solidity"],
                            "CS__extent": g["extent"], "CS__crop_side": float(max(a.shape)), "R_equiv": g["R"]})
            rows.append(rec)
    return pd.DataFrame(rows)


def make_pipe(hidden, alpha, seed):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("cf", CorrFilter(0.95)), ("sc", StandardScaler()),
                     ("clf", MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, max_iter=800, random_state=seed))])


def fold_assign(samp, rep, k=5):
    fm = {}
    for f, (_, te) in enumerate(StratifiedKFold(k, shuffle=True, random_state=SEED + rep).split(samp.index, samp.values)):
        for s in samp.index[te]:
            fm[s] = f
    return fm


def fast_auc(y, s):
    y = np.asarray(y)
    return roc_auc_score(y, s) if len(np.unique(y)) > 1 else np.nan


def ens_predict(X, y, Xte, hidden, alpha, n_seeds, seed0=0):
    return np.mean([make_pipe(hidden, alpha, seed0 + s).fit(X, y).predict_proba(Xte)[:, 1] for s in range(n_seeds)], axis=0)


def select_params(X, y, g, samp_y, d_sample, seeds=(0, 1, 2)):
    """Pick MLP hyper-parameters by serum-grouped CV AUC (used on the training part only)."""
    best, best_auc = GRID[0], -1
    for hidden, alpha in GRID:
        aucs = []
        for sd in seeds:
            oof = np.zeros(len(y))
            for tr, te in StratifiedGroupKFold(3, shuffle=True, random_state=sd).split(X, y, g):
                oof[te] = make_pipe(hidden, alpha, sd).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            aucs.append(fast_auc(y, oof))
        if np.mean(aucs) > best_auc:
            best, best_auc = (hidden, alpha), float(np.mean(aucs))
    return best


def cluster_boot_ci(y_s, s_s, n=2000, seed=0):
    """CI of the AUC by resampling sera (class-stratified); one value per serum."""
    rng = np.random.default_rng(seed)
    i1, i0 = np.where(y_s == 1)[0], np.where(y_s == 0)[0]
    out = []
    for _ in range(n):
        ii = np.r_[rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]
        out.append(fast_auc(y_s[ii], s_s[ii]))
    return np.nanpercentile(out, [2.5, 97.5])


def spot_ci(df, n=1000, seed=0):
    """Cluster bootstrap at the spot level: resample whole sera with all their spots."""
    rng = np.random.default_rng(seed)
    groups = {k: v for k, v in df.groupby("sample_id")}
    pos = [k for k, v in groups.items() if v.y.iloc[0] == 1]
    neg = [k for k, v in groups.items() if v.y.iloc[0] == 0]
    out = []
    for _ in range(n):
        keys = list(rng.choice(pos, len(pos))) + list(rng.choice(neg, len(neg)))
        b = pd.concat([groups[k] for k in keys])
        out.append(fast_auc(b.y.values, b.score.values))
    return np.nanpercentile(out, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--dataset", default=os.path.join(here, "..", "Dot_Blot_Model_ML_CNN", "Dataset"),
                    help="folder containing Grayscale_Test 1/2/3 (default: sibling folder of this repository)")
    ap.add_argument("--out", default=os.path.join(here, "model", "dotblot_q4_mlp.joblib"))
    ap.add_argument("--lux", type=int, default=250)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--members", type=int, default=20, help="MLP seeds in the final ensemble")
    a = ap.parse_args()
    t0 = time.time()

    tab = build_table(a.dataset, a.lux)
    print(f"images at {a.lux} lux: {len(tab)}; valid: {int(tab.valid.sum())}; invalid: {tab[~tab.valid].reason.value_counts().to_dict()}")
    d = tab[tab.valid & tab.membrane.isin([2, 3])].reset_index(drop=True)          # MIX training set
    cols = [c for c in d.columns if c.startswith("Q4__")]
    X, y, g = d[cols].values.astype(float), d.y.values, d.sample_id.values
    samp = d.groupby("sample_id").y.first()
    print(f"training set (membranes 2+3): {len(d)} spots, {len(samp)} sera ({int(samp.sum())} positive / {int((1 - samp).sum())} negative), {len(cols)} features")

    # ---------- nested, serum-grouped cross-validation of the deployed pipeline ----------
    oof_all = np.zeros((a.repeats, len(d)))
    chosen = []
    for rep in range(a.repeats):
        fold = d.sample_id.map(fold_assign(samp, rep)).values
        for k in range(5):
            tr, te = fold != k, fold == k
            h, al = select_params(X[tr], y[tr], g[tr], None, None, seeds=(rep,))
            chosen.append((h, al))
            oof_all[rep, te] = ens_predict(X[tr], y[tr], X[te], h, al, n_seeds=3, seed0=100 * rep)
        print(f"  CV repeat {rep + 1}/{a.repeats} done ({time.time() - t0:.0f}s)")
    d["score"] = oof_all.mean(0)
    spot_auc_rep = [fast_auc(y, oof_all[r]) for r in range(a.repeats)]
    sm = d.groupby("sample_id").agg(y=("y", "first"), score=("score", "mean"))
    sample_auc = fast_auc(sm.y.values, sm.score.values)
    spot_auc = float(np.mean(spot_auc_rep))
    ci_spot = spot_ci(d)
    ci_sample = cluster_boot_ci(sm.y.values, sm.score.values)
    print(f"spot-level AUC {spot_auc:.3f} (95% CI {ci_spot[0]:.3f}-{ci_spot[1]:.3f}); sample-level AUC (mean of replicate spots) {sample_auc:.3f} ({ci_sample[0]:.3f}-{ci_sample[1]:.3f})")

    # ---------- final hyper-parameters on all training sera ----------
    hidden, alpha = select_params(X, y, g, None, None, seeds=(0, 1, 2, 3, 4))
    print("final hyper-parameters:", hidden, alpha)

    # ---------- permutation test (serum-level label shuffling, fixed hyper-parameters) ----------
    def cv_auc(labels_by_sample, seed):
        yy = d.sample_id.map(labels_by_sample).values
        fold = d.sample_id.map(fold_assign(labels_by_sample, 0)).values
        oof = np.zeros(len(d))
        for k in range(5):
            tr, te = fold != k, fold == k
            oof[te] = ens_predict(X[tr], yy[tr], X[te], hidden, alpha, n_seeds=1, seed0=seed)
        return fast_auc(yy, oof)
    obs = cv_auc(samp, 0)
    rng = np.random.default_rng(SEED)
    perm = [cv_auc(pd.Series(rng.permutation(samp.values), index=samp.index), 0) for _ in range(a.n_perm)]
    p_perm = (1 + sum(p >= obs for p in perm)) / (1 + a.n_perm)
    print(f"permutation test: observed {obs:.3f}, null mean {np.mean(perm):.3f}, p = {p_perm:.3f}")

    # ---------- artefact control: shape-only logistic regression ----------
    cs = [c for c in d.columns if c.startswith("CS__")]
    Xc = d[cs].values.astype(float)
    oof_c = np.zeros(len(d))
    fold = d.sample_id.map(fold_assign(samp, 0)).values
    for k in range(5):
        tr, te = fold != k, fold == k
        pc = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                       ("clf", LogisticRegression(C=0.1, class_weight="balanced", max_iter=4000))]).fit(Xc[tr], y[tr])
        oof_c[te] = pc.predict_proba(Xc[te])[:, 1]
    shape_auc = fast_auc(y, oof_c)
    print(f"shape-only control (spot AUC): {shape_auc:.3f}")

    # ---------- decision rule from out-of-fold scores ----------
    fpr, tpr, thr = roc_curve(y, d.score.values)
    t_youden = float(thr[np.argmax(tpr - fpr)])
    hi = float(np.quantile(d.score[d.y == 0], 0.90))     # >= hi: <=10% of negative spots lie above (specificity 90%)
    lo = float(np.quantile(d.score[d.y == 1], 0.10))     # <= lo: <=10% of positive spots lie below (sensitivity 90%)
    if lo >= hi:                                          # no usable separation -> everything between is indeterminate
        lo, hi = min(lo, hi), max(lo, hi)
    yhat = np.where(d.score >= hi, 1, np.where(d.score <= lo, 0, -1))
    dec = yhat >= 0
    cov = float(dec.mean())
    sens_dec = float(((yhat == 1) & (y == 1)).sum() / max((y == 1).sum(), 1))
    spec_dec = float(((yhat == 0) & (y == 0)).sum() / max((y == 0).sum(), 1))
    ppv = float(((yhat == 1) & (y == 1)).sum() / max((yhat == 1).sum(), 1))
    npv = float(((yhat == 0) & (y == 0)).sum() / max((yhat == 0).sum(), 1))
    yb = (d.score.values >= t_youden).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yb, labels=[0, 1]).ravel()
    print(f"Youden threshold {t_youden:.3f}: sens {tp / (tp + fn):.2f} spec {tn / (tn + fp):.2f} | dual thresholds lo={lo:.3f} hi={hi:.3f}: decided {cov:.0%} of spots")

    # ---------- Platt calibration of the OOF score ----------
    z = np.log(np.clip(d.score.values, 1e-6, 1 - 1e-6) / (1 - np.clip(d.score.values, 1e-6, 1 - 1e-6)))
    pl = LogisticRegression(C=1e6).fit(z[:, None], y)
    calib = (float(pl.coef_[0, 0]), float(pl.intercept_[0]))

    # ---------- reference data for the decision-support figures ----------
    rings = [f"Q3__ring{i}" for i in range(1, 7)]
    rad = {}
    for cls, nm in ((0, "neg"), (1, "pos")):
        v = d.loc[d.y == cls, rings].values
        rad[nm] = dict(q25=np.nanpercentile(v, 25, 0), q50=np.nanpercentile(v, 50, 0), q75=np.nanpercentile(v, 75, 0), n=int(len(v)))
    reference = dict(oof_score=d.score.values.copy(), oof_y=y.copy(), oof_sample=g.copy(), roc=dict(fpr=fpr, tpr=tpr, thr=thr),
                     radial=rad, ring_edges=F.RAD_EDGES.copy(), sample_score=sm.score.values.copy(), sample_y=sm.y.values.copy())

    metrics = dict(spot_auc=spot_auc, spot_auc_ci=[float(ci_spot[0]), float(ci_spot[1])],
                   spot_auc_by_repeat=[float(x) for x in spot_auc_rep],
                   sample_auc_mean_of_replicates=float(sample_auc), sample_auc_ci=[float(ci_sample[0]), float(ci_sample[1])],
                   permutation_p=float(p_perm), permutation_null_mean=float(np.mean(perm)), permutation_observed=float(obs),
                   shape_only_spot_auc=float(shape_auc), youden_threshold=t_youden,
                   youden_sens=float(tp / (tp + fn)), youden_spec=float(tn / (tn + fp)),
                   dual_lo=lo, dual_hi=hi, coverage=cov, sens_decided=sens_dec, spec_decided=spec_dec, ppv=ppv, npv=npv,
                   n_spots=int(len(d)), n_sera=int(len(samp)), n_pos_sera=int(samp.sum()), n_neg_sera=int((1 - samp).sum()),
                   hyperparameters=dict(hidden=list(hidden), alpha=alpha),
                   chosen_in_nested_cv=[f"{h}|{al}" for h, al in chosen])

    # ---------- final ensemble on all training spots ----------
    members = [make_pipe(hidden, alpha, 1000 + s).fit(X, y) for s in range(a.members)]
    meta = dict(lux=a.lux, features="Q4 texture (abs OD, inner ROI 85%)", training="membranes 2+3 (MIX)",
                reference_radius_px=F.REFERENCE_RADIUS_PX, built=datetime.datetime.now().isoformat(timespec="seconds"),
                sklearn=sklearn.__version__, python=platform.python_version(), seed=SEED,
                note="Research prototype. Cross-validated performance is stored in 'metrics'.")
    model = DiagnosisModel(cols, members, t_youden, (lo, hi), calib, reference, metrics, meta)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    joblib.dump(model, a.out, compress=3)
    json.dump(metrics, open(os.path.splitext(a.out)[0] + "_metrics.json", "w"), indent=2)
    print(f"saved {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
