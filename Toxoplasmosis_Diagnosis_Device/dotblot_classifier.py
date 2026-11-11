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
dotblot_classifier.py
=====================
Deployable single-spot classifier for the anti-GRA7 dot-blot reader.

Model   : Q4 (rotation-invariant texture of the inner ROI) + MLP, 250 lux
Training: Membranes 2+3 only ("MIX": both classes on the same membrane, i.e. no
          batch confound), grouped by serum; see train_diagnosis_model.py.

The class is stored in a joblib bundle (model/dotblot_q4_mlp.joblib).  It must be
importable as ``dotblot_classifier.DiagnosisModel`` when the bundle is loaded.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CorrFilter(BaseEstimator, TransformerMixin):
    """Drops one of every pair of features with |r| > thr (and constant features)."""

    def __init__(self, thr=0.95):
        self.thr = thr

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        n = X.shape[1]
        keep = np.ones(n, bool)
        if X.shape[0] > 3 and n > 1:
            with np.errstate(all="ignore"):
                C = np.abs(np.nan_to_num(np.corrcoef(X, rowvar=False)))
            for i in range(n):
                if keep[i]:
                    keep[(C[i] > self.thr) & (np.arange(n) > i)] = False
        keep &= (X.std(0) > 1e-12)
        self.keep_ = keep if keep.any() else np.ones(n, bool)
        return self

    def transform(self, X):
        return np.asarray(X, float)[:, self.keep_]


class DiagnosisModel:
    """Ensemble of identical-architecture MLP pipelines + decision rule + reference data."""

    def __init__(self, feature_names, members, threshold, band, calib, reference, metrics, meta):
        self.feature_names = list(feature_names)
        self.members = members            # list of sklearn Pipelines (imputer -> CorrFilter -> scaler -> MLP)
        self.threshold = float(threshold)  # decision threshold on the raw score
        self.band = tuple(band)           # (low, high) raw-score band reported as INDETERMINATE
        self.calib = calib                # (a, b): p = sigmoid(a * logit(score) + b)
        self.reference = reference        # dict with OOF scores, ROC, radial-profile reference
        self.metrics = metrics            # cross-validated performance of the deployed pipeline
        self.meta = meta

    # -- scoring ------------------------------------------------------------
    def raw_score(self, feats: dict) -> float:
        x = np.array([[feats.get(k, np.nan) for k in self.feature_names]], float)
        return float(np.mean([m.predict_proba(x)[0, 1] for m in self.members]))

    def calibrated(self, score: float) -> float:
        a, b = self.calib
        z = np.log(np.clip(score, 1e-6, 1 - 1e-6) / (1 - np.clip(score, 1e-6, 1 - 1e-6)))
        return float(1 / (1 + np.exp(-(a * z + b))))

    def predict(self, feats: dict) -> dict:
        s = self.raw_score(feats)
        lo, hi = self.band
        label = "POSITIVE" if s >= hi else ("NEGATIVE" if s <= lo else "INDETERMINATE")
        return dict(score=s, probability=self.calibrated(s), label=label,
                    threshold=self.threshold, band=self.band)
