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
dotblot_features.py
===================
Artefact-free quantification of a single anti-GRA7 dot-blot spot.

The functions here are a line-by-line port of the feature code used in the
classification notebook ("Dot-Blot Model.ipynb", sections 4-6), so that the
application computes exactly the same features that the model was trained on.

Input  : a grayscale crop of ONE spot (uint8, white = 255 outside the circle),
         i.e. the output of the YOLO segmentation + circular auto-crop stage.
Output : QC verdict, geometry, inner-ROI optical-density (OD) map and the
         Q3 (radial profile) / Q4 (rotation-invariant texture) feature values.

Only the inner disk (85 % of the equivalent radius) is analysed, so that the
size/shape of the contour - a pressing/cropping artefact - never enters the
features.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from scipy import ndimage as ndi
from skimage import feature, measure, morphology

# --------------------------------------------------------------------------- #
# Constants (identical to the notebook configuration)                          #
# --------------------------------------------------------------------------- #
ROI_FRAC = 0.85          # inner-ROI radius as a fraction of the equivalent spot radius
MIN_AREA_PX = 2000       # minimum mask area for a valid crop
I0 = 255.0               # white reference for optical density
RAD_EDGES = np.linspace(0, ROI_FRAC, 7)   # 6 concentric rings (Q3)
REFERENCE_RADIUS_PX = 94.4                # mean equivalent radius of the training crops (250 lux)


# --------------------------------------------------------------------------- #
# Basic geometry / QC                                                          #
# --------------------------------------------------------------------------- #
def to_od(a: np.ndarray) -> np.ndarray:
    """Optical density (Beer-Lambert): OD = -log10(I / I0)."""
    return -np.log10(np.clip(a.astype(np.float32), 1, 255) / I0)


def spot_mask(a: np.ndarray) -> Optional[np.ndarray]:
    """Largest non-white connected component after closing + hole filling."""
    m = a < 255
    if m.sum() < 50:
        return None
    m = ndi.binary_closing(m, structure=morphology.disk(2))
    m = ndi.binary_fill_holes(m)
    lab, n = ndi.label(m)
    if n == 0:
        return None
    sz = ndi.sum(m, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sz)))


def geometry(m: np.ndarray) -> Dict[str, float]:
    area = int(m.sum())
    cy, cx = ndi.center_of_mass(m)
    R = math.sqrt(area / math.pi)
    rp = measure.regionprops(m.astype(np.uint8))[0]
    edge_touch = int(m[0].sum() + m[-1].sum() + m[:, 0].sum() + m[:, -1].sum())
    return dict(area=area, cy=cy, cx=cx, R=R, perimeter=rp.perimeter, solidity=rp.solidity,
                eccentricity=rp.eccentricity, extent=rp.extent, edge_touch=edge_touch)


def qc_reason(g: Dict[str, float]) -> str:
    """Acquisition QC only (never based on feature values)."""
    if g["area"] < MIN_AREA_PX:
        return "small_mask"
    if g["solidity"] < 0.90 or g["edge_touch"] > 0.25 * 2 * math.pi * g["R"]:
        return "truncated"
    return "ok"


def roi_masks(m: np.ndarray, g: Dict[str, float]):
    yy, xx = np.ogrid[:m.shape[0], :m.shape[1]]
    d = np.hypot(yy - g["cy"], xx - g["cx"]) / g["R"]      # distance normalised by radius
    roi = m & (d <= ROI_FRAC)
    return roi, d


# --------------------------------------------------------------------------- #
# Q3 - radial OD profile                                                       #
# --------------------------------------------------------------------------- #
def radial_profile(od: np.ndarray, m: np.ndarray, d: np.ndarray) -> np.ndarray:
    prof = []
    for i in range(6):
        sel = m & (d >= RAD_EDGES[i]) & (d < RAD_EDGES[i + 1])
        prof.append(od[sel].mean() if sel.any() else np.nan)
    return np.array(prof, dtype=float)


def q3_radial(od, m, d):
    prof = radial_profile(od, m, d)
    x = np.arange(6)
    ok = np.isfinite(prof)
    slope = np.polyfit(x[ok], prof[ok], 1)[0] if ok.sum() > 2 else np.nan
    curv = np.polyfit(x[ok], prof[ok], 2)[0] if ok.sum() > 3 else np.nan
    out = {f"Q3__ring{i + 1}": prof[i] for i in range(6)}
    out.update({"Q3__slope": slope, "Q3__curv": curv, "Q3__center_to_edge": prof[0] - prof[-1]})
    return out


# --------------------------------------------------------------------------- #
# Q4 - rotation-invariant texture                                              #
# --------------------------------------------------------------------------- #
def _glcm_feats(img_q, valid, levels, tag):
    # img_q: int (0..levels-1); outside the ROI = levels (extra level, discarded)
    q = np.where(valid, img_q, levels).astype(np.uint8)
    out = {}
    for dist in (1, 2):
        G = feature.graycomatrix(q, [dist], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                 levels=levels + 1, symmetric=True, normed=False)[:levels, :levels, 0, :]
        Gs = G.astype(np.float64)
        Gs = Gs / np.maximum(Gs.sum(axis=(0, 1), keepdims=True), 1)
        i, j = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")
        con = (Gs * (i - j)[:, :, None] ** 2).sum((0, 1))
        dis = (Gs * np.abs(i - j)[:, :, None]).sum((0, 1))
        hom = (Gs / (1 + (i - j)[:, :, None] ** 2)).sum((0, 1))
        asm = (Gs ** 2).sum((0, 1))
        mu_i = (Gs * i[:, :, None]).sum((0, 1))
        mu_j = (Gs * j[:, :, None]).sum((0, 1))
        si = np.sqrt((Gs * (i[:, :, None] - mu_i) ** 2).sum((0, 1)))
        sj = np.sqrt((Gs * (j[:, :, None] - mu_j) ** 2).sum((0, 1)))
        cor = (Gs * (i[:, :, None] - mu_i) * (j[:, :, None] - mu_j)).sum((0, 1)) / np.maximum(si * sj, 1e-9)
        ent = -(Gs * np.log2(np.where(Gs > 0, Gs, 1))).sum((0, 1))
        for nm, val in dict(contrast=con, dissim=dis, homog=hom, asm=asm, corr=cor, entropy=ent).items():
            out[f"{tag}__glcm_{nm}_d{dist}"] = float(np.mean(val))     # mean of 4 directions -> rotation invariant
    return out


def q4_texture(od, roi):
    ys, xs = np.nonzero(roi)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = od[y0:y1, x0:x1]
    r = roi[y0:y1, x0:x1]
    v = sub[r]
    lo, hi = np.percentile(v, 1), np.percentile(v, 99)
    q = np.clip((sub - lo) / max(hi - lo, 1e-6), 0, 1)
    q = np.minimum((q * 32).astype(int), 31)                       # per-image normalisation: pure texture
    out = _glcm_feats(q, r, 32, "Q4")
    filled = np.where(r, sub, v.mean())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lbp = feature.local_binary_pattern(filled, 8, 1, method="uniform")
    inner = ndi.binary_erosion(r, iterations=2)
    hist = np.bincount(lbp[inner].astype(int), minlength=10)[:10].astype(float)
    hist /= max(hist.sum(), 1)
    out.update({f"Q4__lbp{k}": hist[k] for k in range(10)})
    sig = [1, 2, 4, 8]
    G = [ndi.gaussian_filter(filled, s) for s in sig]
    for k in range(3):
        out[f"Q4__dog{k + 1}_energy"] = float((G[k] - G[k + 1])[inner].std())
    out["Q4__lowfreq_std"] = float(G[3][inner].std())
    return out


Q4_FEATURES = None   # filled lazily; the order of keys of q4_texture() defines the column order


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class SpotAnalysis:
    ok: bool
    reason: str                                   # 'ok' | 'blank' | 'small_mask' | 'truncated'
    gray: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    roi: Optional[np.ndarray] = None
    od: Optional[np.ndarray] = None
    geom: Dict[str, float] = field(default_factory=dict)
    features: Dict[str, float] = field(default_factory=dict)
    radial_profile: Optional[np.ndarray] = None


def analyze_spot(gray: np.ndarray, compute_features: bool = True) -> SpotAnalysis:
    """Run QC and (if valid) feature extraction on one grayscale spot crop."""
    if gray.ndim == 3:
        raise ValueError("analyze_spot expects a single-channel (grayscale) uint8 image")
    gray = np.ascontiguousarray(gray, dtype=np.uint8)
    m = spot_mask(gray)
    if m is None:
        return SpotAnalysis(False, "blank", gray=gray)
    g = geometry(m)
    reason = qc_reason(g)
    roi, d = roi_masks(m, g)
    od = to_od(gray)
    res = SpotAnalysis(reason == "ok", reason, gray=gray, mask=m, roi=roi, od=od, geom=g)
    if reason != "ok" or not compute_features:
        return res
    f = {}
    f.update(q3_radial(od, m, d))
    f.update(q4_texture(od, roi))
    res.features = f
    res.radial_profile = radial_profile(od, m, d)
    return res
