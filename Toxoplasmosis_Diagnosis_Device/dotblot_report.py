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
dotblot_report.py
=================
Figures and PDF report of one diagnosis.

* ``fig_od_map`` / ``fig_radial_profile`` / ``fig_score_position`` build the three decision-support figures. The
  application draws them at the size of its canvases, the PDF report at print size.
* ``export_pdf`` writes the report (A4): captured spot, QC and geometry, the three figures, the descriptive
  result, a clinical interpretation and the cross-validated performance of the classifier.
"""
from __future__ import annotations

import datetime
import html
from typing import Dict, List, Optional

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from PyQt5 import QtCore, QtGui
from PyQt5.QtPrintSupport import QPrinter

import dotblot_features as dbf

C_NEG, C_POS = "#3b6fb6", "#d1495b"
LABEL_COLORS = {"POSITIVE": "#c0392b", "NEGATIVE": "#1e8449", "INDETERMINATE": "#d68910", "INVALID": "#7f8c8d"}


# =============================================================================
# Figures
# =============================================================================
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7)


def fig_od_map(an, w_px=400, h_px=320, dpi=100) -> Figure:
    """Figure 1: optical-density map of the inner ROI (the only region that is quantified)."""
    fig = Figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
    ax = fig.add_axes([0.05, 0.27, 0.9, 0.63])
    od = np.where(an.roi, an.od, np.nan)
    im = ax.imshow(od, cmap="magma", vmin=0, vmax=max(0.25, float(np.nanpercentile(od, 99))))
    g = an.geom
    ax.add_patch(Circle((g["cx"], g["cy"]), dbf.ROI_FRAC * g["R"], fill=False, ec="#00a0ff", lw=1.2))
    ax.axis("off")
    ax.set_title("Optical-density map of the inner ROI", fontsize=9)
    cax = fig.add_axes([0.2, 0.17, 0.6, 0.035])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=7)
    cb.set_label(f"OD  (mean {np.nanmean(od):.3f}, SD {np.nanstd(od):.3f})", fontsize=7)
    return fig


def fig_radial_profile(an, clf, w_px=400, h_px=320, dpi=100) -> Figure:
    """Figure 2: radial OD profile (centre -> edge) of this spot against the reference spots."""
    ref = clf.reference
    fig = Figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    edges = ref["ring_edges"]
    x = (edges[:-1] + edges[1:]) / 2
    for key, col, nm in (("neg", C_NEG, "negative"), ("pos", C_POS, "positive")):
        rr = ref["radial"][key]
        ax.fill_between(x, rr["q25"], rr["q75"], color=col, alpha=0.25, label=f"{nm}: median, IQR")
        ax.plot(x, rr["q50"], color=col, lw=1.2)
    ax.plot(x, an.radial_profile, "k-o", lw=2, ms=4, label="this spot")
    ax.set_xlabel("distance from centre (fraction of spot radius)", fontsize=8)
    ax.set_ylabel("mean optical density", fontsize=8)
    ax.set_title("Radial OD profile (centre → edge)", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.36)
    return fig


def fig_score_position(res, clf, w_px=400, h_px=320, dpi=100) -> Figure:
    """Figure 3: model score of this spot against the cross-validated scores of the validation spots."""
    ref = clf.reference
    score = res["score"]
    lo, hi = clf.band
    fig = Figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    bins = np.linspace(min(ref["oof_score"].min(), score) - 0.02, max(ref["oof_score"].max(), score) + 0.02, 22)
    neg = ref["oof_score"][ref["oof_y"] == 0]
    pos = ref["oof_score"][ref["oof_y"] == 1]
    ax.hist(neg, bins=bins, color=C_NEG, alpha=0.55, density=True, label=f"negative spots (n={len(neg)})")
    ax.hist(pos, bins=bins, color=C_POS, alpha=0.55, density=True, label=f"positive spots (n={len(pos)})")
    ax.axvspan(lo, hi, color="#f5b041", alpha=0.18, label="indeterminate zone")
    ax.axvline(score, color="k", lw=2.2, label=f"this spot ({score:.2f})")
    ax.set_xlabel("model score (cross-validated)", fontsize=8)
    ax.set_ylabel("density", fontsize=8)
    ax.set_title("Spot score vs. validation spots", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.38)
    return fig


def fig_to_qimage(fig: Figure) -> QtGui.QImage:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())
    h, w = buf.shape[:2]
    return QtGui.QImage(buf.data, w, h, 4 * w, QtGui.QImage.Format_RGBA8888).copy()


def bgr_to_qimage(bgr: np.ndarray) -> QtGui.QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888).copy()


# =============================================================================
# Descriptive text
# =============================================================================
def reference_range_note(an, clf) -> Optional[str]:
    """Warn when the spot is far outside the optical-density range of the reference (training) spots."""
    rad = clf.reference["radial"]
    ref_med = float(np.mean([np.mean(rad["neg"]["q50"]), np.mean(rad["pos"]["q50"])]))
    this = float(np.nanmean(an.radial_profile))
    ratio = this / max(ref_med, 1e-6)
    if ratio > 3 or ratio < 1 / 3:
        return (f"Mean optical density of this spot ({this:.3f}) is {ratio:.1f}× the median of the reference spots "
                f"({ref_med:.3f}): the spot lies far outside the range the classifier was trained on. "
                "Check illumination (≈250 lux), focus, exposure and the sample before relying on this result.")
    return None


def zone_sentence(res, clf) -> str:
    lo, hi = clf.band
    s = res["score"]
    if res["label"] == "POSITIVE":
        return f"The score {s:.2f} is at or above the POSITIVE threshold ({hi:.2f})."
    if res["label"] == "NEGATIVE":
        return f"The score {s:.2f} is at or below the NEGATIVE threshold ({lo:.2f})."
    return (f"The score {s:.2f} lies between the NEGATIVE ({lo:.2f}) and POSITIVE ({hi:.2f}) thresholds; "
            "the classifier cannot make a reliable call for this spot.")


def clinical_interpretation(label: str) -> List[str]:
    if label == "POSITIVE":
        return [
            "Anti-GRA7 reactivity was detected by the reader. In the intended use of the assay this suggests the presence of "
            "<i>Toxoplasma gondii</i>-specific reactivity in the tested serum.",
            "The dot-blot does not distinguish recent (acute) from past (chronic) infection.",
            "Recommended: confirm with the reference serology (IgM/IgG, IgG avidity) and correlate with clinical findings. In "
            "pregnant women, neonates and immunocompromised patients (e.g. HIV, transplantation) a positive result warrants "
            "prompt specialist assessment.",
        ]
    if label == "NEGATIVE":
        return [
            "No anti-GRA7 reactivity above the negative threshold was detected.",
            "A negative result does not exclude infection: very early infection (before seroconversion) and "
            "immunosuppression can give false-negative results.",
            "If clinical suspicion persists (e.g. pregnancy, lymphadenopathy, retinochoroiditis, immunosuppression), repeat the "
            "test after 2–3 weeks or use the reference serology.",
        ]
    return [
        "The result is inconclusive: the classifier could not assign the spot to the negative or the positive group with "
        "sufficient confidence.",
        "Re-capture the spot (whole spot in view, in focus, illumination ≈250 lux) and, if possible, test replicate spots of "
        "the same serum.",
        "Use the reference serology (IgM/IgG, IgG avidity) to establish the serostatus; do not base clinical decisions on "
        "an inconclusive result.",
    ]


# =============================================================================
# PDF report
# =============================================================================
def _e(x) -> str:
    return html.escape(str(x))


def build_html(d: Dict) -> str:
    res, an, clf, info = d["res"], d["an"], d["clf"], d.get("info") or {}
    lab = res["label"]
    col = LABEL_COLORS.get(lab, "#555")
    m = clf.metrics
    g = an.geom
    od = np.where(an.roi, an.od, np.nan)
    warn = reference_range_note(an, clf)
    lo_ci, hi_ci = m["spot_auc_ci"]
    ts = d["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
    sample_id = d.get("sample_id") or "—"

    def tr(k, v):
        return f"<tr><td style='color:#555;padding:2px 8px 2px 0;'>{_e(k)}</td><td style='padding:2px 0;'><b>{v}</b></td></tr>"

    h = [f"""
<html><body style="font-family:Arial; font-size:10pt; color:#222;">
<table width="100%" cellspacing="0" cellpadding="0"><tr>
<td><span style="font-size:17pt; font-weight:bold;">Toxoplasmosis Dot-Blot Diagnosis Report</span><br>
<span style="color:#555;">PoCT – Toxoplasmosis Diagnosis Device · anti-GRA7 dot-blot reader</span></td></tr></table>
<hr>
<table cellspacing="0" cellpadding="0" width="100%"><tr><td valign="top" width="50%"><table cellspacing="0">
{tr("Report date/time", _e(ts))}{tr("Sample ID", _e(sample_id))}{tr("Capture file", _e(info.get("file", "—")))}
</table></td><td valign="top"><table cellspacing="0">
{tr("Classifier", _e(clf.meta.get("features", "").split(" (")[0]) + " + MLP")}{tr("Illumination (model)", f"{_e(clf.meta.get('lux'))} lux")}
{tr("Classifier build", _e(clf.meta.get("built", "—")))}
</table></td></tr></table>

<table width="100%" cellspacing="0" cellpadding="8" style="margin-top:8px; border:2px solid {col};"><tr>
<td align="center" bgcolor="#fafafa"><span style="font-size:24pt; font-weight:bold; color:{col};">{_e(lab)}</span><br>
<span style="font-size:11pt;">{"Toxoplasmosis (anti-GRA7 reactivity)" if lab != "INDETERMINATE" else "inconclusive – no reliable call"}</span><br>
<span style="font-size:10pt; color:#444;">model score {res['score']:.3f} · calibrated probability ≈ {res['probability'] * 100:.0f} %</span></td></tr></table>
"""]
    if warn:
        h.append(f"<p style='color:#c0392b;'><b>Warning:</b> {_e(warn)}</p>")

    h.append("""<h3 style="margin-bottom:2px;">1. Captured spot and acquisition quality</h3>
<table width="100%" cellspacing="0" cellpadding="4"><tr>
<td width="38%" valign="top"><img src="crop.png" width="210"></td><td valign="top"><table cellspacing="0">""")
    h.append(tr("Acquisition QC", "passed" if an.ok else _e(an.reason)))
    h.append(tr("Detection confidence", f"{info.get('conf', float('nan')):.2f}"))
    h.append(tr("Equivalent spot radius", f"{g['R']:.1f} px"))
    h.append(tr("Mask area / solidity", f"{g['area']:.0f} px² / {g['solidity']:.3f}"))
    h.append(tr("Crop rescaling", f"× {info.get('scale', 1.0):.2f}" + (" (to training spot size)" if info.get("scale", 1.0) != 1.0 else " (none)")))
    h.append(tr("Inner ROI", f"{dbf.ROI_FRAC * 100:.0f} % of the equivalent radius"))
    h.append(tr("Mean OD in ROI (± SD)", f"{np.nanmean(od):.3f} ± {np.nanstd(od):.3f}"))
    h.append("</table></td></tr></table>")

    h.append("""<h3 style="margin-bottom:2px;">2. Quantitative decision-support figures</h3>
<table width="100%" cellspacing="0" cellpadding="2"><tr>
<td width="33%"><img src="fig1.png" width="205"></td><td width="33%"><img src="fig2.png" width="205"></td><td width="33%"><img src="fig3.png" width="205"></td></tr></table>
<p style="font-size:8.5pt; color:#555;"><b>Fig. 1</b> optical density inside the inner ROI (blue circle), the only region that is quantified; contour
size and shape are excluded to avoid pressing/cropping artefacts. <b>Fig. 2</b> mean OD in six concentric rings against the
median and interquartile range of the negative and positive reference spots. <b>Fig. 3</b> score of this spot against the
cross-validated scores of the validation spots; the orange band is the indeterminate zone.</p>""")

    h.append('<h3 style="margin-bottom:2px;">3. Descriptive result</h3><ul>')
    h.append(f"<li>{_e(zone_sentence(res, clf))}</li>")
    h.append(f"<li>Decision rule: NEGATIVE ≤ {clf.band[0]:.2f} &lt; INDETERMINATE &lt; {clf.band[1]:.2f} ≤ POSITIVE "
             f"(thresholds from cross-validated scores of validation spots; Youden threshold {clf.threshold:.2f}).</li>")
    rp = an.radial_profile
    h.append(f"<li>Radial OD profile (centre → edge): " + ", ".join(f"{v:.3f}" for v in rp) +
             f"; centre-to-edge difference {rp[0] - rp[-1]:+.3f}.</li>")
    h.append("</ul>")

    h.append('<h3 style="margin-bottom:2px;">4. Clinical interpretation</h3><ul>')
    for t in clinical_interpretation(lab):
        h.append(f"<li>{t}</li>")
    h.append("</ul>")

    perf_chance = lo_ci <= 0.5
    h.append(f"""<h3 style="margin-bottom:2px;">5. Classifier performance and limitations</h3>
<table cellspacing="0" cellpadding="3" border="1" style="border-collapse:collapse; border-color:#bbb;">
<tr bgcolor="#eeeeee"><td><b>Cross-validated metric (single spot)</b></td><td><b>Value</b></td></tr>
<tr><td>Spot-level AUC (95 % CI)</td><td>{m['spot_auc']:.2f} ({lo_ci:.2f}–{hi_ci:.2f})</td></tr>
<tr><td>Serum-level AUC, mean of replicate spots (95 % CI)</td><td>{m['sample_auc_mean_of_replicates']:.2f} ({m['sample_auc_ci'][0]:.2f}–{m['sample_auc_ci'][1]:.2f})</td></tr>
<tr><td>Label-permutation test</td><td>p = {m['permutation_p']:.2f}</td></tr>
<tr><td>Share of validation spots with a POSITIVE/NEGATIVE call</td><td>{m['coverage'] * 100:.0f} %</td></tr>
<tr><td>Training data</td><td>{m['n_spots']} spots, {m['n_sera']} sera ({m['n_pos_sera']} positive / {m['n_neg_sera']} negative), membranes 2+3, {clf.meta.get('lux')} lux</td></tr>
</table>""")
    if perf_chance:
        h.append("<p style='color:#c0392b;'><b>The 95 % confidence interval of the single-spot AUC includes 0.5: the performance of the "
                 "classifier on one spot is not distinguishable from chance.</b></p>")
    h.append("""<p style="font-size:9pt; color:#444;"><b>Research prototype.</b> This report was generated by experimental software
and is not a validated diagnostic device. It must not be used alone for clinical decisions; confirm with the reference assay and
clinical assessment.</p>
<hr><p style="font-size:8pt; color:#777;">PoCT – Toxoplasmosis Diagnosis Device (Dec, 2026) · Dodit Suprianto ·
Department of Electrical Engineering, Politeknik Negeri Malang; ATOM Research Group, Faculty of Medicine, Universitas Brawijaya, Malang, Indonesia.</p>
</body></html>""")
    return "".join(h)


def export_pdf(path: str, d: Dict) -> None:
    """Write the A4 PDF report. ``d`` keys: res, an, clf, crop_bgr, info, timestamp, sample_id (optional)."""
    res, an, clf = d["res"], d["an"], d["clf"]
    d = dict(d)
    d.setdefault("timestamp", datetime.datetime.now())
    doc = QtGui.QTextDocument()
    doc.addResource(QtGui.QTextDocument.ImageResource, QtCore.QUrl("crop.png"), bgr_to_qimage(d["crop_bgr"]))
    for name, fig in (("fig1.png", fig_od_map(an, 700, 560, 150)), ("fig2.png", fig_radial_profile(an, clf, 700, 560, 150)),
                      ("fig3.png", fig_score_position(res, clf, 700, 560, 150))):
        doc.addResource(QtGui.QTextDocument.ImageResource, QtCore.QUrl(name), fig_to_qimage(fig))
    doc.setHtml(build_html(d))
    printer = QPrinter(QPrinter.ScreenResolution)
    printer.setOutputFormat(QPrinter.PdfFormat)
    printer.setPageSize(QtGui.QPageSize(QtGui.QPageSize.A4))
    printer.setPageMargins(15, 14, 15, 14, QPrinter.Millimeter)
    printer.setOutputFileName(str(path))
    doc.setPageSize(QtCore.QSizeF(printer.pageRect().size()))
    doc.print_(printer)
