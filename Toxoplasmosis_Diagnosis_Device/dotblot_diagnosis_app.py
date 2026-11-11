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
dotblot_diagnosis_app.py
========================
Desktop application of the anti-GRA7 dot-blot reader for *Toxoplasma gondii* serodiagnosis.

Tabs (defined in gui.py):
  1. Diagnosis   - ONE dot-blot spot -> POSITIVE / NEGATIVE / INDETERMINATE (+ decision-support figures)
  2. Labelling   - builds the labelled crop dataset (file name = column_row_label_lux.png)
  3. Raw capture - stores raw photographs for segmentation learning

Diagnosis workflow
  live camera -> YOLOv8-seg spot detection -> [Capture] circular auto-crop of the most confident spot
  -> [Diagnosis] acquisition QC -> inner-ROI optical density -> Q4 texture features -> MLP ensemble
  -> decision + three figures (score among validation spots, radial OD profile, cross-validated ROC).

Run:    python dotblot_diagnosis_app.py                 (camera)
        python dotblot_diagnosis_app.py --image x.png   (still image instead of the camera, for demos/tests)

Files needed next to this script: gui.py, model/best.pt (YOLO), model/dotblot_q4_mlp.joblib (classifier,
created by train_diagnosis_model.py), dotblot_features.py, dotblot_classifier.py.
"""
import argparse
import csv
import datetime
import os
import re
import sys
import uuid
from pathlib import Path

import cv2
import joblib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QGraphicsScene, QMessageBox
from ultralytics import YOLO

import time

import config as cfg
import dotblot_features as dbf
import dotblot_classifier  # noqa: F401  (needed to unpickle the classifier bundle)
import dotblot_report as rep
from gui import Ui_FormDiagnosisToxoplasmosis

YOLO_PATH = cfg.YOLO_WEIGHTS
CLASSIFIER_PATH = cfg.CLASSIFIER_PATH
DIAG_DIR = cfg.DIAGNOSIS_DIR                      # crops + diagnosis_log.csv of the Diagnosis tab
LABEL_DIR = cfg.LABELLING_DIR                     # Labelling tab output
RAW_DIR = cfg.RAW_CAPTURE_DIR                     # Raw-capture tab output
CROP_PADDING = cfg.CROP_PADDING
SCALE_TOLERANCE = cfg.SCALE_TOLERANCE

COLORS = {"POSITIVE": "#c0392b", "NEGATIVE": "#1e8449", "INDETERMINATE": "#d68910", "INVALID": "#7f8c8d"}
C_NEG, C_POS = "#3b6fb6", "#d1495b"


# =============================================================================
# Helpers
# =============================================================================
def dbg(msg):
    """Console diagnostics, enabled with DOTBLOT_DEBUG=1 (or cfg.DEBUG = True)."""
    if cfg.DEBUG:
        print(f"[debug] {msg}", flush=True)


def cvimg_to_qpixmap(bgr_img: np.ndarray) -> QtGui.QPixmap:
    if bgr_img is None or bgr_img.size == 0:
        return QtGui.QPixmap()
    if bgr_img.ndim == 2:
        bgr_img = cv2.cvtColor(bgr_img, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg.copy())


def fig_to_pixmap(fig: Figure) -> QtGui.QPixmap:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())
    h, w = buf.shape[:2]
    qimg = QtGui.QImage(buf.data, w, h, 4 * w, QtGui.QImage.Format_RGBA8888)
    return QtGui.QPixmap.fromImage(qimg.copy())


def show_pixmap_in_graphicsview(view: QtWidgets.QGraphicsView, pixmap: QtGui.QPixmap):
    scene = QGraphicsScene(view)
    scene.addPixmap(pixmap)
    view.setScene(scene)
    view.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
    view.fitInView(scene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)
    view.viewport().update()


def sanitize_filename_part(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "NA"
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r'[\\/:*?"<>|]+', "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s if s else "NA"


def detections_from_result(r):
    """Return (circles, conf) from a YOLO result; circle = (cx, cy, radius) taken from the bounding box."""
    if r.boxes is None or len(r.boxes) == 0:
        return None, None
    boxes = r.boxes.xyxy.cpu().numpy()
    try:
        conf = r.boxes.conf.cpu().numpy()
    except Exception:
        conf = np.ones((boxes.shape[0],), dtype=np.float32)
    circles = []
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        circles.append((int((x1 + x2) / 2), int((y1 + y2) / 2), int(min(x2 - x1, y2 - y1) / 2)))
    return circles, conf


def crop_circle(frame: np.ndarray, circle, scale: float = 1.0):
    """Circular auto-crop with white background (same geometry as the training crops).

    If ``scale`` != 1 the square region is resized first and the circular mask is applied afterwards, so
    that no interpolation artefact appears at the spot edge.
    """
    h_img, w_img = frame.shape[:2]
    cx, cy, radius = [int(v) for v in circle]
    rp = int(radius * CROP_PADDING)
    x1, y1 = max(0, cx - rp), max(0, cy - rp)
    x2, y2 = min(w_img, cx + rp), min(h_img, cy + rp)
    region = frame[y1:y2, x1:x2].copy()
    if region.size == 0:
        return None
    mask = np.zeros(region.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (cx - x1, cy - y1), rp, 255, -1)
    if abs(scale - 1.0) > 1e-3:
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        region = cv2.resize(region, None, fx=scale, fy=scale, interpolation=interp)
        mask = cv2.resize(mask, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_NEAREST)
    bg = np.full_like(region, 255)
    return np.where(mask[..., None] > 0, region, bg)


class StillImageCapture:
    """Minimal cv2.VideoCapture look-alike that always returns the same image (demos / tests without camera)."""

    def __init__(self, path):
        self.img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if self.img is None:
            raise FileNotFoundError(path)

    def isOpened(self):
        return True

    def read(self):
        return True, self.img.copy()

    def set(self, *_):
        return True

    def get(self, prop):
        return float(self.img.shape[1] if prop == cv2.CAP_PROP_FRAME_WIDTH else self.img.shape[0])

    def release(self):
        pass


# =============================================================================
# Main window
# =============================================================================
class App(QtWidgets.QMainWindow, Ui_FormDiagnosisToxoplasmosis):
    def __init__(self, still_image=None, camera_index=None):
        super().__init__()
        self.setupUi(self)
        self.still_image = still_image
        self.preferred_camera = camera_index

        # ---- models ----
        self.model_path = str(YOLO_PATH)
        self.model = None                      # YOLO
        self.clf = None                        # dotblot_classifier.DiagnosisModel

        # ---- camera ----
        self.cap = None
        self.timer = QtCore.QTimer(self)
        self.current_camera_index = None
        self.camera_initialized = False
        self.frame_counter = 0

        # ---- state: Diagnosis (one spot) ----
        self.last_frame_bgr = None
        self.last_circles = None
        self.last_conf = None
        self.captured_crop = None              # BGR crop of the selected spot
        self.captured_info = None              # dict(circle, scale, file)
        self.diag_result = None

        # ---- state: Labelling ----
        self.last_frame_bgr_lab = None
        self.last_boxes_xyxy_lab = None
        self.last_conf_lab = None
        self.last_masks_lab = None
        self.last_circles_lab = None
        self.last_label_crops = []
        self.label_views = [self.graphicsViewCroppedImageSpotSegmentation1, self.graphicsViewCroppedImageSpotSegmentation2,
                            self.graphicsViewCroppedImageSpotSegmentation3, self.graphicsViewCroppedImageSpotSegmentation4]

        # ---- state: Raw capture ----
        self.last_frame_bgr_raw = None

        # ---- UI ----
        self._configure_diagnosis_tab()
        self.btnDotBlotSampleCapture.clicked.connect(self.on_capture_clicked)
        self.btnDotBlotSegmentationDiagnosis.clicked.connect(self.on_diagnosis_clicked)
        self.btnExportPDF.clicked.connect(self.on_export_pdf_clicked)
        self.btnRawCaptureSegmentation.clicked.connect(self.on_labelling_capture_clicked)
        self.btnLabelling.clicked.connect(self.on_labelling_save_clicked)
        self.btnRawImageCapture.clicked.connect(self.on_raw_image_capture_clicked)
        self.tabWidgetDiagnosis.currentChanged.connect(self.on_tab_changed)

        self.ensure_classifier_loaded()
        self.initialize_camera()
        self.tabWidgetDiagnosis.setCurrentIndex(0)
        self.on_tab_changed(0)
        self.reset_result_panel()

    # ------------------------------------------------------------------
    # Diagnosis tab layout (one spot instead of four quadrants)
    # ------------------------------------------------------------------
    def _configure_diagnosis_tab(self):
        """Re-use the widgets of gui.py, but lay the tab out with layouts so it follows the window size:
        ONE enlarged segmentation canvas, a scrollable Result group and an 'Export PDF' button on its right."""
        for v in (self.graphicsViewDotBlotSegmentation2, self.graphicsViewDotBlotSegmentation3,
                  self.graphicsViewDotBlotSegmentation4):
            v.hide()
            v.setParent(None)
        self.dot_views = [self.graphicsViewDotBlotSegmentation1]
        self.label_10.setText("Dot Blot Sample (place ONE dot-blot spot)")
        self.label_9.setText("Dot Blot Segmentation")
        self.btnDotBlotSampleCapture.setText("Capture spot")
        self.btnDotBlotSegmentationDiagnosis.setText("Diagnosis")
        self.groupBox_2.setTitle("Result")
        self.groupBox.setTitle("Quantitative decision-support figures")

        # ---- Result group: everything inside ONE scroll area (horizontal + vertical scroll bars as needed) ----
        for w in (self.labelResult, self.labelResult_2):
            w.setParent(None)
        self.labelResult.setAlignment(QtCore.Qt.AlignCenter)
        self.labelResult_2.setAlignment(QtCore.Qt.AlignCenter)
        fr = QtGui.QFont()
        fr.setPointSize(16)
        fr.setBold(True)
        self.labelResult.setFont(fr)
        f2 = QtGui.QFont()
        f2.setPointSize(11)
        f2.setBold(True)
        self.labelResult_2.setFont(f2)
        self.labelProb = QtWidgets.QLabel()
        self.labelProb.setAlignment(QtCore.Qt.AlignCenter)
        fp = QtGui.QFont()
        fp.setPointSize(9)
        self.labelProb.setFont(fp)
        self.labelDetail = QtWidgets.QLabel()
        self.labelDetail.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.labelDetail.setWordWrap(False)                 # long lines are scrolled, not clipped
        fd = QtGui.QFont()
        fd.setPointSize(9)
        fd.setBold(False)
        self.labelDetail.setFont(fd)
        self.labelDetail.setTextFormat(QtCore.Qt.RichText)
        self.labelDetail.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        inner = QtWidgets.QWidget()
        iv = QtWidgets.QVBoxLayout(inner)
        iv.setContentsMargins(8, 4, 8, 4)
        iv.setSpacing(4)
        for w in (self.labelResult, self.labelResult_2, self.labelProb):
            iv.addWidget(w)
        iv.addWidget(self.labelDetail, 1)
        self.resultScroll = QtWidgets.QScrollArea()
        self.resultScroll.setWidgetResizable(True)
        self.resultScroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.resultScroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.resultScroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.resultScroll.setWidget(inner)
        gl = QtWidgets.QVBoxLayout(self.groupBox_2)
        gl.setContentsMargins(6, 24, 6, 6)
        gl.addWidget(self.resultScroll)

        # ---- Export button (to the right of the Result group) ----
        self.btnExportPDF = QtWidgets.QPushButton("Export\nPDF report", self.tabDiagnosis)
        fb = QtGui.QFont()
        fb.setPointSize(11)
        fb.setBold(True)
        self.btnExportPDF.setFont(fb)
        self.btnExportPDF.setMinimumSize(120, 80)
        self.btnExportPDF.setEnabled(False)
        self.btnExportPDF.setToolTip("Save a PDF test report (captured spot, 3 figures, result, clinical interpretation)")

        # ---- charts group: three equal canvases ----
        cl = QtWidgets.QHBoxLayout(self.groupBox)
        cl.setContentsMargins(8, 26, 8, 8)
        cl.setSpacing(8)
        for v in (self.chart1, self.chart2, self.chart3):
            v.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            cl.addWidget(v, 1)

        # ---- tab layout ----
        for v in (self.graphicsViewDotBlotSample, self.graphicsViewDotBlotSegmentation1):
            v.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        g = QtWidgets.QGridLayout(self.tabDiagnosis)
        g.setContentsMargins(10, 8, 10, 8)
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(6)
        g.addWidget(self.label_10, 0, 0)
        g.addWidget(self.label_9, 0, 1)
        g.addWidget(self.graphicsViewDotBlotSample, 1, 0)
        g.addWidget(self.graphicsViewDotBlotSegmentation1, 1, 1)
        g.addWidget(self.btnDotBlotSampleCapture, 2, 0)
        g.addWidget(self.btnDotBlotSegmentationDiagnosis, 2, 1)
        g.addWidget(self.groupBox_2, 0, 2, 3, 1)
        g.addWidget(self.btnExportPDF, 0, 3, 1, 1, QtCore.Qt.AlignTop)
        g.addWidget(self.groupBox, 3, 0, 1, 4)
        for c, st in enumerate((5, 4, 5, 0)):
            g.setColumnStretch(c, st)
        g.setRowStretch(1, 5)
        g.setRowStretch(3, 4)
        g.setRowMinimumHeight(2, 51)
        self.btnDotBlotSampleCapture.setMinimumHeight(51)
        self.btnDotBlotSegmentationDiagnosis.setMinimumHeight(51)
        self.groupBox_2.setMinimumWidth(380)
        self.groupBox.setMinimumHeight(240)

        for v in (self.chart1, self.chart2, self.chart3, self.graphicsViewDotBlotSample, self.graphicsViewDotBlotSegmentation1):
            v.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            v.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._chart_timer = QtCore.QTimer(self)             # debounce chart redraw while the window is resized
        self._chart_timer.setSingleShot(True)
        self._chart_timer.timeout.connect(self._redraw_after_resize)
        self.diag_ctx = None                                # (res, an) of the last diagnosis
        self.setMinimumSize(1000, 700)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.tabWidgetDiagnosis.setGeometry(0, 0, self.width(), self.height())
        if hasattr(self, "_chart_timer"):
            self._chart_timer.start(200)

    def _redraw_after_resize(self):
        try:
            if self.captured_crop is not None:
                show_pixmap_in_graphicsview(self.dot_views[0], cvimg_to_qpixmap(self.captured_crop))
            if self.diag_ctx is not None:
                self.draw_charts(*self.diag_ctx)
        except Exception as e:
            print(f"Redraw after resize failed: {e}")

    def reset_result_panel(self, message="Capture a spot, then press Diagnosis."):
        self.labelResult.setText("—")
        self.labelResult.setStyleSheet("color: #7f8c8d;")
        self.labelResult_2.setText("Toxoplasmosis")
        self.labelProb.setText("")
        self.labelDetail.setText(f"<i>{message}</i>")
        self.diag_ctx = None
        if hasattr(self, "btnExportPDF"):
            self.btnExportPDF.setEnabled(False)
        for v in (self.chart1, self.chart2, self.chart3):
            v.setScene(QGraphicsScene(v))

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    def ensure_model_loaded(self):
        if self.model is None:
            try:
                self.model = YOLO(self.model_path)
                print(f"YOLO model loaded from {self.model_path}")
            except Exception as e:
                QMessageBox.critical(self, "Model error", f"Could not load the YOLO model: {self.model_path}\n\n{e}")
                self.model = None

    def ensure_classifier_loaded(self):
        if self.clf is not None:
            return True
        try:
            self.clf = joblib.load(CLASSIFIER_PATH)
            print(f"Classifier loaded from {CLASSIFIER_PATH}")
            return True
        except Exception as e:
            self.clf = None
            QMessageBox.warning(self, "Classifier not found",
                                f"Could not load {CLASSIFIER_PATH}.\nRun train_diagnosis_model.py first.\n\n{e}\n\n"
                                "The Diagnosis button is disabled; the other tabs work normally.")
            self.btnDotBlotSegmentationDiagnosis.setEnabled(False)
            return False

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------
    def initialize_camera(self):
        self.ensure_model_loaded()
        if self.model is None:
            return False
        if self.still_image:
            self.cap = StillImageCapture(self.still_image)
            self.camera_initialized = True
            print(f"Using still image instead of a camera: {self.still_image}")
            return True
        indices = [self.preferred_camera] if self.preferred_camera is not None else list(cfg.CAMERA_INDICES)
        for idx in indices:
            if self.open_camera(idx):
                self.current_camera_index = idx
                self.camera_initialized = True
                print(f"Camera initialised at index {idx}")
                return True
        QMessageBox.warning(self, "Camera warning",
                            "Could not open the camera (indices tried: " + ", ".join(map(str, indices)) + ").\n"
                            "Close other programs that use the camera (Windows Camera, Teams, another instance of this app) "
                            "and restart. Details are printed in the console.\n\nThe application runs without a camera.")
        self.camera_initialized = False
        return False

    def open_camera(self, camera_index=0):
        """Open a camera with 4K resolution (same procedure as final6.py, which is known to work on this PC)."""
        try:
            # default back-end first, DirectShow as fallback
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if self.cap.isOpened():
                # 4K request (the camera falls back automatically if unsupported)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.CAMERA_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAMERA_HEIGHT)
                self.cap.set(cv2.CAP_PROP_FPS, cfg.CAMERA_FPS)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, cfg.CAMERA_BUFFER_SIZE)   # lower latency
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    print(f"Camera {camera_index} opened successfully with resolution: {actual_width}x{actual_height}")
                    return True
                self.cap.release()
                self.cap = None
                return False
            return False
        except Exception as e:
            print(f"Error opening camera {camera_index}: {e}")
            return False

    def reconnect_camera(self):
        print("Attempting to reconnect camera...")
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.camera_initialized = False
        return self.initialize_camera()

    def stop_camera(self):
        self.timer.stop()
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                print(f"Error releasing camera: {e}")
        self.cap = None
        self.camera_initialized = False

    def closeEvent(self, event):
        self.stop_camera()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Timer routing per tab
    # ------------------------------------------------------------------
    def _retarget_timer(self, slot):
        self.timer.stop()
        try:
            self.timer.timeout.disconnect()            # PyQt5: without arguments this removes all connections
        except TypeError:
            pass                                       # nothing was connected yet
        self.timer.timeout.connect(slot)
        self.timer.start(cfg.PREVIEW_INTERVAL_MS)

    def on_tab_changed(self, idx: int):
        """0 = Diagnosis, 1 = Labelling, 2 = Raw capture."""
        if idx == 0:
            self._retarget_timer(self.on_frame_tick)
        elif idx == 1:
            self._retarget_timer(self.on_frame_tick_labelling)
        elif idx == 2:
            self._retarget_timer(self.on_frame_tick_rawcapture)
        else:
            self.timer.stop()

    def _camera_placeholder(self, view):
        placeholder = np.full((480, 640, 3), 200, dtype=np.uint8)
        cv2.putText(placeholder, "Camera Not Available", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        show_pixmap_in_graphicsview(view, cvimg_to_qpixmap(placeholder))

    # ============================================================
    # TAB 1 - DIAGNOSIS (one spot)
    # ============================================================
    def best_detection(self):
        """Index of the most confident detection (the single dot-blot spot to be analysed)."""
        if not self.last_circles:
            return None
        conf = self.last_conf if self.last_conf is not None and len(self.last_conf) == len(self.last_circles) \
            else np.ones(len(self.last_circles))
        return int(np.argmax(conf))

    def on_frame_tick(self):
        if not self.camera_initialized or self.cap is None:
            if not self.reconnect_camera():
                self._camera_placeholder(self.graphicsViewDotBlotSample)
            return
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.frame_counter += 1
                if self.frame_counter > 30:
                    self.frame_counter = 0
                    self.reconnect_camera()
                return
            self.frame_counter = 0
            self.last_frame_bgr = frame.copy()
            t_pred = time.time()
            try:
                r = self.model.predict(source=frame, verbose=False, **cfg.YOLO_PREDICT_KWARGS)[0]
            except Exception as e:
                print(f"Model prediction error: {e}")
                show_pixmap_in_graphicsview(self.graphicsViewDotBlotSample, cvimg_to_qpixmap(frame))
                return
            self.last_circles, self.last_conf = detections_from_result(r)
            self._tick_n = getattr(self, "_tick_n", 0) + 1
            if self._tick_n % 30 == 1:
                dbg(f"tick {self._tick_n}: frame {frame.shape[1]}x{frame.shape[0]}, detections "
                    f"{0 if not self.last_circles else len(self.last_circles)}, predict {1000 * (time.time() - t_pred):.0f} ms "
                    f"(timer period {cfg.PREVIEW_INTERVAL_MS} ms), canvas {self.graphicsViewDotBlotSample.viewport().width()}x"
                    f"{self.graphicsViewDotBlotSample.viewport().height()}")

            preview = frame.copy()
            best = self.best_detection()
            if self.last_circles:
                for i, (cx, cy, rad) in enumerate(self.last_circles):
                    if i == best:
                        cv2.circle(preview, (cx, cy), rad, (0, 200, 0), 15)          # selected spot: green
                        cv2.circle(preview, (cx, cy), 4, (0, 200, 0), -1)
                    else:
                        cv2.circle(preview, (cx, cy), rad, (160, 160, 160), 6)       # other detections: thin grey
            show_pixmap_in_graphicsview(self.graphicsViewDotBlotSample, cvimg_to_qpixmap(preview))
        except Exception as e:
            print(f"Error in on_frame_tick: {e}")

    # ---- capture ---------------------------------------------------
    def on_capture_clicked(self, _checked=False, quiet=False):
        """Freeze the current frame and crop the most confident spot (one spot = one test)."""
        dbg(f"capture clicked: frame={'none' if self.last_frame_bgr is None else self.last_frame_bgr.shape}, "
            f"detections={0 if not self.last_circles else len(self.last_circles)}")
        if self.last_frame_bgr is None:
            QMessageBox.information(self, "Info", "No camera frame available yet.")
            return False
        best = self.best_detection()
        if best is None:
            self.captured_crop = None
            self.reset_result_panel("No spot detected. Place the dot-blot in the field of view.")
            self.dot_views[0].setScene(QGraphicsScene(self.dot_views[0]))
            return False
        circle = self.last_circles[best]
        r_equiv = dbf.REFERENCE_RADIUS_PX
        scale = r_equiv / max(circle[2] * CROP_PADDING, 1.0)
        if abs(scale - 1.0) <= SCALE_TOLERANCE:
            scale = 1.0                                   # within the size range of the training crops
        crop = crop_circle(self.last_frame_bgr, circle, scale)
        if crop is None or crop.size == 0:
            QMessageBox.warning(self, "Capture", "Cropping failed. Try again.")
            return False
        self.captured_crop = crop
        DIAG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = DIAG_DIR / f"{stamp}_crop.png"
        cv2.imwrite(str(fname), crop)
        self.captured_info = dict(circle=circle, scale=scale, file=fname.name, conf=float(self.last_conf[best]) if self.last_conf is not None else 1.0)
        self.diag_result = None
        self.reset_result_panel("Spot captured. Press <b>Diagnosis</b> to analyse it.")
        show_pixmap_in_graphicsview(self.dot_views[0], cvimg_to_qpixmap(crop))
        v = self.dot_views[0]
        dbg(f"capture: circle={circle}, scale={scale:.2f}, crop={crop.shape}, saved={fname}, canvas visible={v.isVisible()}, "
            f"viewport={v.viewport().width()}x{v.viewport().height()}, scene items={len(v.scene().items())}")
        return True

    # ---- diagnosis -------------------------------------------------
    def on_diagnosis_clicked(self):
        if self.clf is None and not self.ensure_classifier_loaded():
            return
        if self.captured_crop is None:
            if not self.on_capture_clicked(quiet=True):
                return
        gray = cv2.cvtColor(self.captured_crop, cv2.COLOR_BGR2GRAY)
        an = dbf.analyze_spot(gray)
        if not an.ok:
            self.show_invalid(an.reason)
            return
        res = self.clf.predict(an.features)
        self.diag_result = dict(res, qc=an.reason, geom=an.geom)
        self.diag_ctx = (res, an)
        self.show_result(res, an)
        self.draw_charts(res, an)
        self.append_log(res, an)
        self.btnExportPDF.setEnabled(True)

    def show_invalid(self, reason):
        why = {"blank": "the crop is empty (no spot found in the crop)",
               "small_mask": "the detected spot is too small",
               "truncated": "the spot is truncated or touches the crop border"}.get(reason, reason)
        self.labelResult.setText("INVALID")
        self.labelResult.setStyleSheet(f"color: {COLORS['INVALID']};")
        self.labelResult_2.setText("capture rejected")
        self.labelProb.setText("")
        self.labelDetail.setText(f"Acquisition QC failed: {why}.<br><br>Re-position the dot-blot so that the whole<br>"
                                 "spot is visible and in focus, then<br><b>Capture spot</b> again.")
        self.diag_ctx = None
        self.btnExportPDF.setEnabled(False)
        for v in (self.chart1, self.chart2, self.chart3):
            v.setScene(QGraphicsScene(v))

    def show_result(self, res, an):
        lab = res["label"]
        self.labelResult.setText(lab)
        self.labelResult.setStyleSheet(f"color: {COLORS[lab]}; font-weight: bold;")
        self.labelResult_2.setText("Toxoplasmosis" if lab != "INDETERMINATE" else "inconclusive")
        self.labelProb.setText(f"score {res['score']:.2f}  |  P ≈ {res['probability'] * 100:.0f} %")
        m = self.clf.metrics
        lo, hi = m["spot_auc_ci"]
        g = an.geom
        od = np.where(an.roi, an.od, np.nan)
        L = [f"<b>Decision rule</b><br>NEGATIVE ≤ {self.clf.band[0]:.2f} &lt; INDETERMINATE &lt; {self.clf.band[1]:.2f} ≤ POSITIVE<br>",
             f"<b>Model</b>: {self.clf.meta['features'].split(' (')[0]} + MLP, {self.clf.meta['lux']} lux<br>",
             f"<b>Spot</b>: R = {g['R']:.1f} px, mean OD {np.nanmean(od):.3f} ± {np.nanstd(od):.3f}<br>"]
        if self.captured_info and self.captured_info["scale"] != 1.0:
            L.append(f"Crop rescaled ×{self.captured_info['scale']:.2f} to the training spot size.<br>")
        note = rep.reference_range_note(an, self.clf)
        if note:
            L.append("<span style='color:#c0392b'><b>Warning</b>: spot OD far outside the training range;<br>check illumination, focus and exposure.</span><br>")
        L.append(f"<br><b>Cross-validated (single spot)</b><br>AUC {m['spot_auc']:.2f} (95% CI {lo:.2f}–{hi:.2f}), permutation p = {m['permutation_p']:.2f}<br>"
                 f"{m['coverage'] * 100:.0f}% of validation spots receive a POSITIVE/NEGATIVE call.<br>")
        if lo <= 0.5:
            L.append("<span style='color:#c0392b'><b>The CI includes 0.5: single-spot performance is<br>not distinguishable from chance.</b></span><br>")
        L.append("<br><b>Clinical interpretation</b><br>")
        import re as _re
        for t in rep.clinical_interpretation(lab):
            t = _re.sub(r"<[^>]+>", "", t)
            import textwrap
            L.append("• " + "<br>&nbsp;&nbsp;".join(textwrap.wrap(t, 58)) + "<br>")
        L.append("<br><span style='color:#7f8c8d'>Research prototype - not a validated diagnostic device.<br>Confirm with the reference assay.</span>")
        self.labelDetail.setText("".join(L))

    # ---- figures ---------------------------------------------------
    def _canvas_size(self, view):
        return max(view.viewport().width(), 300), max(view.viewport().height(), 250)

    def draw_charts(self, res, an):
        """The three decision-support figures (shared with the PDF report, see dotblot_report.py)."""
        for view, build in ((self.chart1, lambda w, h: rep.fig_od_map(an, w, h)),
                            (self.chart2, lambda w, h: rep.fig_radial_profile(an, self.clf, w, h)),
                            (self.chart3, lambda w, h: rep.fig_score_position(res, self.clf, w, h))):
            w, h = self._canvas_size(view)
            show_pixmap_in_graphicsview(view, fig_to_pixmap(build(w, h)))

    # ---- PDF report ------------------------------------------------
    def on_export_pdf_clicked(self):
        if self.diag_ctx is None or self.captured_crop is None:
            QMessageBox.information(self, "Export", "Run a diagnosis first.")
            return
        res, an = self.diag_ctx
        sample_id, ok = QtWidgets.QInputDialog.getText(self, "Sample ID", "Sample / patient ID (optional, printed on the report):")
        if not ok:
            return
        REPORT_DIR = DIAG_DIR.parent / "reports"
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        now = datetime.datetime.now()
        default = str(REPORT_DIR / f"report_{now.strftime('%Y%m%d_%H%M%S')}.pdf")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save PDF report", default, "PDF files (*.pdf)")
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        try:
            rep.export_pdf(path, dict(res=res, an=an, clf=self.clf, crop_bgr=self.captured_crop, info=self.captured_info,
                                      timestamp=now, sample_id=sample_id.strip()))
        except Exception as e:
            QMessageBox.warning(self, "Export failed", f"Could not write the PDF report:\n{e}")
            return
        self.statusTip = path
        QMessageBox.information(self, "Export", f"PDF report saved:\n{path}")

    # ---- logging ---------------------------------------------------
    def append_log(self, res, an):
        try:
            DIAG_DIR.mkdir(parents=True, exist_ok=True)
            path = DIAG_DIR / "diagnosis_log.csv"
            new = not path.exists()
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["timestamp", "crop_file", "score", "probability", "label", "qc", "rescale",
                                "model_built", "model_lux"])
                w.writerow([datetime.datetime.now().isoformat(timespec="seconds"),
                            self.captured_info["file"] if self.captured_info else "", f"{res['score']:.4f}",
                            f"{res['probability']:.4f}", res["label"], an.reason,
                            f"{self.captured_info['scale']:.3f}" if self.captured_info else "",
                            self.clf.meta.get("built", ""), self.clf.meta.get("lux", "")])
        except Exception as e:
            print(f"Could not write the diagnosis log: {e}")

    # ============================================================
    # TAB 2 - LABELLING (unchanged behaviour)
    # ============================================================
    def _get_quadrant(self, x: int, y: int, mid_x: int, mid_y: int) -> int:
        if x < mid_x and y < mid_y:
            return 2
        if x >= mid_x and y < mid_y:
            return 1
        if x < mid_x and y >= mid_y:
            return 3
        return 4

    def on_frame_tick_labelling(self):
        if not self.camera_initialized or self.cap is None:
            if not self.reconnect_camera():
                self._camera_placeholder(self.graphicsViewRawImageImageSpotSegmentation)
            return
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return
            self.last_frame_bgr_lab = frame.copy()
            try:
                r = self.model.predict(source=frame, verbose=False, **cfg.YOLO_PREDICT_KWARGS)[0]
            except Exception:
                preview_fail = frame.copy()
                h, w = preview_fail.shape[:2]
                cx, cy = w // 2, h // 2
                cv2.line(preview_fail, (0, cy), (w - 1, cy), (255, 0, 0), 5)
                cv2.line(preview_fail, (cx, 0), (cx, h - 1), (255, 0, 0), 5)
                cv2.circle(preview_fail, (cx, cy), 6, (0, 255, 0), -1)
                show_pixmap_in_graphicsview(self.graphicsViewRawImageImageSpotSegmentation, cvimg_to_qpixmap(preview_fail))
                return
            self.last_circles_lab, self.last_conf_lab = detections_from_result(r)
            preview = frame.copy()
            if self.last_circles_lab is not None:
                for (cx, cy, rad) in self.last_circles_lab:
                    cv2.circle(preview, (cx, cy), rad, (0, 0, 255), 15)
                    cv2.circle(preview, (cx, cy), 3, (255, 0, 0), -1)
            h, w = preview.shape[:2]
            cx, cy = w // 2, h // 2
            cv2.line(preview, (0, cy), (w - 1, cy), (0, 255, 0), 7)
            cv2.line(preview, (cx, 0), (cx, h - 1), (0, 255, 0), 7)
            cv2.circle(preview, (cx, cy), 6, (0, 255, 0), -1)
            show_pixmap_in_graphicsview(self.graphicsViewRawImageImageSpotSegmentation, cvimg_to_qpixmap(preview))
        except Exception as e:
            print(f"Error in on_frame_tick_labelling: {e}")

    def clear_label_views(self):
        for v in self.label_views:
            v.setScene(QGraphicsScene(v))

    def on_labelling_capture_clicked(self):
        try:
            if self.last_frame_bgr_lab is None:
                QMessageBox.information(self, "Info", "No camera frame available yet (Labelling).")
                return
            frame = self.last_frame_bgr_lab
            if not self.last_circles_lab:
                QMessageBox.information(self, "Info", "No object detected to capture (Labelling).")
                self.clear_label_views()
                self.last_label_crops = []
                return
            circles = self.last_circles_lab
            conf = self.last_conf_lab
            if conf is None or len(conf) != len(circles):
                conf = np.ones((len(circles),), dtype=np.float32)
            h_img, w_img = frame.shape[:2]
            mid_x, mid_y = w_img // 2, h_img // 2

            best_idx = {1: None, 2: None, 3: None, 4: None}
            best_score = {1: -1.0, 2: -1.0, 3: -1.0, 4: -1.0}
            for idx, (cx, cy, r) in enumerate(circles):
                q = self._get_quadrant(int(cx), int(cy), mid_x, mid_y)
                if float(conf[idx]) > best_score[q]:
                    best_score[q] = float(conf[idx])
                    best_idx[q] = idx

            quad_order = [2, 1, 3, 4]
            crops, first_valid = [], None
            for q in quad_order:
                if best_idx[q] is not None:
                    tmp = crop_circle(frame, circles[best_idx[q]])
                    if tmp is not None:
                        first_valid = tmp
                        break
            placeholder = np.full((256, 256, 3), 255, dtype=np.uint8) if first_valid is None else np.full_like(first_valid, 255)
            for q in quad_order:
                c = crop_circle(frame, circles[best_idx[q]]) if best_idx[q] is not None else None
                crops.append(placeholder.copy() if c is None else c)

            self.clear_label_views()
            for i in range(4):
                show_pixmap_in_graphicsview(self.label_views[i], cvimg_to_qpixmap(crops[i]))
            self.last_label_crops = crops
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during capture (Labelling):\n{e}")

    def _get_label_fields_for_index(self, i: int):
        lux = sanitize_filename_part(self.textEditLuxLumination.toPlainText())
        col = sanitize_filename_part(getattr(self, f"textEditColumnPosition{i + 1}").toPlainText())
        row = sanitize_filename_part(getattr(self, f"textEditRowPosition{i + 1}").toPlainText())
        lab = sanitize_filename_part(getattr(self, f"comboBoxLabel{i + 1}").currentText())
        return col, row, lab, lux

    def on_labelling_save_clicked(self):
        if not self.last_label_crops:
            QMessageBox.information(self, "Info", "No crops yet. Click 'Raw Capture Segmentation' first.")
            return
        LABEL_DIR.mkdir(parents=True, exist_ok=True)
        saved = []
        for i in range(min(4, len(self.last_label_crops))):
            col, row, lab, lux = self._get_label_fields_for_index(i)
            filename = f"{col}_{row}_{lab}_{lux}.png"
            if cv2.imwrite(str(LABEL_DIR / filename), self.last_label_crops[i]):
                saved.append(filename)
        if saved:
            QMessageBox.information(self, "Labelling saved", f"PNG files saved in:\n{LABEL_DIR}\n\nFile names:\n- " + "\n- ".join(saved))
        else:
            QMessageBox.warning(self, "Failed", "No file could be saved. Check folder/drive permissions.")

    # ============================================================
    # TAB 3 - RAW CAPTURE (unchanged behaviour)
    # ============================================================
    def on_frame_tick_rawcapture(self):
        if not self.camera_initialized or self.cap is None:
            if not self.reconnect_camera():
                self._camera_placeholder(self.graphicsViewRawImageSpot)
            return
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return
            self.last_frame_bgr_raw = frame.copy()
            show_pixmap_in_graphicsview(self.graphicsViewRawImageSpot, cvimg_to_qpixmap(frame))
        except Exception as e:
            print(f"Error in on_frame_tick_rawcapture: {e}")

    def on_raw_image_capture_clicked(self):
        if self.last_frame_bgr_raw is None:
            QMessageBox.information(self, "Info", "No camera frame available yet (Raw capture).")
            return
        frame = self.last_frame_bgr_raw
        show_pixmap_in_graphicsview(self.graphicsViewRawImageCaptured, cvimg_to_qpixmap(frame))
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        filepath = RAW_DIR / f"{uuid.uuid4().hex}.png"
        if cv2.imwrite(str(filepath), frame):
            QMessageBox.information(self, "Saved", f"Image saved:\n{filepath}")
        else:
            QMessageBox.warning(self, "Failed", "Could not save the PNG file. Check folder/drive permissions.")


def main():
    ap = argparse.ArgumentParser(description="Anti-GRA7 dot-blot diagnosis application")
    ap.add_argument("--image", help="use this still image instead of the camera (demo / test)")
    ap.add_argument("--camera", type=int, default=cfg.CAMERA_INDEX, help="camera index (default: config.CAMERA_INDICES)")
    args, qt_args = ap.parse_known_args()
    app = QtWidgets.QApplication([sys.argv[0]] + qt_args)
    w = App(still_image=args.image, camera_index=args.camera)
    w.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
