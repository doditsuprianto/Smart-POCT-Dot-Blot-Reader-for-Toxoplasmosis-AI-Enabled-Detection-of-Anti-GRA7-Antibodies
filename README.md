# Anti-GRA7 dot-blot reader for *Toxoplasma gondii* serodiagnosis — code, data and models

Code, notebooks and results accompanying the manuscript *Development and Proof-of-Concept Validation of an Image-Segmentation and Machine-Learning System for Automated Interpretation of Anti-GRA7 Dot-Blot Assays in Toxoplasma gondii Serodiagnosis* (manuscript and supplementary material are distributed separately).

The system has four parts, each in its own folder:

| Folder | Content | Main entry point |
|---|---|---|
| `Body Case/` | SketchUp (`.skp`) models of the camera/reader enclosure (body, top cover, panel cover, object slider, assemblies) | open in SketchUp |
| `DotBlot YOLO Segmentation/` | YOLOv8n-seg spot detection: audited dataset, training, validation, spot-level metrics, cropping; published weights and results | `Segmentation_Dotblot_YOLOv8_Audited.ipynb` |
| `Dot-Blot Model ML CNN/` | Artefact-controlled classification from grayscale crops: quantification families (Q1–Q6) + 9 ML algorithms (pipeline A) and 5 CNNs (pipeline B), controls, permutation tests, ML-vs-CNN comparison | `Dot-Blot Model.ipynb` |
| `Toxoplasmosis_Diagnosis_Device/` | Desktop application of the point-of-care reader (camera, YOLO detection, cropping, classification, labelling and raw-capture tabs); has its own README | `dotblot_diagnosis_app.py` |
| `Supplementary_Code/` | Stand-alone scripts for the supplementary analyses (artefact diagnostics, membrane-group sensitivity, selection-aware permutation, mask-geometry control, manuscript figures) | see table below |

The segmentation and auto-cropping stage produces the grayscale crops; the classification notebook **starts from those crops** (it does not repeat segmentation).

## Repository layout

```
Body Case/                      *.skp
DotBlot YOLO Segmentation/
  Segmentation_Dotblot_YOLOv8_Audited.ipynb   audited dataset -> training -> validation -> cropping
  Figure5_Segmentation_Example.ipynb           segmentation example figure (ground truth vs prediction)
  dataset/{train,validation,test}/          YOLO-segmentation format (images/, labels/); see "Data"
  yolo_results/                             results of the manuscript run (weights/best.pt, metrics, reports)
  figures/
Dot-Blot Model ML CNN/
  Dot-Blot Model.ipynb                      full classification pipeline (Colab or local)
  Dataset/Grayscale_Test {1,2,3}/      1,360 grayscale crops (3 tests x 5 illuminances)
  Results/                                  tables (T01-T20), figures (F01-F20), cache, RESULTS_SUMMARY.md, run_manifest.json
Toxoplasmosis_Diagnosis_Device/             application, config.py, trained classifier, README (see its own README)
Supplementary_Code/
```

## Data

* **Grayscale crops (classification):** `Dot-Blot Model ML CNN/Dataset/` — 480 + 440 + 440 = 1,360 PNG crops (46 sera, 272 spots, five illuminances: 250, 350, 470, 560, 850 lux). Included in the repository (about 14 MB).
* **Segmentation images (YOLO):** 124 full-membrane photographs (3,840 × 2,160 px, about 25 MB each): 85 training and 39 validation images with 1,615 annotated spots (1,090 and 525). The polygon labels (`labels/`) and list files are in the repository; the **images are not** (about 3 GB, excluded by `.gitignore`) and must be placed in
  `DotBlot YOLO Segmentation/dataset/train/images/Train/` and `.../dataset/validation/images/Validation/`.
  Download: *[add Zenodo/Drive link or DOI here]*.
* **Model weights:** `yolo_results/dotblot-yolov8n-seg/weights/best.pt` (manuscript model). The 60 cross-validation CNN checkpoints (`Results/models/`, about 0.9 GB) are not tracked; the notebook re-creates them.

## Quick start

### Environment

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

The notebooks also run unchanged in Google Colab (they install missing packages and mount Drive when needed).

### 1. Segmentation (YOLOv8n-seg)

Open `DotBlot YOLO Segmentation/Segmentation_Dotblot_YOLOv8_Audited.ipynb` from inside that folder (or set `DOTBLOT_YOLO_ROOT` to it). Paths are resolved automatically; nothing has to be edited.

| Environment variable | Default | Meaning |
|---|---|---|
| `DOTBLOT_YOLO_ROOT` | auto-detected (Colab Drive folder, current folder, sub-/parent folder) | folder that contains `dataset/` |
| `DOTBLOT_TRAIN` | `1` | `0` = skip training and evaluate the published weights in `yolo_results/` (CPU is sufficient) |
| `DOTBLOT_OUT_DIR` | `<root>/yolo_results_rerun` | where re-runs are written (the published `yolo_results/` is never overwritten) |
| `DOTBLOT_TMP` | system temp folder | scratch space for the corrected working copy of the dataset |

Training (50 epochs, 1,920 px, batch 4) needs a GPU. The dataset folders are read in place and never modified; a corrected working copy is built in the scratch folder and every correction is logged.
`Figure5_Segmentation_Example.ipynb` draws the example figure from the validation image and `best.pt` (`DOTBLOT_WEIGHTS` overrides the weights file).

### 2. Classification (ML and CNN)

Open `Dot-Blot Model ML CNN/Dot-Blot Model.ipynb` from inside that folder.

| Environment variable | Default | Meaning |
|---|---|---|
| `RUN_MODE` | `FULL` | `FAST` = reduced configuration for a quick functional test |
| `WORK_ROOT` | current folder (local) / `MyDrive/Dotblot_GRA7` (Colab) | folder that contains `Dataset/` |
| `DATA_ROOT`, `OUT_DIR` | `<WORK_ROOT>/Dataset`, `<WORK_ROOT>/Results_rerun` (local) | dataset and output folders |
| `N_JOBS` | `-1` | parallel workers for the ML part |

The `FULL` run (5 CNN architectures × 2 protocols × 5 folds, nested tuning, bootstrap and permutation tests) is intended for a GPU; outputs are written to `Results_rerun/` so that the published `Results/` stays intact. `Results/RESULTS_SUMMARY.md` and `Results/run_manifest.json` document the published run (seed 42, 250 lux, per-membrane normalisation).

### 3. Supplementary scripts

Run from `Supplementary_Code/` after the classification results exist in `Dot-Blot Model ML CNN/Results/`. Outputs go to `Supplementary_Code/output/` (`DOTBLOT_RESULTS` and `DOTBLOT_SUPP_OUT` override the locations).

| Script | Purpose | Output |
|---|---|---|
| `artefact_diagnostics_250lux.py` | crop-size / grey-level / membrane diagnostics, silhouette scores, PCA at 250 lux | `stats250.json`, `N05_artefact_250.png`, `N06_pca_250.png` |
| `artefact_diagnostics_all_lux.py` | the same diagnostics at all five illuminances | `multi_lux.csv` |
| `extra_membrane_sensitivity.py` | ALL-protocol models re-scored within membrane groups (M1 vs M2/M3) | `ALL_membrane_sensitivity.csv` |
| `extra_selection_aware_permutation_and_position_control.py` | selection-aware label-permutation test (MIX) and well-position control (`NP` = number of permutations, default 300) | `sel_perm.pkl` |
| `geometry_control_250lux.py` | mask-geometry negative control (univariate statistics, SVM, spot-level vs group-wise CV) | `geo250_univariate.csv`, `geo250_svm.csv` |
| `n03.py` | permutation / membrane-group figure (needs the two scripts above) | `N03_perm_membrane.png` |
| `newfigs.py` | workflow and cohort-composition figures | `N01_workflow.png`, `N02_composition.png` |

Reference outputs of the manuscript run are provided alongside the scripts (`*.csv`, `artefact_diagnostics_250lux.json`).

## Reproducibility notes

* Random seeds are fixed (classification: 42 + repeat; YOLO: 0, deterministic mode). Re-training on a different GPU/driver can still change the last digits.
* The per-membrane normalisation (`'mem'`) and the OD-histogram bin edges are label-free but are computed once on all data (transductive); all other pre-processing, tuning and early stopping use training sera only (see the Limitations section and Supplementary Figure S11 of the manuscript).
* Spot-level and image-level metrics are not independent observations: the statistical unit is the serum (n = 46; MIX = 22).
* `Dot-Blot Model.ipynb` and `Segmentation_Dotblot_YOLOv8_Audited.ipynb` contain the outputs of the manuscript runs (executed in Colab).

## Verification status

Functional checks performed on this folder layout (CPU-only machine, 2 cores, 8 GB RAM; Python 3.13, ultralytics 8.4.208, torch 2.15):

| Component | Check | Result |
|---|---|---|
| `Segmentation_Dotblot_YOLOv8_Audited.ipynb` | executed end to end with `DOTBLOT_TRAIN=0` (published weights; training skipped) on the 39 validation images and 8 of the 85 training images | runs without errors; Ultralytics metrics of `best.pt` and `last.pt`, mask-F1 peaks, spot-level TP/FP/FN at confidence 0.25 / 0.50 / 0.75 (512/55/13, 502/22/23, 421/3/104) and the image-size sensitivity table are **identical** to `yolo_results/…_report.json` |
| `Dot-Blot Model.ipynb` | executed end to end with `RUN_MODE=FAST` (reduced configuration, 1,360 crops) | runs without errors and writes all tables, figures and models; `T01` and `T03` identical to the published tables, `T04` (features) identical to within 3e-5. The `FAST` mode is a functional test, its AUC values are not those of the manuscript |
| `Supplementary_Code/*.py` | executed against `Results/` | `ALL_membrane_sensitivity.csv`, `multi_lux.csv`, `stats250.json` and the univariate geometry table reproduce exactly; the geometry-SVM table reproduces for ALL, while the near-chance MIX AUC varies by about 0.04 because `SVC(probability=True)` is unseeded; the selection-aware permutation script was run with 3 permutations only (`NP=3`) |

**Not re-run here:** YOLO training (needs a GPU), the `FULL` ML/CNN run with pretrained CNNs (needs a GPU and several hours), `Figure5_Segmentation_Example.ipynb` (configuration cell only checked), and the 300-permutation selection-aware test.
The YOLO spot-level cell was rewritten to rasterise masks inside their joint bounding box (same IoU, about 1,000 times less memory); the original full-image version needed more than 10 GB of RAM.

## Credit

PoCT - Toxoplasmosis Diagnosis Device (Dec, 2026)  
Dodit Suprianto  
Email: Dodit.suprianto@polinema.ac.id  
Affiliation:
- Department of Electrical Engineering, Politeknik Negeri Malang, Malang, Indonesia
- AIDS, Toxoplasma, Opportunistic Disease and Malaria (ATOM) Research Group, Faculty of Medicine, Universitas Brawijaya, Malang, Indonesia

## License and citation

*[License to be chosen by the authors, e.g. MIT for code and CC BY 4.0 for data and figures.]*
*[Citation / DOI of the published manuscript to be added.]*
