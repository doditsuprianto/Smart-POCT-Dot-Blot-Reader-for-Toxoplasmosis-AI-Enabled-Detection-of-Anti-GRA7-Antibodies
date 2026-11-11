# Smart POCT Dot-Blot Reader for Toxoplasmosis: AI-Enabled Detection of Anti-GRA7 Antibodies

Code, data, trained models and results accompanying the manuscript *Development and Proof-of-Concept Validation of an Image-Segmentation and Machine-Learning System for Automated Interpretation of Anti-GRA7 Dot-Blot Assays in Toxoplasma gondii Serodiagnosis* (manuscript and supplementary material are distributed separately).

The system has five parts, each in its own folder:

| Folder | Content | Main entry point |
|---|---|---|
| `Body_Case/` | SketchUp (`.skp`) models of the camera/reader enclosure (body, top cover, panel cover, object slider, assemblies) | open in SketchUp |
| `Dot_Blot_YOLO_Segmentation/` | YOLOv8n-seg spot detection: audited dataset, training, validation, spot-level metrics, cropping; published weights and results | `Segmentation_Dotblot_YOLOv8_Audited.ipynb` |
| `Dot_Blot_Model_ML_CNN/` | Artefact-controlled classification from grayscale crops: quantification families (Q1–Q6) + 9 ML algorithms (pipeline A) and 5 CNNs (pipeline B), controls, permutation tests, ML-vs-CNN comparison | `Dot-Blot Model.ipynb` |
| `Toxoplasmosis_Diagnosis_Device/` | Desktop application of the point-of-care reader (camera, YOLO detection, cropping, classification, labelling and raw-capture tabs); has its own README | `dotblot_diagnosis_app.py` |
| `Supplementary_Code/` | Stand-alone scripts for the supplementary analyses (artefact diagnostics, membrane-group sensitivity, selection-aware permutation, mask-geometry control, manuscript figures) | see table below |

The segmentation and auto-cropping stage produces the grayscale crops; the classification notebook **starts from those crops** (it does not repeat segmentation).

## Repository layout

```
Body_Case/                      *.skp
Dot_Blot_YOLO_Segmentation/
  Segmentation_Dotblot_YOLOv8_Audited.ipynb   audited dataset -> training -> validation -> cropping
  Figure5_Segmentation_Example.ipynb           segmentation example figure (ground truth vs prediction)
  dataset/{train,validation,test}/          YOLO-segmentation format (images/, labels/); see "Data"
  yolo_results/                             results of the manuscript run (weights/best.pt, metrics, reports)
  figures/
Dot_Blot_Model_ML_CNN/
  Dot-Blot Model.ipynb                      full classification pipeline (Colab or local)
  Dataset/Grayscale_Test {1,2,3}/      1,360 grayscale crops (3 tests x 5 illuminances)
  Results/                                  tables (T01-T20), figures (F01-F20), cache, RESULTS_SUMMARY.md, run_manifest.json
Toxoplasmosis_Diagnosis_Device/             application, config.py, trained classifier, README (see its own README)
Supplementary_Code/
```

## Data

Everything needed to reproduce the study is in this repository (no data are available only on request). Large binary files are handled as follows:

| Content | Location | Size | How it is stored |
|---|---|---|---|
| Grayscale crops (classification) | `Dot_Blot_Model_ML_CNN/Dataset/Grayscale_Test {1,2,3}/` — 480 + 440 + 440 = 1,360 PNG crops (46 de-identified sera, 272 spots, five illuminances: 250, 350, 470, 560, 850 lux) | about 16 MB | regular git |
| Sample index / QC / features | `Dot_Blot_Model_ML_CNN/Results/tables/T01_image_index.csv`, `T03_qc_flags.csv`, `T04_features_raw.csv` (anonymised sample codes only) | small | regular git |
| Segmentation images (YOLO) | `Dot_Blot_YOLO_Segmentation/dataset/{train,validation}/images/` — 124 full-membrane photographs (3,840 × 2,160 px, about 25 MB each): 85 training and 39 validation images with 1,615 annotated spots (1,090 and 525); polygon labels in `labels/`; extra example photographs in `dataset/test/` | about 3 GB | regular git (PNG) |
| YOLO weights | `Dot_Blot_YOLO_Segmentation/yolo_results/dotblot-yolov8n-seg/weights/{best,last}.pt` (manuscript model) and `Toxoplasmosis_Diagnosis_Device/model/best.pt` (weights of the device) | few MB | **Git LFS** |
| CNN checkpoints | `Dot_Blot_Model_ML_CNN/Results/models/*.pt` (60 cross-validation checkpoints) and `Results/cache/*.pkl` | about 0.9 GB | **Git LFS** |
| Enclosure models | `Body_Case/*.skp` | small | **Git LFS** |

### Git LFS

`*.pt`, `*.pkl`, `*.skp` and `*.zip` are stored with [Git LFS](https://git-lfs.com). Without it you receive 130-byte pointer files instead of weights and checkpoints. Install it once and clone:

```bash
git lfs install
git clone https://github.com/doditsuprianto/Smart-POCT-Dot-Blot-Reader-for-Toxoplasmosis-AI-Enabled-Detection-of-Anti-GRA7-Antibodies.git
# already cloned without LFS?  ->  git lfs pull
```

The full clone is large (several GB, mostly the 124 membrane photographs). To fetch only the code, notebooks and the 1,360 crops, use a sparse checkout (`git sparse-checkout set Dot_Blot_Model_ML_CNN Supplementary_Code Toxoplasmosis_Diagnosis_Device`). The notebooks and the application re-create everything else.

### Ethics and de-identification

Sera were de-identified before imaging; file names and tables use anonymised sample codes (e.g. `S09`) and contain, besides the images, only the reference serology used as labels (CLIA status, IgM and IgG values); no names, identifiers, dates or other personal or clinical data are included. Patient-related output of the application (`Toxoplasmosis_Diagnosis_Device/output/`, including PDF reports) is excluded by `.gitignore` and must never be committed.

## Quick start

### Environment

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

The notebooks also run unchanged in Google Colab (they install missing packages and mount Drive when needed).

### 1. Segmentation (YOLOv8n-seg)

Open `Dot_Blot_YOLO_Segmentation/Segmentation_Dotblot_YOLOv8_Audited.ipynb` from inside that folder (or set `DOTBLOT_YOLO_ROOT` to it). Paths are resolved automatically; nothing has to be edited.

| Environment variable | Default | Meaning |
|---|---|---|
| `DOTBLOT_YOLO_ROOT` | auto-detected (Colab Drive folder, current folder, sub-/parent folder) | folder that contains `dataset/` |
| `DOTBLOT_TRAIN` | `1` | `0` = skip training and evaluate the published weights in `yolo_results/` (CPU is sufficient) |
| `DOTBLOT_OUT_DIR` | `<root>/yolo_results_rerun` | where re-runs are written (the published `yolo_results/` is never overwritten) |
| `DOTBLOT_TMP` | system temp folder | scratch space for the corrected working copy of the dataset |

Training (50 epochs, 1,920 px, batch 4) needs a GPU. The dataset folders are read in place and never modified; a corrected working copy is built in the scratch folder and every correction is logged.
`Figure5_Segmentation_Example.ipynb` draws the example figure from the validation image and `best.pt` (`DOTBLOT_WEIGHTS` overrides the weights file).

### 2. Classification (ML and CNN)

Open `Dot_Blot_Model_ML_CNN/Dot-Blot Model.ipynb` from inside that folder.

| Environment variable | Default | Meaning |
|---|---|---|
| `RUN_MODE` | `FULL` | `FAST` = reduced configuration for a quick functional test |
| `WORK_ROOT` | current folder (local) / `MyDrive/Dotblot_GRA7` (Colab) | folder that contains `Dataset/` |
| `DATA_ROOT`, `OUT_DIR` | `<WORK_ROOT>/Dataset`, `<WORK_ROOT>/Results_rerun` (local) | dataset and output folders |
| `N_JOBS` | `-1` | parallel workers for the ML part |

The `FULL` run (5 CNN architectures × 2 protocols × 5 folds, nested tuning, bootstrap and permutation tests) is intended for a GPU; outputs are written to `Results_rerun/` so that the published `Results/` stays intact. `Results/RESULTS_SUMMARY.md` and `Results/run_manifest.json` document the published run (seed 42, 250 lux, per-membrane normalisation).

### 3. Supplementary scripts

Run from `Supplementary_Code/` after the classification results exist in `Dot_Blot_Model_ML_CNN/Results/`. Outputs go to `Supplementary_Code/output/` (`DOTBLOT_RESULTS` and `DOTBLOT_SUPP_OUT` override the locations).

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

**License:** to be chosen by the authors before publication (e.g. MIT for code, CC BY 4.0 for data and figures); add a `LICENSE` file. Until then all rights are reserved.

**Citation:** the reference and DOI of the manuscript, and the Zenodo DOI of the archived release, will be added here.

**Disclaimer.** Research prototype. It is not a validated medical device and must not be used for clinical decisions.
