# License of data, figures and results

Unless stated otherwise, the **data, figures, tables and documentation** of this repository are licensed under the
**Creative Commons Attribution 4.0 International License (CC BY 4.0)**:
<https://creativecommons.org/licenses/by/4.0/legalcode>

Covered: grayscale spot crops (`Dot_Blot_Model_ML_CNN/Dataset/`), annotated membrane photographs and labels
(`Dot_Blot_YOLO_Segmentation/dataset/`), result tables T01–T20 and figures F01–F20 (`Dot_Blot_Model_ML_CNN/Results/`,
`Dot_Blot_YOLO_Segmentation/yolo_results/`, `figures/`), the trained model files (`*.pt`, `*.joblib`, `*.pkl`), the enclosure
models (`Body_Case/`) and the README files.

You are free to share and adapt the material for any purpose, including commercially, provided that you give appropriate
credit (cite the manuscript and/or this repository, see `CITATION.cff`), link to the license and indicate if changes were
made.

The **source code** (`*.py`, `*.ipynb`, `*.ui`, `requirements*.txt`) is licensed under the MIT License (see `LICENSE`).

## Third-party software

This repository depends on open-source packages under their own licenses. Note in particular that **Ultralytics YOLO
(`ultralytics`) is distributed under AGPL-3.0** (or an Ultralytics Enterprise License). The YOLOv8n-seg weights in this
repository were trained with it, and the application imports it; anyone who redistributes or deploys a derivative work
(e.g. as a network service or a product) has to comply with the Ultralytics license terms: <https://www.ultralytics.com/license>.

## Samples and privacy

The images derive from archived human sera that were de-identified before imaging. Do not attempt to re-identify donors.
