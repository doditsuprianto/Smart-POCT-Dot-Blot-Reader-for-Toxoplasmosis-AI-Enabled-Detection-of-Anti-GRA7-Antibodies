# AUTOMATIC RESULTS SUMMARY
Generated: 2026-11-02 08:57 | mode=FULL | selected lux=250 | normalization=mem

## Data
- Images: 1360 (valid: 1296; failed QC: 64); unique spots: 272; sera: 46 (positive 32, negative 14).

## Artefact diagnostic (univariate AUC of crop size @850 lux)
| subset | n_spot | AUC_crop_side | AUC_mean_gray |
|---|---|---|---|
| ALL (M1+M2+M3) | 272 | 0.759 | 0.597 |
| MIX (M2+M3) | 176 | 0.512 | 0.567 |

## Conditional conclusions (automatic)
- **ALL**: best ML model `COMBINED_Q1-Q6+MLP` sample AUC = 0.929 (95% CI 0.839-0.989); best artefact control `CTRL_Shape+ET` AUC = 0.900. Lower CI bound > 0.5 (discrimination above chance). Permutation test (LR): p = 0.010. Exceeds the artefact control.
- **MIX**: best ML model `Q4_Texture+MLP` sample AUC = 0.821 (95% CI 0.598-0.991); best artefact control `CTRL_MaskOnly` AUC = 0.536. Lower CI bound > 0.5 (discrimination above chance). Permutation test (LR): p = 0.238. Exceeds the artefact control.

## ML vs CNN comparison - ALL
| model | family | AUC | AUC_lo | AUC_hi | BACC | SENS | SPEC | F1 | MCC | PR_AUC | Brier | input | params_M | train_s_per_fold | infer_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| COMBINED_Q1-Q6+MLP | ML | 0.929 | 0.839 | 0.989 | 0.835 | 0.812 | 0.857 | 0.867 | 0.631 | 0.972 | 0.110 | 250 lux | nan | nan | nan |
| Q4_Texture+ET | ML | 0.929 | 0.837 | 0.982 | 0.846 | 0.906 | 0.786 | 0.906 | 0.692 | 0.971 | 0.178 | 250 lux | nan | nan | nan |
| Q5_Histogram+kNN | ML | 0.908 | 0.817 | 0.978 | 0.855 | 0.781 | 0.929 | 0.862 | 0.659 | 0.965 | 0.119 | 250 lux | nan | nan | nan |
| FUSION_5lux+XGB | ML | 0.902 | 0.806 | 0.975 | 0.875 | 0.750 | 1.000 | 0.857 | 0.691 | 0.964 | 0.123 | 5 lux (fusi) | nan | nan | nan |
| CTRL_Shape+ET | CONTROL-ML | 0.900 | 0.810 | 0.973 | 0.859 | 0.719 | 1.000 | 0.836 | 0.661 | 0.961 | 0.177 | CTRL_Shape | nan | nan | nan |
| CTRL_Batch+LR | CONTROL-ML | 0.854 | 0.730 | 0.954 | 0.875 | 0.750 | 1.000 | 0.857 | 0.691 | 0.943 | 0.181 | CTRL_Batch | nan | nan | nan |
| SimpleCNN | CNN | 0.596 | 0.411 | 0.775 | 0.654 | 0.594 | 0.714 | 0.691 | 0.283 | 0.737 | 0.244 | all lux | 0.294 | 5.330 | 0.210 |
| ResNet18 | CNN | 0.855 | 0.739 | 0.955 | 0.728 | 0.812 | 0.643 | 0.825 | 0.447 | 0.942 | 0.152 | all lux | 11.171 | 4.954 | 0.232 |
| MobileNetV3S | CNN | 0.708 | 0.525 | 0.877 | 0.632 | 0.906 | 0.357 | 0.829 | 0.320 | 0.833 | 0.202 | all lux | 1.519 | 4.815 | 0.209 |
| EfficientNetB0 | CNN | 0.513 | 0.335 | 0.699 | 0.533 | 0.781 | 0.286 | 0.746 | 0.072 | 0.699 | 0.236 | all lux | 4.008 | 8.398 | 0.346 |
| DenseNet121 | CNN | 0.855 | 0.734 | 0.951 | 0.783 | 0.781 | 0.786 | 0.833 | 0.535 | 0.946 | 0.149 | all lux | 6.949 | 18.293 | 0.863 |
| CTRL_MaskOnly | CONTROL-CNN | 0.862 | 0.750 | 0.951 | 0.844 | 0.688 | 1.000 | 0.815 | 0.633 | 0.948 | 0.203 | mask silhouette | 0.294 | 5.562 | 0.160 |

## ML vs CNN comparison - MIX
| model | family | AUC | AUC_lo | AUC_hi | BACC | SENS | SPEC | F1 | MCC | PR_AUC | Brier | input | params_M | train_s_per_fold | infer_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q4_Texture+MLP | ML | 0.821 | 0.598 | 0.991 | 0.688 | 0.375 | 1.000 | 0.545 | 0.526 | 0.823 | 0.171 | 250 lux | nan | nan | nan |
| Q4_Texture+LR | ML | 0.786 | 0.535 | 0.973 | 0.830 | 0.875 | 0.786 | 0.778 | 0.638 | 0.650 | 0.180 | 250 lux | nan | nan | nan |
| Q4_Texture+ET | ML | 0.732 | 0.500 | 0.929 | 0.491 | 0.125 | 0.857 | 0.182 | -0.025 | 0.521 | 0.229 | 250 lux | nan | nan | nan |
| FUSION_5lux+LR | ML | 0.634 | 0.375 | 0.866 | 0.652 | 0.375 | 0.929 | 0.500 | 0.379 | 0.576 | 0.227 | 5 lux (fusi) | nan | nan | nan |
| CTRL_Shape+LR | CONTROL-ML | 0.527 | 0.295 | 0.786 | 0.536 | 0.500 | 0.571 | 0.444 | 0.069 | 0.488 | 0.249 | CTRL_Shape | nan | nan | nan |
| SimpleCNN | CNN | 0.536 | 0.241 | 0.804 | 0.491 | 0.625 | 0.357 | 0.455 | -0.018 | 0.563 | 0.273 | all lux | 0.294 | 2.659 | 0.166 |
| ResNet18 | CNN | 0.366 | 0.152 | 0.616 | 0.286 | 0.000 | 0.571 | 0.000 | -0.463 | 0.311 | 0.298 | all lux | 11.171 | 3.779 | 0.204 |
| MobileNetV3S | CNN | 0.304 | 0.080 | 0.571 | 0.411 | 0.750 | 0.071 | 0.444 | -0.250 | 0.312 | 0.286 | all lux | 1.519 | 3.564 | 0.229 |
| EfficientNetB0 | CNN | 0.402 | 0.116 | 0.723 | 0.545 | 0.375 | 0.714 | 0.400 | 0.092 | 0.470 | 0.250 | all lux | 4.008 | 5.964 | 0.290 |
| DenseNet121 | CNN | 0.464 | 0.205 | 0.723 | 0.500 | 0.000 | 1.000 | 0.000 | 0.000 | 0.435 | 0.270 | all lux | 6.949 | 11.256 | 0.797 |
| CTRL_MaskOnly | CONTROL-CNN | 0.536 | 0.268 | 0.813 | 0.473 | 0.375 | 0.571 | 0.353 | -0.052 | 0.552 | 0.249 | mask silhouette | 0.294 | 2.571 | 0.161 |

## Paired tests
| protocol | A | B | dAUC_A_minus_B | ci_lo | ci_hi | p_boot | McNemar_only_A_correct | McNemar_only_B_correct | p_McNemar | spearman_scores |
|---|---|---|---|---|---|---|---|---|---|---|
| ALL | ML COMBINED_Q1-Q6+MLP | CNN DenseNet121 | 0.077 | -0.013 | 0.181 | 0.114 | 3.000 | 1.000 | 0.625 | 0.714 |
| ALL | ML COMBINED_Q1-Q6+MLP | CTRL-ML: CTRL_Shape+ET | 0.033 | -0.058 | 0.132 | 0.532 | nan | nan | nan | nan |
| ALL | CNN DenseNet121 | CTRL-ML: CTRL_Shape+ET | -0.044 | -0.156 | 0.051 | 0.392 | nan | nan | nan | nan |
| ALL | ML COMBINED_Q1-Q6+MLP | CNN: CTRL_MaskOnly | 0.070 | -0.018 | 0.172 | 0.116 | nan | nan | nan | nan |
| ALL | CNN DenseNet121 | CNN: CTRL_MaskOnly | -0.007 | -0.109 | 0.101 | 0.906 | nan | nan | nan | nan |
| MIX | ML Q4_Texture+MLP | CNN SimpleCNN | 0.293 | 0.062 | 0.536 | 0.016 | 9.000 | 2.000 | 0.065 | 0.350 |
| MIX | ML Q4_Texture+MLP | CTRL-ML: CTRL_Shape+LR | 0.300 | -0.089 | 0.625 | 0.104 | nan | nan | nan | nan |
| MIX | CNN SimpleCNN | CTRL-ML: CTRL_Shape+LR | 0.007 | -0.473 | 0.447 | 0.962 | nan | nan | nan | nan |
| MIX | ML Q4_Texture+MLP | CNN: CTRL_MaskOnly | 0.293 | -0.063 | 0.661 | 0.108 | nan | nan | nan | nan |
| MIX | CNN SimpleCNN | CNN: CTRL_MaskOnly | -0.000 | -0.322 | 0.366 | 0.974 | nan | nan | nan | nan |

## Dose-response (Spearman score vs log10 IgG, positive sera)
| protocol | model | n_pos | spearman_rho | p |
|---|---|---|---|---|
| ALL | COMBINED_Q1-Q6+MLP | 31 | 0.484 | 0.006 |
| ALL | CTRL_Shape(best) | 31 | 0.472 | 0.007 |
| MIX | Q4_Texture+MLP | 7 | -0.357 | 0.432 |
| MIX | CTRL_Shape(best) | 7 | 0.143 | 0.760 |

## Feature-family contribution
| protocol | model | dropped | sample_AUC | delta_vs_combined |
|---|---|---|---|---|
| ALL | MLP | Q1_Densitometry | 0.886 | -0.033 |
| ALL | MLP | Q2_RelativeOD | 0.889 | -0.030 |
| ALL | MLP | Q3_Radial | 0.873 | -0.046 |
| ALL | MLP | Q4_Texture | 0.858 | -0.061 |
| ALL | MLP | Q5_Histogram | 0.894 | -0.025 |
| ALL | MLP | Q6_Spectrum | 0.893 | -0.026 |
| ALL | MLP | PERM:Q1_Densitometry | nan | -0.031 |
| ALL | MLP | PERM:Q2_RelativeOD | nan | -0.017 |
| ALL | MLP | PERM:Q3_Radial | nan | -0.005 |
| ALL | MLP | PERM:Q4_Texture | nan | -0.119 |
| ALL | MLP | PERM:Q5_Histogram | nan | -0.243 |
| ALL | MLP | PERM:Q6_Spectrum | nan | -0.015 |
| MIX | MLP | Q1_Densitometry | 0.494 | -0.062 |
| MIX | MLP | Q2_RelativeOD | 0.509 | -0.048 |
| MIX | MLP | Q3_Radial | 0.512 | -0.045 |
| MIX | MLP | Q4_Texture | 0.473 | -0.083 |
| MIX | MLP | Q5_Histogram | 0.658 | 0.101 |
| MIX | MLP | Q6_Spectrum | 0.637 | 0.080 |
| MIX | MLP | PERM:Q1_Densitometry | nan | 0.033 |
| MIX | MLP | PERM:Q2_RelativeOD | nan | -0.057 |
| MIX | MLP | PERM:Q3_Radial | nan | 0.021 |
| MIX | MLP | PERM:Q4_Texture | nan | -0.054 |
| MIX | MLP | PERM:Q5_Histogram | nan | 0.044 |
| MIX | MLP | PERM:Q6_Spectrum | nan | 0.020 |
