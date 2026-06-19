# Youth Debt Crisis DSR Prediction: An XAI-Driven Regression Framework

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3.0-green.svg)](https://lightgbm.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.44.1-red.svg)](https://shap.readthedocs.io/)
[![DiCE](https://img.shields.io/badge/DiCE--ML-0.12-purple.svg)](https://interpret.ml/DiCE/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## Overview

This repository contains the replication code for the paper:

> **Anonymous (under review). "An XAI-Driven Early Warning and Policy Prescription Framework for Youth Debt Crisis: A Regression Approach Using Korean Welfare Panel Data." *Anonymous Journal*.**

This paper extends a prior binary classification framework (DSR ≥ 40% as crisis indicator) to a **continuous regression framework** in which the debt service ratio (DSR) is modelled directly as a continuous outcome. Using the 19th wave (2024) of the Korea Welfare Panel Study (KOWEPS, n = 1,916 youth aged 19–39), we train a LightGBM–XGBoost ensemble regressor, apply SHAP-based feature attribution, generate DiCE counterfactual explanations under a desired DSR range, and simulate the heterogeneous effects of debt relief and income support policies across employment status subgroups.

---

## Key Findings

- The ensemble model achieves a cross-validated R² of approximately 0.54 and test-set R² of 0.60 using only publicly available welfare panel data.
- SHAP analysis identifies **financial institution loan** and **temporary wage income** as the two dominant predictors of DSR, consistent across five random seeds and three percentile-based case definitions.
- The relationship between temporary wage income and DSR is **non-monotonic**: low income is associated with higher DSR, but the direction reverses above the median (≈14.4 million KRW annually), suggesting that income stability rather than income level is the primary protective factor.
- DiCE counterfactual explanations achieve 100% coverage with a mean of 3.20 feature changes per case. This rate is maintained even under strict actionability constraints that exclude non-policy-relevant features (age, sex, region, education, marital status).
- Policy simulation reveals substantial heterogeneity: **debt relief (Policy A) is optimal for employed youth**; **combined income and debt support (Policy C) is optimal for unemployed youth** (DSR reduction = 0.162, 95% CI [0.075, 0.260]).
- Low-magnitude income support (B-20, B-40) paradoxically worsens predicted DSR for employed youth, a finding attributable to the non-monotonic income–DSR relationship.
- All main findings are robust to the choice of high-risk case definition (top 10%/20%/30% of predicted DSR).

---

## Repository Structure

```
youth-debt-xai-regression/
│
├── data/                          # Data directory (raw data not included)
│   ├── youth_debt_preprocessed_regression.csv
│   ├── X_train_reg.csv
│   ├── X_test_reg.csv
│   ├── y_train_reg.csv
│   ├── y_test_reg.csv
│   ├── feature_cols_reg.csv
│   └── y_pred_ens_reg.csv
│
├── outputs/                       # Model files and result outputs
│   ├── model_lgbm_reg.pkl
│   ├── model_xgb_reg.pkl
│   ├── imputer_reg.pkl
│   ├── shap_importance_reg.csv
│   ├── dice_cf_summary_reg.csv
│   ├── dice_cf_diversity_reg.csv
│   ├── policy_simulation_summary_reg.csv
│   ├── bootstrap_ci_model_reg.csv
│   ├── bootstrap_ci_policy_reg.csv
│   ├── bootstrap_ci_subgroup_reg.csv
│   └── [figures 01–40: .png]
│
├── notebooks/
│   ├── 01_preprocessing_regression.py
│   ├── 02_eda_regression.py
│   ├── 03_model_regression.py
│   ├── 04_shap_analysis_regression.py
│   ├── 05_dice_counterfactual_regression.py
│   ├── 05b_dice_diversity_regression.py
│   ├── 05c_case_comparison_regression.py
│   ├── 06_policy_simulation_regression.py
│   ├── 07_sensitivity_analysis_regression.py
│   ├── 08_bootstrap_ci_regression.py
│   ├── 09_heterogeneity_extended_regression.py
│   ├── 10_shap_nonlinear_cf_actionability_regression.py
│   └── 11_dsr_percentile_robustness_regression.py
│
├── regression_results_summary.md  # Full results summary with paper notes
├── requirements.txt
├── requirements_dice_cf.txt       # Separate environment for DiCE
└── README.md
```

---

## Data

This study uses the **19th wave (2024) of the Korea Welfare Panel Study (KOWEPS)**, jointly conducted by the Korea Institute for Health and Social Affairs (KIHASA) and Seoul National University.

- The raw data file (`koweps_hpc19_2024_beta2.dta`) is **not included** in this repository due to data access restrictions.
- The data can be obtained from the [KOWEPS website](https://www.koweps.re.kr) upon registration.
- Once the raw data file is placed in the `data/` directory, running the notebooks in order (01 → 11) will reproduce all results.

The analytical sample is restricted to youth aged 19–39 with positive reported income (n = 1,916). The regression target is the household debt service ratio (DSR), winsorised at 500%.

---

## Environment Setup

Two separate conda environments are required due to package conflicts between DiCE and other dependencies.

### Main environment (notebooks 01–04, 06–11)

```bash
conda create -n youth-debt python=3.10 -y
conda activate youth-debt

pip install numpy==1.26.4
pip install pandas scikit-learn matplotlib seaborn
pip install lightgbm==4.3.0
pip install xgboost==2.0.3
pip install shap==0.44.1
pip install scipy joblib Pillow
```

### DiCE environment (notebooks 05, 05b, 05c, 07 Part C, 10 Part B, 11 Part C)

DiCE has package conflicts with the main environment and requires a separate conda environment.

```bash
conda create -n dice_cf python=3.10 -y
conda activate dice_cf

pip install numpy==1.26.4
pip install pandas scikit-learn matplotlib seaborn
pip install lightgbm==4.3.0
pip install xgboost==2.0.3
pip install shap==0.44.1
pip install dice-ml==0.12
pip install click scipy joblib Pillow

# Register Jupyter kernel
pip install ipykernel
python -m ipykernel install --user --name dice_cf --display-name "Python (dice_cf)"
```

### requirements.txt

```
numpy==1.26.4
pandas>=2.0
scikit-learn==1.3.2
lightgbm==4.3.0
xgboost==2.0.3
shap==0.44.1
scipy>=1.11
joblib>=1.3
matplotlib>=3.7
seaborn>=0.12
Pillow>=9.0
```

### requirements_dice_cf.txt

```
numpy==1.26.4
pandas>=2.0
scikit-learn==1.3.2
lightgbm==4.3.0
xgboost==2.0.3
shap==0.44.1
dice-ml==0.12
click
scipy>=1.11
joblib>=1.3
matplotlib>=3.7
seaborn>=0.12
Pillow>=9.0
```

---

## Notebook Execution Order

All notebooks should be run in sequence. Notebooks 05, 05b, 05c, 07 (Part C only), 10 (Part B only), and 11 (Part C only) require the `alibi_cf` kernel.

| # | Notebook | Kernel | Description |
|---|---|---|---|
| 01 | preprocessing_regression | youth-debt | Data loading, cleaning, DSR construction |
| 02 | eda_regression | youth-debt | Exploratory data analysis |
| 03 | model_regression | youth-debt | LightGBM + XGBoost ensemble training |
| 04 | shap_analysis_regression | youth-debt | SHAP feature attribution |
| 05 | dice_counterfactual_regression | dice_cf | DiCE CF1 generation |
| 05b | dice_diversity_regression | dice_cf | DiCE CF1/CF2/CF3 diversity analysis |
| 05c | case_comparison_regression | dice_cf | Case-level CF comparison tables |
| 06 | policy_simulation_regression | youth-debt | Policy simulation and heterogeneity |
| 07 | sensitivity_analysis_regression | dice_cf* | Seed robustness, SHAP stability, CF range |
| 08 | bootstrap_ci_regression | youth-debt | Bootstrap 95% CI for all estimates |
| 09 | heterogeneity_extended_regression | youth-debt | Full policy × subgroup heterogeneity |
| 10 | shap_nonlinear_cf_actionability | dice_cf* | SHAP non-linearity + CF actionability |
| 11 | dsr_percentile_robustness_regression | dice_cf* | Percentile-based robustness check |

\* Only the DiCE-related cells in notebooks 07, 10, and 11 require the `dice_cf` kernel. All other cells can be run in the `youth-debt` kernel.

---

## Analytical Framework

The analysis proceeds in three stages:

```
Stage 1: Early Warning Model
  └─ LightGBM + XGBoost regression ensemble
     └─ Target: continuous DSR
     └─ Metrics: RMSE, MAE, R² (+ 95% bootstrap CI)

Stage 2: XAI Explanation
  ├─ SHAP (TreeExplainer)
  │   ├─ Global: beeswarm, bar, dependence plots
  │   ├─ Local: waterfall plots (high-DSR cases)
  │   └─ Non-linearity: inflection analysis
  └─ DiCE (regression, desired_range mode)
      ├─ CF1: primary counterfactual (sparsity focus)
      ├─ CF1/CF2/CF3: diversity analysis
      └─ Actionability-constrained CF

Stage 3: Policy Simulation
  ├─ 7 scenarios: A-20/40/60, B-20/40/60, C (combined)
  ├─ Dose-response curves
  ├─ Subgroup heterogeneity (employment status)
  └─ Percentile-based robustness check
```

---

## Key Software Versions

| Package | Version |
|---|---|
| Python | 3.10 |
| LightGBM | 4.3.0 |
| XGBoost | 2.0.3 |
| SHAP | 0.44.1 |
| DiCE-ML | 0.12 |
| scikit-learn | 1.3.2 |
| NumPy | 1.26.4 |
| pandas | ≥ 2.0 |
| SciPy | ≥ 1.11 |

---

## Note on NiCE

The original classification paper used NiCE (Brughmans et al., 2024) as a second counterfactual method for cross-method comparison. NiCE is not applicable in this regression extension because it relies on `predict_proba` and class-label-based nearest-unlike-neighbour search, which are not defined for continuous regression targets. DiCE's `desired_range` parameter provides an analogous mechanism for specifying target outcome intervals in the regression setting.

---

## Citation

If you use this code, please cite:

```bibtex
@article{anonymous2025youth,
  title   = {An XAI-Driven Early Warning and Policy Prescription Framework
             for Youth Debt Crisis: A Regression Approach Using Korean
             Welfare Panel Data},
  author  = {Anonymous},
  journal = {Anonymous Journal},
  year    = {under review}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

This research uses data from the Korea Welfare Panel Study (KOWEPS), jointly conducted by the Korea Institute for Health and Social Affairs (KIHASA) and Seoul National University. The authors thank the data providers for making these data publicly available.
