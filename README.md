# Machine learning for differentiating influenza types using routine parameters among outpatients

Analysis code accompanying the manuscript:

> **Machine Learning for Differentiating Influenza Types Using Routine Parameters Among Outpatients:
> A Multicenter Retrospective Study**
> Xiefei Hu, Chunmei Duan, Jiangtao Mou, Jun Liu, Qiubin Wan, Cui Fan, Huajian Chen, Xun Li, Gen Luo,
> Hongshu Fang, Yongquan Lu, Jianda Liu, Shenshen Zhi\*, Wei Li\*
> *BMC Medical Informatics and Decision Making* (under revision)

This repository contains the complete analysis pipeline used in the study: preprocessing and data-quality
screening, model selection and hyperparameter tuning, training of the final LightGBM model, external
validation in three independent cohorts, interpretability analyses (SHAP), sensitivity analyses
(month-of-presentation ablation, feature-encoding checks, correlation checks), bootstrap comparisons and
decision-curve analysis.

**No patient data are included in this repository.** The datasets are de-identified clinical records held by
the participating hospitals and cannot be shared publicly; see *Data requirements* below.

---

## 1. Environment

The analyses reported in the manuscript were run on **Python 3.11.5** with the package versions listed in
`requirements.txt` (scikit-learn 1.7.2, lightgbm 4.2.0, xgboost 2.0.3, shap 0.44.1, numpy 1.23.5,
pandas 2.0.3, scipy 1.11.1, matplotlib 3.7.2, seaborn). All procedures use a fixed random seed
(`random_state = 42`).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The sensitivity checks added during revision (in-fold preprocessing re-run, month-encoding comparison) were
also reproduced with newer library versions (scikit-learn 1.9.1, lightgbm 4.7.0) and gave the same results.

## 2. Data requirements

Every script reads its inputs from one directory, taken from the environment variable `INFLUENZA_DATA_DIR`
(default: the path used on the authors' machine). Point it at the folder that contains the four de-identified
cohort files:

```bash
export INFLUENZA_DATA_DIR=/path/to/cohort_data      # Windows (cmd): set INFLUENZA_DATA_DIR=D:\cohort_data
```

Expected files (one row per patient; column names as documented in `DATA_DICTIONARY.md`):

| File | Cohort | n | Role |
|---|---|---|---|
| `a附三院数据_processed.csv` | Third Affiliated Hospital of Chongqing Medical University (TAH) | 11,358 | derivation cohort |
| `b新桥数据_processed.csv` | Second Affiliated Hospital of Army Medical University (SAH) | 5,686 | external validation |
| `c急救数据.csv` | Chongqing Emergency Medical Center (CEMC) | 5,236 | external validation |
| `d丰都数据_processed.csv` | Fengdu General Hospital (FDH) | 3,411 | external validation |

The scripts write their outputs below `<INFLUENZA_DATA_DIR>/代码存档/运行结果/<analysis>/`. De-identified
aggregate datasets can be requested from the corresponding author (liwei0111@cqu.edu.cn); requests are
reviewed by the Institutional Review Board of Chongqing Emergency Medical Center.

## 3. Running order

| # | Script | Purpose | Manuscript output |
|---|---|---|---|
| 01 | `01_data_preprocessing.py` | column alignment, KNN imputation (k = 5), plausibility-range screening | eTable 2 |
| 02 | `02_model_selection_grid_search.py` | six algorithms, grid search, 5-fold CV, 70:30 split | eTable 6, eTable 7 |
| 03 | `03_train_final_lightgbm.py` | final LightGBM model + serialized model/preprocessor | final model |
| 04 | `04_external_validation.py` | external validation in SAH, CEMC, FDH (+ predictions) | eTable 8, eFigure 1 |
| 05 | `05_calibration_pr_brier.py` | calibration curves, PR curves, Brier scores | Figure 2E–H, eTable 9 |
| 06 | `06_leave_one_center_out.py` | leave-one-centre-out derivation comparison | eFigure 1 |
| 07 | `07_shap_analysis.py` | SHAP values for all cohorts with the final model | Figure 3, eFigures 3–5, eTable 15 |
| 08 | `08_external_metrics_from_predictions.py` | class-wise operating characteristics, confusion matrices | eTable 10, eTable 11 |
| 09 | `09_binary_any_influenza_metrics.py` | binary any-influenza vs non-influenza task | eTable 13 |
| 10 | `10_month_ablation.py` | month-only / CBC+age+sex / full-model ablation | eTable 16 |
| 11 | `11_plausibility_range_screening.py` | physiological plausibility rules vs ±3 SD screening | eTable 2, Panels B and C |
| 12 | `12_in_fold_pipeline_rerun.py` | preprocessing fitted inside each CV fold (leakage check) | response to reviewers |
| 13 | `13_paired_bootstrap.py` | paired bootstrap comparison of LightGBM vs XGBoost vs RF | Discussion |
| 14 | `14_decision_curve_analysis.py` | decision-curve analysis (net benefit) | eTable 17, eFigure 6 |
| 15 | `15_wbc_correlation.py` | correlation structure of the CBC features | response to reviewers |
| 16 | `16_class_level_youden.py` | class-level Youden-optimal thresholds | eTable 10 |
| 17 | `17_month_encoding_sensitivity.py` | ordinal vs cyclical vs categorical encoding of month | Methods |

Scripts 01–14 were used for the originally submitted analyses; 12 and 17 were added during revision, and
11, 15 and 16 document the data-quality and sensitivity checks.

## 4. Methodological points worth noting

* **No ±3 SD winsorization.** Values outside pre-specified *physiological* plausibility ranges are treated as
  data-entry errors and set to missing (14 of 25,691 × 23 screened cells, 0.002%); extreme but plausible
  values are retained, because gradient-boosted trees split on rank order. See `11_plausibility_range_screening.py`.
* **No class weighting or resampling.** All six algorithms are selected on macro-averaged AUROC, which
  weights the three outcome classes equally.
* **The three-class decision rule** is the class with the highest predicted probability; class-wise
  performance is additionally reported at the 0.5 and Youden-optimal thresholds.
* **Preprocessing inside folds.** `12_in_fold_pipeline_rerun.py` re-runs the whole workflow with scaling and
  encoding embedded in the cross-validation pipeline; the external macro-AUROCs are unchanged (Δ ≤ 0.002).
* **SHAP** values are computed with the final model trained on the derivation cohort and applied unchanged
  to each external cohort; no centre-specific model is retrained.

## 5. Licence and citation

Released under the MIT Licence (`LICENSE`). If you use this code, please cite the manuscript above.
Questions: Wei Li (liwei0111@cqu.edu.cn) or Shenshen Zhi (zss179394047@cqu.edu.cn).
