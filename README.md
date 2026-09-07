# Explainable Credit Risk & Early Warning Decision System

An end-to-end machine learning system for retail credit-risk assessment,
probability calibration, cost-sensitive decisioning, explainability,
expected-loss estimation, risk segmentation, and model monitoring.

## Project Overview

The project develops a credit-risk decision framework using the public
**Home Credit Default Risk** dataset.

The system covers:

- Exploratory data analysis and business-oriented feature engineering
- Multiple credit-risk classification models
- XGBoost champion model selection
- Probability calibration using isotonic regression
- Cost-sensitive threshold optimisation
- SHAP-based applicant-level explanations
- Expected Loss estimation using PD × LGD × EAD
- Risk segmentation and decision rules
- New-applicant decisioning: APPROVE / MANUAL REVIEW / DECLINE
- Existing-customer Early Warning System (EWS) monitoring
- PSI and KS-based monitoring
- Portfolio-level risk dashboard
- Applicant-level explanation reports

## Model Performance

Final performance is reported on the held-out test set:

| Metric | Result |
|---|---:|
| ROC-AUC | 0.7684 |
| PR-AUC | 0.2476 |
| Brier Score | 0.0672 |

The probability threshold was selected on the calibration set using
a cost-sensitive objective with:

- False Negative cost = 5
- False Positive cost = 1

Optimal threshold:

**0.1676**

Risk tiers are anchored to this decision threshold.

## Explainability

SHAP is used as a post-hoc explainability layer to identify the
features contributing to individual applicant risk predictions and
to provide portfolio-level feature importance.

## Expected Loss

Expected Loss is estimated as:

**EL = PD × LGD × EAD**

For demonstration purposes, LGD is set to an illustrative 45% assumption.
EAD is proxied using the application credit amount available in the
public dataset.

## Monitoring

The notebook includes PSI and KS monitoring using simulated
score-distribution periods derived from the held-out test predictions.

These monitoring results should be interpreted as a demonstration of
the monitoring framework rather than genuine longitudinal production
stability monitoring.

## Dataset

This project uses the public **Home Credit Default Risk** dataset
available through Kaggle.

The raw dataset is **not included** in this repository.

To reproduce the notebook, download the dataset separately and place
the required files in the expected data directory.

## Repository Structure

```text
credit-risk-ews/
│
├── credit_risk_ews.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
├── figures/
│   ├── eda_business_findings.png
│   ├── roc_pr_curves.png
│   ├── calibration_analysis.png
│   ├── threshold_optimisation.png
│   ├── shap_beeswarm.png
│   ├── shap_bar.png
│   ├── shap_dependence.png
│   ├── shap_waterfall_high_risk.png
│   ├── shap_waterfall_low_risk.png
│   ├── expected_loss_analysis.png
│   ├── decision_outcomes.png
│   ├── portfolio_dashboard.png
│   └── model_monitoring.png
│
├── models/
│   ├── xgb_credit_default_raw.pkl
│   └── xgb_credit_default_calibrated.pkl
│
└── reports/
    ├── model_summary.json
    └── applicant_scores.csv

Disclaimer

This is a portfolio/academic project based on a public dataset.
It does not contain confidential or proprietary banking data and
should not be interpreted as an actual production credit decision
system.
