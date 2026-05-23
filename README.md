# Insurance Risk Analytics – ACIS

Predictive risk modelling and pricing optimisation for AlphaCare Insurance Solutions (ACIS), using 18 months of historical South African auto-insurance claim data (Feb 2014 – Aug 2015).

## Business Context

ACIS is preparing for aggressive growth in the SA auto-insurance market. This project provides the analytical foundation for:

- Identifying **low-risk client segments** where premiums can be reduced to attract new business.
- Statistically validating **risk hypotheses** across provinces, zip codes, and demographics.
- Building **predictive models** for claim probability and severity that power dynamic, risk-based pricing.

## Project Structure

```
insurance-risk-analytics/
├── .github/workflows/ci.yml   # GitHub Actions CI (lint + test on every push)
├── data/                      # Tracked by DVC — not committed to Git
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis (Task 1)
│   ├── 02_hypothesis_testing.ipynb  # A/B hypothesis tests (Task 3)
│   └── 03_modeling.ipynb      # Predictive modelling (Task 4)
├── src/
│   ├── data_loader.py         # DataLoader class — load, validate, enrich
│   ├── eda_utils.py           # EDA helper functions and plots
│   ├── hypothesis_tests.py    # Statistical test utilities (Task 3)
│   └── modeling.py            # Model training and evaluation (Task 4)
├── reports/
│   └── final_report.md        # Medium-style business report
├── tests/                     # pytest test suite
├── .dvc/                      # DVC config (committed)
├── dvc.yaml                   # DVC pipeline stages
├── requirements.txt
└── README.md
```

## Key Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Loss Ratio** | `TotalClaims / TotalPremium` | Core measure of portfolio profitability |
| **Margin** | `TotalPremium - TotalClaims` | Per-policy profit contribution |
| **Claim Frequency** | `count(claims > 0) / count(policies)` | Probability of a claim occurring |
| **Claim Severity** | `mean(TotalClaims | TotalClaims > 0)` | Average cost when a claim occurs |

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/<your-username>/insurance-risk-analytics.git
cd insurance-risk-analytics
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Restore the dataset via DVC

```bash
dvc pull
```

This downloads the raw data to `data/` from the configured remote storage. See Task 2 for DVC setup details.

### 3. Run tests

```bash
pytest tests/ -v
```

### 4. Launch notebooks

```bash
jupyter notebook notebooks/01_eda.ipynb
```

## Tasks

| Task | Branch | Status | Description |
|------|--------|--------|-------------|
| Task 1 | `task-1` | In progress | Git setup + EDA |
| Task 2 | `task-2` | Pending | DVC data versioning |
| Task 3 | `task-3` | Pending | A/B hypothesis testing |
| Task 4 | `task-4` | Pending | Predictive modelling & SHAP |

## CI/CD

GitHub Actions runs on every push to any branch:

1. `flake8` — syntax errors and undefined names (hard fail).
2. `flake8` — style warnings, max line length 120 (soft fail, exit-zero).
3. `pytest tests/` — unit tests for `src/` modules.

## DVC Data Pipeline

See `dvc.yaml` for pipeline stage definitions. To reproduce:

```bash
dvc repro
```

## Data

The dataset covers car-insurance policy, client, vehicle, and claim information. It is pipe-delimited (`|`) and stored at `data/MachineLearningRating_v3.txt` (DVC-tracked, not in Git).

**Field groups:** Policy · Transaction · Client · Location · Vehicle · Plan · Payment & Claim

## References

- [FSRAO Insurance Sector Resources](https://www.fsca.co.za)
- [DVC Documentation](https://dvc.org/doc)
- [SHAP Documentation](https://shap.readthedocs.io)
