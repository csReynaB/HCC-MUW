# Survival XGBoost–Cox Pipeline

This repository contains a modular and reproducible pipeline for **survival analysis using gradient-boosted trees (XGBoost) with a Cox proportional hazards objective**.  
It is designed for **biomedical cohorts**, supports **nested cross-validation**, and integrates **model interpretation (SHAP)** and **survival-specific evaluation metrics**.

The scripts have been tested for **HPC/SLURM environments**, and can also be run locally.

---

## 🚀 Features

- **XGBoost Cox survival models**
- **Nested cross-validation** (outer / inner folds)
- **Bayesian hyperparameter optimization** (scikit-optimize)
- **Time-dependent AUC and concordance index**
- **Kaplan–Meier and log-rank testing**
- **SHAP-based feature interpretation**
- HPC-ready **SLURM array execution**

---

## 📁 Repository structure

```text
survival_project/
├── README.md
├── pyproject.toml
├── dockerfile
├── ML_env.yml
├── installations_and_runs/
│   ├── example_run_cli_with_docker.sh
│   ├── example_run_jupyter_with_docker.sh
│   ├── get_docker_container.sh
│   └── install_phipsurv_local.sh
│ 
│ 
├── scripts/
│   ├── run_phipsurv.sh
│
├── src/
│   └── survival/
│       ├── cli/
│       │   └── train_test.py
│       ├── io/
│       │   └── data_handler.py
│       ├── survival/
│       │   └── helpers.py
│       ├── plots/
│       │   ├── metrics_helpers.py
│       │   └── helpers.py
│       └── utils/
│           └── peptides_filter.py
│
├── configs/
│   ├── config.yaml
│
├── data/
│
│
├── notebooks/
│   └── example.ipynb
│
``` 

---

## 🧠 Method overview

The pipeline implements a **Cox proportional hazards model via XGBoost**, allowing non-linear effects and interactions while preserving survival-time censoring.

Key steps:
1. Data loading and preprocessing
2. Feature filtering (prevalence thresholds, optional covariates)
3. Nested cross-validation
4. Bayesian hyperparameter tuning
5. Model fitting and evaluation
6. Survival-specific metrics and plots
7. SHAP-based feature interpretation

---

## ⚙️ Requirements

The pipeline is designed to run in a **conda environment**.

Core dependencies:
- `numpy`, `pandas`, `scipy`
- `scikit-learn`
- `xgboost`
- `scikit-survival`
- `lifelines`
- `shap`
- `matplotlib`, `seaborn`
- `joblib`, `pyyaml`, `tqdm`

Formatting / linting (optional):
- `black`, `isort`, `ruff`

Main dependencies are documented in `pyproject.toml`.  
Indications of how to install package or container are found in installation_and_runs folder.

---

## 🧪 Environment local setup (example)

```bash
micromamba create \
  --yes \
  --name phipsurv \
  --file ML_env.yml

micromamba activate phipsurv

python -m pip install . --no-deps

## Running array in SLURM

sbatch --array=1-10 \
  scripts/run_phipsurv.sh \
  configs/survival.yaml \
  seeds.txt # one random number per line

### Docker installation
docker build -t phipsurv:latest .

## check if install correctly
docker run --rm phipsurv:latest python -m pip show phipsurv

## Run main script
docker run --rm phipsurv:latest python -m phipsurv.cli.train_test  --help
#or
docker run --rm phipsurv:latest phipsurv --help

## save image for exporting 
docker save phipsurv:latest -o phipsurv.tar

### Internally running
python -m phipsurv.cli.train_test -h
#or
phipsurv -h
