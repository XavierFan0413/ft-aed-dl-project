# FT-AED Deep Learning Project

This project studies anomalous event detection on freeway sensor time series using the FT-AED dataset. The task is to predict rare anomalous events (`human_label`) from lane-level traffic measurements under severe class imbalance. The project is implemented in PyTorch and currently includes a baseline MLP and a GRU-based temporal model.

## Task

This project is framed as imbalanced supervised time-series classification.

- Primary target label: `human_label`
- Split strategy: time-based split only
- Baseline model: per-timestep MLP
- Main model: GRU sequence classifier
- Primary metrics: Precision, Recall, F1, PR-AUC

## Dataset

Dataset page:
`https://acoursey3.github.io/ft-aed/`

Dataset repository:
`https://github.com/acoursey3/freeway-anomaly-data`

Download the file `nashville_freeway_anomaly.csv` and place it at:

`data/raw/nashville_freeway_anomaly.csv`

The dataset itself is not included in this repository.

## Repository structure

- `data/` — dataset notes, raw data folder, processed outputs
- `notebooks/` — optional exploration notebooks
- `src/` — preprocessing, training, and evaluation code
- `results/` — saved model outputs, metrics, and plots
- `docs/` — proposal and report-related materials

## Current implemented pipeline

- dataset inspection
- time-based train / validation / test split
- baseline feature preprocessing
- baseline MLP training
- GRU sequence window construction
- GRU training
- saved weights, metrics, and plots for both models

## Environment setup

Install dependencies with:

`pip install -r requirements.txt`

## How to run

1. Inspect the dataset schema and label distribution

`python src/inspect_ftaed.py`

2. Preprocess flat features for the baseline model

`python src/preprocess.py`

This creates:

- `data/processed/baseline_data.npz`
- `data/processed/baseline_metadata.json`

3. Train the baseline MLP

`python src/train_baseline.py`

This saves outputs under `results/baseline/`, including:

- `best_model.pt`
- `metrics.json`
- `history.json`
- `loss_curve.png`
- `val_ap_curve.png`
- `test_pr_curve.png`
- `test_outputs.npz`

4. Build sequence windows for the GRU

`python src/data_loader.py`

This creates:

- `data/processed/gru_train_w10.npz`
- `data/processed/gru_val_w10.npz`
- `data/processed/gru_test_w10.npz`
- `data/processed/gru_metadata_w10.json`

5. Train the GRU model

`python src/train_gru.py`

This saves outputs under `results/gru/`, including:

- `best_model.pt`
- `metrics.json`
- `history.json`
- `loss_curve.png`
- `val_ap_curve.png`
- `test_pr_curve.png`
- `test_outputs.npz`

## Reproduce saved results

To reproduce evaluation metrics and PR curves from saved checkpoints without retraining:

`python -m src.evaluate --model baseline`

`python -m src.evaluate --model gru`

## Current status

The end-to-end project pipeline is working. Both the baseline MLP and the GRU model can be trained successfully, and both produce saved weights, metrics, and plots.

At the current stage, model performance is still modest. The dataset is highly imbalanced, and threshold selection is unstable because the anomaly class is extremely rare. This will be discussed further in the final report.

## Notes

- Framework used: PyTorch
- Main reported label: `human_label`
- The dataset is excluded from the repository and should be downloaded separately
- Final deliverables will include the report, code, saved outputs, and reproducibility instructions