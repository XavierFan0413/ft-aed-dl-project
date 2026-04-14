# FT-AED Deep Learning Project

This project studies anomalous event detection on freeway sensor time series using the FT-AED dataset. The task is to predict rare anomalous events (`human_label`) from lane-level traffic measurements under severe class imbalance. The project is implemented in PyTorch and currently includes four model variants: a baseline MLP, a GRU, a GRU with imbalance-aware sampling, and a TCN with imbalance-aware sampling.

## Task

This project is framed as imbalanced supervised time-series classification.

- Primary target label: `human_label`
- Split strategy: time-based split only
- Baseline model: per-timestep MLP
- Temporal models: GRU, GRU + sampler, TCN + sampler
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
- `src/` — preprocessing, training, evaluation, and analysis code
- `results/` — saved model outputs, metrics, plots, and comparison tables
- `docs/` — proposal and report-related materials

## Current implemented pipeline

- dataset inspection
- time-based train / validation / test split
- baseline feature preprocessing
- baseline MLP training
- GRU sequence window construction
- GRU training
- GRU training with weighted sampling for imbalance handling
- TCN training with weighted sampling for imbalance handling
- saved weights, metrics, and plots for all trained models
- reproducible saved-model evaluation without retraining
- threshold sweep comparison across multiple fixed thresholds

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

4. Build sequence windows for the temporal models

`python src/data_loader.py`

This creates:

- `data/processed/gru_train_w10.npz`
- `data/processed/gru_val_w10.npz`
- `data/processed/gru_test_w10.npz`
- `data/processed/gru_metadata_w10.json`

5. Train the plain GRU model

`python src/train_gru.py`

This saves outputs under `results/gru/`.

6. Train the GRU with weighted sampling

`python src/train_gru_sampler.py`

This saves outputs under `results/gru_sampler/`.

7. Train the TCN with weighted sampling

`python src/train_tcn_sampler.py`

This saves outputs under `results/tcn_sampler/`.

8. Reproduce saved results without retraining

`python -m src.evaluate --model baseline`

`python -m src.evaluate --model gru`

`python -m src.evaluate --model gru_sampler`

`python -m src.evaluate --model tcn_sampler`

This saves reproduced metrics and PR curves for each model.

9. Run threshold sweep analysis

`python src/threshold_sweep.py`

This saves comparison files under `results/comparison/`, including:

- `baseline_threshold_sweep.csv`
- `gru_threshold_sweep.csv`
- `gru_sampler_threshold_sweep.csv`
- `tcn_sampler_threshold_sweep.csv`
- `threshold_sweep_comparison.csv`
- `threshold_sweep_best_f1_summary.csv`

## Current status

The end-to-end project pipeline is working. All implemented models can be trained successfully, and saved checkpoints can be reloaded to reproduce evaluation outputs without retraining.

The project now includes four model variants with different tradeoffs:

- The baseline MLP provides a simple per-timestep reference point.
- The plain GRU adds temporal context but shows unstable threshold behavior at the validation-chosen threshold.
- The GRU with weighted sampling improves recall and gives the best thresholded F1 among the current variants.
- The TCN with weighted sampling gives the strongest test AP among the current variants.

Overall performance is still modest in absolute terms because the dataset is extremely imbalanced and operating-threshold selection is unstable. However, the experiments now provide a clearer comparison of how temporal modeling and imbalance handling affect anomaly detection behavior.

## Notes

- Framework used: PyTorch
- Main reported label: `human_label`
- The dataset is excluded from the repository and should be downloaded separately
- Generated `.npz` artifacts are not tracked in git
- Final deliverables include the report, code, saved outputs, and reproducibility instructions