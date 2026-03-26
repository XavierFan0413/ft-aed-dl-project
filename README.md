# FT-AED Deep Learning Project

Deep learning project for anomalous event detection on freeway sensor time series (FT-AED).

## Task
This project is framed as imbalanced supervised time-series classification.

## Current frozen decisions
- Target label: `human_label`
- Split strategy: time-based split only
- Baseline model: MLP per timestep
- Main model: GRU
- Primary metrics: Precision, Recall, F1, PR-AUC

## Planned structure
- `data/` for dataset notes and local data folders
- `notebooks/` for exploration
- `src/` for preprocessing, training, and evaluation code
- `results/` for saved outputs
- `docs/` for proposal and report-related materials
