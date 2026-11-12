# FT-AED: Freeway Anomaly Detection

Can lane-level freeway sensor readings identify rare traffic anomalies? This PyTorch project compares a per-timestep MLP with three temporal configurations on FT-AED, using chronological train, validation, and test splits. The held-out days contain 98 positives in both the 188,160-reading and 186,200-window test sets.

## Held-out test results

| Configuration | Test AP | Validation cutoff | Test F1 at that cutoff |
| --- | ---: | ---: | ---: |
| [MLP baseline](results/baseline/metrics.json) | 0.000621 | 0.59 | 0.001109 |
| [GRU](results/gru/metrics.json) | 0.000614 | 0.81 | 0.000000 |
| [GRU + weighted sampling](results/gru_sampler/metrics.json) | 0.000578 | 0.99 | **0.001175** |
| [TCN + weighted sampling](results/tcn_sampler/metrics.json) | **0.000891** | 0.99 | 0.001143 |

AP is scikit-learn average precision, a recall-weighted summary of precision across test-score cutoffs. For each saved checkpoint, the cutoff that maximized validation F1 on the 0.01–0.99 grid was applied to the test scores to report test F1. The weighted TCN leads on AP and the weighted GRU leads on F1 at the validation-selected cutoffs.

[Final report](docs/ft_aed_dl_report_final.pdf) · [Presentation](docs/Deep_Learning_Project_Presentation.pdf) · [Validation AP curves](results/final_figures/figure1_val_ap_grid.png) · [Exploratory threshold sweep](#exploratory-test-threshold-sweep)

## Reproduce the saved MLP checkpoint

Download the [FT-AED CSV](https://github.com/acoursey3/freeway-anomaly-data/blob/main/nashville_freeway_anomaly.csv) from the upstream GitHub file page. Use its download control to get the Git LFS file and save it as `data/raw/nashville_freeway_anomaly.csv`. From the repository root, in a Python environment, run:

```bash
python -m pip install -r requirements.txt
python src/preprocess.py
python -m src.evaluate --model baseline --output-dir reproduced/baseline
```

The checkpoint is included in the repository. The command writes metrics, a precision–recall plot, and predictions to `reproduced/baseline/`.

### Reproduce the temporal checkpoints

Use the same CSV and dependencies. The loader builds 10-step windows for the three temporal configurations.

```bash
python src/data_loader.py
python -m src.evaluate --model gru --output-dir reproduced/gru
python -m src.evaluate --model gru_sampler --output-dir reproduced/gru_sampler
python -m src.evaluate --model tcn_sampler --output-dir reproduced/tcn_sampler
```

Evaluation uses CUDA when available and also supports CPU. Each run produces `reproduced_metrics.json`, `reproduced_test_pr_curve.png`, and `reproduced_test_outputs.npz` in its output directory.

## Exploratory test threshold sweep

The [threshold-sweep figure](results/final_figures/figure2_threshold_vs_f1.png) plots F1 on the **test set** at cutoffs from 0.10 through 0.90 in steps of 0.10, plus each model's validation-selected cutoff. The maxima below were selected after inspecting test F1. They describe sensitivity to the cutoff, and the held-out comparison above uses the validation-selected cutoffs.

| Configuration | Cutoff with highest F1 in test sweep | Highest F1 in test sweep |
| --- | ---: | ---: |
| MLP baseline | 0.70 | 0.001527 |
| GRU | 0.30 | 0.001311 |
| GRU + weighted sampling | 0.80 | 0.001390 |
| TCN + weighted sampling | 0.50 | 0.001344 |

To regenerate the sweep, copy each `reproduced/<model>/reproduced_test_outputs.npz` into its matching `results/<model>/` directory. Then run:

```bash
python src/threshold_sweep.py
python src/make_figure2_threshold_plot.py
```

The sweep writes CSVs under `results/comparison/`, including `threshold_sweep_comparison.csv` and `threshold_sweep_best_f1_summary.csv`. The plotting command rewrites the tracked threshold-sweep figure.

## Data and training

The [FT-AED dataset page](https://acoursey3.github.io/ft-aed/) describes the traffic measurements and annotations. The target is `human_label`. The MLP scores individual readings and the GRU and TCN score 10-step windows from the same held-out days. The raw CSV and generated data files are kept outside version control.

To train from scratch after downloading the dataset, run the following from the repository root. Training writes to `results/<model>/`, including the checkpoints and metrics, so use a separate checkout to keep a copy of the saved artifacts.

```bash
python src/inspect_ftaed.py
python src/preprocess.py
python src/train_baseline.py
python src/data_loader.py
python src/train_gru.py
python src/train_gru_sampler.py
python src/train_tcn_sampler.py
```

The training scripts save validation histories, loss and AP curves, and test precision–recall plots. To rebuild the [validation AP grid](results/final_figures/figure1_val_ap_grid.png) from those curves, run `python src/make_figure1_grid.py`.
