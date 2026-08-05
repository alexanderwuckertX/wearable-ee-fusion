# Energy Expenditure Estimation from Wearable Biosignals

Single notebook (`main.ipynb`) predicting energy expenditure (EE, in W/kg) from wearable
sensor signals. Trains a shared 1D-CNN backbone (`Net`) as 16 single-signal baselines
(Phase 1), then compares early, intermediate, and late multi-modal fusion strategies
(`EarlyFusionNet` / `IntermediateFusionNet` / `LateFusionNet`) on predefined signal
groups G1–G6 (Phase 2). All models are evaluated under Leave-One-Subject-Out (LOSO)
cross-validation across the 10 subjects.

## 1. Environment

Runs on CPU (no `.to(device)`/CUDA calls in the notebook — a GPU is not used or required).
Packages, installed via `pip` in the notebook's first cells:

```
pip install torch torchvision h5py scikit-learn matplotlib numpy
```

(`torchvision` is installed but not actually used by the code.)

## 2. Dataset

The raw `SubjectXX.mat` files are not tracked in this repo (see `.gitignore`). Download
them from [figshare](https://figshare.com/articles/dataset/Predicting_energy_cost_from_wearable_sensors/7473191)
and place `Subject01.mat` ... `Subject10.mat` directly under `./data/`. All paths in the
notebook are relative to the project root, so no other configuration is needed.

## 3. Run order

`main.ipynb` is a single top-to-bottom pipeline; Phase 2 depends on outputs Phase 1
writes to disk, so cells must be run in order:

1. **Phase 1 — single-signal baselines.** Computes EE ground truth and saves
   `labels.pkl`, then LOSO-trains `Net` separately on each of the 16 candidate signals
   (fixed hyperparameters: Adam, lr=1e-3, weight_decay=1e-4, tuned once on Heart Rate).
   - Saves `phase1_results.pkl` (per-signal RMSE/MAE/R², averaged over folds) and
     `phase1_signal_ranking.png`.
   - Raw sensor arrays are cached to `hdf5_cache.pkl` as they're read, so re-running
     is faster on subsequent passes.
2. **Phase 2 — multi-modal fusion.** Reuses `labels.pkl` and the cache from step 1.
   Defines signal groups G1–G6 and LOSO-trains all three fusion architectures on each
   group, with the same locked hyperparameters as Phase 1.
   - Saves `phase2_results.pkl` (per-group, per-fusion-strategy metrics),
     `learning_curves.png`, and `scatter_best.png`.
   - Later cells re-run a subset of groups with input normalization and gradient
     clipping as additional regularization checks, saving
     `phase2_normalized_results.pkl`.

Both phases are resumable: results already saved to the `.pkl` files can be reloaded
instead of recomputing finished LOSO folds.
