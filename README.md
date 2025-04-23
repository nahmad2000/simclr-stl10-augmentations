# 🧠 Exploring Augmentation Strategies in Self-Supervised Learning

🔬 This repository provides a reproducible and flexible implementation of the original SimCLR framework [SimCLR repository by sthalles](https://github.com/sthalles/SimCLR/tree/master) for **self-supervised learning** on the **STL-10 dataset**. It offers a modular SimCLR implementation for STL-10 with YAML-based augmentation control and full evaluation workflow.

---

## 📌 Project Overview

- **Framework**: SimCLR (contrastive learning with InfoNCE)
- **Backbone**: ResNet-18 (no ImageNet pretraining)
- **Dataset**: STL-10 (Unlabeled for training, Train/Test for evaluation)

**Goals:**
- Modular config-based augmentation control
- Standard + extended augmentations (e.g., blur, rotation, solarize)
- Linear probing & k-NN evaluation
- Checkpoint-based performance tracking
- Centralized results and plots for easy analysis

---

## 🗂️ Directory Layout

```
configs/                  # YAML files for each experiment
  └── simclr_*.yaml       # Augmentation setups

Utils/                    # Helper modules
  ├── augmentations.py    # Custom transform logic
  ├── plotting.py         # Comparison plots
  └── train_utils.py      # Logging, seeding, DDP helpers

data/stl10_binary/        # STL-10 dataset after downloading

results/                  # Experiment outputs
  ├── simclr_<exp>/       # Checkpoints, logs, metrics
  ├── plots/              # Aggregated plots
  ├── consolidated_results.csv
  └── summary_report_final_epoch.md

simclr.py                 # Pretraining script
linear_probe.py           # Evaluation via frozen features
knn_eval.py               # Evaluation via nearest neighbors
run.py                    # Unified launcher
```

---

## ⚙️ Setup Instructions

```bash
# Clone and navigate to repo
git clone <repo-url>
cd <repo-dir>

# Virtual environment
python -m venv ssl_env
source ssl_env/bin/activate  # or .\ssl_env\Scripts\activate on Windows

# Install packages
pip install torch torchvision torchaudio
pip install PyYAML scikit-learn matplotlib seaborn tqdm numpy pandas

# Download dataset
python download_stl10.py
```

---

## 🚀 Run Experiments

```bash
# Single experiment (e.g., baseline)
python run.py baseline

# Multiple experiments
python run.py blur color all_extended

# All experiments (Defined in run.py)
python run.py all

# Custom hyperparameters
python run.py rotation --epochs 200 --batch_size 128 --gpus 2 --saving_epoch 25

# Force rerun or cleanup
python run.py baseline --force_rerun
python run.py baseline --cleanup_intermediate
```

**Experiment names** map to YAML files in `configs/`.

---

## 🧩 Add New Augmentation

1. Define params (if needed) in `base_simclr.yaml`
2. Create `simclr_myexp.yaml` with augmentations enabled/disabled
3. Implement logic (if new transform) in `augmentations.py`
4. Run: `python run.py myexp --epochs 100 --batch_size 256`
5. Add to `ALL_STANDARD_EXPERIMENTS` in `run.py` (optional)

---

## 📁 Output Summary

### For Each Experiment:
- `final_model.pth`, `model_checkpoint.pth`, `model_epoch_*.pth`
- `training.log`, `linear_probe.log`, `knn_eval.log`
- `training_loss_*.csv`, `*_acc_epoch_*.txt`
- `loss_*.png`, `pseudo_acc_*.png`

### Centralized Outputs:
- `consolidated_results.csv`: All metrics, all epochs
- `summary_report_final_epoch.md`: Final performance table
- `plots/`: Comparisons across experiments
  - `loss_all_comparison.png`
  - `linear_probe_top1_comparison_latest.png`
  - `knn_top1_comparison_latest.png`
  - `ablation_performance_drop_top1_latest.png`
  - `linear_probe_top1_evolution.png`

---

## 👤 Author

**Ahmad Nayfeh**  
Master’s Student @ KFUPM

