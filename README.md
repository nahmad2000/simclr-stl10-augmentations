# 🔍 SimCLR Augmentation Analysis on STL-10

This repository implements a modular, reproducible pipeline for **Self-Supervised Learning (SSL)** using **SimCLR** on the **STL-10** dataset. The focus is on analyzing how **different image augmentations** impact the learned representations.

---

## 📌 Project Goals

- Reproduce SimCLR with full control over augmentations
- Evaluate representations using:
  - Linear probing
  - k-Nearest Neighbors (k-NN)
- Compare augmentation strategies: `baseline`, `color`, `blur`, `gray`, and `all`

---

## 🧰 Prerequisites & Setup

### 1. Environment

```bash
# Using venv
python -m venv ssl_env
source ssl_env/bin/activate

# OR using conda
# conda create --name ssl_env python=3.10 -y
# conda activate ssl_env
```

### 2. Dependencies

Install PyTorch (check [official instructions](https://pytorch.org/get-started/locally/) for your CUDA version) and then:

```bash
pip install PyYAML scikit-learn matplotlib seaborn tqdm numpy pandas
```

### 3. Download STL-10

```bash
python download_stl10.py
```

---

## 🗂️ Directory Structure

```bash
├── configs/
│   ├── base_simclr.yaml
│   ├── simclr_baseline.yaml
│   ├── simclr_color.yaml
│   ├── simclr_blur.yaml
│   ├── simclr_gray.yaml
│   └── simclr_all.yaml
│
├── Utils/
│   ├── augmentations.py
│   └── train_utils.py
│
├── results/
│   ├── simclr_baseline/
│   │   ├── final_model.pth
│   │   ├── model_checkpoint.pth
│   │   ├── best_model.pth
│   │   ├── training.log
│   │   ├── effective_config.yaml
│   │   ├── training_loss_simclr_baseline.csv
│   │   ├── linear_probe_acc_simclr_baseline.txt
│   │   ├── knn_acc_simclr_baseline.txt
│   │   └── linear_probe.log / knn_eval.log
│   └── ...
│
├── download_stl10.py
├── simclr.py
├── linear_probe.py
├── knn_eval.py
├── run_simclr_experiments.sh
├── plot_results.py
└── README.md
```

---

## 🚀 Running Experiments

### 1. Pretraining

```bash
torchrun --nproc_per_node=2 simclr.py --config configs/simclr_baseline.yaml

# Override CLI params:
torchrun --nproc_per_node=2 simclr.py --config configs/simclr_baseline.yaml --epochs 5 --batch_size 128
```

### 2. Linear Probe Evaluation

```bash
python linear_probe.py \
  --checkpoint results/simclr_baseline/final_model.pth \
  --config configs/simclr_baseline.yaml
```

### 3. k-NN Evaluation

```bash
python knn_eval.py \
  --checkpoint results/simclr_baseline/final_model.pth \
  --config configs/simclr_baseline.yaml
```

---

## 📊 Plotting Results

```bash
python plot_results.py
```

Plots saved to `results/plots/`.

---

## 📁 Outputs (per run)

Each folder under `results/simclr_<aug>/` contains:

- `final_model.pth`: Final encoder weights (projection head removed)
- `model_checkpoint.pth`: Last checkpoint during training
- `best_model.pth`: Lowest-loss model (if early stopping enabled)
- `effective_config.yaml`: Full config used (base + specific + CLI overrides)
- `training.log`: Full pretraining log
- `training_loss_simclr_<aug>.csv`: Epoch-wise loss, pseudo accuracy, LR
- `linear_probe_acc_simclr_<aug>.txt`: Linear evaluation results
- `knn_acc_simclr_<aug>.txt`: k-NN results for k = 1, 5, 10
- `linear_probe.log`, `knn_eval.log`: Evaluation logs

---

## 🔧 Extending Experiments

- Create a new YAML in `configs/` and modify `augmentations:` section.
- Adjust training/evaluation hyperparameters in `base_simclr.yaml` or specific config.
- Use CLI overrides for quick changes (`--epochs`, `--batch_size`, etc.).

---

## 🧠 Key Features

- ✅ Modular YAML configuration
- ✅ Distributed training via PyTorch DDP
- ✅ Cosine LR scheduler
- ✅ Early stopping support
- ✅ Mixed-precision training (optional)
- ✅ Logging + CSV outputs for reproducibility

---

## 🧪 Augmentation Variants

| Variant   | Color Jitter | Blur | Grayscale |
|-----------|--------------|------|-----------|
| baseline  | ❌           | ❌   | ❌        |
| color     | ✅           | ❌   | ❌        |
| blur      | ❌           | ✅   | ❌        |
| gray      | ❌           | ❌   | ✅        |
| all       | ✅           | ✅   | ✅        |

---

## 📚 Acknowledgements & Citation

This project is inspired by:

- [SimCLR (ICML 2020)](https://arxiv.org/abs/2002.05709) – Chen et al.

---

## 👤 Author

**Ahmad Nayfeh**  
Master's Student @ KFUPM  
---
