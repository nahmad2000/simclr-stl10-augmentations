# 🧠 Exploring Augmentation Strategies in Self-Supervised Learning

> 🔬 ****

This repository provides a reproducible and flexible implementation of the original SimCLR framework [SimCLR repository by sthalles](https://github.com/sthalles/SimCLR/tree/master) for **self-supervised learning** on the **STL-10 dataset**. It offers a modular SimCLR implementation for STL-10 with YAML-based augmentation control and full evaluation workflow


---

## 📌 Project Goals

- Reproduce SimCLR with full control over augmentations
- Evaluate representations using:
  - Linear probing
  - k-Nearest Neighbors (k-NN)
- Compare augmentation strategies: `baseline`, `color`, `blur`, `gray`, and `all`

---


## 📌 Project Overview

- **Framework**: SimCLR (contrastive learning using InfoNCE)
- **Backbone**: ResNet-18 (no ImageNet pretraining)
- **Dataset**: STL-10 (Unlabeled for pretraining, Train/Test for evaluation)
- **Goal**: Evaluate how augmentation strategies affect learned representations

---

## 🗂️ Directory Structure

```
├── configs/
│   ├── base_simclr.yaml
│   ├── simclr_baseline.yaml
│   ├── simclr_blur.yaml
│   ├── simclr_color.yaml
│   ├── simclr_gray.yaml
│   ├── simclr_all_standard.yaml
│   └── simclr_custom.yaml
│
├── data/
│   └── stl10_binary/
│
├── Utils/
│   ├── augmentations.py
│   ├── plotting.py
│   └── train_utils.py
│
├── results/              # Auto-generated per experiment
│   └── plots/            # Comparison plots for all experiments
├── simclr.py             # Main pretraining loop
├── linear_probe.py       # Linear evaluation script
├── knn_eval.py           # k-NN evaluation script
├── run.py                # Unified experiment launcher
├── download_stl10.py     # STL-10 dataset fetcher
└── README.md             # This file
```

---

## ⚙️ Setup Instructions

```bash
# Create environment
python -m venv ssl_env
source ssl_env/bin/activate

# Install PyTorch and dependencies
pip install torch torchvision torchaudio
pip install PyYAML scikit-learn matplotlib seaborn tqdm numpy pandas

# Download STL-10 dataset
python download_stl10.py
```

---

## 🚀 How to Run Experiments

Use `run.py` to orchestrate pretraining, evaluation, and plotting:

```bash
# Run baseline experiment (pretrain + linear + k-NN + plot)
python run.py baseline

# Run multiple specific experiments
python run.py blur color all_standard

# Run all defined configs in run.py
python run.py all
```

Override hyperparameters if needed:
```bash
python run.py baseline --epochs 200 --batch_size 128 --gpus 2
```

---

## 🔬 Individual Evaluation Scripts

```bash
# Pretraining only
torchrun --nproc_per_node=2 simclr.py --config configs/simclr_blur.yaml

# Linear probe evaluation
python linear_probe.py --checkpoint results/simclr_blur/final_model.pth --config configs/simclr_blur.yaml

# k-NN evaluation
python knn_eval.py --checkpoint results/simclr_blur/final_model.pth --config configs/simclr_blur.yaml
```

---

## 📈 Plotting Results

```bash
python Utils/plotting.py --experiments baseline blur color gray all_standard
```
Generates `.png` files in `results/plots/` and individual run folders.

---

## 🧩 How to Add a New Augmentation Config

1. 📄 Duplicate a config YAML:
   ```bash
   cp configs/simclr_baseline.yaml configs/simclr_myaug.yaml
   ```
2. 🧪 Modify the `augmentations:` section:
   ```yaml
   augmentations:
     color_jitter: { enabled: true }
     grayscale: { enabled: false }
   ```
3. 🔧 (Optional) Extend `augmentations.py` if you add custom logic.
4. ▶️ Run your experiment:
   ```bash
   python run.py myaug --epochs 100 --batch_size 256
   ```

---

## 📁 Output Artifacts (Per Experiment)

Each `results/simclr_<name>/` folder includes:
- `final_model.pth`, `best_model.pth`, `model_checkpoint.pth`
- `training.log`, `linear_probe.log`, `knn_eval.log`
- `effective_config.yaml`, `training_loss_*.csv`
- Accuracy summaries in `.txt` files
- Auto-generated plots: `loss_*.png`, `pseudo_acc_*.png`

---

## 📊 Augmentation Config Matrix

| Config Name    | Color Jitter | Blur | Grayscale |
| -------------- | ------------ | ---- | --------- |
| baseline       | ❌            | ❌    | ❌         |
| color          | ✅            | ❌    | ❌         |
| blur           | ❌            | ✅    | ❌         |
| gray           | ❌            | ❌    | ✅         |
| all_standard   | ✅            | ✅    | ✅         |


---


