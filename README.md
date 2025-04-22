# 🧠 Exploring Augmentation Strategies in Self-Supervised Learning

> 🔬 This repository provides a reproducible and flexible implementation of the original SimCLR framework [SimCLR repository by sthalles](https://github.com/sthalles/SimCLR/tree/master) for **self-supervised learning** on the **STL-10 dataset**. It offers a modular SimCLR implementation for STL-10 with YAML-based augmentation control and full evaluation workflow

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
│   └── simclr_myaug.yaml
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

> Notes:
> - Default Parameters are --> `epochs=100`, `batch_size=256`, `gpus=1`
> - If you didn't pass experiment name, then the by default it will run `baseline` (Which is `simclr_baseline.yaml`)
> - If you want to run all experiments in one go, then you can pass `all` as the experiment name

---

## 🧩 How to Add a New Augmentation Config

1. Create a new config YAML file under `configs/`. For example: `simclr_myaug.yaml`
2. Follow same structure as `simclr_blur.yaml`, `simclr_color.yaml`, `simclr_gray.yaml`, and `simclr_baseline.yaml`
3. Extend `augmentations.py` if you add custom logic.
4. Run your experiment:
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

## 👤 Author

**Ahmad Nayfeh**  
Master’s Student @ KFUPM


