# 🔍 SimCLR Augmentation Analysis on STL-10

> 🧭 Based on the original [SimCLR repository by sthalles](https://github.com/sthalles/SimCLR/tree/master)

This repository implements a modular, reproducible pipeline for **Self-Supervised Learning (SSL)** using **SimCLR** on the **STL-10** dataset. The focus is on analyzing how **different image augmentations** impact the learned representations.

---

## 📌 Project Goals

- Reproduce SimCLR with full control over augmentations
- Evaluate representations using:
  - Linear probing
  - k-Nearest Neighbors (k-NN)
- Compare augmentation strategies: `baseline`, `color`, `blur`, `gray`, and `all`

---
## ⚙️ Implementation Notes & Comparison

This project is heavily inspired by and based on the SimCLR implementation found at [sthalles/SimCLR](https://github.com/sthalles/SimCLR/tree/master). While the core SimCLR principles are maintained, this repository introduces several structural and implementation choices aimed at modularity and specific research goals:

**Key Similarities:**

- **Core Framework:** Implements the SimCLR contrastive learning approach using the InfoNCE loss objective.
- **Dataset:** Primarily targets the STL-10 dataset, using the unlabeled set for pretraining and labeled train/test sets for evaluation.
- **Backbone:** Utilizes a ResNet-18 encoder architecture, loaded without ImageNet pretraining.
- **Projection Head:** Employs a 2-layer MLP projection head following the ResNet backbone to produce 128-dimensional embeddings for the contrastive loss. (Note: Minor difference in final layer bias term exists).
- **Optimization:**  Uses the Adam optimizer with a learning rate of 0.0003 and weight decay of 1e-4, along with a Cosine Annealing learning rate scheduler (stepped per epoch after 10 epochs warmup), aligning with the reference repository's settings.
- **Temperature:** Uses a contrastive temperature tau=0.07, matching the reference default.
- **Evaluation Concept:** Adopts the linear probing protocol (training a linear classifier on frozen features) as a primary evaluation method. The linear probe also uses the Adam optimizer (LR=3e-4, WD=8e-4) matching the reference evaluation notebook.

**Key Differences & Enhancements:**

- **Configuration:** This project uses a modular YAML-based configuration system (`configs/base_simclr.yaml` + specific overrides), allowing easier management of hyperparameters and augmentation strategies compared to the reference's `argparse`-based approach.
- **Augmentation Modularity:** Augmentations are defined declaratively in the YAML configuration and implemented in `Utils/augmentations.py`. This allows for explicit control and comparison of different augmentation pipelines (e.g., `baseline`, `color`, `blur`, `gray`, `all`), which is the core goal of this project. The reference implementation uses a single, hardcoded combined augmentation pipeline.
- **Gaussian Blur:** This implementation uses the standard `torchvision.transforms.GaussianBlur`, whereas the reference repository uses a custom implementation based on `nn.Conv2d`.
- **Distributed Training:** This project utilizes modern PyTorch DistributedDataParallel (DDP) launched via `torchrun` for efficient multi-GPU training. The reference implementation appears geared towards single-GPU or potentially older `DataParallel` methods.
- **Evaluation Suite:** Includes both linear probing (`linear_probe.py`) and k-Nearest Neighbors (`knn_eval.py`) evaluation scripts. The reference repository primarily provides a notebook for linear evaluation.
- **Evaluation Normalization:** Based on empirical results, ImageNet normalization is currently **omitted** during linear probe evaluation as it yielded better results for features trained with this setup, aligning with the apparent behavior in the reference evaluation notebook.
- **Code Structure:** Organized with separate utility (`Utils/`) and configuration (`configs/`) directories, and distinct scripts for pretraining and evaluation.


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
├── results/  # 🔄 This will be automatically generated after running any experiment
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

## ✅ Running All 5 Experiments In One Click

_(Make sure you specify desired number of **`EPOCHS`** and **`BATCH_SIZE`** in **`run_simclr_experiments.sh`**)  
Default is `EPOCHS=300`, and `BATCH_SIZE=256`._

Make sure you are in the root directory path, then run:

```bash
chmod +x run_simclr_experiments.sh
./run_simclr_experiments.sh
```

This will automatically run **Pretraining**, **Linear Probe**, and **k-NN Evaluation** for all five augmentation configs in sequence.

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

