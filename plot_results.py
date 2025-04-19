import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# plot_results.py
# Script to generate plots for InfoNCE loss, pseudo-accuracy, LR schedule,
# linear-probe and k-NN results, saving all figures into results/plots/.

# Configuration
AUGS = ['baseline', 'color', 'blur', 'gray', 'all']
RESULTS_DIR = 'results'
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
KS = [1, 5, 10]

def read_training_loss(aug):
    path = os.path.join(RESULTS_DIR, f'simclr_{aug}', f'training_loss_simclr_{aug}.csv')
    return pd.read_csv(path)

def read_linear_probe(aug):
    path = os.path.join(RESULTS_DIR, f'simclr_{aug}', f'linear_probe_acc_simclr_{aug}.txt')
    lines = open(path).read().splitlines()
    top1 = float(re.search(r'Top-1 Accuracy: ([\d.]+)%', lines[0]).group(1))
    top5 = float(re.search(r'Top-5 Accuracy: ([\d.]+)%', lines[1]).group(1))
    return top1, top5

def read_knn(aug):
    path = os.path.join(RESULTS_DIR, f'simclr_{aug}', f'knn_acc_simclr_{aug}.txt')
    result = {}
    for line in open(path):
        m = re.search(r'\(k=(\d+)\): ([\d.]+)%', line)
        if m:
            k = int(m.group(1))
            val = float(m.group(2))
            result[k] = val
    return result

def plot_loss():
    # Per-experiment
    for aug in AUGS:
        df = read_training_loss(aug)
        plt.figure()
        plt.plot(df['epoch'], df['avg_loss'])
        plt.xlabel('Epoch'); plt.ylabel('InfoNCE Loss')
        plt.title(f'InfoNCE Loss vs Epoch ({aug})')
        plt.savefig(os.path.join(PLOTS_DIR, f'loss_{aug}.png'))
    # Combined
    plt.figure()
    for aug in AUGS:
        df = read_training_loss(aug)
        plt.plot(df['epoch'], df['avg_loss'], label=aug)
    plt.xlabel('Epoch'); plt.ylabel('InfoNCE Loss')
    plt.legend(); plt.title('InfoNCE Loss vs Epoch (all)')
    plt.savefig(os.path.join(PLOTS_DIR, 'loss_all.png'))

def plot_pseudo_accuracy():
    for aug in AUGS:
        df = read_training_loss(aug)
        plt.figure()
        plt.plot(df['epoch'], df['avg_top1_acc'])
        plt.xlabel('Epoch'); plt.ylabel('Pseudo Acc (%)')
        plt.title(f'Pseudo-Accuracy vs Epoch ({aug})')
        plt.savefig(os.path.join(PLOTS_DIR, f'acc_{aug}.png'))
    plt.figure()
    for aug in AUGS:
        df = read_training_loss(aug)
        plt.plot(df['epoch'], df['avg_top1_acc'], label=aug)
    plt.xlabel('Epoch'); plt.ylabel('Pseudo Acc (%)')
    plt.legend(); plt.title('Pseudo-Accuracy vs Epoch (all)')
    plt.savefig(os.path.join(PLOTS_DIR, 'acc_all.png'))

def plot_lr_schedule():
    for aug in AUGS:
        df = read_training_loss(aug)
        plt.figure()
        plt.plot(df['epoch'], df['learning_rate'])
        plt.xlabel('Epoch'); plt.ylabel('Learning Rate')
        plt.title(f'LR Schedule vs Epoch ({aug})')
        plt.savefig(os.path.join(PLOTS_DIR, f'lr_{aug}.png'))
    plt.figure()
    for aug in AUGS:
        df = read_training_loss(aug)
        plt.plot(df['epoch'], df['learning_rate'], label=aug)
    plt.xlabel('Epoch'); plt.ylabel('Learning Rate')
    plt.legend(); plt.title('LR Schedule vs Epoch (all)')
    plt.savefig(os.path.join(PLOTS_DIR, 'lr_all.png'))

def plot_linear_probe():
    top1_vals = []
    for aug in AUGS:
        top1, _ = read_linear_probe(aug)
        top1_vals.append(top1)
    plt.figure()
    plt.bar(AUGS, top1_vals)
    plt.xlabel('Augmentation'); plt.ylabel('Linear-Probe Top-1 Acc (%)')
    plt.title('Linear-Probe Top-1 Accuracy')
    plt.savefig(os.path.join(PLOTS_DIR, 'linear_probe_top1.png'))

def plot_knn():
    plt.figure()
    X = range(len(AUGS))
    width = 0.2
    for idx, k in enumerate(KS):
        vals = [read_knn(aug).get(k, 0) for aug in AUGS]
        plt.bar([x + idx*width for x in X], vals, width=width, label=f'k={k}')
    plt.xticks([x + width for x in X], AUGS)
    plt.xlabel('Augmentation'); plt.ylabel('k-NN Top-1 Acc (%)')
    plt.legend(); plt.title('k-NN Top-1 Accuracy for k')
    plt.savefig(os.path.join(PLOTS_DIR, 'knn_top1.png'))

if __name__ == '__main__':
    os.makedirs(PLOTS_DIR, exist_ok=True)
    for aug in AUGS:
        os.makedirs(os.path.join(RESULTS_DIR, f'simclr_{aug}'), exist_ok=True)
    plot_loss()
    plot_pseudo_accuracy()
    plot_lr_schedule()
    plot_linear_probe()
    plot_knn()
    print(f"All plots saved under '{PLOTS_DIR}' directory.")
