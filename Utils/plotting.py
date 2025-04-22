# Utils/plotting.py
# Updated plotting script to work with specific experiments

import argparse
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import logging
from pathlib import Path

# Basic logging setup for the plotting script
logging.basicConfig(level=logging.INFO, format='%(asctime)s [Plotter] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# ====== Config ======
# DEFAULT_AUGS = ['baseline', 'color', 'blur', 'gray', 'all'] # Default list if none provided via args
RESULTS_DIR = Path('results')
CENTRAL_PLOTS_DIR = RESULTS_DIR / 'plots'
MODEL_NAME = "simclr" # Prefix for result directories and files
KS = [1, 5, 10]

# High-resolution plot settings
matplotlib.rcParams.update({
    "font.size": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

# ====== Readers (with Error Handling) ======
def read_training_loss(aug_name):
    """Reads training loss CSV for a specific augmentation name."""
    run_dir = RESULTS_DIR / f'{MODEL_NAME}_{aug_name}'
    path = run_dir / f'training_loss_{MODEL_NAME}_{aug_name}.csv'
    try:
        df = pd.read_csv(path)
        logging.debug(f"Read training loss from {path}")
        return df
    except FileNotFoundError:
        logging.warning(f"Training loss file not found: {path}")
        return None
    except Exception as e:
        logging.error(f"Error reading {path}: {e}")
        return None

def read_linear_probe(aug_name):
    """Reads linear probe results TXT for a specific augmentation name."""
    run_dir = RESULTS_DIR / f'{MODEL_NAME}_{aug_name}'
    path = run_dir / f'linear_probe_acc_{MODEL_NAME}_{aug_name}.txt'
    try:
        with open(path, 'r') as f:
            lines = f.read().splitlines()
        top1 = float(re.search(r'Top-1 Accuracy: ([\d.]+)%', lines[0]).group(1))
        top5 = float(re.search(r'Top-5 Accuracy: ([\d.]+)%', lines[1]).group(1))
        logging.debug(f"Read linear probe from {path}: Top-1={top1}, Top-5={top5}")
        return top1, top5
    except FileNotFoundError:
        logging.warning(f"Linear probe file not found: {path}")
        return None, None
    except (IndexError, AttributeError, ValueError) as e:
        logging.error(f"Error parsing {path}: {e}")
        return None, None
    except Exception as e:
        logging.error(f"Error reading {path}: {e}")
        return None, None


def read_knn(aug_name):
    """Reads k-NN results TXT for a specific augmentation name."""
    run_dir = RESULTS_DIR / f'{MODEL_NAME}_{aug_name}'
    path = run_dir / f'knn_acc_{MODEL_NAME}_{aug_name}.txt'
    result = {}
    try:
        with open(path, 'r') as f:
            for line in f:
                m = re.search(r'\(k=(\d+)\): ([\d.]+)%', line)
                if m:
                    k = int(m.group(1))
                    val = float(m.group(2))
                    result[k] = val
        logging.debug(f"Read kNN results from {path}: {result}")
        return result
    except FileNotFoundError:
        logging.warning(f"k-NN file not found: {path}")
        return {}
    except Exception as e:
        logging.error(f"Error reading {path}: {e}")
        return {}

# ====== Plotters (Updated Save Paths) ======

def plot_loss(experiments_to_plot):
    """Plots individual and combined loss curves."""
    plt.figure() # For combined plot
    for aug_name in experiments_to_plot:
        df = read_training_loss(aug_name)
        if df is not None:
            run_dir = RESULTS_DIR / f'{MODEL_NAME}_{aug_name}'
            run_dir.mkdir(parents=True, exist_ok=True) # Ensure individual dir exists
            individual_plot_path = run_dir / f'loss_{aug_name}.png'
            combined_plot_path = CENTRAL_PLOTS_DIR / 'loss_all.png'

            # Individual Plot
            plt.figure() # New figure for individual plot
            plt.plot(df['epoch'], df['avg_loss'])
            plt.xlabel('Epoch'); plt.ylabel('InfoNCE Loss')
            plt.grid(True)
            plt.title(f'InfoNCE Loss vs Epoch ({aug_name})')
            plt.savefig(individual_plot_path, dpi=300, bbox_inches='tight')
            plt.close() # Close individual plot figure
            logging.info(f"Saved individual loss plot: {individual_plot_path}")

            # Add to combined plot
            plt.figure(1) # Switch back to combined plot figure (figure number 1)
            plt.plot(df['epoch'], df['avg_loss'], label=aug_name)

    # Finalize Combined Plot
    plt.figure(1)
    if plt.gca().has_data(): # Check if any data was plotted
        plt.xlabel('Epoch'); plt.ylabel('InfoNCE Loss')
        plt.grid(True)
        plt.legend(); plt.title(f'InfoNCE Loss vs Epoch (Comparison)')
        CENTRAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved combined loss plot: {combined_plot_path}")
    else:
         logging.warning("No data found to plot combined loss.")
    plt.close() # Close combined plot figure


def plot_pseudo_accuracy(experiments_to_plot):
    """Plots individual and combined pseudo-accuracy curves."""
    plt.figure() # For combined plot
    for aug_name in experiments_to_plot:
        df = read_training_loss(aug_name)
        if df is not None and 'avg_top1_acc' in df.columns:
            run_dir = RESULTS_DIR / f'{MODEL_NAME}_{aug_name}'
            run_dir.mkdir(parents=True, exist_ok=True)
            individual_plot_path = run_dir / f'pseudo_acc_{aug_name}.png'
            combined_plot_path = CENTRAL_PLOTS_DIR / 'pseudo_acc_all.png'

            # Individual Plot
            plt.figure()
            plt.plot(df['epoch'], df['avg_top1_acc'])
            plt.xlabel('Epoch'); plt.ylabel('Pseudo Accuracy (%)')
            plt.grid(True); plt.ylim(bottom=0) # Ensure y-axis starts at 0
            plt.title(f'Pseudo-Accuracy vs Epoch ({aug_name})')
            plt.savefig(individual_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"Saved individual pseudo-accuracy plot: {individual_plot_path}")

            # Add to combined plot
            plt.figure(1)
            plt.plot(df['epoch'], df['avg_top1_acc'], label=aug_name)
        elif df is not None:
             logging.warning(f"'avg_top1_acc' column not found in {run_dir / f'training_loss_{MODEL_NAME}_{aug_name}.csv'}")


    # Finalize Combined Plot
    plt.figure(1)
    if plt.gca().has_data():
        plt.xlabel('Epoch'); plt.ylabel('Pseudo Accuracy (%)')
        plt.grid(True); plt.ylim(bottom=0)
        plt.legend(); plt.title(f'Pseudo-Accuracy vs Epoch (Comparison)')
        CENTRAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved combined pseudo-accuracy plot: {combined_plot_path}")
    else:
         logging.warning("No data found to plot combined pseudo-accuracy.")
    plt.close()


def plot_linear_probe(experiments_to_plot):
    """Plots Linear Probe Top-1 accuracy as a bar chart."""
    top1_vals = []
    valid_augs = []
    for aug_name in experiments_to_plot:
        top1, _ = read_linear_probe(aug_name)
        if top1 is not None:
            top1_vals.append(top1)
            valid_augs.append(aug_name)

    if not valid_augs:
        logging.warning("No valid linear probe data found to plot.")
        return

    plt.figure()
    plt.bar(valid_augs, top1_vals)
    plt.xlabel('Augmentation'); plt.ylabel('Top-1 Accuracy (%)')
    plt.title('Linear-Probe Top-1 Accuracy')
    plt.grid(True, axis='y')
    plt.ylim(bottom=0)
    # Rotate labels if many experiments
    if len(valid_augs) > 5:
        plt.xticks(rotation=45, ha='right')

    plot_path = CENTRAL_PLOTS_DIR / 'linear_probe_top1.png'
    CENTRAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved linear probe plot: {plot_path}")
    plt.close()


def plot_knn(experiments_to_plot):
    """Plots k-NN Top-1 accuracy as a grouped bar chart."""
    knn_data = {k: [] for k in KS}
    valid_augs = []

    for aug_name in experiments_to_plot:
        results = read_knn(aug_name)
        # Check if results were found for this aug
        if any(k in results for k in KS):
             valid_augs.append(aug_name)
             for k in KS:
                 knn_data[k].append(results.get(k, 0)) # Append 0 if k is missing for this aug
        else:
            # If no results found for any k for this aug, pad with zeros
            # only if we intend to show the aug label anyway (currently we only show valid_augs)
            pass


    if not valid_augs:
        logging.warning("No valid k-NN data found to plot.")
        return

    plt.figure()
    num_augs = len(valid_augs)
    x_indices = range(num_augs) # Indices for bars [0, 1, 2,...]
    num_ks = len(KS)
    width = 0.8 / num_ks # Calculate width based on number of K values

    for i, k in enumerate(KS):
        # Calculate offset for this k value's bars
        offset = (i - (num_ks - 1) / 2) * width
        plt.bar([x + offset for x in x_indices], knn_data[k], width=width, label=f'k={k}')

    plt.xticks(x_indices, valid_augs) # Place ticks at the center of the groups
    plt.xlabel('Augmentation'); plt.ylabel('Top-1 Accuracy (%)')
    plt.title('k-NN Classification Accuracy')
    plt.grid(True, axis='y')
    plt.ylim(bottom=0)
    plt.legend()
    # Rotate labels if many experiments
    if num_augs > 5:
         plt.xticks(rotation=45, ha='right')

    plot_path = CENTRAL_PLOTS_DIR / 'knn_top1.png'
    CENTRAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved k-NN plot: {plot_path}")
    plt.close()

# ====== Runner ======
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate plots for SimCLR experiments.")
    parser.add_argument(
        "--experiments",
        nargs='+',
        required=True, # Make it required when run standalone
        # default=DEFAULT_AUGS, # Remove default when required
        help="List of experiment augmentation names to plot (e.g., baseline color all)."
    )
    args = parser.parse_args()

    logging.info(f"Generating plots for experiments: {args.experiments}")

    # Ensure central plots directory exists
    CENTRAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Call plotting functions with the list of experiments from args
    plot_loss(args.experiments)
    plot_pseudo_accuracy(args.experiments)
    plot_linear_probe(args.experiments)
    plot_knn(args.experiments)

    print(f"Plotting complete. Check individual run directories and '{CENTRAL_PLOTS_DIR}'")