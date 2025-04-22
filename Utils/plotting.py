# Utils/plotting.py
# Updated plotting script with improved readability, ablation plot,
# and a function to generate a summary text report.

import argparse
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import logging
from pathlib import Path
from collections import defaultdict
import itertools # For line styles

# Basic logging setup for the plotting script
logging.basicConfig(level=logging.INFO, format='%(asctime)s [Plotter] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# ====== Config ======
RESULTS_DIR = Path('results')
CENTRAL_PLOTS_DIR = RESULTS_DIR / 'plots'
SUMMARY_REPORT_FILE = RESULTS_DIR / 'summary_report.md' # Using Markdown for better formatting
MODEL_NAME = "simclr" # Prefix for result directories and files
DEFAULT_ABLATION_BASELINE = 'all_extended' # *** IMPORTANT: Set this to the name of your full augmentation combo experiment ***
ABLATION_PREFIX = 'all_minus_' # Prefix for experiments where one augmentation is removed
KNN_K_VALUES_FOR_SUMMARY = [1, 5, 10] # K values to include in the summary report

# Define a list of line styles to cycle through for combined plots
LINE_STYLES = ['-', '--', '-.', ':']

# High-resolution plot settings + Style
# plt.style.use('seaborn-v0_8-paper') # Example style, choose one you like
matplotlib.rcParams.update({
    "font.size": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.titlesize": 12, # Slightly larger titles
    "axes.labelsize": 10,
    "legend.fontsize": 8, # Slightly smaller legend for many items
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.figsize": (10, 6) # Default figure size
})

# ====== Readers (with Error Handling - unchanged from previous) ======
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
    path = run_dir / f'linear_probe_acc_{MODEL_NAME}_{aug_name}.txt' #
    try:
        with open(path, 'r') as f:
            lines = f.read().splitlines()
        top1 = float(re.search(r'Top-1 Accuracy: ([\d.]+)%', lines[0]).group(1))
        # Handle potential absence of Top-5
        top5_match = re.search(r'Top-5 Accuracy: ([\d.]+)%', lines[1])
        top5 = float(top5_match.group(1)) if top5_match else None
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
    path = run_dir / f'knn_acc_{MODEL_NAME}_{aug_name}.txt' #
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

# ====== Plotters (Unchanged from previous update) ======

def plot_loss(experiments_to_plot):
    """Plots individual and combined loss curves with improved style cycling."""
    plt.figure() # For combined plot
    style_cycler = itertools.cycle(LINE_STYLES)
    color_cycler = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    for aug_name in experiments_to_plot:
        df = read_training_loss(aug_name)
        if df is not None:
            run_dir = RESULTS_DIR / f'{MODEL_NAME}_{aug_name}'
            run_dir.mkdir(parents=True, exist_ok=True) # Ensure individual dir exists
            individual_plot_path = run_dir / f'loss_{aug_name}.png'
            combined_plot_path = CENTRAL_PLOTS_DIR / 'loss_all_comparison.png'

            # Individual Plot
            plt.figure(figsize=(8, 5)) # Smaller individual plot
            plt.plot(df['epoch'], df['avg_loss'])
            plt.xlabel('Epoch'); plt.ylabel('InfoNCE Loss')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.title(f'InfoNCE Loss vs Epoch ({aug_name})')
            plt.tight_layout()
            plt.savefig(individual_plot_path, dpi=300, bbox_inches='tight')
            plt.close() # Close individual plot figure
            logging.info(f"Saved individual loss plot: {individual_plot_path}")

            # Add to combined plot
            plt.figure(1) # Switch back to combined plot figure (figure number 1)
            plt.plot(df['epoch'], df['avg_loss'],
                     label=aug_name,
                     linestyle=next(style_cycler),
                     color=next(color_cycler)) # Use style and color cyclers

    # Finalize Combined Plot
    plt.figure(1)
    if plt.gca().has_data(): # Check if any data was plotted
        plt.xlabel('Epoch'); plt.ylabel('InfoNCE Loss')
        plt.grid(True, linestyle='--', alpha=0.6)
        # Adjust legend position if too many items
        num_items = len(plt.gca().get_lines())
        if num_items > 10:
             plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        else:
             plt.legend(loc='best')
        plt.title(f'InfoNCE Loss vs Epoch (Comparison)')
        CENTRAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.tight_layout(rect=[0, 0, 0.85, 1] if num_items > 10 else None) # Adjust layout if legend is outside
        plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved combined loss plot: {combined_plot_path}")
    else:
         logging.warning("No data found to plot combined loss.")
    plt.close() # Close combined plot figure


def plot_pseudo_accuracy(experiments_to_plot):
    """Plots individual and combined pseudo-accuracy curves with improved style cycling."""
    plt.figure() # For combined plot
    style_cycler = itertools.cycle(LINE_STYLES)
    color_cycler = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    for aug_name in experiments_to_plot:
        df = read_training_loss(aug_name)
        if df is not None and 'avg_top1_acc' in df.columns:
            run_dir = RESULTS_DIR / f'{MODEL_NAME}_{aug_name}'
            run_dir.mkdir(parents=True, exist_ok=True)
            individual_plot_path = run_dir / f'pseudo_acc_{aug_name}.png'
            combined_plot_path = CENTRAL_PLOTS_DIR / 'pseudo_acc_all_comparison.png'

            # Individual Plot
            plt.figure(figsize=(8, 5)) # Smaller individual plot
            plt.plot(df['epoch'], df['avg_top1_acc'])
            plt.xlabel('Epoch'); plt.ylabel('Pseudo Accuracy (%)')
            plt.grid(True, linestyle='--', alpha=0.6); plt.ylim(bottom=0) # Ensure y-axis starts at 0
            plt.title(f'Pseudo-Accuracy vs Epoch ({aug_name})')
            plt.tight_layout()
            plt.savefig(individual_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"Saved individual pseudo-accuracy plot: {individual_plot_path}")

            # Add to combined plot
            plt.figure(1)
            plt.plot(df['epoch'], df['avg_top1_acc'],
                     label=aug_name,
                     linestyle=next(style_cycler),
                     color=next(color_cycler)) # Use style and color cyclers
        elif df is not None:
             logging.warning(f"'avg_top1_acc' column not found in {run_dir / f'training_loss_{MODEL_NAME}_{aug_name}.csv'}")


    # Finalize Combined Plot
    plt.figure(1)
    if plt.gca().has_data():
        plt.xlabel('Epoch'); plt.ylabel('Pseudo Accuracy (%)')
        plt.grid(True, linestyle='--', alpha=0.6); plt.ylim(bottom=0)
        # Adjust legend position if too many items
        num_items = len(plt.gca().get_lines())
        if num_items > 10:
             plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        else:
             plt.legend(loc='best')
        plt.title(f'Pseudo-Accuracy vs Epoch (Comparison)')
        CENTRAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.tight_layout(rect=[0, 0, 0.85, 1] if num_items > 10 else None) # Adjust layout if legend is outside
        plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved combined pseudo-accuracy plot: {combined_plot_path}")
    else:
         logging.warning("No data found to plot combined pseudo-accuracy.")
    plt.close()


def plot_linear_probe(experiments_to_plot):
    """Plots Linear Probe Top-1 and Top-5 accuracy as horizontal bar charts for better readability."""
    top1_vals = []
    top5_vals = []
    valid_augs = []
    for aug_name in experiments_to_plot:
        top1, top5 = read_linear_probe(aug_name)
        if top1 is not None: # Only include if Top-1 exists
            top1_vals.append(top1)
            top5_vals.append(top5 if top5 is not None else 0) # Use 0 if Top-5 missing
            valid_augs.append(aug_name)

    if not valid_augs:
        logging.warning("No valid linear probe data found to plot.")
        return

    # --- Top-1 Plot ---
    plt.figure(figsize=(8, max(5, len(valid_augs) * 0.4))) # Adjust height based on number of experiments
    y_pos = range(len(valid_augs))
    plt.barh(y_pos, top1_vals) # Horizontal bars
    plt.yticks(y_pos, valid_augs) # Experiment names on y-axis
    plt.xlabel('Top-1 Accuracy (%)')
    plt.ylabel('Experiment')
    plt.title('Linear Probe Top-1 Accuracy Comparison')
    plt.grid(True, axis='x', linestyle='--', alpha=0.6) # Grid on x-axis
    plt.xlim(left=0) # Ensure x-axis starts at 0
    plt.gca().invert_yaxis() # Display top experiment at the top

    plot_path_top1 = CENTRAL_PLOTS_DIR / 'linear_probe_top1_comparison.png'
    CENTRAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path_top1, dpi=300, bbox_inches='tight')
    logging.info(f"Saved linear probe Top-1 plot: {plot_path_top1}")
    plt.close()

    # --- Top-5 Plot ---
    if any(v > 0 for v in top5_vals): # Only plot if there's valid Top-5 data
        plt.figure(figsize=(8, max(5, len(valid_augs) * 0.4))) # Adjust height
        y_pos = range(len(valid_augs))
        plt.barh(y_pos, top5_vals) # Horizontal bars
        plt.yticks(y_pos, valid_augs) # Experiment names on y-axis
        plt.xlabel('Top-5 Accuracy (%)')
        plt.ylabel('Experiment')
        plt.title('Linear Probe Top-5 Accuracy Comparison')
        plt.grid(True, axis='x', linestyle='--', alpha=0.6) # Grid on x-axis
        plt.xlim(left=0) # Ensure x-axis starts at 0
        plt.gca().invert_yaxis() # Display top experiment at the top

        plot_path_top5 = CENTRAL_PLOTS_DIR / 'linear_probe_top5_comparison.png'
        plt.tight_layout()
        plt.savefig(plot_path_top5, dpi=300, bbox_inches='tight')
        logging.info(f"Saved linear probe Top-5 plot: {plot_path_top5}")
        plt.close()
    else:
        logging.info("Skipping linear probe Top-5 plot as no valid Top-5 data was found.")


def plot_knn(experiments_to_plot):
    """Plots k-NN Top-1 accuracy as a grouped horizontal bar chart."""
    knn_data_dict = {} # {aug_name: {k: acc, ...}}
    valid_augs = []
    all_k_values = set()

    for aug_name in experiments_to_plot:
        results = read_knn(aug_name) # results is {k: acc}
        if results:
             valid_augs.append(aug_name)
             knn_data_dict[aug_name] = results
             all_k_values.update(results.keys())

    if not valid_augs:
        logging.warning("No valid k-NN data found to plot.")
        return

    sorted_k_values = sorted(list(all_k_values))
    num_ks = len(sorted_k_values)
    num_augs = len(valid_augs)

    plt.figure(figsize=(10, max(5, num_augs * 0.3 * num_ks))) # Adjust height based on experiments and k's
    y_indices = range(num_augs) # Indices for experiments [0, 1, 2,...]
    height = 0.8 / num_ks # Height of each bar within a group

    for i, k in enumerate(sorted_k_values):
        # Calculate offset for this k value's bars along the y-axis
        offset = (i - (num_ks - 1) / 2) * height
        k_accuracies = [knn_data_dict[aug].get(k, 0) for aug in valid_augs] # Get acc for this k for all augs
        plt.barh([y + offset for y in y_indices], k_accuracies, height=height, label=f'k={k}')

    plt.yticks([y for y in y_indices], valid_augs) # Place ticks at the center of the groups
    plt.xlabel('Top-1 Accuracy (%)')
    plt.ylabel('Experiment')
    plt.title('k-NN Classification Accuracy Comparison')
    plt.grid(True, axis='x', linestyle='--', alpha=0.6)
    plt.xlim(left=0)
    plt.legend(title="k Value")
    plt.gca().invert_yaxis() # Display top experiment at the top

    plot_path = CENTRAL_PLOTS_DIR / 'knn_top1_comparison.png'
    CENTRAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved k-NN comparison plot: {plot_path}")
    plt.close()


def plot_ablation_study(experiments_to_plot):
    """
    Generates plots specifically for the ablation study experiments.
    Assumes a baseline experiment (e.g., 'all_extended') and
    ablation experiments named with a prefix (e.g., 'all_minus_color').
    Plots absolute performance and performance drop relative to baseline.
    """
    ablation_experiments = [exp for exp in experiments_to_plot if exp.startswith(ABLATION_PREFIX)]
    baseline_exp = DEFAULT_ABLATION_BASELINE

    if baseline_exp not in experiments_to_plot:
        logging.warning(f"Ablation baseline '{baseline_exp}' not found in experiments list. Skipping ablation plots.")
        return
    if not ablation_experiments:
        logging.warning(f"No ablation experiments (starting with '{ABLATION_PREFIX}') found. Skipping ablation plots.")
        return

    baseline_top1, baseline_top5 = read_linear_probe(baseline_exp)
    if baseline_top1 is None:
        logging.warning(f"Could not read results for ablation baseline '{baseline_exp}'. Skipping ablation plots.")
        return

    ablation_data = defaultdict(dict) # {removed_aug: {'name': name, 'top1': t1, 'top5': t5}}
    valid_ablations = []

    for exp_name in ablation_experiments:
        top1, top5 = read_linear_probe(exp_name)
        if top1 is not None:
            removed_aug = exp_name.replace(ABLATION_PREFIX, "") # Extract removed augmentation name
            ablation_data[removed_aug]['name'] = exp_name
            ablation_data[removed_aug]['top1'] = top1
            ablation_data[removed_aug]['top5'] = top5 if top5 is not None else 0
            valid_ablations.append(removed_aug)

    if not valid_ablations:
        logging.warning("Could not read results for any ablation experiments. Skipping ablation plots.")
        return

    sorted_removed_augs = sorted(valid_ablations)

    # --- Plot 1: Absolute Performance Comparison ---
    plt.figure(figsize=(8, max(5, len(sorted_removed_augs) * 0.5)))
    y_pos = range(len(sorted_removed_augs) + 1) # +1 for baseline
    labels = [f"Baseline ({baseline_exp})"] + [f"- {aug}" for aug in sorted_removed_augs]
    top1_scores = [baseline_top1] + [ablation_data[aug]['top1'] for aug in sorted_removed_augs]

    plt.barh(y_pos, top1_scores)
    plt.yticks(y_pos, labels)
    plt.xlabel('Linear Probe Top-1 Accuracy (%)')
    plt.ylabel('Experiment (Baseline vs. Augmentation Removed)')
    plt.title('Ablation Study: Absolute Performance')
    plt.grid(True, axis='x', linestyle='--', alpha=0.6)
    plt.xlim(left=0)
    plt.gca().invert_yaxis()

    plot_path_abs = CENTRAL_PLOTS_DIR / 'ablation_absolute_perf_top1.png'
    CENTRAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path_abs, dpi=300, bbox_inches='tight')
    logging.info(f"Saved ablation absolute performance plot: {plot_path_abs}")
    plt.close()

    # --- Plot 2: Performance Drop Relative to Baseline ---
    plt.figure(figsize=(8, max(5, len(sorted_removed_augs) * 0.4)))
    y_pos_drop = range(len(sorted_removed_augs))
    labels_drop = [f"{aug}" for aug in sorted_removed_augs] # Label is the removed aug
    drops = [baseline_top1 - ablation_data[aug]['top1'] for aug in sorted_removed_augs]

    colors = ['red' if drop > 0 else 'green' for drop in drops] # Red for drop, green for gain (unlikely)
    plt.barh(y_pos_drop, drops, color=colors)
    plt.yticks(y_pos_drop, labels_drop)
    plt.xlabel('Performance Drop (Top-1 Accuracy % Points)')
    plt.ylabel('Removed Augmentation')
    plt.title(f'Ablation Study: Impact of Removing Augmentations (Baseline: {baseline_exp})')
    plt.grid(True, axis='x', linestyle='--', alpha=0.6)
    plt.axvline(0, color='grey', linewidth=0.8) # Line at zero drop

    plot_path_drop = CENTRAL_PLOTS_DIR / 'ablation_performance_drop_top1.png'
    plt.tight_layout()
    plt.savefig(plot_path_drop, dpi=300, bbox_inches='tight')
    logging.info(f"Saved ablation performance drop plot: {plot_path_drop}")
    plt.close()


# ====== NEW: Summary Report Function ======
def generate_summary_report(experiments_to_summarize, output_file_path):
    """
    Generates a text summary report (Markdown table) of key evaluation metrics
    for the completed experiments.
    """
    logging.info(f"Generating summary report for: {experiments_to_summarize}")
    results_data = []

    for aug_name in experiments_to_summarize:
        top1, top5 = read_linear_probe(aug_name)
        knn_results = read_knn(aug_name) # Returns dict {k: acc}

        # Prepare data for this row, handling missing values
        row_data = {"Experiment": aug_name}
        row_data["Linear Top-1 (%)"] = f"{top1:.2f}" if top1 is not None else "N/A"
        row_data["Linear Top-5 (%)"] = f"{top5:.2f}" if top5 is not None else "N/A"

        for k in KNN_K_VALUES_FOR_SUMMARY:
            acc = knn_results.get(k)
            row_data[f"k-NN Top-1 (k={k}) (%)"] = f"{acc:.2f}" if acc is not None else "N/A"

        results_data.append(row_data)

    if not results_data:
        logging.warning("No results found to generate summary report.")
        return

    # Use pandas to create and format the table easily
    df = pd.DataFrame(results_data)

    # Optional: Sort by experiment name or a specific metric
    df = df.sort_values(by="Experiment")
    # Example sort by Linear Top-1 (requires handling 'N/A')
    # df['SortMetric'] = pd.to_numeric(df['Linear Top-1 (%)'], errors='coerce')
    # df = df.sort_values(by='SortMetric', ascending=False).drop('SortMetric', axis=1)


    # Convert DataFrame to Markdown string
    try:
        # Ensure pandas >= 1.0 for to_markdown
        markdown_table = df.to_markdown(index=False, floatfmt=".2f")
        report_content = "# Experiment Summary Report\n\n"
        report_content += markdown_table
        report_content += "\n\n*N/A indicates results file was missing or could not be parsed.*\n"
    except AttributeError:
        logging.warning("Pandas < 1.0 detected, cannot use to_markdown. Saving as CSV instead.")
        report_content = df.to_csv(index=False)
        output_file_path = output_file_path.with_suffix(".csv") # Change extension

    # Write the report to the file
    try:
        with open(output_file_path, 'w') as f:
            f.write(report_content)
        logging.info(f"Summary report saved to: {output_file_path}")
    except Exception as e:
        logging.error(f"Failed to write summary report to {output_file_path}: {e}")


# ====== Runner ======
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate improved plots and summary report for SimCLR experiments.")
    parser.add_argument(
        "--experiments",
        nargs='+',
        required=True,
        help="List of experiment augmentation names to plot/summarize (e.g., baseline color all_extended all_minus_color)."
    )
    args = parser.parse_args()

    logging.info(f"Processing results for experiments: {args.experiments}")

    # Ensure central plots directory exists
    CENTRAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Generate Plots ---
    logging.info("\n--- Generating Plots ---")
    plot_loss(args.experiments)
    plot_pseudo_accuracy(args.experiments)
    plot_linear_probe(args.experiments) # Now plots Top-1 and Top-5 horizontally
    plot_knn(args.experiments) # Now plots grouped horizontal bars
    plot_ablation_study(args.experiments) # New function for ablation plots
    logging.info("--- Plot generation finished. ---")


    # --- Generate Summary Report ---
    logging.info("\n--- Generating Summary Report ---")
    generate_summary_report(args.experiments, SUMMARY_REPORT_FILE)
    logging.info("--- Summary report generation finished. ---")


    print(f"\nResult processing complete.")
    print(f"Check individual run directories and '{CENTRAL_PLOTS_DIR}' for plots.")
    print(f"Check '{SUMMARY_REPORT_FILE}' for the summary report.")