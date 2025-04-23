# Utils/plotting.py
# Updated plotting script to read from a single consolidated results CSV.

import argparse
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import logging
import numpy as np
import sys
from pathlib import Path
from collections import defaultdict
import itertools # For line styles
# No longer need glob

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s [Plotter] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# ====== Config ======
RESULTS_DIR = Path('results')
CENTRAL_PLOTS_DIR = RESULTS_DIR / 'plots'
# --- !! Path to the consolidated results file !! ---
CONSOLIDATED_RESULTS_CSV = RESULTS_DIR / "consolidated_results.csv"
SUMMARY_REPORT_FILE = RESULTS_DIR / 'summary_report_final_epoch.md' # Report still uses final epoch
MODEL_NAME = "simclr" # Used for reading training loss CSVs if needed
DEFAULT_ABLATION_BASELINE = 'all_extended' # *** IMPORTANT: Set this to the name of your full augmentation combo experiment ***
ABLATION_PREFIX = 'all_minus_' # Prefix for experiments where one augmentation is removed
KNN_K_VALUES_FOR_SUMMARY = [1, 5, 10, 20] # K values to include in summary report
DEFAULT_KNN_K_FOR_EVOLUTION_PLOT = 10 # Which k value to plot

# Plotting styles
LINE_STYLES = ['-', '--', '-.', ':']
MARKER_STYLES = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
matplotlib.rcParams.update({
    "font.size": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.figsize": (10, 6) # Default figure size
})

# ====== Data Loading Function ======
def load_consolidated_results(csv_path=CONSOLIDATED_RESULTS_CSV):
    """Loads the consolidated results CSV into a pandas DataFrame."""
    csv_path = Path(csv_path) # Ensure it's a Path object
    if not csv_path.exists():
        logging.error(f"Consolidated results file not found: {csv_path}")
        # Optionally, try to find it elsewhere or provide more guidance
        return None
    try:
        df = pd.read_csv(csv_path)
        logging.info(f"Successfully loaded consolidated results ({len(df)} rows) from: {csv_path}")
        # Basic validation
        expected_cols = ['experiment_name', 'epoch', 'metric_type', 'metric_name', 'k_value', 'value']
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
             logging.warning(f"Consolidated CSV missing expected columns: {missing_cols}. Found: {list(df.columns)}")
             # Attempt to continue if possible, or return None depending on severity
        # Convert relevant columns to numeric, coercing errors
        df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df['k_value'] = pd.to_numeric(df['k_value'], errors='coerce')
        # Drop rows where essential numeric conversions failed
        df.dropna(subset=['epoch', 'value'], inplace=True)
        return df
    except Exception as e:
        logging.error(f"Error loading or parsing consolidated results file {csv_path}: {e}", exc_info=True)
        return None


# ====== Plotters (Modified to use DataFrame) ======

# plot_loss and plot_pseudo_accuracy still read individual training_loss CSVs
# as these contain metrics *during* training, not the evaluation metrics.
def plot_loss(experiments_to_plot):
    """Plots individual and combined loss curves (reads training_loss CSV)."""
    plt.figure() # For combined plot
    style_cycler = itertools.cycle(LINE_STYLES)
    color_cycler = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    did_plot = False

    for aug_name in experiments_to_plot:
        # This function reads the single training loss CSV, not epoch-specific files
        run_dir = RESULTS_DIR / f'{MODEL_NAME}_{aug_name}'
        loss_csv_path = run_dir / f'training_loss_{MODEL_NAME}_{aug_name}.csv'
        df = None
        try:
            df = pd.read_csv(loss_csv_path)
            logging.debug(f"Read training loss from {loss_csv_path}")
        except FileNotFoundError:
            logging.warning(f"Training loss file not found: {loss_csv_path}")
            continue # Skip this experiment for loss plot
        except Exception as e:
            logging.error(f"Error reading {loss_csv_path}: {e}")
            continue

        if df is not None and not df.empty and 'epoch' in df.columns and 'avg_loss' in df.columns:
            did_plot = True
            run_dir.mkdir(parents=True, exist_ok=True)
            individual_plot_path = run_dir / f'loss_{aug_name}.png'
            combined_plot_path = CENTRAL_PLOTS_DIR / 'loss_all_comparison.png'

            # Individual Plot
            plt.figure(figsize=(8, 5))
            plt.plot(df['epoch'], df['avg_loss'])
            plt.xlabel('Epoch'); plt.ylabel('InfoNCE Loss')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.title(f'InfoNCE Loss vs Epoch ({aug_name})')
            plt.tight_layout()
            plt.savefig(individual_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"Saved individual loss plot: {individual_plot_path}")

            # Add to combined plot
            plt.figure(1) # Switch back to combined plot figure (figure number 1)
            plt.plot(df['epoch'], df['avg_loss'],
                     label=aug_name,
                     linestyle=next(style_cycler),
                     color=next(color_cycler)) # Use style and color cyclers
        elif df is not None:
             logging.warning(f"Skipping loss plot for {aug_name}: Missing 'epoch' or 'avg_loss' column in {loss_csv_path}")


    # Finalize Combined Plot
    plt.figure(1) # Ensure we are on the combined plot figure
    if did_plot: # Check if any data was actually plotted
        plt.xlabel('Epoch'); plt.ylabel('InfoNCE Loss')
        plt.grid(True, linestyle='--', alpha=0.6)
        # Adjust legend position if too many items
        num_items = len(plt.gca().get_lines())
        legend_props = {'bbox_to_anchor':(1.05, 1), 'loc':'upper left', 'borderaxespad':0.} if num_items > 10 else {'loc':'best'}
        plt.legend(**legend_props)
        plt.title(f'InfoNCE Loss vs Epoch (Comparison)')
        CENTRAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.tight_layout(rect=[0, 0, 0.85, 1] if num_items > 10 else None) # Adjust layout if legend is outside
        plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved combined loss plot: {combined_plot_path}")
    else:
         logging.warning("No data found to plot combined loss.")
    plt.close() # Close combined plot figure


def plot_pseudo_accuracy(experiments_to_plot):
    """Plots individual and combined pseudo-accuracy curves (reads training_loss CSV)."""
    plt.figure() # For combined plot
    style_cycler = itertools.cycle(LINE_STYLES)
    color_cycler = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    did_plot = False

    for aug_name in experiments_to_plot:
        # Reads from the single training loss CSV
        run_dir = RESULTS_DIR / f'{MODEL_NAME}_{aug_name}'
        loss_csv_path = run_dir / f'training_loss_{MODEL_NAME}_{aug_name}.csv'
        df = None
        try:
            df = pd.read_csv(loss_csv_path)
            logging.debug(f"Read training loss for pseudo-acc from {loss_csv_path}")
        except FileNotFoundError:
            logging.warning(f"Training loss file not found for pseudo-acc: {loss_csv_path}")
            continue
        except Exception as e:
            logging.error(f"Error reading {loss_csv_path} for pseudo-acc: {e}")
            continue

        # --- !! Use the correct column name from simclr.py !! ---
        acc_col_name = 'avg_contrastive_acc' # Make sure this matches the column saved in simclr.py history

        if df is not None and not df.empty and 'epoch' in df.columns and acc_col_name in df.columns:
            did_plot = True
            run_dir.mkdir(parents=True, exist_ok=True)
            individual_plot_path = run_dir / f'pseudo_acc_{aug_name}.png'
            combined_plot_path = CENTRAL_PLOTS_DIR / 'pseudo_acc_all_comparison.png'

            # Individual Plot
            plt.figure(figsize=(8, 5)) # Smaller individual plot
            plt.plot(df['epoch'], df[acc_col_name]) # Use correct column name
            plt.xlabel('Epoch'); plt.ylabel('Pseudo Accuracy (InfoNCE) (%)')
            plt.grid(True, linestyle='--', alpha=0.6); plt.ylim(bottom=0) # Ensure y-axis starts at 0
            plt.title(f'Pseudo-Accuracy vs Epoch ({aug_name})')
            plt.tight_layout()
            plt.savefig(individual_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"Saved individual pseudo-accuracy plot: {individual_plot_path}")

            # Add to combined plot
            plt.figure(1)
            plt.plot(df['epoch'], df[acc_col_name], # Use correct column name
                     label=aug_name,
                     linestyle=next(style_cycler),
                     color=next(color_cycler)) # Use style and color cyclers
        elif df is not None:
             logging.warning(f"Skipping pseudo-acc plot for {aug_name}: Missing 'epoch' or '{acc_col_name}' column in {loss_csv_path}")


    # Finalize Combined Plot
    plt.figure(1)
    if did_plot:
        plt.xlabel('Epoch'); plt.ylabel('Pseudo Accuracy (InfoNCE) (%)')
        plt.grid(True, linestyle='--', alpha=0.6); plt.ylim(bottom=0)
        # Adjust legend position if too many items
        num_items = len(plt.gca().get_lines())
        legend_props = {'bbox_to_anchor':(1.05, 1), 'loc':'upper left', 'borderaxespad':0.} if num_items > 10 else {'loc':'best'}
        plt.legend(**legend_props)
        plt.title(f'Pseudo-Accuracy vs Epoch (Comparison)')
        CENTRAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.tight_layout(rect=[0, 0, 0.85, 1] if num_items > 10 else None) # Adjust layout if legend is outside
        plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved combined pseudo-accuracy plot: {combined_plot_path}")
    else:
         logging.warning("No data found to plot combined pseudo-accuracy.")
    plt.close()


def plot_linear_probe(results_df: pd.DataFrame, experiments_to_plot: list):
    """Plots Linear Probe Top-1/Top-5 (latest epoch) from the results DataFrame."""
    if results_df is None or results_df.empty:
        logging.warning("Consolidated results DataFrame is empty. Skipping linear probe plots.")
        return

    # Filter for relevant experiments and metric type
    df_filtered = results_df[
        results_df['experiment_name'].isin(experiments_to_plot) &
        (results_df['metric_type'] == 'LinearProbe')
    ].copy()

    if df_filtered.empty:
        logging.warning(f"No LinearProbe data found for experiments: {experiments_to_plot} in consolidated results.")
        return

    # Find the latest epoch for each experiment
    # Group by experiment, find max epoch index within each group, select those rows
    latest_indices = df_filtered.groupby('experiment_name')['epoch'].idxmax()
    latest_epochs = df_filtered.loc[latest_indices]

    # Separate Top-1 and Top-5 data for the latest epoch
    latest_top1 = latest_epochs[latest_epochs['metric_name'] == 'Top1Acc'].set_index('experiment_name')['value']
    latest_top5 = latest_epochs[latest_epochs['metric_name'] == 'Top5Acc'].set_index('experiment_name')['value']

    # Get experiment names that have at least Top1 data
    valid_augs = latest_top1.index.unique().tolist()

    if not valid_augs:
        logging.warning("No valid latest Linear Probe Top-1 data found in consolidated results.")
        return

    # Reindex both series to ensure they match the valid_augs order, fill missing Top-1 with 0 (shouldn't happen if valid_augs based on latest_top1)
    latest_top1 = latest_top1.reindex(valid_augs, fill_value=0)


    # --- Top-1 Plot ---
    plt.figure(figsize=(8, max(5, len(valid_augs) * 0.4)))
    y_pos = range(len(valid_augs))
    plt.barh(y_pos, latest_top1.values) # Use reindexed values
    plt.yticks(y_pos, valid_augs) # Use reindexed labels
    plt.xlabel('Top-1 Accuracy (%) [Latest Epoch]')
    plt.ylabel('Experiment')
    plt.title('Linear Probe Top-1 Accuracy Comparison (Latest Epoch)')
    plt.grid(True, axis='x', linestyle='--', alpha=0.6); plt.xlim(left=0); plt.gca().invert_yaxis()
    plot_path_top1 = CENTRAL_PLOTS_DIR / 'linear_probe_top1_comparison_latest.png'
    CENTRAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True); plt.tight_layout()
    try:
        plt.savefig(plot_path_top1, dpi=300, bbox_inches='tight')
        logging.info(f"Saved linear probe Top-1 plot (latest epoch): {plot_path_top1}")
    except Exception as e:
        logging.error(f"Failed to save plot {plot_path_top1}: {e}")
    plt.close()

    # --- Top-5 Plot ---
    if not latest_top5.empty:
        # Reindex Top5 to match Top1 order, fill missing with 0
        latest_top5 = latest_top5.reindex(valid_augs, fill_value=0)

        plt.figure(figsize=(8, max(5, len(valid_augs) * 0.4)))
        y_pos = range(len(valid_augs))
        plt.barh(y_pos, latest_top5.values) # Use reindexed values
        plt.yticks(y_pos, valid_augs) # Use reindexed labels
        plt.xlabel('Top-5 Accuracy (%) [Latest Epoch]')
        plt.ylabel('Experiment')
        plt.title('Linear Probe Top-5 Accuracy Comparison (Latest Epoch)')
        plt.grid(True, axis='x', linestyle='--', alpha=0.6); plt.xlim(left=0); plt.gca().invert_yaxis()
        plot_path_top5 = CENTRAL_PLOTS_DIR / 'linear_probe_top5_comparison_latest.png'
        plt.tight_layout()
        try:
            plt.savefig(plot_path_top5, dpi=300, bbox_inches='tight')
            logging.info(f"Saved linear probe Top-5 plot (latest epoch): {plot_path_top5}")
        except Exception as e:
             logging.error(f"Failed to save plot {plot_path_top5}: {e}")
        plt.close()
    else:
        logging.info("Skipping linear probe Top-5 plot (no latest Top-5 data found).")


def plot_knn(results_df: pd.DataFrame, experiments_to_plot: list):
    """Plots k-NN Top-1 accuracy (latest epoch) as grouped horizontal bar chart."""
    if results_df is None or results_df.empty:
        logging.warning("Consolidated results DataFrame is empty. Skipping k-NN plots.")
        return

    # Filter for relevant experiments and metric type/name
    df_filtered = results_df[
        results_df['experiment_name'].isin(experiments_to_plot) &
        (results_df['metric_type'] == 'kNN') &
        (results_df['metric_name'] == 'Top1Acc') # Assuming kNN results are Top1
    ].copy()

    if df_filtered.empty:
        logging.warning(f"No kNN Top1Acc data found for experiments: {experiments_to_plot} in consolidated results.")
        return

    # Find the latest epoch results for each experiment and k_value
    latest_indices = df_filtered.groupby(['experiment_name', 'k_value'])['epoch'].idxmax()
    latest_epoch_data = df_filtered.loc[latest_indices]

    # Pivot table to get experiments as index, k_values as columns, accuracy as values
    try:
        # Need to handle cases where an experiment might miss a k_value for the latest epoch
        pivot_df = latest_epoch_data.pivot_table(index='experiment_name', columns='k_value', values='value')
    except Exception as e:
        logging.error(f"Could not pivot k-NN data: {e}. Data:\n{latest_epoch_data}")
        return

    # Get valid experiments and k values after pivoting
    valid_augs = pivot_df.index.unique().tolist()
    # Ensure k_values are numeric and sorted
    sorted_k_values = sorted([k for k in pivot_df.columns if pd.notna(k)])


    if not valid_augs or not sorted_k_values:
        logging.warning("No valid latest k-NN data found after pivoting.")
        return

    # Plotting logic (similar to before, but uses pivot_df)
    num_ks = len(sorted_k_values)
    num_augs = len(valid_augs)
    plt.figure(figsize=(10, max(5, num_augs * 0.3 * num_ks)))
    y_indices = np.arange(num_augs) # Use numpy arange for consistent indexing
    height = 0.8 / num_ks

    for i, k in enumerate(sorted_k_values):
        offset = (i - (num_ks - 1) / 2) * height
        # Get accuracies for this k, handling potential NaNs from pivot
        # Ensure we access data aligned with valid_augs order
        k_accuracies = pivot_df.loc[valid_augs, k].fillna(0).values
        plt.barh(y_indices + offset, k_accuracies, height=height, label=f'k={k}')

    plt.yticks(y_indices, valid_augs) # Place ticks at the center of the groups
    plt.xlabel('Top-1 Accuracy (%) [Latest Epoch]')
    plt.ylabel('Experiment')
    plt.title('k-NN Classification Accuracy Comparison (Latest Epoch)')
    plt.grid(True, axis='x', linestyle='--', alpha=0.6); plt.xlim(left=0)
    plt.legend(title="k Value", bbox_to_anchor=(1.05, 1), loc='upper left'); plt.gca().invert_yaxis()
    plot_path = CENTRAL_PLOTS_DIR / 'knn_top1_comparison_latest.png'
    CENTRAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True); plt.tight_layout(rect=[0, 0, 0.9, 1])
    try:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved k-NN comparison plot (latest epoch): {plot_path}")
    except Exception as e:
        logging.error(f"Failed to save plot {plot_path}: {e}")
    plt.close()


def plot_ablation_study(results_df: pd.DataFrame, experiments_to_plot: list):
    """Generates ablation plots using latest epoch linear probe Top-1 results from the DataFrame."""
    if results_df is None or results_df.empty:
        logging.warning("Consolidated results DataFrame is empty. Skipping ablation plots.")
        return

    ablation_experiments = [exp for exp in experiments_to_plot if exp.startswith(ABLATION_PREFIX)]
    baseline_exp = DEFAULT_ABLATION_BASELINE

    # Check if baseline experiment name exists in the provided list
    if baseline_exp not in experiments_to_plot:
        logging.warning(f"Ablation baseline '{baseline_exp}' not found in the list of experiments to plot: {experiments_to_plot}. Skipping ablation plots.")
        return
    # Check if any ablation experiment names exist in the list
    actual_ablation_exps_in_list = [exp for exp in ablation_experiments if exp in experiments_to_plot]
    if not actual_ablation_exps_in_list:
        logging.warning(f"No ablation experiments (starting with '{ABLATION_PREFIX}') found in the list of experiments to plot: {experiments_to_plot}. Skipping ablation plots.")
        return

    # Filter data for Top-1 Linear Probe results for relevant experiments
    df_filtered = results_df[
        results_df['experiment_name'].isin(experiments_to_plot) & # Filter by provided list
        (results_df['metric_type'] == 'LinearProbe') &
        (results_df['metric_name'] == 'Top1Acc')
    ].copy()

    if df_filtered.empty:
        logging.warning("No LinearProbe Top1Acc data found for ablation relevant experiments.")
        return

    # Get latest epoch data for these experiments
    latest_indices = df_filtered.groupby('experiment_name')['epoch'].idxmax()
    latest_accuracies = df_filtered.loc[latest_indices].set_index('experiment_name')['value']

    # Get baseline accuracy
    if baseline_exp not in latest_accuracies.index:
         logging.warning(f"Could not find latest result for ablation baseline '{baseline_exp}' in consolidated data. Skipping ablation plots.")
         return
    baseline_top1 = latest_accuracies[baseline_exp]

    # Get ablation accuracies for experiments that are BOTH in the ablation list AND have data
    ablation_data = {} # {removed_aug: top1}
    valid_ablations = []
    for exp_name in actual_ablation_exps_in_list: # Iterate only through relevant ablation exps
        if exp_name in latest_accuracies.index:
            removed_aug = exp_name.replace(ABLATION_PREFIX, "")
            ablation_data[removed_aug] = latest_accuracies[exp_name]
            valid_ablations.append(removed_aug)
        else:
             logging.warning(f"Could not find latest result for ablation exp: {exp_name}")

    if not valid_ablations:
        logging.warning("No valid latest results found for any specified ablation experiments.")
        return

    sorted_removed_augs = sorted(valid_ablations)

    # Plotting logic (uses extracted latest_accuracies)
    # --- Plot 1: Absolute Performance ---
    plt.figure(figsize=(8, max(5, len(sorted_removed_augs) * 0.5)))
    y_pos = range(len(sorted_removed_augs) + 1); labels = [f"Baseline ({baseline_exp})"] + [f"- {aug}" for aug in sorted_removed_augs]
    top1_scores = [baseline_top1] + [ablation_data[aug] for aug in sorted_removed_augs]
    plt.barh(y_pos, top1_scores); plt.yticks(y_pos, labels); plt.xlabel('Linear Probe Top-1 Accuracy (%) [Latest Epoch]'); plt.ylabel('Experiment')
    plt.title('Ablation Study: Absolute Performance (Latest Epoch)'); plt.grid(True, axis='x', linestyle='--', alpha=0.6); plt.xlim(left=min(0, min(top1_scores)-5)); plt.gca().invert_yaxis()
    plot_path_abs = CENTRAL_PLOTS_DIR / 'ablation_absolute_perf_top1_latest.png'
    CENTRAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True); plt.tight_layout()
    try: plt.savefig(plot_path_abs, dpi=300, bbox_inches='tight'); logging.info(f"Saved ablation absolute plot: {plot_path_abs}")
    except Exception as e: logging.error(f"Failed to save plot {plot_path_abs}: {e}")
    plt.close()

    # --- Plot 2: Performance Drop ---
    plt.figure(figsize=(8, max(5, len(sorted_removed_augs) * 0.4)))
    y_pos_drop = range(len(sorted_removed_augs)); labels_drop = [f"{aug}" for aug in sorted_removed_augs]
    drops = [baseline_top1 - ablation_data[aug] for aug in sorted_removed_augs]; colors = ['#d62728' if drop > 0 else '#2ca02c' for drop in drops] # Red for drop, green for gain/no change
    plt.barh(y_pos_drop, drops, color=colors); plt.yticks(y_pos_drop, labels_drop); plt.xlabel('Performance Drop (Top-1 Accuracy % Points) [Latest Epoch]'); plt.ylabel('Removed Augmentation')
    plt.title(f'Ablation Study: Impact of Removing Augmentations (Latest Epoch)\nBaseline: {baseline_exp} ({baseline_top1:.2f}%)'); plt.grid(True, axis='x', linestyle='--', alpha=0.6); plt.axvline(0, color='grey', linewidth=0.8)
    plot_path_drop = CENTRAL_PLOTS_DIR / 'ablation_performance_drop_top1_latest.png'
    plt.tight_layout()
    try: plt.savefig(plot_path_drop, dpi=300, bbox_inches='tight'); logging.info(f"Saved ablation drop plot: {plot_path_drop}")
    except Exception as e: logging.error(f"Failed to save plot {plot_path_drop}: {e}")
    plt.close()


# ====== Evolution Plotting Functions (Using DataFrame) ======

def plot_linear_evolution(results_df: pd.DataFrame, experiments_to_plot: list):
    """Plots the evolution of Linear Probe Top-1 accuracy over saved epochs from the DataFrame."""
    if results_df is None or results_df.empty:
        logging.warning("Consolidated results DataFrame is empty. Skipping linear evolution plot.")
        return

    df_filtered = results_df[
        results_df['experiment_name'].isin(experiments_to_plot) &
        (results_df['metric_type'] == 'LinearProbe') &
        (results_df['metric_name'] == 'Top1Acc')
    ].copy()

    if df_filtered.empty:
        logging.warning(f"No LinearProbe Top1Acc data found for evolution plot for experiments: {experiments_to_plot}")
        return

    plt.figure()
    style_cycler = itertools.cycle(LINE_STYLES)
    marker_cycler = itertools.cycle(MARKER_STYLES)
    color_cycler = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    has_data = False

    # Group by experiment and plot
    for exp_name, group in df_filtered.groupby('experiment_name'):
        group = group.sort_values('epoch') # Ensure data is sorted by epoch
        if not group.empty:
            has_data = True
            plt.plot(group['epoch'], group['value'],
                     label=exp_name,
                     linestyle=next(style_cycler), marker=next(marker_cycler),
                     color=next(color_cycler))

    if has_data:
        plt.xlabel('Epoch'); plt.ylabel('Linear Probe Top-1 Accuracy (%)')
        plt.title('Linear Probe Top-1 Accuracy Evolution'); plt.grid(True, linestyle='--', alpha=0.6); plt.ylim(bottom=0)
        # Adjust legend position
        num_items = len(plt.gca().get_lines()); legend_props = {'bbox_to_anchor':(1.05, 1), 'loc':'upper left', 'borderaxespad':0.} if num_items > 10 else {'loc':'best'}
        plt.legend(**legend_props)
        CENTRAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True); plt.tight_layout(rect=[0, 0, 0.85, 1] if num_items > 10 else None)
        plot_path = CENTRAL_PLOTS_DIR / 'linear_probe_top1_evolution.png'
        try: plt.savefig(plot_path, dpi=300, bbox_inches='tight'); logging.info(f"Saved linear probe evolution plot: {plot_path}")
        except Exception as e: logging.error(f"Failed to save plot {plot_path}: {e}")
    else:
        logging.warning("No data found to plot linear probe evolution.")
    plt.close()


def plot_knn_evolution(results_df: pd.DataFrame, experiments_to_plot: list, k_value=DEFAULT_KNN_K_FOR_EVOLUTION_PLOT):
    """Plots k-NN Top-1 accuracy evolution for a specific k from the DataFrame."""
    if results_df is None or results_df.empty:
        logging.warning("Consolidated results DataFrame is empty. Skipping kNN evolution plot.")
        return

    df_filtered = results_df[
        results_df['experiment_name'].isin(experiments_to_plot) &
        (results_df['metric_type'] == 'kNN') &
        (results_df['metric_name'] == 'Top1Acc') &
        # Ensure k_value comparison handles potential NaN/float issues
        (results_df['k_value'].fillna(-1).astype(int) == k_value) # Filter for specific k
    ].copy()

    if df_filtered.empty:
        logging.warning(f"No kNN Top1Acc (k={k_value}) data found for evolution plot for experiments: {experiments_to_plot}")
        return

    plt.figure()
    style_cycler = itertools.cycle(LINE_STYLES)
    marker_cycler = itertools.cycle(MARKER_STYLES)
    color_cycler = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    has_data = False

    # Group by experiment and plot
    for exp_name, group in df_filtered.groupby('experiment_name'):
        group = group.sort_values('epoch')
        if not group.empty:
            has_data = True
            plt.plot(group['epoch'], group['value'],
                     label=exp_name,
                     linestyle=next(style_cycler), marker=next(marker_cycler),
                     color=next(color_cycler))

    if has_data:
        plt.xlabel('Epoch'); plt.ylabel(f'k-NN Top-1 Accuracy (k={k_value}) (%)')
        plt.title(f'k-NN Top-1 Accuracy (k={k_value}) Evolution'); plt.grid(True, linestyle='--', alpha=0.6); plt.ylim(bottom=0)
        # Adjust legend position
        num_items = len(plt.gca().get_lines()); legend_props = {'bbox_to_anchor':(1.05, 1), 'loc':'upper left', 'borderaxespad':0.} if num_items > 10 else {'loc':'best'}
        plt.legend(**legend_props)
        CENTRAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True); plt.tight_layout(rect=[0, 0, 0.85, 1] if num_items > 10 else None)
        plot_path = CENTRAL_PLOTS_DIR / f'knn_top1_k{k_value}_evolution.png'
        try: plt.savefig(plot_path, dpi=300, bbox_inches='tight'); logging.info(f"Saved k-NN (k={k_value}) evolution plot: {plot_path}")
        except Exception as e: logging.error(f"Failed to save plot {plot_path}: {e}")
    else:
        logging.warning(f"No data found to plot k-NN (k={k_value}) evolution.")
    plt.close()


# ====== Summary Report (Using Latest Epoch from DataFrame) ======
def generate_summary_report(results_df: pd.DataFrame, experiments_to_plot: list, output_file_path: Path):
    """Generates Markdown summary using latest epoch results from the DataFrame."""
    if results_df is None or results_df.empty:
        logging.warning("Consolidated results DataFrame is empty. Skipping summary report.")
        return

    df_filtered = results_df[results_df['experiment_name'].isin(experiments_to_plot)].copy()
    if df_filtered.empty:
        logging.warning(f"No data found for specified experiments in summary: {experiments_to_plot}")
        return

    # Find latest epoch results for each metric group
    latest_indices = df_filtered.groupby(['experiment_name', 'metric_type', 'metric_name', 'k_value'], dropna=False)['epoch'].idxmax()
    latest_results = df_filtered.loc[latest_indices].reset_index(drop=True) # Use reset_index for easier access

    # Pivot for easier table creation
    summary_data = []
    # Ensure experiments_to_plot only contains experiments present in latest_results
    valid_experiments_for_summary = sorted(list(latest_results['experiment_name'].unique()))

    for exp_name in valid_experiments_for_summary:
        exp_data = latest_results[latest_results['experiment_name'] == exp_name]
        if exp_data.empty: continue # Should not happen if valid_experiments_for_summary is correct

        row = {"Experiment": exp_name}
        # Get Linear Probe results
        lp_top1_row = exp_data[(exp_data['metric_type'] == 'LinearProbe') & (exp_data['metric_name'] == 'Top1Acc')]
        lp_top5_row = exp_data[(exp_data['metric_type'] == 'LinearProbe') & (exp_data['metric_name'] == 'Top5Acc')]
        lp_top1 = lp_top1_row['value'].iloc[0] if not lp_top1_row.empty else None
        lp_top5 = lp_top5_row['value'].iloc[0] if not lp_top5_row.empty else None
        row["Linear Top-1 (%)"] = f"{lp_top1:.2f}" if lp_top1 is not None else "N/A"
        row["Linear Top-5 (%)"] = f"{lp_top5:.2f}" if lp_top5 is not None else "N/A"

        # Get k-NN results
        knn_data = exp_data[(exp_data['metric_type'] == 'kNN') & (exp_data['metric_name'] == 'Top1Acc')]
        for k in KNN_K_VALUES_FOR_SUMMARY:
            # Need to handle k_value being float after read_csv sometimes
            knn_val_row = knn_data[knn_data['k_value'].fillna(-1).astype(int) == k]
            knn_val = knn_val_row['value'].iloc[0] if not knn_val_row.empty else None
            row[f"k-NN Top-1 (k={k}) (%)"] = f"{knn_val:.2f}" if knn_val is not None else "N/A"
        summary_data.append(row)

    if not summary_data:
        logging.warning("No latest results to build summary table.")
        return

    summary_df = pd.DataFrame(summary_data)
    # Ensure column order is consistent
    cols_order = ["Experiment", "Linear Top-1 (%)", "Linear Top-5 (%)"] + \
                 [f"k-NN Top-1 (k={k}) (%)" for k in KNN_K_VALUES_FOR_SUMMARY]
    summary_df = summary_df[cols_order]
    # Sort by experiment name (already sorted if valid_experiments_for_summary was sorted)
    # summary_df = summary_df.sort_values(by="Experiment")


    # Save as Markdown
    try:
        markdown_table = summary_df.to_markdown(index=False, floatfmt=".2f")
        report_content = "# Experiment Summary Report (Latest Epoch Results)\n\n" + markdown_table + "\n\n*N/A indicates results were missing for the latest epoch.*\n"
        output_file_path_md = output_file_path.with_suffix(".md")
        with open(output_file_path_md, 'w') as f: f.write(report_content)
        logging.info(f"Summary report (latest epoch) saved to: {output_file_path_md}")
    except Exception as e:
        logging.error(f"Failed to write summary report: {e}")
        # Optional: Save as CSV fallback
        try:
             output_file_path_csv = output_file_path.with_suffix(".csv")
             summary_df.to_csv(output_file_path_csv, index=False, floatfmt="%.2f")
             logging.info(f"Saved summary report fallback as CSV: {output_file_path_csv}")
        except Exception as e_csv:
              logging.error(f"Failed to write summary CSV fallback: {e_csv}")


# ====== Runner ======
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate plots and summary report from consolidated CSV.")
    parser.add_argument(
        "--experiments",
        nargs='+',
        required=True,
        help="List of experiment names to include in plots/report (e.g., baseline color all_extended)."
        )
    # Optional: Add argument to specify CSV path if needed
    parser.add_argument(
        "--csv_path",
        type=str,
        default=str(CONSOLIDATED_RESULTS_CSV), # Default to configured path
        help="Path to the consolidated results CSV file."
        )
    args = parser.parse_args()

    logging.info(f"Processing results for experiments: {args.experiments}")
    logging.info(f"Reading data from: {args.csv_path}")

    # --- Load Consolidated Data ---
    results_df = load_consolidated_results(Path(args.csv_path)) # Use path from args

    if results_df is None:
        logging.error("Could not load consolidated results. Exiting plotting script.")
        sys.exit(1) # Exit if data cannot be loaded

    # Ensure central plots directory exists
    CENTRAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Filter DataFrame for requested experiments immediately after loading
    results_df_filtered = results_df[results_df['experiment_name'].isin(args.experiments)].copy()
    if results_df_filtered.empty:
         logging.warning(f"No data found in {args.csv_path} for the specified experiments: {args.experiments}. Plots may be empty or fail.")
         # Decide whether to exit or proceed with empty plots
         # exit() # Option to exit early

    # --- Generate Plots (using filtered DataFrame) ---
    logging.info("\n--- Generating Comparison Plots (Latest Epoch Results) ---")
    # Pass the filtered DataFrame to plotting functions
    plot_loss(args.experiments) # Still reads separate CSVs for training loss
    plot_pseudo_accuracy(args.experiments) # Still reads separate CSVs for training accuracy
    plot_linear_probe(results_df_filtered, args.experiments)
    plot_knn(results_df_filtered, args.experiments)
    plot_ablation_study(results_df_filtered, args.experiments)
    logging.info("--- Comparison plot generation finished. ---")

    logging.info("\n--- Generating Evolution Plots ---")
    plot_linear_evolution(results_df_filtered, args.experiments)
    plot_knn_evolution(results_df_filtered, args.experiments, k_value=DEFAULT_KNN_K_FOR_EVOLUTION_PLOT)
    # Add calls for other k values if desired, e.g.:
    # plot_knn_evolution(results_df_filtered, args.experiments, k_value=5)
    # plot_knn_evolution(results_df_filtered, args.experiments, k_value=20)
    logging.info("--- Evolution plot generation finished. ---")

    logging.info("\n--- Generating Summary Report (Latest Epoch Results) ---")
    generate_summary_report(results_df_filtered, args.experiments, SUMMARY_REPORT_FILE) # Pass filtered df
    logging.info("--- Summary report generation finished. ---")

    print(f"\nResult processing complete.")
    print(f"Check individual run directories for logs and checkpoints.")
    print(f"Check '{CENTRAL_PLOTS_DIR}' for plots.")
    print(f"Check '{args.csv_path}' for raw consolidated data.")
    print(f"Check '{SUMMARY_REPORT_FILE.with_suffix('.md')}' (or .csv) for the summary report.")