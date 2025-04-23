# run.py
# Orchestrates SimCLR experiments: pretrain (with intermediate eval), final evaluate, and plot.
# UPDATED for intermediate evaluation trigger.

import argparse
import os
import subprocess
import sys
from pathlib import Path
import logging
import shutil # For potentially cleaning up intermediate checkpoints

# --- Configuration ---
CONFIG_DIR = "configs"
RESULTS_DIR = "results"
MODEL_NAME = "simclr"
UTILS_DIR = "Utils"
# Example: List experiments you want to run with the 'all' keyword
# Add your new experiment config names here if you want 'all' to include them
ALL_STANDARD_EXPERIMENTS = [
    'baseline', 'color', 'blur', 'gray', 'all_standard',
    'rotation', 'erasing', 'solarize', 'all_extended',
    'all_minus_color', 'all_minus_blur', 'all_minus_gray',
    'all_minus_rotation', 'all_minus_erasing', 'all_minus_solarize'
    ]


logging.basicConfig(level=logging.INFO, format='%(asctime)s [Runner] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# --- Helper Functions ---
def check_config_exists(exp_name):
    config_path = Path(CONFIG_DIR) / f"{MODEL_NAME}_{exp_name}.yaml"
    if not config_path.is_file():
        # Check for alternative naming (e.g., all_extended.yaml without simclr_ prefix)
        config_path_alt = Path(CONFIG_DIR) / f"{exp_name}.yaml"
        if config_path_alt.is_file():
            return config_path_alt # Return alternative path if it exists
        logging.warning(f"Config file not found: {config_path} or {config_path_alt} - Skipping experiment '{exp_name}'.")
        return None
    return config_path

def run_command(command_list):
    """Runs a command using subprocess, allowing its stdout and stderr
    to go directly to the terminal. Returns True on success, False on failure.
    """
    command_str = " ".join(map(str, command_list))
    logging.info(f"  Executing: {command_str}")
    logging.info(f"  (Output will appear directly below)")

    try:
        process = subprocess.run(
            command_list,
            stdout=None, # Inherit terminal stdout
            stderr=None, # Inherit terminal stderr
            text=True,
            check=True # Raise CalledProcessError if return code is non-zero
        )
        return True # Success is implicit if check=True doesn't raise
    except subprocess.CalledProcessError as e:
        logging.error(f"  Command failed with exit code {e.returncode}.")
        logging.error(f"  Failed command: {command_str}")
        return False
    except FileNotFoundError:
        logging.error(f"  Command not found (e.g., 'torchrun' or 'python'). Is it in your PATH? Failed command: {command_str}")
        return False
    except Exception as e:
        logging.error(f"  An unexpected error occurred while running command: {e}")
        logging.error(f"  Failed command: {command_str}")
        return False

# --- Main Orchestration Logic ---
def main(args):
    experiments_to_run = args.experiments
    if 'all' in experiments_to_run:
        logging.info(f"'all' keyword detected. Running standard experiments: {ALL_STANDARD_EXPERIMENTS}")
        experiments_to_run = ALL_STANDARD_EXPERIMENTS
    elif isinstance(experiments_to_run, str):
        experiments_to_run = [experiments_to_run]

    # Determine GPUs to use
    gpus_per_node = args.gpus
    if gpus_per_node <= 0:
        try:
            # Use nvidia-smi if available
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
                capture_output=True, text=True, check=True
            )
            num_cuda_devices = int(result.stdout.strip().splitlines()[0])
            if num_cuda_devices > 0:
                gpus_per_node = num_cuda_devices
                logging.info(f"Auto-detected {gpus_per_node} GPUs using nvidia-smi.")
            else:
                logging.warning("nvidia-smi reported 0 GPUs. Trying torch.cuda.device_count().")
                # Fallback using torch
                import torch
                if torch.cuda.is_available():
                     gpus_per_node = torch.cuda.device_count()
                     logging.info(f"Detected {gpus_per_node} GPUs using torch.")
                else:
                     logging.warning("torch.cuda reports no available GPUs. Defaulting to 1 GPU.")
                     gpus_per_node = 1
        except (FileNotFoundError, subprocess.CalledProcessError, ImportError, Exception) as e:
            logging.warning(f"GPU auto-detection failed ({e}). Defaulting to 1 GPU.")
            gpus_per_node = 1
    else:
        logging.info(f"Using specified {gpus_per_node} GPUs.")

    # --- NEW: Handle saving_epoch default ---
    saving_epoch = args.saving_epoch
    if saving_epoch <= 0:
        saving_epoch = args.epochs # If 0 or less, only evaluate at the very end
        logging.info(f"Evaluation interval (--saving_epoch) not specified or <= 0. Evaluating only at final epoch {saving_epoch}.")
    else:
        logging.info(f"Evaluation interval (--saving_epoch) set to every {saving_epoch} epochs.")


    logging.info(f"Starting experiments with EPOCHS={args.epochs}, BATCH_SIZE={args.batch_size}, GPUS_PER_NODE={gpus_per_node}, EVAL_INTERVAL={saving_epoch}")
    logging.info(f"Running for experiments: {', '.join(experiments_to_run)}")

    successful_experiments = []
    failed_experiments = []

    for exp_name in experiments_to_run:
        logging.info(f"\n{'='*60}\n--- Starting Experiment: {MODEL_NAME} | Augmentation = {exp_name} ---\n{'='*60}")

        config_path = check_config_exists(exp_name)
        if config_path is None:
            failed_experiments.append(exp_name)
            continue

        run_dir = Path(RESULTS_DIR) / f"{MODEL_NAME}_{exp_name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        # Final checkpoint path (simclr.py should ensure this exists at the end)
        final_checkpoint_path = run_dir / "final_model.pth"

        # --- Stage 1: Pretraining (with Internal Intermediate Evaluation) ---
        logging.info(f"--> [1/1] Pretraining Stage for '{exp_name}' (includes intermediate evaluation)")

        pretrain_command = [
            "torchrun", f"--nproc_per_node={gpus_per_node}",
            "simclr.py",
            "--config", str(config_path),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--saving_epoch", str(saving_epoch) # Pass the interval
            # Add seed if needed: "--seed", str(args.seed)
        ]

        # Check if final checkpoint already exists to potentially skip
        pretrain_needed = True
        if final_checkpoint_path.exists() and not args.force_rerun:
            logging.info(f"  Final checkpoint found: {final_checkpoint_path}. Assuming pretraining completed. Skipping pretrain.")
            pretrain_success = True
            pretrain_needed = False
        elif args.force_rerun:
            logging.info("  --force_rerun specified. Running pretraining even if checkpoint exists.")
            pretrain_needed = True

        if pretrain_needed:
            logging.info(f"  Starting pretraining process...")
            pretrain_success = run_command(pretrain_command)
            if pretrain_success:
                # Verify final checkpoint creation after successful run
                if final_checkpoint_path.exists():
                    logging.info(f"  ✅ Pretraining process completed successfully. Final checkpoint verified.")
                else:
                    logging.error(f"  ❌ Pretraining command finished, but expected final checkpoint '{final_checkpoint_path}' was NOT created. Check simclr.py logs.")
                    pretrain_success = False
            else:
                logging.error(f"  ❌ Pretraining process failed.")


        if pretrain_success:
            successful_experiments.append(exp_name)
            # Optionally clean up intermediate checkpoints
            if args.cleanup_intermediate and pretrain_needed: # Only cleanup if we actually ran pretraining
                 logging.info(f"  Cleaning up intermediate checkpoints (model_epoch_*.pth)...")
                 count = 0
                 for item in run_dir.glob("model_epoch_*.pth"):
                     try:
                         item.unlink()
                         count += 1
                     except OSError as e:
                         logging.warning(f"    Could not delete intermediate checkpoint {item}: {e}")
                 logging.info(f"    Deleted {count} intermediate checkpoints.")

        else:
            failed_experiments.append(exp_name)
            logging.warning(f"--- Experiment '{exp_name}' failed during pretraining stage. ---")
            continue

        logging.info(f"\n--- ✅ Finished Experiment: {exp_name} ---")


    # --- Final Summary ---
    logging.info(f"\n{'='*60}\n--- All Specified Experiments Finished ---\n{'='*60}")
    if successful_experiments:
        logging.info(f"Successfully completed or skipped experiments: {', '.join(successful_experiments)}")
    if failed_experiments:
        logging.warning(f"Failed experiments: {', '.join(failed_experiments)}")
    logging.info(f"Check '{RESULTS_DIR}/' for detailed outputs, logs, and intermediate results.")

    # --- Stage 2: Plotting (Run once at the end, using all generated results) ---
    if successful_experiments:
        logging.info(f"\n{'='*60}\n--- Starting Plotting Stage ---\n{'='*60}")
        plot_script_path = Path(UTILS_DIR) / "plotting.py"
        # Pass the list of experiments that completed successfully
        plot_command = [sys.executable, str(plot_script_path), "--experiments"] + successful_experiments

        logging.info(f"  Plotting results for experiments: {', '.join(successful_experiments)}...")
        plotting_success = run_command(plot_command)

        if plotting_success:
            logging.info(f"  ✅ Plotting process completed successfully.")
            logging.info(f"  Check individual run directories and '{RESULTS_DIR}/plots/' for outputs.")
            logging.info(f"  Check '{RESULTS_DIR}/summary_report.md' for the summary.")
        else:
            logging.warning(f"  ⚠️ Plotting process failed.")
    elif failed_experiments and not successful_experiments:
         logging.warning("\n--- Plotting skipped as no experiments completed successfully. ---")
    else:
        logging.info("\n--- No experiments were run, skipping plotting. ---")


# --- Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SimCLR experiments: pretrain (with intermediate eval), evaluate, and plot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "experiments",
        nargs='*',
        help="Which experiment(s) to run (e.g., baseline color all_extended). Use 'all' to run standard experiments. Corresponds to '[model_name]_[experiment].yaml' or '[experiment].yaml' in configs/. If none specified, runs 'baseline'."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of pretraining epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Global batch size for pretraining (total across all GPUs)."
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=0,
        help="Number of GPUs per node (torchrun --nproc_per_node). 0=auto-detect."
    )
    # --- NEW ARGUMENT ---
    parser.add_argument(
        "--saving_epoch", # Changed name to --saving_epoch as requested
        type=int,
        default=0, # Default 0 means only evaluate at the end
        help="Interval for saving intermediate checkpoints and running evaluations. 0 means evaluate only at the final epoch."
    )
    parser.add_argument(
        "--cleanup_intermediate",
        action='store_true',
        help="If set, delete intermediate checkpoints (model_epoch_*.pth) after training finishes successfully."
    )
    parser.add_argument(
        "--force_rerun",
        action='store_true',
        help="If set, force pretraining even if a final checkpoint exists."
    )
    # Add --seed if you want to control it from run.py
    # parser.add_argument('--seed', type=int, default=None, help='Random seed for pretraining')


    parsed_args = parser.parse_args()

    # Default experiment if none provided
    if not parsed_args.experiments:
         logging.info("No experiments specified, defaulting to 'baseline'.")
         parsed_args.experiments = ['baseline']
    elif 'all' in parsed_args.experiments:
        # Handle 'all' case - already done inside main based on the list
        pass

    main(parsed_args)