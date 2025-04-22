# run.py
# Orchestrates SimCLR experiments: pretrain, evaluate, and plot.
# CORRECTED VERSION

import argparse
import os
import subprocess
import sys
from pathlib import Path
import logging

# --- Configuration ---
CONFIG_DIR = "configs"
RESULTS_DIR = "results"
MODEL_NAME = "simclr"
UTILS_DIR = "Utils"
ALL_STANDARD_EXPERIMENTS = ['baseline', 'color', 'blur', 'gray', 'all_standard']

logging.basicConfig(level=logging.INFO, format='%(asctime)s [Runner] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# --- Helper Functions ---
def check_config_exists(exp_name):
    config_path = Path(CONFIG_DIR) / f"{MODEL_NAME}_{exp_name}.yaml"
    if not config_path.is_file():
        logging.warning(f"Config file not found: {config_path} - Skipping experiment '{exp_name}'.")
        return None
    return config_path

# In run.py

# In run.py

def run_command(command_list): # Removed log_path and stream_stdout
    """Runs a command using subprocess, allowing its stdout and stderr
    to go directly to the terminal.
    """
    command_str = " ".join(map(str, command_list))
    logging.info(f"  Executing: {command_str}")
    logging.info(f"  (Output will appear directly below)")

    try:
        # stdout=None and stderr=None inherit terminal streams
        process = subprocess.run(
            command_list,
            stdout=None,
            stderr=None,
            text=True,
            check=True # Raise CalledProcessError if return code is non-zero
        )
        # Success is implicit if no exception is raised by check=True
        return True
    except subprocess.CalledProcessError as e:
        # Error messages from the command should have appeared directly on the terminal (stderr)
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
        logging.info("'all' keyword detected. Running all standard experiments.")
        experiments_to_run = ALL_STANDARD_EXPERIMENTS
    elif isinstance(experiments_to_run, str): # Handle single experiment case if passed without list
        experiments_to_run = [experiments_to_run]

    # Determine GPUs to use (More Robust Detection)
    gpus_per_node = args.gpus
    if gpus_per_node <= 0:
        try:
            # (GPU detection logic remains the same - omitted for brevity)
            nvidia_smi_path = subprocess.check_output("which nvidia-smi", shell=True).strip().decode()
            if nvidia_smi_path:
                result = subprocess.run(
                    [nvidia_smi_path, "--query-gpu=count", "--format=csv,noheader"],
                    capture_output=True, text=True, check=True
                )
                first_line = result.stdout.splitlines()[0].strip()
                num_cuda_devices = int(first_line)
                if num_cuda_devices > 0:
                    gpus_per_node = num_cuda_devices
                    logging.info(f"Auto-detected {gpus_per_node} GPUs using nvidia-smi.")
                else:
                    logging.warning("nvidia-smi reported 0 GPUs. Defaulting to 1.")
                    gpus_per_node = 1
            else:
                logging.warning("nvidia-smi not found in PATH. Defaulting to 1 GPU.")
                gpus_per_node = 1
        except Exception as e:
            logging.warning(f"GPU auto-detection failed ({e}). Defaulting to 1 GPU.")
            gpus_per_node = 1
    else:
        logging.info(f"Using specified {gpus_per_node} GPUs.")


    logging.info(f"Starting experiments with EPOCHS={args.epochs}, BATCH_SIZE={args.batch_size}, GPUS_PER_NODE={gpus_per_node}")
    logging.info(f"Running for experiments: {', '.join(experiments_to_run)}")

    successful_experiments = []
    failed_experiments = []

    for exp_name in experiments_to_run:
        logging.info(f"\n{'='*60}\n--- Starting Experiment: {MODEL_NAME} | Augmentation = {exp_name} ---\n{'='*60}")

        config_path = check_config_exists(exp_name)
        if config_path is None:
            failed_experiments.append(exp_name)
            continue

        # Define run directory and paths - this is correct
        run_dir = Path(RESULTS_DIR) / f"{MODEL_NAME}_{exp_name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = run_dir / "final_model.pth" # Assumes simclr.py saves here by default/config
        pretrain_log = run_dir / "pretrain_runner.log"
        linear_probe_log = run_dir / "linear_probe_runner.log"
        knn_eval_log = run_dir / "knn_eval_runner.log"

        # --- CORRECTED COMMAND LISTS (removed --run_dir) ---
        pretrain_command = [
            "torchrun", f"--nproc_per_node={gpus_per_node}",
            "simclr.py",
            "--config", str(config_path),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size)
            # Removed --run_dir argument
        ]
        linear_probe_command = [
            sys.executable, "linear_probe.py",
            "--checkpoint", str(checkpoint_path),
            "--config", str(config_path)
             # Removed --run_dir argument
        ]
        knn_eval_command = [
            sys.executable, "knn_eval.py",
            "--checkpoint", str(checkpoint_path),
            "--config", str(config_path)
             # Removed --run_dir argument
        ]
        # --- END CORRECTION ---


        # --- Stage 1: Pretraining ---
        logging.info(f"--> [1/3] Pretraining Stage for '{exp_name}'")
        pretrain_needed = True
        pretrain_success = False # Assume failure until proven otherwise
        if checkpoint_path.exists():
            logging.info(f"  Checkpoint found: {checkpoint_path}. Assuming pretraining completed earlier. Skipping.")
            pretrain_needed = False
            pretrain_success = True

        if pretrain_needed:
            logging.info(f"  Starting pretraining process...")
            pretrain_success = run_command(pretrain_command)
            if pretrain_success:
                 # Verify checkpoint creation AFTER successful command run
                 # It's crucial that simclr.py (or its config) knows to save the checkpoint
                 # to the expected `checkpoint_path` location.
                if checkpoint_path.exists():
                    logging.info(f"  ✅ Pretraining process completed successfully. Checkpoint verified.")
                else:
                    # This might indicate an issue in simclr.py not saving correctly
                    logging.error(f"  ❌ Pretraining command finished, but expected checkpoint file '{checkpoint_path}' was NOT created. Check simclr.py's save logic and log: {pretrain_log}")
                    pretrain_success = False
            else:
                # run_command already logged the failure details
                logging.error(f"  ❌ Pretraining process failed.")

        if not pretrain_success:
            failed_experiments.append(exp_name)
            logging.warning(f"--- Skipping evaluation stages for '{exp_name}' due to pretraining failure or missing checkpoint. ---")
            continue

        logging.info("-" * 40) # Separator

        # --- Stage 2: Linear Probe Evaluation ---
        logging.info(f"--> [2/3] Linear Probe Stage for '{exp_name}'")
        logging.info(f"  Starting linear probe process...")
        linear_probe_success = run_command(linear_probe_command)
        if linear_probe_success:
            logging.info(f"  ✅ Linear Probe process completed successfully.")
        else:
            logging.warning(f"  ⚠️ Linear Probe process failed. Check log: {linear_probe_log}")

        logging.info("-" * 40) # Separator

        # --- Stage 3: k-NN Evaluation ---
        logging.info(f"--> [3/3] k-NN Evaluation Stage for '{exp_name}'")
        logging.info(f"  Starting k-NN evaluation process...")
        knn_eval_success = run_command(knn_eval_command)
        if knn_eval_success:
            logging.info(f"  ✅ k-NN Evaluation process completed successfully.")
        else:
            logging.warning(f"  ⚠️ k-NN Evaluation process failed. Check log: {knn_eval_log}")

        logging.info(f"\n--- ✅ Finished All Stages for Experiment: {exp_name} ---")
        successful_experiments.append(exp_name)


    # --- Final Summary ---
    logging.info(f"\n{'='*60}\n--- All Specified Experiments Finished ---\n{'='*60}")
    if successful_experiments:
        logging.info(f"Successfully processed experiments: {', '.join(successful_experiments)}")
    if failed_experiments:
        logging.warning(f"Failed/Skipped experiments (Config missing or Pretrain failed): {', '.join(failed_experiments)}")
    logging.info(f"Check '{RESULTS_DIR}/' for detailed outputs and logs.")

    # --- Stage 4: Plotting ---
    if successful_experiments:
        logging.info(f"\n{'='*60}\n--- Starting Plotting Stage ---\n{'='*60}")
        plot_script_path = Path(UTILS_DIR) / "plotting.py"
        plot_log_path = Path(RESULTS_DIR) / "plotting_runner.log"
        plot_command = [sys.executable, str(plot_script_path), "--experiments"] + [str(exp) for exp in successful_experiments]

        logging.info(f"  Plotting results for experiments: {', '.join(successful_experiments)}...")
        plotting_success = run_command(plot_command)

        if plotting_success:
            logging.info(f"  ✅ Plotting process completed successfully.")
            logging.info(f"  Check individual run directories and potentially '{RESULTS_DIR}/plots/' for outputs.")
        else:
            logging.warning(f"  ⚠️ Plotting process failed. Check log: {plot_log_path}")
    elif failed_experiments and not successful_experiments:
         logging.warning("\n--- Plotting skipped as no experiments completed successfully. ---")
    else:
        logging.info("\n--- No experiments were run, skipping plotting. ---")


# --- Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SimCLR experiments: pretrain, evaluate, and plot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "experiments",
        nargs='*',
        help="Which experiment(s) to run (e.g., baseline color all_standard). Use 'all' to run all standard experiments. Corresponds to 'simclr_[experiment].yaml' in configs/. If none specified, runs 'baseline'."
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

    parsed_args = parser.parse_args()

    if not parsed_args.experiments:
         logging.info("No experiments specified, defaulting to 'baseline'.")
         parsed_args.experiments = ['baseline']
    elif 'all' in parsed_args.experiments:
        parsed_args.experiments = ['all']

    main(parsed_args)