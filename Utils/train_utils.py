# Utils/train_utils.py
import torch
import numpy as np
import random
import os
import logging
import sys
import torch.distributed as dist

def set_seed(seed):
    """Sets seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # Can reduce performance, but needed for full determinism

def setup_ddp(rank: int, world_size: int):
    """
    Initializes the distributed process group.
    Assumes environment variables MASTER_ADDR, MASTER_PORT are set.
    These are typically set by the process launcher (e.g., torchrun).
    """
    # If MASTER_ADDR and MASTER_PORT are not set, set defaults for local debugging
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '12355') # Choose an arbitrary free port

    # Initialize the process group
    dist.init_process_group(
        backend='nccl',        # Use NCCL backend for NVIDIA GPUs
        init_method='env://',  # Use environment variables
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank) # Set the current device for this process
    print(f"DDP Setup: Rank {rank}/{world_size} initialized on device {torch.cuda.current_device()}.")

def cleanup_ddp():
    """Destroys the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
        print("DDP Cleanup: Process group destroyed.")

def is_main_process() -> bool:
    """Checks if the current process is the main process (rank 0)."""
    if not dist.is_available() or not dist.is_initialized():
        return True # Treat as main process if not distributed
    return dist.get_rank() == 0

def setup_logging(run_dir: str, log_name: str = "training.log"):
    """
    Sets up logging to file and console.
    Logs INFO level and above to a file in `run_dir`.
    Logs INFO level and above to stdout ONLY for the main process (rank 0).
    """
    log_file = os.path.join(run_dir, log_name)
    os.makedirs(run_dir, exist_ok=True)

    # Determine the logging level and format
    log_level = logging.INFO
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Create handlers
    file_handler = logging.FileHandler(log_file, mode='a') # Append mode
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

    handlers = [file_handler]

    # Add console handler only for the main process
    if is_main_process():
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        handlers.append(console_handler)

    # Configure root logger
    logging.basicConfig(level=log_level, format=log_format, datefmt=date_format, handlers=handlers, force=True)

    if is_main_process():
        logging.info(f"Logging setup complete. Log file: {log_file}")
        logging.info("Console logging enabled for main process.")
    # else:
        # Optionally log a message indicating file-only logging for non-main processes
        # logging.info(f"Logging setup complete for rank {dist.get_rank()}. Log file: {log_file}. Console logging disabled.")

# Example usage (not typically run directly)
if __name__ == '__main__':
    # This part is mostly for demonstration; DDP setup requires launching multiple processes.
    print("Utils/train_utils.py - Contains helpers for DDP, logging, and seeding.")
    print("To test DDP, run a script using torchrun or mp.spawn.")

    # Example seed setting
    set_seed(42)
    print(f"Seed set to 42. Torch manual seed: {torch.initial_seed()}")

    # Example logging setup (simulating main process)
    if not dist.is_initialized(): # Simulate main process for logging demo
        print("\nSimulating logging setup for main process:")
        demo_run_dir = "./temp_log_dir"
        setup_logging(demo_run_dir, log_name="demo_log.txt")
        logging.info("This is an info message.")
        logging.warning("This is a warning message.")
        print(f"Check '{os.path.join(demo_run_dir, 'demo_log.txt')}'")
        # Clean up dummy dir
        # import shutil
        # if os.path.exists(demo_run_dir):
        #    shutil.rmtree(demo_run_dir)