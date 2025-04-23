# simclr.py (Updated)
# Main script for SimCLR pre-training with integrated periodic evaluation
# and appending results to a consolidated CSV file.

# ------------------------- Imports -------------------------
import argparse
import os
import yaml
import logging
import sys
import random
import time
import pandas as pd
from pathlib import Path # Use pathlib
import torch
# (Other imports: nn, optim, F, dist, mp, DDP, DataLoader, DistributedSampler, GradScaler, autocast, STL10, torchvision_models, tqdm)
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torchvision.datasets import STL10
import torchvision.models as torchvision_models
from tqdm import tqdm


# --- Import from Utils ---
from Utils.augmentations import ContrastiveTransform
from Utils.train_utils import set_seed, setup_ddp, cleanup_ddp, is_main_process, setup_logging

# --- Import Evaluation Functions ---
try:
    from linear_probe import evaluate_linear_probe # Assumes returns list of dicts
    from knn_eval import evaluate_knn # Assumes returns list of dicts
    EVAL_FUNCTIONS_IMPORTED = True
    # Log success after logger is set up in main_worker
except ImportError as e:
    EVAL_FUNCTIONS_IMPORTED = False
    # Log warning after logger is set up in main_worker


# ---------------------- Helper Functions ---------------------
def deep_update(source, overrides):
    """Recursively update a dict."""
    for key, value in overrides.items():
        if isinstance(value, dict) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# --------------------- Model Definition --------------------
class ResNetSimCLR(nn.Module):
    """Encoder network using ResNet with an MLP projection head."""
    def __init__(self, base_model_name='resnet18', out_dim=128):
        super(ResNetSimCLR, self).__init__()
        resnet_constructor = getattr(torchvision_models, base_model_name)
        # Load backbone without pretrained weights, setting num_classes arbitrarily (will be replaced)
        self.backbone = resnet_constructor(weights=None, num_classes=out_dim)
        dim_mlp = self.backbone.fc.in_features
        # Replace the final layer with the SimCLR projection head
        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, out_dim, bias=True) # SimCLR uses bias in the final layer
        )
        # Use standard logging after logger is initialized
        # logging.debug(f"Built ResNetSimCLR: backbone={base_model_name}, proj_dim={out_dim}, feature_dim={dim_mlp}")

    def forward(self, x):
        return self.backbone(x)

# --------------------- Argument Parsing --------------------
def parse_args():
    parser = argparse.ArgumentParser(description='SimCLR Training Script')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--seed', type=int, default=None, help='Override random seed')
    parser.add_argument('--output_dir', type=str, default=None, help='Override base output directory')
    parser.add_argument('--epochs', type=int, default=None, help='Override number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Override global batch size')
    # --- NEW ARGUMENT ---
    parser.add_argument('--saving_epoch', type=int, default=0, help='Interval for saving/evaluating checkpoints. 0=only final.')
    return parser.parse_args()

# -------------------- Configuration Loading ------------------
def load_config(config_path, cli_args):
    # Load base config
    base_config_path = os.path.join(os.path.dirname(config_path), 'base_simclr.yaml')
    config = {}
    if os.path.exists(base_config_path):
        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    # else: log warning after logger setup

    # Load specific config and merge/override base
    with open(config_path, 'r') as f:
        specific_config = yaml.safe_load(f) or {}
    config = deep_update(config, specific_config)

    # Override with CLI arguments
    if cli_args.seed is not None: config['seed'] = cli_args.seed
    if cli_args.output_dir is not None: config['output']['results_dir'] = cli_args.output_dir
    if cli_args.epochs is not None: config['pretrain']['epochs'] = cli_args.epochs
    if cli_args.batch_size is not None: config['pretrain']['batch_size'] = cli_args.batch_size
    # Add saving_epoch to config
    config['pretrain']['saving_epoch'] = cli_args.saving_epoch

    # Set defaults and derived paths
    config.setdefault('seed', random.randint(1, 10000))
    config['model_name'] = config.get('model', {}).get('name', 'simclr')

    # Determine augmentation name from config filename
    config_filename = os.path.basename(config_path)
    aug_name = os.path.splitext(config_filename)[0]
    if aug_name.startswith(f"{config['model_name']}_"):
         aug_name = aug_name.split(f"{config['model_name']}_", 1)[1]
    config['augmentation_name'] = aug_name

    # Define run directory
    # Ensure results_dir exists in config before constructing run_dir
    config.setdefault('output', {}).setdefault('results_dir', './results')
    config['output']['run_dir'] = os.path.join(
        config['output']['results_dir'],
        f"{config['model_name']}_{config['augmentation_name']}"
    )

    return config


# ---------------------- Data Loading -----------------------
def get_dataloader(config, rank, world_size):
    try:
        transform = ContrastiveTransform(config['augmentations'])
        dataset = STL10(
            root=config['data']['data_dir'],
            split='unlabeled',
            download=False, # Assume downloaded
            transform=transform
        )
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config['seed']
        )
        # Ensure global batch size is divisible by world size
        global_batch_size = config['pretrain']['batch_size']
        if global_batch_size % world_size != 0:
            # Log error after logger setup
            raise ValueError(f"Global batch size {global_batch_size} must be divisible by world size {world_size}")
        per_gpu_batch_size = global_batch_size // world_size

        loader = DataLoader(
            dataset,
            batch_size=per_gpu_batch_size,
            sampler=sampler,
            num_workers=config['data'].get('num_workers', 4), # Provide default
            pin_memory=True,
            drop_last=True
        )
        # Log info after logger setup
        # logging.info(f"Rank {rank}: DataLoader created...")
        return loader, sampler
    except FileNotFoundError:
        # Log error after logger setup
        print(f"ERROR: STL10 dataset not found at {config['data']['data_dir']}. Please run download script or check path.")
        raise
    except KeyError as e:
        # Log error after logger setup
        print(f"ERROR: Missing key in data config: {e}")
        raise
    except Exception as e:
        # Log error after logger setup
        print(f"ERROR: Error creating dataloader: {e}")
        raise


# ---------------------- Model Building -------------------
def build_model(config):
    model_cfg = config.get('model', {})
    base_model = model_cfg.get('backbone', 'resnet18')
    out_dim = model_cfg.get('projection_dim', 128)
    try:
        model = ResNetSimCLR(base_model_name=base_model, out_dim=out_dim)
        return model
    except AttributeError:
        # Log error after logger setup
        print(f"ERROR: Invalid backbone model name: {base_model}. Check torchvision models.")
        raise
    except Exception as e:
        # Log error after logger setup
        print(f"ERROR: Error building model: {e}")
        raise


# ------------------- Optimizer & Scheduler -----------------
def build_optimizer_scheduler(model_params, config, steps_per_epoch):
    optim_cfg = config['pretrain']
    optimizer_name = optim_cfg.get('optimizer', 'adam').lower()
    lr = float(optim_cfg.get('learning_rate', 0.0003)) # Ensure float
    wd = float(optim_cfg.get('weight_decay', 1e-4)) # Ensure float

    if optimizer_name == 'adam':
        optimizer = optim.Adam(model_params, lr=lr, weight_decay=wd)
        # Log info after logger setup
    elif optimizer_name == 'sgd':
        momentum = float(optim_cfg.get('momentum', 0.9))
        optimizer = optim.SGD(model_params, lr=lr, momentum=momentum, weight_decay=wd)
        # Log info after logger setup
    # Add AdamW, LARS etc. if needed
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    scheduler_name = optim_cfg.get('lr_schedule', 'cosine').lower()
    epochs = optim_cfg.get('epochs', 100)
    warmup_epochs = optim_cfg.get('warmup_epochs', 10) # Add warmup definition

    if scheduler_name == 'cosine':
        t_max_epochs = epochs - warmup_epochs
        if t_max_epochs <= 0:
             # Log warning after logger setup
             t_max_epochs = 1 # Avoid T_max=0 error
        eta_min = float(optim_cfg.get('eta_min', 0.0))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max_epochs, eta_min=eta_min)
        # Log info after logger setup
    # Add other schedulers if needed
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    return optimizer, scheduler, warmup_epochs # Return warmup_epochs


# ---------------------- Loss Function ----------------------
def info_nce_loss(features, batch_size, n_views, temperature, device):
    """Calculates InfoNCE loss for contrastive learning."""
    # Features shape: [n_views * batch_size, proj_dim]
    # Create labels that indicate positive pairs (views of the same image)
    labels = torch.cat([torch.arange(batch_size) for _ in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device, non_blocking=True)

    # Normalize features
    features = F.normalize(features, dim=1)

    # Compute similarity matrix (cosine similarity)
    similarity_matrix = torch.matmul(features, features.T)
    # Shape: [n_views * batch_size, n_views * batch_size]

    # Discard diagonal elements (similarity of an image to itself)
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device, non_blocking=True)
    labels_no_diag = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix_no_diag = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # Shape: [n_views * batch_size, n_views * batch_size - 1]

    # Select positive similarities (views of the same image, excluding self-similarity)
    positives = similarity_matrix_no_diag[labels_no_diag.bool()].view(labels.shape[0], -1)
    # Shape: [n_views * batch_size, n_views - 1]

    # Select negative similarities (views of different images)
    negatives = similarity_matrix_no_diag[~labels_no_diag.bool()].view(similarity_matrix.shape[0], -1)
    # Shape: [n_views * batch_size, (batch_size - 1) * n_views]

    # Concatenate positive and negative similarities for logits
    logits = torch.cat([positives, negatives], dim=1)

    # Labels for cross-entropy are all zeros (positive similarity is the target)
    criterion_labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device, non_blocking=True)

    # Scale logits by temperature
    logits = logits / temperature

    # Calculate cross-entropy loss
    criterion = nn.CrossEntropyLoss().to(device)
    loss = criterion(logits, criterion_labels)

    # Calculate accuracy (optional, represents classifying positive sample correctly)
    with torch.no_grad():
        top1, _ = accuracy(logits, criterion_labels, topk=(1, 5)) # Using helper function

    return loss, logits, criterion_labels, top1.item() # Return loss and top-1 accuracy


# ---------------------- Training Loop ----------------------
def train_one_epoch(model, loader, sampler, optimizer, scheduler, criterion_fn, config, epoch, rank, world_size, scaler, warmup_epochs):
    model.train()
    sampler.set_epoch(epoch) # Ensure proper shuffling with DDP

    pretrain_cfg = config['pretrain']
    batch_size_global = pretrain_cfg['batch_size']
    n_views = pretrain_cfg.get('n_views', 2)
    temperature = float(pretrain_cfg.get('temperature', 0.07)) # Ensure float
    use_amp = pretrain_cfg.get('fp16_precision', False)
    batch_size_per_gpu = batch_size_global // world_size

    epoch_loss = 0.0
    epoch_contrastive_acc = 0.0 # Renamed from top1_acc for clarity
    step_losses = [] # For detailed logging if needed

    pbar = None
    if is_main_process():
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{pretrain_cfg['epochs']}", leave=False, file=sys.stdout, dynamic_ncols=True)
    else:
        pbar = loader # No progress bar for non-main processes

    start_time = time.time()
    for step, batch_data in enumerate(pbar):
        # Handle cases where dataset yields (images, _) or just images
        if isinstance(batch_data, (list, tuple)):
            images = batch_data[0] # Assuming first element is the pair of views
        else:
            # This case is less likely with ContrastiveTransform returning tuples
            images = batch_data
            logging.warning(f"Rank {rank}, Epoch {epoch+1}, Step {step}: Unexpected batch data format. Expected list/tuple of views.")
            # Attempt to proceed might fail later in loss calculation
            # Need logic here depending on how data is formatted if not tuple

        # ContrastiveTransform returns (view1, view2), DataLoader collates B x (view1, view2)
        # into ([B x view1], [B x view2])
        if isinstance(images, (list, tuple)) and len(images) == n_views:
             combined_images = torch.cat(images, dim=0).to(rank, non_blocking=True)
        else:
             # This might happen if transform is not ContrastiveTransform or loader doesn't handle it right
             logging.error(f"Rank {rank}, Epoch {epoch+1}, Step {step}: Incorrect data format from DataLoader. Expected {n_views} views.")
             # Handle error appropriately, e.g., skip batch or raise error
             continue # Skip batch

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            features = model(combined_images) # Get projections [N*V, D]
            # Ensure batch_size_per_gpu is correct for loss calculation
            loss, logits, labels, contrastive_acc = criterion_fn(
                features, batch_size_per_gpu, n_views, temperature, rank
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Accumulate metrics
        current_loss = loss.item()
        epoch_loss += current_loss
        epoch_contrastive_acc += contrastive_acc
        step_losses.append({'epoch': epoch, 'step': step, 'loss': current_loss, 'contrastive_acc': contrastive_acc})

        # Update progress bar (only on main process)
        if is_main_process():
            # Get LR from optimizer during warmup, scheduler after
            if epoch < warmup_epochs:
                 lr = optimizer.param_groups[0]['lr']
            else:
                 # Avoid error if scheduler hasn't stepped yet in first non-warmup epoch
                 try:
                     lr = scheduler.get_last_lr()[0]
                 except: # Handle potential errors if get_last_lr fails initially
                      lr = optimizer.param_groups[0]['lr'] # Fallback
            pbar.set_postfix({"Loss": f"{current_loss:.4f}", "Acc(InfoNCE)": f"{contrastive_acc:.2f}%", "LR": f"{lr:.6f}"})

    # --- Scheduler Step (after epoch) ---
    # Step the scheduler only *after* the warmup phase
    if epoch >= warmup_epochs:
        scheduler.step()
        if is_main_process():
             new_lr = scheduler.get_last_lr()[0]
             logging.debug(f"Epoch {epoch+1}: Stepped scheduler. New LR: {new_lr:.6f}")
    elif is_main_process():
         logging.debug(f"Epoch {epoch+1}: Warmup phase. Optimizer LR: {optimizer.param_groups[0]['lr']:.6f}")

    end_time = time.time()
    epoch_duration = end_time - start_time
    steps_per_epoch = len(loader)
    avg_epoch_loss = epoch_loss / steps_per_epoch if steps_per_epoch > 0 else 0
    avg_epoch_contrastive_acc = epoch_contrastive_acc / steps_per_epoch if steps_per_epoch > 0 else 0

    # Log epoch summary on main process
    if is_main_process():
        logging.info(f"Epoch {epoch+1} finished. Avg Loss: {avg_epoch_loss:.4f}, "
                     f"Avg Contrastive Acc: {avg_epoch_contrastive_acc:.2f}%, Duration: {epoch_duration:.2f}s")

    # Return average metrics for the epoch
    return avg_epoch_loss, avg_epoch_contrastive_acc, step_losses


# ------------------- Periodic Evaluation & CSV Appending -------------
def run_periodic_evaluation_and_log(checkpoint_path, config, epoch, device, consolidated_csv_path):
    """
    Runs evaluations and appends results to the consolidated CSV.
    Only runs on the main process.
    """
    if not is_main_process(): # Should already be checked before calling, but double-check
        return

    if not EVAL_FUNCTIONS_IMPORTED:
        logging.warning(f"Skipping evaluation for epoch {epoch} - functions not imported.")
        return

    logging.info(f"--- Starting Evaluation & Logging for Epoch {epoch} ---")
    eval_start_time = time.time()
    all_results_for_epoch = [] # Collect results from all eval types

    # --- Linear Probe ---
    try:
        logging.info(f"Running Linear Probe for epoch {epoch}...")
        linear_results = evaluate_linear_probe(
            checkpoint_path=str(checkpoint_path), # Ensure path is string
            config=config, # Pass the full config dictionary
            epoch=epoch,
            device=device
        )
        if linear_results: # Check if list is not empty
             all_results_for_epoch.extend(linear_results)
             logging.info(f"Linear Probe completed for epoch {epoch}.")
        else:
             logging.warning(f"Linear Probe returned no results for epoch {epoch}.")
    except Exception as e:
        logging.error(f"Linear Probe evaluation failed for epoch {epoch}: {e}", exc_info=True)

    # --- k-NN Evaluation ---
    try:
        logging.info(f"Running k-NN Evaluation for epoch {epoch}...")
        knn_results = evaluate_knn(
            checkpoint_path=str(checkpoint_path), # Ensure path is string
            config=config, # Pass the full config dictionary
            epoch=epoch,
            device=device
        )
        if knn_results: # Check if list is not empty
            all_results_for_epoch.extend(knn_results)
            logging.info(f"k-NN evaluation completed for epoch {epoch}.")
        else:
            logging.warning(f"k-NN evaluation returned no results for epoch {epoch}.")
    except Exception as e:
        logging.error(f"k-NN evaluation failed for epoch {epoch}: {e}", exc_info=True)

    # --- Append results to Consolidated CSV ---
    if all_results_for_epoch:
        try:
            df_new_results = pd.DataFrame(all_results_for_epoch)
            # Add experiment name column
            exp_name = config.get('augmentation_name', 'unknown_exp') # Get exp name from config
            df_new_results['experiment_name'] = exp_name

            # Define column order
            cols = ['experiment_name', 'epoch', 'metric_type', 'metric_name', 'k_value', 'value']
            # Ensure all columns exist, adding missing ones as None/NaN if necessary
            for col in cols:
                 if col not in df_new_results.columns:
                      df_new_results[col] = None
            df_new_results = df_new_results[cols] # Reorder columns

            # Check if file exists to determine if header is needed
            file_exists = consolidated_csv_path.exists()

            # Append to CSV
            df_new_results.to_csv(
                consolidated_csv_path,
                mode='a',          # Append mode
                header=not file_exists, # Write header only if file doesn't exist
                index=False        # Don't write pandas index
            )
            logging.info(f"Appended {len(df_new_results)} result(s) for epoch {epoch} to {consolidated_csv_path}")

        except Exception as e:
            logging.error(f"Failed to append results to {consolidated_csv_path} for epoch {epoch}: {e}", exc_info=True)

    eval_duration = time.time() - eval_start_time
    logging.info(f"--- Evaluation & Logging for Epoch {epoch} finished. Duration: {eval_duration:.2f}s ---")


# ---------------------- Main Worker Function (Updated Eval Call) --------
def main_worker(rank, world_size, config):
    """Main DDP training worker process."""
    setup_ddp(rank, world_size) # Initialize DDP
    run_dir = Path(config['output']['run_dir'])
    log_name = config['output'].get('log_name', 'training.log') # Default log name
    # Setup logging *after* DDP setup to ensure rank info is available
    setup_logging(str(run_dir), log_name) # setup_logging expects string path
    main_proc = is_main_process() # Check if this is the main process (rank 0)
    device = torch.device(f"cuda:{rank}") # Assign device based on rank

    # Log import status now that logger is configured
    if main_proc:
        if not EVAL_FUNCTIONS_IMPORTED:
             logging.warning("Evaluation functions (linear_probe, knn_eval) import failed. Intermediate evaluations will be skipped.")
        else:
             logging.info("Evaluation functions imported successfully.")
        # Log config loading warnings if any
        base_config_path = Path(config['output']['run_dir']).parent.parent / 'configs' / 'base_simclr.yaml' # Infer base config path relative to run_dir might be fragile
        # Better: Pass base config path or check explicitly
        # if not Path(os.path.join(os.path.dirname(config_path_arg), 'base_simclr.yaml')).exists(): # Need config_path_arg here
        #      logging.warning("Base config file may be missing.")


    # --- !! Define path for consolidated results file !! ---
    # Place it one level up from individual run dirs, in the base results dir
    base_results_dir = Path(config['output'].get('results_dir', './results'))
    # Ensure base results dir exists (especially for rank 0)
    if main_proc:
         os.makedirs(base_results_dir, exist_ok=True)
    consolidated_csv_path = base_results_dir / "consolidated_results.csv"

    if main_proc:
        logging.info("=====================================================")
        logging.info(f"Starting SimCLR Training: {config.get('augmentation_name', 'N/A')}")
        logging.info(f"World Size: {world_size}, Rank: {rank}, Device: {device}")
        logging.info(f"Run Directory: {run_dir}")
        logging.info(f"Consolidated Results CSV: {consolidated_csv_path}") # Log path
        # Save the effective configuration used for this run
        os.makedirs(run_dir, exist_ok=True)
        final_config_path = run_dir / 'effective_config.yaml'
        try:
            with open(final_config_path, 'w') as f:
                 yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            logging.info(f"Effective configuration saved to {final_config_path}")
        except Exception as e:
            logging.error(f"Could not save effective config: {e}")
        logging.info("=====================================================")

    # Set seed for reproducibility
    set_seed(config['seed'] + rank)
    if main_proc:
        logging.info(f"Random seed (Rank 0) set to {config['seed']}")

    # --- Create DataLoader ---
    try:
        train_loader, train_sampler = get_dataloader(config, rank, world_size)
        steps_per_epoch = len(train_loader)
        if main_proc:
            logging.info(f"DataLoader created. Steps per epoch: {steps_per_epoch}")
    except Exception as e:
        logging.error(f"Failed to create DataLoader: {e}", exc_info=True)
        cleanup_ddp(); sys.exit(1) # Exit if dataloader fails

    # --- Build Model, Optimizer, Scheduler ---
    try:
        model = build_model(config).to(device)
        # Log model structure details if needed (e.g., number of parameters)
        if main_proc:
             logging.info(f"Model '{config['model']['backbone']}' built successfully.")
             # total_params = sum(p.numel() for p in model.parameters())
             # logging.info(f"Total model parameters: {total_params:,}")

    except Exception as e:
        logging.error(f"Failed to build model: {e}", exc_info=True)
        cleanup_ddp(); sys.exit(1)

    # Wrap model AFTER moving to device
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    try:
        optimizer, scheduler, warmup_epochs = build_optimizer_scheduler(model.parameters(), config, steps_per_epoch)
        if main_proc:
             logging.info(f"Optimizer '{config['pretrain']['optimizer']}' and Scheduler '{config['pretrain']['lr_schedule']}' built.")
    except Exception as e:
        logging.error(f"Failed to build optimizer/scheduler: {e}", exc_info=True)
        cleanup_ddp(); sys.exit(1)

    # --- Loss Criterion and AMP Scaler ---
    criterion_fn = info_nce_loss
    use_amp = config['pretrain'].get('fp16_precision', False)
    scaler = GradScaler(enabled=use_amp)
    if main_proc and use_amp:
        logging.info("Automatic Mixed Precision (AMP) enabled.")

    # --- Training State & History ---
    training_history = [] # Store per-epoch summary
    all_step_losses = [] # Store per-step details if needed later
    start_epoch = 0 # Resume logic placeholder
    total_epochs = config['pretrain']['epochs']
    saving_epoch_interval = config['pretrain']['saving_epoch']

    # --- Early Stopping Setup ---
    early_stopping_enabled = config['pretrain'].get('early_stopping', {}).get('enabled', False)
    best_loss = float('inf'); epochs_no_improve = 0
    if main_proc and early_stopping_enabled:
        patience = config['pretrain']['early_stopping'].get('patience', 20)
        min_delta = float(config['pretrain']['early_stopping'].get('min_delta', 0.001))
        logging.info(f"Early stopping enabled: Patience={patience}, Min Delta={min_delta}")

    # --- Main Training Loop ---
    if main_proc:
        logging.info(f"Starting training loop for {total_epochs} epochs...")
    global_start_time = time.time()

    for epoch in range(start_epoch, total_epochs):
        epoch_start_time = time.time()

        # Run one training epoch
        avg_loss, avg_contrastive_acc, step_losses = train_one_epoch(
            model, train_loader, train_sampler, optimizer, scheduler, criterion_fn,
            config, epoch, rank, world_size, scaler, warmup_epochs
        )

        # --- Logging, Checkpointing, Evaluation (Main Process Only) ---
        stop_signal_val = 0
        if main_proc:
            epoch_duration = time.time() - epoch_start_time
            # Determine current LR for logging
            if epoch < warmup_epochs: current_lr = optimizer.param_groups[0]['lr']
            else:
                 try: current_lr = scheduler.get_last_lr()[0]
                 except: current_lr = optimizer.param_groups[0]['lr'] # Fallback

            # Record epoch history
            training_history.append({'epoch': epoch + 1, 'avg_loss': avg_loss, 'avg_contrastive_acc': avg_contrastive_acc, 'duration_sec': epoch_duration, 'learning_rate': current_lr})
            all_step_losses.extend(step_losses) # Accumulate step details

            # --- Checkpointing ---
            latest_checkpoint_path = run_dir / config['output']['checkpoint_name']
            save_data = {'epoch': epoch + 1, 'state_dict': model.module.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'config': config, 'scaler': scaler.state_dict() if use_amp else None}
            torch.save(save_data, latest_checkpoint_path)
            logging.debug(f"Saved latest checkpoint to {latest_checkpoint_path} after epoch {epoch+1}")

            # --- Determine if evaluation should happen ---
            should_evaluate = (saving_epoch_interval > 0 and (epoch + 1) % saving_epoch_interval == 0) or \
                              (epoch + 1 == total_epochs)
            epoch_checkpoint_path = None
            if should_evaluate:
                 # Save epoch-specific checkpoint (only model weights needed for eval)
                 epoch_checkpoint_path = run_dir / f"model_epoch_{epoch+1}.pth"
                 torch.save(model.module.state_dict(), epoch_checkpoint_path)
                 logging.info(f"Saved epoch-specific checkpoint for evaluation: {epoch_checkpoint_path}")

                 # --- !! Call Periodic Evaluation & Logging !! ---
                 # Pass the path to the epoch-specific weights
                 run_periodic_evaluation_and_log(
                     epoch_checkpoint_path, config, epoch + 1, device, consolidated_csv_path
                 )

            # --- Early Stopping Check ---
            if early_stopping_enabled:
                 current_loss_for_es = avg_loss
                 if current_loss_for_es < best_loss - min_delta:
                     best_loss = current_loss_for_es; epochs_no_improve = 0
                     logging.info(f"ES: New best loss: {best_loss:.4f}. Saving best model.")
                     best_model_path = run_dir / 'best_model.pth'
                     torch.save(model.module.state_dict(), best_model_path) # Save only weights
                 else:
                     epochs_no_improve += 1; logging.info(f"ES: No improvement for {epochs_no_improve}/{patience} epochs.")
                 if epochs_no_improve >= patience:
                     logging.warning(f"Early stopping triggered after epoch {epoch + 1}.")
                     stop_signal_val = 1
        # --- End Main Process Block ---

        # --- Synchronize and Check Stop Signal ---
        stop_signal = torch.tensor(stop_signal_val, dtype=torch.int).to(device)
        dist.broadcast(stop_signal, src=0)
        dist.barrier()
        if stop_signal.item() == 1:
            if main_proc: logging.info("Stop signal received, terminating training.")
            break

    # --- End of Training Loop ---

    # --- Final Actions (Main Process Only) ---
    if main_proc:
        total_duration = time.time() - global_start_time
        # Use final epoch value reached (could be less than total_epochs if early stopping)
        final_epoch = epoch + 1 if 'epoch' in locals() else start_epoch
        logging.info("=====================================================")
        logging.info(f"Training finished after {final_epoch} epochs.")
        logging.info(f"Total training time: {total_duration / 3600:.2f} hours")

        # Save the final model state dict (redundant if last epoch was saved, but ensures it exists)
        final_model_path = run_dir / 'final_model.pth'
        torch.save(model.module.state_dict(), final_model_path)
        logging.info(f"Final model state dict saved to {final_model_path}")

        # Save training history (loss, acc per epoch)
        history_df = pd.DataFrame(training_history)
        # Ensure output filenames are correctly formatted from config
        loss_csv_fname = config['output'].get('train_loss_csv', 'training_loss_{model}_{aug}.csv')
        history_csv_path = run_dir / loss_csv_fname.format(model=config['model_name'], aug=config['augmentation_name'])
        try:
            history_df.to_csv(history_csv_path, index=False)
            logging.info(f"Training history saved to {history_csv_path}")
        except Exception as e:
            logging.error(f"Could not save training history CSV: {e}")

        # Log location of consolidated results
        logging.info(f"Consolidated evaluation results saved to: {consolidated_csv_path}")
        logging.info("=====================================================")

    # --- Cleanup DDP ---
    cleanup_ddp()


# ---------------- Execution Guard & DDP Launch ---------------
if __name__ == "__main__":
    args = parse_args()
    try:
        config = load_config(args.config, args)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # Check GPU availability and batch size divisibility
    world_size = torch.cuda.device_count()
    if world_size < 1:
        print("Error: No CUDA GPUs found. Exiting.")
        sys.exit(1)
    # Check batch size divisibility using the loaded config value
    global_bs = config.get('pretrain', {}).get('batch_size')
    if global_bs is None:
         print("Error: 'pretrain.batch_size' not found in configuration.")
         sys.exit(1)
    if global_bs % world_size != 0:
         print(f"Error: Global batch size {global_bs} is not divisible by the number of GPUs {world_size}.")
         sys.exit(1)

    # DDP Launch Check (using environment variables set by torchrun)
    try:
        rank = int(os.environ["RANK"])
        world_size_env = int(os.environ["WORLD_SIZE"])
        # Optional: Check consistency, though torchrun should handle it
        if world_size_env != world_size:
             print(f"Warning: WORLD_SIZE env ({world_size_env}) != torch.cuda.device_count() ({world_size}). Using env value.")
             world_size = world_size_env

        print(f"Launching DDP process. Rank: {rank}, World Size: {world_size}")
        main_worker(rank=rank, world_size=world_size, config=config)

    except KeyError:
        print("Error: RANK or WORLD_SIZE environment variables not set.")
        print("Please launch this script using 'torchrun --nproc_per_node=<num_gpus> simclr.py ...'")
        sys.exit(1)
    except ValueError as e: # Catch specific errors like batch size issue
         print(f"Configuration Error: {e}")
         # Attempt cleanup even on error (might fail if DDP didn't init fully)
         try: cleanup_ddp()
         except: pass
         sys.exit(1)
    except Exception as e:
         # General error catching
         print(f"An error occurred during training: {e}")
         # Log the full traceback if logging is set up (might not be if error is early)
         if logging.getLogger().hasHandlers():
              logging.error("Training script terminated with an error.", exc_info=True)
         else:
              import traceback
              traceback.print_exc() # Print traceback to stderr if logger not ready
         try: cleanup_ddp()
         except: pass
         sys.exit(1) # Indicate failure

    print("SimCLR script finished successfully.")