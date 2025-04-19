# simclr.py
# Main script for SimCLR pre-training

# ------------------------- Imports -------------------------
import argparse
import os
import yaml
import logging
import sys
import random
import time
import pandas as pd
from collections import deque # For early stopping

import torch
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
import torchvision.models as torchvision_models # Import torchvision models
from tqdm import tqdm

# --- Import from Utils ---
from Utils.augmentations import ContrastiveTransform
from Utils.train_utils import set_seed, setup_ddp, cleanup_ddp, is_main_process, setup_logging

# ---------------------- Helper Functions ---------------------
def deep_update(source, overrides):
    """
    Recursively update a dict.
    Subdict's values are updated instead of being overwritten.
    """
    for key, value in overrides.items():
        if isinstance(value, dict) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source

# --------------------- Model Definition --------------------
# (Included directly in simclr.py for consolidation)
class ResNetSimCLR(nn.Module):
    """
    Encoder network using ResNet with an MLP projection head.
    """
    def __init__(self, base_model_name='resnet18', out_dim=128):
        super(ResNetSimCLR, self).__init__()
        # Dynamically get the ResNet model constructor from torchvision
        resnet_constructor = getattr(torchvision_models, base_model_name)
        self.backbone = resnet_constructor(weights=None, num_classes=out_dim) # Use weights=None for random init
        # Note: Original repo used pretrained=False which might differ slightly

        dim_mlp = self.backbone.fc.in_features # Get the input dim of the final layer

        # Replace the final layer with the projection head
        # SimCLR paper uses a 2-layer MLP projection head
        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, out_dim, bias=False) # Original paper didn't use bias here
        )

    def forward(self, x):
        return self.backbone(x)

# --------------------- Argument Parsing --------------------
def parse_args():
    parser = argparse.ArgumentParser(description='SimCLR Training Script')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file (e.g., configs/simclr_baseline.yaml)')
    parser.add_argument('--seed', type=int, default=None, help='Override random seed')
    parser.add_argument('--output_dir', type=str, default=None, help='Override base output directory')
    parser.add_argument('--epochs', type=int, default=None, help='Override number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Override global batch size')
    # Add more overrides if needed (e.g., --lr, --workers)
    return parser.parse_args()

# -------------------- Configuration Loading ------------------
def load_config(config_path, cli_args):
    # Determine base config path relative to the specific config file
    base_config_path = os.path.join(os.path.dirname(config_path), 'base_simclr.yaml')

    # Load base config if it exists
    if os.path.exists(base_config_path):
        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f)
            if config is None: # Handle empty base file
                config = {}
    else:
        print(f"Warning: Base config file not found at {base_config_path}. Starting with empty config.")
        config = {}

    # Load specific config
    with open(config_path, 'r') as f:
        specific_config = yaml.safe_load(f)
        if specific_config is None: # Handle empty specific file
             print(f"Warning: Specific config file {config_path} is empty.")
             specific_config = {}

    # Deep merge specific config into base config
    config = deep_update(config, specific_config)

    # Apply CLI overrides (only if provided)
    if cli_args.seed is not None: config['seed'] = cli_args.seed
    if cli_args.output_dir is not None: config['output']['results_dir'] = cli_args.output_dir
    if cli_args.epochs is not None: config['pretrain']['epochs'] = cli_args.epochs
    if cli_args.batch_size is not None: config['pretrain']['batch_size'] = cli_args.batch_size
    # Add more overrides...

    # Generate dynamic fields if missing
    config.setdefault('seed', random.randint(1, 10000)) # Default seed if none provided
    config['model_name'] = config.get('model', {}).get('name', 'simclr')
    # Infer augmentation name from config filename (e.g., simclr_baseline.yaml -> baseline)
    aug_name = os.path.splitext(os.path.basename(config_path))[0]
    if aug_name.startswith('simclr_'):
        aug_name = aug_name.split('simclr_', 1)[1]
    config['augmentation_name'] = aug_name
    config['output']['run_dir'] = os.path.join(
        config['output']['results_dir'],
        f"{config['model_name']}_{config['augmentation_name']}"
    )

    return config

# ---------------------- Data Loading -----------------------
def get_dataloader(config, rank, world_size):
    transform = ContrastiveTransform(config['augmentations'])
    dataset = STL10(
        root=config['data']['data_dir'],
        split='unlabeled', # Use unlabeled data for pretraining
        download=False,    # Assume already downloaded
        transform=transform
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True, # Shuffle data each epoch
        seed=config['seed'] # Ensure consistent shuffling across epochs if seed is fixed
    )
    # Calculate batch size per GPU
    assert config['pretrain']['batch_size'] % world_size == 0, \
        f"Global batch size {config['pretrain']['batch_size']} must be divisible by world size {world_size}"
    per_gpu_batch_size = config['pretrain']['batch_size'] // world_size

    loader = DataLoader(
        dataset,
        batch_size=per_gpu_batch_size,
        sampler=sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True # Drop last incomplete batch
    )
    return loader, sampler

# ---------------------- Model Building -------------------
def build_model(config):
    base_model = config['model']['backbone'] # e.g., 'resnet18'
    out_dim = config['model']['projection_dim'] # e.g., 128
    model = ResNetSimCLR(base_model_name=base_model, out_dim=out_dim)
    return model

# ------------------- Optimizer & Scheduler -----------------
# Corrected version for simclr.py
def build_optimizer_scheduler(model_params, config, steps_per_epoch):
    optim_cfg = config['pretrain']
    optimizer_name = optim_cfg['optimizer'].lower()
    lr = float(optim_cfg['learning_rate']) # Also ensure LR is float
    wd = float(optim_cfg['weight_decay'])  # <<< CORRECTED: Explicitly convert wd to float

    if optimizer_name == 'sgd':
        momentum = float(optim_cfg.get('momentum', 0.9)) # Ensure momentum is float
        optimizer = optim.SGD(model_params, lr=lr, momentum=momentum, weight_decay=wd)
    elif optimizer_name == 'adam':
         optimizer = optim.Adam(model_params, lr=lr, weight_decay=wd)
    # Add AdamW if needed
    # elif optimizer_name == 'adamw':
    #     optimizer = optim.AdamW(model_params, lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    scheduler_name = optim_cfg['lr_schedule'].lower()
    epochs = optim_cfg['epochs']
    if scheduler_name == 'cosine':
         total_steps = steps_per_epoch * epochs
         # Ensure eta_min is float if specified, default is 0 which is fine
         eta_min = float(optim_cfg.get('eta_min', 0.0))
         scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)
    elif scheduler_name == 'step': # Example: Add step decay if needed
        # milestones = optim_cfg.get('lr_milestones', [epochs // 2, epochs * 3 // 4])
        # gamma = optim_cfg.get('lr_gamma', 0.1)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        raise NotImplementedError("StepLR scheduler needs configuration")
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    return optimizer, scheduler

# ---------------------- Loss Function ----------------------
def info_nce_loss(features, batch_size, n_views, temperature, device):
    """
    Calculates the InfoNCE loss for SimCLR.
    Adapted from the original SimCLR repo.
    Args:
        features (Tensor): Combined features from all views (N*views, proj_dim).
        batch_size (int): Batch size *before* concatenating views.
        n_views (int): Number of views per image (usually 2).
        temperature (float): Softmax temperature.
        device: Torch device.
    Returns:
        Tuple[Tensor, Tensor, Tensor]: Loss, Logits, Labels for accuracy calculation.
    """
    labels = torch.cat([torch.arange(batch_size) for _ in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (n_views * batch_size, n_views * batch_size)
    # assert similarity_matrix.shape == labels.shape

    # Discard diagonal elements (self-similarities)
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels_no_diag = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix_no_diag = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix_no_diag.shape == labels_no_diag.shape

    # Select positive similarities (mask by labels)
    positives = similarity_matrix_no_diag[labels_no_diag.bool()].view(labels.shape[0], -1)

    # Select negative similarities (mask by ~labels)
    negatives = similarity_matrix_no_diag[~labels_no_diag.bool()].view(similarity_matrix.shape[0], -1)

    # Concatenate positive and negative similarities for logits
    logits = torch.cat([positives, negatives], dim=1)

    # Labels for CrossEntropyLoss are all zeros (positive similarity is target)
    criterion_labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    # Apply temperature scaling
    logits = logits / temperature

    # Calculate CrossEntropy loss
    criterion = nn.CrossEntropyLoss().to(device)
    loss = criterion(logits, criterion_labels)

    return loss, logits, criterion_labels

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # Adapted from original repo utils.py
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True) # Original had reshape
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# ---------------------- Training Loop ----------------------
def train_one_epoch(model, loader, sampler, optimizer, scheduler, criterion_fn, config, epoch, rank, world_size, scaler):
    """Runs one epoch of SimCLR pretraining."""
    model.train()
    sampler.set_epoch(epoch) # Ensure proper shuffling with DDP

    # Get config values
    batch_size_global = config['pretrain']['batch_size']
    n_views = config['pretrain']['n_views']
    temperature = config['pretrain']['temperature']
    use_amp = config['pretrain']['fp16_precision']
    batch_size_per_gpu = batch_size_global // world_size

    epoch_loss = 0.0
    epoch_top1_acc = 0.0
    step_losses = [] # Collect loss per step for detailed logging if needed

    # Setup progress bar only on main process
    pbar = None
    if is_main_process():
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{config['pretrain']['epochs']}", leave=True)
    else:
        pbar = loader # Iterate directly if not main process

    start_time = time.time()
    for step, (images, _) in enumerate(pbar): # STL10 unlabeled loader returns images, _
        # `images` is a list of tensors [view1_batch, view2_batch] from ContrastiveTransform
        # Combine views along the batch dimension
        combined_images = torch.cat(images, dim=0).to(rank, non_blocking=True)

        with autocast(enabled=use_amp):
            features = model(combined_images) # Forward pass through DDP model
            loss, logits, labels = criterion_fn(
                features, batch_size_per_gpu, n_views, temperature, rank
            )

        # Backpropagation
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Step the scheduler (Cosine Annealing is typically per step)
        if config['pretrain']['lr_schedule'].lower() == 'cosine':
             scheduler.step()

        # Accumulate metrics
        current_loss = loss.item()
        epoch_loss += current_loss
        step_losses.append({'epoch': epoch, 'step': step, 'loss': current_loss})

        # Calculate accuracy (optional, but good for monitoring)
        top1, _ = accuracy(logits, labels, topk=(1, 5)) # Use the labels created for the loss function
        epoch_top1_acc += top1.item()

        # Update progress bar on main process
        if is_main_process():
            lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({"Loss": f"{current_loss:.4f}", "Top-1 Acc": f"{top1.item():.2f}%", "LR": f"{lr:.6f}"})

    end_time = time.time()
    epoch_duration = end_time - start_time
    steps_per_epoch = len(loader)
    avg_epoch_loss = epoch_loss / steps_per_epoch
    avg_epoch_top1_acc = epoch_top1_acc / steps_per_epoch

    # Log epoch summary on main process
    if is_main_process():
        logging.info(f"Epoch {epoch+1} finished. Avg Loss: {avg_epoch_loss:.4f}, "
                     f"Avg Top-1 Acc: {avg_epoch_top1_acc:.2f}%, Duration: {epoch_duration:.2f}s")

    return avg_epoch_loss, avg_epoch_top1_acc, step_losses # Return epoch average loss/acc for early stopping

# ---------------------- Main Function ----------------------
def main_worker(rank, world_size, config):
    # 4. Initialize DDP
    setup_ddp(rank, world_size)

    # 5. Setup Logging (Call after DDP setup to use is_main_process)
    run_dir = config['output']['run_dir']
    log_name = config['output']['log_name']
    setup_logging(run_dir, log_name)

    main_proc = is_main_process() # Cache the result

    if main_proc:
        logging.info("=====================================================")
        logging.info(f"Starting SimCLR Training Run: {config['model_name']}_{config['augmentation_name']}")
        logging.info(f" 세계 크기(World Size): {world_size}, 순위(Rank): {rank}")
        logging.info(f" 구성(Config): {config}")
        logging.info("=====================================================")
        # Save final effective config to run directory
        final_config_path = os.path.join(run_dir, 'effective_config.yaml')
        with open(final_config_path, 'w') as f:
             yaml.dump(config, f, default_flow_style=False)
        logging.info(f"Effective configuration saved to {final_config_path}")

    # 3. Set Random Seed (Call after DDP setup to ensure seed consistency across processes if needed, though DDP sampler handles data shuffling seed)
    set_seed(config['seed'] + rank) # Add rank to seed for potentially different initializations if desired (e.g., data augmentation workers)
    if main_proc:
        logging.info(f"Random seed set to {config['seed']} (rank 0 seed)")

    # 6. Build Data Loader
    train_loader, train_sampler = get_dataloader(config, rank, world_size)
    steps_per_epoch = len(train_loader)
    if main_proc:
        logging.info(f"DataLoader created. Steps per epoch: {steps_per_epoch}")

    # 7. Build Model, Optimizer, Scheduler
    model = build_model(config).to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False) # Set find_unused_parameters=True if encountering issues
    optimizer, scheduler = build_optimizer_scheduler(model.parameters(), config, steps_per_epoch)
    if main_proc:
        logging.info("Model, Optimizer, and Scheduler built and wrapped in DDP.")
        # logging.info(f"Model Structure:\n{model}") # Optional: Log model structure

    # 8. Define Loss Function (already defined globally as info_nce_loss)
    criterion_fn = info_nce_loss

    # AMP Grad Scaler
    scaler = GradScaler(enabled=config['pretrain']['fp16_precision'])
    if main_proc and config['pretrain']['fp16_precision']:
        logging.info("Automatic Mixed Precision (AMP) enabled.")

    # Training History & Early Stopping Setup
    training_history = [] # List to store epoch summaries
    early_stopping_enabled = config['pretrain']['early_stopping']['enabled']
    if main_proc and early_stopping_enabled:
        patience = config['pretrain']['early_stopping']['patience']
        min_delta = config['pretrain']['early_stopping']['min_delta']
        best_loss = float('inf')
        epochs_no_improve = 0
        loss_history = deque(maxlen=patience) # Store recent losses
        logging.info(f"Early stopping enabled: Patience={patience}, Min Delta={min_delta}")

    # Resume logic placeholder (implement if needed)
    start_epoch = 0
    # checkpoint_path = os.path.join(run_dir, config['output']['checkpoint_name'])
    # if os.path.exists(checkpoint_path):
    #     # Load checkpoint logic...
    #     # map_location = {'cuda:%d' % 0: 'cuda:%d' % rank} # Map location for DDP
    #     # checkpoint = torch.load(checkpoint_path, map_location=map_location)
    #     # model.module.load_state_dict(checkpoint['state_dict'])
    #     # optimizer.load_state_dict(checkpoint['optimizer'])
    #     # scheduler.load_state_dict(checkpoint['scheduler'])
    #     # start_epoch = checkpoint['epoch']
    #     # best_loss = checkpoint.get('best_loss', float('inf')) # Resume best loss if saved
    #     # scaler.load_state_dict(checkpoint['scaler']) # Resume scaler state if using AMP
    #     # if main_proc: logging.info(f"Resuming training from epoch {start_epoch + 1}")
    #     pass


    # 9. Training Loop
    if main_proc:
        logging.info(f"Starting training for {config['pretrain']['epochs']} epochs...")

    global_start_time = time.time()
    all_step_losses = [] # Collect step losses from main process over all epochs

    for epoch in range(start_epoch, config['pretrain']['epochs']):
        epoch_start_time = time.time()

        avg_loss, avg_acc, step_losses = train_one_epoch(
            model, train_loader, train_sampler, optimizer, scheduler, criterion_fn, config, epoch, rank, world_size, scaler
        )

        # Collect results on main process
        if main_proc:
            epoch_duration = time.time() - epoch_start_time
            training_history.append({
                'epoch': epoch + 1,
                'avg_loss': avg_loss,
                'avg_top1_acc': avg_acc,
                'duration_sec': epoch_duration,
                'learning_rate': scheduler.get_last_lr()[0]
            })
            all_step_losses.extend(step_losses) # Append step losses from this epoch

            # 10. Checkpoint & Metrics Saving (Main Process Only)
            # Save checkpoint periodically or based on performance if needed
            # Example: Save last checkpoint
            checkpoint_path = os.path.join(run_dir, config['output']['checkpoint_name'])
            save_data = {
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(), # Use model.module with DDP
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'config': config, # Save config for reference
                'best_loss': best_loss if early_stopping_enabled else None, # Save best loss for early stopping resume
                'scaler': scaler.state_dict() if config['pretrain']['fp16_precision'] else None
            }
            torch.save(save_data, checkpoint_path)
            # logging.info(f"Checkpoint saved to {checkpoint_path}") # Can be noisy, log less often

            # Early Stopping Check
            if early_stopping_enabled:
                current_loss = avg_loss # Use average epoch loss
                if current_loss < best_loss - min_delta:
                    best_loss = current_loss
                    epochs_no_improve = 0
                    logging.info(f"EarlyStopping: New best loss: {best_loss:.4f}. Saving best model.")
                    best_model_path = os.path.join(run_dir, 'best_model.pth')
                    torch.save(model.module.state_dict(), best_model_path)
                else:
                    epochs_no_improve += 1
                    logging.info(f"EarlyStopping: No improvement for {epochs_no_improve}/{patience} epochs.")

                if epochs_no_improve >= patience:
                    logging.warning(f"Early stopping triggered after epoch {epoch + 1} due to no improvement for {patience} epochs.")
                    stop_signal = torch.tensor(1).to(rank) # Signal to stop
                else:
                    stop_signal = torch.tensor(0).to(rank)
            else:
                 stop_signal = torch.tensor(0).to(rank) # No early stopping

        else:
            stop_signal = torch.tensor(0).to(rank) # Placeholder for non-main processes

        # Broadcast stop signal from main process to all processes
        dist.broadcast(stop_signal, src=0)

        # Synchronize processes before checking stop condition or starting next epoch
        dist.barrier()

        if stop_signal.item() == 1:
            break # Exit loop if stop signal received

    # --- End of Training Loop ---

    # Final Actions (Main Process Only)
    if main_proc:
        total_duration = time.time() - global_start_time
        logging.info("=====================================================")
        logging.info(f"Training finished after {epoch + 1} epochs.")
        logging.info(f"Total training time: {total_duration / 3600:.2f} hours")

        # Save final model state explicitly
        final_model_path = os.path.join(run_dir, 'final_model.pth')
        torch.save(model.module.state_dict(), final_model_path)
        logging.info(f"Final model state dict saved to {final_model_path}")

        # Save training history (epoch summaries) to CSV
        history_df = pd.DataFrame(training_history)
        history_csv_path = os.path.join(run_dir, config['output']['train_loss_csv'].format(
            model=config['model_name'], aug=config['augmentation_name']
        ))
        history_df.to_csv(history_csv_path, index=False)
        logging.info(f"Training history saved to {history_csv_path}")

        # Optional: Save detailed step losses if needed
        # step_loss_df = pd.DataFrame(all_step_losses)
        # step_loss_csv_path = os.path.join(run_dir, 'step_losses.csv')
        # step_loss_df.to_csv(step_loss_csv_path, index=False)
        # logging.info(f"Detailed step losses saved to {step_loss_csv_path}")

        # 11. Evaluation Placeholders
        logging.info("Starting evaluation phase (placeholders)...")
        try:
            # --- Placeholder call for Linear Probing ---
            # run_linear_probe(final_model_path, config, rank, world_size) # Pass rank/world_size if needed
            logging.info("Placeholder: run_linear_probe() would run here.")
            # Example: Write dummy accuracy file
            linear_acc_file = os.path.join(run_dir, config['output']['linear_acc_txt'].format(
                 model=config['model_name'], aug=config['augmentation_name']
            ))
            with open(linear_acc_file, 'w') as f:
                f.write("Linear Probe Accuracy: N/A (Not Implemented)\n")

            # --- Placeholder call for k-NN Evaluation ---
            # run_knn_evaluation(final_model_path, config, rank, world_size)
            logging.info("Placeholder: run_knn_evaluation() would run here.")
            # Example: Write dummy accuracy file
            knn_acc_file = os.path.join(run_dir, config['output']['knn_acc_txt'].format(
                 model=config['model_name'], aug=config['augmentation_name']
            ))
            with open(knn_acc_file, 'w') as f:
                f.write("k-NN Accuracy: N/A (Not Implemented)\n")

            logging.info("Evaluation phase finished (placeholders).")
        except Exception as e:
            logging.error(f"Error during placeholder evaluation phase: {e}", exc_info=True)

        logging.info("=====================================================")

    # 12. Cleanup DDP
    cleanup_ddp()

# ---------------- Execution Guard & DDP Launch ---------------
if __name__ == "__main__":
    # 1. Parse Arguments
    args = parse_args()

    # 2. Load Configuration
    config = load_config(args.config, args)

    # Determine world size (number of GPUs)
    world_size = torch.cuda.device_count()
    if world_size < 1:
        print("No CUDA GPUs found. Exiting.")
        sys.exit(1)
    if config['pretrain']['batch_size'] % world_size != 0:
         print(f"Error: Global batch size {config['pretrain']['batch_size']} must be divisible by the number of GPUs {world_size}.")
         sys.exit(1)

    # --- Launch with torchrun ---
    # torchrun automatically sets MASTER_ADDR, MASTER_PORT, RANK, LOCAL_RANK, WORLD_SIZE
    # We just need to get rank and world_size inside the worker function.
    try:
        rank = int(os.environ["RANK"])
        world_size_env = int(os.environ["WORLD_SIZE"])
        if world_size_env != world_size:
             print(f"Warning: WORLD_SIZE from env ({world_size_env}) does not match torch.cuda.device_count() ({world_size}). Using env value.")
             world_size = world_size_env
        print(f"Launching DDP training via torchrun. Rank: {rank}, World Size: {world_size}")
        main_worker(rank=rank, world_size=world_size, config=config)
    except KeyError:
        print("Error: RANK or WORLD_SIZE environment variables not set.")
        print("Please launch this script using 'torchrun --nproc_per_node=<num_gpus> simclr.py ...'")
        sys.exit(1)
    except Exception as e:
         print(f"An error occurred during training: {e}")
         # Ensure cleanup happens even if there's an error in the worker
         cleanup_ddp()
         # Log the exception if logging was set up
         if logging.getLogger().hasHandlers():
              logging.error("Training script terminated with an error.", exc_info=True)
         sys.exit(1)

    print("SimCLR script finished successfully.")