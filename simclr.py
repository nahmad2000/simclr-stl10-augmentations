# simclr.py (Updated)
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
class ResNetSimCLR(nn.Module):
    """
    Encoder network using ResNet with an MLP projection head.
    """
    def __init__(self, base_model_name='resnet18', out_dim=128):
        super(ResNetSimCLR, self).__init__()
        resnet_constructor = getattr(torchvision_models, base_model_name)
        self.backbone = resnet_constructor(weights=None, num_classes=out_dim)
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, out_dim, bias=True)
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
    return parser.parse_args()

# -------------------- Configuration Loading ------------------
def load_config(config_path, cli_args):
    base_config_path = os.path.join(os.path.dirname(config_path), 'base_simclr.yaml')
    if os.path.exists(base_config_path):
        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    else:
        print(f"Warning: Base config file not found at {base_config_path}. Starting with empty config.")
        config = {}

    with open(config_path, 'r') as f:
        specific_config = yaml.safe_load(f) or {}
        if specific_config is None:
             print(f"Warning: Specific config file {config_path} is empty.")
             specific_config = {}

    config = deep_update(config, specific_config)

    if cli_args.seed is not None: config['seed'] = cli_args.seed
    if cli_args.output_dir is not None: config['output']['results_dir'] = cli_args.output_dir
    if cli_args.epochs is not None: config['pretrain']['epochs'] = cli_args.epochs
    if cli_args.batch_size is not None: config['pretrain']['batch_size'] = cli_args.batch_size

    config.setdefault('seed', random.randint(1, 10000))
    config['model_name'] = config.get('model', {}).get('name', 'simclr')
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
        split='unlabeled',
        download=False,
        transform=transform
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config['seed']
    )
    assert config['pretrain']['batch_size'] % world_size == 0, \
        f"Global batch size {config['pretrain']['batch_size']} must be divisible by world size {world_size}"
    per_gpu_batch_size = config['pretrain']['batch_size'] // world_size

    loader = DataLoader(
        dataset,
        batch_size=per_gpu_batch_size,
        sampler=sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    return loader, sampler

# ---------------------- Model Building -------------------
def build_model(config):
    base_model = config['model']['backbone']
    out_dim = config['model']['projection_dim']
    model = ResNetSimCLR(base_model_name=base_model, out_dim=out_dim)
    return model

# ------------------- Optimizer & Scheduler (UPDATED) -----------------
def build_optimizer_scheduler(model_params, config, steps_per_epoch):
    optim_cfg = config['pretrain']
    optimizer_name = optim_cfg['optimizer'].lower()
    lr = float(optim_cfg['learning_rate'])
    wd = float(optim_cfg['weight_decay'])

    # --- UPDATED Optimizer Creation ---
    if optimizer_name == 'sgd':
        momentum = float(optim_cfg.get('momentum', 0.9))
        optimizer = optim.SGD(model_params, lr=lr, momentum=momentum, weight_decay=wd)
        logging.info(f"Using SGD optimizer: LR={lr}, Momentum={momentum}, WD={wd}")
    elif optimizer_name == 'adam':
         optimizer = optim.Adam(model_params, lr=lr, weight_decay=wd)
         logging.info(f"Using Adam optimizer: LR={lr}, WD={wd}")
    # Add AdamW if needed
    # elif optimizer_name == 'adamw':
    #     optimizer = optim.AdamW(model_params, lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    scheduler_name = optim_cfg['lr_schedule'].lower()
    epochs = optim_cfg['epochs']
    warmup_epochs = 10 # Define the warmup period consistent with reference simclr.py

    if scheduler_name == 'cosine':
        # --- UPDATED T_max Calculation ---
        # Calculate T_max based on epochs AFTER warmup
        # Reference simclr.py only starts stepping scheduler after 10 epochs
        t_max_epochs = epochs - warmup_epochs
        if t_max_epochs <= 0:
             logging.warning(f"Total epochs ({epochs}) <= warmup epochs ({warmup_epochs}). Scheduler will not decay.")
             t_max_epochs = 1 # Avoid T_max=0 error

        eta_min = float(optim_cfg.get('eta_min', 0.0))
        # T_max for CosineAnnealingLR is the number of *scheduling steps* (epochs in this case)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max_epochs, eta_min=eta_min)
        if is_main_process(): # Log only on main process
            logging.info(f"Using CosineAnnealingLR scheduler: T_max={t_max_epochs} epochs (stepping after epoch {warmup_epochs}), eta_min={eta_min}")
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    return optimizer, scheduler

# ---------------------- Loss Function ----------------------
def info_nce_loss(features, batch_size, n_views, temperature, device):
    labels = torch.cat([torch.arange(batch_size) for _ in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)

    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels_no_diag = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix_no_diag = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix_no_diag[labels_no_diag.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix_no_diag[~labels_no_diag.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    criterion_labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
    logits = logits / temperature

    criterion = nn.CrossEntropyLoss().to(device)
    loss = criterion(logits, criterion_labels)
    return loss, logits, criterion_labels

def accuracy(output, target, topk=(1,)):
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

# ---------------------- Training Loop (UPDATED) ----------------------
def train_one_epoch(model, loader, sampler, optimizer, scheduler, criterion_fn, config, epoch, rank, world_size, scaler):
    model.train()
    sampler.set_epoch(epoch)

    batch_size_global = config['pretrain']['batch_size']
    n_views = config['pretrain']['n_views']
    temperature = float(config['pretrain']['temperature']) # Ensure temperature is float
    use_amp = config['pretrain']['fp16_precision']
    batch_size_per_gpu = batch_size_global // world_size

    epoch_loss = 0.0
    epoch_top1_acc = 0.0
    step_losses = []

    pbar = None
    if is_main_process():
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{config['pretrain']['epochs']}", leave=True, file=sys.stdout, dynamic_ncols=True)
    else:
        pbar = loader

    start_time = time.time()
    for step, (images, _) in enumerate(pbar):
        combined_images = torch.cat(images, dim=0).to(rank, non_blocking=True)

        with autocast(enabled=use_amp):
            features = model(combined_images)
            loss, logits, labels = criterion_fn(
                features, batch_size_per_gpu, n_views, temperature, rank
            )

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # --- REMOVED SCHEDULER STEP FROM HERE ---

        current_loss = loss.item()
        epoch_loss += current_loss
        step_losses.append({'epoch': epoch, 'step': step, 'loss': current_loss})

        top1, _ = accuracy(logits, labels, topk=(1, 5))
        epoch_top1_acc += top1.item()

        if is_main_process():
            # Get LR from optimizer during warmup, scheduler after
            if epoch < 10: # Assuming 10 warmup epochs
                 lr = optimizer.param_groups[0]['lr']
            else:
                 lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({"Loss": f"{current_loss:.4f}", "Top-1 Acc": f"{top1.item():.2f}%", "LR": f"{lr:.6f}"})

    end_time = time.time()
    epoch_duration = end_time - start_time
    steps_per_epoch = len(loader)
    avg_epoch_loss = epoch_loss / steps_per_epoch
    avg_epoch_top1_acc = epoch_top1_acc / steps_per_epoch

    if is_main_process():
        logging.info(f"Epoch {epoch+1} finished. Avg Loss: {avg_epoch_loss:.4f}, "
                     f"Avg Top-1 Acc: {avg_epoch_top1_acc:.2f}%, Duration: {epoch_duration:.2f}s")

    return avg_epoch_loss, avg_epoch_top1_acc, step_losses

# ---------------------- Main Function (UPDATED) ----------------------
def main_worker(rank, world_size, config):
    setup_ddp(rank, world_size)
    run_dir = config['output']['run_dir']
    log_name = config['output']['log_name']
    setup_logging(run_dir, log_name)
    main_proc = is_main_process()

    if main_proc:
        logging.info("=====================================================")
        logging.info(f"Starting SimCLR Training Run: {config['model_name']}_{config['augmentation_name']}")
        logging.info(f"World Size: {world_size}, Rank: {rank}")
        logging.info(f"Config: {config}")
        logging.info("=====================================================")
        final_config_path = os.path.join(run_dir, 'effective_config.yaml')
        with open(final_config_path, 'w') as f:
             yaml.dump(config, f, default_flow_style=False)
        logging.info(f"Effective configuration saved to {final_config_path}")

    set_seed(config['seed'] + rank)
    if main_proc:
        logging.info(f"Random seed set to {config['seed']} (rank 0 seed)")

    train_loader, train_sampler = get_dataloader(config, rank, world_size)
    steps_per_epoch = len(train_loader)
    if main_proc:
        logging.info(f"DataLoader created. Steps per epoch: {steps_per_epoch}")

    model = build_model(config).to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    optimizer, scheduler = build_optimizer_scheduler(model.parameters(), config, steps_per_epoch)
    if main_proc:
        logging.info("Model, Optimizer, and Scheduler built and wrapped in DDP.")

    criterion_fn = info_nce_loss
    scaler = GradScaler(enabled=config['pretrain']['fp16_precision'])
    if main_proc and config['pretrain']['fp16_precision']:
        logging.info("Automatic Mixed Precision (AMP) enabled.")

    training_history = []
    early_stopping_enabled = config['pretrain']['early_stopping']['enabled']
    warmup_epochs = 10 # Define warmup period

    if main_proc and early_stopping_enabled:
        patience = config['pretrain']['early_stopping']['patience']
        min_delta = float(config['pretrain']['early_stopping']['min_delta']) # Ensure float
        best_loss = float('inf')
        epochs_no_improve = 0
        logging.info(f"Early stopping enabled: Patience={patience}, Min Delta={min_delta}")

    start_epoch = 0
    # Resume logic placeholder (implement if needed)
    global_start_time = time.time()
    all_step_losses = []

    if main_proc:
        logging.info(f"Starting training for {config['pretrain']['epochs']} epochs...")

    for epoch in range(start_epoch, config['pretrain']['epochs']):
        epoch_start_time = time.time()

        avg_loss, avg_acc, step_losses = train_one_epoch(
            model, train_loader, train_sampler, optimizer, scheduler, criterion_fn, config, epoch, rank, world_size, scaler
        )

        stop_signal_val = 0 # Default to continue
        if main_proc:
            epoch_duration = time.time() - epoch_start_time
            # --- UPDATED: Get correct LR for logging ---
            if epoch < warmup_epochs:
                 current_lr = optimizer.param_groups[0]['lr']
            else:
                 current_lr = scheduler.get_last_lr()[0]

            training_history.append({
                'epoch': epoch + 1,
                'avg_loss': avg_loss,
                'avg_top1_acc': avg_acc,
                'duration_sec': epoch_duration,
                'learning_rate': current_lr
            })
            all_step_losses.extend(step_losses)

            # --- ADDED: Scheduler Step Logic ---
            if epoch >= warmup_epochs:
                 scheduler.step()
                 new_lr = scheduler.get_last_lr()[0]
                 logging.debug(f"Epoch {epoch+1}: Stepped scheduler. New LR: {new_lr:.6f}") # Debug level log
            else:
                 logging.debug(f"Epoch {epoch+1}: Warmup phase. No scheduler step.") # Debug level log

            # Checkpoint Saving
            checkpoint_path = os.path.join(run_dir, config['output']['checkpoint_name'])
            save_data = {
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'config': config,
                'best_loss': best_loss if early_stopping_enabled else None,
                'scaler': scaler.state_dict() if config['pretrain']['fp16_precision'] else None
            }
            torch.save(save_data, checkpoint_path)

            # Early Stopping Check
            if early_stopping_enabled:
                current_loss = avg_loss
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
                    logging.warning(f"Early stopping triggered after epoch {epoch + 1}.")
                    stop_signal_val = 1 # Signal to stop
        # --- End Main Proc Block ---

        # Broadcast stop signal from main process to all processes
        stop_signal = torch.tensor(stop_signal_val).to(rank)
        dist.broadcast(stop_signal, src=0)

        # Synchronize processes before checking stop condition or starting next epoch
        dist.barrier()

        if stop_signal.item() == 1:
            break # Exit loop if stop signal received

    # --- End of Training Loop ---

    if main_proc:
        total_duration = time.time() - global_start_time
        logging.info("=====================================================")
        logging.info(f"Training finished after {epoch + 1} epochs.") # Use final epoch value
        logging.info(f"Total training time: {total_duration / 3600:.2f} hours")

        final_model_path = os.path.join(run_dir, 'final_model.pth')
        torch.save(model.module.state_dict(), final_model_path)
        logging.info(f"Final model state dict saved to {final_model_path}")

        history_df = pd.DataFrame(training_history)
        history_csv_path = os.path.join(run_dir, config['output']['train_loss_csv'].format(
            model=config['model_name'], aug=config['augmentation_name']
        ))
        history_df.to_csv(history_csv_path, index=False)
        logging.info(f"Training history saved to {history_csv_path}")

        logging.info("Evaluation phase placeholders...")
        linear_acc_file = os.path.join(run_dir, config['output']['linear_acc_txt'].format(
                model=config['model_name'], aug=config['augmentation_name']
        ))
        with open(linear_acc_file, 'w') as f: f.write("Linear Probe Accuracy: Needs to be run separately.\n")
        knn_acc_file = os.path.join(run_dir, config['output']['knn_acc_txt'].format(
                model=config['model_name'], aug=config['augmentation_name']
        ))
        with open(knn_acc_file, 'w') as f: f.write("k-NN Accuracy: Needs to be run separately.\n")
        logging.info("=====================================================")

    cleanup_ddp()

# ---------------- Execution Guard & DDP Launch ---------------
if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config, args)
    world_size = torch.cuda.device_count()
    if world_size < 1:
        print("No CUDA GPUs found. Exiting.")
        sys.exit(1)
    if config['pretrain']['batch_size'] % world_size != 0:
         print(f"Error: Global batch size {config['pretrain']['batch_size']} is not divisible by the number of GPUs {world_size}.")
         sys.exit(1)

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
         cleanup_ddp()
         if logging.getLogger().hasHandlers():
              logging.error("Training script terminated with an error.", exc_info=True)
         sys.exit(1)

    print("SimCLR script finished successfully.")