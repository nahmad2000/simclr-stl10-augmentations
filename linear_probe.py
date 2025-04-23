# linear_probe.py (Updated)
# Trains and evaluates a linear classifier, returns results, and saves epoch-specific txt.

import argparse
import os
import yaml
import logging
import time
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torchvision.datasets import STL10
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torchvision.models as torchvision_models

# --- Helper Functions (unchanged) ---
def deep_update(source, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source

def load_config_for_eval(config_path):
    base_config_path = Path(config_path).parent / 'base_simclr.yaml'
    config = {}
    if base_config_path.exists():
        with open(base_config_path, 'r') as f: config = yaml.safe_load(f) or {}
    else: logging.warning(f"Base config {base_config_path} not found.")

    with open(config_path, 'r') as f: specific_config = yaml.safe_load(f) or {}
    config = deep_update(config, specific_config)

    config.setdefault('model', {})
    config.setdefault('data', {})
    config.setdefault('evaluation', {}).setdefault('linear_probe', {})
    config.setdefault('output', {})
    config['model_name'] = config['model'].get('name', 'simclr')
    config_filename = Path(config_path).name
    aug_name = Path(config_path).stem
    if aug_name.startswith(f"{config['model_name']}_"):
         aug_name = aug_name.split(f"{config['model_name']}_", 1)[1]
    config['augmentation_name'] = aug_name
    config['output']['run_dir'] = Path(config['output'].get('results_dir', './results')) / \
                                   f"{config['model_name']}_{config['augmentation_name']}"
    return config

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

# --- Backbone Model Definition (unchanged) ---
class ResNetBackbone(nn.Module):
    def __init__(self, base_model_name='resnet18'):
        super(ResNetBackbone, self).__init__()
        try:
            resnet_constructor = getattr(torchvision_models, base_model_name)
            self.backbone = resnet_constructor(weights=None, num_classes=10)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            logging.debug(f"Created ResNetBackbone: {base_model_name}, feature_dim={self.feature_dim}")
        except AttributeError:
             logging.error(f"Invalid backbone model name: {base_model_name}")
             raise ValueError(f"Invalid backbone model name: {base_model_name}")
    def forward(self, x): return self.backbone(x)

# --- Feature Extraction Dataset Wrapper (unchanged) ---
class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __len__(self): return len(self.features)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

# --- Feature Extraction Function (unchanged) ---
@torch.no_grad()
def extract_features(encoder, loader, device):
    encoder.eval()
    all_features, all_labels = [], []
    logging.info(f"Extracting features using device: {device}...")
    for images, labels in tqdm(loader, desc="Feature Extraction", leave=False):
        images = images.to(device, non_blocking=True)
        features = encoder(images)
        all_features.append(features.cpu())
        all_labels.append(labels.cpu())
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    logging.info(f"Finished feature extraction. Feature shape: {all_features.shape}")
    return all_features, all_labels

# --- CORE EVALUATION FUNCTION (Updated Return Value) ---
def evaluate_linear_probe(checkpoint_path: str, config: dict, epoch: int, device: torch.device) -> list:
    """
    Performs linear probe evaluation and returns results as a list of dicts.
    Also saves results to an epoch-specific .txt file.

    Returns:
        List of dictionaries, e.g.,
        [
            {'epoch': 50, 'metric_type': 'LinearProbe', 'metric_name': 'Top1Acc', 'k_value': None, 'value': 75.3},
            {'epoch': 50, 'metric_type': 'LinearProbe', 'metric_name': 'Top5Acc', 'k_value': None, 'value': 92.1}
        ]
        Returns empty list on failure.
    """
    run_dir = Path(config['output']['run_dir'])
    run_dir.mkdir(parents=True, exist_ok=True)

    eval_cfg = config['evaluation']['linear_probe']
    data_cfg = config['data']
    model_cfg = config['model']
    results_list = [] # Initialize empty list for results

    logging.info(f"--- Starting Linear Probe Eval Epoch {epoch} ---")
    # (Logging checkpoint/device unchanged)

    # --- Load Backbone Model (unchanged logic) ---
    encoder = ResNetBackbone(base_model_name=model_cfg.get('backbone', 'resnet18'))
    try:
        map_location = device
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            backbone_state_dict = {k.replace('module.backbone.', 'backbone.'): v for k, v in state_dict.items() if 'backbone.' in k}
            backbone_state_dict = {k.replace('backbone.',''): v for k,v in backbone_state_dict.items() if not k.startswith('backbone.fc.')}
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
            backbone_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if 'backbone.' in k}
            backbone_state_dict = {k.replace('backbone.', ''): v for k, v in backbone_state_dict.items() if not k.startswith('fc.')}
        else: raise ValueError("Checkpoint format not recognized.")
        if not backbone_state_dict:
             logging.warning(f"No 'backbone.' keys found in {checkpoint_path}. Trying without prefix.")
             backbone_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
        if not backbone_state_dict: raise ValueError("Could not extract backbone state dict.")
        msg = encoder.load_state_dict(backbone_state_dict, strict=False)
        logging.info(f"Backbone state_dict loaded. Missing: {msg.missing_keys}, Unexpected: {msg.unexpected_keys}")
    except FileNotFoundError:
        logging.error(f"Checkpoint file not found: {checkpoint_path}")
        return [] # Return empty list on critical error
    except Exception as e:
        logging.error(f"Error loading checkpoint {checkpoint_path}: {e}", exc_info=True)
        return []
    encoder = encoder.to(device)
    encoder.eval()
    for param in encoder.parameters(): param.requires_grad = False
    # (Logging backbone loaded unchanged)

    # --- Prepare DataLoaders (unchanged logic) ---
    eval_transform = T.Compose([
        T.Resize((data_cfg.get('image_size', 96), data_cfg.get('image_size', 96))),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    try:
        train_dataset = STL10(root=data_cfg['data_dir'], split='train', download=False, transform=eval_transform)
        test_dataset = STL10(root=data_cfg['data_dir'], split='test', download=False, transform=eval_transform)
    except Exception as e:
        logging.error(f"Error loading STL10 dataset from {data_cfg['data_dir']}: {e}")
        return []
    eval_batch_size = eval_cfg.get('batch_size', 256)
    num_workers = data_cfg.get('num_workers', 4)
    train_feature_loader = DataLoader(train_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_feature_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    # (Logging dataloaders prepared unchanged)

    # --- Extract Features (unchanged logic) ---
    train_features, train_labels = extract_features(encoder, train_feature_loader, device)
    test_features, test_labels = extract_features(encoder, test_feature_loader, device)
    del encoder
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- Train Linear Classifier (unchanged logic) ---
    classifier = nn.Linear(train_features.shape[1], 10).to(device)
    optimizer_name = eval_cfg.get('optimizer', 'adam').lower()
    lr = float(eval_cfg.get('learning_rate', 0.0003))
    wd = float(eval_cfg.get('weight_decay', 0.0))
    if optimizer_name == 'adam':
        optimizer = optim.Adam(classifier.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_name == 'sgd':
        momentum = float(eval_cfg.get('momentum', 0.9))
        optimizer = optim.SGD(classifier.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    else:
         logging.error(f"Unsupported optimizer for linear probe: {optimizer_name}")
         return []
    criterion = nn.CrossEntropyLoss()
    probe_epochs = eval_cfg.get('epochs', 100)
    probe_batch_size = eval_cfg.get('probe_batch_size', 1024)
    train_probe_dataset = FeatureDataset(train_features, train_labels)
    train_probe_loader = DataLoader(train_probe_dataset, batch_size=probe_batch_size, shuffle=True, num_workers=0)
    logging.info(f"Training linear classifier for {probe_epochs} epochs...")
    classifier.train()
    train_start_time = time.time()
    for probe_epoch in range(probe_epochs):
        # (Training loop unchanged)
        epoch_loss, epoch_acc1, epoch_acc5 = 0.0, 0.0, 0.0
        for features, labels in train_probe_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            epoch_loss += loss.item()
            epoch_acc1 += acc1.item()
            epoch_acc5 += acc5.item()
        # (Logging progress unchanged)
        if (probe_epoch + 1) % 10 == 0 or probe_epoch == 0 or (probe_epoch + 1) == probe_epochs:
             logging.info(f"  Probe Epoch [{probe_epoch+1}/{probe_epochs}] Loss: {epoch_loss/len(train_probe_loader):.4f}, Train Acc@1: {epoch_acc1/len(train_probe_loader):.2f}%")
    # (Logging training finished unchanged)

    # --- Evaluate on Test Set (unchanged logic) ---
    logging.info("Evaluating linear classifier on the test set...")
    classifier.eval()
    with torch.no_grad():
        test_outputs = classifier(test_features.to(device))
        test_loss = criterion(test_outputs, test_labels.to(device))
        top1_acc, top5_acc = accuracy(test_outputs, test_labels.to(device), topk=(1, 5))
        final_top1_accuracy = top1_acc.item()
        final_top5_accuracy = top5_acc.item()
    logging.info(f"Test Loss: {test_loss.item():.4f}")
    logging.info(f"Test Top-1 Accuracy: {final_top1_accuracy:.2f}%")
    logging.info(f"Test Top-5 Accuracy: {final_top5_accuracy:.2f}%")

    # --- Save Results to TXT file (unchanged logic) ---
    output_filename_format = config['output'].get('linear_acc_txt', 'linear_probe_acc_{model}_{aug}.txt')
    if '{epoch}' in output_filename_format:
         output_filename = output_filename_format.format(model=config['model_name'], aug=config['augmentation_name'], epoch=epoch)
    else:
         base_name = output_filename_format.format(model=config['model_name'], aug=config['augmentation_name'])
         output_filename = f"{Path(base_name).stem}_epoch_{epoch}.txt"
    output_file = run_dir / output_filename
    result_string = f"Checkpoint Epoch: {epoch}\nTop-1 Accuracy: {final_top1_accuracy:.2f}%\nTop-5 Accuracy: {final_top5_accuracy:.2f}%\n"
    try:
        with open(output_file, 'w') as f: f.write(result_string)
        logging.info(f"Linear probe text result for epoch {epoch} saved to: {output_file}")
    except Exception as e:
        logging.error(f"Failed to write linear probe text result file {output_file}: {e}")

    # --- !! NEW: Prepare results for return !! ---
    results_list.append({
        'epoch': epoch,
        'metric_type': 'LinearProbe',
        'metric_name': 'Top1Acc',
        'k_value': None, # Not applicable for Linear Probe
        'value': round(final_top1_accuracy, 4) # Round for consistency
    })
    results_list.append({
        'epoch': epoch,
        'metric_type': 'LinearProbe',
        'metric_name': 'Top5Acc',
        'k_value': None,
        'value': round(final_top5_accuracy, 4)
    })

    logging.info(f"--- Finished Linear Probe Eval Epoch {epoch} ---")
    return results_list # Return the structured results


# --- Standalone Execution (unchanged) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Linear Probe Evaluation Script (Standalone)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path config')
    parser.add_argument('--epoch', type=int, default=0, help='Epoch number (for naming)')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [LP Standalone] %(levelname)s %(message)s", handlers=[logging.StreamHandler()])
    try:
        config = load_config_for_eval(args.config)
        eval_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        run_dir = Path(config['output']['run_dir']); run_dir.mkdir(parents=True, exist_ok=True)
        # Call the main function - results are returned but not used here
        evaluate_linear_probe(checkpoint_path=args.checkpoint, config=config, epoch=args.epoch, device=eval_device)
    except Exception as e: logging.error("Linear probe script failed.", exc_info=True)
    finally: logging.info("Linear probe standalone script finished.")