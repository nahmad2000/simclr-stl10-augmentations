# knn_eval.py (Updated)
# Evaluates features using k-NN, returns results, and saves epoch-specific txt.

import argparse
import os
import yaml
import logging
import time
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.datasets import STL10
from torch.utils.data import DataLoader
from tqdm import tqdm
try:
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import normalize as sk_normalize
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not found. k-NN evaluation will be skipped.")
import torchvision.models as torchvision_models

# --- Helper Functions (unchanged) ---
def deep_update(source, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else: source[key] = overrides[key]
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
    config.setdefault('evaluation', {}).setdefault('knn', {})
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

# --- Feature Extraction Function (unchanged) ---
@torch.no_grad()
def extract_features_numpy(encoder, loader, device):
    encoder.eval()
    all_features, all_labels = [], []
    logging.info(f"Extracting features for k-NN using device: {device}...")
    for images, labels in tqdm(loader, desc="k-NN Feature Extraction", leave=False):
        images = images.to(device, non_blocking=True)
        features = encoder(images)
        all_features.append(features.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    logging.info(f"Finished k-NN feature extraction. Feature shape: {all_features.shape}")
    return all_features, all_labels

# --- CORE EVALUATION FUNCTION (Updated Return Value) ---
def evaluate_knn(checkpoint_path: str, config: dict, epoch: int, device: torch.device) -> list:
    """
    Performs k-NN evaluation and returns results as a list of dicts.
    Also saves results to an epoch-specific .txt file.

    Returns:
        List of dictionaries, e.g.,
        [
            {'epoch': 50, 'metric_type': 'kNN', 'metric_name': 'Top1Acc', 'k_value': 1, 'value': 70.1},
            {'epoch': 50, 'metric_type': 'kNN', 'metric_name': 'Top1Acc', 'k_value': 5, 'value': 74.5},
            ...
        ]
        Returns empty list on failure or if scikit-learn is missing.
    """
    if not SKLEARN_AVAILABLE:
        logging.warning("Skipping k-NN evaluation because scikit-learn is not installed.")
        return []

    run_dir = Path(config['output']['run_dir'])
    run_dir.mkdir(parents=True, exist_ok=True)

    eval_cfg = config['evaluation']['knn']
    data_cfg = config['data']
    model_cfg = config['model']
    results_list = [] # Initialize empty list

    logging.info(f"--- Starting k-NN Eval Epoch {epoch} ---")
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
             logging.warning(f"No 'backbone.' keys found in {checkpoint_path}. Trying without.")
             backbone_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
        if not backbone_state_dict: raise ValueError("Could not extract backbone state dict.")
        msg = encoder.load_state_dict(backbone_state_dict, strict=False)
        logging.info(f"Backbone state_dict loaded. Missing: {msg.missing_keys}, Unexpected: {msg.unexpected_keys}")
    except FileNotFoundError:
        logging.error(f"Checkpoint file not found: {checkpoint_path}")
        return []
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
    eval_batch_size = config['evaluation'].get('linear_probe', {}).get('batch_size', 256)
    num_workers = data_cfg.get('num_workers', 4)
    train_feature_loader = DataLoader(train_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_feature_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    # (Logging dataloaders prepared unchanged)

    # --- Extract Features (unchanged logic) ---
    train_features, train_labels = extract_features_numpy(encoder, train_feature_loader, device)
    test_features, test_labels = extract_features_numpy(encoder, test_feature_loader, device)
    del encoder
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- Perform k-NN Classification (unchanged logic) ---
    k_values = eval_cfg.get('k_values', [1, 5, 10])
    if isinstance(k_values, int): k_values = [k_values]
    metric = eval_cfg.get('metric', 'cosine')
    if metric == 'cosine':
        logging.info("Normalizing features (L2 norm) for cosine distance...")
        train_features = sk_normalize(train_features, axis=1)
        test_features = sk_normalize(test_features, axis=1)
    knn_results_dict = {} # Store {k: acc}
    logging.info(f"Performing k-NN classification for k={k_values}, metric='{metric}'...")
    knn_start_time = time.time()
    for k in k_values:
        logging.info(f"  Running k-NN for k={k}...")
        knn_classifier = KNeighborsClassifier(n_neighbors=k, metric=metric, n_jobs=-1)
        knn_classifier.fit(train_features, train_labels)
        test_predictions = knn_classifier.predict(test_features)
        accuracy_val = accuracy_score(test_labels, test_predictions) * 100
        knn_results_dict[k] = accuracy_val
        logging.info(f"    k={k} | Test Top-1 Accuracy: {accuracy_val:.2f}%")
    # (Logging k-NN finished unchanged)

    # --- Save Results to TXT file (unchanged logic) ---
    output_filename_format = config['output'].get('knn_acc_txt', 'knn_acc_{model}_{aug}.txt')
    if '{epoch}' in output_filename_format:
         output_filename = output_filename_format.format(model=config['model_name'], aug=config['augmentation_name'], epoch=epoch)
    else:
         base_name = output_filename_format.format(model=config['model_name'], aug=config['augmentation_name'])
         output_filename = f"{Path(base_name).stem}_epoch_{epoch}.txt"
    output_file = run_dir / output_filename
    result_lines = [f"Checkpoint Epoch: {epoch}"]
    for k in sorted(knn_results_dict.keys()):
         result_lines.append(f"k-NN Top-1 Accuracy (k={k}): {knn_results_dict[k]:.2f}%")
    result_string = "\n".join(result_lines) + "\n"
    try:
        with open(output_file, 'w') as f: f.write(result_string)
        logging.info(f"k-NN evaluation text results for epoch {epoch} saved to: {output_file}")
    except Exception as e:
        logging.error(f"Failed to write k-NN text result file {output_file}: {e}")

    # --- !! NEW: Prepare results for return !! ---
    for k, acc in knn_results_dict.items():
        results_list.append({
            'epoch': epoch,
            'metric_type': 'kNN',
            'metric_name': 'Top1Acc',
            'k_value': k,
            'value': round(acc, 4) # Round for consistency
        })

    logging.info(f"--- Finished k-NN Eval Epoch {epoch} ---")
    return results_list # Return structured results


# --- Standalone Execution (unchanged) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='k-NN Evaluation Script (Standalone)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path config')
    parser.add_argument('--epoch', type=int, default=0, help='Epoch number (for naming)')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [KNN Standalone] %(levelname)s %(message)s", handlers=[logging.StreamHandler()])
    if not SKLEARN_AVAILABLE: logging.error("Cannot run k-NN: scikit-learn not installed.")
    else:
        try:
            config = load_config_for_eval(args.config)
            eval_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            run_dir = Path(config['output']['run_dir']); run_dir.mkdir(parents=True, exist_ok=True)
            evaluate_knn(checkpoint_path=args.checkpoint, config=config, epoch=args.epoch, device=eval_device)
        except Exception as e: logging.error("k-NN script failed.", exc_info=True)
        finally: logging.info("k-NN standalone script finished.")