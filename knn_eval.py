# knn_eval.py (Updated)
# Evaluates features from a pretrained SimCLR model using k-NN classification
# for k = 1, 5, 10.

import argparse
import os
import yaml
import logging
import time
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.datasets import STL10
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize # For cosine similarity

import torchvision.models as torchvision_models # Import torchvision models

# --- Helper Functions (Copied from simclr.py for standalone execution) ---
def deep_update(source, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source

def load_config(config_path):
    base_config_path = os.path.join(os.path.dirname(config_path), 'base_simclr.yaml')
    if os.path.exists(base_config_path):
        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    with open(config_path, 'r') as f:
        specific_config = yaml.safe_load(f) or {}

    config = deep_update(config, specific_config)

    # Add model/aug names if needed for output formatting
    config['model_name'] = config.get('model', {}).get('name', 'simclr')
    aug_name = os.path.splitext(os.path.basename(config_path))[0]
    if aug_name.startswith('simclr_'):
        aug_name = aug_name.split('simclr_', 1)[1]
    elif aug_name.startswith('byol_'): # Add basic support for BYOL naming
         aug_name = aug_name.split('byol_', 1)[1]
    config['augmentation_name'] = aug_name
    # Derive run_dir based on config location
    config['output']['run_dir'] = os.path.join(
        config['output']['results_dir'],
        f"{config['model_name']}_{config['augmentation_name']}"
    )
    return config

# --- Model Definition (Copied from simclr.py) ---
class ResNetSimCLR(nn.Module):
    def __init__(self, base_model_name='resnet18', out_dim=128):
        super(ResNetSimCLR, self).__init__()
        resnet_constructor = getattr(torchvision_models, base_model_name)
        # Load backbone ONLY, without projection head for k-NN eval
        self.backbone = resnet_constructor(weights=None, num_classes=10) # Use num_classes=10 just as placeholder

        # Get the feature dimension before the final layer
        self.feature_dim = self.backbone.fc.in_features

        # Remove the final classification layer of the original ResNet
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        return self.backbone(x) # Outputs features before final FC

# --- Feature Extraction ---
@torch.no_grad()
def extract_features(encoder, loader, device):
    encoder.eval()
    all_features = []
    all_labels = []
    logging.info("Extracting features...")
    for images, labels in tqdm(loader, desc="Feature Extraction"):
        images = images.to(device)
        features = encoder(images)
        all_features.append(features.cpu().numpy()) # Convert to numpy for sklearn
        all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_features), np.concatenate(all_labels)

# --- Main k-NN Evaluation Logic ---
def run_knn_evaluation(args):
    # 1. Load Config
    config = load_config(args.config)
    eval_cfg = config['evaluation']['knn']
    data_cfg = config['data']
    output_cfg = config['output']
    run_dir = output_cfg['run_dir']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup basic logging for the script
    log_file = os.path.join(run_dir, "knn_eval.log")
    # Ensure run_dir exists for logging
    os.makedirs(run_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()]) # Use mode 'w' to overwrite log
    logging.info(f"Starting k-NN Evaluation for checkpoint: {args.checkpoint}")
    logging.info(f"Using config: {args.config}")
    logging.info(f"Device: {device}")

    # 2. Load Pretrained Encoder
    encoder = ResNetSimCLR(base_model_name=config['model']['backbone'])
    logging.info(f"Loading checkpoint from: {args.checkpoint}")
    map_location = lambda storage, loc: storage.cuda() if torch.cuda.is_available() else storage.cpu()
    checkpoint = torch.load(args.checkpoint, map_location=map_location)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        encoder_state_dict = {k.replace('module.', '').replace('backbone.', ''): v
                              for k, v in state_dict.items()
                              if k.startswith('module.backbone.') and not k.startswith('module.backbone.fc.')}
    elif isinstance(checkpoint, dict) and not any(k.startswith('backbone.') or k.startswith('module.') for k in checkpoint.keys()):
         state_dict = checkpoint
         encoder_state_dict = {k.replace('backbone.', ''): v
                               for k, v in state_dict.items()
                               if k.startswith('backbone.') and not k.startswith('backbone.fc.')}
    else:
        logging.warning("Checkpoint format unclear, attempting direct load.")
        encoder_state_dict = checkpoint

    msg = encoder.load_state_dict(encoder_state_dict, strict=False)
    logging.info(f"Encoder state_dict loaded. Missing keys: {msg.missing_keys}, Unexpected keys: {msg.unexpected_keys}")
    expected_unexpected = {'backbone.fc.0.weight', 'backbone.fc.0.bias', 'backbone.fc.2.weight', 'backbone.fc.2.bias'}
    unexpected_keys_check = set(msg.unexpected_keys) <= expected_unexpected
    logging.info(f"Unexpected keys check (are they projector layers?): {unexpected_keys_check}")
    encoder = encoder.to(device)

    # 3. Freeze Encoder
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    logging.info("Encoder frozen.")

    # 4. Load STL10 Data
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet normalization
    ])
    train_dataset = STL10(root=data_cfg['data_dir'], split='train', download=False, transform=transform)
    test_dataset = STL10(root=data_cfg['data_dir'], split='test', download=False, transform=transform)

    eval_batch_size = config['evaluation']['linear_probe'].get('batch_size', 256) # Borrow batch size
    train_loader = DataLoader(train_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=data_cfg['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=data_cfg['num_workers'])
    logging.info("STL10 train/test datasets loaded.")

    # 5. Extract Features
    train_features, train_labels = extract_features(encoder, train_loader, device)
    test_features, test_labels = extract_features(encoder, test_loader, device)
    logging.info(f"Train features shape: {train_features.shape}, Test features shape: {test_features.shape}")

    logging.info("Normalizing features (L2 norm) for cosine distance...")
    train_features = normalize(train_features, axis=1)
    test_features = normalize(test_features, axis=1)

    # 6. Perform k-NN Classification for multiple k
    # **** UPDATED: Loop over k values ****
    k_values = [1, 5, 10] # Use default from config as well? eval_cfg.get('k', 10)
    if eval_cfg.get('k', 10) not in k_values: # Ensure the config's k is included if different
        k_values.append(eval_cfg.get('k', 10))
        k_values.sort()
    knn_results = {}
    metric = eval_cfg.get('metric', 'cosine')
    # Ensure metric is compatible with sklearn's implementation
    if metric == 'cosine':
         # Cosine distance = 1 - cosine similarity. Sklearn handles this.
         pass
    elif metric == 'euclidean': # Example if you wanted Euclidean
        pass
    else:
        logging.warning(f"Metric '{metric}' might not be directly supported by sklearn KNeighborsClassifier in the way intended. Using it anyway.")

    logging.info(f"Performing k-NN classification for k={k_values}, metric='{metric}'...")

    for k in k_values:
        logging.info(f"Running k-NN for k={k}...")
        knn_start_time = time.time()
        knn_classifier = KNeighborsClassifier(n_neighbors=k, metric=metric, n_jobs=-1) # Use all available CPU cores
        knn_classifier.fit(train_features, train_labels)
        test_predictions = knn_classifier.predict(test_features)
        knn_duration = time.time() - knn_start_time

        # Compute Top-1 Accuracy for this k
        top1_accuracy = accuracy_score(test_labels, test_predictions) * 100
        knn_results[k] = top1_accuracy
        logging.info(f"k={k} | Test Top-1 Accuracy: {top1_accuracy:.2f}% | Duration: {knn_duration:.2f}s")
    # **** END UPDATE ****

    # 7. Save Accuracy Result
    output_file = os.path.join(run_dir, output_cfg['knn_acc_txt'].format(
        model=config['model_name'], aug=config['augmentation_name']
    ))
    # **** UPDATED: Format output for multiple k ****
    result_lines = []
    for k in sorted(knn_results.keys()):
         result_lines.append(f"k-NN Top-1 Accuracy (k={k}): {knn_results[k]:.2f}%")
    result_string = "\n".join(result_lines) + "\n"
     # **** END UPDATE ****
    with open(output_file, 'w') as f:
        f.write(result_string)
    logging.info(f"k-NN evaluation results saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='k-NN Evaluation Script')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the pretrained model checkpoint (e.g., final_model.pth)')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file used for pretraining')
    args = parser.parse_args()
    try:
        run_knn_evaluation(args)
    except Exception as e:
        logging.error("k-NN evaluation script failed.", exc_info=True)
        raise e
    finally:
        logging.info("k-NN evaluation script finished.")