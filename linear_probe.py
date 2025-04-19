# linear_probe.py (Updated)
# Trains and evaluates a linear classifier on features from a pretrained SimCLR model.
# Outputs Top-1 and Top-5 accuracy.

import argparse
import os
import yaml
import logging
import time
import pandas as pd
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torchvision.datasets import STL10
from torch.utils.data import DataLoader
from tqdm import tqdm

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

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
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

# --- Model Definition (Copied from simclr.py) ---
class ResNetSimCLR(nn.Module):
    def __init__(self, base_model_name='resnet18', out_dim=128):
        super(ResNetSimCLR, self).__init__()
        resnet_constructor = getattr(torchvision_models, base_model_name)
        # Load backbone ONLY, without projection head for linear probing
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
        all_features.append(features.cpu())
        all_labels.append(labels.cpu())
    return torch.cat(all_features), torch.cat(all_labels)

# --- Main Linear Probing Logic ---
def run_linear_probe(args):
    # 1. Load Config
    config = load_config(args.config)
    eval_cfg = config['evaluation']['linear_probe']
    data_cfg = config['data']
    output_cfg = config['output']
    run_dir = output_cfg['run_dir']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup basic logging for the script
    log_file = os.path.join(run_dir, "linear_probe.log")
    # Ensure run_dir exists for logging
    os.makedirs(run_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()]) # Use mode 'w' to overwrite log
    logging.info(f"Starting Linear Probe Evaluation for checkpoint: {args.checkpoint}")
    logging.info(f"Using config: {args.config}")
    logging.info(f"Device: {device}")

    # 2. Load Pretrained Encoder
    encoder = ResNetSimCLR(base_model_name=config['model']['backbone'])
    logging.info(f"Loading checkpoint from: {args.checkpoint}")
    map_location = lambda storage, loc: storage.cuda() if torch.cuda.is_available() else storage.cpu()
    # Use weights_only=True if you trust the source or saved it yourself.
    # For now, keeping default False but acknowledging the warning.
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
    # Ensure projector keys are among the unexpected keys if present in the checkpoint
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

    eval_batch_size = eval_cfg.get('batch_size', 256)
    train_loader = DataLoader(train_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=data_cfg['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=data_cfg['num_workers'])
    logging.info("STL10 train/test datasets loaded.")

    # 5. Extract Features & Train Linear Classifier
    train_features, train_labels = extract_features(encoder, train_loader, device)
    test_features, test_labels = extract_features(encoder, test_loader, device)
    logging.info(f"Train features shape: {train_features.shape}, Test features shape: {test_features.shape}")

    classifier = nn.Linear(encoder.feature_dim, 10).to(device) # 10 classes in STL10
    optimizer = optim.SGD(
        classifier.parameters(),
        lr=float(eval_cfg['learning_rate']),
        momentum=float(eval_cfg['momentum']),
        weight_decay=0
    )
    criterion = nn.CrossEntropyLoss()

    logging.info("Training the linear classifier...")
    classifier.train()
    train_start_time = time.time()
    for epoch in range(eval_cfg['epochs']):
        optimizer.zero_grad()
        outputs = classifier(train_features.to(device))
        loss = criterion(outputs, train_labels.to(device))
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == eval_cfg['epochs']: # Log first, last, and every 10 epochs
             with torch.no_grad():
                 acc1, acc5 = accuracy(outputs, train_labels.to(device), topk=(1, 5))
                 logging.info(f"Epoch [{epoch+1}/{eval_cfg['epochs']}] Loss: {loss.item():.4f}, Train Acc@1: {acc1.item():.2f}%, Train Acc@5: {acc5.item():.2f}%")

    train_duration = time.time() - train_start_time
    logging.info(f"Linear classifier training finished in {train_duration:.2f}s.")

    # 6. Evaluate on Test Set
    logging.info("Evaluating on the test set...")
    classifier.eval()
    with torch.no_grad():
        test_outputs = classifier(test_features.to(device))
        test_loss = criterion(test_outputs, test_labels.to(device))
        # **** UPDATED: Get Top-1 and Top-5 ****
        top1_acc, top5_acc = accuracy(test_outputs, test_labels.to(device), topk=(1, 5))
        top1_accuracy = top1_acc.item()
        top5_accuracy = top5_acc.item()
        # **** END UPDATE ****

    logging.info(f"Test Loss: {test_loss.item():.4f}")
    logging.info(f"Test Top-1 Accuracy: {top1_accuracy:.2f}%")
    logging.info(f"Test Top-5 Accuracy: {top5_accuracy:.2f}%") # Log Top-5

    # 7. Save Accuracy Result
    output_file = os.path.join(run_dir, output_cfg['linear_acc_txt'].format(
        model=config['model_name'], aug=config['augmentation_name']
    ))
    # **** UPDATED: Write both Top-1 and Top-5 ****
    result_string = f"Top-1 Accuracy: {top1_accuracy:.2f}%\nTop-5 Accuracy: {top5_accuracy:.2f}%\n"
    # **** END UPDATE ****
    with open(output_file, 'w') as f:
        f.write(result_string)
    logging.info(f"Linear probe result saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Linear Probe Evaluation Script')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the pretrained model checkpoint (e.g., final_model.pth)')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file used for pretraining')
    args = parser.parse_args()
    try:
        run_linear_probe(args)
    except Exception as e:
        logging.error("Linear probe script failed.", exc_info=True)
        raise e
    finally:
         logging.info("Linear probe script finished.")