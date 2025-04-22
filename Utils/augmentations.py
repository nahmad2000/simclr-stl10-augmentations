# Utils/augmentations.py
# UPDATED to read parameters from base config and apply based on 'enabled' flags

import torch
import torchvision.transforms as T
import random
from typing import List, Tuple, Dict, Any

class ContrastiveTransform:
    """
    A transform class that applies a specified pipeline of augmentations twice
    to generate two correlated views for contrastive learning.

    Reads detailed parameters from the provided aug_config (expected to be merged
    from base_simclr.yaml and a specific augmentation config like simclr_all.yaml),
    but applies augmentations based on the 'enabled' flag within each section.
    """
    def __init__(self, aug_config: Dict[str, Any]):
        # It's assumed aug_config is the result of loading base_simclr.yaml
        # and merging the specific config (e.g., simclr_all.yaml) which
        # mainly overrides the 'enabled' flags.
        self.transform = self._build_transform(aug_config)
        self.base_parameters = aug_config # Store base params if needed for debugging

    def _build_transform(self, aug_config: Dict[str, Any]) -> T.Compose:
        """Builds the torchvision transform pipeline from the config."""
        pipeline = []

        # --- Get augmentation sub-configs ---
        # Use .get() defensively, although base_simclr.yaml should define them
        crop_cfg = aug_config.get('random_resized_crop', {})
        flip_cfg = aug_config.get('random_horizontal_flip', {})
        cj_cfg = aug_config.get('color_jitter', {})
        gray_cfg = aug_config.get('grayscale', {})
        blur_cfg = aug_config.get('gaussian_blur', {})

        # 1. Random Resized Crop (Always enabled implicitly by SimCLR standard)
        pipeline.append(T.RandomResizedCrop(
            size=crop_cfg.get('size', 96),
            scale=tuple(crop_cfg.get('scale', [0.2, 1.0]))
        ))

        # 2. Random Horizontal Flip (Always enabled implicitly by SimCLR standard)
        pipeline.append(T.RandomHorizontalFlip(p=flip_cfg.get('p', 0.5)))

        # 3. Color Jitter (Apply based on 'enabled' flag)
        if cj_cfg.get('enabled', False):
            # Use parameters defined in the config (now matching reference)
            transform = T.ColorJitter(
                brightness=cj_cfg.get('brightness', 0.8), # Defaulting to reference values
                contrast=cj_cfg.get('contrast', 0.8),
                saturation=cj_cfg.get('saturation', 0.8),
                hue=cj_cfg.get('hue', 0.2)
            )
            # Use probability 'p' also from config
            pipeline.append(T.RandomApply([transform], p=cj_cfg.get('p', 0.8)))

        # 4. Grayscale (Apply based on 'enabled' flag)
        if gray_cfg.get('enabled', False):
             # Use probability 'p' from config
            pipeline.append(T.RandomGrayscale(p=gray_cfg.get('p', 0.2)))

        # 5. Gaussian Blur (Apply based on 'enabled' flag, p=1.0)
        if blur_cfg.get('enabled', False):
            # Ensure kernel size is odd
            kernel_size = blur_cfg.get('kernel_size', 9)
            if kernel_size % 2 == 0:
                kernel_size += 1
            # Use parameters defined in the config
            transform = T.GaussianBlur(
                kernel_size=(kernel_size, kernel_size),
                sigma=tuple(blur_cfg.get('sigma', [0.1, 2.0]))
            )
            # Apply directly (implicitly p=1.0, matching reference logic)
            pipeline.append(transform)

        # 6. Convert to Tensor (Always last for augmentations)
        pipeline.append(T.ToTensor())

        return T.Compose(pipeline)

    def __call__(self, image) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the transform pipeline twice to the input image."""
        view1 = self.transform(image)
        view2 = self.transform(image)
        return view1, view2

# Quick test snippet (remains the same)
if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    # Create a dummy sample image (replace with actual path if needed)
    try:
        print("Creating a dummy 128x128 RGB image for testing.")
        sample_array = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        sample = Image.fromarray(sample_array, 'RGB')
    except Exception as e:
        print(f"Error creating image: {e}. Exiting test.")
        exit()

    print("\n--- Testing 'All' Augmentation (should use reference params from base) ---")
    # Simulate loading base config and enabling all in 'simclr_all.yaml'
    # (Manually constructing the merged dict here for test purposes)
    effective_all_cfg = {
        "random_resized_crop": {"size": 96, "scale": [0.2, 1.0]},
        "random_horizontal_flip": {"p": 0.5},
        "color_jitter": {"enabled": True, "brightness": 0.8, "contrast": 0.8, "saturation": 0.8, "hue": 0.2, "p": 0.8},
        "gaussian_blur": {"enabled": True, "kernel_size": 9, "sigma": [0.1, 2.0]}, # No 'p' here
        "grayscale": {"enabled": True, "p": 0.2}
    }
    transform_all = ContrastiveTransform(effective_all_cfg)
    v1_all, v2_all = transform_all(sample)
    print(f"Input PIL image size: {sample.size}")
    print(f"Output view sizes: {v1_all.shape}, {v2_all.shape}") # Shape for tensors
    print(f"Output view types: {type(v1_all)}, {type(v2_all)}")
    print(f"Output tensor ranges: [{v1_all.min():.2f}, {v1_all.max():.2f}], [{v2_all.min():.2f}, {v2_all.max():.2f}]")

    print("\n--- Testing 'Baseline' Augmentation (only spatial) ---")
    effective_baseline_cfg = {
        "random_resized_crop": {"size": 96, "scale": [0.2, 1.0]},
        "random_horizontal_flip": {"p": 0.5},
        "color_jitter": {"enabled": False, "brightness": 0.8, "contrast": 0.8, "saturation": 0.8, "hue": 0.2, "p": 0.8}, # Params present but disabled
        "gaussian_blur": {"enabled": False, "kernel_size": 9, "sigma": [0.1, 2.0]}, # Disabled
        "grayscale": {"enabled": False, "p": 0.2} # Disabled
    }
    transform_baseline = ContrastiveTransform(effective_baseline_cfg)
    v1_base, v2_base = transform_baseline(sample)
    print(f"Input PIL image size: {sample.size}")
    print(f"Output view sizes: {v1_base.shape}, {v2_base.shape}") # Shape for tensors
    print(f"Output view types: {type(v1_base)}, {type(v2_base)}")
    print(f"Output tensor ranges: [{v1_base.min():.2f}, {v1_base.max():.2f}], [{v2_base.min():.2f}, {v2_base.max():.2f}]")