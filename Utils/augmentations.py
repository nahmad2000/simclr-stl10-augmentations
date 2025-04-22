# Utils/augmentations.py
# UPDATED to read parameters from base config and apply based on 'enabled' flags,
# including RandomRotation, RandomErasing, and RandomSolarize.

import torch
import torchvision.transforms as T
import random
from typing import List, Tuple, Dict, Any
from PIL import Image # Import Image explicitly if needed for type hints or checks

class ContrastiveTransform:
    """
    A transform class that applies a specified pipeline of augmentations twice
    to generate two correlated views for contrastive learning.

    Reads detailed parameters from the provided aug_config (expected to be merged
    from base_simclr.yaml and a specific augmentation config),
    but applies augmentations based on the 'enabled' flag within each section.
    """
    def __init__(self, aug_config: Dict[str, Any]):
        # It's assumed aug_config is the result of loading base_simclr.yaml
        # and merging the specific config (e.g., simclr_rotation.yaml) which
        # mainly overrides the 'enabled' flags.
        self.transform = self._build_transform(aug_config)
        self.base_parameters = aug_config # Store base params if needed for debugging
        print(f"Augmentation pipeline built with config keys: {list(aug_config.keys())}")
        print("Enabled augmentations:")
        for k, v in aug_config.items():
            if isinstance(v, dict) and v.get('enabled'):
                 print(f"  - {k}")


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
        # --- NEW ---
        rot_cfg = aug_config.get('random_rotation', {})
        erase_cfg = aug_config.get('random_erasing', {})
        solar_cfg = aug_config.get('random_solarize', {})

        # --- Build Pipeline ---

        # 1. Random Resized Crop (Always enabled implicitly by SimCLR standard)
        pipeline.append(T.RandomResizedCrop(
            size=crop_cfg.get('size', 96),
            scale=tuple(crop_cfg.get('scale', [0.2, 1.0]))
        ))

        # 2. Random Horizontal Flip (Always enabled implicitly by SimCLR standard)
        pipeline.append(T.RandomHorizontalFlip(p=flip_cfg.get('p', 0.5)))

        # --- Geometric Augmentations (Example Placement: After Flip, Before Color) ---
        # NEW: Random Rotation
        if rot_cfg.get('enabled', False):
            # Apply directly or using RandomApply based on your preference / config structure
            # Using RandomApply here as it's common for augmentations beyond the core spatial ones
            transform = T.RandomRotation(degrees=rot_cfg.get('degrees', 15)) # Get degrees from config
            pipeline.append(T.RandomApply([transform], p=rot_cfg.get('p', 0.5))) # Use probability 'p' from config

        # --- Color Augmentations ---
        # 3. Color Jitter (Apply based on 'enabled' flag)
        if cj_cfg.get('enabled', False):
            transform = T.ColorJitter(
                brightness=cj_cfg.get('brightness', 0.8),
                contrast=cj_cfg.get('contrast', 0.8),
                saturation=cj_cfg.get('saturation', 0.8),
                hue=cj_cfg.get('hue', 0.2)
            )
            pipeline.append(T.RandomApply([transform], p=cj_cfg.get('p', 0.8))) # Use probability 'p' from config

        # 4. Grayscale (Apply based on 'enabled' flag)
        if gray_cfg.get('enabled', False):
            pipeline.append(T.RandomGrayscale(p=gray_cfg.get('p', 0.2))) # Use probability 'p' from config

        # --- Other Appearance Augmentations ---
        # NEW: Random Solarize (Example Placement: Before Blur)
        if solar_cfg.get('enabled', False):
             transform = T.RandomSolarize(threshold=solar_cfg.get('threshold', 128)) # Get threshold from config
             pipeline.append(T.RandomApply([transform], p=solar_cfg.get('p', 0.2))) # Use probability 'p' from config

        # 5. Gaussian Blur (Apply based on 'enabled' flag, p=1.0)
        if blur_cfg.get('enabled', False):
            kernel_size = blur_cfg.get('kernel_size', 9)
            if kernel_size % 2 == 0: kernel_size += 1 # Ensure kernel size is odd
            transform = T.GaussianBlur(
                kernel_size=(kernel_size, kernel_size),
                sigma=tuple(blur_cfg.get('sigma', [0.1, 2.0]))
            )
            # SimCLR reference applies blur with p=1.0 if enabled for one view, 0.1 for other?
            # Sticking to simpler p=1.0 application if enabled, consistent with original version here.
            # If p<1 is desired, wrap in T.RandomApply like others.
            pipeline.append(transform) # Apply directly if enabled

        # --- Convert to Tensor (Needs to happen BEFORE Random Erasing) ---
        # Random Erasing works on Tensors
        pipeline.append(T.ToTensor())

        # --- Occlusion Augmentation (Applied last, on Tensor) ---
        # NEW: Random Erasing
        if erase_cfg.get('enabled', False):
            pipeline.append(T.RandomErasing(
                p=erase_cfg.get('p', 0.5), # Probability of applying erasing
                scale=tuple(erase_cfg.get('scale', [0.02, 0.33])), # Proportion of image to erase
                ratio=tuple(erase_cfg.get('ratio', [0.3, 3.3])), # Aspect ratio of erased area
                value=erase_cfg.get('value', 0), # Value to fill ('random' or number)
                inplace=False # Default is False, keep it that way
            ))

        # --- Normalization (Optional, usually done outside contrastive transform in dataloader) ---
        # If you need normalization as part of this specific transform pipeline:
        # norm_cfg = aug_config.get('normalize', {'enabled': False})
        # if norm_cfg.get('enabled', False):
        #    pipeline.append(T.Normalize(mean=norm_cfg.get('mean', [0.485, 0.456, 0.406]),
        #                                std=norm_cfg.get('std', [0.229, 0.224, 0.225])))

        return T.Compose(pipeline)

    def __call__(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the transform pipeline twice to the input image."""
        try:
            view1 = self.transform(image)
            view2 = self.transform(image)
            return view1, view2
        except Exception as e:
            print(f"Error during transform application: {e}")
            # Optionally re-raise or return None/dummy data
            raise e


# Quick test snippet (Updated to test a new transform)
if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    # Create a dummy sample image
    try:
        print("Creating a dummy 128x128 RGB image for testing.")
        sample_array = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        sample = Image.fromarray(sample_array, 'RGB')
    except Exception as e:
        print(f"Error creating image: {e}. Exiting test.")
        exit()

    print("\n--- Testing 'Rotation' Augmentation (Example) ---")
    # Simulate loading base config and enabling only rotation
    effective_rotation_cfg = {
        "random_resized_crop": {"size": 96, "scale": [0.2, 1.0]},
        "random_horizontal_flip": {"p": 0.5},
        "random_rotation": {"enabled": True, "degrees": 30, "p": 1.0}, # Enable rotation
        "color_jitter": {"enabled": False, "brightness": 0.8, "contrast": 0.8, "saturation": 0.8, "hue": 0.2, "p": 0.8},
        "gaussian_blur": {"enabled": False, "kernel_size": 9, "sigma": [0.1, 2.0]},
        "grayscale": {"enabled": False, "p": 0.2},
        "random_erasing": {"enabled": False, "p": 0.5},
        "random_solarize": {"enabled": False, "threshold": 128, "p": 0.2}
    }
    transform_rot = ContrastiveTransform(effective_rotation_cfg)
    v1_rot, v2_rot = transform_rot(sample)
    print(f"Input PIL image size: {sample.size}")
    print(f"Output view sizes: {v1_rot.shape}, {v2_rot.shape}") # Shape for tensors
    print(f"Output view types: {type(v1_rot)}, {type(v2_rot)}")
    print(f"Output tensor ranges: [{v1_rot.min():.2f}, {v1_rot.max():.2f}], [{v2_rot.min():.2f}, {v2_rot.max():.2f}]")

    print("\n--- Testing 'All Extended' Augmentation (Example) ---")
    effective_all_ext_cfg = {
        "random_resized_crop": {"size": 96, "scale": [0.2, 1.0]},
        "random_horizontal_flip": {"p": 0.5},
        "random_rotation": {"enabled": True, "degrees": 15, "p": 0.5},
        "color_jitter": {"enabled": True, "brightness": 0.8, "contrast": 0.8, "saturation": 0.8, "hue": 0.2, "p": 0.8},
        "gaussian_blur": {"enabled": True, "kernel_size": 9, "sigma": [0.1, 2.0]}, # Applied directly
        "grayscale": {"enabled": True, "p": 0.2},
        "random_solarize": {"enabled": True, "threshold": 128, "p": 0.2},
        "random_erasing": {"enabled": True, "p": 0.5} # Applied after ToTensor
    }
    transform_all_ext = ContrastiveTransform(effective_all_ext_cfg)
    v1_all, v2_all = transform_all_ext(sample)
    print(f"Input PIL image size: {sample.size}")
    print(f"Output view sizes: {v1_all.shape}, {v2_all.shape}")
    print(f"Output view types: {type(v1_all)}, {type(v2_all)}")
    print(f"Output tensor ranges: [{v1_all.min():.2f}, {v1_all.max():.2f}], [{v2_all.min():.2f}, {v2_all.max():.2f}]")