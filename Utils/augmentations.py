# Utils/augmentations.py
import torch
import torchvision.transforms as T
import random
from typing import List, Tuple, Dict, Any

class ContrastiveTransform:
    """
    A transform class that applies a specified pipeline of augmentations twice
    to generate two correlated views for contrastive learning.

    Args:
        aug_config (Dict[str, Any]): Configuration dictionary for augmentations.
            Expected keys:
            - random_resized_crop (Dict): {'size': int, 'scale': List[float]}
            - random_horizontal_flip (Dict): {'p': float}
            - color_jitter (Dict): {'enabled': bool, 'brightness': float, 'contrast': float,
                                    'saturation': float, 'hue': float, 'p': float}
            - gaussian_blur (Dict): {'enabled': bool, 'kernel_size': int, 'sigma': List[float], 'p': float}
            - grayscale (Dict): {'enabled': bool, 'p': float}
    """
    def __init__(self, aug_config: Dict[str, Any]):
        self.transform = self._build_transform(aug_config)

    def _build_transform(self, aug_config: Dict[str, Any]) -> T.Compose:
        """Builds the torchvision transform pipeline from the config."""
        pipeline = []

        # 1. Random Resized Crop (Always enabled)
        crop_cfg = aug_config.get('random_resized_crop', {'size': 96, 'scale': [0.2, 1.0]})
        pipeline.append(T.RandomResizedCrop(size=crop_cfg['size'], scale=tuple(crop_cfg['scale'])))

        # 2. Random Horizontal Flip (Always enabled)
        flip_cfg = aug_config.get('random_horizontal_flip', {'p': 0.5})
        pipeline.append(T.RandomHorizontalFlip(p=flip_cfg['p']))

        # 3. Color Jitter (Optional)
        cj_cfg = aug_config.get('color_jitter', {'enabled': False})
        if cj_cfg.get('enabled', False):
            transform = T.ColorJitter(
                brightness=cj_cfg.get('brightness', 0.4),
                contrast=cj_cfg.get('contrast', 0.4),
                saturation=cj_cfg.get('saturation', 0.4),
                hue=cj_cfg.get('hue', 0.1)
            )
            pipeline.append(T.RandomApply([transform], p=cj_cfg.get('p', 0.8)))

        # 4. Grayscale (Optional)
        gray_cfg = aug_config.get('grayscale', {'enabled': False})
        if gray_cfg.get('enabled', False):
            pipeline.append(T.RandomGrayscale(p=gray_cfg.get('p', 0.2)))

        # 5. Gaussian Blur (Optional)
        blur_cfg = aug_config.get('gaussian_blur', {'enabled': False})
        if blur_cfg.get('enabled', False):
            # Ensure kernel size is odd
            kernel_size = blur_cfg.get('kernel_size', 9)
            if kernel_size % 2 == 0:
                kernel_size += 1
            transform = T.GaussianBlur(
                kernel_size=(kernel_size, kernel_size),
                sigma=tuple(blur_cfg.get('sigma', [0.1, 2.0]))
            )
            pipeline.append(T.RandomApply([transform], p=blur_cfg.get('p', 0.5)))

        # 6. Convert to Tensor (Always last for augmentations)
        pipeline.append(T.ToTensor())

        return T.Compose(pipeline)

    def __call__(self, image) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the transform pipeline twice to the input image."""
        view1 = self.transform(image)
        view2 = self.transform(image)
        return view1, view2

# Quick test snippet
if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    # Create a dummy sample image (replace with actual path if needed)
    try:
        # Try loading a real image first
        # Make sure to replace "path/to/sample.jpg" with an actual image path
        # sample_img_path = "path/to/sample.jpg"
        # sample = Image.open(sample_img_path).convert("RGB")
        # print(f"Loaded sample image from: {sample_img_path}")

        # Fallback to creating a dummy image if path doesn't exist
        print("Creating a dummy 128x128 RGB image for testing.")
        sample_array = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        sample = Image.fromarray(sample_array, 'RGB')

    except FileNotFoundError:
        print("Sample image path not found. Creating a dummy 128x128 RGB image.")
        sample_array = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        sample = Image.fromarray(sample_array, 'RGB')
    except Exception as e:
        print(f"Error loading/creating image: {e}. Exiting test.")
        exit()


    print("\n--- Testing Baseline Augmentation (A) ---")
    baseline_cfg = {
        "random_resized_crop": {"size": 96, "scale": [0.2, 1.0]},
        "random_horizontal_flip": {"p": 0.5},
        "color_jitter": {"enabled": False},
        "gaussian_blur": {"enabled": False},
        "grayscale": {"enabled": False}
    }
    transform_baseline = ContrastiveTransform(baseline_cfg)
    v1_base, v2_base = transform_baseline(sample)
    print(f"Input PIL image size: {sample.size}")
    print(f"Output view sizes: {v1_base.shape}, {v2_base.shape}") # Shape for tensors
    print(f"Output view types: {type(v1_base)}, {type(v2_base)}")
    print(f"Output tensor ranges: [{v1_base.min():.2f}, {v1_base.max():.2f}], [{v2_base.min():.2f}, {v2_base.max():.2f}]")


    print("\n--- Testing Strong Augmentation (E: B+C+D) ---")
    strong_cfg = {
        "random_resized_crop": {"size": 96, "scale": [0.2, 1.0]},
        "random_horizontal_flip": {"p": 0.5},
        "color_jitter": {"enabled": True, "brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1, "p": 0.8},
        "gaussian_blur": {"enabled": True, "kernel_size": 9, "sigma": [0.1, 2.0], "p": 0.5},
        "grayscale": {"enabled": True, "p": 0.2}
    }
    transform_strong = ContrastiveTransform(strong_cfg)
    v1_strong, v2_strong = transform_strong(sample)
    print(f"Input PIL image size: {sample.size}")
    print(f"Output view sizes: {v1_strong.shape}, {v2_strong.shape}") # Shape for tensors
    print(f"Output view types: {type(v1_strong)}, {type(v2_strong)}")
    print(f"Output tensor ranges: [{v1_strong.min():.2f}, {v1_strong.max():.2f}], [{v2_strong.min():.2f}, {v2_strong.max():.2f}]")