"""Paths and image preprocessing shared by the scripts."""

from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
DATA = ROOT / "data"
SAMPLES = ROOT / "samples"
RESULTS = ROOT / "results"

# ImageNet normalisation used when MobileNetV2 was trained.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_rgb(image: np.ndarray) -> Image.Image:
    """skimage images can be grayscale, RGB or RGBA; MobileNetV2 wants RGB."""
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        image = (255 * (image - image.min()) / max(float(np.ptp(image)), 1e-6)).astype(np.uint8)
    return Image.fromarray(image)


def to_tensor(image: Image.Image) -> np.ndarray:
    """224x224 RGB image -> normalised float32 [3, 224, 224] (channels first)."""
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - MEAN) / STD
    return array.transpose(2, 0, 1).astype(np.float32)


def center_crop(image: Image.Image, size: int = 224) -> Image.Image:
    """Resize the short side to 256, then take the central 224x224, as in ImageNet evaluation."""
    w, h = image.size
    scale = 256 / min(w, h)
    image = image.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    w, h = image.size
    left, top = (w - size) // 2, (h - size) // 2
    return image.crop((left, top, left + size, top + size))


def random_crop(image: Image.Image, rng: np.random.Generator, size: int = 224) -> Image.Image:
    """A random square covering 40-100% of the short side, resized to 224, maybe flipped."""
    w, h = image.size
    side = int(min(w, h) * rng.uniform(0.4, 1.0))
    left = int(rng.integers(0, w - side + 1))
    top = int(rng.integers(0, h - side + 1))
    crop = image.crop((left, top, left + side, top + side)).resize((size, size), Image.BILINEAR)
    return crop.transpose(Image.FLIP_LEFT_RIGHT) if rng.random() < 0.5 else crop
