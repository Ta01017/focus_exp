from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
from PIL import Image


def resolve_path(value: str, base_dir: Path) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    p = Path(text).expanduser()
    if not p.is_absolute():
        p = base_dir / p
    return p.resolve()


def load_rgb(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def ensure_same_shape(*images: np.ndarray) -> None:
    shapes = [image.shape for image in images if image is not None]
    if len(set(shapes)) > 1:
        raise ValueError(f"Image shapes differ: {shapes}")


def image_extensions() -> Tuple[str, ...]:
    return (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def list_images(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in image_extensions():
            yield path
