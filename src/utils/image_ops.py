from pathlib import Path

import numpy as np
from PIL import Image


def rgb_to_grayscale(input_path: Path) -> np.ndarray:
    img = Image.open(input_path, "r")

    # convert to grayscale
    grayscale_img = img.convert("L")
    grayscale_np = image_to_np_array(grayscale_img)
    grayscale_img.save(input_path.parent.parent / "grayscale" / input_path.name)

    return grayscale_np


def image_to_np_array(input_img: Path | Image.Image) -> np.ndarray:
    if isinstance(input_img, Path):
        input_img = Image.open(input_img, "r")

    img_np = np.array(input_img)
    return img_np
