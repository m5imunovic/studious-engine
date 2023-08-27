from pathlib import Path

import numpy as np
from PIL import Image


def rgb_to_grayscale(input_path: Path) -> np.ndarray:
    """Opens image and converts it to grayscale valued numpy array.

    Args:
        input_path (Path): path to the image

    Returns:
        np.ndarray: 2-D numpy array
    """
    img = Image.open(input_path, "r")

    # convert to grayscale
    grayscale_img = img.convert("L")
    grayscale_np = image_to_np_array(grayscale_img)
    grayscale_img.save(input_path.parent.parent / "grayscale" / input_path.name)

    return grayscale_np


def image_to_np_array(input_img: Path | Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(input_img, np.ndarray):
        return input_img

    if isinstance(input_img, Path):
        input_img = Image.open(input_img, "r")

    img_np = np.array(input_img)

    return img_np


def normalize_image(x: np.ndarray) -> np.ndarray:
    x = np.float32(x)
    x = x - x.min()
    x /= x.max()
    return x


def mix_images(input_images: list[Path | Image.Image | np.ndarray], mixing_matrix: np.ndarray) -> list:
    """Mix the images using the random mixing matrix.

    Args:
        input_images (list[Path  |  Image.Image  |  np.ndarray]): list of source images to mix
        mixing_matrix (np.ndarray): weights of the mixture

    Returns:
        list: mixed targets, same number of targets as input sources
    """

    matrix_rows, matrix_cols = mixing_matrix.shape
    assert matrix_cols == matrix_rows, "We expect the same number of sources and targets"

    assert len(input_images) == matrix_cols, "Number of images does not match mixing matrix dimensions"

    images = [image_to_np_array(image_path) for image_path in input_images]

    # Flatten the images into 1D arrays
    flat_images = [image.flatten() for image in images]
    # Combine the flattened images into a single array
    combined_images = np.vstack(flat_images)
    mixed_images = np.dot(mixing_matrix, combined_images)

    return mixed_images
