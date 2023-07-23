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


def image_to_np_array(input_img: Path | Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(input_img, np.ndarray):
        return input_img

    if isinstance(input_img, Path):
        input_img = Image.open(input_img, "r")

    img_np = np.array(input_img)
    return img_np


def mix_images(input_images: list[Path | Image.Image | np.ndarray], mixing_matrix: np.ndarray) -> list:

    matrix_rows, matrix_cols = mixing_matrix.shape
    assert matrix_cols == matrix_rows, "We expect the same number of sources and targets"

    assert len(input_images) == matrix_cols, "Number of images does not match mixing matrix dimensions"

    images = [image_to_np_array(image_path) for image_path in input_images]

    # Flatten the images into 1D arrays
    flat_images = [image.flatten() for image in images]

    # Combine the flattened images into a single array
    combined_images = np.vstack(flat_images)

    # Mix the images using the random mixing matrix
    mixed_images = np.dot(mixing_matrix, combined_images)

    # Reshape the mixed images back to their original shape
    mixed_images_list = [mixed_images[i].reshape(images[i].shape) for i in range(len(images))]

    return mixed_images_list
