from pathlib import Path

import numpy as np
from PIL import Image

from utils.image_ops import image_to_np_array, mix_images, rgb_to_grayscale


def test_rgb_to_grayscale(four_faces_path: Path):
    for rgb_image_sample in (four_faces_path / "rgb").glob("*"):

        grayscale_image = rgb_to_grayscale(rgb_image_sample)
        assert grayscale_image.shape == (112, 112)

        grayscale_image_sample = four_faces_path / "grayscale" / rgb_image_sample.name

        expected_grayscale_image = image_to_np_array(grayscale_image_sample)
        expected_grayscale_image = np.array(Image.open(grayscale_image_sample))
        assert (expected_grayscale_image == grayscale_image).all()


def test_mix_images():
    img1 = np.array([[4, 2], [6, 8]])
    img2 = np.array([[2, 4], [8, 16]])

    mixing_matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
    mixed_images = mix_images([img1, img2], mixing_matrix=mixing_matrix)
    assert mixed_images.shape == (2, 4)

    mixed_img1 = mixed_images[0, :]
    expected_mixed_img1 = np.array([3, 3, 7, 12])
    assert (mixed_img1 == expected_mixed_img1).all()
    mixed_img2 = mixed_images[1, :]
    expected_mixed_img2 = np.array([3, 3, 7, 12])
    assert (mixed_img2 == expected_mixed_img2).all()

    mixing_matrix = np.array([[0.25, 0.75], [0.75, 0.25]])
    mixed_images = mix_images([img1, img2], mixing_matrix=mixing_matrix)

    mixed_img1 = mixed_images[0, :]
    expected_mixed_img1 = np.array([2.5, 3.5, 7.5, 14])
    assert (mixed_img1 == expected_mixed_img1).all()

    mixed_img2 = mixed_images[1, :]
    expected_mixed_img2 = np.array([3.5, 2.5, 6.5, 10])
    assert (mixed_img1 == expected_mixed_img1).all()
