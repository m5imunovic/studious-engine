from pathlib import Path

import numpy
from PIL import Image

from utils.image_ops import image_to_np_array, rgb_to_grayscale


def test_rgb_to_grayscale(four_faces_path: Path):
    for rgb_image_sample in (four_faces_path / "rgb").glob("*"):

        grayscale_image = rgb_to_grayscale(rgb_image_sample)
        assert grayscale_image.shape == (112, 112)

        grayscale_image_sample = four_faces_path / "grayscale" / rgb_image_sample.name

        expected_grayscale_image = image_to_np_array(grayscale_image_sample)
        expected_grayscale_image = numpy.array(Image.open(grayscale_image_sample))
        assert (expected_grayscale_image == grayscale_image).all()
