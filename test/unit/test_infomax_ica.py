import numpy as np

from infomax import infomax_ica
from utils.image_ops import mix_images


def test_infomax_ica():
    img1 = np.array([[4, 2], [6, 8]])
    img2 = np.array([[2, 4], [8, 16]])

    mixing_matrix = np.array([[0.25, 0.75], [0.75, 0.25]])
    mixed_images = mix_images([img1, img2], mixing_matrix=mixing_matrix)
    assert mixed_images.shape == (2, 4)

    M = infomax_ica(mixed_images, algo="sup")
    assert M.shape == (2, 2)
