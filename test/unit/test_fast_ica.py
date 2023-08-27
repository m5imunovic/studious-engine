import numpy as np
from sklearn.decomposition import FastICA

from fast_ica import fast_ica_defl, fastIca
from utils.image_ops import mix_images


def test_fast_ica_defl():
    img1 = np.array([[4, 2], [6, 8]])
    img2 = np.array([[2, 4], [8, 16]])

    mixing_matrix = np.array([[0.25, 0.75], [0.75, 0.25]])
    mixed_images = mix_images([img1, img2], mixing_matrix=mixing_matrix)
    assert mixed_images.shape == (2, 4)

    M = fast_ica_defl(mixed_images, axis=1)
    assert M.shape == (2, 2)

    fastIca(mixed_images)
    transformer = FastICA(n_components=2, random_state=0, whiten="unit-variance")
    X_transformed = transformer.fit_transform(mixed_images.T)
    print(X_transformed.shape)
