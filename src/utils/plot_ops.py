from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np


def plot_dependency(images: list[np.ndarray], normalize: bool = True):
    """Plot dependency of pairwise combinations of images.

    Args:
        images (list[np.ndarray]): List of images to mix
        normalize (bool, optional): Normalize images to range [0, 1]
    """

    all_comb = list(combinations(range(len(images)), 2))

    assert (all(image.dtype == np.uint8) for image in images)

    if normalize:
        images = [(image / 255) for image in images]

    _, ax = plt.subplots(nrows=1, ncols=len(all_comb), figsize=[20, 3], squeeze=False)
    # _, ax = plt.subplots(1, len(all_comb), figsize=[18, 10])

    print(type(ax))

    for idx in range(len(all_comb)):
        iidx1, iidx2 = all_comb[idx]
        ax[0][idx].scatter(images[iidx1], images[iidx2])
        ax[0][idx].tick_params(labelsize=12)
        ax[0][idx].set_title(f"Sources {iidx1} - {iidx2}", fontsize=14)
        ax[0][idx].set_xlim([-0.1, 1.1])
        ax[0][idx].set_ylim([-0.1, 1.1])
        # ax[0][idx].set_xticks([])
        # ax[0][idx].set_yticks([])

    plt.show()
