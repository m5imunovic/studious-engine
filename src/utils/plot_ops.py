from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np


def plot_dependency(signals: list[np.ndarray], titles: list[str] = None, normalize: bool = True):
    """Plot dependency of pairwise combinations of images.

    Args:
        signals (list[np.ndarray]): List of images to mix
        titles (list[str]): list of titles for plots
        normalize (bool, optional): Normalize images to range [0, 1]
    """

    for sig_idx, images in enumerate(signals):

        all_comb = list(combinations(range(len(images)), 2))

        def norm(image: np.ndarray) -> np.ndarray:
            min_val = image.min()
            max_val = image.max()
            return (image - min_val) / (max_val - min_val)

        if normalize:
            images = [norm(image) for image in images]

        _, ax = plt.subplots(nrows=1, ncols=len(all_comb), figsize=[4, 3], squeeze=False)

        for idx in range(len(all_comb)):
            iidx1, iidx2 = all_comb[idx]
            ax[0][idx].scatter(images[iidx1], images[iidx2])
            ax[0][idx].tick_params(labelsize=12)
            if titles is None:
                ax[0][idx].set_title(f"Sources {iidx1} - {iidx2}", fontsize=14)
            else:
                ax[0][idx].set_title(titles[sig_idx], fontsize=14)

            # ax[0][idx].set_xlim([-0.1, 1.1])
            # ax[0][idx].set_ylim([-0.1, 1.1])
            ax[0][idx].set_xticks([])
            ax[0][idx].set_yticks([])

    plt.show()
