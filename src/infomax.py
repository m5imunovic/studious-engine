import numpy as np

from utils.stat_ops import center, whiten_evd, whiten_svd


def infomax_ica(
    x: np.ndarray,
    steps: int = 5000,
    tolerance: float = 1e-3,
    learning_rate: float = 0.0001,
    whiten: str = "svd",
    algo: str = "sub",
) -> np.ndarray:
    # l_rate = 0.01 / math.log(n_features**2.0)

    num_components, _ = x.shape
    x, _ = center(x)

    if whiten == "svd":
        x = whiten_svd(x)
    elif whiten == "evd":
        x = whiten_evd(x)
    else:
        raise Exception(f"Unsupported whitening operation {whiten}. Excepted 'svd' or 'evd'")

    W = np.random.rand(num_components, num_components)
    I = np.identity(n=num_components)  # noqa: E741

    def f(x: np.ndarray, algo: str) -> np.ndarray:
        if algo == "sup":
            return np.tanh(x)
        elif algo == "sub":
            return x - np.tanh(x)
        else:
            return 2.0 / (1 - np.exp(-x))

    for _ in range(steps):
        update = (I - np.dot(f(x, algo=algo), x.T)) * W
        Wnew = W + learning_rate * update
        if np.abs(Wnew - W).sum() < tolerance:
            break
        W = Wnew

    return W
