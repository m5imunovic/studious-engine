import numpy as np

from utils.stat_ops import center, whiten_evd, whiten_svd


def _gs_decorrelation(w, W, j):
    """Orthonormalize w wrt the first j rows of W.

    Parameters
    ----------
    w : ndarray of shape (n,)
        Array to be orthogonalized

    W : ndarray of shape (p, n)
        Null space definition

    j : int < p
        The no of (from the first) rows of Null space W wrt which w is
        orthogonalized.

    Notes
    -----
    Assumes that W is orthogonal
    w changed in place
    """
    w -= np.linalg.multi_dot([w, W[:j].T, W[:j]])
    return w


def _logcosh(x):
    alpha = 1.0  # fun_args.get("alpha", 1.0)  # comment it out?

    x *= alpha
    gx = np.tanh(x, x)  # apply the tanh inplace
    g_x = np.empty(x.shape[0], dtype=x.dtype)
    # XXX compute in chunks to avoid extra allocation
    for i, gx_i in enumerate(gx):  # please don't vectorize.
        g_x[i] = (alpha * (1 - gx_i**2)).mean()
    return gx, g_x


def fast_ica_defl(x: np.ndarray, axis: int = 1, whiten: str = "svd", threshold: float = 1e-04, steps: int = 5000):
    x_centered, _ = center(x, axis=axis)

    if whiten == "svd":
        z = whiten_svd(x_centered)
    elif whiten == "evd":
        z = whiten_evd(x_centered)
    else:
        raise Exception("Unsupported whitening option")

    m, _ = x_centered.shape

    # Initialize random weights of mixing matrix
    W = np.zeros(shape=(m, m))

    for p in range(m):
        w_p = np.random.rand(m, 1)
        # normalize to unit vector
        w_p = w_p / (np.linalg.norm(w_p) + 1e-8)

        limit = np.inf
        for step in range(steps):
            if step == 9500:
                print(limit)
            if limit < threshold:
                break
            g = np.tanh(np.dot(w_p.T, z))
            g_prim = 1 - np.square(np.tanh(np.dot(w_p.T, z)))
            w_p = (z * g).mean(axis=1) - g_prim.mean() * w_p.squeeze()

            w_orthog = np.zeros_like(w_p)
            for j in range(p):
                w_orthog = (w_p.T * W[j, :]) * w_p
            w_p = w_p - w_orthog

            # normalize to unit vector
            w_p = w_p / (np.linalg.norm(w_p) + 1e-8)

            limit = np.abs(w_p * W[p, :]).sum() - 1
            limit = np.abs(limit)
            W[p, :] = w_p.copy()

        print(f"Returns from optimization loop after {step} steps, reached threshold {limit}")

    return W


def fastIca(signals, alpha=1, thresh=1e-8, iterations=5000):
    m, n = signals.shape

    # Initialize random weights
    W = np.random.rand(m, m)

    for c in range(m):
        w = W[c, :].copy().reshape(m, 1)
        w = w / np.sqrt((w**2).sum())

        i = 0
        lim = 100
        while (lim > thresh) & (i < iterations):

            # Dot product of weight and signal
            ws = np.dot(w.T, signals)

            # Pass w*s into contrast function g
            wg = np.tanh(ws * alpha).T

            # Pass w*s into g prime
            wg_ = (1 - np.square(np.tanh(ws))) * alpha

            # Update weights
            wNew = (signals * wg.T).mean(axis=1) - wg_.mean() * w.squeeze()

            # Decorrelate weights
            wNew = wNew - np.dot(np.dot(wNew, W[:c].T), W[:c])
            wNew = wNew / np.sqrt((wNew**2).sum())

            # Calculate limit condition
            lim = np.abs(np.abs((wNew * w).sum()) - 1)

            # Update weights
            w = wNew

            # Update counter
            i += 1

        W[c, :] = w.T
    return W
