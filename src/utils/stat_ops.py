import numpy as np


def extract_components_features(x: np.ndarray, shape: str = "tall") -> tuple:
    if shape == "wide":
        n_components, n_features = x.shape
        assert n_features >= n_components, "Expected wide but got tall matrix as input"
    else:
        n_features, n_components = x.shape
        assert n_features >= n_components, "Expected tall but got wide matrix as input"

    return n_components, n_features


def center(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x_mean = np.mean(x, axis=axis, keepdims=True)
    x_centered = x - x_mean
    return x_centered, x_mean


def covariance(x: np.ndarray, axis: int = 1, shape: str = "wide") -> np.ndarray:
    _, n_features = extract_components_features(x, shape)

    mean = np.mean(x, axis=axis, keepdims=True)
    m = x - mean

    return (m.dot(m.T)) / (n_features - 1)


def whiten_svd(x: np.ndarray, shape: str = "wide") -> np.ndarray:
    covariance_x = covariance(x, shape=shape)

    # Single value decoposition
    U, S, _ = np.linalg.svd(covariance_x)

    # Calculate diagonal matrix of eigenvalues and create whitening matrix
    d = np.diag(1.0 / np.sqrt(S))
    whitening_m = np.dot(U, np.dot(d, U.T))

    # Project onto whitening matrix
    x_whitened = np.dot(whitening_m, x)

    return x_whitened


def whiten_evd(x: np.ndarray) -> np.ndarray:
    covariance_x = covariance(x)

    # Eigenvalue decomposition
    eigenval, eigenvec = np.linalg.eig(covariance_x)

    # Calculate diagonal matrix of eigenvalues and create whitening matrix
    d = np.diag(1.0 / eigenval)
    whitening_m = np.dot(eigenvec, np.dot(d, eigenvec.T))

    # Project onto whitening matrix
    x_whitened = np.dot(whitening_m, x)
    return x_whitened
