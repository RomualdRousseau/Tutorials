import numpy as np


EPS = 1e-7


def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + EPS)


def almost(v: float | np.ndarray, w: float | np.ndarray, eps=0.0001) -> bool:
    return np.allclose(v, w, 0, eps)
