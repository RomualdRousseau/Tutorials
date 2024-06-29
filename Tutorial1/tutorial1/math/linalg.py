import numpy as np
import numpy.typing as npt
from numba import njit

EPS = 1e-7


def lst_2_arr(a: npt.ArrayLike) -> npt.NDArray[np.float64]:
    return np.array(a, dtype=np.float64)


@njit
def norm(v: npt.NDArray[np.float64]) -> float:
    return np.sqrt(np.sum(v**2))


@njit
def normalize(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return v / (norm(v) + EPS)


@njit
def almost(v: npt.ArrayLike, w: npt.ArrayLike, eps: float = 0.0001) -> bool:
    return np.allclose(v, w, 0.0, eps)
