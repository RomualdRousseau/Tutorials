from typing import Optional

import numpy as np
import numpy.typing as npt
from numba import njit

EPS = 1e-7


def lst_2_arr(a: npt.ArrayLike) -> npt.NDArray[np.float64]:
    return np.array(a, dtype=np.float64)


@njit
def det(a: npt.ArrayLike) -> float:
    return np.linalg.det(np.array(a))


@njit
def norm(v: npt.NDArray[np.float64]) -> float:
    return np.sqrt(np.sum(v**2))


@njit
def normalize(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return v / (norm(v) + EPS)


@njit
def almost(v: npt.ArrayLike, w: npt.ArrayLike, eps: float = 0.0001) -> bool:
    return np.allclose(v, w, 0.0, eps)


@njit
def intersect(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    c: npt.NDArray[np.float64],
    d: npt.NDArray[np.float64],
    atol: float,
) -> Optional[npt.NDArray[np.float64]]:
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    x4, y4 = d

    match det([[x1 - x2, y1 - y2], [x3 - x4, y3 - y4]]):
        case dd if dd != 0:
            match -det([[x1 - x2, y1 - y2], [x1 - x3, y1 - y3]]) / dd:
                case u if abs(u - 0.5) <= (atol + 0.5):
                    match det([[x1 - x3, y1 - y3], [x3 - x4, y3 - y4]]) / dd:
                        case t if abs(t - 0.5) <= (atol + 0.5):
                            return np.array(
                                [
                                    np.interp(t, [0, 1], [x1, x2]),
                                    np.interp(t, [0, 1], [y1, y2]),
                                ]
                            )

    return None
