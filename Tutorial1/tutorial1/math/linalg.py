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


@njit
def intersect(x1, y1, x2, y2, x3, y3, x4, y4, atol):
    match np.linalg.det(np.array([[x1 - x2, y1 - y2], [x3 - x4, y3 - y4]])):
        case d if d != 0:
            match -np.linalg.det(np.array([[x1 - x2, y1 - y2], [x1 - x3, y1 - y3]])) / d:
                case u if abs(u - 0.5) <= (atol + 0.5):
                    match np.linalg.det(np.array([[x1 - x3, y1 - y3], [x3 - x4, y3 - y4]])) / d:
                        case t if abs(t - 0.5) <= (atol + 0.5):
                            return np.array(
                                [
                                    np.interp(t, [0, 1], [x1, x2]),
                                    np.interp(t, [0, 1], [y1, y2]),
                                ]
                            )
    return None
