import numpy as np

from tutorial1.math.linalg import almost, normalize


def test_almost():
    assert almost(1, 0.9, 0.1)


def test_points_not_almost():
    assert not almost(1, 0.9, 0.01)


def test_normalize():
    a = np.array([1, 1])
    b = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)])
    assert almost(normalize(a), b)
