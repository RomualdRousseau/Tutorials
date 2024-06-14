import numpy as np

from tutorial1.math.geom import Point, distance


def test_points_equals():
    a = Point(np.array([1, 1]))
    b = Point(np.array([1, 1]))
    assert a == b


def test_points_not_equals():
    a = Point(np.array([1, 1]))
    b = Point(np.array([2, 2]))
    assert a != b


def test_points_almost():
    a = Point(np.array([1, 1]))
    b = Point(np.array([0.9, 0.9]))
    assert a.almost(b, 0.1)


def test_points_not_almost():
    a = Point(np.array([1, 1]))
    b = Point(np.array([0.9, 0.9]))
    assert not a.almost(b, 0.01)


def test_points_distance():
    a = Point(np.array([1, 1]))
    b = Point(np.array([2, 2]))
    assert distance(a, b) == np.sqrt(2)
