import numpy as np
import pyray as pr

from tutorial1.math.geom import (
    Point,
    Segment,
    distance,
    distance_point_segment,
    nearest_point_segment,
    point_in_polygon,
    point_on_segment,
)
from tutorial1.math.linalg import almost


def test_point_to_vector2():
    a = Point(np.array([1, 1]))
    assert pr.vector2_equals(a.to_vec(), pr.Vector2(1, 1))


def test_point_hash():
    a = Point(np.array([1, 1]))
    b = 2
    assert a.__hash__() == b


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


def test_point_on_segment():
    a = Point(np.array([1, 1]))
    b = Point(np.array([2, 2]))
    c = Point(np.array([1.5, 1.5]))
    e = Segment(a, b)
    assert point_on_segment(c, e)


def test_point_not_on_segment_within_bounding_box():
    a = Point(np.array([1, 1]))
    b = Point(np.array([2, 2]))
    c = Point(np.array([1, 2]))
    e = Segment(a, b)
    assert not point_on_segment(c, e)


def test_point_not_on_segment_out_of_bounding_box():
    a = Point(np.array([1, 1]))
    b = Point(np.array([2, 2]))
    c = Point(np.array([4, 4]))
    e = Segment(a, b)
    assert not point_on_segment(c, e)


def test_point_in_polygon_strict():
    a = Point(np.array([1, 1]))
    b = Point(np.array([2, 1]))
    c = Point(np.array([2, 2]))
    d = Point(np.array([1, 2]))
    x = Point(np.array([1.5, 1.5]))
    assert point_in_polygon(x, [a, b, c, d])


def test_point_not_in_polygon_strict():
    a = Point(np.array([1, 1]))
    b = Point(np.array([2, 1]))
    c = Point(np.array([2, 2]))
    d = Point(np.array([1, 2]))
    x = Point(np.array([1, 1.5]))
    assert not point_in_polygon(x, [a, b, c, d])


def test_point_in_polygon_not_strict():
    a = Point(np.array([1, 1]))
    b = Point(np.array([2, 1]))
    c = Point(np.array([2, 2]))
    d = Point(np.array([1, 2]))
    x = Point(np.array([1, 1.5]))
    assert point_in_polygon(x, [a, b, c, d], False)


def test_point_not_in_polygon_not_strict():
    a = Point(np.array([1, 1]))
    b = Point(np.array([2, 1]))
    c = Point(np.array([2, 2]))
    d = Point(np.array([1, 2]))
    x = Point(np.array([0, 0]))
    assert not point_in_polygon(x, [a, b, c, d], False)


def test_point_distance_with_segment():
    a = Point(np.array([1, 1]))
    b = Point(np.array([2, 2]))
    c = Point(np.array([1, 2]))
    e = Segment(a, b)
    assert almost(distance_point_segment(c, e), np.sqrt(2) / 2)


def test_point_distance_out_of_segment():
    a = Point(np.array([1, 1]))
    b = Point(np.array([2, 2]))
    c = Point(np.array([0, 0]))
    e = Segment(a, b)
    assert distance_point_segment(c, e) == np.inf


def test_point_nearest_with_segment():
    a = Point(np.array([1, 1]))
    b = Point(np.array([2, 2]))
    c = Point(np.array([1, 2]))
    e = Segment(a, b)
    x = nearest_point_segment(c, e)
    assert x is not None
    assert x.almost(Point(np.array([1.5, 1.5])))


def test_point_nearest_out_of_segment():
    a = Point(np.array([1, 1]))
    b = Point(np.array([2, 2]))
    c = Point(np.array([0, 0]))
    e = Segment(a, b)
    assert nearest_point_segment(c, e) is None
