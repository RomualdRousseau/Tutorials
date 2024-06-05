import math
from tutorial1.util.geom import Point, distance


def test_points_equals():
    a = Point(1, 1)
    b = Point(1, 1)
    assert a == b


def test_points_not_equals():
    a = Point(1, 1)
    b = Point(2, 2)
    assert a != b


def test_points_distance():
    a = Point(1, 1)
    b = Point(2, 2)
    assert distance(a, b) == math.sqrt(2)
