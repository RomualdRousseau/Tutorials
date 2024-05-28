from tutorial1.math import Point


def test_points_equals():
    a = Point(1, 1)
    b = Point(1, 1)
    assert a == b


def test_points_not_equals():
    a = Point(1, 1)
    b = Point(2, 2)
    assert a != b
