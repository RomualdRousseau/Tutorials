import numpy as np

from tutorial1.util.geom import (
    Point,
    Segment,
    distance_point_segment,
    intersect,
    point_on_segment,
)


def test_segments_equals():
    a = Point(np.array([1, 1]))
    b = Point(np.array([2, 2]))
    e1 = Segment(a, b)
    e2 = Segment(a, b)
    assert e1 == e1
    assert e2 == e2
    assert e1 == e2


def test_segments_not_equals():
    a = Point(np.array([1, 1]))
    b = Point(np.array([2, 2]))
    c = Point(np.array([3, 3]))
    e1 = Segment(a, b)
    e2 = Segment(a, c)
    assert e1 == e1
    assert e2 == e2
    assert e1 != e2


def test_segments_non_directed():
    a = Point(np.array([1, 1]))
    b = Point(np.array([2, 2]))
    e1 = Segment(a, b)
    e2 = Segment(b, a)
    assert e1 == e1
    assert e2 == e2
    assert e1 == e2


def test_intersect():
    a = Point(np.array([1, 1]))
    b = Point(np.array([3, 3]))
    c = Point(np.array([1, 3]))
    d = Point(np.array([3, 1]))
    e1 = Segment(a, b)
    e2 = Segment(c, d)
    p = intersect(e1, e2)
    assert p and p.almost(Point(np.array([2, 2])))
    p = intersect(e2, e1)
    assert p and p.almost(Point(np.array([2, 2])))


def test_intersect_horiz():
    a = Point(np.array([1, 1]))
    b = Point(np.array([3, 3]))
    c = Point(np.array([1, 2]))
    d = Point(np.array([3, 2]))
    e1 = Segment(a, b)
    e2 = Segment(c, d)
    p = intersect(e1, e2)
    assert p and p.almost(Point(np.array([2, 2])))
    p = intersect(e2, e1)
    assert p and p.almost(Point(np.array([2, 2])))


def test_intersect_verti():
    a = Point(np.array([1, 1]))
    b = Point(np.array([3, 3]))
    c = Point(np.array([2, 3]))
    d = Point(np.array([2, 1]))
    e1 = Segment(a, b)
    e2 = Segment(c, d)
    p = intersect(e1, e2)
    assert p and p.almost(Point(np.array([2, 2])))
    p = intersect(e2, e1)
    assert p and p.almost(Point(np.array([2, 2])))


def test_intersect_parallel():
    a = Point(np.array([1, 1]))
    b = Point(np.array([3, 3]))
    c = Point(np.array([1, 2]))
    d = Point(np.array([3, 4]))
    e1 = Segment(a, b)
    e2 = Segment(c, d)
    p = intersect(e1, e2)
    assert p == None
    p = intersect(e2, e1)
    assert p == None


def test_not_intersect():
    a = Point(np.array([1, 1]))
    b = Point(np.array([3, 3]))
    c = Point(np.array([4, 5]))
    d = Point(np.array([5, 3]))
    e1 = Segment(a, b)
    e2 = Segment(c, d)
    p = intersect(e1, e2)
    assert p == None
    p = intersect(e2, e1)
    assert p == None


def test_point_on_segment():
    a = Point(np.array([1, 1]))
    b = Point(np.array([2, 2]))
    c = Point(np.array([1.5, 1.5]))
    e = Segment(a, b)
    assert point_on_segment(c, e)


def test_point_not_on_segment():
    a = Point(np.array([1, 1]))
    b = Point(np.array([2, 2]))
    c = Point(np.array([1, 2]))
    e = Segment(a, b)
    assert not point_on_segment(c, e)


def test_segment_distance_with_point():
    a = Point(np.array([1, 1]))
    b = Point(np.array([2, 2]))
    c = Point(np.array([1, 2]))
    e = Segment(a, b)
    assert distance_point_segment(c, e) == np.sqrt(2) / 2
