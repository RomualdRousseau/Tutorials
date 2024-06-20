import numpy as np

from tutorial1.math.geom import (
    Point,
    Segment,
    break_segment,
    cast_ray_segments,
    collision_circle_segment,
    intersect,
    points_to_segments,
)


def test_segment_length():
    a = Point(np.array([1, 1]))
    b = Point(np.array([2, 2]))
    e = Segment(a, b)
    assert e.length == np.sqrt(2)


def test_segment_middle():
    a = Point(np.array([1, 1]))
    b = Point(np.array([2, 2]))
    e = Segment(a, b)
    assert e.middle == Point(np.array([1.5, 1.5]))


def test_segment_angle():
    a = Point(np.array([1, 1]))
    b = Point(np.array([2, 2]))
    e = Segment(a, b)
    assert e.angle == np.pi / 4


def test_segment_almost():
    a = Point(np.array([1, 1]))
    b = Point(np.array([2, 2]))
    c = Point(np.array([2.01, 2.01]))
    e = Segment(a, b)
    f = Segment(a, c)
    assert e.almost(f, 0.5)


def test_segment_not_almost():
    a = Point(np.array([1, 1]))
    b = Point(np.array([2, 2]))
    c = Point(np.array([2.6, 2.6]))
    e = Segment(a, b)
    f = Segment(a, c)
    assert not e.almost(f, 0.5)


def test_segments_equals():
    a = Point(np.array([1, 1]))
    b = Point(np.array([2, 2]))
    e1 = Segment(a, b)
    e2 = Segment(a, b)
    assert e1 == Segment(a, b)
    assert e2 == Segment(a, b)
    assert e1 == e2


def test_segments_not_equals():
    a = Point(np.array([1, 1]))
    b = Point(np.array([2, 2]))
    c = Point(np.array([3, 3]))
    e1 = Segment(a, b)
    e2 = Segment(a, c)
    assert e1 == Segment(a, b)
    assert e2 == Segment(a, c)
    assert e1 != e2


def test_segments_non_directed():
    a = Point(np.array([1, 1]))
    b = Point(np.array([2, 2]))
    e1 = Segment(a, b)
    e2 = Segment(b, a)
    assert e1 == Segment(a, b)
    assert e2 == Segment(b, a)
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
    assert p is None
    p = intersect(e2, e1)
    assert p is None


def test_not_intersect():
    a = Point(np.array([1, 1]))
    b = Point(np.array([3, 3]))
    c = Point(np.array([4, 5]))
    d = Point(np.array([5, 3]))
    e1 = Segment(a, b)
    e2 = Segment(c, d)
    p = intersect(e1, e2)
    assert p is None
    p = intersect(e2, e1)
    assert p is None


def test_segment_collision_circle():
    a = Point(np.array([1, 1]))
    b = Point(np.array([3, 3]))
    c = Point(np.array([1, 3]))
    e = Segment(a, b)
    assert collision_circle_segment(c, 3, e) is not None


def test_segment_not_collision_circle():
    a = Point(np.array([1, 1]))
    b = Point(np.array([3, 3]))
    x = Point(np.array([1, 3]))
    e = Segment(a, b)
    assert collision_circle_segment(x, 1, e) is None


def test_cast_ray_segments_intersect():
    n = np.array([1, -1]) / np.sqrt(2)
    a = Point(np.array([1, 1]))
    b = Point(np.array([2, 2]))
    c = Point(np.array([1, 2]))
    d = Point(np.array([2, 0]))
    e = Point(np.array([6, 4]))
    x = Point(np.array([1.5, 1.5]))
    e1 = Segment(a, b)
    e2 = Segment(d, e)
    assert cast_ray_segments(c, n, 2, [e1, e2]) == Segment(c, x)


def test_cast_ray_segments_not_intersect():
    n = np.array([-1, 0])
    a = Point(np.array([1, 1]))
    b = Point(np.array([2, 2]))
    c = Point(np.array([1, 2]))
    x = Point(np.array([-1, 2]))
    e = Segment(a, b)
    assert cast_ray_segments(c, n, 2, [e]) == Segment(c, x)


def test_points_to_segments_closed():
    a = Point(np.array([1, 1]))
    b = Point(np.array([2, 1]))
    c = Point(np.array([2, 2]))
    d = Point(np.array([1, 2]))
    assert points_to_segments([a, b, c, d]) == [
        Segment(a, b),
        Segment(b, c),
        Segment(c, d),
        Segment(d, a),
    ]


def test_points_to_segments_not_closed():
    a = Point(np.array([1, 1]))
    b = Point(np.array([2, 1]))
    c = Point(np.array([2, 2]))
    d = Point(np.array([1, 2]))
    assert points_to_segments([a, b, c, d], False) == [
        Segment(a, b),
        Segment(b, c),
        Segment(c, d),
    ]


def test_segment_break():
    a = Point(np.array([1, 1]))
    b = Point(np.array([2, 1]))
    c = Point(np.array([2, 2]))
    d = Point(np.array([1, 2]))
    x = Point(np.array([1.5, 1.5]))
    e1 = Segment(a, c)
    e2 = Segment(b, d)
    assert break_segment(e1, e2) == [
        Segment(b, x),
        Segment(x, d),
    ]


def test_segment_not_break():
    a = Point(np.array([1, 1]))
    b = Point(np.array([2, 1]))
    c = Point(np.array([0, 0]))
    d = Point(np.array([1, 2]))
    e1 = Segment(a, c)
    e2 = Segment(b, d)
    assert break_segment(e1, e2) == [e2]
