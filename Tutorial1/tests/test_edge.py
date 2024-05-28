from tutorial1.math import Point, intersect
from tutorial1.graph import Edge


def test_edges_equals():
    a = Point(1, 1)
    b = Point(2, 2)
    e1 = Edge(a, b)
    e2 = Edge(a, b)
    assert e1 == e1
    assert e2 == e2
    assert e1 == e2


def test_edges_not_equals():
    a = Point(1, 1)
    b = Point(2, 2)
    c = Point(3, 3)
    e1 = Edge(a, b)
    e2 = Edge(a, c)
    assert e1 == e1
    assert e2 == e2
    assert e1 != e2


def test_edges_non_directed():
    a = Point(1, 1)
    b = Point(2, 2)
    e1 = Edge(a, b)
    e2 = Edge(b, a)
    assert e1 == e1
    assert e2 == e2
    assert e1 == e2


def test_intersect():
    a = Point(1, 1)
    b = Point(3, 3)
    c = Point(1, 3)
    d = Point(3, 1)
    e1 = Edge(a, b)
    e2 = Edge(c, d)
    p = intersect(e1, e2)
    assert p == Point(2, 2)
    p = intersect(e2, e1)
    assert p == Point(2, 2)


def test_intersect_horiz():
    a = Point(1, 1)
    b = Point(3, 3)
    c = Point(1, 2)
    d = Point(3, 2)
    e1 = Edge(a, b)
    e2 = Edge(c, d)
    p = intersect(e1, e2)
    assert p == Point(2, 2)
    p = intersect(e2, e1)
    assert p == Point(2, 2)


def test_intersect_verti():
    a = Point(1, 1)
    b = Point(3, 3)
    c = Point(2, 3)
    d = Point(2, 1)
    e1 = Edge(a, b)
    e2 = Edge(c, d)
    p = intersect(e1, e2)
    assert p == Point(2, 2)
    p = intersect(e2, e1)
    assert p == Point(2, 2)


def test_intersect_parallel():
    a = Point(1, 1)
    b = Point(3, 3)
    c = Point(1, 2)
    d = Point(3, 4)
    e1 = Edge(a, b)
    e2 = Edge(c, d)
    p = intersect(e1, e2)
    assert p == None
    p = intersect(e2, e1)
    assert p == None


def test_not_intersect():
    a = Point(1, 1)
    b = Point(3, 3)
    c = Point(4, 5)
    d = Point(5, 3)
    e1 = Edge(a, b)
    e2 = Edge(c, d)
    p = intersect(e1, e2)
    assert p == None
    p = intersect(e2, e1)
    assert p == None
