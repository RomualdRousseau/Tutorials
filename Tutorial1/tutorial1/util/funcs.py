def identity():
    """Return the identity function"""
    return lambda func: func


def compose(func):
    """Compose 2 functions: func(other(x))"""
    return lambda other: lambda x: func(other(x))


def apply(value):
    """Return a function with the value as argument"""
    return lambda func: func(value)


def curry(func):
    """Return a function curry of its last argument"""
    return lambda *x: lambda y: func(*x, y)


def extend(func, lst):
    """Return a list of func from the list"""
    return [curry(func)(*e) for e in lst]


def constant(value):
    """Return a function that returns the value"""
    return lambda _: value


def pack_args(*x):
    return x
