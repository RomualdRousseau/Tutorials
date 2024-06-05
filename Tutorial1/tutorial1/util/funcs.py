from typing import Callable, TypeVar

T = TypeVar("T")


def apply(func: Callable[[T], None], x: T) -> T:
    func(x)
    return x