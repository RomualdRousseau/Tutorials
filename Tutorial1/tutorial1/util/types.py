from typing import Protocol


class Scene(Protocol):
    def reset(self) -> None:
        ...

    def update(self, dt: float) -> str:
        ...

    def draw(self) -> None:
        ...


class Entity(Protocol):
    def is_alive(self) -> bool:
        ...

    def reset(self) -> None:
        ...

    def hit(self, damage: int) -> None:
        ...

    def update(self, dt: float) -> None:
        ...

    def draw(self, layer: int) -> None:
        ...


def bit_set(num: int, pos: int) -> int:
    return num | (1 << pos)


def bit_set_if(num: int, pos: int, pred: bool) -> int:
    return bit_set(num, pos) if pred else bit_unset(num, pos)


def bit_unset(num: int, pos: int) -> int:
    return num & ~(1 << pos)


def is_bit_set(num: int, pos: int) -> bool:
    return bool(num & (1 << pos))
