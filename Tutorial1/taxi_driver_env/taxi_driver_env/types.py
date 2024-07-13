from typing import Protocol

import pyray as pr


class Scene(Protocol):
    def reset(self) -> None:
        ...

    def update(self, dt: float) -> str:
        ...

    def draw(self) -> None:
        ...


class Camera(Protocol):
    def reset(self) -> None:
        ...

    def update(self, dt: float) -> str:
        ...

    def draw(self) -> None:
        ...


class Entity(Protocol):
    def is_alive(self) -> bool:
        ...

    def hit(self, damage: int) -> None:
        ...

    def reset(self) -> None:
        ...

    def update(self, dt: float) -> None:
        ...

    def draw(self, layer: int) -> None:
        ...


class Widget(Protocol):
    def get_bound(self) -> pr.Rectangle:
        ...

    def reset(self) -> None:
        ...

    def update(self, dt: float) -> None:
        ...

    def draw(self) -> None:
        ...


class Effect(Protocol):
    def get_bound(self) -> pr.Rectangle:
        ...

    def is_playing(self, latency: float) -> bool:
        ...

    def reset(self) -> None:
        ...

    def update(self, dt: float) -> None:
        ...

    def draw(self) -> None:
        ...
