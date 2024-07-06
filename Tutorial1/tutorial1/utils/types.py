from typing import Protocol


class Scene(Protocol):
    def reset(self) -> None:
        ...

    def update(self, dt: float) -> str:
        ...

    def draw(self) -> None:
        ...


class Widget(Protocol):
    def is_playing(self, latency: float = 0.1) -> bool:
        ...

    def reset(self) -> None:
        ...

    def update(self, dt: float) -> None:
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
