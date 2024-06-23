from typing import Optional, Protocol, Union

import numpy as np

from tutorial2.pyflow.core import Layer, Model
from tutorial2.pyflow.functions import __functions__


class Individual(Protocol):
    def get_fitness(self) -> float: ...

    def set_fitness(self, v: float) -> None: ...


class GeneticPool:

    def __init__(self, pool: list[Individual] = []):
        self.pool = pool

    def sample(self, sample_count: Optional[int] = None):
        if sample_count is None:
            sample_count = int(np.floor(np.random.rand() * len(self.pool)))
        sample_count = max(1, sample_count)
        self.pool = sorted(self.pool, key=lambda x: x.get_fitness(), reverse=True)
        if len(self.pool) > sample_count:
            self.pool = self.pool[:sample_count]

    def normalize(self):
        sum = 0
        for individual in self.pool:
            sum += individual.get_fitness()
        for individual in self.pool:
            individual.set_fitness(individual.get_fitness() / sum)

    def select_parent(self):
        r = np.random.rand()
        best_index = 0
        while r > 0:
            r -= self.pool[best_index].get_fitness()
            best_index += 1
        return self.pool[best_index - 1]


class GeneticModel(Model):

    def __init__(self, layers: list[Layer], write_mask: Optional[list[bool]] = None):
        super().__init__(layers, write_mask)

    def compile(self, **kargs) -> None:
        optimizer: str = kargs.get("optimizer", "rmsprop")
        rate: float = kargs.get("rate", 0.1)
        variance: float = kargs.get("variance", 0.1)
        optimizer_func = __functions__[optimizer]["func"]

        def call(layers: list[Layer], result: list[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray]:
            match layers:
                case []:
                    return result, result[-1]
                case _:
                    head, *tail = layers
                    return call(tail, [*result, head.call(result[-1])])

        def gradient(
            layers: list[Layer], result: list[tuple[np.ndarray, np.ndarray]] = []
        ) -> list[tuple[np.ndarray, np.ndarray]]:
            match layers:
                case []:
                    return result
                case _:
                    *head, tail = layers
                    grad = tail.gradient(rate=rate, variance=variance)
                    return gradient(head, [grad, *result])

        def update(layers: list[Layer], write_mask: list[bool], weights: list[tuple[np.ndarray, np.ndarray]]) -> None:
            match layers:
                case []:
                    pass
                case _:
                    head, *tail = layers
                    if write_mask[0]:
                        head.update_weights(optimizer_func(weights[0][0], head.W[1], head.W[2]))
                        if head.B is not None:
                            head.update_biases(optimizer_func(weights[0][1], head.B[1], head.B[2]))
                    update(tail, write_mask[1:], weights[1:])

        self._call = lambda x: call(self.layers, [x])
        self._train = lambda: update(
            self.layers,
            self.write_mask,
            gradient(self.layers),
        )

    def fit(self, x: np.ndarray = [], y: np.ndarray = [], **kargs) -> dict[str, np.ndarray]:
        self._train()
        return {"loss": [], "accuracy": []}

    def evaluate(self, x: np.ndarray, **kargs) -> tuple[dict[str, np.ndarray], np.ndarray]:
        _, yhat = self._call(x)
        return {"loss": [], "accuracy": []}, yhat

    def predict(self, x: np.ndarray, **kargs) -> np.ndarray:
        _, yhat = self._call(x)
        return yhat
