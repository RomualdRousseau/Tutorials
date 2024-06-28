from __future__ import annotations

from typing import Optional, Protocol

import numpy as np

from tutorial2.pyflow.core import Model
from tutorial2.pyflow.sequential import Sequential


class GeneticIndividual(Protocol):
    def get_model(self) -> Sequential:
        ...

    def get_fitness(self) -> float:
        ...

    def set_fitness(self, v: float) -> None:
        ...


class GeneticPool:
    def __init__(self, pool: list[GeneticIndividual]) -> None:
        self.pool = pool

    def sample(self, sample_count: Optional[int] = None) -> None:
        if sample_count is None:
            sample_count = int(np.floor(np.random.rand() * len(self.pool)))
        sample_count = max(1, sample_count)

        self.pool.sort(key=lambda x: x.get_fitness(), reverse=True)

        if len(self.pool) > sample_count:
            self.pool = self.pool[:sample_count]

    def normalize(self) -> None:
        s = sum((max(0, individual.get_fitness()) for individual in self.pool))
        for individual in self.pool:
            individual.set_fitness(max(0, individual.get_fitness()) / s)

    def best_parent(self) -> GeneticIndividual:
        return self.pool[0]

    def select_parent(self) -> GeneticIndividual:
        r = np.random.rand()
        best_index = 0
        while r > 0:
            r -= self.pool[best_index].get_fitness()
            best_index += 1
        return self.pool[best_index - 1]


class GeneticTrainer:
    def __init__(self, rate: float = 0.1, variance: float = 1):
        self.rate = rate
        self.variance = variance

    def train(self, model: Model, x: Optional[np.ndarray], y: Optional[np.ndarray]) -> Optional[np.ndarray]:
        assert isinstance(model, Sequential)
        assert x is None
        assert y is None

        gradients: list[tuple[np.ndarray, np.ndarray]] = []
        for lr in reversed(model.layers):
            dw, db = lr.backward(self.rate, self.variance)
            gradients = [(dw, db), *gradients]

        for lr, gr in zip(model.layers, gradients, strict=True):
            lr.apply_gradient(gr, model.optimizer_func)

        return None
