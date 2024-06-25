from __future__ import annotations

from typing import Optional, Protocol

import numpy as np

from tutorial2.pyflow.core import Layer, Model


class GeneticIndividual(Protocol):
    def get_model(self) -> Genetic:
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
        s = sum((x.get_fitness() for x in self.pool))
        for individual in self.pool:
            individual.set_fitness(individual.get_fitness() / s)

    def select_parent(self) -> GeneticIndividual:
        r = np.random.rand()
        best_index = 0
        while r > 0:
            r -= self.pool[best_index].get_fitness()
            best_index += 1
        return self.pool[best_index - 1]


class Genetic(Model):
    def __init__(
        self, layers: list[Layer], write_mask: Optional[list[bool]] = None, rate: float = 0.1, variance: float = 0.1
    ):
        super().__init__(layers, write_mask)
        self.rate = rate
        self.variance = variance

    def train_step(self, x: Optional[np.ndarray], y: Optional[np.ndarray]) -> tuple[float, float]:
        def mutate(
            layers: list[Layer], result: list[tuple[np.ndarray, np.ndarray]]
        ) -> list[tuple[np.ndarray, np.ndarray]]:
            match layers:
                case []:
                    new_result = result
                case *head, tail:
                    dW, dB = tail.optimize(self.rate, self.variance)
                    new_result = mutate(head, [(dW, dB), *result])
            return new_result

        self.update_step(mutate(self.layers, []))
        return 0, 0
