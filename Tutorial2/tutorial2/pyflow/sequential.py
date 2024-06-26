from __future__ import annotations

import copy
import json
from functools import reduce
from typing import Optional

import numpy as np

from tutorial2.pyflow.core import Layer, Model, Trainer


class Sequential(Model):
    def __init__(self, layers: list[Layer], trainer: Optional[Trainer]):
        from tutorial2.pyflow import gradient

        super().__init__(trainer if trainer is not None else gradient)
        self.layers = layers

    def call(self, x: np.ndarray, training: bool = False) -> list[np.ndarray]:
        forward = lambda res, lr: [*res, lr.call(res[-1], training=training)]
        return reduce(forward, self.layers, [x])

    def clone(self) -> Model:
        cloned = copy.copy(self)
        cloned.layers = [x.clone() for x in self.layers]
        return cloned

    def load(self, file_path: str) -> None:
        with open(file_path, "r") as f:
            model_data = json.load(f)
        for i, dt in enumerate(model_data["layers"]):
            self.layers[i].from_dict(dt)

    def save(self, file_path: str) -> None:
        model_data = {"layers": [lr.to_dict() for lr in self.layers]}
        with open(file_path, "w") as f:
            json.dump(model_data, f, indent=4)
