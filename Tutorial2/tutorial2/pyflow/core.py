from __future__ import annotations

import copy
import json
from typing import Optional, Protocol, Union
import numpy as np

from tutorial2.pyflow.functions import __functions__


class Params:
    """This class is responsible for handling parameters with a fixed shape. It allows for initialization of parameters,
    supports item setting and retrieval, copying, converting to and from list representations, and checking for equality.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        initializer: str = "zeros",
        data=Optional[np.ndarray],
    ) -> None:
        if data is None:
            init_func = __functions__[initializer]["func"]
            data = np.array(
                [
                    init_func(shape[0], shape[1]),
                    np.zeros(shape),
                    np.zeros(shape),
                ]
            )
        self.data = data
        self.shape = data.shape

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.data[idx]

    def __setitem__(self, idx: int, data: np.ndarray) -> None:
        self.data[idx] = data

    def __eq__(self, other) -> bool:
        if not isinstance(other, Params):
            return NotImplemented
        return np.array_equal(self.data, other.data)

    def copy(self) -> Params:
        return Params(self.data[0].shape, data=self.data.copy())

    def to_list(self) -> list:
        return self.data.tolist()

    def from_list(self, alist: list):
        self.data = np.asarray(alist)


class Trainable(Protocol):
    def call(self, x: np.ndarray, **kargs) -> np.ndarray: ...

    def gradient(self, **kargs) -> tuple[np.ndarray, np.ndarray]: ...


class Layer:
    """This is the base class for all layers. It provides the structure for storing weights and biases,
    and outlines the necessary methods that every layer should implement or support.
    """

    def __init__(self, W: np.ndarray, B: np.ndarray) -> None:
        self.W = W
        self.B = B

    def call(self, x: np.ndarray, **kargs) -> np.ndarray:
        raise NotImplemented

    def gradient(self, **kargs) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplemented

    def update_weights(self, dW: np.ndarray) -> None:
        self.W[0] = self.W[0] + dW[0]
        self.W[1] = dW[1]
        self.W[2] = dW[2]

    def update_biases(self, dB: np.ndarray) -> None:
        self.B[0] = self.B[0] + dB[0]
        self.B[1] = dB[1]
        self.B[2] = dB[2]

    def clone(self) -> Layer:
        cloned = copy.copy(self)
        cloned.W = self.W.copy()
        if self.B is not None:
            cloned.B = self.B.copy()
        return cloned

    def to_dict(self) -> dict[str, list]:
        if self.B is not None:
            return {"W": self.W.to_list(), "B": self.B.to_list()}
        else:
            return {"W": self.W.to_list()}

    def from_dict(self, adict: dict[str, list]) -> None:
        self.W.from_list(adict["W"])
        if self.B is not None:
            self.B.from_list(adict["B"])


class Trainer(Protocol):
    def compile(self, **kargs) -> None:
        """Prepares the training process by setting up the necessary configurations such as
        the optimizer, loss function, and metrics.
        """

    def fit(self, x: np.ndarray, y: np.ndarray, **kargs) -> dict[str, np.ndarray]:
        """Trains the model on the given dataset."""

    def evaluate(self, x: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Evaluates the performance of the trained model on a test set."""

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Uses the trained model to make predictions on new data."""


class Model:
    """This is the base class for all models. It provides methods to manage the model's layers, clone the model, load
    model parameters from a file, and save the model to a file.
    """

    def __init__(self, layers: list[Layer], write_mask: Optional[list[bool]] = None):
        self.layers = layers
        self.write_mask = [True] * len(layers) if write_mask is None else write_mask

    def compile(self, **kargs) -> None:
        raise NotImplemented

    def fit(self, x: np.ndarray, y: np.ndarray, **kargs) -> dict[str, np.ndarray]:
        raise NotImplemented

    def evaluate(self, x: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
        raise NotImplemented

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplemented

    def clone(self) -> Model:
        cloned = copy.copy(self)
        cloned.layers = [x.clone() for x in self.layers]
        return cloned

    def load(self, file_path: str) -> None:
        with open(file_path, "r") as f:
            model_data = json.load(f)
        for i, x in enumerate(model_data["layers"]):
            self.layers[i].from_dict(x)
        self.write_mask = model_data["write_mask"]

    def save(self, file_path: str) -> None:
        model_data = {
            "layers": [x.to_dict() for x in self.layers],
            "write_mask": self.write_mask,
        }
        with open(file_path, "w") as f:
            json.dump(model_data, f, indent=4)
