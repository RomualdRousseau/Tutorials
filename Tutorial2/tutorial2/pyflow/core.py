from __future__ import annotations

import copy
import json
from typing import Callable, Optional

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
        data: Optional[np.ndarray] = None,
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

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.data[idx]

    def __setitem__(self, idx: int, data: np.ndarray) -> None:
        self.data[idx] = data

    def __eq__(self, other) -> bool:
        if not isinstance(other, Params):
            return NotImplemented
        return np.array_equal(self.data, other.data)

    def update(self, data: np.ndarray) -> None:
        self[0] += data[0]
        self[1] = data[1]
        self[2] = data[2]

    def copy(self) -> Params:
        return Params(self.data[0].shape, data=self.data.copy())

    def to_list(self) -> list:
        return self.data.tolist()

    def from_list(self, alist: list):
        self.data = np.asarray(alist)


class Layer:
    """This is the base class for all layers. It provides the structure for storing weights and biases,
    and outlines the necessary methods that every layer should implement or support.
    """

    def __init__(self, kernel: Params, bias: Params) -> None:
        self.kernel = kernel
        self.bias = bias

    def call(self, x: np.ndarray, training: bool = False, **kwargs) -> np.ndarray:
        """Performs the logic of applying the layer to the input arguments."""
        raise NotImplementedError

    def optimize(self, *args, **kwargs) -> list[np.ndarray]:
        """Performs the logic of optimizing the layer to the input arguments."""
        raise NotImplementedError

    def update_step(self, trainable: bool, weights: tuple[np.ndarray, np.ndarray], optimizer_func: Callable) -> Layer:
        """This method updates either the weights or both weights and biases of the layer using a given
        optimization function."""
        if trainable:
            self.kernel.update(optimizer_func(weights[0], self.kernel[1], self.kernel[2]))
            self.bias.update(optimizer_func(weights[1], self.bias[1], self.bias[2]))
        return self

    def clone(self) -> Layer:
        cloned = copy.copy(self)
        cloned.kernel = self.kernel.copy()
        cloned.bias = self.bias.copy()
        return cloned

    def to_dict(self) -> dict[str, list]:
        return {"W": self.kernel.to_list(), "B": self.bias.to_list()}

    def from_dict(self, adict: dict[str, list]) -> None:
        self.kernel.from_list(adict["W"])
        self.bias.from_list(adict["B"])


class Model:
    """This is the base class for all models. It provides methods to manage the model's layers, clone the model, load
    model parameters from a file, and save the model to a file.
    """

    def __init__(self, layers: list[Layer], write_mask: Optional[list[bool]] = None):
        self.layers = layers
        self.write_mask = [True] * len(layers) if write_mask is None else write_mask

    def call(self, x: np.ndarray, training: bool = False) -> list[np.ndarray]:
        return self.compiled_call(x, training)

    def optimize(
        self, x: Optional[np.ndarray], y: Optional[np.ndarray]
    ) -> tuple[Optional[np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
        raise NotImplementedError

    def compile(self, optimizer: str = "rmsprop", loss: str = "mse") -> None:
        """Prepares the training process by setting up the necessary configurations such as
        the optimizer, loss function, and metrics.
        """

        self.optimizer_func: Callable = __functions__[optimizer]["func"]
        self.loss_func: Callable = __functions__[loss]["func"]
        self.loss_prime: Callable = __functions__[loss]["prime"]
        self.loss_acc: Callable = __functions__[loss]["acc"]

        def call_(layers: list[Layer], result: list[np.ndarray], training: bool) -> list[np.ndarray]:
            match layers:
                case []:
                    new_result = result
                case head, *tail:
                    new_result = call_(tail, [*result, head.call(result[-1], training=training)], training)
            return new_result

        def update_(
            layers: list[Layer], write_mask: list[bool], weights: list[tuple[np.ndarray, np.ndarray]]
        ) -> list[Layer]:
            match layers:
                case []:
                    new_result = []
                case head, *tail:
                    new_result = [
                        head.update_step(write_mask[0], weights[0], self.optimizer_func),
                        *update_(tail, write_mask[1:], weights[1:]),
                    ]
            return new_result

        self.compiled_call = lambda x, t: call_(self.layers, [x], t)
        self.compiled_update = lambda x: update_(self.layers, self.write_mask, x)

    def fit(
        self,
        x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        epochs: int = 10,
        batch_size: int = 128,
        shuffle: bool = True,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Trains the model on the given dataset."""

        history: dict[str, list[float]] = {"loss": [], "accuracy": []}

        N = x.shape[0] if x is not None else 0

        batch_sample = np.arange(N)
        batch_count = max(1, np.ceil(N / batch_size))

        first_pass = True

        for e in range(1, 1 + epochs):
            if verbose:
                print(f"Epoch {e}/{epochs}")

            if shuffle:
                np.random.shuffle(batch_sample)

            train_loss = 0
            train_accuracy = 0

            for i in range(batch_count):
                sample = batch_sample[i * batch_size : (i + 1) * batch_size]
                loss, accuracy = self.train_batch(x, y, sample)

                if first_pass:
                    first_pass = False
                    history["loss"].append(loss)
                    history["accuracy"].append(accuracy)

                if verbose:
                    beta = (i + 1) / batch_count
                    bar = "=" * int(30 * beta) + ">" + "." * int(30 * (1 - beta))
                    print(f"{i:3d}/{batch_count} [{bar}]\r", end="")

                train_loss += loss / batch_count
                train_accuracy += accuracy / batch_count

            history["loss"].append(train_loss)
            history["accuracy"].append(train_accuracy)

            if verbose:
                bar = "=" * 30
                print(f"{batch_count}/{batch_count} [{bar}] - loss: {train_loss:.4f} - accuracy: {train_accuracy:.4f}")

        return history

    def evaluate(self, x: np.ndarray, y: np.ndarray, verbose=True) -> tuple[float, float, np.ndarray]:
        """Evaluates the performance of the trained model on a test set."""
        loss, accuracy, yhat = self.test_step(x, y)
        if verbose:
            print(f"Test loss: {loss}")
            print(f"Test accuracy: {accuracy}")
        return loss, accuracy, yhat

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Uses the trained model to make predictions on new data."""
        *_, yhat = self.call(x)
        return yhat

    def train_step(self, x: Optional[np.ndarray], y: Optional[np.ndarray]) -> tuple[float, float]:
        yhat, weights = self.optimize(x, y)
        self.layers = self.compiled_update(weights)
        loss, accuracy = (self.loss_func(y, yhat).mean(), self.loss_acc(y, yhat).mean()) if yhat is not None else (0, 0)
        return loss, accuracy

    def train_batch(self, x: Optional[np.ndarray], y: Optional[np.ndarray], sample: np.ndarray) -> tuple[float, float]:
        return self.train_step(x[sample] if x is not None else x, y[sample] if y is not None else y)

    def test_step(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
        *_, yhat = self.call(x)
        loss, accuracy = (
            self.loss_func(y, yhat).mean(),
            self.loss_acc(y, yhat).mean(),
        )
        return loss, accuracy, yhat

    def clone(self) -> Model:
        cloned = copy.copy(self)
        cloned.layers = [x.clone() for x in self.layers]
        return cloned

    def load(self, file_path: str) -> None:
        with open(file_path, "r") as f:
            model_data = json.load(f)
        for i, dt in enumerate(model_data["layers"]):
            self.layers[i].from_dict(dt)
        self.write_mask = model_data["write_mask"]

    def save(self, file_path: str) -> None:
        model_data = {
            "layers": [lr.to_dict() for lr in self.layers],
            "write_mask": self.write_mask,
        }
        with open(file_path, "w") as f:
            json.dump(model_data, f, indent=4)
