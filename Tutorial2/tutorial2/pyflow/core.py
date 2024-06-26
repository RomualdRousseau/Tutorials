from __future__ import annotations

import copy
from typing import Callable, Optional, Protocol

import numpy as np

from tutorial2.pyflow.functions import __functions__


class Trainer(Protocol):
    def train(self, model: Model, x: Optional[np.ndarray], y: Optional[np.ndarray]) -> Optional[np.ndarray]:
        ...


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

    def __init__(self, kernel: Params, bias: Params, trainable: bool = True) -> None:
        self.kernel = kernel
        self.bias = bias
        self.trainable = trainable

    def call(self, x: np.ndarray, training: bool = False, **kwargs) -> np.ndarray:
        """Performs the logic of applying the layer to the input arguments."""
        raise NotImplementedError

    def backward(self, *args, **kwargs) -> list[np.ndarray]:
        """Performs the logic of optimizing the layer to the input arguments."""
        raise NotImplementedError

    def update_gradients(self, gradients: tuple[np.ndarray, np.ndarray], optimizer_func: Callable) -> Layer:
        """This method updates either the weights or both weights and biases of the layer using a given
        optimization function."""
        if self.trainable:
            self.kernel.update(optimizer_func(gradients[0], self.kernel[1], self.kernel[2]))
            self.bias.update(optimizer_func(gradients[1], self.bias[1], self.bias[2]))
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

    def __init__(self, trainer: Trainer):
        self.trainer = trainer

    def call(self, x: np.ndarray, training: bool = False) -> list[np.ndarray]:
        raise NotImplementedError

    def clone(self) -> Model:
        raise NotImplementedError

    def load(self, file_path: str) -> None:
        raise NotImplementedError

    def save(self, file_path: str) -> None:
        raise NotImplementedError

    def compile(self, optimizer: str = "rmsprop", loss: str = "mse") -> None:
        """Prepares the training process by setting up the necessary configurations such as
        the optimizer, loss function, and metrics.
        """
        self.optimizer_func: Callable = __functions__[optimizer]["func"]
        self.loss_func: Callable = __functions__[loss]["func"]
        self.loss_prime: Callable = __functions__[loss]["prime"]
        self.loss_acc: Callable = __functions__[loss]["acc"]

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

    def evaluate(self, x: np.ndarray, y: np.ndarray, verbose: bool = True) -> tuple[float, float, np.ndarray]:
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

    def train_batch(self, x: Optional[np.ndarray], y: Optional[np.ndarray], sample: np.ndarray) -> tuple[float, float]:
        x_sample = x[sample] if x is not None else x
        y_sample = y[sample] if y is not None else y
        return self.train_step(x_sample, y_sample)

    def train_step(self, x: Optional[np.ndarray], y: Optional[np.ndarray]) -> tuple[float, float]:
        yhat = self.trainer.train(self, x, y)
        loss, accuracy = self.compute_stats(y, yhat)
        return loss, accuracy

    def test_step(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
        *_, yhat = self.call(x)
        loss, accuracy = self.compute_stats(y, yhat)
        return loss, accuracy, yhat

    def compute_stats(self, y: Optional[np.ndarray], yhat: Optional[np.ndarray]) -> tuple[float, float]:
        if y is None or yhat is None:
            return 0, 0
        return self.loss_func(y, yhat).mean(), self.loss_acc(y, yhat).mean()
