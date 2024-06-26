import numpy as np

from tutorial2.pyflow.core import Layer, Params
from tutorial2.pyflow.functions import __functions__


class Dense(Layer):
    def __init__(
        self,
        inputs: int,
        outputs: int,
        activation: str = "linear",
        kernel_initializer: str = "gorot",
        bias_initializer: str = "zeros",
    ):
        super().__init__(
            Params((inputs, outputs), initializer=kernel_initializer),
            Params((1, outputs), initializer=bias_initializer),
        )
        self.activation = __functions__[activation]["func"]
        self.activation_prime = __functions__[activation]["prime"]

    def call(self, x: np.ndarray, *args, training: bool = False, **kwargs) -> np.ndarray:
        return self.activation(x @ self.kernel[0] + self.bias[0])

    def backward(self, *args, **kwargs) -> list[np.ndarray]:
        x1, x0, error = args
        error = error * self.activation_prime(x1)
        dW = x0.T @ error
        dB = error.sum(axis=0, keepdims=True)
        error = error @ self.kernel[0].T
        return [dW, dB, error]
