import numpy as np

from tutorial2.pyflow.core import Params, Layer
from tutorial2.pyflow.functions import __functions__


class GeneticDense(Layer):

    def __init__(
        self,
        inputs,
        outputs,
        activation="linear",
        kernel_initializer="gorot",
        bias_initializer="zeros",
    ):
        super().__init__(
            Params((inputs, outputs), initializer_func=kernel_initializer),
            Params((1, outputs), initializer_func=bias_initializer),
        )
        self.activation = __functions__[activation]["func"]

    def call(self, x: np.ndarray, **kargs) -> np.ndarray:
        return self.activation(x @ self.W[0] + self.B[0])

    def gradient(self, **kargs) -> tuple[np.ndarray, np.ndarray]:
        rate: float = kargs.get("rate", 0.1)
        variance: float = kargs.get("variance", 0.1)

        if np.random.rand() < rate:
            dW = np.random.standard_normal(self.W[0].shape) * variance
            dB = np.random.standard_normal(self.B[0].shape) * variance
        else:
            dW = 0
            dB = 0

        return dW, dB
