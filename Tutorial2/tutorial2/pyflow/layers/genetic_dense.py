import numpy as np

from tutorial2.pyflow.core import Layer, Params
from tutorial2.pyflow.functions import __functions__


class GeneticDense(Layer):
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

    def call(self, x: np.ndarray, *args, training: bool = False, **kwargs) -> np.ndarray:
        return self.activation(x @ self.kernel[0] + self.bias[0])

    def backward(self, *args, **kwargs) -> list[np.ndarray]:
        rate, variance = args
        if np.random.rand() < rate:
            dW = np.random.standard_normal(self.kernel[0].shape) * variance
            dB = np.random.standard_normal(self.bias[0].shape) * variance
        else:
            dW = np.zeros(self.kernel[0].shape)
            dB = np.zeros(self.bias[0].shape)
        return [dW, dB]
