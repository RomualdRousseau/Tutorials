from typing import Optional

import numpy as np

from tutorial2.pyflow.core import Layer, Model


class Sequential(Model):
    def __init__(self, layers: list[Layer], write_mask: Optional[list[bool]] = None):
        super().__init__(layers, write_mask)

    def optimize(
        self, x: Optional[np.ndarray], y: Optional[np.ndarray]
    ) -> tuple[Optional[np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
        assert x is not None and y is not None

        def gradient(
            layers: list[Layer], x: list[np.ndarray], loss: np.ndarray, result: list[tuple[np.ndarray, np.ndarray]]
        ) -> list[tuple[np.ndarray, np.ndarray]]:
            match layers:
                case []:
                    new_result = result
                case *head, tail:
                    dw, db, loss_ = tail.optimize(x[-1], x[-2], loss)
                    new_result = gradient(head, x[:-1], loss_, [(dw, db), *result])
            return new_result

        *xhat, yhat = self.call(x, training=True)
        weights = gradient(self.layers, [*xhat, yhat], self.loss_prime(y, yhat), [])
        return yhat, weights
