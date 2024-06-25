from typing import Optional

import numpy as np

from tutorial2.pyflow.core import Layer, Model


class Sequential(Model):
    def __init__(self, layers: list[Layer], write_mask: Optional[list[bool]] = None):
        super().__init__(layers, write_mask)

    def train_step(self, x: Optional[np.ndarray], y: Optional[np.ndarray]) -> tuple[float, float]:
        assert x is not None and y is not None

        def gradient(
            layers: list[Layer], xhat: list[np.ndarray], dloss: np.ndarray, result: list[tuple[np.ndarray, np.ndarray]]
        ) -> list[tuple[np.ndarray, np.ndarray]]:
            match layers:
                case []:
                    new_result = result
                case *head, tail:
                    dw, db, dloss_ = tail.optimize(xhat[-1], xhat[-2], dloss)
                    new_result = gradient(head, xhat[:-1], dloss_, [(dw, db), *result])
            return new_result

        *xhat, yhat = self.call(x, training=True)
        self.update_step(gradient(self.layers, [*xhat, yhat], self.loss_prime(y, yhat), []))
        loss, accuracy = (
            self.loss_func(y, yhat).mean(),
            self.loss_acc(y, yhat).mean(),
        )
        return loss, accuracy
