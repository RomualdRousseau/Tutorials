from typing import Optional

import numpy as np

from tutorial2.pyflow.core import Model
from tutorial2.pyflow.sequential import Sequential


def train(model: Model, x: Optional[np.ndarray], y: Optional[np.ndarray]) -> Optional[np.ndarray]:
    assert isinstance(model, Sequential)
    assert x is not None
    assert y is not None

    output = model.call(x, training=True)
    yhat = output[-1]
    loss = model.loss_prime(y, yhat)

    gradients: list[tuple[np.ndarray, np.ndarray]] = []
    for i, lr in enumerate(reversed(model.layers), 1):
        dw, db, loss = lr.backward(output[-i], output[-(i + 1)], loss)
        gradients = [(dw, db), *gradients]

    for i, lr in enumerate(model.layers):
        lr.update_gradients(gradients[i], model.optimizer_func)

    return yhat
