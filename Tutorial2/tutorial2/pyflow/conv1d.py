import numpy as np


def conv1d(input: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    n = input.shape[0]
    conv = lambda i: np.array(
        [
            input[i - 1] if i > 0 else 1.0,
            input[i],
            input[i + 1] if i < n - 1 else 1.0,
        ]
    )
    return np.array([np.sum(conv(i) * kernel) for i in range(n)])
