import numpy as np


def conv1d(input, kernel: np.ndarray) -> np.ndarray:
    N = len(input)
    result = []
    for n in range(N):
        conv = np.array(
            [
                input[n - 1] if n > 0 else 1.0,
                input[n],
                input[n + 1] if n < N - 1 else 1.0,
            ]
        )
        result.append(np.sum(conv * kernel))
    return np.array(result)