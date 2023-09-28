from typing import Callable, Generator, Literal, List, Tuple

import numpy as np

from knn_tspi.constants import MINIMUM_DATA_SIZE_SIZE_FOR_ROLLING_WINDOW
from knn_tspi.exceptions import InvalidRollingWindowUpperBoundException
from knn_tspi.predict import predict


def calculate_residuals(
    data: np.ndarray,
    k: int,
    len_query: int,
    weights: Literal["uniform", "distance"],
    g: Callable[[np.ndarray], np.ndarray] | None,
) -> np.ndarray:
    residuals = []
    for train_idxs, test_idxs in rolling_window(
        data, MINIMUM_DATA_SIZE_SIZE_FOR_ROLLING_WINDOW * len_query
    ):
        training_data = data[train_idxs]
        validation_data = data[test_idxs]
        queue = training_data[-len_query:]
        f = predict(data, queue, g, k, len_query, weights)
        residual = validation_data[0] - f
        residuals.append(residual)
    return np.array(residuals)


def bootstrap_residuals(
    y: np.ndarray, residuals: np.ndarray, h: int, n_iter: int = 250
):
    return np.array([y + np.random.choice(residuals, h) for _ in range(n_iter)])


def rolling_window(
    data: np.ndarray, initial: int
) -> Generator[Tuple[List[int], List[int]], None, None]:
    upper_bound = len(data) - initial
    if upper_bound == 0:
        raise InvalidRollingWindowUpperBoundException(
            f"Data must have more than {initial} observations to bootstrap residuals!"
        )
    for i in range(upper_bound):
        yield [j for j in range(i, i + initial)], [i + initial]
