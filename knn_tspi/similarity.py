from typing import Tuple

import numpy as np

from knn_tspi.constants import INFINITY
from knn_tspi.exceptions import DistanceIsNaNException
from knn_tspi.scores import z_score1
from knn_tspi.distances import distance


def similarity_search(
    data: np.ndarray, query: np.ndarray, k: int, len_query: int
) -> Tuple[np.ndarray, np.ndarray]:
    query = z_score1(query)

    min_dists = create_array_of_infinities(k)
    indexes = create_array_of_zeros(k)
    upper_bound = calculate_upper_bound(data, len_query)

    for i in range(k):
        for j in range(upper_bound):
            if not trivial_match(len_query, j, indexes, i):
                subseq = data[j : j + len_query]
                subseq = z_score1(subseq)
                d = distance(subseq, query)
                check_distance(d)

                if d < min_dists[i]:
                    min_dists[i] = d
                    indexes[i] = j + 1

    return (indexes.astype(int), min_dists)


def trivial_match(len_query: int, pos: int, indexes: np.ndarray, inc: int) -> bool:
    tm = False
    for i in range(inc):
        if np.abs(pos - indexes[i]) <= len_query:
            tm = True
            break
    return tm


def create_array_of_infinities(k: int) -> np.ndarray:
    return np.ones(k) * INFINITY


def create_array_of_zeros(k: int) -> np.ndarray:
    return np.zeros(k)


def check_distance(d: float) -> None:
    if np.isnan(d) or np.isinf(d):
        raise DistanceIsNaNException(
            f"Distance is not a number, got {d}! Please, check your data and model hyperparameters"  # noqa: E501
        )


def calculate_upper_bound(data: np.ndarray, len_query: int) -> int:
    return data.shape[0] - (2 * len_query) - 1
