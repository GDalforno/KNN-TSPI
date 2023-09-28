from typing import Callable, Literal

import numpy as np

from knn_tspi.constants import EPSILON
from knn_tspi.exceptions import WeightsCalculationException, InvalidMinimumKException
from knn_tspi.similarity import similarity_search
from knn_tspi.scores import z_score2


def predict(
    data: np.ndarray,
    query: np.ndarray,
    g: Callable[[np.ndarray], np.ndarray] | None,
    k: int,
    len_query: int,
    weights: Literal["uniform", "distance"],
) -> float:
    (indexes, min_dists) = similarity_search(data, query, k, len_query)
    min_k = calculate_min_k(indexes)
    indexes -= 1
    query_mean = np.mean(query)
    query_std = np.std(query)

    check_min_k(min_k)

    predictions = []
    for i in range(min_k):
        subseq = data[indexes[i] : indexes[i] + len_query + 1]
        subseq = z_score2(subseq, query)
        subseq = subseq * query_std + query_mean
        predictions.append(subseq[len_query])

    predictions_array = np.array(predictions)

    if weights == "uniform":
        return np.mean(predictions_array)
    else:
        return weighted_average(min_dists[:min_k], predictions_array, g)


def weighted_average(
    min_dists: np.ndarray,
    predictions: np.ndarray,
    g: Callable[[np.ndarray], np.ndarray] | None,
) -> float:
    if g is None:
        min_dists = np.where(min_dists == 0, EPSILON, min_dists)
        min_dists = 1 / min_dists
    else:
        try:
            min_dists = g(min_dists)
            min_dists = min_dists.reshape(-1)
        except Exception:
            raise WeightsCalculationException(
                "Error during weights calculation, please check the g function implementation!"  # noqa: E501
            )
    return np.sum(min_dists * predictions) / np.sum(min_dists)


def calculate_min_k(indexes: np.ndarray) -> int:
    return int(np.sum(np.mod(-indexes, indexes + 1)))


def check_min_k(min_k: int) -> None:
    if min_k == 0:
        raise InvalidMinimumKException(
            "min_k is zero, this usually means the"
            + " time series is too short and/or the query length is too big."
        )
