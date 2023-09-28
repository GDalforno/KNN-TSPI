from typing import Literal

import numpy as np

from knn_tspi.exceptions import (
    InvalidParameterException,
    ModelNotFittedException,
    InvalidDataException,
)


def validate_hyperparameters(
    k: int, len_query: int, weights: Literal["uniform", "distance"]
) -> None:
    if type(k) != int or k <= 0:
        raise InvalidParameterException(
            f"Number of neighbors must be an integer greater than 1, got {k}!"
        )
    if type(len_query) != int or len_query <= 2:
        raise InvalidParameterException(
            f"Query length must be an integer greater than 3, got {len_query}!"
        )
    if weights not in (
        "uniform",
        "distance",
    ):
        raise InvalidParameterException(
            f"Weights must be either uniform or distance, got {weights}!"
        )


def validate_fit(data: np.ndarray) -> None:
    if type(data) != np.ndarray:
        raise InvalidDataException("Data must be a numpy array!")
    if len(data.shape) > 1:
        raise InvalidDataException("Only 1-d arrays are supported!")
    if data.dtype.kind not in ("f", "i"):
        raise InvalidDataException("Data must be numerical!")
    if np.isnan(data).any():
        raise InvalidDataException("Data has missing values on it!")


def validate_predict(h: int, is_fitted: bool) -> None:
    if type(h) != int or h <= 0:
        raise InvalidParameterException(
            f"Horizon must be an integer greater than 0, got {h}!"
        )
    if not is_fitted:
        raise ModelNotFittedException("Model hasn't been fitted yet")
