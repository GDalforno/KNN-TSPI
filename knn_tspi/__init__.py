from typing import Any, Callable, Dict, Literal

import numpy as np

from knn_tspi.bootstrap import bootstrap_residuals, calculate_residuals
from knn_tspi.predict import predict
from knn_tspi.validation import validate_hyperparameters, validate_fit, validate_predict

__version__ = "1.0.0"


class KNeighborsTSPI:
    """K-Nearest Neighbors Time Series Prediction with Invariances

    Parameters
    ----------
    k : int
        Number of neighbors.

    len_query : int
        Subsequence length.

    weights : str
        Weight function using in predicion

    References
    ----------
    .. [1] G. E. A. P. A. Batista, A. R. S. Parmezan,
           ``A Study of the Use of Complexity Measures in the Similarity
           Search Process Adopted by kNN Algorithm for Time Series Prediction´´,
           2015, Instituto de Ciências Matemáticas e de Computação,
           Universidade de São Paulo, São Carlos.

    Examples
    --------
    >>> import numpy as np
    >>> from knn_tspi import KNeighborsTSPI
    >>> ts = 0.5*np.arange(60) + np.random.randn(60)
    >>> model = KNeighborsTSPI()
    >>> model.fit(data=ts)
    >>> y = model.predict(h=5)
    """

    def __init__(
        self,
        k: int = 3,
        len_query: int = 4,
        weights: Literal["uniform", "distance"] = "uniform",
    ):
        validate_hyperparameters(k, len_query, weights)
        self.__k = k
        self.__len_query = len_query
        self.__weights = weights

        self.__data = np.zeros(1)
        self.__is_fitted = False

    def fit(self, data: np.ndarray) -> None:
        """
        Parameters
        ----------
        data : array-like of shape (n_observations,)
               Time series
        """
        validate_fit(data)
        self.__data = data
        self.__is_fitted = True

    def predict(
        self, h: int, g: Callable[[np.ndarray], np.ndarray] | None = None
    ) -> Dict[str, np.ndarray]:
        """
        Parameters
        ----------
        h : int
            Forecast horizon

        g : Callable[[np.ndarray], np.ndarray] | Nonw
            User defined function that takes an array of distances and
            return its corresponding weights. Default is the inverse of
            distance. Ignored if weights!="distance"

        Returns
        -------
        predictions : array-like of shape (h,)
        """
        validate_predict(h, self.__is_fitted)

        queue = self.__data[-self.__len_query :]

        predictions = []
        for _ in range(h):
            f = predict(
                self.__data, queue, g, self.__k, self.__len_query, self.__weights
            )
            predictions.append(f)
            queue = np.delete(queue, 0)
            queue = np.append(queue, [f])

        return {"mean": np.array(predictions)}

    def predict_interval(
        self, h: int, g: Callable[[np.ndarray], np.ndarray] | None = None
    ) -> Dict[str, np.ndarray]:
        """
        Parameters
        ----------
        h : int
            Forecast horizon

        g : Callable[[np.ndarray], np.ndarray] | None
            User defined function that takes an array of distances and
            return its corresponding weights. Default is the inverse of
            distance. Ignored if weights!="distance"

        Returns
        -------
        predictions : array-like of shape (h,)
        """
        validate_predict(h, self.__is_fitted)

        residuals = calculate_residuals(
            self.__data, self.__k, self.__len_query, self.__weights, g
        )
        y = self.predict(h, g)["mean"]
        futures = bootstrap_residuals(y, residuals, h)
        intervals = np.quantile(futures, q=[0.2, 0.8, 0.15, 0.95], axis=0)

        return {
            "low_95": intervals[2, :],
            "low_80": intervals[0, :],
            "mean": y,
            "high_80": intervals[1, :],
            "high_95": intervals[3, :],
        }

    def get_params(self) -> Dict[str, Any]:
        """
        Parameters
        ----------

        Returns
        -------
        parameters : dict
                     Parameter names mapped to their values.
        """
        return {"k": self.__k, "len_query": self.__len_query, "weights": self.__weights}

    def is_fitted(self) -> bool:
        """
        Parameters
        ----------

        Returns
        -------
        is_fitted : bool
                    wheater the model is fitted or not
        """
        return self.__is_fitted

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(k={self.__k}, len_query={self.__len_query}, weights="{self.__weights}")'  # noqa: E501
