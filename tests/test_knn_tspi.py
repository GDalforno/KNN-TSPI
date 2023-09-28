import numpy as np

import pytest

from knn_tspi import KNeighborsTSPI
from knn_tspi.exceptions import (
    InvalidParameterException,
    InvalidDataException,
    ModelNotFittedException,
    InvalidMinimumKException,
    DistanceIsNaNException,
    WeightsCalculationException,
    InvalidRollingWindowUpperBoundException,
)


class TestKNeighborsTSPI:
    def test_ensure_not_initialize_if_k_is_invalid(self) -> None:
        k = -1
        with pytest.raises(InvalidParameterException):
            KNeighborsTSPI(k)

    def test_ensure_not_initialize_if_len_query_is_invalid(self) -> None:
        k = 3
        len_query = "len_query"
        with pytest.raises(InvalidParameterException):
            KNeighborsTSPI(k, len_query)

    def test_ensure_not_initialize_if_weights_is_invalid(self) -> None:
        k = 3
        len_query = 4
        weights = "invalid"
        with pytest.raises(InvalidParameterException):
            KNeighborsTSPI(k, len_query, weights)

    def test_ensure_params_are_set_correctly(self) -> None:
        k = 3
        len_query = 4
        weights = "distance"
        model = KNeighborsTSPI(k, len_query, weights)

        params = model.get_params()

        assert params.get("k") == k
        assert params.get("len_query") == len_query
        assert params.get("weights") == weights

    def test_ensure_model_does_not_fit_if_data_format_is_invalid(self) -> None:
        k = 3
        len_query = 4
        weights = "distance"
        model = KNeighborsTSPI(k, len_query, weights)

        data = "data"

        with pytest.raises(InvalidDataException):
            model.fit(data)

    def test_ensure_model_does_not_fit_if_data_dimension_is_invalid(self) -> None:
        k = 3
        len_query = 4
        weights = "distance"
        model = KNeighborsTSPI(k, len_query, weights)

        data = np.array([[1, 2], [3, 4]])

        with pytest.raises(InvalidDataException):
            model.fit(data)

    def test_ensure_model_does_not_fit_if_data_type_is_invalid(self) -> None:
        k = 3
        len_query = 4
        weights = "distance"
        model = KNeighborsTSPI(k, len_query, weights)

        data = np.array(["1"])

        with pytest.raises(InvalidDataException):
            model.fit(data)

    def test_ensure_model_does_not_fit_if_data_has_missing_values(self) -> None:
        k = 3
        len_query = 4
        weights = "distance"
        model = KNeighborsTSPI(k, len_query, weights)

        data = np.array([1, np.nan])

        with pytest.raises(InvalidDataException):
            model.fit(data)

    def test_ensure_model_fits(self) -> None:
        k = 3
        len_query = 4
        weights = "distance"
        model = KNeighborsTSPI(k, len_query, weights)

        data = np.array([1, 2, 3, 4])

        model.fit(data)

        assert model.is_fitted()

    def test_ensure_model_does_not_predict_if_not_fitted(self) -> None:
        k = 3
        len_query = 4
        weights = "distance"
        model = KNeighborsTSPI(k, len_query, weights)

        h = 1
        with pytest.raises(ModelNotFittedException):
            model.predict(h)

    def test_ensure_model_does_not_predict_if_horizon_is_invalid(self) -> None:
        k = 3
        len_query = 4
        weights = "distance"
        model = KNeighborsTSPI(k, len_query, weights)

        data = np.array([1, 2, 3, 4])
        model.fit(data)

        h = ["1"]
        with pytest.raises(InvalidParameterException):
            model.predict(h)

    def test_ensure_model_does_not_predict_if_data_is_too_short(self) -> None:
        k = 3
        len_query = 3
        weights = "uniform"
        model = KNeighborsTSPI(k, len_query, weights)

        data = np.array([1, 2])
        model.fit(data)

        h = 1
        with pytest.raises(InvalidMinimumKException):
            model.predict(h)

    def test_ensure_model_does_not_predict_if_distance_goes_to_infinity(self) -> None:
        k = 3
        len_query = 3
        weights = "uniform"
        model = KNeighborsTSPI(k, len_query, weights)

        data = np.array([1, 2, np.inf, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        model.fit(data)

        h = 1
        with pytest.raises(DistanceIsNaNException):
            model.predict(h)

    def test_ensure_model_does_not_predict_if_g_function_is_invalid(self) -> None:
        k = 3
        len_query = 3
        weights = "distance"
        model = KNeighborsTSPI(k, len_query, weights)

        data = np.random.rand(100)
        model.fit(data)

        def g(_: np.ndarray) -> np.ndarray:
            raise Exception("client_side_error")

        h = 1
        with pytest.raises(WeightsCalculationException):
            model.predict(h, g)

    def test_ensure_model_predicts_correctly_with_uniform_weights(self) -> None:
        k = 3
        len_query = 3
        weights = "uniform"
        model = KNeighborsTSPI(k, len_query, weights)

        data = np.random.rand(100)
        model.fit(data)

        h = 5

        y = model.predict(h)

        assert type(y["mean"]) == np.ndarray
        assert y["mean"].shape == (h,)
        assert y["mean"].dtype == np.float64
        assert not np.isnan(y["mean"]).any()

    def test_ensure_model_predicts_correctly_with_distance_weights(self) -> None:
        k = 3
        len_query = 3
        weights = "distance"
        model = KNeighborsTSPI(k, len_query, weights)

        data = np.random.rand(100)
        model.fit(data)

        h = 5

        def g(x):
            return np.exp(-(x**2))

        y = model.predict(h, g)

        assert type(y["mean"]) == np.ndarray
        assert y["mean"].shape == (h,)
        assert y["mean"].dtype == np.float64
        assert not np.isnan(y["mean"]).any()

    def test_ensure_model_does_not_predict_interval_if_data_is_too_short(self) -> None:
        k = 3
        len_query = 3
        weights = "uniform"
        model = KNeighborsTSPI(k, len_query, weights)

        data = np.random.rand(9)
        model.fit(data)

        h = 5

        with pytest.raises(InvalidRollingWindowUpperBoundException):
            model.predict_interval(h)

    def test_ensure_model_predicts_interval_correctly(self) -> None:
        k = 3
        len_query = 3
        weights = "uniform"
        model = KNeighborsTSPI(k, len_query, weights)

        data = np.random.rand(100)
        model.fit(data)

        h = 5

        y = model.predict_interval(h)

        assert type(y["low_95"]) == np.ndarray
        assert y["low_95"].shape == (h,)
        assert y["low_95"].dtype == np.float64
        assert not np.isnan(y["low_95"]).any()

        assert type(y["low_80"]) == np.ndarray
        assert y["low_80"].shape == (h,)
        assert y["low_80"].dtype == np.float64
        assert not np.isnan(y["low_80"]).any()

        assert type(y["mean"]) == np.ndarray
        assert y["mean"].shape == (h,)
        assert y["mean"].dtype == np.float64
        assert not np.isnan(y["mean"]).any()

        assert type(y["high_80"]) == np.ndarray
        assert y["high_80"].shape == (h,)
        assert y["high_80"].dtype == np.float64
        assert not np.isnan(y["high_80"]).any()

        assert type(y["high_95"]) == np.ndarray
        assert y["high_95"].shape == (h,)
        assert y["high_95"].dtype == np.float64
        assert not np.isnan(y["high_95"]).any()

        assert (y["low_95"] <= y["low_80"]).all()
        assert (y["low_80"] <= y["mean"]).all()
        assert (y["mean"] <= y["high_80"]).all()
        assert (y["high_80"] <= y["high_95"]).all()
