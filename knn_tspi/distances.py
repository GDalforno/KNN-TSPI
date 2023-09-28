import numpy as np

from knn_tspi.constants import EPSILON


def ED(s: np.ndarray, t: np.ndarray) -> float:
    return np.sqrt(np.sum((s - t) ** 2))


def CID(Q: np.ndarray, C: np.ndarray) -> float:
    CE_Q = np.sqrt(np.sum(np.diff(Q) ** 2)) + EPSILON
    CE_C = np.sqrt(np.sum(np.diff(C) ** 2)) + EPSILON
    return ED(Q, C) * (max(CE_Q, CE_C) / min(CE_Q, CE_C))


def distance(t: np.ndarray, s: np.ndarray) -> float:
    return CID(t, s)
