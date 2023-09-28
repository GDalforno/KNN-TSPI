import numpy as np


def z_score1(s: np.ndarray) -> np.ndarray:
    mean, std = np.mean(s), np.std(s)
    return s - mean if std == 0 else (s - mean) / std


def z_score2(s: np.ndarray, q: np.ndarray) -> np.ndarray:
    s1 = s[:-1]
    if np.var(s1) > np.var(q):
        m = np.mean(s1)
        d = np.std(s1)
    else:
        m = np.mean(s)
        d = np.std(s)
    return s - m if d == 0 else (s - m) / d
