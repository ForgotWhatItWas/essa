import numpy as np
from scipy.linalg import svd
from typing import List

class AntiHankelDecompose:
    def __init__(self, time_series: np.ndarray, window_size: int):
        self.time_series = time_series
        self.window_size = window_size
        self.ts_size = len(time_series)

    def _trajectory_matrix(self) -> np.ndarray:
        K = self.ts_size - self.window_size + 1
        X = np.zeros((self.window_size, K))
        for i in range(self.window_size):
            X[i, :] = self.time_series[K - i - 1:K - i - 1 + K]
        return X

    def fit(self) -> None:
        X = self._trajectory_matrix()
        U, s, Vt = svd(X)
        self.U, self.sigma, self.V = U, s, Vt.T
        self.d = np.linalg.matrix_rank(X)
        self.components = [s[i] * np.outer(U[:, i], self.V[:, i]) for i in range(self.d)]
