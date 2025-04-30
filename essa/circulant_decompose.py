import numpy as np
from scipy.linalg import circulant, svd
from typing import List

class CirculantDecompose:
  def __init__(self, time_series: np.ndarray, window_size: int) -> None:
    self.time_series = time_series
    self.ts_size = len(time_series)
    self.window_size = window_size

  def _trajectory_matrix(self) -> np.ndarray:
    base_col = np.concatenate([self.time_series, np.zeros(self.window_size - 1)])
    return circulant(base_col)[:self.window_size, :self.ts_size]

  def fit(self) -> None:
    X = self._trajectory_matrix()
    U, s, Vt = svd(X)
    self.U, self.sigma, self.V = U, s, Vt.T
    self.d = np.linalg.matrix_rank(X)
    self.components = [s[i] * np.outer(U[:, i], self.V[:, i]) for i in range(self.d)]
