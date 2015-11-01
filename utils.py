import numpy as np


def is_row_stochastic(M: np.ndarray) -> bool:
    assert (M.ndim == 2)
    return np.allclose(M.sum(axis=1), (1. for x in range(0, M.shape[0])))
