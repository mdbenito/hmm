from functools import reduce
import numpy as np


def is_row_stochastic(M: np.ndarray) -> bool:
    assert(M.ndim == 1 or M.ndim == 2)

    if M.ndim == 1:
        return np.isclose(M.sum(), 1.) and (M >= 0).all()
    elif M.ndim == 2:
        return np.allclose(M.sum(axis=1), [1. for x in range(0, M.shape[0])]) and (M >= 0).all()


def doto (val, *funs):
    reduce(lambda x, f: f(x), funs, val)
