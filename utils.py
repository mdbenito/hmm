from functools import reduce
from time import time
import numpy as np


def is_row_stochastic(M: np.ndarray) -> bool:
    assert(M.ndim == 1 or M.ndim == 2)

    if M.ndim == 1:
        return np.isclose(M.sum(), 1.) and (M >= 0).all()
    elif M.ndim == 2:
        return np.allclose(M.sum(axis=1), np.ones(M.shape[0])) and (M >= 0).all()


class Timer:
    def __init__(self, msg: str = ""):
        self.msg = msg

    def __enter__(self):
        self.start = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time()
        print("{0} done in {1:>6.4}s".format(self.msg, self.end - self.start,))
