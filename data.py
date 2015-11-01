#############################################################################
# Data R/W and synthetic datasets
#
#############################################################################
import numpy as np
from h5py import File


# Dumb containers to have dot notation instead of the cumbersome dictionary
# notation. (Python Cookbook §4.8)
class Data:
    """
    Contains:
        L = Number of observations
        M = Number of symbols which may be observed
        V = { 0, 1, ..., M-1} = possible observations
        Y = {Y_0, ..., Y_{L-1} } = sequence of observations

    """

    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def load(filename: str = ''):
    with File(filename, 'r') as f:
        [M, L, Y] = do_stuff()  # ...
    return Data(M=M, L=L, Y=Y)


def save(filename: str = ''):
    return False


def generate(N=4, M=10, L=1000) -> [Data, dict]:
    """
        N = Number of states
        M = Number of possible emissions
        L = Number of emissions generated
    """
    p = np.ndarray((1, N))
    A = np.random.random((N, N))
    B = np.random.random((N, M))
    Y = np.ndarray((L, ))
    [p, A, B] = map(lambda X: X / X.sum(axis=1)[:, None], [p, A, B])

    q = np.random.choice(N, p=p[0, :])  # Initial state
    for t in range(1, L):
        Y[t] = np.random.choice(M, p=B[q])  # Emission
        q = np.random.choice(N, p=A[q])  # Jump to next state

    return [Data(M=M, L=L, Y=Y), {'p': p[0], 'A': A, 'B': B}]
