import numpy as np
# from h5py import File


# Dumb containers to have dot notation instead of the cumbersome dictionary
# notation. (Python Cookbook §4.8)
# FIXME Look up class "bunch"
class Data:
    """
    Contains:
        L = Number of observations
        M = Number of symbols which may be observed
        V = { 0, 1, ..., M-1} = possible observations (TODO)
        Y = {Y_0, ..., Y_{L-1} } = sequence of observations
        generator = dictionary with model parameters if the data is synthetic
    """

    # Hints for the IDE:
    L = int()
    M = int()
    Y = np.ndarray((0, 0), dtype='i2')
    generator = {}

    def __init__(self, **keywords):
        self.__dict__.update(keywords)


# def load(filename: str = ''):
#     with File(filename, 'r') as f:
#         [M, L, Y] = do_stuff()  # ...
#     return Data(M=M, L=L, Y=Y)


def generate(N: int=4, M: int=10, L: int=1000, p=None, A=None, B=None) -> Data:
    """
    Constructs a random model and generates data using it.
    Returns a Data object including a special dictionary containing the model
    parameters for verification.

    :param N: Number of states
    :param M: Number of possible emissions
    :param L: Number of emissions generated
    """

    if p is None:
        p = np.random.random((1, N))  # Careful, this needs reshaping/extraction
    else:
        p = p.reshape((1, N))
    if A is None:
        A = np.random.random((N, N))
    if B is None:
        B = np.random.random((N, M))
    assert (p.shape == (1, N) and A.shape == (N, N) and B.shape == (N, M)), \
        "Wrong shape for initial parameters (N = {0}, M = {1}):\n" \
        "p:{2}\nA:{3}\nB:{4}".format(N, M, p.shape, A.shape, B.shape)

    Y = np.ndarray((L, ), dtype='i2')
    Q = np.ndarray((L, ), dtype='i2')
    # Normalize probabilities (make row-stochastic)
    [p, A, B] = map(lambda X: X / X.sum(axis=1)[:, None], [p, A, B])

    Q[0] = np.random.choice(N, p=p[0, :])  # Initial state, sampled from prior
    Y[0] = np.random.choice(M, p=B[Q[0]])  # Emission
    for t in range(1, L):
        Q[t] = np.random.choice(N, p=A[Q[t-1]])  # Jump to next state
        Y[t] = np.random.choice(M, p=B[Q[t]])    # Emmit

    return Data(M=M, L=L, Y=Y,
                generator={'N': N, 'p': p[0], 'A': A, 'B': B, 'Q': Q})
