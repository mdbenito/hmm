from functools import reduce
import numpy as np
from data import Data


# TODO: remove me or move to test_inference
def is_row_stochastic(M: np.ndarray) -> bool:
    assert (M.ndim == 2)
    return np.allclose(M.sum(axis=1), (1. for x in range(0, M.shape[0])))


class Model:
    """
    Model attributes:
        N = Number of states in the model
        Q = { q_0, ..., q_{N-1} } = states of the Markov process
        A = State transition matrix (NxN): A[i,j] = Prob(X_{t+1} = j | X_t = i)
        B = Observation probability matrix (NxM): B(q, y) = Prob(Y_t = y | X_t = q)
            TODO: check whether usage pattern for B makes the transpose a more sensible choice
        C = B^t @ A^t. Can be used for the computation of α_t, see TODO above.
        p = Prior distribution for the initial state

    Model parameters:
        alpha (TxN)
        beta (TxN)

    Configuration:
        max_iterations
    """

    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def init(d: Data) -> Model:
    N = 4  # TODO: compute best value for N
    # TODO: find better initialization values for p, A, B
    #       (e.g. 1/N±ε (but careful not to use 1/N or we start at a local max)
    p = np.array(size=N)
    A = np.random.random((N, N))
    B = np.random.random((N, d.M))
    [p, A, B] = map(lambda M: M / M.sum(axis=1)[:, None], [p, A, B])

    return Model(p=p, A=A, B=B, BT=np.copy(B.T), C=B.T @ A.T)


def alpha_pass(d: Data, m: Model) -> Model:
    """
    """

    # Compute α_0
    c = np.array(size=d.L)
    alpha = np.ndarray(shape=(d.L, m.N))
    alpha[0] = m.p @ m.B[:, d.Y[0]]  # α_0(i) = π_i * P(Emission = Y[0] | State = i) = π_i * B(i, Y[0])
    # Using B^t:
    # alpha[0] = m.p @ m.BT[d.Y[0], :]

    # Rescale α_0
    c[0] = 1. / alpha[0].sum()
    alpha[0] *= c[0]

    # Compute α_t, rescaling at each stage
    for t in range(1, d.L):
        alpha[t] = m.C[d.Y[t], :] @ alpha[t - 1]  # FIXME?
        c[t] = 1. / alpha[t].sum()
        alpha[t] *= c[t]

    m.alpha = alpha  # Should I copy?

    # Store scaling for use in beta_pass / computation of log likelihood of observations
    m.c = c
    return m


def beta_pass(d: Data, m: Model) -> Model:
    """
    Computes β.

    Rescaling is done with a new variable instead of using m.c, in order to enable parallel execution.
    """
    # assert(hasattr(m, 'c'))
    beta = np.ndarray(shape=(d.L, m.N))
    e = np.array(size=d.L)

    # Set β_{L-1}[i]=1*c[L-1]
    beta[d.L - 1].fill(1.)
    e[d.L - 1] = 1. / beta[d.L - 1].sum()
    beta[d.L - 1] *= e[d.L - 1]

    for t in range(d.L - 1, 0, -1):
        beta[t] = (m.C[t + 1, :] @ beta[t + 1])  # FIXME?
        e[t] = 1. / beta[t].sum()
        beta[t] *= e[t]

    m.beta = beta
    return m


def gammas(d: Data, m: Model) -> Model:
    assert (hasattr(m, 'alpha') and hasattr(m, 'beta'))
    digamma = np.ndarray(shape=(d.L - 2, m.N, m.N))
    gamma = np.array(size=d.L - 2)

    for t in range(0, d.L - 2):
        digamma[t] = m.alpha[t] @ m.C[t + 1, :] @ m.beta[t + 1]  # FIXME?
        digamma[t] /= digamma[t].sum()
        gamma[t] = digamma[t].sum(axis=1)

    m.digamma = digamma
    m.gamma = gamma
    return m


def estimate(d: Data, m: Model) -> Model:
    assert (hasattr(m, 'gamma') and hasattr(m, 'digamma'))

    # Re-estimate π
    m.p = np.copy(m.gamma[0])

    # Re-estimate transition matrix A
    m.A = m.digamma.sum(axis=0) / m.gamma.sum(axis=0)[:, None]

    # Re-estimate emission matrix B FIXME! This is going to be sloooooow!
    m.B.fill(0.)
    for j in range(0, d.N):
        for k in range(0, d.M):
            for t in range(0, d.L - 2):
                m.B[j, k] += m.gamma[t, j] if d.Y[t] == k else 0.
    m.B /= m.gamma.sum(axis=0)[:, None]

    # Compute (log) likelihood of the observed emissions under the current model parameters
    m.ll = - np.log(m.c).sum()

    # Sanity checks:
    assert (is_row_stochastic(m.A) and is_row_stochastic(m.B) and is_row_stochastic(m.p))

    return m


def iterate(d: Data, maxiter=10) -> Model:
    run = True
    ll = 0.
    it = 1
    print('Initializing model...')
    m = init(d)
    while run:
        print('Running iteration ' + it + ':')
        print('=====================')
        reduce(lambda x, f: f(d, x), [alpha_pass, beta_pass, gammas, estimate], m)
        it += 1
        run = it <= maxiter and m.ll > ll
        ll = m.ll
        print('Computed likelihood for the observations: ' + ll)


def viterbi_path(d: Data, m: Model) -> np.array:
    """
    Returns the sequence of states maximizing the expected number of correct states.
    """

    path = np.array(d.L)
    return path
