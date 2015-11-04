from functools import reduce
import numpy as np
import config
from data import Data
from utils import is_row_stochastic


class Model:
    """
    Model attributes:
        N = Number of states in the model
        Q = { q_0, ..., q_{N-1} } = states of the Markov process
        A = State transition matrix (NxN): A[i,j] = Prob(X_{t+1} = j | X_t = i)
        B = Observation probability matrix (NxM): B(q, y) = Prob(Y_t = y | X_t = q)
            TODO: check whether usage pattern for B makes the transpose a more sensible choice
        p = Prior distribution for the initial state

    Model parameters:
        alpha (TxN)
        beta (TxN)

    Configuration:
        max_iterations
    """

    # Hints for the IDE:
    A = np.ndarray((0, 0)); B = np.ndarray((0, 0)); p = np.ndarray((0,)); N = int()
    alpha = np.ndarray((0, 0)); beta = np.ndarray((0, 0)); gamma = np.ndarray((0, 0)); digamma = np.ndarray((0, 0, 0))

    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def init(d: Data, N: int=4) -> Model:
    """
    Initializes the model to random values.
    We must be careful not to use 1/N for initialization or we start at a local max. Instead we use 1/N ± ε

    TODO: compute best value for N

    """
    eps = 1e-2
    p = np.random.random((1, N))*eps + 1./N
    A = np.random.random((N, N))*eps + 1./N
    B = np.random.random((N, d.M))*eps + 1./d.M

    # Normalize probabilities (make row-stochastic)
    [p, A, B] = map(lambda M: M / M.sum(axis=1)[:, None], [p, A, B])

    return Model(N=N, p=p[0], A=A, B=B)  # BT=np.copy(B.T)


def alpha_pass(d: Data, m: Model) -> Model:
    """
    """

    # Compute α_0
    c = np.ndarray((d.L,))
    alpha = np.ndarray((d.L, m.N))
    alpha[0] = m.p * m.B[:, d.Y[0]]  # α_0(i) = π_i * P(Emission = Y[0] | State = i) = π_i * B(i, Y[0])
    # Using B^t:
    # alpha[0] = m.p * m.BT[d.Y[0], :]

    # Rescale α_0
    c[0] = 1. / alpha[0].sum()
    alpha[0] *= c[0]

    # Compute α_t, rescaling at each stage
    for t in range(1, d.L):
        alpha[t] = (alpha[t - 1] @ m.A) * m.B[:, d.Y[t]]
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
    beta = np.ndarray((d.L, m.N))
    e = np.ndarray((d.L, ))

    # Set β_{L-1}[i]=1*c[L-1]
    beta[d.L - 1].fill(1./d.L)
    e[d.L - 1] = 1. / d.L

    for t in range(d.L - 2, -1, -1):
        beta[t] = m.A @ (m.B[:, d.Y[t + 1]] * beta[t + 1])
        e[t] = 1. / beta[t].sum()
        beta[t] *= e[t]

    m.beta = beta
    return m


def gammas(d: Data, m: Model) -> Model:
    assert (hasattr(m, 'alpha') and hasattr(m, 'beta'))
    digamma = np.ndarray((d.L - 1, m.N, m.N))
    gamma = np.ndarray((d.L - 1, m.N))

    for t in range(0, d.L - 1):
        digamma[t] = m.alpha[t] * (m.A * m.B[:, d.Y[t+1]]) * m.beta[t+1].reshape(m.N, 1)
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
    for j in range(0, m.N):
        for k in range(0, d.M):
            for t in range(0, d.L - 1):
                m.B[j, k] += m.gamma[t, j] if d.Y[t] == k else 0.
    m.B /= m.gamma.sum(axis=0)[:, None]

    # Compute (log) likelihood of the observed emissions under the current model parameters
    m.ll = - np.log(m.c).sum()

    # Sanity check
    assert (is_row_stochastic(m.A) and is_row_stochastic(m.B) and is_row_stochastic(m.p))

    return m


def iterate(d: Data, m: Model=None, maxiter=10) -> Model:
    run = True
    ll = - np.inf
    it = 1
    print('Running (maxiter = ' + str(maxiter) + '):')
    if m is None:
        print('Initializing model...')
        m = init(d)
    while run:
        print('Iteration ' + str(it), end=', ', flush=True)
        m = reduce(lambda x, f: f(d, x), [alpha_pass, beta_pass, gammas, estimate], m)
        it += 1
        run = it <= maxiter and m.ll > ll and not np.isclose(m.ll, ll, atol=1e-10)
        ll = m.ll
        print('likelihood = ' + str(ll))
    return m


def viterbi_path(d: Data, m: Model) -> np.ndarray:
    """
    Returns the sequence of states maximizing the expected number of correct states.
    """

    path = np.ndarray((d.L, ))
    return path


def save(m: Model, filename: str = ''):
    return False

# def check_data(d: Data, m: Model=None) -> bool:
#     assert(d.Y.shape == (d.L,))
#     if m is not None:
#         assert(m.A.shape == (m.N, m.N))
#         assert(m.B.shape == (m.N, d.M))
#         assert(map(is_row_stochastic, [m.p, m.A, m.B]).all())
