from functools import reduce
import numpy as np
from data import Data
from utils import is_row_stochastic
from time import time
import config


class Model:
    """
    Model attributes and data:
        N = Number of states in the model
        X = [X_0, ..., X_{L-1}] = chain of states
        Y = [Y_0, ..., Y_{L-1}] = observed emissions

    Model parameters:
        A = State transition matrix (NxN): A[i,j] = Prob(X_{t+1} = j | X_t = i)
        B = Observation probability matrix (NxM): B(j, k) = Prob(Y_t = k | X_t = j)
            TODO: check whether usage pattern for B makes the transpose a more sensible choice
        p = Prior distribution for the initial state

    Internal data:
        alpha (LxN)
        beta (LxN)
        gamma (LxN)
        digamma(LxNxN)

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
    TODO: compute best value for N

    :param N: Number of states in the model
    :param d: Data object
    """
    # We must be careful not to use 1/N for initialization or we start at a local max.
    # Instead we use 1/N ± ε(N), with ε(N)~U(-1/(2N),1/(2N))
    p = (0.5 + np.random.random((1, N)))/N
    A = (0.5 + np.random.random((N, N)))/N
    B = (0.5 + np.random.random((N, d.M)))/d.M

    # Normalize probabilities (make row-stochastic)
    [p, A, B] = map(lambda M: M / M.sum(axis=1)[:, None], [p, A, B])

    return Model(N=N, p=p[0], A=A, B=B,
                 alpha=np.ndarray((d.L, N)),
                 c=np.ndarray((d.L,)),  # Scaling for α (and β if not concurrently run) / computation of log likelihood
                 beta=np.ndarray((d.L, N)),
                 e=np.ndarray((d.L, )),  # Scaling of β (used if alpha_pass and beta_pass are run concurrently)
                 gamma=np.ndarray((d.L - 1, N)), digamma=np.ndarray((d.L - 1, N, N)))


def alpha_pass(d: Data, m: Model) -> Model:
    """
    Computes
        α_t(i) = α(t, i) = P(X_t = i, Y_0 = y_0, ..., Y_t = y_t)
    """

    # Compute α_0
    m.alpha[0] = m.p * m.B[:, d.Y[0]]  # α_0(i) = π_i * P(Emission = Y[0] | State = i) = π_i * B(i, Y[0])

    # Rescale α_0
    m.c[0] = 1. / m.alpha[0].sum()
    m.alpha[0] *= m.c[0]

    # Compute α_t, rescaling at each stage
    for t in range(1, d.L):
        m.alpha[t] = (m.alpha[t - 1] @ m.A) * m.B[:, d.Y[t]]
        m.c[t] = 1. / m.alpha[t].sum()
        m.alpha[t] *= m.c[t]

    return m


def beta_pass(d: Data, m: Model) -> Model:
    """
    Computes
        β_t(i) = β(t, i) = P(Y_{t+1} = y_{t+1}, ..., Y_{L-1} = y_{l-1} | X_t = i)

    TODO: Rescaling can be done with a new variable instead of using m.c, in order to enable parallel execution.
    """
    # assert(hasattr(m, 'c'))

    # Set β_{L-1}[i]=1*c[L-1]
    m.beta[d.L-1] = m.c[d.L-1]

    # m.beta[d.L - 1].fill(1./d.L)
    # m.e[d.L - 1] = 1. / d.L

    for t in range(d.L - 2, -1, -1):
        # m.beta[t] = m.A @ (m.B[:, d.Y[t + 1]] * m.beta[t + 1])
        m.beta[t] = 0.0
        for i in range(m.N):
            m.beta[t, i] += (m.A.T[:, i] * m.B.T[d.Y[t + 1], :] * m.beta[t + 1]).sum(axis=0)
        m.beta[t] *= m.c[t]
        # m.e[t] = 1. / m.beta[t].sum()
        # m.beta[t] *= m.e[t]

    return m


def gammas(d: Data, m: Model) -> Model:
    """
    Computes
        ɣ(t, i)    = P(X_t = i | Y_0, ..., Y_{L-1})
        ɣ(t, i, j) = P(X_t = i, X_{t+1} = j | Y_0, ..., Y_{L-1})
    """
    assert (hasattr(m, 'alpha') and hasattr(m, 'beta'))

    V = m.B.T[d.Y[1:]] * m.beta[1:]
    for t in range(d.L - 1):
        m.digamma[t] = m.alpha[t].reshape((m.N, 1)) * (m.A * V[t])

    # FIXME: why does each matrix digamma[t,·, ·] sum to one?
    assert (np.allclose(1.0, m.digamma.sum(axis=(1, 2))))  # Just in case it's not always true...
    # m.digamma /= m.digamma.sum(axis=(1, 2)).reshape(d.L-1, 1, 1)
    m.gamma = m.digamma.sum(axis=2)

    return m


def estimate(d: Data, m: Model) -> Model:
    assert (hasattr(m, 'gamma') and hasattr(m, 'digamma'))

    # \sum_{t=0}^{L-2} ɣ(t, i) = Expected number of transitions made *from* state i
    e_transitions_from = m.gamma[:-1, :].sum(axis=0)
    # \sum_{t=0}^{L-1} ɣ(t, i) = Expected number of times that state i is visited
    e_visited = e_transitions_from + m.gamma[-1, :]

    # Re-estimate π
    m.p = np.copy(m.gamma[0].reshape(1, m.N))

    # Re-estimate transition matrix A FIXME: this is wrong!
    m.A = m.digamma[:-1, :, :].sum(axis=0) / e_transitions_from.reshape((m.N, 1))
    # for i in range(m.N):
    #     for j in range(m.N):
    #         num = 0.0
    #         den = 0.0
    #         for t in range(d.L - 1):
    #             num += m.digamma[t, i, j]
    #             den += m.gamma[t, i]
    #         m.A[i, j] = num / den

    # Re-estimate emission matrix B FIXME! This is going to be sloooooow!
    m.B.fill(0.)
    for j in range(m.N):
        for k in range(d.M):
            # for t in range(0, d.L - 1):
            #    m.B[j, k] += m.gamma[t, j] if d.Y[t] == k else 0.  # FIXME: test performance wrt line below
            m.B[j, k] += m.gamma[d.Y[:-1] == k, j].sum(axis=0)

    m.B /= e_visited.reshape((m.N, 1))
    # m.B /= m.B.sum(axis=1).reshape((m.N, 1))  # This should be equivalent to  /= e_visited, and it's O(N) as well.

    # Compute (log) likelihood of the observed emissions under the current model parameters
    m.ll = - np.log(m.c).sum()

    return m


def iterate(d: Data, m: Model=None, maxiter=10, eps=config.iteration_margin, verbose=False) -> Model:
    run = True
    ll = - np.inf
    it = 1
    if verbose: print('\nRunning up to maxiter = {0} with threshold {1}%:'.format(maxiter, eps))
    if m is None:
        if verbose: print('Initializing model...')
        m = init(d)
    start = time()
    while run:
        m = reduce(lambda x, f: f(d, x), [alpha_pass, beta_pass, gammas, estimate], m)
        it += 1
        delta = 100.0 * (m.ll - ll) / np.abs(ll) if ll > -np.inf else np.inf
        run = it <= maxiter and delta >= eps
        ll = m.ll
        if it % 10 == 0:
            end = time()
            print("Iteration {} finishes after {:.4}s with an increase of {:.3}% in the likelihood".
                  format(it, end - start, delta))
            start = time()
    return m


def viterbi_path(d: Data, m: Model) -> np.ndarray:
    """
    TODO: Returns the sequence of states maximizing the expected number of correct states.
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
