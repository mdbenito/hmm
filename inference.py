from functools import reduce
from data import Data
from time import time
import numpy as np
import config


# For the type hints
Scalar = np.float64
Array = np.ndarray


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
        xi(LxNxN)

    Configuration:
        max_iterations
    """

    # Hints for the IDE:
    A = np.ndarray((0, 0))
    B = np.ndarray((0, 0))
    p = np.ndarray((0,))
    N = int()
    alpha = np.ndarray((0, 0))
    beta = np.ndarray((0, 0))
    gamma = np.ndarray((0, 0))
    xi = np.ndarray((0, 0, 0))
    c = np.ndarray((0, 0))

    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def init(d: Data, N: int=4) -> Model:
    """
    Initializes the model to random values.
    TODO: compute best value for N

    :param N: Number of states in the model
    :param d: Data object
    """

    p = np.random.random((1, N))
    A = np.random.random((N, N))
    B = np.random.random((N, d.M))

    # Normalize probabilities (make row-stochastic)
    [p, A, B] = map(lambda M: M / M.sum(axis=1)[:, None], [p, A, B])

    return Model(N=N, p=p.reshape((N,)), A=A, B=B,
                 alpha=np.ndarray((d.L, N)),
                 # Scaling for α (and β if not concurrently run) / computation of log likelihood
                 c=np.ndarray((d.L,)),
                 beta=np.ndarray((d.L, N)),
                 # TODO: Scaling of β (used if forward and backward recursions are run concurrently)
                 e=np.ndarray((d.L, )),
                 gamma=np.ndarray((d.L - 1, N)), xi=np.ndarray((d.L - 1, N, N)))


def forward(d: Data, m: Model) -> Model:
    """
    Computes

        α_t(i) = α(t, i) = P(X_t = i, Y_0 = y_0, ..., Y_t = y_t)

    as well as the scaling coefficients c[t].
    Requires:
        - m.B is a valid (row stochastic) emission probability matrix (N x M)
        - m.p is a valid initial probability vector (N x 1)
    Ensures:
        - m.alpha contains the *scaled* "forward probabilities"
            m.alpha[t] = α[t] / (m.c[0] * ... * m.c[t])

    WARNING! c[t] is the INVERSE of Rabiner's c[t]
    """

    # Compute and rescale α_0
    #   α_0(i) = π_i * P(Emission = Y[0] | State = i) = π_i * B(i, Y[0])
    m.alpha[0] = m.p * m.B[:, d.Y[0]]
    m.c[0] = m.alpha[0].sum()
    m.alpha[0] /= m.c[0]

    # Compute and rescale α_t
    for t in range(1, d.L):
        m.alpha[t] = (m.alpha[t - 1] @ m.A) * m.B[:, d.Y[t]]
        m.c[t] = m.alpha[t].sum()
        m.alpha[t] /= m.c[t]

    return m


def backward(d: Data, m: Model) -> Model:
    """
    Computes
        β_t(i) = β(t, i) = P(Y_{t+1} = y_{t+1}, ..., Y_{L-1} = y_{l-1} | X_t = i)

    TODO: Rescaling can be done with a new variable instead of using m.c, in order to enable
    parallel execution.
    """
    # assert hasattr(m, 'c')
    m.beta[d.L-1].fill(1.)
    for t in range(d.L - 2, -1, -1):
        m.beta[t] = (m.A @ (m.B[:, d.Y[t + 1]] * m.beta[t + 1])) / m.c[t + 1]

    return m


def posteriors(d: Data, m: Model) -> Model:
    """
    Computes
        ɣ(t, i)    = P(X_t = i | Y_0, ..., Y_{L-1})
        ξ(t, i, j) = P(X_t = i, X_{t+1} = j | Y_0, ..., Y_{L-1})
    """
    # assert (hasattr(m, 'alpha') and hasattr(m, 'beta'))
    V = m.B.T[d.Y[1:]] * m.beta[1:]
    for t in range(d.L-1):
        m.xi[t] = (m.alpha[t].reshape((m.N, 1)) * (m.A * V[t])) / m.c[t + 1]

    m.gamma = m.alpha * m.beta

    return m


def estimate(d: Data, m: Model) -> Model:
    # assert hasattr(m, 'gamma') and hasattr(m, 'xi')

    # Expected number of transitions made *from* state i
    e_transitions_from = m.gamma[:-1, :].sum(axis=0)
    # Expected number of times that state i is visited
    e_visited = e_transitions_from + m.gamma[-1, :]

    # Re-estimate π, the transition matrix A and the emission matrix B
    m.p = np.copy(m.gamma[0].reshape((m.N,)))
    m.A = m.xi.sum(axis=0) / e_transitions_from.reshape((m.N, 1))
    m.B.fill(0.)
    for j, k in np.ndindex(m.N, d.M):
        m.B[j, k] += m.gamma[d.Y == k, j].sum()
    m.B /= e_visited.reshape((m.N, 1))

    # Log likelihood of the observed emissions under the current model parameters
    m.ll = np.log(m.c).sum()

    return m


def estimate_poisson(d: Data, m: Model) -> Model:
    # Expected number of transitions made *from* state i
    e_transitions_from = m.gamma[:-1, :].sum(axis=0)

    # Re-estimate π, the transition matrix A and the emission matrix B
    m.p = np.copy(m.gamma[0].reshape((m.N,)))
    m.A = m.xi.sum(axis=0) / e_transitions_from.reshape((m.N, 1))

    time_step = 10  # Milliseconds
    rates = np.ndarray((m.N,))
    rates = (d.Y @ m.gamma) / time_step
    for j, k in np.ndindex(m.N, d.M):
        m.B[j, k] = np.power(rates[j] * time_step, k) * np.exp(- rates[j] * time_step) / \
                    np.math.factorial(k)

    # Log likelihood of the observed emissions under the current model parameters
    m.ll = np.log(m.c).sum()

    return m


def iterate(d: Data, m: Model=None, maxiter=10, eps=config.iteration_margin, verbose=False)-> Model:
    run = True
    ll = - np.inf
    it = 1
    if verbose: print("\nRunning up to maxiter = {0} with threshold {1}%:".format(maxiter, eps))
    if m is None:
        if verbose: print("Initializing model...")
        m = init(d)
    start = time()
    total = 0.
    while run:
        m = reduce(lambda x, f: f(d, x), [forward, backward, posteriors, estimate], m)
        it += 1
        avg = (np.abs(ll) + np.abs(m.ll) + config.eps) / 2
        delta = 100.0 * (m.ll - ll) / avg if avg != np.inf else np.inf
        run = it <= maxiter and delta >= eps
        ll = m.ll
        if it % 10 == 0:
            end = time()
            total += end
            if verbose:
                done = 100 * it/maxiter
                left = 100 - done
                print("\r[{0}{1}] {2}%   Iteration: {3:>{it_width}}, time delta: {4:>6.4}s, "
                      "log likelihood delta {5:.3}%".
                      format('■' * np.floor(done/2), '·' * np.ceil(left/2), int(done),
                             it, end-start, delta, it_width=int(np.ceil(np.log10(maxiter)))),
                      end='')
                # print("Iteration {} finishes after {:.4}s with an increase of "
                #       "{:.3}% in the likelihood".format(it, end - start, delta))
            start = time()
    if verbose:
        print("\nDone after {0} iterations. Total running time: {1:>8.4}s"
              .format(it, total - start))
    return m


def viterbi_path(d: Data, m: Model) -> Array:
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
