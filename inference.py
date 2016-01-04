from functools import reduce
from data import Data
from time import time
import numpy as np
import config


# For the type hints
Scalar = np.float64
Vector = Matrix = Array = np.ndarray


class Model:
    """
    Model parameters:
        N = Number of states in the model
        A = State transition matrix (NxN): A[i,j] = Prob(X_{t+1} = j | X_t = i)
        B = Observation probability matrix (NxM):
                B(j, k) = Prob(Y_t = k | X_t = j)
            TODO: check whether usage pattern for B makes the transpose faster
        p = Distribution for the initial state
    Internal data:
        alpha (LxN)
        beta (LxN)
        gamma (LxN)
        xi(LxNxN)
        c:  Scaling for α (and β if not concurrently run)
            also used to compute the log likelihood (since α itself is scaled)
        ll: log likelihood of the observed emissions
    Configuration:
        max_iterations
    Notation:
        X = [X_0, ..., X_{L-1}] = chain of states
        Y = [Y_0, ..., Y_{L-1}] = observed emissions
    """

    # Hints for the IDE:
    N = int(1)
    M = int(1)
    L = int(1)
    A = np.ndarray((N, N))
    B = np.ndarray((N, M))
    p = np.ndarray((N,))
    alpha = np.ndarray((L, N))
    beta = np.ndarray((L, N))
    gamma = np.ndarray((L-1, N))
    xi = np.ndarray((L-1, N, N))
    c = np.ndarray((L, ))
    ll = - np.inf
    visited = np.array((L,))
    transitions_from = np.array((L, ))
    iteration = 0

    # If using Poisson we also have:
    dt = np.float64
    rates = np.ndarray((N, ))

    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def init_multinomial(d: Data, N: int=4) -> Model:
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
                 c=np.ndarray((d.L,)),
                 beta=np.ndarray((d.L, N)),
                 # TODO: Scaling of β
                 # used if forward and backward recursions are run concurrently
                 # e=np.ndarray((d.L, )),
                 gamma=np.ndarray((d.L - 1, N)), xi=np.ndarray((d.L - 1, N, N)))


def poisson_emissions(rates: Vector, dt: Scalar, length: int) -> Vector:
    N = len(rates)
    B = np.ndarray((N, length))  # FIXME: should work in-place
    for n, j in np.ndindex(N, length):
        # FIXME: precompute factorials?
        B[n, j] = np.exp(-rates[n]*dt)*np.power(rates[n]*dt, j) /\
                  np.math.factorial(j)
    # FIXME: is it ok to normalize? m.B[t] (inhomog.) definitely needs it...
    return B / B.sum(axis=1).reshape((N, 1))


def init_poisson(d: Data, N: int=4) -> Model:
    p = np.random.random((1, N))
    A = np.random.random((N, N))
    [p, A] = map(lambda M: M / M.sum(axis=1)[:, None], [p, A])  # Normalize
    rates = 0.8 * np.random.random((N, )) + 0.1
    dt = np.int(7 + 6 * np.random.random())
    B = poisson_emissions(rates, dt, length=d.M)
    print("WARNING: normalizing emissions! Ok?")

    return Model(N=N, p=p.reshape((N,)), A=A, rates=rates, B=B, dt=dt,
                 alpha=np.ndarray((d.L, N)),
                 c=np.ndarray((d.L,)),
                 beta=np.ndarray((d.L, N)),
                 gamma=np.ndarray((d.L - 1, N)),
                 xi=np.ndarray((d.L - 1, N, N)))


def forward(d: Data, m: Model) -> Model:
    """
    Requires:
        - m.A is a valid (row stochastic) transition probability matrix (N x N)
        - m.B is a valid (row stochastic) emission probability matrix (N x M)
        - m.p is a valid initial probability vector (N x 1)

    Ensures:
        - Computed m.alpha with the *scaled* forward probabilities
            m.alpha[t] = α[t] / (m.c[0] * ... * m.c[t])
          where
            α(t, i) = P(X_t = i, Y_0 = y_0, ..., Y_t = y_t)
        - Computed scaling coefficients m.c, valid for backward() pass

    WARNING! m.c[t] is the INVERSE of Rabiner's c[t]
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
    Requires:
        - m.A is a valid (row stochastic) transition probability matrix (N x N)
        - m.B is a valid (row stochastic) emission probability matrix (N x M)
        - Precomputed scaling coefficients m.c (using forward())
    Ensures:
        - Computed backward probabilities:
            β(t, i) = P(Y_{t+1} = y_{t+1}, ..., Y_{L-1} = y_{l-1} | X_t = i)

    TODO: For execution in parallel with forward(), rescaling can be done with
          a new variable instead of using m.c.
    """
    m.beta[d.L-1].fill(1.)
    for t in range(d.L - 2, -1, -1):
        m.beta[t] = (m.A @ (m.B[:, d.Y[t + 1]] * m.beta[t + 1])) / m.c[t + 1]

    return m


def posteriors(d: Data, m: Model) -> Model:
    """
    Requires:
        - Precomputed m.alpha (using forward())
        - Precomputed m.beta (using backward())
    Ensures:
        - Computed m.gamma:
            ɣ(t, i) = P(X_t = i | Y_0, ..., Y_{L-1})
        - Computed m.xi:
            ξ(t, i, j) = P(X_t = i, X_{t+1} = j | Y_0, ..., Y_{L-1})
          For each t, ξ(t) is an NxN matrix whose entries sum to 1
    """
    V = m.B.T[d.Y[1:]] * m.beta[1:]
    for t in range(d.L-1):
        m.xi[t] = (m.alpha[t].reshape((m.N, 1)) * (m.A * V[t])) / m.c[t + 1]

    m.gamma = m.alpha * m.beta

    # The following are used in the M-step (estimate_*):

    # Expected number of transitions made *from* state i
    m.transitions_from = m.gamma[:-1, :].sum(axis=0)
    # Expected number of times that state i is visited
    m.visited = m.transitions_from + m.gamma[-1, :]
    return m


def estimate_multinomial(d: Data, m: Model) -> Model:
    """
    Performs one M-step in the EM algorithm, estimating the parameters of a
    model with multinomial emissions: the initial probabilities π, the
    transition matrix A and the emission matrix B
    """
    m.p = np.copy(m.gamma[0].reshape((m.N,)))
    m.A = m.xi.sum(axis=0) / m.transitions_from.reshape((m.N, 1))
    m.B.fill(0.)
    for j, k in np.ndindex(m.N, d.M):
        m.B[j, k] += m.gamma[d.Y == k, j].sum()
    m.B /= m.visited.reshape((m.N, 1))

    # Log likelihood of the observed emissions under current model parameters
    m.ll = np.log(m.c).sum()

    m.iteration += 1
    return m


def estimate_poisson(d: Data, m: Model) -> Model:
    """
    Performs one M-step in the EM algorithm, estimating the parameters of a
    model with Poisson emissions.

    Requires:
        - Precomputed m.alpha, m.beta, m.gamma, m.xi
        - Valid m.dt
    Ensures:
        - Computed m.A, m.B, m.p, m.ll
    """

    # from scipy.optimize import fsolve
    # u = d.Y @ m.gamma
    # def gradQ3(x) -> np.ndarray:
    #     nonlocal d, m, u
    #     return u/x - m.dt * m.visited
    #
    # def hessQ3(x) -> np.ndarray:
    #     nonlocal d, m, u
    #     return np.diag(- u / (m.rates * m.rates))
    #
    # new_rates = fsolve(gradQ3, m.rates, fprime=hessQ3)

    m.p = np.copy(m.gamma[0].reshape((m.N,)))
    m.A = m.xi.sum(axis=0) / m.transitions_from.reshape((m.N, 1))
    m.rates = (d.Y @ m.gamma) / (m.dt * m.visited)  # np.ndarray((m.N,))
    B = poisson_emissions(m.rates, m.dt, d.M)

    # Log likelihood of the data under the current model parameters
    m.ll = np.log(m.c).sum()
    m.iteration += 1

    return m


def iterate(d: Data, m: Model=None, maxiter=10, eps=config.iteration_margin,
            verbose=False)-> Model:
    run = True
    ll = - np.inf
    it = 1
    if verbose: print("\nRunning up to maxiter = {0}"
                      " with threshold {1}%:".format(maxiter, eps))
    if m is None:
        if verbose: print("Initializing model...")
        m = init_poisson(d)
    start = time()
    total = 0.
    while run:
        m = reduce(lambda x, f: f(d, x),
                   [forward, backward, posteriors, estimate_poisson], m)
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
                print("\r[{0}{1}] {2}%   Iteration: {3:>{it_width}},"
                      " time delta: {4:>6.4}s, log likelihood delta {5:.3}%".
                      format('■' * np.floor(done/2), '·' * np.ceil(left/2),
                             int(done),
                             it, end-start, delta,
                             it_width=int(np.ceil(np.log10(maxiter)))),
                      end='')
            start = time()
    if verbose:
        print("\nDone after {0} iterations. Total running time: {1:>8.4}s"
              .format(it, total - start))
    return m


def viterbi_path(d: Data, m: Model) -> Array:
    """
    TODO: Returns the sequence of states maximizing the expected number of
          correct states.
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
