import numpy as np
from data import Data
from config import Scalar, Vector, Matrix
from .base import HMM


class Discrete(HMM):
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
            also used to compute the log likelihood (since α itm is scaled)
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
    iteration = 0
    visited = np.array((L,))
    transitions_from = np.array((L, ))

    def __init__(m, d: Data, N: int, **kwds):
        super().__init__(d, N, **kwds)
        m.alpha = np.ndarray((d.L, N))
        m.c = np.ndarray((d.L,))
        m.beta = np.ndarray((d.L, N))
        # TODO: Scaling of β
        # used if forward and backward recursions are run concurrently
        # e=np.ndarray((d.L, )),
        m.gamma = np.ndarray((d.L - 1, N))
        m.xi = np.ndarray((d.L - 1, N, N))
        m.data = d

    def forward(m):
        """
        Requires:
            - m.A is a valid (row stochastic) transition probability matrix (N x N)
            - m.B is a valid emission probability matrix (N x M)
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
        m.alpha[0] = m.p * m.B[:, m.data.Y[0]]
        m.c[0] = m.alpha[0].sum()
        m.alpha[0] /= m.c[0]
    
        # Compute and rescale α_t
        for t in range(1, m.data.L):
            m.alpha[t] = (m.alpha[t - 1] @ m.A) * m.B[:, m.data.Y[t]]
            m.c[t] = m.alpha[t].sum()
            m.alpha[t] /= m.c[t]

        return m

    def backward(m):
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
        m.beta[m.data.L - 1].fill(1.)
        for t in range(m.data.L - 2, -1, -1):
            m.beta[t] = (m.A @ (m.B[:, m.data.Y[t + 1]] * m.beta[t + 1])) / m.c[t + 1]
    
        return m

    def posteriors(m):
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
        V = m.B.T[m.data.Y[1:]] * m.beta[1:]
        for t in range(m.data.L-1):
            m.xi[t] = (m.alpha[t].reshape((m.N, 1)) * (m.A * V[t])) / m.c[t + 1]

        m.gamma = m.alpha * m.beta

        # The following are used in the M-step:

        # Expected number of transitions made *from* state i
        m.transitions_from = m.gamma[:-1, :].sum(axis=0)
        # Expected number of times that state i is visited
        m.visited = m.transitions_from + m.gamma[-1, :]
        return m


class Multinomial(Discrete):
    def __init__(m, d: Data, N: int, **kwds):
        """
        Initializes the model to random values.

        :param N: Number of states in the model
        :param d: Data object
        """
        super().__init__(d, N, **kwds)
        if m.__dict__.get('p') is None:
            m.p = np.random.random((1, N))
            m.p /= m.p.sum(axis=1)[:, None]
        if m.__dict__.get('A') is None:
            m.A = np.random.random((N, N))
            m.A /= m.A.sum(axis=1)[:, None]
        if m.__dict__.get('B') is None:
            m.B = np.random.random((N, d.M))  # FIXME! Do something better
            m.B /= m.B.sum(axis=1)[:, None]
        m.p = m.p.reshape((N, ))
    
    def estimate(m):
        """
        Performs one M-step in the EM algorithm, estimating the parameters of a
        model with multinomial emissions: the initial probabilities π, the
        transition matrix A and the emission matrix B
        """
        m.p = np.copy(m.gamma[0].reshape((m.N,)))
        m.A = m.xi.sum(axis=0) / m.transitions_from.reshape((m.N, 1))
        m.B.fill(0.)
        for j, k in np.ndindex(m.N, m.data.M):
            m.B[j, k] += m.gamma[m.data.Y == k, j].sum()
        m.B /= m.visited.reshape((m.N, 1))
    
        # Log likelihood of the observed emissions under current model parameters
        m.ll = np.log(m.c).sum()
    
        m.iteration += 1
        return m


class Poisson(Discrete):
    # Hints for the IDE
    dt = Scalar
    rates = np.ndarray((1, ))

    def __init__(m, d: Data, N: int, **kwds):
        super().__init__(d, N, **kwds)
        if m.__dict__.get('p') is None:
            m.p = np.random.random((1, N))
            m.p /= m.p.sum(axis=1)[:, None]
        if m.__dict__.get('A') is None:
            m.A = np.random.random((N, N))
            m.A /= m.A.sum(axis=1)[:, None]
        m.p = m.p.reshape((N, ))
        
        m.rates = 0.8 * np.random.random((N, )) + 0.1
        m.dt = np.int(7 + 6 * np.random.random())
        m.B = m.emissions(m.rates, m.dt, length=d.M)
        m.p = m.p.reshape((N, ))

    def estimate(m):
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
        m.rates = (m.data.Y @ m.gamma) / (m.dt * m.visited)  # np.ndarray((m.N,))
        m.B = m.emissions(m.rates, m.dt, m.d.M)

        # Log likelihood of the data under the current model parameters
        m.ll = np.log(m.c).sum()
        m.iteration += 1

        return m

    @staticmethod
    def emissions(rates: Vector, dt: Scalar, length: int) -> Vector:
        N = len(rates)
        B = np.ndarray((N, length))  # FIXME: should work in-place
        for n, j in np.ndindex(N, length):
            # FIXME: precompute factorials?
            B[n, j] = np.exp(-rates[n]*dt)*np.power(rates[n]*dt, j) /\
                      np.math.factorial(j)
        # FIXME: is it ok to normalize? m.B[t] (inhomog.) definitely needs it...
        print("WARNING: normalizing emissions! Ok?")
        return B / B.sum(axis=1).reshape((N, 1))
