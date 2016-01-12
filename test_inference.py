import unittest as ut
import numpy as np
import matplotlib.pyplot as plt
import inference as infer
from models.base import HMM
from models.discrete import Multinomial, Poisson
import data
import config
from utils import is_row_stochastic
import warnings
warnings.filterwarnings('error')


# FIXME: We only do possible permutations of the labels for THREE states
def check_permutations(m: HMM, truth: HMM, permutations: int=3, name: str=""):
    permutations = [[0, 1, 2], [0, 2, 1], [2, 1, 0], [1, 0, 2]]
    p_ok = A_ok = B_ok = all_ok = True
    for per in permutations:
        alt_p = truth.p[per]
        alt_A = truth.A[per].T[per].T
        alt_B = truth.B[per]
        p_ok = p_ok and np.allclose(alt_p, m.p, atol=config.test_eps)
        A_ok = A_ok and np.allclose(alt_A, m.A, atol=config.test_eps)
        B_ok = B_ok and np.allclose(alt_B, m.B, atol=config.test_eps)
        all_ok = p_ok and A_ok and B_ok
        if all_ok:
            break

    if not all_ok:
        print("Tests failed. Saving state to /tmp/")
        np.savetxt("/tmp/test_{}.Y".format(name), m.d.Y, fmt="%d")
        np.savetxt("/tmp/test_{}.p".format(name), truth.p)
        np.savetxt("/tmp/test_{}.A".format(name), truth.A)
        np.savetxt("/tmp/test_{}.B".format(name), truth.B)

    return [m.p, p_ok, m.A, A_ok, m.B, B_ok]


class TestMultinomial(ut.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.d = data.generate_multinomial(N=3, M=4, L=500)

    def test_init(self):
        m = Multinomial(self.d)
        for M in [m.p, m.A, m.B]:
            self.assertTrue(is_row_stochastic(M),
                            "Initial model parameters are not probabilities")

        with self.subTest("Test initial distribution"):
            self.assertEqual(m.p.shape, (m.N, ), "Shapes don't match")
        with self.subTest("Test transition"):
            self.assertEqual(m.A.shape, (m.N, m.N), "Shapes don't match")
        with self.subTest("Test emission"):
            self.assertEqual(m.B.shape, (m.N, self.d.M), "Shapes don't match")

    def test_forward(self):
        m = Multinomial(self.d)
        m.forward()
        self.assertEqual(m.alpha.shape, (self.d.L, m.N), "Shapes don't match")

        alpha = np.zeros_like(m.alpha)
        c = np.zeros_like(m.c)
        alpha[0] = m.p * m.B[:, self.d.Y[0]]
        c[0] = alpha[0].sum()
        alpha[0] /= c[0]
        for t in range(1, self.d.L):
            for i in range(m.N):
                for j in range(m.N):
                    alpha[t, i] += alpha[t-1, j] * m.A[j, i]
                alpha[t, i] *= m.B[i, self.d.Y[t]]
                c[t] += alpha[t, i]
            alpha[t] /= c[t]
        self.assertTrue(np.allclose(alpha, m.alpha) and np.allclose(c, m.c),
                        "Computation is wrong")

    def test_backward(self):
        # FIXME: compute proper scaling for beta instead of relying on m.c
        m = Multinomial(self.d)
        m.forward()
        m.backward()
        self.assertEqual(m.beta.shape, (self.d.L, m.N), "Shapes don't match")

        beta = np.zeros_like(m.beta)
        beta[self.d.L - 1] = 1.  # m.c[self.d.L - 1]
        for t in range(self.d.L-2, -1, -1):
            for i in range(m.N):
                for j in range(m.N):
                    beta[t, i] += m.A[i, j] * m.B[j, self.d.Y[t+1]] *\
                                  beta[t+1, j]
                beta[t, i] /= m.c[t+1]
        self.assertTrue((np.allclose(beta, m.beta)), "Computation is wrong")

    def test_posteriors(self):
        m = Multinomial(self.d)
        m.forward()
        m.backward()
        m.posteriors()

        gamma = np.zeros_like(m.gamma)
        xi = np.ndarray(shape=m.xi.shape)
        for t in range(self.d.L - 1):
            norm = 0.0
            for i, j in np.ndindex(m.N, m.N):
                norm += m.alpha[t, i] * m.A[i, j] * m.B[j, self.d.Y[t+1]] *\
                        m.beta[t+1, j]
            for i, j in np.ndindex(m.N, m.N):
                xi[t, i, j] = m.alpha[t, i] * m.A[i, j] * \
                                   m.B[j, self.d.Y[t+1]] * m.beta[t+1, j] / norm
                gamma[t, i] += xi[t, i, j]

        with self.subTest('Test gamma'):
            self.assertEqual(m.gamma.shape, (self.d.L, m.N),
                             "Shapes don't match")
            # HACK: gamma has one element less than m.gamma (!)
            self.assertTrue(np.allclose(gamma[:-1], m.gamma[:-1]),
                            "Computation of gamma is wrong")

        with self.subTest('Test xi'):
            self.assertEqual(m.xi.shape, (self.d.L - 1, m.N, m.N),
                             "Shapes don't match")
            # Each matrix xi[t,·, ·] is P(x_t, x_{t+1} | Y)
            self.assertTrue(np.allclose(1.0, m.xi.sum(axis=(1, 2))),
                            "Xi doesn't sum up to one")
            self.assertTrue(np.allclose(xi, m.xi), "Computation of xi is wrong")

    def test_estimate(self):
        m = Multinomial(self.d)
        m.forward()
        m.backward()
        m.posteriors()
        m.estimate()

        # Sanity checks
        self.assertTrue(is_row_stochastic(m.A), "A is not row stochastic")
        self.assertTrue(is_row_stochastic(m.B), "B is not row stochastic")
        self.assertTrue(is_row_stochastic(m.p), "p is not a probability table")

        # Did we accidentally share data somewhere?
        self.assertTrue(m.p.base is None)
        self.assertTrue(m.A.base is None)
        self.assertTrue(m.B.base is None)
        self.assertTrue(m.xi.base is None)

        with self.subTest("Test transition matrix"):
            A = np.ndarray(shape=m.A.shape)
            for i, j in np.ndindex(m.N, m.N):
                num = 0.0
                den = 0.0
                for t in range(self.d.L - 1):
                    num += m.xi[t, i, j]
                    den += m.gamma[t, i]
                A[i, j] = num / den
            self.assertTrue(np.allclose(A, m.A), "Wrong estimation")

        with self.subTest("Test emission matrix"):
            B = np.ndarray(shape=m.B.shape)
            for i, k in np.ndindex(m.N, self.d.M):
                num = 0.0
                den = 0.0
                for t in range(self.d.L):
                    if self.d.Y[t] == k:
                        num += m.gamma[t, i]
                    den += m.gamma[t, i]
                B[i, k] = num / den
            self.assertTrue(np.allclose(B, m.B), "Wrong estimation")


class TestIterate(ut.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def test_iterate_simple(self):
        N = 3
        M = 2
        p = np.array([1, 0, 0])
        A = np.array([[0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]])
        B = np.array([[1, 0], [0, 1], [0, 1]])
        d = data.generate_multinomial(N=N, M=M, L=400, p=p, A=A, B=B)

        initial_models = [Multinomial(dd, N) for dd in [d] * 16]
        truth = Multinomial(d, N=N, p=p, A=A, B=B)
        m = infer.run_multiple_models(initial_models)
        [mp, p_ok, mA, A_ok, mB, B_ok] = \
            check_permutations(m, truth=truth, permutations=3, name="simple")

        with self.subTest("Test initial distribution"):
            if not p_ok:
                self.fail("Estimation of initial distribution failed."
                          "\np: {0}\nm.p: {1}".format(p, np.round(mp, 2)))
        with self.subTest("Test transition"):
            if not A_ok:
                self.fail("Estimation of transition matrix failed."
                          "\nA:\n{0}\nm.A:\n{1}".format(A, np.round(mA, 2)))
        with self.subTest("Test emission"):
            if not B_ok:
                self.fail("Estimation of emission matrix failed."
                          "\nB:\n{0}\nm.B:\n{1}".format(B, np.round(mB, 2)))

    def test_iterate(self):
        d = data.generate_multinomial(N=3, M=4, L=500)
        [N, p, A, B] = [d.generator[k] for k in 'NpAB']
        initial_models = [Multinomial(dd, N) for dd in [self.d] * 16]
        truth = Multinomial(N=N, p=p, A=A, B=B)
        m = infer.run_multiple_models(initial_models)
        [mp, p_ok, mA, A_ok, mB, B_ok] = \
            check_permutations(m, truth=truth, permutations=3, name="multinomial")

        with self.subTest("Test initial distribution"):
            if not p_ok:
                self.fail("Estimation of initial distribution failed."
                          "\np: {0}\nm.p: {1}".format(p, np.round(mp, 2)))
        with self.subTest("Test transition"):
            if not A_ok:
                self.fail("Estimation of transition matrix failed."
                          "\nA:\n{0}\nm.A:\n{1}".format(A, np.round(mA, 2)))
        with self.subTest("Test emission"):
            if not B_ok:
                self.fail("Estimation of emission matrix failed."
                          "\nB:\n{0}\nm.B:\n{1}".format(B, np.round(mB, 2)))

    def test_poisson(self):
        N = 3
        M = 8
        dt = 10
        rates = np.array([0.8, 0.4, 0.04])
        B = np.ndarray((N, M))
        for j, k in np.ndindex(N, M):
            B[j, k] = np.exp(-rates[j]*dt)*np.power(rates[j]*dt, k) /\
                      np.math.factorial(k)
        d = data.generate_discrete(N=N, M=M, L=400, B=B)
        [N, p, A] = [d.generator[k] for k in ['N', 'p', 'A']]
        initial_models = [infer.init_poisson(dd, N) for dd in [d] * 16]
        truth = infer.Model(N=N, p=p, A=A, B=B, rates=rates, dt=dt)
        m = infer.run_multiple_models(initial_models)
        [mp, p_ok, mA, A_ok, mB, B_ok] = \
            check_permutations(m, truth=truth, permutations=3, name="Poisson")

        plt.subplot(3, 1, 1)
        plt.plot(range(M), B[0], 'g')
        plt.plot(range(M), mB[0], 'b')
        plt.subplot(3, 1, 2)
        plt.plot(range(M), B[1], 'g')
        plt.plot(range(M), mB[1], 'b')
        plt.subplot(3, 1, 3)
        plt.plot(range(M), B[2], 'g')
        plt.plot(range(M), mB[2], 'b')
        plt.show()

        return [M, p, mp, A, mA, B, mB]


if __name__ == '__main__':
    ut.main()
