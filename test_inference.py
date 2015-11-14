from functools import reduce
import unittest as ut
import numpy as np
import inference as infer
import data
import config
from utils import is_row_stochastic


class TestMethods(ut.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.d = data.generate(N=4, M=8, L=1000)

    @ut.skip('Basic test has to pass first')
    def test_init(self):
        m = infer.init(self.d)
        for M in [m.p, m.A, m.B]:
            self.assertTrue(is_row_stochastic(M), 'Initial model parameters are not probabilities')

        with self.subTest('Test initial distribution'):
            self.assertEqual(m.p.shape, (m.N, ), 'Shapes don\'t match')
        with self.subTest('Test transition'):
            self.assertEqual(m.A.shape, (m.N, m.N), 'Shapes don\'t match')
        with self.subTest('Test emission'):
            self.assertEqual(m.B.shape, (m.N, self.d.M), 'Shapes don\'t match')

    def test_alpha_pass(self):
        m = infer.init(self.d)
        m = infer.alpha_pass(self.d, m)
        self.assertEqual(m.alpha.shape, (self.d.L, m.N), 'Shapes don\'t match')

        alpha = np.ndarray(shape=m.alpha.shape)
        c = np.zeros_like(m.c)
        alpha[0] = m.p * m.B[:, self.d.Y[0]]
        c[0] = 1. / alpha[0].sum()
        alpha[0] *= c[0]
        for t in range(1, self.d.L):
            for i in range(m.N):
                alpha[t, i] = 0.0
                for j in range(m.N):
                    alpha[t, i] += alpha[t-1, j] * m.A[j, i]
                alpha[t, i] *= m.B[i, self.d.Y[t]]
                c[t] += alpha[t, i]
            c[t] = 1.0 / c[t]
            alpha[t] *= c[t]
        self.assertTrue(np.allclose(alpha, m.alpha) and np.allclose(c, m.c), 'Computation is wrong')

    def test_beta_pass(self):
        m = infer.init(self.d)
        m = infer.alpha_pass(self.d, m)  # FIXME: compute proper scaling in beta_pass instead of relying on alpha_pass
        m = infer.beta_pass(self.d, m)
        self.assertEqual(m.beta.shape, (self.d.L, m.N), 'Shapes don\'t match')

        beta = np.ndarray(shape=m.beta.shape)
        beta[self.d.L - 1] = m.c[self.d.L - 1]
        for t in range(self.d.L-2, -1, -1):
            for i in range(m.N):
                beta[t, i] = 0.0
                for j in range(m.N):
                    beta[t, i] += m.A[i, j] * m.B[j, self.d.Y[t+1]] * beta[t+1, j]
                beta[t, i] *= m.c[t]
        self.assertTrue((np.allclose(beta, m.beta)), 'Computation is wrong')

    def test_gammas(self):
        m = reduce(lambda x, f: f(self.d, x), [infer.alpha_pass, infer.beta_pass, infer.gammas], infer.init(self.d))
        gamma = np.zeros_like(m.gamma)
        digamma = np.ndarray(shape=m.digamma.shape)
        for t in range(self.d.L - 1):
            norm = 0.0
            for i in range(m.N):
                for j in range(m.N):
                    norm += m.alpha[t, i] * m.A[i, j] * m.B[j, self.d.Y[t+1]] * m.beta[t+1, j]
            for i in range(m.N):
                gamma[t, i] = 0.0
                for j in range(m.N):
                    digamma[t, i, j] = m.alpha[t, i] * m.A[i, j] * m.B[j, self.d.Y[t+1]] * m.beta[t+1, j] / norm
                    gamma[t, i] += digamma[t, i, j]

        with self.subTest('Test gamma'):
            self.assertEqual(m.gamma.shape, (self.d.L-1, m.N), 'Shapes don\'t match')
            self.assertTrue(np.allclose(gamma, m.gamma), 'Computation is wrong')
        with self.subTest('Test digamma'):
            self.assertEqual(m.digamma.shape, (self.d.L-1, m.N, m.N), 'Shapes don\'t match')
            self.assertTrue(np.allclose(digamma, m.digamma), 'Computation is wrong')

    def test_estimate_simple(self):
        N = 2
        p = np.array([1, 0])
        A = np.array([[0.1, 0.9], [0.1, 0.9]])
        B = np.array([[1, 0], [0, 1]])
        d = data.generate(N=N, M=2, L=1000, p=p, A=A, B=B)
        m = infer.init(d, N)
        m = infer.iterate(d, m, maxiter=1000, eps=config.test_eps)

        with self.subTest('Test initial distribution'):
            if not np.allclose(p, m.p, atol=config.test_eps):
                self.fail('Estimated initial distribution diverges from generator:\np: {0}\nm.p: {1}'.format(p, m.p))
        with self.subTest('Test transition'):
            if not np.allclose(A, m.A, atol=config.test_eps):
                self.fail('Estimated transition matrix diverges from generator:\nA:\n{0}\nm.A:\n{1}'.format(A, m.A))
        with self.subTest('Test emission'):
            if not np.allclose(B, m.B, atol=config.test_eps):
                self.fail('Estimated emission matrix diverges from generator\nB:\n{0}\nm.B:\n{1}'.format(B, m.B))

    @ut.skip('Basic test has to pass first')
    def test_estimate(self):
        [N, p, A, B] = [self.d.generator[k] for k in ['N', 'p', 'A', 'B']]
        m = infer.init(self.d, N)
        m = infer.iterate(self.d, m, maxiter=1000)

        with self.subTest('Test initial distribution'):
            if not np.allclose(p, m.p, atol=config.test_eps):
                self.fail('Estimated initial distribution diverges from generator:\np: {0}\nm.p: {1}'.format(p, m.p))
        with self.subTest('Test transition'):
            if not np.allclose(A, m.A, atol=config.test_eps):
                self.fail('Estimated transition matrix diverges from generator:\nA:\n{0}\nm.A:\n{1}'.format(A, m.A))
        with self.subTest('Test emission'):
            if not np.allclose(B, m.B, atol=config.test_eps):
                self.fail('Estimated emission matrix diverges from generator\nB:\n{0}\nm.B:\n{1}'.format(B, m.B))

if __name__ == '__main__':
    ut.main()
