from functools import reduce
import unittest as ut
import numpy as np
import inference as i
import data
import config
from utils import is_row_stochastic


class TestMethods(ut.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.d = data.generate(N=4, M=8, L=1000)

    @ut.skip('Basic test has to pass first')
    def test_init(self):
        m = i.init(self.d)
        for M in [m.p, m.A, m.B]:
            self.assertTrue(is_row_stochastic(M), 'Initial model parameters are not probabilities')

        with self.subTest('Test initial distribution'):
            self.assertEqual(m.p.shape, (m.N, ), 'Shapes don\'t match')
        with self.subTest('Test transition'):
            self.assertEqual(m.A.shape, (m.N, m.N), 'Shapes don\'t match')
        with self.subTest('Test emission'):
            self.assertEqual(m.B.shape, (m.N, self.d.M), 'Shapes don\'t match')

    @ut.skip('Basic test has to pass first')
    def test_alpha_pass(self):
        m = i.init(self.d)
        m = i.alpha_pass(self.d, m)
        self.assertEqual(m.alpha.shape, (self.d.L, m.N), 'Shapes don\'t match')

    @ut.skip('Basic test has to pass first')
    def test_beta_pass(self):
        m = i.init(self.d)
        m = i.beta_pass(self.d, m)
        self.assertEqual(m.beta.shape, (self.d.L, m.N), 'Shapes don\'t match')

    @ut.skip('Basic test has to pass first')
    def test_gammas(self):
        m = reduce(lambda x, f: f(self.d, x), [i.alpha_pass, i.beta_pass, i.gammas, i.estimate], i.init(self.d))
        with self.subTest('Test gamma'):
            self.assertEqual(m.gamma.shape, (self.d.L-1, m.N), 'Shapes don\'t match')
        with self.subTest('Test digamma'):
            self.assertEqual(m.digamma.shape, (self.d.L-1, m.N, m.N), 'Shapes don\'t match')

    def test_estimate_simple(self):
        N = 2
        p = np.array([1, 0])
        A = np.array([[0.1, 0.9], [0.1, 0.9]])
        B = np.array([[1, 0], [0, 1]])
        d = data.generate(N=N, M=2, L=1000, p=p, A=A, B=B)
        m = i.init(d, N)
        m = i.iterate(d, m, maxiter=1000, eps=config.test_eps)

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
        m = i.init(self.d, N)
        m = i.iterate(self.d, m, maxiter=1000)

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
