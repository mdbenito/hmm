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
        self.d = data.Data(M=10, L=100, Y=np.random.random_integers(low=0, high=9, size=(1, 100)).reshape((100,)))

    def test_init(self):
        m = i.init(self.d)
        for M in [m.p, m.A, m.B]:
            self.assertTrue(is_row_stochastic(M), 'Initial model parameters are not probabilities')

        self.assertEqual(m.p.shape, (m.N, ), 'Shapes don\'t match')
        self.assertEqual(m.A.shape, (m.N, m.N), 'Shapes don\'t match')
        self.assertEqual(m.B.shape, (m.N, self.d.M), 'Shapes don\'t match')

    def test_alpha_pass(self):
        m = i.init(self.d)
        m = i.alpha_pass(self.d, m)
        self.assertEqual(m.alpha.shape, (self.d.L, m.N), 'Shapes don\'t match')

    def test_beta_pass(self):
        m = i.init(self.d)
        m = i.beta_pass(self.d, m)
        self.assertEqual(m.beta.shape, (self.d.L, m.N), 'Shapes don\'t match')

    def test_gammas(self):
        m = reduce(lambda x, f: f(self.d, x), [i.alpha_pass, i.beta_pass, i.gammas, i.estimate], i.init(self.d))
        self.assertEqual(m.digamma.shape, (self.d.L-1, m.N, m.N), 'Shapes don\'t match')
        self.assertEqual(m.gamma.shape, (self.d.L-1, m.N), 'Shapes don\'t match')

    def test_estimate(self):
        d = data.generate(N=2, M=2, L=100)
        m = i.init(d, 2)
        m = i.iterate(d, m, maxiter=100)
        [A, B] = [d.generator[k] for k in ['A', 'B']]

        if not np.allclose(A, m.A, atol=config.eps):
            self.fail('Estimated transition matrix diverges from generator:\nA:\n' + str(A) + '\nm.A:\n' + str(m.A))
        if not np.allclose(B, m.B, atol=config.eps):
            self.fail('Estimated emission matrix diverges from generator\nB:\n' + str(B) + '\nm.B\n' + str(m.B))

if __name__ == '__main__':
    ut.main()
