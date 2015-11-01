import unittest as ut
import numpy as np
import inference as i
import data
from utils import is_row_stochastic


class TestMethods(ut.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.d = data.Data(M=10, L=100, Y=np.random.random_integers(low=0, high=9, size=(1, 100)).reshape((100,)))

    def test_init(self):
        m = i.init(self.d)
        for M in [m.p, m.A, m.B]:
            self.assertTrue(is_row_stochastic(M))
            map(self.assertTrue, M >= 0)

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
        m = i.init(self.d)
        m = i.alpha_pass(self.d, m)
        m = i.beta_pass(self.d, m)
        m = i.gammas(self.d, m)
        self.assertEqual(m.digamma.shape, (self.d.L-2, m.N, m.N), 'Shapes don\'t match')
        self.assertEqual(m.gamma.shape, (self.d.L-2, m.N), 'Shapes don\'t match')

    def test_estimate(self):
        d = data.generate(N=4, M=10, L=1000)
        m = i.iterate(d, maxiter=10)
        [A, B] = [d.generator[k] for k in ('A', 'B')]

        if not np.allclose(A, m.A, atol=1.e-5):
            self.fail('Estimated transition matrix diverges from generator:\nA:\n' + str(A) + '\nm.A:\n' + str(m.A))
        if not np.allclose(B, m.B, atol=1.e-5):
            self.fail('Estimated emission matrix diverges from generator\nB:\n' + str(B) + '\nm.B\n' + str(m.B))

if __name__ == '__main__':
    ut.main()
