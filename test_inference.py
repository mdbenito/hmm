import unittest as ut
import numpy as np
import inference
import data
from functools import partial


class TestMethods(ut.TestCase):
    def test_init(self):
        m = inference.init(data.Data(M=10, L=100, Y=np.random.random_integers(low=0, high=9, size=(1, 100))))
        for M in [m.p, m.A, m.B]:
            map(partial(self.assertAlmostEqual, 1.), M.sum(axis=1))

if __name__ == '__main__':
    ut.main()
