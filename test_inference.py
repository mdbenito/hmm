import unittest as ut
import numpy as np
import inference
import data
from utils import *


class TestMethods(ut.TestCase):
    def test_init(self):
        m = inference.init(data.Data(M=10, L=100, Y=np.random.random_integers(low=0, high=9, size=(1, 100))))
        for M in [m.p, m.A, m.B]:
            self.assertTrue(is_row_stochastic(M))


if __name__ == '__main__':
    ut.main()
