from utils import *
import unittest as ut
import data


class TestMethods(ut.TestCase):

    def test_generate_trivial(self):
        N = 2
        M = 2
        p = np.array([1, 0])
        A = np.array([[0, 1], [1, 0]])  # Should produce 0,1,0,1,0,1,0,1,0,1,...
        B = np.array([[1, 0], [0, 1]])  # Should produce 0,1,0,1,0,1,0,1,0,1,...
        d = data.generate(N=N, M=M, L=100, p=p, A=A, B=B)
        self.assertTrue((d.Y == d.generator['Q']).all(), 'Trivial generation fails')
        # TODO: count

    def test_generate(self):
        d = data.generate(N=4, M=10, L=100)
        for k, v in [[k, d.generator[k]] for k in ['p', 'A', 'B']]:
            self.assertTrue(is_row_stochastic(v), "{0} doesn't define a probability: {1}".format(k, v))
        self.assertEqual(d.Y.size, d.L)

        if not (d.Y >= 0.).all() or not (d.Y <= 9).all():
            self.fail('Generated emissions are out of range')

        # TODO: more of this

    # def test_load(self):
    #     self.assertEqual(True, False, 'loading not implemented')

    # def test_save(self):
    #     self.assertTrue(False)


if __name__ == '__main__':
    ut.main()
