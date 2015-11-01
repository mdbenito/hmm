from utils import *
import unittest as ut
import data


class TestMethods(ut.TestCase):
    def test_generate(self):
        d = data.generate(N=4, M=10, L=1000)
        for M in d.generator.values():
            self.assertTrue(is_row_stochastic(M))
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
