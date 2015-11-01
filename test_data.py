from utils import *
import unittest as ut
import data


class TestMethods(ut.TestCase):
    def test_load(self):
        self.assertEqual(1, 2)

    def test_generate(self):
        d = data.generate()
        for M in d.generator.values:
            self.assertTrue(is_row_stochastic(M))
        self.assertEqual(d.Y.size, d.L)
        # TODO: more of this

    # def test_save(self):
    #     self.assertTrue(False)


if __name__ == '__main__':
    ut.main()
