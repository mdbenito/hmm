import unittest as ut
import test_data
import test_inference

if __name__ == '__main__':
    suite = ut.TestSuite()
    for case in [test_data.TestMethods,
                 test_inference.TestMethods]:
        suite.addTest(ut.defaultTestLoader.loadTestsFromTestCase(case))
    ut.TextTestRunner(verbosity=2).run(suite)
