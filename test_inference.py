from functools import reduce
from queue import Queue, Empty
import unittest as ut
import numpy as np
import inference as infer
import data
import config
from utils import is_row_stochastic, available_cpu_count, new_channel
from concurrent.futures import ProcessPoolExecutor
from time import sleep


def messenger(q: Queue):
    while True:
        sleep(0.1)
        try:
            print(q.get_nowait())
        except Empty:
            pass


def run_multiple_iterations(N, M, p, A, B, d, runs_multiplier=4, name=""):
    q = Queue()
    winner = None
    procs = available_cpu_count()
    runs = procs * runs_multiplier
    with ProcessPoolExecutor(max_workers=procs+1) as ex:
        msg = ex.submit(messenger, q)
        runs = procs * 4
        initial_models = map(lambda dd: infer.init(dd, N), [d] * runs)
        maxiter = 2000
        iterations = [maxiter] * runs
        channels = [new_channel(q, x) for x in range(runs)]
        max_ll = - np.inf
        print("Starting {0} tests using {1} processes (maxiter={2})".format(runs, procs, maxiter))
        # tasks = [[infer.iterate, dd, infer.init(dd, N), 2000, new_channel(q, k)]
        #          for dd, k in zip([d] * runs, range(runs))]
        # running_tasks = []
        # while True:
        #     for t in running_tasks:
        #         if free_slot:
        #             append new task
        #         if any_done:
        #             pop task
        #     if any_messages:
        #         print messages
        #     sleep(10ms)
        for m in ex.map(infer.iterate, [d] * runs, initial_models, iterations, channels):
            if m.ll > max_ll:
                max_ll = m.ll
                winner = m
                print("Worker done: new winner with log likelihood = {0}".format(m.ll))
            else:
                print("Worker done.")

    m = winner
    # FIXME: All possible permutations of the labels for THREE states
    permutations = [[0, 1, 2], [0, 2, 1], [2, 1, 0], [1, 0, 2]]
    p_ok = A_ok = B_ok = all_ok = True
    for per in permutations:
        alt_p = p[per]
        alt_A = A[per].T[per].T
        alt_B = B[per]
        p_ok = p_ok and np.allclose(alt_p, m.p, atol=config.test_eps)
        A_ok = A_ok and np.allclose(alt_A, m.A, atol=config.test_eps)
        B_ok = B_ok and np.allclose(alt_B, m.B, atol=config.test_eps)
        all_ok = p_ok and A_ok and B_ok
        if all_ok:
            break

    if not all_ok:
        print("Tests failed. Saving state to /tmp/")
        np.savetxt("/tmp/test_iterate_{}.Y".format(name), d.Y, fmt="%d")
        np.savetxt("/tmp/test_iterate_{}.p".format(name), p)
        np.savetxt("/tmp/test_iterate_{}.A".format(name), A)
        np.savetxt("/tmp/test_iterate_{}.B".format(name), B)

    return [m.p, p_ok, m.A, A_ok, m.B, B_ok]


class TestMethods(ut.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        np.random.seed(2015)
        self.d = data.generate(N=3, M=4, L=1000)

    def test_init(self):
        m = infer.init(self.d)
        for M in [m.p, m.A, m.B]:
            self.assertTrue(is_row_stochastic(M), "Initial model parameters are not probabilities")

        with self.subTest("Test initial distribution"):
            self.assertEqual(m.p.shape, (m.N, ), "Shapes don't match")
        with self.subTest("Test transition"):
            self.assertEqual(m.A.shape, (m.N, m.N), "Shapes don't match")
        with self.subTest("Test emission"):
            self.assertEqual(m.B.shape, (m.N, self.d.M), "Shapes don't match")

    def test_forward(self):
        m = infer.init(self.d)
        m = infer.forward(self.d, m)
        self.assertEqual(m.alpha.shape, (self.d.L, m.N), "Shapes don't match")

        alpha = np.zeros_like(m.alpha)
        c = np.zeros_like(m.c)
        alpha[0] = m.p * m.B[:, self.d.Y[0]]
        c[0] = alpha[0].sum()
        alpha[0] /= c[0]
        for t in range(1, self.d.L):
            for i in range(m.N):
                for j in range(m.N):
                    alpha[t, i] += alpha[t-1, j] * m.A[j, i]
                alpha[t, i] *= m.B[i, self.d.Y[t]]
                c[t] += alpha[t, i]
            alpha[t] /= c[t]
        self.assertTrue(np.allclose(alpha, m.alpha) and np.allclose(c, m.c), "Computation is wrong")

    def test_backward(self):
        # FIXME: compute proper scaling in beta_pass instead of relying on alpha_pass
        m = reduce(lambda x, f: f(self.d, x),
                   [infer.forward, infer.backward],
                   infer.init(self.d))
        self.assertEqual(m.beta.shape, (self.d.L, m.N), "Shapes don't match")

        beta = np.zeros_like(m.beta)
        beta[self.d.L - 1] = 1.  # m.c[self.d.L - 1]
        for t in range(self.d.L-2, -1, -1):
            for i in range(m.N):
                for j in range(m.N):
                    beta[t, i] += m.A[i, j] * m.B[j, self.d.Y[t+1]] * beta[t+1, j]
                beta[t, i] /= m.c[t+1]
        self.assertTrue((np.allclose(beta, m.beta)), "Computation is wrong")

    def test_posteriors(self):
        m = reduce(lambda x, f: f(self.d, x),
                   [infer.forward, infer.backward, infer.posteriors],
                   infer.init(self.d))
        gamma = np.zeros_like(m.gamma)
        digamma = np.ndarray(shape=m.xi.shape)
        for t in range(self.d.L - 1):
            norm = 0.0
            for i, j in np.ndindex(m.N, m.N):
                norm += m.alpha[t, i] * m.A[i, j] * m.B[j, self.d.Y[t+1]] * m.beta[t+1, j]
            for i, j in np.ndindex(m.N, m.N):
                digamma[t, i, j] = m.alpha[t, i] * m.A[i, j] * \
                                   m.B[j, self.d.Y[t+1]] * m.beta[t+1, j] / norm
                gamma[t, i] += digamma[t, i, j]

        with self.subTest('Test gamma'):
            self.assertEqual(m.gamma.shape, (self.d.L, m.N), "Shapes don't match")
            self.assertTrue(np.allclose(m.xi.sum(axis=(1, 2)), np.ones(self.d.L - 1)))
            # HACK: gamma has one element less than m.gamma (!)
            self.assertTrue(np.allclose(gamma[:-1], m.gamma[:-1]), 'Computation is wrong')

        with self.subTest('Test xi'):
            self.assertEqual(m.xi.shape, (self.d.L - 1, m.N, m.N), "Shapes don't match")
            # Each matrix xi[t,·, ·] is P(x_t, x_{t+1} | Y)
            self.assertTrue(np.allclose(1.0, m.xi.sum(axis=(1, 2))),
                            "Digammas don't sum up to one")
            self.assertTrue(np.allclose(digamma, m.xi), "Computation is wrong")

    def test_estimate(self):
        m = reduce(lambda x, f: f(self.d, x),
                   [infer.forward, infer.backward, infer.posteriors, infer.estimate],
                   infer.init(self.d))
        # Sanity checks
        self.assertTrue(is_row_stochastic(m.A), "A is not row stochastic")
        self.assertTrue(is_row_stochastic(m.B), "B is not row stochastic")
        self.assertTrue(is_row_stochastic(m.p), "p is not a probability table")

        # Did we accidentally share data somewhere?
        self.assertTrue(m.p.base is None)
        self.assertTrue(m.A.base is None)
        self.assertTrue(m.B.base is None)
        self.assertTrue(m.xi.base is None)

        with self.subTest("Test transition matrix"):
            A = np.ndarray(shape=m.A.shape)
            for i, j in np.ndindex(m.N, m.N):
                num = 0.0
                den = 0.0
                for t in range(self.d.L - 1):
                    num += m.xi[t, i, j]
                    den += m.gamma[t, i]
                A[i, j] = num / den
            self.assertTrue(np.allclose(A, m.A), "Wrong estimation")

        with self.subTest("Test emission matrix"):
            B = np.ndarray(shape=m.B.shape)
            for i, k in np.ndindex(m.N, self.d.M):
                num = 0.0
                den = 0.0
                for t in range(self.d.L):
                    if self.d.Y[t] == k:
                        num += m.gamma[t, i]
                    den += m.gamma[t, i]
                B[i, k] = num / den
            self.assertTrue(np.allclose(B, m.B), "Wrong estimation")

    def test_iterate_simple(self):
        N = 3
        M = 2
        p = np.array([1, 0, 0])
        A = np.array([[0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]])
        B = np.array([[1, 0], [0, 1], [0, 1]])
        d = data.generate(N=N, M=M, L=1000, p=p, A=A, B=B)
        [mp, p_ok, mA, A_ok, mB, B_ok] = run_multiple_iterations(N=N, M=M, p=p, A=A, B=B, d=d,
                                                                 name="simple")

        with self.subTest("Test initial distribution"):
            if not p_ok:
                self.fail("Estimated initial distribution diverges from generator:"
                          "\np: {0}\nm.p: {1}".format(p, np.round(mp, 2)))
        with self.subTest("Test transition"):
            if not A_ok:
                self.fail("Estimated transition matrix diverges from generator:"
                          "\nA:\n{0}\nm.A:\n{1}".format(A, np.round(mA, 2)))
        with self.subTest("Test emission"):
            if not B_ok:
                self.fail("Estimated emission matrix diverges from generator:"
                          "\nB:\n{0}\nm.B:\n{1}".format(B, np.round(mB, 2)))

    @ut.skip("Blah")
    def test_iterate(self):
        [N, p, A, B] = [self.d.generator[k] for k in ['N', 'p', 'A', 'B']]
        [mp, p_ok, mA, A_ok, mB, B_ok] = run_multiple_iterations(N=N, M=self.d.M, p=p, A=A, B=B,
                                                                 d=self.d)

        with self.subTest("Test initial distribution"):
            if not p_ok:
                self.fail("Estimated initial distribution diverges from generator:"
                          "\np: {0}\nm.p: {1}".format(p, np.round(mp, 2)))
        with self.subTest("Test transition"):
            if not A_ok:
                self.fail("Estimated transition matrix diverges from generator:"
                          "\nA:\n{0}\nm.A:\n{1}".format(A, np.round(mA, 2)))
        with self.subTest("Test emission"):
            if not B_ok:
                self.fail("Estimated emission matrix diverges from generator:"
                          "\nB:\n{0}\nm.B:\n{1}".format(B, np.round(mB, 2)))

if __name__ == '__main__':
    ut.main()
