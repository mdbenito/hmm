from data import Data
from time import time
import numpy as np
from models.base import HMM
from config import iteration_margin, Array
from utils import available_cpu_count
from concurrent.futures import ProcessPoolExecutor
from typing import Sequence


def iterate(m: HMM, maxiter=10, eps=iteration_margin, verbose=False) -> HMM:
    run = True
    ll = - np.inf
    if verbose: print("\nRunning up to maxiter = {0}"
                      " with threshold {1}%:".format(maxiter, eps))

    start = time()
    total = 0.
    while run:
        m.forward()
        m.backward()
        m.posteriors()
        m.estimate()
        avg = (np.abs(ll) + np.abs(m.ll) + eps) / 2
        delta = 100.0 * (m.ll - ll) / avg if avg != np.inf else np.inf
        run = m.iteration <= maxiter and delta >= eps
        ll = m.ll
        if m.iteration % 10 == 0:
            end = time()
            total += end
            if verbose:
                done = 100 * m.iteration / maxiter
                left = 100 - done
                print("\r[{0}{1}] {2}%   Iteration: {3:>{it_width}},"
                      " time delta: {4:>6.4}s, log likelihood delta {5:.3}%".
                      format('■' * np.floor(done/2), '·' * np.ceil(left/2),
                             int(done),
                             m.iteration, end-start, delta,
                             it_width=int(np.ceil(np.log10(maxiter)))),
                      end='')
            start = time()
    if verbose:
        print("\nDone after {0} iterations. Total running time: {1:>8.4}s"
              .format(m.iteration, total - start))
    return m


def run_multiple_models(models: Sequence[HMM], maxiter=1000, verbose=False):
    """
    Runs iterate() on a list of models with same data and returns the one assigning highest
    likelihood to it.
    """
    winner = None
    procs = available_cpu_count()
    runs = len(models)
    with ProcessPoolExecutor(max_workers=procs) as ex:
        iterations = [maxiter] * runs
        verbose = [False] * runs
        max_ll = - np.inf
        if verbose:
            print("Starting {0} runs using {1} processes (maxiter={2})".
                  format(runs, procs, maxiter))
        for m in ex.map(iterate, models, iterations, verbose):
            if m.ll > max_ll:
                max_ll = m.ll
                winner = m
                if verbose:
                    print("Worker done: new winner with log likelihood = {0}".
                          format(m.ll))
            elif verbose:
                print("Worker done.")
    return winner


def viterbi_path(d: Data, m: HMM) -> Array:
    """
    TODO: Returns the sequence of states maximizing the expected number of
          correct states.
    """

    path = np.ndarray((d.L, ))
    return path


def save(m: HMM, filename: str = ''):
    return False

# def check_data(d: Data, m: HMM=None) -> bool:
#     assert(d.Y.shape == (d.L,))
#     if m is not None:
#         assert(m.A.shape == (m.N, m.N))
#         assert(m.B.shape == (m.N, d.M))
#         assert(map(is_row_stochastic, [m.p, m.A, m.B]).all())
