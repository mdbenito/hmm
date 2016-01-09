from data import Data


class HMM:
    """ Dummy (TODO: does this add one indirection? """

    def __init__(self, d: Data, N: int, **kwds):
        self.__dict__.update(kwds)

    def forward(m):
        raise RuntimeError("Calling base model's forward()")

    def backward(m):
        raise RuntimeError("Calling base model's backward()")

    def posteriors(m):
        raise RuntimeError("Calling base model's posteriors()")

    def estimate(m):
        raise RuntimeError("Calling base model's estimate()")
