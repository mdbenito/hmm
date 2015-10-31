#############################################################################
# Data R/W and synthetic datasets
#
#############################################################################
from h5py import File


# Dumb containers to have dot notation instead of the cumbersome dictionary
# notation. (Python Cookbook §4.8)
class Data:
    """
    Contains:
        L = Number of observations
        M = Number of symbols which may be observed
        V = { 0, 1, ..., M-1} = possible observations
        Y = {Y_0, ..., Y_{L-1} } = sequence of observations

    """

    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def load(filename: str = ''):
    with File(filename, 'r') as f:
        [M, L, Y] = do_stuff()  # ...
    return Data(M=M, L=L, Y=Y)


def save(filename: str = ''):
    return False
