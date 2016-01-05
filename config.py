# Probably not much here...
import numpy as np

eps = 1e-10              # Precision to use when testing with np.isclose()
test_eps = 1e-1          # Precision for unit tests
iteration_margin = 1e-8  # Minimal percentage increase in log likelihood before stopping

# For the type hints (FIXME: not working)
Scalar = np.float64
Vector = Matrix = Array = np.ndarray
