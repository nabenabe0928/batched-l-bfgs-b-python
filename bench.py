import time

import numpy as np
import torch

from batched_lbfgsb import batched_lbfgsb
from scipy.optimize import fmin_l_bfgs_b


k_batched = torch.rand(50, 100, 100)
C = torch.rand(100, 100)


def rastrigin_and_grad(x: np.ndarray, batched: bool) -> tuple[np.ndarray, np.ndarray]:
    if batched:
        k = k_batched
    else:
        k = k_batched[0]

    A = 10.0
    dim = x.shape[-1]
    _2pi_x = 2 * np.pi * x

    f = A * dim + np.sum(x**2 - A * np.cos(_2pi_x), axis=-1)
    g = 2 * x + 2 * np.pi * A * np.sin(_2pi_x)
    k @ C
    return f, g


rng = np.random.RandomState(42)
X0 = rng.random((10, 50)) * 10.24 - 5.12
start = time.time()
batched_lbfgsb(lambda x: rastrigin_and_grad(x, True), x0=X0)
print(f"Batched: {(time.time() - start)*1000:.2f} ms")
start = time.time()
for x0 in X0:
    fmin_l_bfgs_b(lambda x: rastrigin_and_grad(x, False), x0=x0)
print(f"SciPy: {(time.time() - start)*1000:.2f} ms")
