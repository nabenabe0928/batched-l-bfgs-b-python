import time
from typing import Any

import pytest
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from batched_lbfgsb import batched_lbfgsb


def rastrigin_and_grad(x: np.ndarray, *args) -> tuple[np.ndarray, np.ndarray]:
    A = 10.0
    dim = x.shape[-1]
    _2pi_x = 2 * np.pi * x

    f = A * dim + np.sum(x**2 - A * np.cos(_2pi_x), axis=-1)
    g = 2 * x + 2 * np.pi * A * np.sin(_2pi_x)
    return f, g


def X0_and_bounds(dim: int, n_localopts: int) -> tuple[np.ndarray, np.ndarray]:
    R = 5.12
    rng = np.random.RandomState(0)
    X0 = rng.random((n_localopts, dim)) * 2 * R - R
    bounds = np.array([[-R, R]]*dim)
    return X0, bounds


def _verify_results(X0: np.ndarray, kwargs_ours: Any, kwargs_scipy: Any) -> None:
    _, fs1, is1 = batched_lbfgsb(func_and_grad=rastrigin_and_grad, x0=X0, **kwargs_ours)
    fs2 = []
    n_evals2 = []
    n_iters2 = []
    is_converged2 = []
    for x0 in X0:
        _, f, i2 = fmin_l_bfgs_b(rastrigin_and_grad, x0=x0, **kwargs_scipy)
        fs2.append(f.item())
        n_evals2.append(i2["funcalls"])
        n_iters2.append(i2["nit"])
        is_converged2.append(i2["warnflag"] == 0)

    assert np.all(is1["is_converged"] == np.array(is_converged2))
    assert np.all(is1["n_evals"] == np.array(n_evals2))
    assert np.all(is1["n_iterations"] == np.array(n_iters2))
    assert np.allclose(fs1, fs2)


@pytest.mark.parametrize(
    "kwargs_ours,kwargs_scipy", [
        ({}, {}), ({"max_evals": 3}, {"maxfun": 3}), ({"max_iters": 3}, {"maxiter": 3})
    ]
)
def test_batched_lbfgsb(kwargs_ours: Any, kwargs_scipy: Any) -> None:
    dim = 10
    n_localopts = 10
    X0, bounds = X0_and_bounds(dim=dim, n_localopts=n_localopts)
    kwargs_ours.update(bounds=bounds)
    kwargs_scipy.update(bounds=bounds)
    _verify_results(X0, kwargs_ours, kwargs_scipy)


@pytest.mark.parametrize(
    "kwargs_ours,kwargs_scipy", [
        ({}, {}), ({"max_evals": 3}, {"maxfun": 3}), ({"max_iters": 3}, {"maxiter": 3})
    ]
)
@pytest.mark.parametrize("l,u", [(-np.inf, None), (None, np.inf), (-np.inf, np.inf), (None, None)])
def test_batched_lbfgsb_without_bounds(
    kwargs_ours: Any, kwargs_scipy: Any, l: float | None, u: float | None
) -> None:
    dim = 10
    n_localopts = 10
    X0, bounds = X0_and_bounds(dim=dim, n_localopts=n_localopts)
    if l is not None:
        bounds[:, 0] = l
    if u is not None:
        bounds[:, 1] = u
    kwargs_ours.update(bounds=bounds)
    kwargs_scipy.update(bounds=bounds)
    _verify_results(X0, kwargs_ours, kwargs_scipy)
