"""Stages 1-3: Gaussian elimination.

    pytest tests/test_elim.py -k forward     # Stage 1
    pytest tests/test_elim.py -k back        # Stage 2
    pytest tests/test_elim.py -k solve       # Stage 3
"""
import numpy as np
import pytest
from linsolve.elim import forward_eliminate, back_substitute, gaussian_solve


def test_forward_makes_upper_triangular():
    A = [[2., 1., -1.],
         [-3., -1., 2.],
         [-2., 1., 2.]]
    b = [8., -11., -3.]
    U, c = forward_eliminate(A, b)
    # everything strictly below the diagonal must be ~0
    assert np.allclose(np.tril(U, -1), 0.0, atol=1e-12)
    # the system must still describe the same solution
    x = np.linalg.solve(np.array(A), np.array(b))
    assert np.allclose(U @ x, c, atol=1e-10)


def test_forward_does_not_mutate_inputs():
    A = np.array([[2., 1.], [1., 3.]])
    b = np.array([1., 2.])
    A0, b0 = A.copy(), b.copy()
    forward_eliminate(A, b)
    assert np.allclose(A, A0) and np.allclose(b, b0)


def test_back_substitute_known_triangular():
    U = [[2., 1., -1.],
         [0., 0.5, 0.5],
         [0., 0., -1.]]
    c = [8., 1., 1.]
    x = back_substitute(U, c)
    assert np.allclose(U @ np.array(x), c, atol=1e-12)


def test_solve_matches_numpy_small():
    A = [[2., 1., -1.], [-3., -1., 2.], [-2., 1., 2.]]
    b = [8., -11., -3.]
    x = gaussian_solve(A, b)
    assert np.allclose(x, [2., 3., -1.], atol=1e-10)   # the textbook answer


def test_solve_matches_numpy_random():
    rng = np.random.default_rng(0)
    for _ in range(20):
        A = rng.standard_normal((6, 6))
        b = rng.standard_normal(6)
        assert np.allclose(gaussian_solve(A, b), np.linalg.solve(A, b), atol=1e-8)


def test_solve_needs_pivoting_when_pivot_is_zero():
    # a zero in the (0,0) slot: no-pivot elimination divides by zero, pivoting saves it
    A = [[0., 1.], [1., 1.]]
    b = [2., 3.]
    x = gaussian_solve(A, b, pivot=True)
    assert np.allclose(x, np.linalg.solve(np.array(A), np.array(b)), atol=1e-12)
    with pytest.raises(Exception):
        gaussian_solve(A, b, pivot=False)


def test_solve_pivoting_is_accurate_on_ill_conditioned():
    # tiny pivot without swapping loses precision; partial pivoting keeps it
    eps = 1e-12
    A = [[eps, 1.], [1., 1.]]
    b = [1., 2.]
    x = gaussian_solve(A, b, pivot=True)
    assert np.allclose(x, np.linalg.solve(np.array(A), np.array(b)), atol=1e-6)


def test_solve_raises_on_singular():
    A = [[1., 2.], [2., 4.]]      # rows are dependent
    b = [1., 2.]
    with pytest.raises(ValueError):
        gaussian_solve(A, b)
