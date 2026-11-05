"""Stage 4: LU factorization.

    pytest tests/test_lu.py
"""
import numpy as np
import pytest
from linsolve.lu import lu_factor, lu_solve


def test_factor_reconstructs_PA_equals_LU():
    rng = np.random.default_rng(1)
    for _ in range(20):
        A = rng.standard_normal((5, 5))
        P, L, U = lu_factor(A)
        assert np.allclose(P @ A, L @ U, atol=1e-10)


def test_L_is_unit_lower_and_U_is_upper():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((5, 5))
    P, L, U = lu_factor(A)
    assert np.allclose(np.diag(L), 1.0)              # unit diagonal
    assert np.allclose(np.triu(L, 1), 0.0)           # nothing above L's diagonal
    assert np.allclose(np.tril(U, -1), 0.0)          # nothing below U's diagonal


def test_P_is_a_permutation():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((5, 5))
    P, L, U = lu_factor(A)
    assert np.allclose(P.sum(axis=0), 1) and np.allclose(P.sum(axis=1), 1)
    assert set(np.unique(P)) <= {0.0, 1.0}


def test_lu_solve_matches_numpy():
    rng = np.random.default_rng(4)
    A = rng.standard_normal((6, 6))
    P, L, U = lu_factor(A)
    for _ in range(5):
        b = rng.standard_normal(6)
        assert np.allclose(lu_solve(P, L, U, b), np.linalg.solve(A, b), atol=1e-8)


def test_factor_once_solve_many():
    # the point of LU: one factorization, reused across many right-hand sides
    rng = np.random.default_rng(5)
    A = rng.standard_normal((7, 7))
    P, L, U = lu_factor(A)                 # done ONCE
    B = rng.standard_normal((7, 4))        # four different b vectors
    for j in range(B.shape[1]):
        x = lu_solve(P, L, U, B[:, j])
        assert np.allclose(A @ x, B[:, j], atol=1e-8)


def test_factor_handles_zero_leading_pivot():
    # needs a row swap at the very first step
    A = np.array([[0., 2., 1.],
                  [1., 0., 0.],
                  [0., 1., 1.]])
    P, L, U = lu_factor(A)
    assert np.allclose(P @ A, L @ U, atol=1e-12)


def test_factor_raises_on_singular():
    A = np.array([[1., 2.], [2., 4.]])
    with pytest.raises(ValueError):
        lu_factor(A)
