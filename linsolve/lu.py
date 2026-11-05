"""LU factorization with partial pivoting: P A = L U.

Once you have L and U, every new right-hand side b is solved with two cheap
triangular sweeps -- no re-elimination. That's the whole reason LU beats plain
Gaussian elimination when you solve A x = b for many different b.

Stage 4 lives here. See GUIDE.md.
"""
import numpy as np
from .elim import back_substitute


def lu_factor(A):
    """Stage 4 -- TASK. Factor A so that  P @ A == L @ U, where
      - P is a permutation matrix (records the pivoting row swaps),
      - L is unit-lower-triangular (1s on the diagonal; the elimination
        multipliers stored below it),
      - U is upper-triangular (the result of elimination).

    It is the same elimination as Stage 3, but instead of touching b you SAVE
    each multiplier f = U[i,k]/U[k,k] into L[i,k]. On a pivot swap of rows k and
    p, swap those rows in U and P, and swap the ALREADY-stored part of L
    (columns < k) so the multipliers stay attached to their rows.

    - Return (P, L, U). Raise ValueError if A is singular.

    See GUIDE.md Stage 4.
    """
    raise NotImplementedError("Stage 4: lu_factor")


def lu_solve(P, L, U, b):
    """Stage 4 -- TASK. Solve A x = b given the factorization P A = L U.

    Since P A = L U, the system A x = b becomes L U x = P b. Let y = U x:
      1. forward-substitute  L y = P b   (L has a unit diagonal), then
      2. back-substitute      U x = y    (reuse back_substitute).

    Return x. See GUIDE.md Stage 4.
    """
    raise NotImplementedError("Stage 4: lu_solve")
