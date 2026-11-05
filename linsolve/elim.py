"""Gaussian elimination: turn A x = b into an upper-triangular system and
solve it, with and without partial pivoting.

Stages 1-3 live here. Implement one function at a time and run that stage's
tests. See GUIDE.md.
"""
import numpy as np


def forward_eliminate(A, b):
    """Stage 1 -- TASK. Eliminate below each pivot so A becomes upper-triangular
    U and b becomes the matching right-hand side c. No pivoting yet.

    - Work on COPIES: `U = np.array(A, dtype=float)`, likewise for c. Do not
      mutate the caller's arrays.
    - For each pivot column k, and each row i below it, subtract
      (U[i,k]/U[k,k]) times row k from row i (and do the same to c).
    - Return (U, c).

    See GUIDE.md Stage 1.
    """
    raise NotImplementedError("Stage 1: forward_eliminate")


def back_substitute(U, c):
    """Stage 2 -- TASK. Solve an already upper-triangular system U x = c, from
    the bottom row upward:

        x[i] = ( c[i] - sum_{j>i} U[i,j] * x[j] ) / U[i,i]

    Return the solution vector x. See GUIDE.md Stage 2.
    """
    raise NotImplementedError("Stage 2: back_substitute")


def gaussian_solve(A, b, pivot=True):
    """Stage 3 -- TASK. Full solve, combining elimination with back-substitution.

    With pivot=True, before eliminating in column k, swap in the row (at or
    below k) whose entry in column k has the largest magnitude -- "partial
    pivoting". This avoids dividing by a tiny or zero pivot and keeps the
    arithmetic numerically stable. Swap the matching entries of b too.

    - Raise ValueError if the column has no usable (nonzero) pivot -- the matrix
      is singular.
    - You can reuse back_substitute for the final step.

    See GUIDE.md Stage 3.
    """
    raise NotImplementedError("Stage 3: gaussian_solve")
