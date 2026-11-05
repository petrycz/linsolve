"""End-to-end demo (Stage 5).

Runs once you've implemented the solvers and the circuit stamps. It solves a
textbook system three ways, shows LU factor-once/solve-many, and solves the
example resistor network.

    python examples/demo.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from linsolve import gaussian_solve, lu_factor, lu_solve, Circuit


def main():
    # ---- 1. one system, solved directly ----
    A = np.array([[2., 1., -1.],
                  [-3., -1., 2.],
                  [-2., 1., 2.]])
    b = np.array([8., -11., -3.])
    x = gaussian_solve(A, b)
    print("A x = b   ->  x =", np.round(x, 6), "   (expected [2, 3, -1])")

    # ---- 2. LU: factor once, solve three different right-hand sides ----
    print("\nLU factor once, solve many:")
    P, L, U = lu_factor(A)
    for rhs in ([8., -11., -3.], [1., 0., 0.], [5., 5., 5.]):
        x = lu_solve(P, L, U, np.array(rhs))
        ok = np.allclose(A @ x, rhs)
        print(f"  b = {str(rhs):<18} ->  x = {np.round(x, 4)}   (A x == b: {ok})")

    # ---- 3. the resistor network ----
    print("\nResistor network (a tiny SPICE):")
    c = Circuit(4)
    c.set_ground(0)
    c.add_current_source(1, 1.5)      # 1.5 A into node 1
    c.add_resistor(1, 2, 2.0)
    c.add_resistor(2, 3, 1.0)
    c.add_resistor(3, 0, 4.0)
    c.add_resistor(1, 0, 5.0)
    volts = c.solve()
    for node in sorted(volts):
        label = " (ground)" if node == c.ground else ""
        print(f"  node {node}: {volts[node]:+.4f} V{label}")


if __name__ == "__main__":
    main()
