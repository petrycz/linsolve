"""Stage 5: the resistor-network solver (a tiny SPICE).

    pytest tests/test_circuit.py
"""
import numpy as np
import pytest
from linsolve.circuit import Circuit
from linsolve.lu import lu_factor, lu_solve


def test_single_resistor_ohms_law():
    # 2 A pushed into node 1, which reaches ground through a 3 ohm resistor.
    # V = I * R = 6 V.
    c = Circuit(2)
    c.set_ground(0)
    c.add_current_source(1, 2.0)
    c.add_resistor(1, 0, 3.0)
    v = c.solve()
    assert v[0] == 0.0
    assert np.isclose(v[1], 6.0)


def test_series_resistors():
    # 1 A from a source into node 1; node1 -2ohm- node2 -3ohm- ground.
    # v2 = I*R2 = 3, v1 = I*(R1+R2) = 5.
    c = Circuit(3)
    c.set_ground(0)
    c.add_current_source(1, 1.0)
    c.add_resistor(1, 2, 2.0)
    c.add_resistor(2, 0, 3.0)
    v = c.solve()
    assert np.isclose(v[1], 5.0)
    assert np.isclose(v[2], 3.0)


def test_parallel_resistors_current_divider():
    # 1 A into node 1, two parallel paths to ground (2 ohm || 2 ohm = 1 ohm).
    # v1 = I * 1 = 1 V.
    c = Circuit(2)
    c.set_ground(0)
    c.add_current_source(1, 1.0)
    c.add_resistor(1, 0, 2.0)
    c.add_resistor(1, 0, 2.0)
    v = c.solve()
    assert np.isclose(v[1], 1.0)


def test_conductance_matrix_is_assembled_correctly():
    c = Circuit(3)
    c.add_resistor(1, 2, 1.0)     # g = 1
    c.add_resistor(2, 0, 0.5)     # g = 2
    # G[1,1]=1, G[2,2]=1+2=3, G[1,2]=G[2,1]=-1, G[2,0]=G[0,2]=-2, G[0,0]=2
    assert np.isclose(c.G[1, 1], 1.0)
    assert np.isclose(c.G[2, 2], 3.0)
    assert np.isclose(c.G[1, 2], -1.0) and np.isclose(c.G[2, 1], -1.0)
    assert np.isclose(c.G[0, 2], -2.0)
    # conductance matrices are symmetric by construction
    assert np.allclose(c.G, c.G.T)


def test_reduced_system_drops_ground():
    c = Circuit(3)
    c.set_ground(0)
    c.add_resistor(1, 2, 1.0)
    c.add_resistor(2, 0, 1.0)
    c.add_current_source(1, 1.0)
    Gr, ir, keep = c.reduced_system()
    assert keep == [1, 2]
    assert Gr.shape == (2, 2)
    assert len(ir) == 1 * 2

    # solution must satisfy the full KCL at the non-ground nodes
    v = c.solve()
    assert np.allclose(Gr @ np.array([v[1], v[2]]), ir, atol=1e-10)


def test_solver_is_pluggable_lu_gives_same_answer():
    def lu_based(A, b):
        P, L, U = lu_factor(A)
        return lu_solve(P, L, U, b)

    c = Circuit(4)
    c.set_ground(0)
    c.add_current_source(1, 1.5)
    c.add_resistor(1, 2, 2.0)
    c.add_resistor(2, 3, 1.0)
    c.add_resistor(3, 0, 4.0)
    c.add_resistor(1, 0, 5.0)
    assert c.solve() == pytest.approx(c.solve(solver=lu_based))
