"""A tiny SPICE: solve a resistor network for its node voltages.

The physics is one equation -- Kirchhoff's current law: at every node, the
current flowing out through the resistors equals the current injected there.
Written for all nodes at once that's a linear system  G v = i , where G is the
conductance matrix. You assemble G and i, then hand them to YOUR solver.

Stage 5 lives here. See GUIDE.md.
"""
import numpy as np
from .elim import gaussian_solve


class Circuit:
    def __init__(self, n_nodes):
        self.n = n_nodes
        self.G = np.zeros((n_nodes, n_nodes))   # conductance matrix
        self.i = np.zeros(n_nodes)              # injected currents
        self.ground = 0

    def set_ground(self, node):
        """Pick the reference node, defined to be 0 V. (given)"""
        self.ground = node

    def add_current_source(self, into_node, amps):
        """Inject `amps` of current into `into_node`. (given)"""
        self.i[into_node] += amps

    def add_resistor(self, a, b, R):
        """Stage 5 -- TASK. Stamp a resistor of resistance R between nodes a, b.

        A resistor of conductance g = 1/R contributes to four entries of G:
            G[a,a] += g    G[b,b] += g    G[a,b] -= g    G[b,a] -= g
        (the current from a to b is g*(v_a - v_b); KCL gives these four terms).

        See GUIDE.md Stage 5.
        """
        raise NotImplementedError("Stage 5: Circuit.add_resistor")

    def reduced_system(self):
        """Stage 5 -- TASK. The ground node's voltage is fixed at 0, so it is not
        an unknown: drop its row and column from G and i.

        Return (Gr, ir, keep), where `keep` is the list of non-ground node
        indices (in increasing order) and Gr, ir are G and i restricted to those
        indices. Hint: `np.ix_(keep, keep)` indexes the submatrix.

        See GUIDE.md Stage 5.
        """
        raise NotImplementedError("Stage 5: Circuit.reduced_system")

    def solve(self, solver=gaussian_solve):
        """Assemble the reduced system, solve it with `solver`, and return a
        dict {node: voltage} with the ground node at 0 V. (given)

        Note `solver` defaults to your gaussian_solve but can be any function
        with the signature solver(A, b) -> x -- e.g. an LU-based one.
        """
        Gr, ir, keep = self.reduced_system()
        v_reduced = solver(Gr, ir)
        volts = {self.ground: 0.0}
        for idx, node in enumerate(keep):
            volts[node] = float(v_reduced[idx])
        return volts
