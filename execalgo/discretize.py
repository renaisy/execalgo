"""Composite trapezoidal discretization of the cost operator.

The continuum operators are

    (P h)(t) = int_0^t G(t-s) h(s) ds,      (Q h)(t) = int_t^1 G(s-t) h(s) ds,

and the linear cost operator is A = 2 eta I + P + Q. On a uniform grid the
trapezoidal rule turns P and Q into the triangular matrices assembled below,
and the share-constrained linear solve becomes the bordered system

    [ A   -1 ] [ v     ]   [ rhs   ]
    [ w^T  0 ] [ ell   ] = [ share ],

which is factored once and reused at every order of the expansion.

Ported from gordonritter/perturbative-impact (MIT license).
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import lu_factor, lu_solve


@dataclass(frozen=True)
class Grid:
    """A uniform grid on [0, horizon] with the discretized kernel operators."""

    nodes: np.ndarray
    spacing: float
    weights: np.ndarray
    past: np.ndarray
    future: np.ndarray

    @property
    def size(self):
        return self.nodes.size

    @property
    def horizon(self):
        return float(self.nodes[-1])

    def cost_operator(self, eta, risk=0.0):
        operator = 2.0 * eta * np.eye(self.size) + self.past + self.future
        if risk:
            operator = operator + self.risk_operator(risk)
        return operator

    def risk_operator(self, risk):
        """The Hessian of risk * int (share - int_0^t v)^2 dt.

        Kernel 2 * risk * (horizon - max(s, t)), symmetric positive
        semidefinite and independent of eps, so it only folds into the
        linear cost operator. Columns carry the quadrature weights.
        """
        tail = self.horizon - np.maximum(self.nodes[:, None], self.nodes[None, :])
        return 2.0 * risk * tail * self.weights[None, :]

    def risk_vector(self, risk, share):
        """The constant right-hand-side part of the risk gradient, 2 * risk * share * (T - t)."""
        return 2.0 * risk * share * (self.horizon - self.nodes)

    def inner(self, left, right):
        """The discrete L^2 pairing given by the quadrature weights."""
        return float(self.weights @ (left * right))

    def norm(self, values):
        return float(np.sqrt(self.inner(values, values)))


def make_grid(size, kernel, horizon=1.0):
    """Assemble the grid and the trapezoidal images of P and Q.

    Row i of `past` integrates over [t_0, t_i] and row i of `future` over
    [t_i, t_{n-1}]. Each row therefore has its own trapezoidal weights, with
    half weight at the two ends of its own interval. The first row of `past`
    and the last row of `future` integrate over a degenerate interval and
    vanish.
    """
    nodes = np.linspace(0.0, horizon, size)
    spacing = nodes[1] - nodes[0]
    kernel_matrix = kernel(nodes[:, None] - nodes[None, :])

    index = np.arange(size)
    past_weights = np.full((size, size), spacing)
    past_weights[:, 0] = spacing / 2.0
    past_weights[index, index] = spacing / 2.0
    past = np.tril(kernel_matrix * past_weights)
    past[0, :] = 0.0

    future_weights = np.full((size, size), spacing)
    future_weights[:, -1] = spacing / 2.0
    future_weights[index, index] = spacing / 2.0
    future = np.triu(kernel_matrix * future_weights)
    future[-1, :] = 0.0

    weights = np.full(size, spacing)
    weights[0] = weights[-1] = spacing / 2.0
    return Grid(nodes, spacing, weights, past, future)


class ConstrainedSolver:
    """The bordered system for A at a fixed slippage coefficient.

    One LU factorization serves the baseline and every perturbative
    coefficient, which is the computational point of the expansion.
    """

    def __init__(self, grid, eta, risk=0.0):
        size = grid.size
        bordered = np.zeros((size + 1, size + 1))
        bordered[:size, :size] = grid.cost_operator(eta, risk)
        bordered[:size, size] = -1.0
        bordered[size, :size] = grid.weights
        self._factorization = lu_factor(bordered)
        self._size = size

    def solve(self, source, share):
        """Solve A v - ell = source subject to w^T v = share."""
        solution = lu_solve(self._factorization, np.concatenate([source, [share]]))
        return solution[:-1], float(solution[-1])
