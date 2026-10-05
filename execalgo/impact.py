"""The power-law deformation family and its derivatives.

The nonlinear impact function is f(x; eps) = sign(x) |x|^{1-eps}. At
eps = 0 it is linear (the exactly solvable baseline); at eps = 1/2 it is
the square-root law. Only positive trading rates occur on the solution
branch, but the signed form keeps Newton steps well-defined.
"""

import numpy as np


def impact(rate, eps):
    magnitude = np.abs(rate)
    return np.sign(rate) * magnitude ** (1.0 - eps)


def impact_slope(rate, eps):
    """f_x(x; eps), the derivative with respect to x."""
    return (1.0 - eps) * np.abs(rate) ** (-eps)


def impact_curvature(rate, eps):
    """f_xx(x; eps), the second derivative with respect to x."""
    magnitude = np.abs(rate)
    return -eps * (1.0 - eps) * np.sign(rate) * magnitude ** (-eps - 1.0)
