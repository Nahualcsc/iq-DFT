"""Linear-response xc kernel of iq-DFT.

The matrix of xc derivatives at I = Q = 0, Eq. (16),

    F_xc = [[dV_xc/dI,   dV_xc/dQ  ],
            [dPsi_xc/dI, dPsi_xc/dQ]],

is related to the KS and interacting Onsager matrices by the Dyson-like
equation L = L_s + L_s F_xc L, Eq. (17), i.e. F_xc = L_s^{-1} - L^{-1}, Eq. (18).
"""
import numpy as np
from scipy.optimize import brentq

from . import siam, ssm


def xc_kernel(Ls, L):
    """F_xc = L_s^{-1} - L^{-1}, Eq. (18)."""
    return np.linalg.inv(Ls) - np.linalg.inv(L)


def dyson(Ls, F):
    """Interacting Onsager matrix from KS matrix and xc kernel:
    L = (L_s^{-1} - F_xc)^{-1}, Eq. (17); equivalent to Eqs. (21)."""
    return np.linalg.inv(np.linalg.inv(Ls) - F)


def unpack(F):
    """(dV_xc/dI, dV_xc/dQ = dPsi_xc/dI, dPsi_xc/dQ) from F_xc."""
    return F[..., 0, 0], F[..., 0, 1], F[..., 1, 1]


def xc_derivatives_ssm(n, T, gamma, U):
    """Analytic parametrization of the xc derivatives as functionals of the
    density: Eqs. (29)-(30) evaluated at the SSM gates v_s(n), v(n), Eq. (31).
    Returns F_xc with shape n.shape + (2, 2)."""
    n = np.asarray(n, dtype=float)
    Ls = siam.L_ks(ssm.vs_of_n(n, T), T, gamma)
    L = siam.L_mbm(ssm.v_of_n(n, T, U), n, T, gamma, U)
    return xc_kernel(Ls, L)


def _invert(func, target, scale):
    """Solve func(x) = target for a monotonically decreasing func."""
    a, b = -scale, scale
    while func(a) < target:
        a *= 2.0
    while func(b) > target:
        b *= 2.0
    return brentq(lambda x: func(x) - target, a, b, xtol=1e-14, rtol=1e-15, maxiter=500)


def vs_of_n_exact(n, T, gamma):
    """Numerical inverse of the KS equilibrium density n_s(v_s) (SIAM, gamma > 0)."""
    scale = 10.0 * (T + gamma)
    return np.vectorize(lambda x: _invert(
        lambda vs: siam.ks_density_eq(vs, T, gamma), x, scale))(n)


def v_of_n_exact(n, T, gamma, U):
    """Numerical inverse of the many-body-model equilibrium density n(v)."""
    scale = 10.0 * (T + gamma + U)
    return np.vectorize(lambda x: _invert(
        lambda v: siam.mbm_density_eq(v, T, gamma, U), x, scale))(n)


def xc_derivatives_exact(n, T, gamma, U):
    """Exact (numerical) reverse engineering of the xc derivatives of the
    many-body model: same as xc_derivatives_ssm but with the gate-density
    relations obtained by numerically inverting Eq. (27a) and Eq. (25a)."""
    n = np.asarray(n, dtype=float)
    Ls = siam.L_ks(vs_of_n_exact(n, T, gamma), T, gamma)
    L = siam.L_mbm(v_of_n_exact(n, T, gamma, U), n, T, gamma, U)
    return xc_kernel(Ls, L)
