"""Self-consistent iq-DFT calculation for the SIAM in linear response
(Sec. IV B of the paper).

Three schemes are available; they differ in how the single-site-model (SSM)
ideas are combined with the KS system of the SIAM (lead coupling gamma):

``"exact-ks"`` (default)
    KS-SIAM with coupling gamma.  The non-interacting part of the functionals
    is treated exactly and only the interacting gate is taken from the SSM:
    v_Hxc(n) = v_s^KS(n) - v^SSM(n) and F_xc = R_s(v_s^KS(n)) - R(v^SSM(n), n),
    where v_s^KS(n) is the (numerical) inverse of the KS-SIAM density.
    The self-consistency condition n = n_s(v + v_Hxc(n)) reduces to
    v^SSM(n) = v, so the density is the SSM density, and the Dyson equation
    gives L = L_MBM(v, n_SSM(v)).
``"paper"``
    Setting used for the figures of the paper: density from the uncoupled KS
    site with the SSM Hxc potential of Eq. (32), analytic kernel of
    Eqs. (30)-(31), KS coefficients with coupling gamma at v_s = v_s^SSM(n).
    Gives the same density and iq-DFT coefficients as ``"exact-ks"``; the KS
    (LB-DFT) coefficients correspond to the KS potential v_s^SSM(n).
``"ssm"``
    KS-SIAM with coupling gamma and the analytic SSM expressions inserted
    directly: Eq. (32) for v_Hxc and Eqs. (30)-(31) for the kernel.
"""
import numpy as np
from scipy.optimize import brentq

from . import siam, ssm, xc

SCHEMES = ("exact-ks", "paper", "ssm")
_EPS = 1e-12


def solve_density(v, T, gamma, U, ks_gamma=None):
    """Self-consistent density n = n_s(v + v_Hxc^SSM(n)) with the SSM Hxc
    potential of Eq. (32) and a KS level with coupling ``ks_gamma``
    (default ``gamma``; ``ks_gamma=0`` gives the uncoupled KS site of the
    ``"paper"`` scheme)."""
    kg = gamma if ks_gamma is None else ks_gamma

    def solve_one(vi):
        g = lambda n: n - siam.ks_density_eq(vi + ssm.v_hxc(n, T, U), T, kg)
        return brentq(g, _EPS, 2.0 - _EPS, xtol=1e-15, rtol=1e-15, maxiter=500)

    return np.vectorize(solve_one)(v)


def linear_response(v, T, gamma, U, scheme="exact-ks"):
    """iq-DFT linear-response transport coefficients, Eq. (21), with the
    SSM-based xc approximations (see the module docstring for ``scheme``).

    Returns a dict with the self-consistent density ``n``, KS potential
    ``vs`` and ``v_hxc``, the KS coefficients (``Gs``, ``Ss``, ``kappas``,
    ``ZTs`` = LB-DFT), the xc derivatives (``dVxc_dI``, ``dVxc_dQ``,
    ``dPsixc_dQ``) and the iq-DFT coefficients (``G``, ``S``, ``kappa``,
    ``ZT``).  G and kappa are in atomic units (multiply by pi for units of G0).
    """
    v = np.asarray(v, dtype=float)
    if scheme == "exact-ks":
        n = ssm.n_of_v(v, T, U)
        vs = xc.vs_of_n_exact(n, T, gamma)
        Ls = siam.L_ks(vs, T, gamma)
        F = xc.xc_kernel(Ls, siam.L_mbm(ssm.v_of_n(n, T, U), n, T, gamma, U))
    elif scheme in ("paper", "ssm"):
        n = solve_density(v, T, gamma, U, ks_gamma=0.0 if scheme == "paper" else gamma)
        vs = v + ssm.v_hxc(n, T, U)
        Ls = siam.L_ks(vs, T, gamma)
        F = xc.xc_derivatives_ssm(n, T, gamma, U)
    else:
        raise ValueError(f"scheme must be one of {SCHEMES}")
    L = xc.dyson(Ls, F)
    Gs, Ss, ks = siam.transport_coefficients(Ls, T)
    G, S, k = siam.transport_coefficients(L, T)
    dVdI, dVdQ, dPdQ = xc.unpack(F)
    return dict(n=n, vs=vs, v_hxc=vs - v,
                Gs=Gs, Ss=Ss, kappas=ks, ZTs=siam.figure_of_merit(Gs, Ss, ks, T),
                dVxc_dI=dVdI, dVxc_dQ=dVdQ, dPsixc_dQ=dPdQ,
                G=G, S=S, kappa=k, ZT=siam.figure_of_merit(G, S, k, T))


def mbm_linear_response(v, T, gamma, U):
    """Reference many-body-model results, Eqs. (25), (26), (29)."""
    v = np.asarray(v, dtype=float)
    n = siam.mbm_density_eq(v, T, gamma, U)
    G, S, k = siam.transport_coefficients(siam.L_mbm(v, n, T, gamma, U), T)
    return dict(n=n, G=G, S=S, kappa=k, ZT=siam.figure_of_merit(G, S, k, T))
