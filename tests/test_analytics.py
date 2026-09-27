"""Checks of the analytic expressions against brute-force numerics."""
import numpy as np
import pytest
from scipy.integrate import quad

from iqdft import siam, ssm, xc
from iqdft.special import trigamma

GAMMA = 1.0


def fermi(x):
    return 0.5 * (1.0 - np.tanh(0.5 * x))


def quad_currents(A, V, Psi, T):
    """(n, I, W, Q) of Eqs. (25) by direct integration for spectral function A."""
    TL, TR = T * (1 + Psi / 2), T * (1 - Psi / 2)
    fL = lambda w: fermi((w - V / 2) / TL)
    fR = lambda w: fermi((w + V / 2) / TR)
    kw = dict(limit=4000, points=[-V / 2, V / 2, 0.0])
    lim = 4000.0
    # the density integrand decays only as 1/w^2: integrate the tails analytically
    n = quad(lambda w: (fL(w) + fR(w)) * A(w), -lim, lim, **kw)[0] / (2 * np.pi)
    I = GAMMA / 2 * quad(lambda w: (fL(w) - fR(w)) * A(w), -lim, lim, **kw)[0] / (2 * np.pi)
    W = GAMMA / 2 * quad(lambda w: (fL(w) - fR(w)) * w * A(w), -lim, lim, **kw)[0] / (2 * np.pi)
    return n, I, W, W - V / 2 * I


def lorentzian(x0):
    return lambda w: GAMMA / ((w - x0) ** 2 + GAMMA ** 2 / 4)


def test_trigamma_against_mpmath():
    mpmath = pytest.importorskip("mpmath")
    z = np.array([0.5 + 0.01j, 0.58 - 3.8j, 2.3 + 40j, 0.6 + 0.0j, 15.0 - 200.0j, 0.5 + 1e-6j])
    ref = np.array([complex(mpmath.polygamma(1, complex(zi))) for zi in z])
    assert np.allclose(trigamma(z), ref, rtol=1e-13, atol=0)


@pytest.mark.parametrize("vs,V,Psi,T", [(0.7, 0.4, 0.3, 1.3), (-1.5, 1.0, -0.5, 1.3),
                                         (0.0, 0.0, 0.6, 2.0), (2.0, -3.0, 0.0, 0.7)])
def test_ks_currents_against_integration(vs, V, Psi, T):
    n, I, W, Q = siam.ks_densities(vs, V, Psi, T, GAMMA)
    nq, Iq, Wq, Qq = quad_currents(lorentzian(vs), V, Psi, T)
    assert abs(n - nq) < 1e-3          # truncated 1/w^2 tails of the density
    assert np.allclose([I, W, Q], [Iq, Wq, Qq], rtol=1e-8, atol=1e-10)


def test_mbm_currents_against_integration():
    v, V, Psi, T, U = -2.0, 0.8, 0.4, 1.5, 3.0
    n, I, W, Q = siam.mbm_densities(v, V, Psi, T, GAMMA, U)
    A = lambda w: (1 - n / 2) * lorentzian(v)(w) + n / 2 * lorentzian(v + U)(w)
    nq, Iq, Wq, Qq = quad_currents(A, V, Psi, T)
    assert abs(n - nq) < 1e-3
    assert np.allclose([I, W, Q], [Iq, Wq, Qq], rtol=1e-8, atol=1e-10)


def finite_difference_L(currents, h=1e-5):
    """Onsager matrix d(I, Q)/d(V, Psi) at V = Psi = 0 by central differences."""
    dV = (np.array(currents(h, 0.0)) - np.array(currents(-h, 0.0))) / (2 * h)
    dP = (np.array(currents(0.0, h)) - np.array(currents(0.0, -h))) / (2 * h)
    return np.array([[dV[1], dP[1]], [dV[3], dP[3]]])


@pytest.mark.parametrize("vs,T", [(0.0, 1.0), (-3.0, 0.5), (4.0, 10.0), (1.2, 3.0)])
def test_M_matrix_is_derivative_of_currents(vs, T):
    L = finite_difference_L(lambda V, P: siam.ks_densities(vs, V, P, T, GAMMA))
    assert np.allclose(siam.M_matrix(vs, T, GAMMA), L, rtol=1e-7, atol=1e-10)


@pytest.mark.parametrize("v,T,U", [(-4.0, 1.0, 8.0), (-1.0, 3.0, 4.0), (2.0, 1.0, 12.0)])
def test_L_mbm_is_derivative_of_currents(v, T, U):
    n = siam.mbm_density_eq(v, T, GAMMA, U)
    L = finite_difference_L(lambda V, P: siam.mbm_densities(v, V, P, T, GAMMA, U))
    assert np.allclose(siam.L_mbm(v, n, T, GAMMA, U), L, rtol=1e-7, atol=1e-10)


def test_M_matrix_against_integration():
    v, T = 0.9, 1.7
    mf = lambda w: np.exp(w / T) / (T * (np.exp(w / T) + 1) ** 2)  # -f'
    pref = GAMMA ** 2 / (4 * np.pi)
    Lint = [pref * quad(lambda w: w ** k * mf(w) / ((w - v) ** 2 + GAMMA ** 2 / 4),
                        -200, 200, limit=500, points=[0, v])[0] for k in range(3)]
    M = siam.M_matrix(v, T, GAMMA)
    # sign convention: (I, Q) = L (V, Psi) with Psi = (T_L - T_R)/T
    assert np.allclose([M[0, 0], M[0, 1], M[1, 1]], [Lint[0], Lint[1], Lint[2]], rtol=1e-10)


def test_transport_coefficients_roundtrip():
    G, S, k, T = 0.05, -0.7, 0.2, 2.0
    L = siam.onsager_matrix(G, S, k, T)
    assert np.allclose(siam.transport_coefficients(L, T), (G, S, k))


def test_ssm_gates_invert_ssm_densities():
    T, U = 1.3, 5.0
    n = np.linspace(0.02, 1.98, 50)
    assert np.allclose(siam.ks_density_eq(ssm.vs_of_n(n, T), T, 0.0), n)
    v = ssm.v_of_n(n, T, U)
    x, y = np.exp(-v / T), np.exp(-U / T)
    n_ssm = (2 * x + 2 * x ** 2 * y) / (1 + 2 * x + x ** 2 * y)  # grand-canonical SSM
    assert np.allclose(n_ssm, n, rtol=1e-12)
    # isolated-site limit of the many-body model is the SSM
    assert np.allclose(siam.mbm_density_eq(v, T, 1e-10, U), n, rtol=1e-8)


def test_xc_kernel_properties():
    T, U = 4.0, 8.0
    n = np.linspace(0.05, 1.95, 30)
    F = xc.xc_derivatives_exact(n, T, GAMMA, U)
    assert np.allclose(F[..., 0, 1], F[..., 1, 0])      # Onsager, Eq. (19)
    # Dyson with the exact kernel reproduces the many-body model exactly
    Ls = siam.L_ks(xc.vs_of_n_exact(n, T, GAMMA), T, GAMMA)
    L = siam.L_mbm(xc.v_of_n_exact(n, T, GAMMA, U), n, T, GAMMA, U)
    assert np.allclose(xc.dyson(Ls, F), L, rtol=1e-9)
    # closed forms of Eqs. (21) agree with the matrix Dyson equation
    Gs, Ss, ks = siam.transport_coefficients(Ls, T)
    dVdI, dVdQ, dPdQ = xc.unpack(F)
    k = ks / (1 - T * dPdQ * ks)
    S = (Ss - ks * dVdQ) / (1 - T * dPdQ * ks)
    G = Gs / (1 - (dVdI + T * S ** 2 / k - T * Ss ** 2 / ks) * Gs)
    assert np.allclose(siam.transport_coefficients(L, T), (G, S, k), rtol=1e-9)



def test_ssm_density_closed_form():
    T, U = 0.7, 6.0
    n = np.linspace(0.01, 1.99, 50)
    assert np.allclose(ssm.n_of_v(ssm.v_of_n(n, T, U), T, U), n, rtol=1e-12)


@pytest.mark.parametrize("T", [1.0, 3.0])
def test_schemes(T):
    """"exact-ks" is a self-consistent KS-SIAM calculation whose density and
    iq-DFT coefficients coincide with those of the published ("paper")
    scheme; both equal the MBM coefficients evaluated at the SSM density."""
    from iqdft import dft
    U = 8.0
    v = np.linspace(-16.0, 8.0, 40)
    e = dft.linear_response(v, T, GAMMA, U, scheme="exact-ks")
    p = dft.linear_response(v, T, GAMMA, U, scheme="paper")
    # self-consistency of the KS-SIAM (coupling gamma) with v_Hxc = v_s^KS(n) - v^SSM(n)
    assert np.allclose(siam.ks_density_eq(v + e["v_hxc"], T, GAMMA), e["n"], rtol=1e-10)
    # (the xc derivatives themselves differ, since the KS potentials differ)
    for key in ("n", "G", "S", "kappa"):
        assert np.allclose(e[key], p[key], rtol=1e-9, atol=1e-12), key
    L = siam.L_mbm(v, ssm.n_of_v(v, T, U), T, GAMMA, U)
    assert np.allclose(siam.transport_coefficients(L, T), (e["G"], e["S"], e["kappa"]),
                       rtol=1e-9, atol=1e-12)
    # "ssm" uses the KS-SIAM density with Eq. (32): self-consistent, different density
    s = dft.linear_response(v, T, GAMMA, U, scheme="ssm")
    assert np.allclose(siam.ks_density_eq(v + s["v_hxc"], T, GAMMA), s["n"], rtol=1e-10)
