"""Single impurity Anderson model (SIAM) in the wide-band limit.

Symmetric coupling gamma_L = gamma_R = gamma/2, symmetric bias
mu_{L/R} = +/- V/2 and symmetric thermal gradient T_{L/R} = T (1 +/- Psi/2).
Atomic units (e = hbar = k_B = 1); the conductance quantum is G0 = 1/pi.
Currents flowing *into* the dot from the left lead are positive.

Two systems are provided:

* the Kohn-Sham (KS) system: a single Lorentzian level at v_s, Eq. (27) of
  the paper;
* the many-body model (MBM) of Eq. (26): two Lorentzians at v and v+U with
  weights (1 - n/2) and n/2.

All functions accept numpy arrays (broadcasting) unless noted otherwise.
"""
import numpy as np

from .special import digamma, trigamma

G0 = 1.0 / np.pi  # quantum of conductance (spin degenerate) in atomic units


# ---------------------------------------------------------------------------
# Densities and currents at finite bias / thermal gradient
# ---------------------------------------------------------------------------
def ks_densities(vs, Vs, Psis, T, gamma):
    """Density n, charge current I, energy current W and heat current Q of the
    non-interacting (KS) level, Eqs. (27) and (A16) of the paper:

        W = gamma^2/(4 pi) [Re psi(z_L) - Re psi(z_R)
                            + log((1 + Psi/2)/(1 - Psi/2))] + v_s I,
        Q = W - (V/2) I.
    """
    vs, Vs, Psis = np.broadcast_arrays(*map(np.asarray, (vs, Vs, Psis)))
    TL = T * (1.0 + 0.5 * Psis)
    TR = T * (1.0 - 0.5 * Psis)
    psiL = digamma(0.5 + (0.5 * gamma + 1j * (vs - 0.5 * Vs)) / (2.0 * np.pi * TL))
    psiR = digamma(0.5 + (0.5 * gamma + 1j * (vs + 0.5 * Vs)) / (2.0 * np.pi * TR))
    n = 1.0 - (psiR.imag + psiL.imag) / np.pi
    I = gamma / (2.0 * np.pi) * (psiR.imag - psiL.imag)
    W = (gamma ** 2 / (4.0 * np.pi)
         * (psiL.real - psiR.real + np.log((1.0 + 0.5 * Psis) / (1.0 - 0.5 * Psis)))
         + vs * I)
    Q = W - 0.5 * Vs * I
    return n, I, W, Q


def mbm_densities(v, V, Psi, T, gamma, U):
    """(n, I, W, Q) of the many-body model, Eqs. (25)+(26).

    Since A(omega) depends linearly on n, the self-consistency
    n = (1 - n/2) n_0 + (n/2) n_U is solved in closed form.
    """
    n0, I0, W0, _ = ks_densities(v, V, Psi, T, gamma)
    nU, IU, WU, _ = ks_densities(np.asarray(v) + U, V, Psi, T, gamma)
    n = n0 / (1.0 + 0.5 * (n0 - nU))
    I = (1.0 - 0.5 * n) * I0 + 0.5 * n * IU
    W = (1.0 - 0.5 * n) * W0 + 0.5 * n * WU
    Q = W - 0.5 * np.asarray(V) * I
    return n, I, W, Q


def ks_density_eq(vs, T, gamma):
    """Equilibrium (V = Psi = 0) KS density n_s(v_s).  gamma = 0 gives the
    isolated site, n = 2 f(v_s)."""
    return 1.0 - 2.0 / np.pi * digamma(
        0.5 + (0.5 * gamma + 1j * np.asarray(vs)) / (2.0 * np.pi * T)).imag


def mbm_density_eq(v, T, gamma, U):
    """Equilibrium density of the many-body model, n(v)."""
    n0 = ks_density_eq(v, T, gamma)
    nU = ks_density_eq(np.asarray(v) + U, T, gamma)
    return n0 / (1.0 + 0.5 * (n0 - nU))


# ---------------------------------------------------------------------------
# Linear response
# ---------------------------------------------------------------------------
def M_matrix(v, T, gamma):
    """Onsager matrix of a single Lorentzian level at v, Eqs. (A17).

    Returns an array of shape v.shape + (2, 2) with
    (I, Q) = M (V, Psi) to first order.
    """
    v = np.asarray(v, dtype=float)
    z0 = 0.5 * gamma + 1j * v
    p1 = trigamma(0.5 + z0 / (2.0 * np.pi * T))
    pref = gamma / (4.0 * np.pi ** 2 * T)
    M11 = pref * (1j * p1).imag
    M12 = pref * (z0 * p1).imag
    M22 = (-gamma ** 2 / (8.0 * np.pi ** 2 * T) * (z0 * p1).real
           + v * M12 + gamma ** 2 / (4.0 * np.pi))
    return np.stack([np.stack([M11, M12], -1), np.stack([M12, M22], -1)], -2)


def L_ks(vs, T, gamma):
    """KS Onsager matrix L_s(v_s), Eq. (28)."""
    return M_matrix(vs, T, gamma)


def L_mbm(v, n, T, gamma, U):
    """Many-body-model Onsager matrix, Eq. (29):
    L(v) = (1 - n/2) M(v) + (n/2) M(v + U)."""
    w = (0.5 * np.asarray(n, dtype=float))[..., None, None]
    return (1.0 - w) * M_matrix(v, T, gamma) + w * M_matrix(np.asarray(v) + U, T, gamma)


def transport_coefficients(L, T):
    """(G, S, kappa) from an Onsager matrix, Eqs. (6):
    G = L11, S = -L12 / (T L11), kappa = (L22 - L12^2 / L11) / T.
    G and kappa are in atomic units (divide by G0 for units of G0)."""
    L11, L12, L22 = L[..., 0, 0], L[..., 0, 1], L[..., 1, 1]
    G = L11
    S = -L12 / (T * L11)
    kappa = (L22 - L12 ** 2 / L11) / T
    return G, S, kappa


def figure_of_merit(G, S, kappa, T):
    """Electronic figure of merit ZT = T G S^2 / kappa."""
    return T * G * S ** 2 / kappa


def onsager_matrix(G, S, kappa, T):
    """Inverse of transport_coefficients, Eq. (5)."""
    G, S, kappa = np.broadcast_arrays(G, S, kappa)
    L12 = -T * G * S
    return np.stack([np.stack([G, L12], -1),
                     np.stack([L12, T * kappa + T ** 2 * G * S ** 2], -1)], -2)
