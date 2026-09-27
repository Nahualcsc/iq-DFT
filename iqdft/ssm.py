"""Single-site model (SSM): an isolated (interacting or non-interacting) site
in contact with a heat and particle bath.  Exact density-gate relations,
Eqs. (31) and (32) of the paper."""
import numpy as np


def vs_of_n(n, T):
    """Non-interacting SSM gate for density n, Eq. (31a): v_s = T log(2/n - 1)."""
    n = np.asarray(n, dtype=float)
    return T * np.log(2.0 / n - 1.0)


def v_of_n(n, T, U):
    """Interacting SSM gate for density n, Eq. (31b).

    v = -U - T log[(dn + sqrt(dn^2 + e^{-U/T}(1 - dn^2))) / (1 - dn)],  dn = n - 1.
    For n < 1 the algebraically identical form
    v = T log[(sqrt(...) - dn) / n] is used to avoid cancellation.
    """
    n = np.asarray(n, dtype=float)
    dn = n - 1.0
    root = np.sqrt(dn ** 2 + np.exp(-U / T) * (1.0 - dn ** 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        upper = -U - T * np.log((dn + root) / (1.0 - dn))
        lower = T * np.log((root - dn) / n)
    return np.where(n >= 1.0, upper, lower)


def v_hxc(n, T, U):
    """Exact Hxc potential of the SSM, Eq. (32): v_Hxc = v_s(n) - v(n)."""
    return vs_of_n(n, T) - v_of_n(n, T, U)


def n_of_v(v, T, U):
    """Grand-canonical density of the interacting SSM (inverse of v_of_n):
    n = 2 (e^{-v/T} + e^{-(2v+U)/T}) / (1 + 2 e^{-v/T} + e^{-(2v+U)/T})."""
    v = np.asarray(v, dtype=float)
    a1, a2 = -v / T, -(2.0 * v + U) / T
    m = np.maximum(0.0, np.maximum(a1, a2))  # factor out the largest exponent
    e0, e1, e2 = np.exp(-m), np.exp(a1 - m), np.exp(a2 - m)
    return 2.0 * (e1 + e2) / (e0 + 2.0 * e1 + e2)
