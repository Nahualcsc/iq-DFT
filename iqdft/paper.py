"""Parameter sets and data tables of Figs. 1-5 of the paper.

Every table has the same column layout as the original data files used for
the published figures (``published_data/``), so that the two can be compared
column by column.  All energies in units of gamma (gamma = 1).
G and kappa columns are in units of G0 = 1/pi.
"""
import numpy as np

from . import dft, xc

GAMMA = 1.0

# iq-DFT scheme of the published figures (see iqdft.dft).  "exact-ks" gives
# identical densities and iq-DFT coefficients; only the KS (LB-DFT) columns differ.
PAPER_SCHEME = "paper"

FIG1_N = np.linspace(0.01, 1.99, 400)
FIG1_T_VALUES = (2.0, 5.0, 10.0, 20.0)   # left column, U = 8
FIG1_U_FIXED = 8.0
FIG1_U_VALUES = (1.0, 4.0, 7.0, 12.0)    # right column, T = 12
FIG1_T_FIXED = 12.0

FIG3_VG = np.linspace(-20.0, 20.0, 400)
FIG3_T_VALUES = (1.0, 3.0, 5.0, 10.0)    # Fig. 2 (upper) and Fig. 3, U = 8
FIG3_U = 8.0

FIG4_T = np.logspace(0.0, 2.0, 400)
FIG4_VG_VALUES = (0.5, 2.0, 4.0, 6.0)    # U = 8
FIG4_U = 8.0

FIG5_VG = np.linspace(-15.0, 15.0, 400)
FIG5_U_VALUES = (1.0, 4.0, 7.0, 12.0)    # Fig. 2 (lower) and Fig. 5, T = 1
FIG5_T = 1.0

XC_COLUMNS = ["n", "dVxc_dI", "dVxc_dQ", "dPsixc_dQ",
              "dVxc_dI_exact", "dVxc_dQ_exact", "dPsixc_dQ_exact"]

LR_COLUMNS = ["x", "n_MBM", "G_MBM", "S_MBM", "kappa_MBM",
              "dVxc_dI", "dVxc_dQ", "dPsixc_dQ", "G_KS", "S_KS", "kappa_KS",
              "v_Hxc", "n_iqDFT", "G_iqDFT", "S_iqDFT", "kappa_iqDFT",
              "ZT_MBM", "ZT_iqDFT"]


def xc_table(T, U, gamma=GAMMA, n=FIG1_N):
    """Fig. 1: analytic (SSM) and exact (numerical) xc derivatives vs n."""
    a = xc.unpack(xc.xc_derivatives_ssm(n, T, gamma, U))
    e = xc.unpack(xc.xc_derivatives_exact(n, T, gamma, U))
    return np.column_stack([n, *a, *e])


def lr_table(x, v, T, U, gamma=GAMMA, scheme=PAPER_SCHEME):
    """Figs. 2-5: MBM, KS (LB-DFT) and iq-DFT linear-response results.

    ``x`` is the abscissa written in column 1 (gate voltage or temperature);
    ``v`` the on-site energy(ies) and ``T`` the temperature(s)."""
    x, v, T = np.broadcast_arrays(np.asarray(x, float), np.asarray(v, float),
                                  np.asarray(T, float))
    rows = []
    for xi, vi, Ti in zip(x, v, T):
        m = dft.mbm_linear_response(vi, Ti, gamma, U)
        d = dft.linear_response(vi, Ti, gamma, U, scheme=scheme)
        rows.append([xi, m["n"], np.pi * m["G"], m["S"], np.pi * m["kappa"],
                     d["dVxc_dI"], d["dVxc_dQ"], d["dPsixc_dQ"],
                     np.pi * d["Gs"], d["Ss"], np.pi * d["kappas"],
                     d["v_hxc"], d["n"],
                     np.pi * d["G"], d["S"], np.pi * d["kappa"],
                     m["ZT"], d["ZT"]])
    return np.array(rows, dtype=float)


def gate_table(T, U=FIG3_U, vg=FIG3_VG, **kw):
    """Figs. 2 (upper) and 3: gate dependence (v_g = v + U/2)."""
    return lr_table(vg, vg - 0.5 * U, T, U, **kw)


def temperature_table(vg, U=FIG4_U, T=FIG4_T, **kw):
    """Fig. 4: temperature dependence at fixed gate."""
    return lr_table(T, vg - 0.5 * U, T, U, **kw)


def interaction_table(U, T=FIG5_T, vg=FIG5_VG, **kw):
    """Figs. 2 (lower) and 5: gate dependence for different U."""
    return lr_table(vg, vg - 0.5 * U, T, U, **kw)

