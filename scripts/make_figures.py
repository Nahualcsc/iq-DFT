"""Recompute the data of Figs. 1-5 of the paper, write them to text files and
plot them.

    python scripts/make_figures.py                       # as in the paper
    python scripts/make_figures.py --scheme exact-ks     # finite-coupling KS-SIAM
    python scripts/make_figures.py --compare-schemes     # extra comparison figure

See iqdft/dft.py for the schemes ("paper", "exact-ks", "ssm").

Output goes to results/ (data) and results/figures/ (png).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from iqdft import paper  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
C = {c: i for i, c in enumerate(paper.LR_COLUMNS)}
MARK = dict(ls="none", marker="D", ms=3.5, markevery=8)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6,
                     "lines.linewidth": 1.6, "legend.frameon": False, "savefig.dpi": 200})


def save(table, path, columns, comment=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (comment + "\n" if comment else "") + " ".join(columns)
    np.savetxt(path, table, header=header, fmt="%.16e")


def fig1(out):
    fig, ax = plt.subplots(3, 2, figsize=(7, 7), sharex=True, sharey="row")
    scale = [1e-3, 1e-2, 1.0]
    ylab = [r"$10^{-3}\,\partial V_{xc}/\partial I$", r"$10^{-2}\,\partial V_{xc}/\partial Q$",
            r"$\partial \Psi_{xc}/\partial Q$"]
    ylims = [(-0.2, 3.6), (-1.5, 1.5), (-0.6, 14)]
    sets = [[(f"T/γ={T:g}", paper.xc_table(T, paper.FIG1_U_FIXED), f"LR_T={T}.dat")
             for T in paper.FIG1_T_VALUES],
            [(f"U/γ={U:g}", paper.xc_table(paper.FIG1_T_FIXED, U), f"LR_U={U}.dat")
             for U in paper.FIG1_U_VALUES]]
    for col, series in enumerate(sets):
        for k, (label, tab, fname) in enumerate(series):
            save(tab, out / "fig1_xc_derivatives" / fname, paper.XC_COLUMNS)
            for row in range(3):
                a = ax[row, col]
                a.plot(tab[:, 0], scale[row] * tab[:, 4 + row], color=COLORS[k],
                       label=label if row == 0 else None)
                a.plot(tab[:, 0], scale[row] * tab[:, 1 + row], color=COLORS[k], **MARK)
    for row in range(3):
        ax[row, 0].set_ylabel(ylab[row])
        ax[row, 0].set_ylim(*ylims[row])
    for col in range(2):
        ax[-1, col].set_xlabel("n")
        ax[0, col].legend(loc="upper center", fontsize=8)
    ax[0, 0].set_title(f"U/γ = {paper.FIG1_U_FIXED:g}")
    ax[0, 1].set_title(f"T/γ = {paper.FIG1_T_FIXED:g}")
    ax[1, 0].plot([], [], "k-", label="exact (numerical) RE")
    ax[1, 0].plot([], [], "kD", ms=3.5, label="analytic (SSM) RE")
    ax[1, 0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figures" / "fig1_xc_derivatives.png")
    plt.close(fig)


def lr_panels(ax, tab, k, label, ks=False):
    """Plot G, S, kappa (and optionally ZT) columns of an lr_table."""
    x = tab[:, 0]
    quantities = [("G", "G/G$_0$"), ("S", "S"), ("kappa", r"$\kappa$/G$_0$"), ("ZT", "ZT")]
    for a, (q, ylabel) in zip(ax, quantities):
        a.plot(x, tab[:, C[q + "_MBM"]], color=COLORS[k], label=label)
        a.plot(x, tab[:, C[q + "_iqDFT"]], color=COLORS[k], **MARK)
        if ks and q != "ZT":
            a.plot(x, tab[:, C[q + "_KS"]], color=COLORS[k], ls="--", lw=1.1)
        a.set_ylabel(ylabel)


def style_legend(a, ks=False, loc="upper left", ncol=2):
    handles, labels = a.get_legend_handles_labels()
    extra = [plt.Line2D([], [], color="k"), plt.Line2D([], [], color="k", **MARK)]
    names = ["MBM", "iq-DFT"]
    if ks:
        extra.append(plt.Line2D([], [], color="k", ls="--", lw=1.1))
        names.append("LB-DFT (KS)")
    a.legend(handles + extra, labels + names, fontsize=7, ncol=ncol, loc=loc)


def fig2_3(out, kw, suffix):
    gate = {T: paper.gate_table(T, **kw) for T in paper.FIG3_T_VALUES}
    inter = {U: paper.interaction_table(U, **kw) for U in paper.FIG5_U_VALUES}
    for T, tab in gate.items():
        save(tab, out / f"fig3_gate{suffix}" / f"LR_T={T}.dat", paper.LR_COLUMNS,
             f"U/gamma={paper.FIG3_U:g}; x = v_g/gamma; G, kappa in units of G0")
    for U, tab in inter.items():
        save(tab, out / f"fig5_interaction{suffix}" / f"LR_U={U}.dat", paper.LR_COLUMNS,
             f"T/gamma={paper.FIG5_T:g}; x = v_g/gamma; G, kappa in units of G0")

    # Fig. 2: densities
    fig, ax = plt.subplots(2, 1, figsize=(4.5, 5.5), sharex=True)
    for k, (T, tab) in enumerate(gate.items()):
        ax[0].plot(tab[:, 0], tab[:, C["n_MBM"]], color=COLORS[k], label=f"T/γ={T:g}")
        ax[0].plot(tab[:, 0], tab[:, C["n_iqDFT"]], color=COLORS[k], **MARK)
    for k, (U, tab) in enumerate(inter.items()):
        ax[1].plot(tab[:, 0], tab[:, C["n_MBM"]], color=COLORS[k], label=f"U/γ={U:g}")
        ax[1].plot(tab[:, 0], tab[:, C["n_iqDFT"]], color=COLORS[k], **MARK)
    for a in ax:
        a.set_xlim(-15, 15)
        a.set_ylabel("n")
        style_legend(a, loc="upper right")
    ax[1].set_xlabel(r"$v_g/\gamma$")
    fig.tight_layout()
    fig.savefig(out / "figures" / f"fig2_densities{suffix}.png")
    plt.close(fig)

    # Fig. 3: gate dependence, U = 8 (the paper shows T = 1, 3, 5)
    fig, ax = plt.subplots(3, 1, figsize=(4.5, 6.5), sharex=True)
    for k, T in enumerate(paper.FIG3_T_VALUES[:3]):
        lr_panels(ax, gate[T], k, f"T/γ={T:g}", ks=True)
    ax[-1].set_xlabel(r"$v_g/\gamma$")
    ax[-1].set_xlim(-15, 15)
    style_legend(ax[0], ks=True)
    fig.tight_layout()
    fig.savefig(out / "figures" / f"fig3_gate{suffix}.png")
    plt.close(fig)

    # Fig. 5: interaction dependence, T = 1
    fig, ax = plt.subplots(2, 2, figsize=(7, 5), sharex=True)
    for k, (U, tab) in enumerate(inter.items()):
        lr_panels(ax.T.ravel(), tab, k, f"U/γ={U:g}")
    for a in ax[-1]:
        a.set_xlabel(r"$v_g/\gamma$")
    # legend in the kappa panel, which has free space in its upper left corner
    style_legend(ax[0, 1], ncol=1)
    fig.tight_layout()
    fig.savefig(out / "figures" / f"fig5_interaction{suffix}.png")
    plt.close(fig)


def fig4(out, kw, suffix):
    fig, ax = plt.subplots(2, 2, figsize=(7, 5), sharex=True)
    for k, vg in enumerate(paper.FIG4_VG_VALUES):
        tab = paper.temperature_table(vg, **kw)
        save(tab, out / f"fig4_temperature{suffix}" / f"LR_vg={vg}.dat", paper.LR_COLUMNS,
             f"U/gamma={paper.FIG4_U:g}; v_g/gamma={vg:g}; x = T/gamma; G, kappa in units of G0")
        lr_panels(ax.T.ravel(), tab, k, f"$v_g$/γ={vg:g}")
    for a in ax.ravel():
        a.set_xscale("log")
    ax[1, 1].set_yscale("log")
    for a in ax[-1]:
        a.set_xlabel(r"$T/\gamma$")
    style_legend(ax[0, 0], loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "figures" / f"fig4_temperature{suffix}.png")
    plt.close(fig)


def compare_schemes(out):
    """iq-DFT and LB-DFT for the three schemes of iqdft.dft at T = gamma, U = 8 gamma."""
    T, U = 1.0, paper.FIG3_U
    tab = {sch: paper.gate_table(T, scheme=sch) for sch in ("paper", "exact-ks", "ssm")}
    x = tab["paper"][:, 0]
    fig, ax = plt.subplots(4, 1, figsize=(5.2, 8.5), sharex=True)
    for axis, (q, ylabel) in zip(ax, [("n", "n"), ("G", "G/G$_0$"), ("S", "S"),
                                      ("kappa", r"$\kappa$/G$_0$")]):
        axis.plot(x, tab["paper"][:, C[q + "_MBM"]], color="#52514e", label="MBM")
        axis.plot(x, tab["exact-ks"][:, C[q + "_iqDFT"]], color=COLORS[0], **MARK,
                  label='iq-DFT "exact-ks" (= published)')
        axis.plot(x, tab["ssm"][:, C[q + "_iqDFT"]], color=COLORS[1], ls="none", marker="o",
                  ms=3.5, markevery=(4, 8), label='iq-DFT "ssm" (Eq. 32 with KS-SIAM)')
        if q != "n":
            axis.plot(x, tab["exact-ks"][:, C[q + "_KS"]], color=COLORS[0], ls="--", lw=1.1,
                      label='LB-DFT "exact-ks"')
            axis.plot(x, tab["paper"][:, C[q + "_KS"]], color=COLORS[2], ls=":", lw=1.4,
                      label="LB-DFT as published")
        axis.set_ylabel(ylabel)
    ax[0].legend(fontsize=7, loc="lower left")
    ax[1].legend(*[h[3:] for h in ax[1].get_legend_handles_labels()], fontsize=7, loc="upper left")
    ax[0].set_title(f"T/γ = {T:g}, U/γ = {U:g}", fontsize=9)
    ax[-1].set_xlabel(r"$v_g/\gamma$")
    ax[-1].set_xlim(-15, 15)
    fig.tight_layout()
    fig.savefig(out / "figures" / "scheme_comparison.png")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scheme", choices=("paper", "exact-ks", "ssm"), default="paper",
                   help="iq-DFT scheme for Figs. 2-5 (default: as in the paper)")
    p.add_argument("--compare-schemes", action="store_true",
                   help="also plot the comparison between the schemes")
    p.add_argument("--out", type=Path, default=ROOT / "results")
    args = p.parse_args()
    (args.out / "figures").mkdir(parents=True, exist_ok=True)
    kw = dict(scheme=args.scheme)
    suffix = "" if args.scheme == "paper" else "_" + args.scheme.replace("-", "_")
    fig1(args.out)
    fig2_3(args.out, kw, suffix)
    fig4(args.out, kw, suffix)
    if args.compare_schemes:
        compare_schemes(args.out)
    print(f"data written to {args.out}, figures to {args.out / 'figures'}")


if __name__ == "__main__":
    main()
