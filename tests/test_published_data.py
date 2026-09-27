"""Regression test: recompute the data of Figs. 1-5 and compare with the data
files that were used to make the published figures (published_data/)."""
from pathlib import Path

import numpy as np
import pytest

from iqdft import paper

DATA = Path(__file__).resolve().parents[1] / "published_data"


def check(new, fname):
    old = np.loadtxt(DATA / fname)[:, :new.shape[1]]
    assert new.shape == old.shape
    np.testing.assert_allclose(new, old, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("T", paper.FIG1_T_VALUES)
def test_fig1_temperatures(T):
    check(paper.xc_table(T, paper.FIG1_U_FIXED), f"fig1_xc_derivatives/LR_T={T}.dat")


@pytest.mark.parametrize("U", paper.FIG1_U_VALUES)
def test_fig1_interactions(U):
    check(paper.xc_table(paper.FIG1_T_FIXED, U), f"fig1_xc_derivatives/LR_U={U}.dat")


@pytest.mark.parametrize("T", paper.FIG3_T_VALUES)
def test_fig2_fig3_gate(T):
    check(paper.gate_table(T), f"fig3_gate/LR_T={T}.dat")


@pytest.mark.parametrize("vg", paper.FIG4_VG_VALUES)
def test_fig4_temperature(vg):
    check(paper.temperature_table(vg), f"fig4_temperature/LR_vg={vg}.dat")


@pytest.mark.parametrize("U", paper.FIG5_U_VALUES)
def test_fig2_fig5_interaction(U):
    check(paper.interaction_table(U), f"fig5_interaction/LR_U={U}.dat")
