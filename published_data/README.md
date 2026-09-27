# Data of the published figures

Original data files (2021) from which the figures of the paper were drawn with
xmgrace. The `.agr` files are the xmgrace projects (the data are embedded in
them; the `comment "Cols 1:k"` lines record which column was plotted). The
final figures as submitted are in `published_eps/`.

All energies are in units of gamma (gamma = 1). G and kappa are in units of
G0 = 1/pi (atomic units), S in atomic units (k_B/e).

## `fig1_xc_derivatives/LR_T=*.dat` (U = 8), `LR_U=*.dat` (T = 12)

| col | quantity |
|---|---|
| 1 | n |
| 2-4 | dV_xc/dI, dV_xc/dQ (= dPsi_xc/dI), dPsi_xc/dQ: analytic SSM parametrization, Eqs. (30)+(31) |
| 5-7 | same, from the exact numerical inversion of Eqs. (25a)/(27a) |

## `fig3_gate/LR_T=*.dat`, `fig4_temperature/LR_vg=*.dat`, `fig5_interaction/LR_U=*.dat`

Fig. 2 (densities) uses `fig3_gate` (upper panel) and `fig5_interaction`
(lower panel); the original `densities/` directory contained identical copies.

| col | quantity |
|---|---|
| 1 | gate voltage v_g = v + U/2 (Figs. 2, 3, 5) or temperature T (Fig. 4) |
| 2-5 | many-body model (MBM): n, G, S, kappa |
| 6-8 | xc derivatives dV_xc/dI, dV_xc/dQ, dPsi_xc/dQ at the iq-DFT density |
| 9-11 | KS (= LB-DFT) G_s, S_s, kappa_s |
| 12 | v_Hxc |
| 13 | iq-DFT density |
| 14-16 | iq-DFT G, S, kappa |
| 17-18 | ZT of MBM and iq-DFT |
| 19 (fig3) | dV_xc/dI + T (S^2/kappa - S_s^2/kappa_s), the i-DFT xc derivative of Eq. (23) (not plotted) |
| 19-20 (fig4) | SSM gates v(n), v_s(n) of Eq. (31) at the iq-DFT density (not plotted) |

Parameters: Fig. 3: U = 8, T = 1, 3, 5, 10 (only T = 1, 3, 5 shown in the
paper); Fig. 4: U = 8, v_g = 0.5, 2, 4, 6, T = 10^(0...2); Fig. 5: T = 1,
U = 1, 4, 7, 12.
