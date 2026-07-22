"""Gera as figuras do artigo em outputs/figures/ (matplotlib Agg, sem GPU).

Reexecutável: gera já as figuras de MDF/MEF, convergência e paramétrico; se
existir ``outputs/pinn_sol_seed*.npz`` (produzido por scripts/pinn_seeds.py),
acrescenta o painel dos três métodos e as figuras da PINN.

Uso:  python scripts/make_figures.py
"""

import csv
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import viz
from src.fdm import solve_fdm
from src.fem import solve_fem
from src.geometry import OFFICIAL

FIG = os.path.join(ROOT, "outputs", "figures")
TAB = os.path.join(ROOT, "outputs", "tables")
os.makedirs(FIG, exist_ok=True)


def _csv(name):
    with open(os.path.join(TAB, name), encoding="utf-8") as f:
        linhas = list(csv.reader(f))
    return np.array([[float(x) if x else np.nan for x in row] for row in linhas[1:]])


def main():
    sol_fdm = solve_fdm(OFFICIAL, N=80)
    sol_fem = solve_fem(OFFICIAL, N=80)
    sols = {"MDF": sol_fdm, "MEF": sol_fem}

    npz = sorted(glob.glob(os.path.join(ROOT, "outputs", "pinn_sol_seed*.npz")))
    if npz:
        d = np.load(npz[0])
        sols["PINN"] = dict(X=d["X"], Y=d["Y"], W=d["W"], txz=d["txz"],
                            tyz=d["tyz"], case=OFFICIAL)
        print(f"PINN carregada de {os.path.basename(npz[0])}")

    # --- Campos e perfis ---
    viz.fig_panel_contourf(sols, OFFICIAL, field="W", save="fig_painel_w_curvas.png")
    viz.fig_field_contourf(sol_fdm, "txz", OFFICIAL, save="fig_txz_mdf.png",
                           title=r"$\tau_{xz}$ - MDF")
    viz.fig_centerline_methods(sols, OFFICIAL, field="W", save="fig_perfil_w.png")
    viz.fig_centerline_methods(sols, OFFICIAL, field="txz", save="fig_perfil_txz.png")

    # --- Convergência (com guias de ordem 1 e 2) ---
    c = _csv("conv_circular_mdf.csv")      # N, e_rel, ordem
    m = _csv("conv_quadrada_mdf_w.csv")
    f = _csv("conv_quadrada_mef_w.csv")
    fig, ax = plt.subplots(figsize=(6, 4.4))
    ax.loglog(c[:, 0], c[:, 1], "o-", label="circular MDF (Dirichlet exato)")
    ax.loglog(m[:, 0], m[:, 1], "s-", label="quadrada MDF (w)")
    ax.loglog(f[:, 0], f[:, 1], "^-", label="quadrada MEF (w)")
    xg = np.array([c[0, 0], c[-1, 0]])
    ax.loglog(xg, c[0, 1] * (c[0, 0] / xg) ** 1, "k--", lw=0.8, label="ordem 1")
    ax.loglog(xg, c[0, 1] * (c[0, 0] / xg) ** 2, "k:", lw=0.8, label="ordem 2")
    ax.set_xlabel("N"); ax.set_ylabel(r"erro relativo $L^2$ (w)")
    ax.set_title("Convergência"); ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.savefig(os.path.join(FIG, "fig_convergencia.png"), dpi=160, bbox_inches="tight")

    # --- Paramétrico: G_ef satura; max|tau| exagera vs p99/max_far ---
    p = _csv("parametrico_contraste.csv")  # ratio,Gef,max,p99,far,l2
    fig, ax1 = plt.subplots(figsize=(6.6, 4.4))
    ax1.semilogx(p[:, 0], p[:, 1], "o-", color="#1f77b4", label=r"$G_{ef}$")
    ax1.set_xlabel(r"$G_i/G_m$"); ax1.set_ylabel(r"$G_{ef}$", color="#1f77b4")
    ax2 = ax1.twinx()
    ax2.loglog(p[:, 0], p[:, 2], "s--", color="#d62728", label=r"$\max|\tau|$")
    ax2.loglog(p[:, 0], p[:, 3], "^--", color="#2ca02c", label=r"$p_{99}|\tau|$")
    ax2.loglog(p[:, 0], p[:, 4], "v--", color="#ff7f0e", label=r"$\max_{far}|\tau|$")
    ax2.set_ylabel(r"$|\tau|$")
    l1, la1 = ax1.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, la1 + la2, fontsize=7, loc="upper left")
    ax1.set_title("Contraste $G_i/G_m$")
    fig.savefig(os.path.join(FIG, "fig_parametrico.png"), dpi=160, bbox_inches="tight")

    # --- Figuras só da PINN ---
    if "PINN" in sols:
        viz.fig_field_contourf(sols["PINN"], "W", OFFICIAL, save="fig_pinn_w.png",
                               title="w(x,y) - PINN")
        viz.fig_scatter_r2(sols["PINN"]["W"], sol_fdm["W"],
                           save="fig_pinn_scatter.png", label="PINN")

    plt.close("all")
    metodos = "+".join(sols.keys())
    print(f"Figuras geradas em outputs/figures/ (metodos: {metodos})")


if __name__ == "__main__":
    main()
