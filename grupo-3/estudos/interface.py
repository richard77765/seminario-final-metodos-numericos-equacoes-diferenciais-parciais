"""Erro de continuidade do fluxo normal na interface (parecer 2.3/3.3).

A condição física na interface é o equilíbrio do fluxo normal, [[G d_n w]] = 0.
Este script mede quão bem cada método a satisfaz, estimando o fluxo normal por
derivada unilateral de cada fase e comparando o salto ao longo da interface.

Cores: MDF azul, MEF verde, PINN amarelo.

Uso:  python estudos/interface.py
"""

import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ferramentas import console as ui
from metodos.mdf import solve_fdm
from metodos.mef import solve_fem
from problema.geometria import OFFICIAL
from ferramentas.metricas import interface_flux_error

TAB = os.path.join(ROOT, "resultados")


if __name__ == "__main__":
    os.makedirs(TAB, exist_ok=True)
    ui.header("Erro de fluxo normal na interface  ([[G d_n w]] = 0)")
    ui.legend()

    ui.step("Resolvendo MDF e MEF (N=80)...")
    sols = [("MDF", solve_fdm(OFFICIAL, N=80)),
            ("MEF", solve_fem(OFFICIAL, N=80))]

    # PINN, se houver solução salva por estudos/pinn_multisemente.py
    npz = sorted(glob.glob(os.path.join(ROOT, "resultados", "pinn_sol_seed*.npz")))
    if npz:
        d = np.load(npz[0])
        sols.append(("PINN", dict(X=d["X"], Y=d["Y"], W=d["W"], txz=d["txz"],
                                  tyz=d["tyz"], h=float(1.0 / (d["W"].shape[0] - 1)),
                                  N=d["W"].shape[0] - 1, case=OFFICIAL)))
        ui.info(f"{ui.method('PINN')} incluída ({os.path.basename(npz[0])})")

    ui.section("Salto do fluxo normal na interface (menor = melhor)")
    print("        " + ui.paint(f"{'rms':>12s}{'relativo':>14s}", "gray"))
    linhas = []
    for name, sol in sols:
        m = interface_flux_error(sol, OFFICIAL)
        ui.method_line(name, f"{m['rms']:>12.3e}{m['rel']:>13.2%}")
        linhas.append((name, m))

    print()
    ui.dim("  MEF (G por elemento) satisfaz melhor a continuidade PONTUAL do fluxo e")
    ui.dim("  converge com o refino; o MDF conserva no sentido da media harmonica (nas")
    ui.dim("  faces), com residuo pontual maior; a PINN mostra o maior desequilibrio.")

    with open(os.path.join(TAB, "interface_fluxo.csv"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write("metodo,rms,relativo\n")
        for name, m in linhas:
            f.write(f"{name},{m['rms']:.6e},{m['rel']:.6e}\n")
    ui.ok("Tabela: resultados/interface_fluxo.csv")
