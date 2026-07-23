"""Comparação no caso oficial (N=80): G_ef, erros L2 e tensões robustas.

Endereça:
  - parecer 2.2: reportar G_ef com dígitos + sanidade Voigt-Reuss.
  - parecer 2.5: MDF é a REFERÊNCIA; proximidade MEF<->MDF NÃO prova acurácia
    absoluta. Reporta-se o erro relativo tomando MDF como referência, com essa
    ressalva explícita.
  - parecer 2.3: max|tau| não é robusto; reporta-se também p99 e max fora das
    quinas (max_far).

Cores: MDF azul, MEF verde (ver src/console.py).

Uso:  python estudos/comparacao.py
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ferramentas import console as ui
from metodos.mdf import solve_fdm
from metodos.mef import solve_fem
from problema.geometria import OFFICIAL
from ferramentas.metricas import field_errors, robust_stress_metrics

TAB = os.path.join(ROOT, "resultados")


def voigt_reuss(case):
    f = (2 * case.a) ** 2
    gv = f * case.Gi + (1 - f) * case.Gm
    gr = 1.0 / (f / case.Gi + (1 - f) / case.Gm)
    return gr, gv, f


if __name__ == "__main__":
    os.makedirs(TAB, exist_ok=True)
    ui.header("Comparacao no caso oficial  (L=1, a=0.15, Gm=1, Gi=5, N=80)")
    ui.legend()

    ui.step("Resolvendo MDF e MEF...")
    sol_fdm = solve_fdm(OFFICIAL, N=80)
    sol_fem = solve_fem(OFFICIAL, N=80)

    gr, gv, f = voigt_reuss(OFFICIAL)
    ui.section("Modulo efetivo  G_ef = <tau_xz>/gamma")
    ui.method_line("MDF", ui.value(f"{sol_fdm['Gef']:.6f}"))
    ui.method_line("MEF", ui.value(f"{sol_fem['Gef']:.6f}"))
    ui.kv("Voigt / Reuss", f"[{gr:.4f} ; {gv:.4f}]   (fracao de area f = {f:.3f})")
    if gr <= sol_fdm["Gef"] <= gv:
        ui.ok("MDF e MEF dentro dos limites de Voigt-Reuss")
    else:
        ui.warn("fora dos limites de Voigt-Reuss")

    err = field_errors(sol_fem, sol_fdm)
    ui.section("Erro relativo L2 do MEF  (referencia = MDF)")
    ui.kv("w", ui.value(f"{err['w']:.3e}"))
    ui.kv("tau_xz", ui.value(f"{err['txz']:.3e}"))
    ui.kv("tau_yz", ui.value(f"{err['tyz']:.3e}"))
    ui.dim("  proximidade ao MDF nao prova acuracia absoluta; as tensoes divergem 10-30%")

    ui.section("Metricas robustas de |tau|  (parecer 2.3)")
    print("        " + ui.paint(f"{'max':>10s}{'p99':>11s}{'max_far':>11s}{'L2':>11s}",
                                 "gray"))
    rows = []
    for name, sol in (("MDF", sol_fdm), ("MEF", sol_fem)):
        m = robust_stress_metrics(sol, OFFICIAL, delta=0.05)
        ui.method_line(name, f"{m['max']:>10.4f}{m['p99']:>11.4f}"
                             f"{m['max_far']:>11.4f}{m['l2']:>11.4f}")
        rows.append((name, m))

    with open(os.path.join(TAB, "comparacao_oficial.csv"), "w",
              encoding="utf-8", newline="\n") as fcsv:
        fcsv.write("grandeza,MDF,MEF\n")
        fcsv.write(f"Gef,{sol_fdm['Gef']:.6f},{sol_fem['Gef']:.6f}\n")
        for k in ("max", "p99", "max_far", "l2"):
            fcsv.write(f"tau_{k},{rows[0][1][k]:.6f},{rows[1][1][k]:.6f}\n")
    with open(os.path.join(TAB, "erros_mef_vs_mdf.csv"), "w",
              encoding="utf-8", newline="\n") as fcsv:
        fcsv.write("campo,erro_rel_L2\n")
        for k in ("w", "txz", "tyz"):
            fcsv.write(f"{k},{err[k]:.6e}\n")

    print()
    ui.ok("Tabelas: resultados/comparacao_oficial.csv, erros_mef_vs_mdf.csv")
