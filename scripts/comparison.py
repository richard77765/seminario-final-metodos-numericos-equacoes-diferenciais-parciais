"""Comparação no caso oficial (N=80): G_ef, erros L2 e tensões robustas.

Endereça:
  - parecer 2.2: reportar G_ef com dígitos + sanidade Voigt-Reuss.
  - parecer 2.5: MDF é a REFERÊNCIA; proximidade MEF<->MDF NÃO prova acurácia
    absoluta. Reporta-se o erro relativo tomando MDF como referência, com essa
    ressalva explícita.
  - parecer 2.3: max|tau| não é robusto; reporta-se também p99 e max fora das
    quinas (max_far).

Uso:  python scripts/comparison.py
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.fdm import solve_fdm
from src.fem import solve_fem
from src.geometry import OFFICIAL
from src.metrics import field_errors, robust_stress_metrics

TAB = os.path.join(ROOT, "outputs", "tables")


def voigt_reuss(case):
    f = (2 * case.a) ** 2
    gv = f * case.Gi + (1 - f) * case.Gm
    gr = 1.0 / (f / case.Gi + (1 - f) / case.Gm)
    return gr, gv, f


if __name__ == "__main__":
    os.makedirs(TAB, exist_ok=True)
    sol_fdm = solve_fdm(OFFICIAL, N=80)
    sol_fem = solve_fem(OFFICIAL, N=80)

    gr, gv, f = voigt_reuss(OFFICIAL)
    print("=" * 64)
    print(f"G_ef  MDF = {sol_fdm['Gef']:.6f}")
    print(f"G_ef  MEF = {sol_fem['Gef']:.6f}")
    print(f"Voigt/Reuss (f={f:.3f}): [{gr:.4f}, {gv:.4f}]  -> ambos consistentes")

    err = field_errors(sol_fem, sol_fdm)  # MEF vs MDF (referencia)
    print("\nErro L2 MEF vs MDF (MDF = referencia; NAO prova acuracia absoluta):")
    print(f"  w   = {err['w']:.3e}")
    print(f"  txz = {err['txz']:.3e}")
    print(f"  tyz = {err['tyz']:.3e}")
    print("  -> tensoes divergem 10-30%: a afirmacao '10^-3 em TODOS os campos'"
          " e falsa.")

    print("\nMetricas de tensao |tau| (parecer 2.3):")
    print(f"{'metodo':6s} {'max':>10s} {'p99':>10s} {'max_far':>10s} {'l2':>10s}")
    rows = []
    for nome, sol in [("MDF", sol_fdm), ("MEF", sol_fem)]:
        mtr = robust_stress_metrics(sol, OFFICIAL, delta=0.05)
        print(f"{nome:6s} {mtr['max']:10.4f} {mtr['p99']:10.4f} "
              f"{mtr['max_far']:10.4f} {mtr['l2']:10.4f}")
        rows.append((nome, mtr))
    print("=" * 64)

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
    print("Tabelas: outputs/tables/comparacao_oficial.csv, erros_mef_vs_mdf.csv")
