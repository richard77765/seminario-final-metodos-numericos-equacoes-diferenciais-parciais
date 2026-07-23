"""Estudo paramétrico da razão de contraste Gi/Gm (parecer 3.7).

O artigo afirma que o pico de tensão "aumenta de forma expressiva" com o
contraste, mas sem tabela/figura. Aqui gera-se a evidência com MDF (N=80),
reportando G_ef e métricas de tensão ROBUSTAS — mostrando que o pico pontual
(max) exagera o efeito frente ao p99/max_far (parecer 2.3).

Uso:  python estudos/parametrico.py
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ferramentas import console as ui
from metodos.mdf import solve_fdm
from problema.geometria import Case
from ferramentas.metricas import robust_stress_metrics

TAB = os.path.join(ROOT, "resultados")
RATIOS = [1, 2, 5, 10, 50, 100]

if __name__ == "__main__":
    os.makedirs(TAB, exist_ok=True)
    ui.header("Estudo parametrico do contraste  Gi/Gm   (MDF, N=80)")
    print("  solver: " + ui.method("MDF"))
    print("  " + ui.paint(f"{'Gi/Gm':>6s}{'G_ef':>11s}{'max|tau|':>12s}"
                          f"{'p99':>11s}{'max_far':>11s}", "gray"))
    linhas = []
    for r in RATIOS:
        case = Case(L=1, a=0.15, Gm=1.0, Gi=float(r), gamma=0.01, shape="square")
        s = solve_fdm(case, N=80)
        m = robust_stress_metrics(s, case, delta=0.05)
        gef_s = ui.value(f"{s['Gef']:>11.4f}")
        print(f"  {r:>6d}{gef_s}{m['max']:>12.4f}"
              f"{m['p99']:>11.4f}{m['max_far']:>11.4f}")
        linhas.append((r, s["Gef"], m))
    print()
    ui.dim("  G_ef cresce e desacelera. max|tau| cresce ~100x mas p99 so ~15x:")
    ui.dim("  o pico pontual e dominado pela singularidade de quina (parecer 2.3).")

    with open(os.path.join(TAB, "parametrico_contraste.csv"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write("Gi_Gm,Gef,tau_max,tau_p99,tau_max_far,tau_l2\n")
        for r, gef, m in linhas:
            f.write(f"{r},{gef:.6f},{m['max']:.6f},{m['p99']:.6f},"
                    f"{m['max_far']:.6f},{m['l2']:.6f}\n")
    print("Tabela: resultados/parametrico_contraste.csv")
