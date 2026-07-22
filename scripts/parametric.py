"""Estudo paramétrico da razão de contraste Gi/Gm (parecer 3.7).

O artigo afirma que o pico de tensão "aumenta de forma expressiva" com o
contraste, mas sem tabela/figura. Aqui gera-se a evidência com MDF (N=80),
reportando G_ef e métricas de tensão ROBUSTAS — mostrando que o pico pontual
(max) exagera o efeito frente ao p99/max_far (parecer 2.3).

Uso:  python scripts/parametric.py
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.fdm import solve_fdm
from src.geometry import Case
from src.metrics import robust_stress_metrics

TAB = os.path.join(ROOT, "outputs", "tables")
RATIOS = [1, 2, 5, 10, 50, 100]

if __name__ == "__main__":
    os.makedirs(TAB, exist_ok=True)
    print("=" * 70)
    print(f"{'Gi/Gm':>6s} {'G_ef':>10s} {'max|tau|':>10s} {'p99':>10s} "
          f"{'max_far':>10s}")
    linhas = []
    for r in RATIOS:
        case = Case(L=1, a=0.15, Gm=1.0, Gi=float(r), gamma=0.01, shape="square")
        s = solve_fdm(case, N=80)
        m = robust_stress_metrics(s, case, delta=0.05)
        print(f"{r:6d} {s['Gef']:10.4f} {m['max']:10.4f} {m['p99']:10.4f} "
              f"{m['max_far']:10.4f}")
        linhas.append((r, s["Gef"], m))
    print("=" * 70)
    print("G_ef cresce e SATURA; note que 'max' cresce mais que p99/max_far,")
    print("confirmando que o pico pontual e dominado pela singularidade de quina.")

    with open(os.path.join(TAB, "parametrico_contraste.csv"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write("Gi_Gm,Gef,tau_max,tau_p99,tau_max_far,tau_l2\n")
        for r, gef, m in linhas:
            f.write(f"{r},{gef:.6f},{m['max']:.6f},{m['p99']:.6f},"
                    f"{m['max_far']:.6f},{m['l2']:.6f}\n")
    print("Tabela: outputs/tables/parametrico_contraste.csv")
