"""Estudos de convergência (parecer 2.1, 3.4, 3.7).

(a) CIRCULAR: MDF com Dirichlet exato da analítica -> erro vs solução exata.
(b) QUADRADA (caso oficial): auto-convergência (Richardson) de MDF e MEF contra
    uma malha fina de referência do próprio método, já que não há solução fechada.
    As malhas grossas são subconjuntos exatos da malha fina (Nf múltiplo de Nc),
    então comparam-se nós coincidentes sem interpolação.

Uso:  python scripts/convergence.py
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import console as ui
from src.analytic import w_analytic
from src.fdm import solve_fdm
from src.fem import solve_fem
from src.geometry import OFFICIAL, VALIDATION
from src.metrics import rel_l2

TAB = os.path.join(ROOT, "outputs", "tables")


def _ordem(prev, e):
    return None if prev is None else float(np.log(prev / e) / np.log(2))


def circular_mdf():
    """MDF (Dirichlet exato) vs analítica circular, erro L2 no interior."""
    rows, prev = [], None
    for N in [20, 40, 80, 160]:
        s = solve_fdm(VALIDATION, N=N, bc_func=w_analytic)
        X, Y = s["X"], s["Y"]
        m = (X > 0.2) & (X < 0.8) & (Y > 0.2) & (Y < 0.8)
        e = rel_l2(s["W"], w_analytic(X, Y, VALIDATION), m)
        rows.append((N, e, _ordem(prev, e)))
        prev = e
    return rows


def square_self(solver, ref_N, tests, field="W"):
    """Auto-convergência no caso quadrado: erro vs malha fina do mesmo método."""
    sref = solver(OFFICIAL, N=ref_N)
    Wref = sref[field]
    rows, prev = [], None
    for N in tests:
        m = ref_N // N
        s = solver(OFFICIAL, N=N)
        e = rel_l2(s[field], Wref[::m, ::m])
        rows.append((N, e, _ordem(prev, e)))
        prev = e
    return rows


def _tabela(nome, rows, header="N,e_rel,ordem"):
    os.makedirs(TAB, exist_ok=True)
    caminho = os.path.join(TAB, nome)
    with open(caminho, "w", encoding="utf-8", newline="\n") as f:
        f.write(header + "\n")
        for N, e, o in rows:
            f.write(f"{N},{e:.6e},{'' if o is None else f'{o:.3f}'}\n")
    return caminho


def _print(titulo, rows, meth=None):
    label = f"{ui.method(meth)}  " if meth else ""
    print()
    print("  " + label + ui.paint(titulo, "bold"))
    for N, e, o in rows:
        oo = ui.paint("  ---", "dim") if o is None else f"ordem~{o:+.2f}"
        print(f"    N={N:<3d}  e_rel={ui.value(f'{e:.4e}')}   {oo}")


if __name__ == "__main__":
    ui.header("Estudos de convergencia")
    ui.legend()
    ui.step("Rodando MDF (circular + quadrada, ref N=240) e MEF (ref N=160)...")

    r_circ = circular_mdf()
    _print("Circular - MDF Dirichlet-exato vs analitica (interior)", r_circ, "MDF")
    _tabela("conv_circular_mdf.csv", r_circ)

    r_sq_mdf_w = square_self(solve_fdm, 240, [20, 40, 80], "W")
    _print("Quadrada - auto-convergencia em w  (ref N=240)", r_sq_mdf_w, "MDF")
    _tabela("conv_quadrada_mdf_w.csv", r_sq_mdf_w)

    r_sq_mdf_t = square_self(solve_fdm, 240, [20, 40, 80], "txz")
    _print("Quadrada - auto-convergencia em tau_xz  (ref N=240)", r_sq_mdf_t, "MDF")
    _tabela("conv_quadrada_mdf_txz.csv", r_sq_mdf_t)

    r_sq_fem_w = square_self(solve_fem, 160, [20, 40, 80], "W")
    _print("Quadrada - auto-convergencia em w  (ref N=160)", r_sq_fem_w, "MEF")
    _tabela("conv_quadrada_mef_w.csv", r_sq_fem_w)

    print()
    ui.dim("  Leitura: circular ~ordem 1.5 (interface em escada). Em w: MDF ~0.9,")
    ui.dim("  MEF ~1.7. Em tau_xz: ~0.5 (concentracao nas quinas -> parecer 2.3).")
    ui.ok("Tabelas salvas em outputs/tables/")
