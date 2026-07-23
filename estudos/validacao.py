"""Validação do solver MDF contra a solução analítica de inclusão CIRCULAR.

Correção do parecer 2.1: a solução analítica clássica é de MEIO INFINITO e NÃO
satisfaz as condições de contorno do quadrado finito (w=0 em x=0, w=gamma*L em
x=L). Comparar contra ela impondo BCs mistas mede a diferença ENTRE DOIS
PROBLEMAS, e o erro estagna (~1,5e-2) por mais que se refine a malha.

Aqui resolve-se o MESMO BVP da referência: impõe-se Dirichlet EXATO da analítica
em TODO o contorno (``bc_func=w_analytic``). Assim a validação passa a ser
consistente e o erro converge com o refino.

Uso:
    python estudos/validacao.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ferramentas import console as ui
from problema.analitico import w_analytic
from metodos.mdf import solve_fdm
from problema.geometria import VALIDATION
from ferramentas.metricas import rel_l2

MASK = lambda X, Y: (X > 0.2) & (X < 0.8) & (Y > 0.2) & (Y < 0.8)
NS = [20, 40, 80, 160]


def convergencia(bc_func):
    """Retorna lista de (N, erro_rel_L2, ordem_estimada)."""
    linhas, prev = [], None
    for N in NS:
        s = solve_fdm(VALIDATION, N=N, bc_func=bc_func)
        X, Y = s["X"], s["Y"]
        e = rel_l2(s["W"], w_analytic(X, Y, VALIDATION), MASK(X, Y))
        ordem = None if prev is None else np.log(prev / e) / np.log(2)
        linhas.append((N, e, ordem))
        prev = e
    return linhas


def imprime(titulo, linhas):
    print()
    print("  " + titulo)
    for N, e, ordem in linhas:
        o = ui.paint("  ---", "dim") if ordem is None else f"ordem~{ordem:+.2f}"
        print(f"    N={N:<3d}  e_rel(L2)={ui.value(f'{e:.4e}')}   {o}")


def salva_csv(caminho, mista, dirichlet):
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    with open(caminho, "w", encoding="utf-8", newline="\n") as f:
        f.write("N,e_rel_bc_mista,e_rel_dirichlet_exato\n")
        for (N, em, _), (_, ed, _) in zip(mista, dirichlet):
            f.write(f"{N},{em:.6e},{ed:.6e}\n")
    ui.ok(f"Tabela: {os.path.relpath(caminho)}")


if __name__ == "__main__":
    ui.header("Validacao do MDF vs solucao analitica (inclusao circular)")
    print("  solver: " + ui.method("MDF"))
    ui.step("Rodando convergencia: BC mista vs Dirichlet exato...")
    mista = convergencia(bc_func=None)         # BCs oficiais (meio infinito != finito)
    dirichlet = convergencia(bc_func=w_analytic)  # mesmo BVP da referência

    imprime(ui.paint("ANTES ", "yellow", "bold")
            + " BC mista (w=0, w=gamma*L, Neumann)  ->  erro ESTAGNA", mista)
    imprime(ui.paint("DEPOIS", "green", "bold")
            + " Dirichlet EXATO no contorno  ->  CONVERGE", dirichlet)

    print()
    ui.dim("  Obs.: a ordem fica ~1,5 (nao 2) porque a interface circular e")
    ui.dim("  representada 'em escada' na malha cartesiana (parecer 3.4).")

    salva_csv(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "resultados", "validacao_convergencia.csv"),
        mista, dirichlet,
    )
