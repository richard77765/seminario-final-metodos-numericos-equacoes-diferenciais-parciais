"""Geometria da celula representativa: dominio, inclusao e campo de modulo G(x,y).

Problema 3 (Equipe 3) - cisalhamento antiplano em celula compósita com inclusao
quadrada central. Tudo aqui e puramente geometrico/material; os solvers consomem
estas funcoes para montar seus sistemas.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Case:
    """Parametros de um caso do problema.

    L  : lado da celula quadrada [0,L]x[0,L]
    a  : "semi-lado" da inclusao quadrada central (lado = 2a); para o caso circular
         de validacao, a e o raio da inclusao circular.
    Gm : modulo de cisalhamento da matriz
    Gi : modulo de cisalhamento da inclusao
    gamma : deformacao de cisalhamento macroscopica imposta (w(L,y) = gamma*L)
    shape : "square" (caso oficial) ou "circle" (validacao analitica)
    """

    L: float = 1.0
    a: float = 0.15
    Gm: float = 1.0
    Gi: float = 5.0
    gamma: float = 0.01
    shape: str = "square"

    @property
    def center(self):
        """Centro do dominio (e da inclusao)."""
        return (self.L / 2.0, self.L / 2.0)


# Caso oficial (secao 1 do CLAUDE.md)
OFFICIAL = Case(L=1.0, a=0.15, Gm=1.0, Gi=5.0, gamma=0.01, shape="square")
# Caso de validacao analitica com inclusao circular (secao 2)
VALIDATION = Case(L=1.0, a=0.15, Gm=1.0, Gi=10.0, gamma=0.01, shape="circle")


def inclusion_mask(x, y, case: Case):
    """Mascara booleana: True onde (x,y) esta DENTRO da inclusao.

    Aceita arrays (broadcast) ou escalares. Para a inclusao quadrada usa a
    norma do infinito (quadrado alinhado aos eixos); para a circular, a norma 2.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cx, cy = case.center
    dx = x - cx
    dy = y - cy
    if case.shape == "square":
        return (np.abs(dx) <= case.a) & (np.abs(dy) <= case.a)
    elif case.shape == "circle":
        return (dx * dx + dy * dy) <= case.a * case.a
    raise ValueError(f"shape desconhecido: {case.shape!r}")


def G_field(x, y, case: Case):
    """Campo material G(x,y): Gi dentro da inclusao, Gm fora. Vetorizado."""
    inside = inclusion_mask(x, y, case)
    return np.where(inside, case.Gi, case.Gm)


def grid(N, case: Case):
    """Malha regular (N+1)x(N+1) sobre [0,L]^2.

    Retorna (X, Y, x1d, h) com X,Y no formato 'ij' (X[i,j], Y[i,j]) de modo que
    o indice i anda em x e j anda em y -- convencao usada pelo solver MDF.
    """
    x1d = np.linspace(0.0, case.L, N + 1)
    X, Y = np.meshgrid(x1d, x1d, indexing="ij")
    h = case.L / N
    return X, Y, x1d, h


def inclusion_outline(case: Case, n=200):
    """Pontos (xs, ys) do contorno da inclusao, para sobrepor nas figuras."""
    cx, cy = case.center
    if case.shape == "square":
        a = case.a
        xs = [cx - a, cx + a, cx + a, cx - a, cx - a]
        ys = [cy - a, cy - a, cy + a, cy + a, cy - a]
        return np.array(xs), np.array(ys)
    else:
        t = np.linspace(0, 2 * np.pi, n)
        return cx + case.a * np.cos(t), cy + case.a * np.sin(t)
