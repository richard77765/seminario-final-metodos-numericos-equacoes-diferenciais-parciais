"""Solver por Metodo de Diferencas Finitas (MDF) para -div(G grad w) = 0.

Discretizacao CONSERVATIVA (forma de fluxo) numa malha regular (N+1)x(N+1):

  -(1/h^2)[ G_{i+1/2,j}(w_{i+1,j}-w_{i,j}) - G_{i-1/2,j}(w_{i,j}-w_{i-1,j})
          + G_{i,j+1/2}(w_{i,j+1}-w_{i,j}) - G_{i,j-1/2}(w_{i,j}-w_{i,j-1}) ] = 0

As condutividades nas FACES usam MEDIA HARMONICA dos G nodais vizinhos -- esse e o
ponto fisico central: trata corretamente o salto de G na interface matriz/inclusao
(continuidade do fluxo normal). BCs:
  - Dirichlet nas laterais (x=0 e x=L): linha identidade no sistema.
  - Neumann homogeneo no topo/base (y=0 e y=L): nos-fantasma por reflexao.

Stack: numpy + scipy apenas.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from problema.geometria import Case, G_field, grid


def _harmonic(a, b):
    """Media harmonica elemento a elemento (faces). Evita divisao por zero."""
    s = a + b
    s = np.where(s == 0.0, 1.0, s)
    return 2.0 * a * b / s


def solve_fdm(case: Case, N=80, bc_func=None):
    """Resolve o problema por MDF.

    Se ``bc_func(x, y, case)`` for fornecida, impoe Dirichlet EXATO dessa funcao
    em TODO o contorno (usado na validacao, secao 7.1, para resolver o MESMO BVP
    da solucao de referencia). Caso contrario usa as BCs oficiais: Dirichlet nas
    laterais (x=0, x=L) e Neumann homogeneo no topo/base.

    Retorna dict com X, Y (malha 'ij'), W (deslocamento), txz, tyz, Gef, h, N, case.
    """
    X, Y, x1d, h = grid(N, case)
    n = N + 1
    Gnod = G_field(X, Y, case)  # G nos nos, shape (n, n) indexado [i, j]

    # Condutividades nas faces (media harmonica entre nos adjacentes).
    # Gx[i,j] = G na face entre (i,j) e (i+1,j), i = 0..N-1
    Gx = _harmonic(Gnod[:-1, :], Gnod[1:, :])      # shape (N, n)
    # Gy[i,j] = G na face entre (i,j) e (i,j+1), j = 0..N-1
    Gy = _harmonic(Gnod[:, :-1], Gnod[:, 1:])      # shape (n, N)

    def idx(i, j):
        return i * n + j

    ndof = n * n
    A = sp.lil_matrix((ndof, ndof))
    b = np.zeros(ndof)
    inv_h2 = 1.0 / (h * h)

    for i in range(n):
        for j in range(n):
            k = idx(i, j)

            on_boundary = (i == 0 or i == N or j == 0 or j == N)

            # --- Validacao (secao 7.1): Dirichlet EXATO da solucao de
            #     referencia em TODO o contorno do quadrado. Faz o problema
            #     numerico resolver o MESMO BVP que a analitica, em vez do
            #     problema de meio infinito -> validacao consistente e
            #     convergencia limpa (parecer 2.1).
            if bc_func is not None:
                if on_boundary:
                    A[k, k] = 1.0
                    b[k] = float(bc_func(X[i, j], Y[i, j], case))
                    continue
            else:
                # --- Caso oficial: Dirichlet nas laterais (x=0, x=L) +
                #     Neumann homogeneo no topo/base (nos-fantasma abaixo) ---
                if i == 0:
                    A[k, k] = 1.0
                    b[k] = 0.0                 # w(0, y) = 0
                    continue
                if i == N:
                    A[k, k] = 1.0
                    b[k] = case.gamma * case.L  # w(L, y) = gamma*L
                    continue

            # Coeficientes das faces. Para j nas bordas (topo/base), Neumann
            # homogeneo via no-fantasma: a face que sairia do dominio tem
            # contribuicao nula (fluxo nulo), o que equivale a remover o termo.
            cw = Gx[i - 1, j] * inv_h2     # face oeste  (i-1/2, j)
            ce = Gx[i, j] * inv_h2         # face leste  (i+1/2, j)
            cs = Gy[i, j - 1] * inv_h2 if j > 0 else 0.0   # face sul (i, j-1/2)
            cn = Gy[i, j] * inv_h2 if j < N else 0.0       # face norte (i, j+1/2)

            diag = cw + ce + cs + cn
            A[k, k] = diag
            A[k, idx(i - 1, j)] = -cw
            A[k, idx(i + 1, j)] = -ce
            if j > 0:
                A[k, idx(i, j - 1)] = -cs
            if j < N:
                A[k, idx(i, j + 1)] = -cn
            # b[k] = 0 (sem fonte)

    W = spsolve(A.tocsr(), b).reshape(n, n)

    txz, tyz = _stresses(W, Gnod, h)
    Gef = float(np.mean(txz) / case.gamma)

    return dict(X=X, Y=Y, W=W, txz=txz, tyz=tyz, Gef=Gef, h=h, N=N, case=case)


def _stresses(W, Gnod, h):
    """tau_xz = G dw/dx, tau_yz = G dw/dy por diferencas centrais (np.gradient)."""
    dwdx = np.gradient(W, h, axis=0)
    dwdy = np.gradient(W, h, axis=1)
    return Gnod * dwdx, Gnod * dwdy
