"""Solver por Metodo de Elementos Finitos (MEF) para -div(G grad w) = 0.

Forma fraca: achar w tal que  int_Omega G grad(w).grad(v) dOmega = 0  para todo v
admissivel. Malha estruturada de elementos quadrilaterais Q4 (bilineares), G
atribuido por elemento (testando o centroide: dentro/fora da inclusao). Quadratura
de Gauss 2x2.

BCs:
  - Neumann homogeneo (topo/base) e condicao NATURAL -> nenhum termo extra.
  - Dirichlet (laterais x=0, x=L) por eliminacao/penalizacao de graus de liberdade.

MEF escrito na mao (transparencia > caixa-preta). Stack: numpy + scipy.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from ..problema3.geometry import Case, grid, inclusion_mask

# Pontos e pesos de Gauss 2x2 no elemento de referencia [-1,1]^2
_GP = 1.0 / np.sqrt(3.0)
_GAUSS = [(-_GP, -_GP), (_GP, -_GP), (_GP, _GP), (-_GP, _GP)]


def _shape(xi, eta):
    """Funcoes de forma Q4 e suas derivadas em (xi, eta). Retorna (Nf, dN)."""
    Nf = 0.25 * np.array([
        (1 - xi) * (1 - eta),
        (1 + xi) * (1 - eta),
        (1 + xi) * (1 + eta),
        (1 - xi) * (1 + eta),
    ])
    # dN/dxi e dN/deta
    dN = 0.25 * np.array([
        [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],   # d/dxi
        [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)],        # d/deta
    ])
    return Nf, dN


def solve_fem(case: Case, N=80):
    """Resolve por MEF Q4 numa malha NxN elementos / (N+1)x(N+1) nos.

    Retorna dict com X, Y (malha 'ij'), W, txz, tyz, Gef, h, N, case.
    """
    X, Y, x1d, h = grid(N, case)
    n = N + 1
    coords = np.column_stack([X.ravel(), Y.ravel()])  # no global k = i*n + j

    def node(i, j):
        return i * n + j

    ndof = n * n
    K = sp.lil_matrix((ndof, ndof))

    # G por elemento via centroide
    for ei in range(N):
        for ej in range(N):
            # nos do elemento (sentido anti-horario): SW, SE, NE, NW
            ns = [node(ei, ej), node(ei + 1, ej), node(ei + 1, ej + 1), node(ei, ej + 1)]
            xe = coords[ns, 0]
            ye = coords[ns, 1]
            cxe, cye = xe.mean(), ye.mean()
            Ge = case.Gi if inclusion_mask(cxe, cye, case) else case.Gm

            Ke = np.zeros((4, 4))
            for (xi, eta) in _GAUSS:
                _, dN = _shape(xi, eta)
                # Jacobiano J = dN . [xe ye]
                J = dN @ np.column_stack([xe, ye])  # 2x2
                detJ = np.linalg.det(J)
                invJ = np.linalg.inv(J)
                dNxy = invJ @ dN  # derivadas em x,y, shape (2,4)
                Ke += Ge * (dNxy.T @ dNxy) * detJ  # peso de Gauss = 1

            for a in range(4):
                for b in range(4):
                    K[ns[a], ns[b]] += Ke[a, b]

    K = K.tocsr()
    F = np.zeros(ndof)

    # --- Dirichlet por eliminacao ---
    # x=0 -> w=0 ; x=L -> w=gamma*L
    fixed = {}
    for j in range(n):
        fixed[node(0, j)] = 0.0
        fixed[node(N, j)] = case.gamma * case.L

    free = np.array([k for k in range(ndof) if k not in fixed], dtype=int)
    fixed_idx = np.array(sorted(fixed.keys()), dtype=int)
    fixed_val = np.array([fixed[k] for k in fixed_idx], dtype=float)

    # Move termos conhecidos para o RHS: F_free -= K[free, fixed] @ fixed_val
    F_eff = F[free] - K[free][:, fixed_idx] @ fixed_val
    K_ff = K[free][:, free]

    w = np.zeros(ndof)
    w[fixed_idx] = fixed_val
    w[free] = spsolve(K_ff, F_eff)

    W = w.reshape(n, n)
    txz, tyz = _stresses_fem(W, X, Y, case, N, h)
    Gef = float(np.mean(txz) / case.gamma)

    return dict(X=X, Y=Y, W=W, txz=txz, tyz=tyz, Gef=Gef, h=h, N=N, case=case)


def _stresses_fem(W, X, Y, case, N, h):
    """Tensoes nodais. Usa G nodal (consistente com a visualizacao) e gradiente
    central de w; e a mesma convencao do MDF, o que torna a comparacao justa."""
    from ..problema3.geometry import G_field
    Gnod = G_field(X, Y, case)
    dwdx = np.gradient(W, h, axis=0)
    dwdy = np.gradient(W, h, axis=1)
    return Gnod * dwdx, Gnod * dwdy
