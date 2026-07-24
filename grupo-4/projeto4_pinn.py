# ================================================================
# PROJETO 4 — Escoamento em Meio Poroso Compósito 1D
# Método de Elementos Finitos (MEF) 
# ================================================================
# Formulação fraca com f(x) dentro do fluxo (Eq. 64):
#   v = -k(p'+f),  -d/dx[k(p'+f)] = s(x)
#
# Multiplicando por w, integrando por partes (w(0)=w(L)=0):
#   ∫ k p' w' dx + ∫ k f w' dx = ∫ s w dx
#   ∫ k p' w' dx = ∫ s w dx - ∫ k f w' dx
#
# A matriz de rigidez K é EXATAMENTE a mesma de antes (só depende
# de k). O vetor de carga F ganha um termo extra:
#   F_i = ∫ s Ni dx  -  ∫ k f Ni' dx
# Para elementos lineares, Ni' = ±1/he (constante no elemento):
#   Fe[0] -= (-1/he) * ∫_e k f dx = + (1/he) ∫_e k f dx
#   Fe[1] -= (+1/he) * ∫_e k f dx = - (1/he) ∫_e k f dx
# ================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

L, GAMMA, K1, K2, P0, PL = 1.0, 0.5, 1.0, 0.01, 0.0, 1.0
K_EF = 1.0 / (GAMMA/K1 + (1-GAMMA)/K2)

PTS3 = [-0.774596669, 0.0, 0.774596669]
WTS3 = [0.555555556, 0.888888889, 0.555555556]

def permeabilidade(x, eps):
    y = (x / eps) % 1.0
    return np.where(y < GAMMA, K1, K2)

def k_elemento_gauss(xe, xe1, eps, n_gauss=3):
    he = xe1 - xe
    k_int = 0.0
    for xi_hat, w in zip(PTS3, WTS3):
        x = (xe + xe1)/2 + (he/2)*xi_hat
        k_int += w * permeabilidade(x, eps)
    return k_int * he / 2

def kf_elemento_gauss(xe, xe1, eps, forca_f):
    """Integra k(x)*f(x) sobre o elemento por quadratura de Gauss
    (necessário para o novo termo de carga vindo de f(x))."""
    he = xe1 - xe
    kf_int = 0.0
    for xi_hat, w in zip(PTS3, WTS3):
        x = (xe + xe1)/2 + (he/2)*xi_hat
        kf_int += w * permeabilidade(x, eps) * forca_f(x)
    return kf_int * he / 2

def montar_sistema_mef(N, eps, fonte_s=None, forca_f=None):
    h = L / N
    xi = np.linspace(0, L, N+1)
    K_global = np.zeros((N+1, N+1))
    F_global = np.zeros(N+1)

    for e in range(N):
        xe, xe1 = xi[e], xi[e+1]
        he = xe1 - xe

        k_int = k_elemento_gauss(xe, xe1, eps, n_gauss=3)
        ke_eff = k_int / he**2
        Ke = ke_eff * np.array([[1, -1], [-1, 1]])

        nos = [e, e+1]
        for i_loc in range(2):
            for j_loc in range(2):
                K_global[nos[i_loc], nos[j_loc]] += Ke[i_loc, j_loc]

        Fe = np.zeros(2)

        # Termo de fonte s(x): ∫ s(x) Ni(x) dx
        if fonte_s is not None:
            for xi_hat, w in zip(PTS3, WTS3):
                x = (xe + xe1)/2 + (he/2)*xi_hat
                N1 = (xe1 - x) / he
                N2 = (x - xe) / he
                s_val = fonte_s(x)
                Fe[0] += w * s_val * N1 * (he/2)
                Fe[1] += w * s_val * N2 * (he/2)

        # Termo de força externa f(x): -∫ k f Ni' dx
        # Ni' é constante no elemento: N1'=-1/he, N2'=+1/he
        if forca_f is not None:
            kf_int = kf_elemento_gauss(xe, xe1, eps, forca_f)
            Fe[0] += kf_int / he     # -(-1/he)*kf_int
            Fe[1] -= kf_int / he     # -(+1/he)*kf_int

        F_global[e]   += Fe[0]
        F_global[e+1] += Fe[1]

    return K_global, F_global, xi

def resolver_mef(N, eps, fonte_s=None, forca_f=None):
    K, F, xi = montar_sistema_mef(N, eps, fonte_s, forca_f)
    K_red = K[1:-1, 1:-1].copy()
    F_red = F[1:-1].copy()
    F_red[0]  -= K[1, 0]   * P0
    F_red[-1] -= K[-2, -1] * PL
    p_int = solve(K_red, F_red)
    return xi, np.concatenate([[P0], p_int, [PL]])

def resolver_mdf(N, eps, fonte_s=None, forca_f=None):
    """MDF v2 (com f corrigido) para comparação direta."""
    h = L / N
    xi = np.linspace(0, L, N+1)
    x_meio = xi[:-1] + h/2
    k_meio = permeabilidade(x_meio, eps)
    k_menos, k_mais = k_meio[:-1], k_meio[1:]
    n = N - 1
    A = np.diag((k_menos+k_mais)/h**2) + \
        np.diag(-k_mais[:-1]/h**2, 1) + np.diag(-k_menos[1:]/h**2, -1)
    s = np.zeros(n) if fonte_s is None else fonte_s(xi[1:-1])
    F = s.copy()
    if forca_f is not None:
        f_meio = forca_f(x_meio)
        f_menos, f_mais = f_meio[:-1], f_meio[1:]
        F += (k_mais*f_mais - k_menos*f_menos) / h
    F[0]  += k_menos[0] * P0 / h**2
    F[-1] += k_mais[-1] * PL / h**2
    p_int = solve(A, F)
    return xi, np.concatenate([[P0], p_int, [PL]])

def solucao_homogeneizada(xi, fonte_s=None, forca_f=None):
    if fonte_s is None and forca_f is None:
        return P0 + (PL - P0) / L * xi
    N = 1000
    h = L / N
    xi_h = np.linspace(0, L, N+1)
    n = N - 1
    x_meio = xi_h[:-1] + h/2
    A = (K_EF/h**2) * (2*np.eye(n) - np.diag(np.ones(n-1),1) - np.diag(np.ones(n-1),-1))
    s = np.zeros(n) if fonte_s is None else fonte_s(xi_h[1:-1])
    F = s.copy()
    if forca_f is not None:
        f_meio = forca_f(x_meio)
        F += K_EF * (f_meio[1:] - f_meio[:-1]) / h
    F[0]  += K_EF * P0 / h**2
    F[-1] += K_EF * PL / h**2
    p_int = solve(A, F)
    p_homo = np.concatenate([[P0], p_int, [PL]])
    return np.interp(xi, xi_h, p_homo)

def erro_relativo(p_num, p_ref):
    return np.linalg.norm(p_num - p_ref) / np.linalg.norm(p_ref)

def forca_externa(x, g0=1.0, alpha=0.5):
    return g0 + alpha * np.cos(2 * np.pi * x / L)

# ── Testes de sanidade ──────────────────────────────────────────
print("="*65)
print("TESTE 1 — MDF vs MEF, caso base s=0,f=0 (deve bater, como antes)")
print("="*65)
N, eps = 400, 0.1
_, p_mdf = resolver_mdf(N, eps)
_, p_mef = resolver_mef(N, eps)
print(f"  Diferença MDF vs MEF: {erro_relativo(p_mef, p_mdf):.4e}")

print("\n" + "="*65)
print("TESTE 2 — MDF vs MEF, caso com f(x) corrigido (dentro do grad.)")
print("="*65)
for eps in [0.5, 0.2, 0.1, 0.05]:
    _, p_mdf_f = resolver_mdf(N, eps, forca_f=forca_externa)
    _, p_mef_f = resolver_mef(N, eps, forca_f=forca_externa)
    print(f"  eps={eps:.3f}: diferença MDF vs MEF = "
          f"{erro_relativo(p_mef_f, p_mdf_f):.4e}")

print("\n✓ Se as diferenças MDF-vs-MEF acima forem pequenas (~1e-2 a "
      "1e-12, igual ao caso base), a implementação de f(x) está "
      "consistente entre os dois métodos.")

# ═══════════════════════════════════════════════════════════════
# Figura 7 (v2) — MDF vs MEF com força externa 
# ═══════════════════════════════════════════════════════════════
import os
os.makedirs('outputs', exist_ok=True)

xi_mdf, p_mdf = resolver_mdf(400, 0.1, forca_f=forca_externa)
xi_mef, p_mef = resolver_mef(400, 0.1, forca_f=forca_externa)
xi_ref = np.linspace(0, L, 1000)
p_homo_f = solucao_homogeneizada(xi_ref, forca_f=forca_externa)

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(xi_ref, p_homo_f, 'k--', lw=2.5, label='Homogeneizado $p_0(x)$', zorder=5)
ax.plot(xi_mdf, p_mdf, 'b-', lw=1.5, alpha=0.8, label='MDF (v2, $f$ no gradiente)')
ax.plot(xi_mef, p_mef, 'r:', lw=2.0, alpha=0.9, label='MEF (v2, $f$ no gradiente)')
ax.set_xlabel('$x$'); ax.set_ylabel('$p(x)$')
ax.set_title('MDF vs MEF com força externa (formulação corrigida) — $\\varepsilon=0.1$')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/fig7_mdf_mef_forca_v2.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nFigura 7 (v2) salva em outputs/fig7_mdf_mef_forca_v2.png")
print("✓ MEF v2 concluído.")
