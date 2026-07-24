# ================================================================
# PROJETO 4 — Escoamento em Meio Poroso Compósito 1D
# Método de Diferenças Finitas (MDF) 
# ================================================================
# Equação (Eq. 64-65 do enunciado):
#   v_eps(x) = -k_eps(x) [ dp/dx(x) + f(x) ]
#   -d/dx[ k_eps(x) ( dp/dx(x) + f(x) ) ] = s(x)   em (0,L)
# Condições de contorno: p(0)=p0, p(L)=pL
#
# Esquema conservativo com f(x) DENTRO do fluxo (corrigido):
#   q_{i+1/2} = k_{i+1/2} [ (p_{i+1}-p_i)/h + f_{i+1/2} ]
#   q_{i-1/2} = k_{i-1/2} [ (p_i-p_{i-1})/h + f_{i-1/2} ]
#   -(q_{i+1/2} - q_{i-1/2})/h = s_i
#
# Isso resulta na MESMA matriz A de antes (só depende de k), mas
# com o lado direito modificado para incluir o termo de f:
#   F_i = s_i + (k_{i+1/2} f_{i+1/2} - k_{i-1/2} f_{i-1/2}) / h
# ================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.linalg import solve

# ── Parâmetros do problema ────────────────────────────────────
L     = 1.0
GAMMA = 0.5
K1    = 1.0
K2    = 0.01
P0    = 0.0
PL    = 1.0

K_EF = 1.0 / (GAMMA/K1 + (1-GAMMA)/K2)
print(f"Permeabilidade efetiva: kef = {K_EF:.6f}")
print(f"Fluxo efetivo (s=0,f=0): vef = {-K_EF*(PL-P0)/L:.6f}")

def permeabilidade(x, eps):
    y = (x / eps) % 1.0
    return np.where(y < GAMMA, K1, K2)

def k_interface(x_mid, eps):
    return permeabilidade(x_mid, eps)

def montar_sistema(N, eps, fonte_s=None, forca_f=None):
    """
    Monta Ax=F pelo MDF conservativo, agora com suporte correto a
    f(x) DENTRO do fluxo (Eq. 64) e s(x) como termo fonte separado
    (Eq. 65). Ambos podem ser usados simultaneamente.

    fonte_s: termo fonte s(x) (lado direito "puro")
    forca_f: força externa f(x), acoplada a k dentro do fluxo
    """
    h = L / N
    xi = np.linspace(0, L, N+1)

    x_meio = xi[:-1] + h/2
    k_meio = k_interface(x_meio, eps)

    k_menos = k_meio[:-1]   # k_{i-1/2}, i=1..N-1
    k_mais  = k_meio[1:]    # k_{i+1/2}, i=1..N-1

    n = N - 1

    diag_princ = (k_menos + k_mais) / h**2
    diag_sup   = -k_mais[:-1] / h**2
    diag_inf   = -k_menos[1:] / h**2
    A = np.diag(diag_princ) + np.diag(diag_sup, 1) + np.diag(diag_inf, -1)

    # Termo fonte s(x)
    if fonte_s is None:
        s = np.zeros(n)
    else:
        s = fonte_s(xi[1:-1])

    F = s.copy()

    # Termo de força externa f(x), corretamente acoplado a k
    # dentro do fluxo (Eq. 64): contribui como
    #   (k_{i+1/2} f_{i+1/2} - k_{i-1/2} f_{i-1/2}) / h
    if forca_f is not None:
        f_meio = forca_f(x_meio)          # f nos pontos médios xi+1/2
        f_menos = f_meio[:-1]             # f_{i-1/2}, i=1..N-1
        f_mais  = f_meio[1:]              # f_{i+1/2}, i=1..N-1
        F += (k_mais * f_mais - k_menos * f_menos) / h

    # Condições de contorno de Dirichlet (mesma forma de antes;
    # o termo de f não altera a forma como P0/PL entram, pois
    # eles multiplicam apenas k, não f)
    F[0]  += k_menos[0]  * P0 / h**2
    F[-1] += k_mais[-1]  * PL / h**2

    return A, F, xi

def resolver_mdf(N, eps, fonte_s=None, forca_f=None):
    A, F, xi = montar_sistema(N, eps, fonte_s, forca_f)
    p_int = solve(A, F)
    p = np.concatenate([[P0], p_int, [PL]])
    return xi, p

def solucao_homogeneizada(xi, fonte_s=None, forca_f=None):
    """
    Modelo homogeneizado (Eq. 70): -d/dx[kef(dp0/dx+f(x))] = s(x)
    Resolvido numericamente com N=1000 para o caso geral, e em
    forma fechada p0(x) = x para o caso s=0, f=0.
    """
    if fonte_s is None and forca_f is None:
        return P0 + (PL - P0) / L * xi

    N = 1000
    h = L / N
    xi_h = np.linspace(0, L, N+1)
    n = N - 1
    x_meio = xi_h[:-1] + h/2

    A = (K_EF/h**2) * (2*np.eye(n) - np.diag(np.ones(n-1),1)
                                    - np.diag(np.ones(n-1),-1))
    s = np.zeros(n) if fonte_s is None else fonte_s(xi_h[1:-1])
    F = s.copy()
    if forca_f is not None:
        f_meio = forca_f(x_meio)
        f_menos = f_meio[:-1]
        f_mais  = f_meio[1:]
        F += K_EF * (f_mais - f_menos) / h
    F[0]  += K_EF * P0 / h**2
    F[-1] += K_EF * PL / h**2
    p_int = solve(A, F)
    p_homo = np.concatenate([[P0], p_int, [PL]])
    return np.interp(xi, xi_h, p_homo)

def erro_relativo(p_num, p_ref):
    return np.linalg.norm(p_num - p_ref) / np.linalg.norm(p_ref)

def forca_externa(x, g0=1.0, alpha=0.5):
    """f(x) = g0 + alpha*cos(2*pi*x/L) — agora tratada como força
    dentro do fluxo (Eq. 64), não mais como termo fonte s(x)."""
    return g0 + alpha * np.cos(2 * np.pi * x / L)

# ── Testes rápidos de sanidade ─────────────────────────────────
mpl.rcParams.update({'figure.dpi': 100})

print("\n" + "="*65)
print("TESTE 1 — Caso base s=0, f=0 (deve ser idêntico à versão v1)")
print("="*65)
N = 400
xi, p = resolver_mdf(N, eps=0.1)
p_ref = solucao_homogeneizada(xi)
print(f"  erro relativo vs homogeneizado: {erro_relativo(p, p_ref):.4e}")

print("\n" + "="*65)
print("TESTE 2 — Caso com f(x) corretamente dentro do gradiente")
print("="*65)
for eps in [0.5, 0.2, 0.1, 0.05]:
    xi, p_f = resolver_mdf(N, eps, forca_f=forca_externa)
    p_homo_f = solucao_homogeneizada(xi, forca_f=forca_externa)
    e = erro_relativo(p_f, p_homo_f)
    print(f"  eps={eps:.3f}: p(0)={p_f[0]:.4f}, p(L)={p_f[-1]:.4f}, "
          f"erro vs homogeneizado = {e:.4e}")

print("\n" + "="*65)
print("TESTE 3 — Comparação direta: f(x) como fonte s(x) (ANTIGO) "
      "vs f(x) dentro do gradiente (NOVO)")
print("="*65)
xi, p_antigo = resolver_mdf(N, eps=0.1, fonte_s=forca_externa)
xi, p_novo   = resolver_mdf(N, eps=0.1, forca_f=forca_externa)
diff = erro_relativo(p_novo, p_antigo)
print(f"  Diferença relativa entre as duas formulações: {diff:.4e}")
print(f"  (uma diferença grande confirma que são fisicamente "
      f"problemas distintos, como o Paulo apontou)")

# ═══════════════════════════════════════════════════════════════
# Regenera Figura 4 (caso com força externa) 
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

xi_ref = np.linspace(0, L, 1000)
p_homo_novo = solucao_homogeneizada(xi_ref, forca_f=forca_externa)
p_homo_antigo = solucao_homogeneizada(xi_ref, fonte_s=forca_externa)

axes[0].plot(xi_ref, p_homo_novo, 'k-', lw=2.5,
             label='$p_0(x)$ homog. — $f$ no gradiente (NOVO)', zorder=5)
axes[0].plot(xi_ref, p_homo_antigo, 'gray', lw=2, linestyle='--',
             label='$p_0(x)$ homog. — $f$ como fonte (ANTIGO)', zorder=4)

cores_f = ['steelblue', 'darkorange', 'green']
for eps, cor in zip([0.2, 0.1, 0.05], cores_f):
    xi, p = resolver_mdf(400, eps, forca_f=forca_externa)
    axes[0].plot(xi, p, color=cor, lw=1.2, alpha=0.85,
                 label=f'$p^\\varepsilon(x)$, $\\varepsilon={eps}$ (NOVO)')

axes[0].set_xlabel('$x$'); axes[0].set_ylabel('$p(x)$')
axes[0].set_title('Pressão com força externa — formulação corrigida')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

x_f = np.linspace(0, L, 500)
axes[1].plot(x_f, forca_externa(x_f), 'r-', lw=2)
axes[1].set_xlabel('$x$'); axes[1].set_ylabel('$f(x)$')
axes[1].set_title('Força externa $f(x) = 1 + 0.5\\cos(2\\pi x/L)$')
axes[1].grid(True, alpha=0.3)

plt.suptitle('Caso com força externa — v2 (f dentro do gradiente)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
import os
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/fig4_forca_externa_v2.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nFigura 4 (v2) salva em outputs/fig4_forca_externa_v2.png")
print("\n✓ MDF v2 concluído.")
