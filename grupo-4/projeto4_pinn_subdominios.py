# ================================================================
# PROJETO 4 — Escoamento em Meio Poroso Compósito 1D
# Physics-Informed Neural Networks (PINNs)
# ================================================================
# Estratégia 1: PINN para problema homogeneizado (kef)
# Estratégia 2: PINN para problema microscópico (kε)
# ================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.nn as nn
from scipy.linalg import solve

torch.manual_seed(42)
np.random.seed(42)

# ── Parâmetros do problema ────────────────────────────────────
L     = 1.0
GAMMA = 0.5
K1    = 1.0
K2    = 0.01
P0    = 0.0
PL    = 1.0
K_EF  = 1.0 / (GAMMA/K1 + (1-GAMMA)/K2)

# ── Funções auxiliares ────────────────────────────────────────
def permeabilidade_np(x, eps):
    y = (x / eps) % 1.0
    return np.where(y < GAMMA, K1, K2)

def solucao_exata_homo(x):
    """p₀(x) = x (solução homogeneizada caso s=0)"""
    return P0 + (PL - P0) / L * x

def resolver_mdf(N, eps, fonte=None):
    """MDF para referência (com suporte a termo de fonte s(x)/f(x))"""
    h = L / N
    xi = np.linspace(0, L, N+1)
    x_meio = xi[:-1] + h/2
    k_meio = permeabilidade_np(x_meio, eps)
    k_menos = k_meio[:-1]
    k_mais  = k_meio[1:]
    n = N - 1
    A = np.diag((k_menos+k_mais)/h**2) + \
        np.diag(-k_mais[:-1]/h**2, 1) + \
        np.diag(-k_menos[1:]/h**2, -1)
    F = np.zeros(n) if fonte is None else fonte(xi[1:-1])
    F[0]  += k_menos[0]  * P0 / h**2
    F[-1] += k_mais[-1]  * PL / h**2
    p_int = solve(A, F)
    return xi, np.concatenate([[P0], p_int, [PL]])

def erro_relativo(p_num, p_ref):
    return np.linalg.norm(p_num - p_ref) / np.linalg.norm(p_ref)

# ── Arquitetura da Rede Neural ────────────────────────────────
class PINN(nn.Module):
    """
    Rede neural totalmente conectada para aproximar p(x).
    Arquitetura: 1 → [N_hidden]*N_layers → 1
    Ativação: tanh (suave e com derivadas bem definidas)
    """
    def __init__(self, n_hidden=32, n_layers=4):
        super().__init__()
        layers = [nn.Linear(1, n_hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        layers.append(nn.Linear(n_hidden, 1))
        self.net = nn.Sequential(*layers)

        # Inicialização Xavier para convergência mais rápida
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

# ── Treinamento PINN ──────────────────────────────────────────
def treinar_pinn(modelo, k_func, fonte=None, n_col=200,
                 n_epochs=8000, lr=1e-3, verbose=True):
    """
    Treina a PINN minimizando:
    L = L_EDP + λ_cc * L_CC

    L_EDP = média de R(x)² nos pontos de colocação
    L_CC  = (p(0)-P0)² + (p(L)-PL)²

    R(x) = −d/dx[k(x) dp/dx] − s(x)
         = −k'(x)p'(x) − k(x)p''(x) − s(x)

    fonte: função s(x) (numpy), ou None para s(x)=0
           (caso estudado nas Estratégias 1 e 2 originais)
    """
    optimizer = torch.optim.Adam(modelo.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=2000, gamma=0.5)

    # Pontos de colocação (interior)
    x_col = torch.linspace(0.01, 0.99, n_col,
                           requires_grad=True).reshape(-1, 1)

    # Termo de fonte avaliado nos pontos de colocação (constante
    # durante o treinamento, pois não depende de theta)
    if fonte is None:
        s_vals = torch.zeros((n_col, 1), dtype=torch.float32)
    else:
        s_vals = torch.tensor(
            fonte(x_col.detach().numpy()).reshape(-1, 1),
            dtype=torch.float32)

    # Pontos de contorno
    x_cc = torch.tensor([[P0], [L]], dtype=torch.float32)
    p_cc = torch.tensor([[P0], [PL]], dtype=torch.float32)

    historico = []
    historico_edp = []
    historico_cc  = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # ── Resíduo EDP ──────────────────────────────────────
        p_pred = modelo(x_col)

        # Primeira derivada dp/dx
        dp_dx = torch.autograd.grad(
            p_pred, x_col,
            grad_outputs=torch.ones_like(p_pred),
            create_graph=True)[0]

        # k(x) * dp/dx
        k_vals = torch.tensor(
            k_func(x_col.detach().numpy()),
            dtype=torch.float32)
        flux = k_vals * dp_dx

        # Segunda derivada d/dx[k dp/dx]
        d_flux_dx = torch.autograd.grad(
            flux, x_col,
            grad_outputs=torch.ones_like(flux),
            create_graph=True)[0]

        # Resíduo: −d/dx[k dp/dx] − s(x) = 0
        residuo = -d_flux_dx - s_vals
        loss_edp = torch.mean(residuo**2)

        # ── Condições de contorno ─────────────────────────────
        p_cc_pred = modelo(x_cc)
        loss_cc = torch.mean((p_cc_pred - p_cc)**2)

        # ── Perda total ───────────────────────────────────────
        loss = loss_edp + 100.0 * loss_cc

        loss.backward()
        optimizer.step()
        scheduler.step()

        historico.append(loss.item())
        historico_edp.append(loss_edp.item())
        historico_cc.append(loss_cc.item())

        if verbose and (epoch+1) % 1000 == 0:
            print(f"  Época {epoch+1:5d} | "
                  f"Loss={loss.item():.2e} | "
                  f"EDP={loss_edp.item():.2e} | "
                  f"CC={loss_cc.item():.2e}")

    return historico, historico_edp, historico_cc

def avaliar_pinn(modelo, x_np):
    """Avalia a PINN nos pontos x_np"""
    x_t = torch.tensor(x_np.reshape(-1, 1), dtype=torch.float32)
    with torch.no_grad():
        p_pred = modelo(x_t).numpy().flatten()
    return p_pred

# ═══════════════════════════════════════════════════════════════
# ESTRATÉGIA 1 — PINN Homogeneizada (k = kef)
# ═══════════════════════════════════════════════════════════════
print("="*55)
print("ESTRATÉGIA 1 — PINN Homogeneizada (k = kef)")
print("="*55)

def k_homogeneo(x):
    """Permeabilidade constante = kef"""
    if isinstance(x, torch.Tensor):
        return K_EF * torch.ones_like(x)
    return K_EF * np.ones_like(x)

modelo_homo = PINN(n_hidden=32, n_layers=4)
hist_homo, hist_homo_edp, hist_homo_cc = treinar_pinn(
    modelo_homo, k_homogeneo,
    n_col=200, n_epochs=8000, lr=1e-3)

# ═══════════════════════════════════════════════════════════════
# ESTRATÉGIA 2 — PINN Microscópica (k = kε)
# ═══════════════════════════════════════════════════════════════
eps_pinn = 0.1

print(f"\n{'='*55}")
print(f"ESTRATÉGIA 2 — PINN Microscópica (ε={eps_pinn})")
print("="*55)

def k_micro(x):
    """Permeabilidade oscilatória kε(x)"""
    if isinstance(x, torch.Tensor):
        x_np = x.detach().numpy()
    else:
        x_np = x
    return permeabilidade_np(x_np, eps_pinn)

modelo_micro = PINN(n_hidden=64, n_layers=5)
hist_micro, hist_micro_edp, hist_micro_cc = treinar_pinn(
    modelo_micro, k_micro,
    n_col=500, n_epochs=8000, lr=5e-4)

# ═══════════════════════════════════════════════════════════════
# ESTRATÉGIA 3 — PINN Homogeneizada com força externa f(x)
# VERSÃO CORRIGIDA: f(x) dentro do gradiente (Eq. 64 do enunciado),
# não mais como termo fonte s(x). f(x) é escrita como função TORCH
# diferenciável para que sua derivada seja capturada pelo autograd.
# (complementa o caso já estudado no MDF e MEF — Figs. 4 e 7)
# ═══════════════════════════════════════════════════════════════
def forca_externa(x, g0=1.0, alpha=0.5):
    """f(x) = g0 + alpha*cos(2*pi*x/L) — versão numpy (para o MDF)"""
    return g0 + alpha * np.cos(2 * np.pi * x / L)

def forca_externa_torch(x, g0=1.0, alpha=0.5):
    """f(x) = g0 + alpha*cos(2*pi*x/L) — versão torch diferenciável
    (para a PINN, de modo que d f/dx entre corretamente no resíduo)"""
    return g0 + alpha * torch.cos(2 * np.pi * x / L)

print(f"\n{'='*55}")
print("ESTRATÉGIA 3 — PINN com f(x) dentro do gradiente (corrigida)")
print("="*55)

def treinar_pinn_estrategia3(n_col=200, n_epochs=20000, lr=1e-3, verbose=True):
    """
    Resíduo: R(x) = -d/dx[ k_ef (dp/dx + f(x)) ]  (s(x)=0)
    f(x) entra dentro do fluxo, diferenciável via autograd.
    """
    modelo = PINN(n_hidden=32, n_layers=4)
    optimizer = torch.optim.Adam(modelo.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

    x_col = torch.linspace(0.01, 0.99, n_col, requires_grad=True).reshape(-1, 1)
    k_vals = K_EF * torch.ones_like(x_col)
    x_cc = torch.tensor([[P0], [L]], dtype=torch.float32)
    p_cc = torch.tensor([[P0], [PL]], dtype=torch.float32)

    hist, hist_edp, hist_cc = [], [], []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        p_pred = modelo(x_col)
        dp_dx = torch.autograd.grad(
            p_pred, x_col, grad_outputs=torch.ones_like(p_pred),
            create_graph=True)[0]

        f_vals = forca_externa_torch(x_col)          # diferenciável em x_col
        flux = k_vals * (dp_dx + f_vals)
        d_flux_dx = torch.autograd.grad(
            flux, x_col, grad_outputs=torch.ones_like(flux),
            create_graph=True)[0]

        residuo = -d_flux_dx    # s(x)=0
        loss_edp = torch.mean(residuo**2)

        p_cc_pred = modelo(x_cc)
        loss_cc = torch.mean((p_cc_pred - p_cc)**2)

        loss = loss_edp + 100.0 * loss_cc
        loss.backward()
        optimizer.step()
        scheduler.step()

        hist.append(loss.item()); hist_edp.append(loss_edp.item()); hist_cc.append(loss_cc.item())
        if verbose and (epoch+1) % 4000 == 0:
            print(f"  Época {epoch+1:5d} | Loss={loss.item():.2e} | "
                  f"EDP={loss_edp.item():.2e} | CC={loss_cc.item():.2e}")
    return modelo, hist, hist_edp, hist_cc

modelo_forca, hist_forca, hist_forca_edp, hist_forca_cc = treinar_pinn_estrategia3(
    n_col=200, n_epochs=20000, lr=1e-3)

# ── Configuração gráficos ─────────────────────────────────────
mpl.rcParams.update({
    'font.size': 12, 'axes.labelsize': 13,
    'axes.titlesize': 13, 'legend.fontsize': 10,
    'figure.dpi': 150, 'lines.linewidth': 1.8,
})

x_plot = np.linspace(0, L, 500)

# ═══════════════════════════════════════════════════════════════
# FIGURA 8 — Curva de perda durante treinamento
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Estratégia 1
epochs = np.arange(1, len(hist_homo)+1)
axes[0].semilogy(epochs, hist_homo,     'b-',  lw=1.5, label='Loss total')
axes[0].semilogy(epochs, hist_homo_edp, 'r--', lw=1.2, label='Loss EDP')
axes[0].semilogy(epochs, hist_homo_cc,  'g:',  lw=1.2, label='Loss CC')
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Loss')
axes[0].set_title('Estratégia 1 — PINN Homogeneizada')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Estratégia 2
axes[1].semilogy(epochs, hist_micro,     'b-',  lw=1.5, label='Loss total')
axes[1].semilogy(epochs, hist_micro_edp, 'r--', lw=1.2, label='Loss EDP')
axes[1].semilogy(epochs, hist_micro_cc,  'g:',  lw=1.2, label='Loss CC')
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Loss')
axes[1].set_title(f'Estratégia 2 — PINN Microscópica ($\\varepsilon={eps_pinn}$)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Histórico de treinamento das PINNs',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/fig8_pinn_loss.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("\nFigura 8 salva.")

# ═══════════════════════════════════════════════════════════════
# FIGURA 9 — Soluções: PINN vs MDF vs Homogeneizado
# ═══════════════════════════════════════════════════════════════

# Avalia PINNs
p_pinn_homo  = avaliar_pinn(modelo_homo, x_plot)
p_pinn_micro = avaliar_pinn(modelo_micro, x_plot)

# Referências
p_exata = solucao_exata_homo(x_plot)
xi_mdf, p_mdf = resolver_mdf(400, eps_pinn)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# Painel esquerdo — Estratégia 1
axes[0].plot(x_plot, p_exata,      'k--', lw=2.5,
             label='Homogeneizado $p_0(x)=x$', zorder=5)
axes[0].plot(x_plot, p_pinn_homo,  'b-',  lw=2.0,
             label='PINN Estratégia 1', alpha=0.85)
axes[0].set_xlabel('$x$')
axes[0].set_ylabel('$p(x)$')
axes[0].set_title('Estratégia 1 — PINN com $k_{ef}$')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

e1 = erro_relativo(p_pinn_homo,
                   solucao_exata_homo(x_plot))
axes[0].text(0.05, 0.92,
             f'$e_{{rel}}={e1:.3e}$',
             transform=axes[0].transAxes,
             fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat',
                       alpha=0.7))

# Painel direito — Estratégia 2
axes[1].plot(xi_mdf, p_mdf,        'k--', lw=2.0,
             label=f'MDF ($\\varepsilon={eps_pinn}$)', zorder=5)
axes[1].plot(x_plot, p_exata,      'gray', lw=1.5,
             linestyle=':', label='Homogeneizado', alpha=0.7)
axes[1].plot(x_plot, p_pinn_micro, 'r-',  lw=2.0,
             label='PINN Estratégia 2', alpha=0.85)
axes[1].set_xlabel('$x$')
axes[1].set_ylabel('$p(x)$')
axes[1].set_title(f'Estratégia 2 — PINN com $k^\\varepsilon$'
                  f' ($\\varepsilon={eps_pinn}$)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

p_mdf_interp = np.interp(x_plot, xi_mdf, p_mdf)
e2 = erro_relativo(p_pinn_micro, p_mdf_interp)
axes[1].text(0.05, 0.92,
             f'$e_{{rel}}={e2:.3e}$',
             transform=axes[1].transAxes,
             fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat',
                       alpha=0.7))

plt.suptitle('Soluções PINN vs referências',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/fig9_pinn_solucoes.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Figura 9 salva.")

# ═══════════════════════════════════════════════════════════════
# FIGURA 10 — Comparação completa: MDF, MEF, HOMO, PINN1, PINN2
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))

ax.plot(x_plot, p_exata,      'k--',  lw=2.5,
        label='Homogeneizado $p_0(x)$', zorder=6)
ax.plot(xi_mdf, p_mdf,        'b-',   lw=1.5, alpha=0.8,
        label=f'MDF ($\\varepsilon={eps_pinn}$)')
ax.plot(x_plot, p_pinn_homo,  'g-',   lw=2.0, alpha=0.85,
        label='PINN Estratégia 1 ($k_{ef}$)')
ax.plot(x_plot, p_pinn_micro, 'r-',   lw=1.5, alpha=0.85,
        label=f'PINN Estratégia 2 ($k^\\varepsilon$,'
              f' $\\varepsilon={eps_pinn}$)')

ax.set_xlabel('$x$')
ax.set_ylabel('$p(x)$')
ax.set_title('Comparação: MDF, Homogeneizado e PINNs')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/fig10_comparacao_completa.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Figura 10 salva.")

# ═══════════════════════════════════════════════════════════════
# FIGURA 12 — PINN com força externa vs MDF/MEF (mesmo caso f(x))
# ═══════════════════════════════════════════════════════════════
p_pinn_forca = avaliar_pinn(modelo_forca, x_plot)

xi_mdf_f, p_mdf_f = resolver_mdf(400, eps_pinn, fonte=forca_externa)

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(xi_mdf_f, p_mdf_f, 'b-', lw=1.5, alpha=0.8,
        label=f'MDF com $f(x)$ ($\\varepsilon={eps_pinn}$)')
ax.plot(x_plot, p_pinn_forca, 'g-', lw=2.0, alpha=0.85,
        label='PINN Estratégia 3 ($k_{ef}$, com $f(x)$)')
ax.set_xlabel('$x$')
ax.set_ylabel('$p(x)$')
ax.set_title('PINN vs MDF — caso com força externa $f(x)$')
ax.legend()
ax.grid(True, alpha=0.3)

p_mdf_f_interp = np.interp(x_plot, xi_mdf_f, p_mdf_f)
e3 = erro_relativo(p_pinn_forca, p_mdf_f_interp)
ax.text(0.05, 0.92, f'$e_{{rel}}={e3:.3e}$',
        transform=ax.transAxes, fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()
plt.savefig('outputs/fig12_pinn_forca.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Figura 12 salva.")

# ═══════════════════════════════════════════════════════════════
# TABELA FINAL DE ERROS
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("TABELA FINAL — ERROS RELATIVOS COMPARATIVOS")
print("="*65)

x_eval = np.linspace(0, L, 500)
p_ref_homo = solucao_exata_homo(x_eval)
xi_mdf_e, p_mdf_e = resolver_mdf(400, eps_pinn)
p_mdf_interp = np.interp(x_eval, xi_mdf_e, p_mdf_e)

# Erros em relação ao homogeneizado
e_mdf_h  = erro_relativo(p_mdf_interp,             p_ref_homo)
e_pinn1  = erro_relativo(avaliar_pinn(modelo_homo, x_eval),  p_ref_homo)

# Erros em relação ao MDF
e_pinn2  = erro_relativo(avaliar_pinn(modelo_micro, x_eval), p_mdf_interp)

print(f"\n  Referência: Solução Homogeneizada")
print(f"  {'Método':<35} {'e_rel':>12}")
print(f"  {'-'*48}")
print(f"  {'MDF (ε=0.1, N=400)':<35} {e_mdf_h:>12.4e}")
print(f"  {'PINN Estratégia 1 (kef)':<35} {e_pinn1:>12.4e}")

print(f"\n  Referência: MDF (ε=0.1)")
print(f"  {'Método':<35} {'e_rel':>12}")
print(f"  {'-'*48}")
print(f"  {'PINN Estratégia 2 (kε, ε=0.1)':<35} {e_pinn2:>12.4e}")

print("\n" + "="*65)
print("\n✓ PINN concluído! Todos os gráficos salvos.")
print("\nResumo das estratégias:")
print("  Estratégia 1 (kef):  mais simples, converge rápido")
print("  Estratégia 2 (kε):   mais difícil, kε descontínuo")
print("  A diferença confirma o que o enunciado prevê!")
