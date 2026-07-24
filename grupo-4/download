# ================================================================
# PROJETO 4 — PINN POR SUBDOMÍNIOS
# ================================================================
# Implementa a Seção 4.8 (item 2) do enunciado: em vez de uma única
# rede global treinada sobre kε(x) descontínuo, o domínio é
# decomposto nos subdomínios de fase (onde k é constante), cada um
# com sua própria pequena rede neural. Nas interfaces microscópicas
# impõem-se as condições de casamento:
#   p1 = p2                                  (continuidade de pressão)
#   k1(dp1/dx + f1) = k2(dp2/dx + f2)         (continuidade de fluxo)
# via um termo extra L_I na função de perda (Eq. 81 do enunciado:
# L = L_EDP + L_CC + L_I).
# ================================================================

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import solve
import matplotlib.pyplot as plt
import os

torch.manual_seed(42)
np.random.seed(42)

L, GAMMA, K1, K2, P0, PL = 1.0, 0.5, 1.0, 0.01, 0.0, 1.0
EPS = 0.1  # mesmo valor usado na comparação original (Estratégia 2)

# ── Referência: MDF microscópico (para comparação) ──────────────
def permeabilidade_np(x, eps):
    y = (x / eps) % 1.0
    return np.where(y < GAMMA, K1, K2)

def resolver_mdf(N, eps):
    h = L / N
    xi = np.linspace(0, L, N+1)
    x_meio = xi[:-1] + h/2
    k_meio = permeabilidade_np(x_meio, eps)
    k_menos, k_mais = k_meio[:-1], k_meio[1:]
    n = N - 1
    A = np.diag((k_menos+k_mais)/h**2) + \
        np.diag(-k_mais[:-1]/h**2, 1) + np.diag(-k_menos[1:]/h**2, -1)
    F = np.zeros(n)
    F[0]  += k_menos[0] * P0 / h**2
    F[-1] += k_mais[-1] * PL / h**2
    p_int = solve(A, F)
    return xi, np.concatenate([[P0], p_int, [PL]])

def erro_relativo(p_num, p_ref):
    return np.linalg.norm(p_num - p_ref) / np.linalg.norm(p_ref)

# ── Decomposição do domínio em subdomínios de fase ──────────────
# Para GAMMA=0.5 e EPS=0.1, cada subdomínio tem comprimento
# GAMMA*EPS = (1-GAMMA)*EPS = 0.05, alternando k1, k2, k1, k2, ...
n_periodos = int(round(L / EPS))
breakpoints = []
fases = []
for m in range(n_periodos):
    x0 = m * EPS
    x_meio = x0 + GAMMA * EPS
    x1 = (m + 1) * EPS
    breakpoints.append((x0, x_meio))
    fases.append(K1)
    breakpoints.append((x_meio, x1))
    fases.append(K2)

n_seg = len(breakpoints)
print(f"Número de subdomínios: {n_seg} (períodos={n_periodos}, "
      f"comprimento por subdomínio={GAMMA*EPS:.3f})")

# ── Rede pequena por subdomínio ─────────────────────────────────
class SubPINN(nn.Module):
    """Rede pequena, válida apenas dentro de um subdomínio.
    Entrada normalizada para [-1,1] dentro do subdomínio."""
    def __init__(self, x0, x1, n_hidden=16, n_layers=2):
        super().__init__()
        self.x0, self.x1 = x0, x1
        layers = [nn.Linear(1, n_hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        layers.append(nn.Linear(n_hidden, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def normalizar(self, x):
        xm = (self.x0 + self.x1) / 2
        xr = (self.x1 - self.x0) / 2
        return (x - xm) / xr

    def forward(self, x):
        return self.net(self.normalizar(x))

# Instancia uma rede por subdomínio
subredes = [SubPINN(x0, x1, n_hidden=16, n_layers=2)
            for (x0, x1) in breakpoints]

todos_parametros = []
for sub in subredes:
    todos_parametros += list(sub.parameters())

# ── Pré-treino (warm start): alinhar cada sub-rede à tendência
# linear homogeneizada p0(x)=x antes do treino físico conjunto.
# Isso evita que a informação das condições de contorno globais
# tenha que se propagar lentamente subdomínio a subdomínio.
CKPT_PATH_PRE = 'checkpoint_subdominios.pt'
if not os.path.exists(CKPT_PATH_PRE):
    print("Pré-treinando sub-redes na tendência linear p0(x)=x...")
    opt_pre = torch.optim.Adam(todos_parametros, lr=5e-3)
    for (x0, x1), sub in zip(breakpoints, subredes):
        xs = torch.linspace(x0, x1, 20).reshape(-1, 1)
        ys = xs.clone()  # p0(x) = x
        for _ in range(300):
            opt_pre.zero_grad()
            pred = sub(xs)
            loss_pre = torch.mean((pred - ys)**2)
            loss_pre.backward()
            opt_pre.step()
    print("Pré-treino concluído.")

optimizer = torch.optim.Adam(todos_parametros, lr=2e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)

# Pontos de colocação internos por subdomínio (evitando as bordas)
N_COL_POR_SEG = 12
x_col_por_seg = []
for (x0, x1) in breakpoints:
    margem = 0.05 * (x1 - x0)
    xs = torch.linspace(x0 + margem, x1 - margem, N_COL_POR_SEG,
                         requires_grad=True).reshape(-1, 1)
    x_col_por_seg.append(xs)

# Pontos de interface (entre subdomínios consecutivos)
x_interfaces = torch.tensor(
    [[breakpoints[i][1]] for i in range(n_seg - 1)],
    dtype=torch.float32, requires_grad=True)

x_cc = torch.tensor([[P0], [L]], dtype=torch.float32)
# nota: P0 é avaliado na 1a subrede, PL na ultima

LAMBDA_CC = 100.0
LAMBDA_I  = 100.0
N_EPOCHS  = 2000   # épocas POR EXECUÇÃO deste script (checkpoint)

CKPT_PATH = 'checkpoint_subdominios.pt'
historico = {'total': [], 'edp': [], 'cc': [], 'interface': []}
epoca_inicial = 0

if os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, weights_only=False)
    for sub, sd in zip(subredes, ckpt['subredes_state']):
        sub.load_state_dict(sd)
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
    historico = ckpt['historico']
    epoca_inicial = ckpt['epoca']
    print(f"Checkpoint carregado: retomando da época {epoca_inicial}")
else:
    print("Nenhum checkpoint encontrado, iniciando do zero.")

print(f"\nTreinando por {N_EPOCHS} épocas adicionais "
      f"(total acumulado alvo: {epoca_inicial + N_EPOCHS})...")
for epoch in range(epoca_inicial, epoca_inicial + N_EPOCHS):
    optimizer.zero_grad()

    # ── Resíduo EDP em cada subdomínio (s=0, sem força) ─────────
    loss_edp_total = 0.0
    for sub, x_col, k_i in zip(subredes, x_col_por_seg, fases):
        p_pred = sub(x_col)
        dp_dx = torch.autograd.grad(
            p_pred, x_col, grad_outputs=torch.ones_like(p_pred),
            create_graph=True)[0]
        flux = k_i * dp_dx  # s(x)=0, f(x)=0 neste teste
        d_flux_dx = torch.autograd.grad(
            flux, x_col, grad_outputs=torch.ones_like(flux),
            create_graph=True)[0]
        residuo = -d_flux_dx
        loss_edp_total += torch.mean(residuo**2)
    loss_edp_total /= n_seg

    # ── Condições de contorno globais ────────────────────────────
    p0_pred = subredes[0](x_cc[0:1])
    pL_pred = subredes[-1](x_cc[1:2])
    loss_cc = (p0_pred - P0)**2 + (pL_pred - PL)**2
    loss_cc = loss_cc.mean()

    # ── Condições de interface: continuidade de p e de fluxo ─────
    loss_interface = 0.0
    for i in range(n_seg - 1):
        x_int = x_interfaces[i:i+1]
        sub_esq, sub_dir = subredes[i], subredes[i+1]
        k_esq, k_dir = fases[i], fases[i+1]

        p_esq = sub_esq(x_int)
        p_dir = sub_dir(x_int)
        cont_p = (p_esq - p_dir)**2

        dp_esq = torch.autograd.grad(
            p_esq, x_int, grad_outputs=torch.ones_like(p_esq),
            create_graph=True)[0]
        dp_dir = torch.autograd.grad(
            p_dir, x_int, grad_outputs=torch.ones_like(p_dir),
            create_graph=True)[0]
        cont_fluxo = (k_esq * dp_esq - k_dir * dp_dir)**2

        loss_interface += (cont_p + cont_fluxo).mean()
    loss_interface /= (n_seg - 1)

    loss = loss_edp_total + LAMBDA_CC * loss_cc + LAMBDA_I * loss_interface
    loss.backward()
    optimizer.step()
    scheduler.step()

    historico['total'].append(loss.item())
    historico['edp'].append(loss_edp_total.item())
    historico['cc'].append(loss_cc.item())
    historico['interface'].append(loss_interface.item())

    if (epoch+1) % 1000 == 0:
        print(f"  Época {epoch+1:5d} | Loss={loss.item():.2e} | "
              f"EDP={loss_edp_total.item():.2e} | "
              f"CC={loss_cc.item():.2e} | Interface={loss_interface.item():.2e}")

epoca_final = epoca_inicial + N_EPOCHS
torch.save({
    'subredes_state': [sub.state_dict() for sub in subredes],
    'optimizer_state': optimizer.state_dict(),
    'scheduler_state': scheduler.state_dict(),
    'historico': historico,
    'epoca': epoca_final,
}, CKPT_PATH)
print(f"\nTreinamento desta execução concluído. Checkpoint salvo "
      f"(época acumulada = {epoca_final}).")

# ── Avaliação da solução completa (por trechos) ──────────────────
def avaliar_subdominios(x_np):
    x_np = np.atleast_1d(x_np)
    p_out = np.zeros_like(x_np)
    for (x0, x1), sub in zip(breakpoints, subredes):
        mask = (x_np >= x0 - 1e-9) & (x_np <= x1 + 1e-9)
        if not np.any(mask):
            continue
        x_t = torch.tensor(x_np[mask].reshape(-1, 1), dtype=torch.float32)
        with torch.no_grad():
            p_out[mask] = sub(x_t).numpy().flatten()
    return p_out

x_plot = np.linspace(0, L, 1000)
p_pinn_sub = avaliar_subdominios(x_plot)

xi_mdf, p_mdf = resolver_mdf(400, EPS)
p_mdf_interp = np.interp(x_plot, xi_mdf, p_mdf)

e_rel = erro_relativo(p_pinn_sub, p_mdf_interp)
print(f"\nErro relativo PINN por subdomínios vs MDF: {e_rel:.4e}")
print(f"(Para comparação: PINN monolítica original tinha e_rel=5,05e-02, "
      f"mas convergia para uma curva SUAVE que ignorava a oscilação.)")

# ═══════════════════════════════════════════════════════════════
# FIGURA 11 — Comparação PINN subdomínios vs MDF vs PINN monolítica
# ═══════════════════════════════════════════════════════════════
os.makedirs('outputs_final', exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

axes[0].plot(xi_mdf, p_mdf, 'k--', lw=2.0, label=f'MDF ($\\varepsilon={EPS}$)', zorder=5)
axes[0].plot(x_plot, p_pinn_sub, 'r-', lw=1.8, alpha=0.85,
             label='PINN por subdomínios')
axes[0].set_xlabel('$x$'); axes[0].set_ylabel('$p(x)$')
axes[0].set_title('PINN por subdomínios vs MDF')
axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[0].text(0.05, 0.92, f'$e_{{rel}}={e_rel:.3e}$', transform=axes[0].transAxes,
             fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# Zoom para mostrar os degraus capturados
xz0, xz1 = 0.3, 0.6
mask_zoom = (x_plot >= xz0) & (x_plot <= xz1)
mask_zoom_mdf = (xi_mdf >= xz0) & (xi_mdf <= xz1)
axes[1].plot(xi_mdf[mask_zoom_mdf], p_mdf[mask_zoom_mdf], 'k--', lw=2.0,
             label='MDF', zorder=5, marker='o', ms=3)
axes[1].plot(x_plot[mask_zoom], p_pinn_sub[mask_zoom], 'r-', lw=2.0,
             label='PINN por subdomínios')
axes[1].set_xlabel('$x$'); axes[1].set_ylabel('$p(x)$')
axes[1].set_title(f'Zoom [{xz0},{xz1}] — detalhe da microestrutura')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.suptitle('PINN por subdomínios (com condições de interface) — Estratégia 2 revisada',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs_final/fig11_pinn_subdominios.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nFigura 11 salva em outputs_final/fig11_pinn_subdominios.png")
