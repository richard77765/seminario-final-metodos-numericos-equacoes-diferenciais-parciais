# Grupo 2
import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt

# -----------------------------
# Parâmetros
# -----------------------------
L = 1.0                 # comprimento [m]
E1I1 = 1.0              # rigidez - segmento 1 [N.m²]
E2I2 = 4.0              # rigidez - segmento 2 [N.m²]
q0 = 1.0                # carga distribuída [N/m]
N = 200                 # número de divisões

# -----------------------------
# Malha
# -----------------------------
x = np.linspace(0, L, N + 1)
h = L / N

# -----------------------------
# ETAPA 1: Resolver M(x)
# d²M/dx² = -q(x)
# M(0)=M(L)=0
# -----------------------------
A_M = np.zeros((N - 1, N - 1))
b_M = np.full(N - 1, -q0 * h**2)

for i in range(N - 1):
    A_M[i, i] = -2
    if i > 0:
        A_M[i, i - 1] = 1
    if i < N - 2:
        A_M[i, i + 1] = 1

M_inner = solve(A_M, b_M)

M = np.zeros(N + 1)
M[1:-1] = M_inner

# -----------------------------
# ETAPA 2: Resolver w(x)
# d²w/dx² = -M(x)/EI(x)
# -----------------------------
EI = np.where(x < L / 2, E1I1, E2I2)

rhs_w = -M / EI

A_w = np.zeros((N - 1, N - 1))
b_w = np.zeros(N - 1)

for i in range(N - 1):
    A_w[i, i] = -2
    if i > 0:
        A_w[i, i - 1] = 1
    if i < N - 2:
        A_w[i, i + 1] = 1

    b_w[i] = rhs_w[i + 1] * h**2

w_inner = solve(A_w, b_w)

w = np.zeros(N + 1)
w[1:-1] = w_inner

# -----------------------------
# ETAPA 3: Esforço cortante
# -----------------------------
V = np.gradient(M, x)

# -----------------------------
# Resultados
# -----------------------------
i_max = np.argmax(w)

print(f"w_max = {w[i_max]:.6e} m em x = {x[i_max]:.4f} m")
print(f"M_max = {np.max(M):.6f} N.m")
print(f"V(0) = {V[0]:.6f} N")

E a implementação da PINN apresentada no artigo:

import torch
import torch.nn as nn

class PINN(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),

            nn.Linear(32, 32),
            nn.Tanh(),

            nn.Linear(32, 32),
            nn.Tanh(),

            nn.Linear(32, 32),
            nn.Tanh(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


def EI_func(x, E1I1=1.0, E2I2=4.0, L=1.0):
    return torch.where(
        x < L/2,
        torch.tensor(E1I1),
        torch.tensor(E2I2)
    )


def loss_fn(model, x_col, x_bc, x_int,
            lam_r=1,
            lam_bc=100,
            lam_int=10):

    x_col.requires_grad_(True)

    w = model(x_col)
    EI = EI_func(x_col)

    # Derivadas automáticas
    w_x = torch.autograd.grad(
        w.sum(),
        x_col,
        create_graph=True
    )[0]

    w_xx = torch.autograd.grad(
        w_x.sum(),
        x_col,
        create_graph=True
    )[0]

    M_pinn = -EI * w_xx

    M_xx = torch.autograd.grad(
        (EI * w_xx).sum(),
        x_col,
        create_graph=True
    )[0]

    M_xx2 = torch.autograd.grad(
        M_xx.sum(),
        x_col,
        create_graph=True
    )[0]

    # Resíduo da EDP
    L_r = ((M_xx2 + 1.0) ** 2).mean()

    # Condições de contorno
    w_bc = model(x_bc)
    L_bc = (w_bc ** 2).sum()

    # Interface
    eps = 1e-5

    x_L = torch.tensor([[0.5 - eps]], dtype=torch.float32)
    x_R = torch.tensor([[0.5 + eps]], dtype=torch.float32)

    L_int = ((model(x_L) - model(x_R)) ** 2).sum()

    return (
        lam_r * L_r +
        lam_bc * L_bc +
        lam_int * L_int
    )

_Espaço reservado para o projeto do Grupo 2._
