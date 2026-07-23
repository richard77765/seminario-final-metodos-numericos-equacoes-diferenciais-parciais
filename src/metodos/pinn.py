"""Physics-Informed Neural Network (PINN) para o cisalhamento antiplano.

Estrategia PRINCIPAL: dominio decomposto (secao 5.3 do CLAUDE.md).
  Como G e constante em cada fase, a EDP -div(G grad w)=0 vira a equacao de
  Laplace  lap(w)=0  dentro de cada subdominio; toda a fisica nao-trivial fica
  na interface. Usamos duas redes:
      net_m -> w na matriz (Omega_m)
      net_i -> w na inclusao (Omega_i)
  Loss = L_EDP + L_CC + L_I
      L_EDP : residuo de Laplace em pontos de colocacao de cada subdominio.
      L_CC  : Dirichlet (x=0, x=L) + Neumann homogeneo (y=0, y=L), so na matriz.
      L_I   : na interface,  wm = wi  e  Gm dwm/dn = Gi dwi/dn.

Estrategia SECUNDARIA (para o relatorio): PINN unica sobre G(x,y) descontinuo,
para evidenciar a dificuldade perto da interface. Ver `train_single_pinn`.

Framework: PyTorch; derivadas via torch.autograd.grad.
"""

import numpy as np
import torch
import torch.nn as nn

from ..problema3.geometry import Case, inclusion_mask


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


class MLP(nn.Module):
    """MLP simples com ativacao tanh. Normaliza a entrada de [0,L]^2 p/ [-1,1]^2."""

    def __init__(self, L=1.0, hidden=48, layers=5):
        super().__init__()
        self.L = L
        acts = []
        dims = [2] + [hidden] * layers + [1]
        mods = []
        for k in range(len(dims) - 1):
            mods.append(nn.Linear(dims[k], dims[k + 1]))
            if k < len(dims) - 2:
                mods.append(nn.Tanh())
        self.net = nn.Sequential(*mods)

    def forward(self, xy):
        z = 2.0 * xy / self.L - 1.0  # normaliza p/ [-1,1]
        return self.net(z)


def _grad(out, inp):
    """d(out)/d(inp), mantendo o grafo para derivadas de ordem superior."""
    return torch.autograd.grad(out, inp, grad_outputs=torch.ones_like(out),
                               create_graph=True)[0]


def _laplacian(w, xy):
    """Laplaciano de w em xy=(x,y) via autograd (segunda derivada)."""
    g = _grad(w, xy)
    wx, wy = g[:, 0:1], g[:, 1:2]
    wxx = _grad(wx, xy)[:, 0:1]
    wyy = _grad(wy, xy)[:, 1:2]
    return wxx + wyy


# --------------------------------------------------------------------------
# Amostragem de pontos de colocacao
# --------------------------------------------------------------------------

def sample_points(case: Case, n_dom=4000, n_bc=400, n_iface=600, seed=0):
    """Gera pontos de colocacao para o caso (inclusao quadrada).

    Retorna dict de arrays numpy (n,2):
      interior_m, interior_i, bc_left, bc_right, bc_top, bc_bot, interface
    """
    rng = np.random.default_rng(seed)
    L, a = case.L, case.a
    cx, cy = case.center

    # Interior: amostra uniforme e separa por fase
    P = rng.uniform(0, L, size=(int(n_dom * 1.6), 2))
    inside = inclusion_mask(P[:, 0], P[:, 1], case)
    interior_i = P[inside][:n_dom]
    interior_m = P[~inside][:n_dom]

    # Contornos externos
    s = rng.uniform(0, L, size=(n_bc, 1))
    bc_left = np.column_stack([np.zeros_like(s), s])
    bc_right = np.column_stack([np.full_like(s, L), s])
    bc_bot = np.column_stack([s, np.zeros_like(s)])
    bc_top = np.column_stack([s, np.full_like(s, L)])

    # Interface: depende da forma da inclusao.
    if case.shape == "square":
        # quadrado |x-cx|=a ou |y-cy|=a; amostra os 4 lados.
        t = rng.uniform(-a, a, size=(n_iface // 4, 1))
        left = np.column_stack([np.full_like(t, cx - a), cy + t])
        right = np.column_stack([np.full_like(t, cx + a), cy + t])
        bot = np.column_stack([cx + t, np.full_like(t, cy - a)])
        top = np.column_stack([cx + t, np.full_like(t, cy + a)])
        interface = np.vstack([left, right, bot, top])
    else:
        # circulo r=a centrado em (cx,cy)
        th = rng.uniform(0, 2 * np.pi, size=(n_iface, 1))
        interface = np.column_stack([cx + a * np.cos(th), cy + a * np.sin(th)])

    return dict(interior_m=interior_m, interior_i=interior_i,
                bc_left=bc_left, bc_right=bc_right, bc_top=bc_top, bc_bot=bc_bot,
                interface=interface)


def _interface_normals(case: Case, pts):
    """Normais externas (apontando da inclusao p/ a matriz) nos pontos da interface."""
    cx, cy = case.center
    a = case.a
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    n = np.zeros_like(pts)
    if case.shape == "square":
        # ponto pertence a face vertical se |dx| ~ a, senao horizontal
        vert = np.abs(np.abs(dx) - a) <= np.abs(np.abs(dy) - a)
        n[vert, 0] = np.sign(dx[vert])
        n[~vert, 1] = np.sign(dy[~vert])
    else:
        # circulo: normal radial unitaria
        r = np.hypot(dx, dy)
        r = np.where(r == 0.0, 1.0, r)
        n[:, 0] = dx / r
        n[:, 1] = dy / r
    return n


# --------------------------------------------------------------------------
# Treino: PINN com dominio decomposto
# --------------------------------------------------------------------------

def train_decomposed(case: Case, epochs=4000, lr=1e-3, hidden=48, layers=5,
                     weights=None, seed=42, device="cpu", verbose=True,
                     points=None):
    """Treina as duas redes acopladas. Retorna (net_m, net_i, history, points)."""
    set_seed(seed)
    weights = weights or dict(edp=1.0, bc=10.0, iface=10.0)
    pts = points or sample_points(case, seed=seed)

    def T(arr, grad=False):
        t = torch.tensor(arr, dtype=torch.float32, device=device)
        t.requires_grad_(grad)
        return t

    Xm = T(pts["interior_m"], grad=True)
    Xi = T(pts["interior_i"], grad=True)
    Xl = T(pts["bc_left"]); Xr = T(pts["bc_right"])
    Xt = T(pts["bc_top"], grad=True); Xb = T(pts["bc_bot"], grad=True)
    Xf = T(pts["interface"], grad=True)
    nrm = T(_interface_normals(case, pts["interface"]))

    net_m = MLP(case.L, hidden, layers).to(device)
    net_i = MLP(case.L, hidden, layers).to(device)
    opt = torch.optim.Adam(list(net_m.parameters()) + list(net_i.parameters()), lr=lr)

    history = dict(total=[], edp=[], bc=[], iface=[])
    mse = nn.MSELoss()

    for ep in range(epochs):
        opt.zero_grad()

        # L_EDP: Laplace em cada subdominio
        lap_m = _laplacian(net_m(Xm), Xm)
        lap_i = _laplacian(net_i(Xi), Xi)
        L_edp = mse(lap_m, torch.zeros_like(lap_m)) + mse(lap_i, torch.zeros_like(lap_i))

        # L_CC: Dirichlet laterais + Neumann topo/base (so a matriz toca o externo)
        wl = net_m(Xl); wr = net_m(Xr)
        L_dir = mse(wl, torch.zeros_like(wl)) + \
            mse(wr, torch.full_like(wr, case.gamma * case.L))
        wt = net_m(Xt); wb = net_m(Xb)
        dwt = _grad(wt, Xt)[:, 1:2]   # dw/dy no topo
        dwb = _grad(wb, Xb)[:, 1:2]   # dw/dy na base
        L_neu = mse(dwt, torch.zeros_like(dwt)) + mse(dwb, torch.zeros_like(dwb))
        L_cc = L_dir + L_neu

        # L_I: continuidade + equilibrio de fluxo na interface
        wm_f = net_m(Xf); wi_f = net_i(Xf)
        L_cont = mse(wm_f, wi_f)
        gm = _grad(wm_f, Xf); gi = _grad(wi_f, Xf)
        dwm_n = (gm * nrm).sum(dim=1, keepdim=True)   # dwm/dn
        dwi_n = (gi * nrm).sum(dim=1, keepdim=True)   # dwi/dn
        L_flux = mse(case.Gm * dwm_n, case.Gi * dwi_n)
        L_iface = L_cont + L_flux

        loss = weights["edp"] * L_edp + weights["bc"] * L_cc + weights["iface"] * L_iface
        loss.backward()
        opt.step()

        history["total"].append(loss.item())
        history["edp"].append(L_edp.item())
        history["bc"].append(L_cc.item())
        history["iface"].append(L_iface.item())
        if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            print(f"  ep {ep:5d}  loss={loss.item():.3e}  edp={L_edp.item():.2e} "
                  f"bc={L_cc.item():.2e} iface={L_iface.item():.2e}")

    return net_m, net_i, history, pts


def evaluate_decomposed(net_m, net_i, case: Case, N=80, device="cpu"):
    """Avalia as redes numa malha (N+1)x(N+1) e retorna dict no formato dos solvers."""
    from ..problema3.geometry import grid, G_field
    X, Y, x1d, h = grid(N, case)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    inside = inclusion_mask(pts[:, 0], pts[:, 1], case)

    net_m.eval(); net_i.eval()
    xy = torch.tensor(pts, dtype=torch.float32, device=device, requires_grad=True)
    # Deslocamento (costurado por fase)
    wm = net_m(xy); wi = net_i(xy)
    gm = _grad(wm, xy).detach().cpu().numpy()
    gi = _grad(wi, xy).detach().cpu().numpy()
    wm_np = wm.detach().cpu().numpy().ravel()
    wi_np = wi.detach().cpu().numpy().ravel()
    W = np.where(inside, wi_np, wm_np).reshape(X.shape)

    # Tensoes via autograd EM CADA SUBDOMINIO (gradiente exato da rede, sem o
    # artefato de diferenciar o campo costurado atraves da interface).
    G = np.where(inside, case.Gi, case.Gm)
    dwdx = np.where(inside, gi[:, 0], gm[:, 0])
    dwdy = np.where(inside, gi[:, 1], gm[:, 1])
    txz = (G * dwdx).reshape(X.shape)
    tyz = (G * dwdy).reshape(X.shape)
    Gef = float(np.mean(txz) / case.gamma)
    return dict(X=X, Y=Y, W=W, txz=txz, tyz=tyz, Gef=Gef, h=h, N=N, case=case)


# --------------------------------------------------------------------------
# Estrategia secundaria: PINN unica sobre G descontinuo
# --------------------------------------------------------------------------

def train_single_pinn(case: Case, epochs=4000, lr=1e-3, hidden=48, layers=5,
                      seed=42, device="cpu", verbose=True):
    """PINN unica resolvendo a forma fraca/forte com G descontinuo.

    Usa a forma  div(G grad w) = 0  diretamente. Como G e descontinuo, a rede
    tem dificuldade na interface -- exatamente o ponto que queremos evidenciar.
    Retorna (net, history).
    """
    set_seed(seed)
    pts = sample_points(case, seed=seed)
    rng = np.random.default_rng(seed)
    # mistura interior das duas fases num unico conjunto
    Xall = np.vstack([pts["interior_m"], pts["interior_i"]])
    rng.shuffle(Xall)

    def T(arr, grad=False):
        t = torch.tensor(arr, dtype=torch.float32, device=device)
        t.requires_grad_(grad)
        return t

    Xd = T(Xall, grad=True)
    Gd = torch.tensor(
        np.where(inclusion_mask(Xall[:, 0], Xall[:, 1], case), case.Gi, case.Gm),
        dtype=torch.float32, device=device).reshape(-1, 1)
    Xl = T(pts["bc_left"]); Xr = T(pts["bc_right"])
    Xt = T(pts["bc_top"], grad=True); Xb = T(pts["bc_bot"], grad=True)

    net = MLP(case.L, hidden, layers).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    mse = nn.MSELoss()
    history = dict(total=[], edp=[], bc=[])

    for ep in range(epochs):
        opt.zero_grad()
        w = net(Xd)
        g = _grad(w, Xd)
        # residuo de div(G grad w); G localmente constante -> G * lap(w)
        lap = _laplacian(w, Xd)
        res = Gd * lap
        L_edp = mse(res, torch.zeros_like(res))

        wl = net(Xl); wr = net(Xr)
        L_dir = mse(wl, torch.zeros_like(wl)) + \
            mse(wr, torch.full_like(wr, case.gamma * case.L))
        wt = net(Xt); wb = net(Xb)
        L_neu = mse(_grad(wt, Xt)[:, 1:2], torch.zeros((Xt.shape[0], 1), device=device)) + \
            mse(_grad(wb, Xb)[:, 1:2], torch.zeros((Xb.shape[0], 1), device=device))
        L_bc = L_dir + L_neu

        loss = L_edp + 10.0 * L_bc
        loss.backward(); opt.step()
        history["total"].append(loss.item())
        history["edp"].append(L_edp.item())
        history["bc"].append(L_bc.item())
        if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            print(f"  [single] ep {ep:5d}  loss={loss.item():.3e}")
    return net, history


def evaluate_single(net, case: Case, N=80, device="cpu"):
    from ..problema3.geometry import grid, G_field
    X, Y, x1d, h = grid(N, case)
    xy = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]),
                      dtype=torch.float32, device=device)
    net.eval()
    with torch.no_grad():
        W = net(xy).cpu().numpy().reshape(X.shape)
    Gnod = G_field(X, Y, case)
    txz = Gnod * np.gradient(W, h, axis=0)
    tyz = Gnod * np.gradient(W, h, axis=1)
    Gef = float(np.mean(txz) / case.gamma)
    return dict(X=X, Y=Y, W=W, txz=txz, tyz=tyz, Gef=Gef, h=h, N=N, case=case)
