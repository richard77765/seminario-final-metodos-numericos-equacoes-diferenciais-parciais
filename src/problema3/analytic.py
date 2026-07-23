"""Solucao analitica do cisalhamento antiplano para INCLUSAO CIRCULAR.

Coordenadas polares com origem no CENTRO da inclusao (= centro do dominio):
    Dentro (r < a):  wi = [ 2 Gm / (Gi + Gm) ] * gamma * r * cos(theta)
    Fora   (r > a):  wm = gamma*r*cos(theta) + gamma*a^2*(Gm-Gi)/(Gm+Gi)*cos(theta)/r

Para que a BC remota gamma*x case com o dominio finito [0,L]^2 (onde impomos
w(0,y)=0 e w(L,y)=gamma*L), trabalhamos com o deslocamento relativo ao centro e
somamos o offset gamma*cx, de modo que a solucao analitica tambem satisfaca
w(0,y) ~ 0 e w(L,y) ~ gamma*L longe da inclusao.
"""

import numpy as np
from .geometry import Case


def w_analytic(x, y, case: Case):
    """Deslocamento w(x,y) da solucao circular analitica.

    O campo base e gamma*x (deformacao remota). A inclusao perturba esse campo.
    Retorna array com a mesma forma de x,y.
    """
    if case.shape != "circle":
        raise ValueError("w_analytic so vale para o caso de inclusao circular.")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cx, cy = case.center
    dx = x - cx
    dy = y - cy
    r = np.hypot(dx, dy)
    a, Gm, Gi, g = case.a, case.Gm, case.Gi, case.gamma

    # Evita divisao por zero no centro (r=0 cai no ramo interno, onde w=0 mesmo).
    r_safe = np.where(r == 0.0, 1.0, r)
    cos_t = dx / r_safe  # = cos(theta); em r=0 o valor e irrelevante (ramo interno)

    inside = r < a
    # Componente relativa ao centro
    w_in = (2.0 * Gm / (Gi + Gm)) * g * r * cos_t
    w_out = g * r * cos_t + g * a * a * (Gm - Gi) / (Gm + Gi) * cos_t / r_safe
    w_rel = np.where(inside, w_in, w_out)

    # Offset para alinhar com a BC do dominio finito: w = gamma*x no campo remoto.
    # w_rel ja contem gamma*dx; somamos gamma*cx para recuperar gamma*x.
    return w_rel + g * cx


def tau_analytic(x, y, case: Case):
    """Tensoes (tau_xz, tau_yz) = G * grad(w) da solucao circular analitica.

    Calculadas por diferenciacao analitica do campo acima. Retorna (txz, tyz).
    """
    if case.shape != "circle":
        raise ValueError("tau_analytic so vale para o caso de inclusao circular.")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cx, cy = case.center
    dx = x - cx
    dy = y - cy
    r2 = dx * dx + dy * dy
    a, Gm, Gi, g = case.a, case.Gm, case.Gi, case.gamma

    r2_safe = np.where(r2 == 0.0, 1.0, r2)
    inside = r2 < a * a

    # Dentro: w = C*gamma*dx com C = 2Gm/(Gi+Gm)  => grad = (C*gamma, 0), G = Gi
    C = 2.0 * Gm / (Gi + Gm)
    dwdx_in = C * g
    dwdy_in = np.zeros_like(dx)

    # Fora: w = gamma*dx + B*dx/r^2 com B = gamma*a^2*(Gm-Gi)/(Gm+Gi)
    #   d/dx [dx/r^2] = (r^2 - 2 dx^2)/r^4 = (dy^2 - dx^2)/r^4
    #   d/dy [dx/r^2] = -2 dx dy / r^4
    B = g * a * a * (Gm - Gi) / (Gm + Gi)
    dwdx_out = g + B * (dy * dy - dx * dx) / (r2_safe * r2_safe)
    dwdy_out = B * (-2.0 * dx * dy) / (r2_safe * r2_safe)

    dwdx = np.where(inside, dwdx_in, dwdx_out)
    dwdy = np.where(inside, dwdy_in, dwdy_out)
    G = np.where(inside, Gi, Gm)
    return G * dwdx, G * dwdy
