"""Todas as funcoes de visualizacao do projeto (secao 6 do CLAUDE.md).

Convencoes:
  - Toda figura de campo sobrepoe o contorno da inclusao (`overlay_inclusion`).
  - Saidas em figuras/ com dpi>=150.
  - Paleta consistente: 'viridis' para w, 'coolwarm'/'RdBu_r' para fluxos e diffs.
  - As malhas seguem a convencao 'ij' (X[i,j], Y[i,j]); para pcolormesh/imshow
    usamos .T para mapear (linhas=y, colunas=x) corretamente.
"""

import os

import matplotlib
matplotlib.use("Agg")  # backend nao-interativo p/ salvar em scripts
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registra projecao 3d)

from problema.geometria import Case, G_field, inclusion_outline, inclusion_mask

# raiz do repositório = três níveis acima de src/comum/viz.py
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(_REPO, "figuras")
W_CMAP = "viridis"
TAU_CMAP = "coolwarm"
DIFF_CMAP = "RdBu_r"
METHOD_COLORS = {"MDF": "#1f77b4", "MEF": "#2ca02c", "PINN": "#d62728",
                 "Analitico": "k", "Referencia": "k"}


def _ensure_dir():
    os.makedirs(FIGDIR, exist_ok=True)


def _save(fig, name):
    _ensure_dir()
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return path


def overlay_inclusion(ax, case: Case, color="k", lw=1.8, ls="-"):
    """Desenha o contorno da inclusao sobre um eixo de campo."""
    xs, ys = inclusion_outline(case)
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=5)


# --------------------------------------------------------------------------
# Geometria / setup (figuras 1-4)
# --------------------------------------------------------------------------

def fig_material_map(case: Case, N=200, save="01_mapa_material.png"):
    """Fig 1: heatmap discreto de G(x,y) (matriz vs inclusao)."""
    x = np.linspace(0, case.L, N)
    X, Y = np.meshgrid(x, x, indexing="ij")
    G = G_field(X, Y, case)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    pc = ax.pcolormesh(X, Y, G, cmap="YlOrRd", shading="auto")
    overlay_inclusion(ax, case)
    fig.colorbar(pc, ax=ax, label="G(x,y)")
    ax.set_title(f"Mapa material  (Gm={case.Gm}, Gi={case.Gi})")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
    return _save(fig, save), fig


def fig_fdm_grid(case: Case, N=20, save="02_malha_mdf.png"):
    """Fig 2: malha MDF com nos da inclusao destacados."""
    x = np.linspace(0, case.L, N + 1)
    X, Y = np.meshgrid(x, x, indexing="ij")
    inside = inclusion_mask(X, Y, case)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(X, Y, "o", color="0.7", ms=3)
    ax.plot(X.T, Y.T, color="0.85", lw=0.5)
    ax.plot(X, Y, color="0.85", lw=0.5)
    ax.plot(X[inside], Y[inside], "o", color="#d62728", ms=4, label="nos na inclusao")
    overlay_inclusion(ax, case)
    ax.set_title(f"Malha MDF (N={N})"); ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(loc="upper left", fontsize=8)
    return _save(fig, save), fig


def fig_fem_mesh(case: Case, N=20, save="03_malha_mef.png"):
    """Fig 3: malha MEF (elementos Q4) colorida por G do elemento."""
    x = np.linspace(0, case.L, N + 1)
    fig, ax = plt.subplots(figsize=(5, 5))
    for ei in range(N):
        for ej in range(N):
            cx = 0.5 * (x[ei] + x[ei + 1])
            cy = 0.5 * (x[ej] + x[ej + 1])
            Ge = case.Gi if inclusion_mask(cx, cy, case) else case.Gm
            color = "#fdbb84" if Ge == case.Gm else "#e34a33"
            ax.add_patch(plt.Rectangle((x[ei], x[ej]), x[ei + 1] - x[ei],
                                       x[ej + 1] - x[ej], facecolor=color,
                                       edgecolor="0.8", lw=0.4))
    overlay_inclusion(ax, case)
    ax.set_xlim(0, case.L); ax.set_ylim(0, case.L); ax.set_aspect("equal")
    ax.set_title(f"Malha MEF Q4 (N={N}) por G do elemento")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    return _save(fig, save), fig


def fig_pinn_points(pts: dict, case: Case, save="04_pontos_pinn.png"):
    """Fig 4: distribuicao dos pontos de colocacao da PINN.

    pts: dict com chaves opcionais 'interior_m','interior_i','boundary','interface'
    cada uma um array (n,2).
    """
    fig, ax = plt.subplots(figsize=(5.2, 5))
    styles = {
        "interior_m": ("matriz (EDP)", "#9ecae1", 4),
        "interior_i": ("inclusao (EDP)", "#fdae6b", 4),
        "boundary": ("contorno", "#31a354", 10),
        "interface": ("interface", "#d62728", 12),
    }
    for key, (lab, col, ms) in styles.items():
        P = pts.get(key)
        if P is not None and len(P):
            ax.scatter(P[:, 0], P[:, 1], s=ms, c=col, label=lab, alpha=0.7)
    overlay_inclusion(ax, case)
    ax.set_aspect("equal"); ax.set_xlim(0, case.L); ax.set_ylim(0, case.L)
    ax.set_title("Pontos de colocacao da PINN")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(fontsize=8, loc="upper right")
    return _save(fig, save), fig


# --------------------------------------------------------------------------
# Campos de solucao (figuras 5-9)
# --------------------------------------------------------------------------

def fig_field_heatmap(sol, field="W", case=None, save=None, title=None,
                      cmap=None, vmin=None, vmax=None):
    """Fig 5/8: heatmap de um campo (W, txz, tyz) com contorno da inclusao."""
    case = case or sol["case"]
    Z = sol[field]
    cmap = cmap or (W_CMAP if field == "W" else TAU_CMAP)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    pc = ax.pcolormesh(sol["X"], sol["Y"], Z, cmap=cmap, shading="auto",
                       vmin=vmin, vmax=vmax)
    overlay_inclusion(ax, case)
    fig.colorbar(pc, ax=ax, label=field)
    ax.set_title(title or field); ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    if save:
        return _save(fig, save), fig
    return None, fig


def fig_surface(sol, case=None, save="06_superficie_w.png", title="w(x,y)"):
    """Fig 6: superficie 3D de w(x,y)."""
    case = case or sol["case"]
    fig = plt.figure(figsize=(6, 4.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(sol["X"], sol["Y"], sol["W"], cmap=W_CMAP,
                    linewidth=0, antialiased=True)
    ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("w")
    return _save(fig, save), fig


def fig_contour(sol, case=None, save="07_curvas_nivel_w.png", title="Curvas de nivel de w"):
    """Fig 7: curvas de nivel de w."""
    case = case or sol["case"]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    cs = ax.contour(sol["X"], sol["Y"], sol["W"], levels=15, cmap=W_CMAP)
    ax.clabel(cs, inline=True, fontsize=7)
    overlay_inclusion(ax, case)
    ax.set_title(title); ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    return _save(fig, save), fig


def fig_field_contourf(sol, field="W", case=None, save=None, title=None,
                       cmap=None, levels=18, vmin=None, vmax=None):
    """Heatmap 'em curvas': bandas de cor (contourf) + isolinhas sobrepostas.

    Mesma informacao de fig_field_heatmap, mas o campo aparece como curvas de
    nivel preenchidas, destacando por onde o campo cresce/decai.
    """
    case = case or sol["case"]
    Z = sol[field]
    cmap = cmap or (W_CMAP if field == "W" else TAU_CMAP)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    cf = ax.contourf(sol["X"], sol["Y"], Z, levels=levels, cmap=cmap,
                     vmin=vmin, vmax=vmax)
    ax.contour(sol["X"], sol["Y"], Z, levels=levels, colors="k",
               linewidths=0.4, alpha=0.45)
    overlay_inclusion(ax, case)
    fig.colorbar(cf, ax=ax, label=field)
    ax.set_title(title or field); ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    if save:
        return _save(fig, save), fig
    return None, fig


def fig_flux_field(sol, case=None, save="09_fluxo.png", step=6, mode="quiver"):
    """Fig 9: campo vetorial do fluxo (txz, tyz) -- quiver ou streamplot."""
    case = case or sol["case"]
    X, Y, U, V = sol["X"], sol["Y"], sol["txz"], sol["tyz"]
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    mag = np.hypot(U, V)
    if mode == "stream":
        # streamplot exige grade 1D crescente e ordem (y, x)
        ax.streamplot(X[:, 0], Y[0, :], U.T, V.T, color=mag.T, cmap=TAU_CMAP, density=1.2)
    else:
        s = slice(None, None, step)
        ax.quiver(X[s, s], Y[s, s], U[s, s], V[s, s], mag[s, s], cmap=TAU_CMAP)
    overlay_inclusion(ax, case)
    ax.set_title("Fluxo de cisalhamento (txz, tyz)"); ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_xlim(0, case.L); ax.set_ylim(0, case.L)
    return _save(fig, save), fig


# --------------------------------------------------------------------------
# Validacao circular (figuras 10-11)
# --------------------------------------------------------------------------

def fig_centerline_validation(sol, w_ref_func, case=None,
                              save="10_perfil_central.png"):
    """Fig 10: perfil de w na linha horizontal central: numerico x analitico."""
    case = case or sol["case"]
    n = sol["W"].shape[1]
    jmid = n // 2
    x = sol["X"][:, jmid]
    y = sol["Y"][:, jmid]
    w_num = sol["W"][:, jmid]
    w_ref = w_ref_func(x, y, case)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, w_ref, "k-", lw=2, label="analitico")
    ax.plot(x, w_num, "o", color="#1f77b4", ms=3, label="numerico")
    ax.axvspan(case.center[0] - case.a, case.center[0] + case.a,
               color="0.85", alpha=0.5, label="inclusao")
    ax.set_xlabel("x"); ax.set_ylabel("w (linha central)")
    ax.set_title("Validacao: perfil central"); ax.legend(fontsize=8)
    return _save(fig, save), fig


def fig_error_map(sol, w_ref_func, case=None, save="11_mapa_erro.png"):
    """Fig 11: mapa de erro absoluto |w_num - w_analitico|."""
    case = case or sol["case"]
    Wref = w_ref_func(sol["X"], sol["Y"], case)
    err = np.abs(sol["W"] - Wref)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    pc = ax.pcolormesh(sol["X"], sol["Y"], err, cmap="magma", shading="auto")
    overlay_inclusion(ax, case, color="w")
    fig.colorbar(pc, ax=ax, label="|w_num - w_ref|")
    ax.set_title("Erro absoluto (vs analitico)"); ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    return _save(fig, save), fig


# --------------------------------------------------------------------------
# PINN treino (figuras 12-13)
# --------------------------------------------------------------------------

def fig_loss_history(history: dict, save="12_loss_pinn.png"):
    """Fig 12: evolucao da loss (total e componentes) em escala log."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for key, vals in history.items():
        ax.semilogy(vals, label=key)
    ax.set_xlabel("epoca"); ax.set_ylabel("loss (log)")
    ax.set_title("Evolucao da loss da PINN"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    return _save(fig, save), fig


def fig_scatter_r2(w_pred, w_ref, save="13_scatter_r2.png", label="PINN"):
    """Fig 13: scatter w_pred x w_ref com R^2 anotado."""
    from .metricas import r2_score
    w_pred = np.asarray(w_pred).ravel()
    w_ref = np.asarray(w_ref).ravel()
    r2 = r2_score(w_pred, w_ref)
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.scatter(w_ref, w_pred, s=5, alpha=0.4, color="#d62728")
    lo, hi = min(w_ref.min(), w_pred.min()), max(w_ref.max(), w_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("w referencia"); ax.set_ylabel(f"w {label}")
    ax.set_title(f"{label}  (R2 = {r2:.4f})"); ax.set_aspect("equal")
    return _save(fig, save), fig


# --------------------------------------------------------------------------
# Comparacao final (figuras 14-19)
# --------------------------------------------------------------------------

def fig_panel_w(sols: dict, case, save="14_painel_w.png"):
    """Fig 14: painel lado a lado de w por metodo, MESMA escala de cor."""
    names = list(sols.keys())
    vmin = min(s["W"].min() for s in sols.values())
    vmax = max(s["W"].max() for s in sols.values())
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4.2))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        s = sols[name]
        pc = ax.pcolormesh(s["X"], s["Y"], s["W"], cmap=W_CMAP, shading="auto",
                           vmin=vmin, vmax=vmax)
        overlay_inclusion(ax, case)
        ax.set_title(f"w - {name}"); ax.set_aspect("equal")
        ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(pc, ax=axes, label="w", fraction=0.046, pad=0.04)
    return _save(fig, save), fig


def fig_panel_contourf(sols: dict, case, field="W", save="14b_painel_curvas.png",
                       levels=18, cmap=None):
    """Painel comparativo dos tres metodos como heatmap 'em curvas' (contourf).

    Todos os subplots compartilham os MESMOS niveis (mesma escala de cor e as
    mesmas isolinhas), para que a comparacao MDF x MEF x PINN seja justa.
    """
    names = list(sols.keys())
    cmap = cmap or (W_CMAP if field == "W" else TAU_CMAP)
    vmin = min(s[field].min() for s in sols.values())
    vmax = max(s[field].max() for s in sols.values())
    lv = np.linspace(vmin, vmax, levels + 1)
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4.2))
    if len(names) == 1:
        axes = [axes]
    cf = None
    for ax, name in zip(axes, names):
        s = sols[name]
        cf = ax.contourf(s["X"], s["Y"], s[field], levels=lv, cmap=cmap,
                         extend="both")
        ax.contour(s["X"], s["Y"], s[field], levels=lv, colors="k",
                   linewidths=0.35, alpha=0.4)
        overlay_inclusion(ax, case)
        ax.set_title(f"{field} - {name}"); ax.set_aspect("equal")
        ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(cf, ax=axes, label=field, fraction=0.046, pad=0.04)
    return _save(fig, save), fig


def fig_panel_diff(sols: dict, case, save="15_painel_diferencas.png"):
    """Fig 15: painel de diferencas entre pares de metodos."""
    names = list(sols.keys())
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.6 * len(pairs), 4.2))
    if len(pairs) == 1:
        axes = [axes]
    for ax, (a, b) in zip(axes, pairs):
        d = sols[a]["W"] - sols[b]["W"]
        lim = np.abs(d).max() or 1e-12
        pc = ax.pcolormesh(sols[a]["X"], sols[a]["Y"], d, cmap=DIFF_CMAP,
                           shading="auto", vmin=-lim, vmax=lim)
        overlay_inclusion(ax, case)
        ax.set_title(f"{a} - {b}"); ax.set_aspect("equal")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, save), fig


def fig_centerline_methods(sols: dict, case, field="W", save="16_perfis_central.png"):
    """Fig 16: perfis na linha central, varios metodos sobrepostos."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, s in sols.items():
        n = s[field].shape[1]
        jmid = n // 2
        ax.plot(s["X"][:, jmid], s[field][:, jmid], label=name,
                color=METHOD_COLORS.get(name), lw=1.8)
    ax.axvspan(case.center[0] - case.a, case.center[0] + case.a,
               color="0.85", alpha=0.5)
    ax.set_xlabel("x"); ax.set_ylabel(f"{field} (linha central)")
    ax.set_title(f"Perfil central de {field}"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    return _save(fig, save), fig


def fig_convergence(Ns, errs, save="17_convergencia.png"):
    """Fig 17: estudo de convergencia e_rel vs N (log-log)."""
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.loglog(Ns, errs, "o-", color="#1f77b4")
    # linha de referencia de ordem 2
    Ns = np.asarray(Ns, float)
    ref = errs[0] * (Ns[0] / Ns) ** 2
    ax.loglog(Ns, ref, "k--", lw=1, label="ordem 2 (ref.)")
    ax.set_xlabel("N"); ax.set_ylabel("e_rel (w)")
    ax.set_title("Convergencia MDF"); ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    return _save(fig, save), fig


def fig_contrast_study(ratios, gefs, tau_peaks, save="18_contraste.png"):
    """Fig 18: Gef e pico de tau na interface vs razao Gi/Gm."""
    fig, ax1 = plt.subplots(figsize=(6.2, 4.2))
    ax1.semilogx(ratios, gefs, "o-", color="#1f77b4", label="Gef")
    ax1.set_xlabel("Gi/Gm"); ax1.set_ylabel("Gef", color="#1f77b4")
    ax2 = ax1.twinx()
    ax2.semilogx(ratios, tau_peaks, "s--", color="#d62728", label="pico |tau|")
    ax2.set_ylabel("pico |tau|", color="#d62728")
    ax1.set_title("Estudo de contraste Gi/Gm"); ax1.grid(alpha=0.3, which="both")
    return _save(fig, save), fig


def fig_error_bars(err_table: dict, save="19_barras_erro.png"):
    """Fig 19: barras dos erros relativos finais (w, txz, tyz) por metodo.

    err_table: {metodo: {'w':..,'txz':..,'tyz':..}}
    """
    methods = list(err_table.keys())
    fields = ["w", "txz", "tyz"]
    x = np.arange(len(fields))
    width = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for k, m in enumerate(methods):
        vals = [err_table[m][f] for f in fields]
        ax.bar(x + k * width, vals, width, label=m, color=METHOD_COLORS.get(m))
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(fields)
    ax.set_ylabel("erro relativo L2"); ax.set_yscale("log")
    ax.set_title("Erros relativos por metodo"); ax.legend(fontsize=8)
    return _save(fig, save), fig


def all_fields_for_method(sol, name, case, prefix):
    """Conveniencia: gera as figuras de campo (5-9) para um metodo, prefixadas."""
    paths = []
    paths.append(fig_field_heatmap(sol, "W", case, f"{prefix}_05_w_heatmap.png",
                                   title=f"w(x,y) - {name}")[0])
    paths.append(fig_surface(sol, case, f"{prefix}_06_w_superficie.png",
                             title=f"w(x,y) - {name}")[0])
    paths.append(fig_contour(sol, case, f"{prefix}_07_w_contorno.png",
                             title=f"Curvas de nivel - {name}")[0])
    paths.append(fig_field_heatmap(sol, "txz", case, f"{prefix}_08a_txz.png",
                                   title=f"txz - {name}")[0])
    paths.append(fig_field_heatmap(sol, "tyz", case, f"{prefix}_08b_tyz.png",
                                   title=f"tyz - {name}")[0])
    paths.append(fig_flux_field(sol, case, f"{prefix}_09_fluxo.png")[0])
    plt.close("all")
    return paths
