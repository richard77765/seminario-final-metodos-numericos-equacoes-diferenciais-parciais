"""Metricas de comparacao entre solucoes (secao 3 do CLAUDE.md)."""

import numpy as np


def rel_l2(num, ref, mask=None):
    """Erro relativo em norma L2:  ||num - ref||_2 / ||ref||_2.

    mask (booleano, opcional) restringe o calculo a uma regiao -- util para
    excluir pontos colados na borda externa na validacao circular.
    """
    num = np.asarray(num, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if mask is not None:
        num = num[mask]
        ref = ref[mask]
    denom = np.linalg.norm(ref.ravel())
    if denom == 0.0:
        return float(np.linalg.norm((num - ref).ravel()))
    return float(np.linalg.norm((num - ref).ravel()) / denom)


def r2_score(pred, ref):
    """Coeficiente de determinacao R^2 entre pred e ref (para scatter da PINN)."""
    pred = np.asarray(pred, dtype=float).ravel()
    ref = np.asarray(ref, dtype=float).ravel()
    ss_res = np.sum((ref - pred) ** 2)
    ss_tot = np.sum((ref - ref.mean()) ** 2)
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def effective_modulus(txz, gamma):
    """Modulo de cisalhamento efetivo Gef = <tau_xz> / gamma."""
    return float(np.mean(txz) / gamma)


def field_errors(sol, ref, mask=None):
    """Dicionario de erros relativos para w, txz, tyz entre duas solucoes (dicts
    com chaves 'W','txz','tyz')."""
    return {
        "w": rel_l2(sol["W"], ref["W"], mask),
        "txz": rel_l2(sol["txz"], ref["txz"], mask),
        "tyz": rel_l2(sol["tyz"], ref["tyz"], mask),
    }
