"""PINN com MÚLTIPLAS sementes — robustez estatística e custo (parecer 2.4).

Uma única execução (semente 42) não permite avaliar a variabilidade do treino.
Aqui treina-se a PINN decomposta com N sementes e reporta-se média ± desvio de
G_ef e dos erros vs MDF, além do TEMPO de parede e do nº de parâmetros treináveis.

Requer PyTorch; roda em CPU (~9 min/semente nesta máquina) ou GPU, se houver.

    python estudos/pinn_multisemente.py --seeds 0,1,2,3,4 --epochs 4000
    python estudos/pinn_multisemente.py --seeds 0,1,2,3,4,5,6,7,8,9  # 10 sementes

Salva resultados/pinn_seeds.csv e ...pinn_seeds_resumo.csv.
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ferramentas import console as ui
from metodos import pinn
from metodos.mdf import solve_fdm
from problema.geometria import OFFICIAL
from ferramentas.metricas import field_errors

TAB = os.path.join(ROOT, "resultados")


def _n_params(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="0,1,2,3,4",
                    help="lista de sementes separadas por vírgula")
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--device", default=None,
                    help="cuda|cpu (padrão: cuda se disponível)")
    args = ap.parse_args()

    import torch  # importado aqui para o script compilar sem torch instalado
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    ui.header("PINN decomposta  -  estudo multi-semente")
    ui.kv("device", device)
    ui.kv("epocas", args.epochs)
    ui.kv("sementes", ", ".join(map(str, seeds)))
    ui.step("Resolvendo referencia MDF e treinando (cada semente ~9 min em CPU)...")
    ref = solve_fdm(OFFICIAL, N=80)  # referência = MDF

    linhas = []
    n_par = None
    t_global = time.time()
    for s in seeds:
        t0 = time.time()
        net_m, net_i, hist, _ = pinn.train_decomposed(
            OFFICIAL, epochs=args.epochs, seed=s, device=device, verbose=False)
        sol = pinn.evaluate_decomposed(net_m, net_i, OFFICIAL, N=80, device=device)
        dt = time.time() - t0
        err = field_errors(sol, ref)
        if s == seeds[0]:  # guarda uma solucao para gerar as figuras da PINN
            np.savez(os.path.join(ROOT, "resultados", f"pinn_sol_seed{s}.npz"),
                     X=sol["X"], Y=sol["Y"], W=sol["W"],
                     txz=sol["txz"], tyz=sol["tyz"])
        if n_par is None:
            n_par = _n_params(net_m) + _n_params(net_i)
        linhas.append(dict(seed=s, Gef=sol["Gef"], err_w=err["w"],
                           err_txz=err["txz"], err_tyz=err["tyz"],
                           loss_final=hist["total"][-1], tempo_s=dt))
        gef_s = ui.value(f"{sol['Gef']:.5f}")
        print(f"  {ui.method('PINN')} seed {s:<2d}  Gef={gef_s}  "
              f"err_w={err['w']:.2e}  err_txz={err['txz']:.2e}  "
              f"{ui.paint(str(round(dt)) + 's', 'gray')}")

    def ms(chave):
        v = np.array([l[chave] for l in linhas], float)
        return v.mean(), v.std(), v.min(), v.max()

    ui.section("Resumo  (media +/- desvio  |  [min, max])")
    ui.kv("parametros treinaveis", f"{n_par}  (2 redes)")
    ui.kv("tempo total", f"{round(time.time() - t_global)} s")
    campos = ["Gef", "err_w", "err_txz", "err_tyz", "tempo_s"]
    for c in campos:
        m, d, lo, hi = ms(c)
        ui.kv(c, f"{m:.5f} +/- {d:.5f}   [{lo:.5f}, {hi:.5f}]")

    gef_m = ms("Gef")[0]
    if gef_m < 1.078:
        ui.warn(f"G_ef = {gef_m:.3f} esta ABAIXO do limite de Reuss (1.078) "
                "-> vies sistematico: a PINN quase ignora a inclusao")
    else:
        ui.ok(f"G_ef = {gef_m:.3f} dentro dos limites fisicos")

    os.makedirs(TAB, exist_ok=True)
    with open(os.path.join(TAB, "pinn_seeds.csv"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write("seed,Gef,err_w,err_txz,err_tyz,loss_final,tempo_s\n")
        for l in linhas:
            f.write(f"{l['seed']},{l['Gef']:.6f},{l['err_w']:.6e},"
                    f"{l['err_txz']:.6e},{l['err_tyz']:.6e},"
                    f"{l['loss_final']:.6e},{l['tempo_s']:.2f}\n")
    with open(os.path.join(TAB, "pinn_seeds_resumo.csv"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write("grandeza,media,desvio,min,max\n")
        for c in campos:
            m, d, lo, hi = ms(c)
            f.write(f"{c},{m:.6f},{d:.6f},{lo:.6f},{hi:.6f}\n")
        f.write(f"n_parametros,{n_par},0,{n_par},{n_par}\n")
    ui.ok("Tabelas: resultados/pinn_seeds.csv, pinn_seeds_resumo.csv")


if __name__ == "__main__":
    main()
