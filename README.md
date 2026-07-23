# Cisalhamento antiplano em compósito bifásico — MDF × MEF × PINN

Seminário Final · Problema 3 · Métodos Numéricos para EDO/EDP
Mestrado em Modelagem Computacional (UFF/EEIMVR)

Resolve-se **o mesmo problema físico por três métodos** e comparam-se os campos
locais e o módulo de cisalhamento efetivo:

| Método | Ideia |
|---|---|
| **MDF** — Diferenças Finitas | forma conservativa, média harmônica de `G` nas faces, Neumann por nós-fantasma |
| **MEF** — Elementos Finitos | forma fraca, elementos Q4 bilineares, Gauss 2×2, `G` por elemento |
| **PINN** — Physics-Informed NN | domínio decomposto (uma rede por fase) acoplado na interface, via PyTorch/autograd |

## Problema

Célula quadrada `Ω = [0,L]²` com inclusão quadrada central de módulo `Gᵢ ≠ Gₘ`.
Cisalhamento antiplano → incógnita escalar `w(x,y)`:

```
-∇·(G(x,y) ∇w) = 0        em Ω
w(0,y)=0,  w(L,y)=γL       (Dirichlet, laterais)
∂w/∂y = 0  em y=0,L        (Neumann homogêneo, topo/base)
[w]=0,  [G ∂ₙw]=0          (continuidade + equilíbrio de fluxo na interface)
```

Tensões `τ_xz = G ∂w/∂x`, `τ_yz = G ∂w/∂y`; módulo aparente `G_ef = ⟨τ_xz⟩/γ`.
Parâmetros oficiais: `L=1, a=0.15, Gₘ=1, Gᵢ=5, γ=0.01`.

## Estrutura

Os diretórios dizem o que contêm — organizados **por problema** e **por método**:

```
problema/       o PROBLEMA físico (comum aos três métodos)
   ├── geometria.py       domínio, inclusão, campo G(x,y), casos
   └── analitico.py       solução analítica de referência (inclusão circular)

metodos/        um arquivo por MÉTODO
   ├── mdf.py             Método de Diferenças Finitas
   ├── mef.py             Método de Elementos Finitos
   └── pinn.py            Physics-Informed Neural Network

ferramentas/    utilitários compartilhados
   ├── metricas.py        erros, módulo efetivo, métricas de tensão e de interface
   ├── graficos.py        geração das figuras
   └── console.py         saída colorida no terminal

estudos/        os experimentos (cada um roda os métodos e gera resultados)
   ├── validacao.py          validação vs solução analítica + convergência
   ├── convergencia.py       convergência (inclusão circular e quadrada)
   ├── comparacao.py         G_ef, erros e métricas de tensão (caso oficial)
   ├── parametrico.py        estudo de contraste Gᵢ/Gₘ
   ├── interface.py          erro de continuidade do fluxo na interface
   ├── pinn_multisemente.py  PINN com várias sementes (média ± desvio, custo)
   └── figuras.py            gera as figuras

resultados/     tabelas .csv geradas pelos estudos
notebook/       seminario_colab.py — notebook Colab original (proveniência)
```

---

# Como executar

O núcleo (MDF, MEF, métricas, figuras) precisa só de **numpy + scipy + matplotlib**.
A **PINN** precisa de **torch** — roda em CPU (~15 min) ou GPU. Os estudos rodam de
qualquer pasta.

## Opção 1 — Windows (PowerShell)

```powershell
cd C:\workspace\seminario-final-problema-3-metodo-numerico-edo-edp

# ambiente virtual (opcional, recomendado)
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# dependências do núcleo
py -m pip install numpy scipy matplotlib

# torch (só p/ a PINN) — no Windows use a build CPU (GPU aqui é via WSL):
py -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# rodar
py estudos\comparacao.py
py estudos\validacao.py
py estudos\pinn_multisemente.py --seeds 0,1,2,3,4 --epochs 4000
```

## Opção 2 — Linux / WSL

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip
cd /mnt/c/workspace/seminario-final-problema-3-metodo-numerico-edo-edp

python3 -m venv ~/venv-seminario
source ~/venv-seminario/bin/activate      # a cada novo terminal

pip install -r requirements.txt           # numpy, scipy, matplotlib, torch
python estudos/comparacao.py
python estudos/pinn_multisemente.py --seeds 0,1,2,3,4 --epochs 4000
```

## Opção 3 — GPU (WSL + CUDA) para a PINN

Pré-requisitos: driver NVIDIA no **Windows** (expõe a GPU ao WSL) e a build **CUDA**
do torch (padrão de `pip install torch` no Linux). Verifique:

```bash
nvidia-smi                                                   # a GPU aparece?
python -c "import torch; print('GPU:', torch.cuda.is_available())"
```

Se `True`, o `pinn_multisemente.py` **usa a GPU automaticamente** (detecta o device).
Para forçar: `--device cuda` (ou `cpu`).

## Opção 4 — Google Colab (notebook original)

`notebook/seminario_colab.py` é autocontido: abra no Colab e **Ambiente de execução →
Executar tudo**. As células `%%writefile` recriam os módulos sozinhas; o Colab já traz
numpy/scipy/matplotlib/torch. (Ative GPU em *Alterar tipo de ambiente* para acelerar a
PINN.)

## O que cada estudo faz

| Estudo | O que faz | Tempo (CPU) | Precisa torch? |
|---|---|---|---|
| `validacao.py` | Validação vs analítica circular + convergência (BC mista vs Dirichlet exato) | ~10 s | não |
| `convergencia.py` | Convergência circular (MDF) e quadrada (MDF+MEF, em `w` e `τ`) | ~20 s | não |
| `comparacao.py` | Caso oficial: `G_ef`, erros L² MEF×MDF, métricas de tensão robustas | ~3 s | não |
| `parametrico.py` | Estudo de contraste `Gᵢ/Gₘ ∈ {1,2,5,10,50,100}` | ~3 s | não |
| `interface.py` | Erro de continuidade do fluxo normal na interface | ~5 s | não |
| `pinn_multisemente.py` | PINN decomposta, **várias sementes** (média±desvio, custo) | ~15 min | **sim** |
| `figuras.py` | Gera as figuras em `figuras/` | ~5 s | não\* |

\* `figuras.py` inclui as figuras da PINN se existir `resultados/pinn_sol_seed*.npz`
(gerado por `pinn_multisemente.py`); sem ele, gera só MDF/MEF.

## Opções e variáveis de ambiente

- `pinn_multisemente.py --seeds 0,1,2,3,4 --epochs 4000 --device cpu|cuda`
- **Cores:** `NO_COLOR=1` desliga · `FORCE_COLOR=1` força (útil ao redirecionar)

**Saída colorida (Windows + Linux).** Os estudos imprimem de forma visual e
consistente, com cor fixa por método — **MDF azul, MEF verde, PINN amarelo**. As cores
ligam sozinhas no terminal e desligam quando a saída vai para arquivo (logs limpos).
Sem dependência extra: `ferramentas/console.py` habilita ANSI no Windows
(`SetConsoleMode`) e usa ANSI nativo no Linux.

> Importar de fora: `from metodos.mdf import solve_fdm`,
> `from problema.geometria import OFFICIAL`, `from ferramentas.metricas import field_errors`.

---

## Resultados principais (apurados do próprio código)

Caso oficial, malha `N=80` (`81×81` nós):

| Grandeza | Valor |
|---|---|
| `G_ef` (MDF) | **1,139769** |
| `G_ef` (MEF) | **1,139824** |
| `G_ef` (PINN, 5 sementes) | **0,982 ± 0,002** abaixo de Reuss |
| Limites Voigt/Reuss | [1,0776 ; 1,3600] → MDF/MEF consistentes; **PINN viola Reuss** |
| Erro L² MEF×MDF — `w` | 3,4×10⁻³ |
| Erro L² MEF×MDF — `τ_xz` | 1,1×10⁻¹ |
| Erro L² MEF×MDF — `τ_yz` | 2,8×10⁻¹ |

Validação (inclusão circular, `Gᵢ=10`), erro L² no interior `0.2 ≤ x,y ≤ 0.8`:

| N | BC mista (original) | Dirichlet exato (corrigido) |
|---|---|---|
| 20 | 1,84×10⁻² | 9,83×10⁻³ |
| 40 | 1,50×10⁻² | 3,44×10⁻³ |
| 80 | 1,61×10⁻² | 2,50×10⁻³ |
| 160 | 1,49×10⁻² (estagna) | 8,34×10⁻⁴ (converge, ordem ~1,5) |

Custo aproximado (CPU, `N=80`): MDF ~0,2 s · MEF ~3 s · PINN ~180 s/semente.
