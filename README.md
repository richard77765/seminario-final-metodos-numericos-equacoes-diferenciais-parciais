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

Organizada **por problema** e **por método**:

```
src/
├── problema3/    geometry.py, analytic.py        # o PROBLEMA: domínio, inclusão, G(x,y),
│                                                 #   casos e solução analítica de referência
├── metodos/      mdf.py, mef.py, pinn.py         # por MÉTODO (MDF, MEF, PINN)
└── comum/        metrics.py, viz.py, console.py  # utilitários compartilhados

scripts/          validate, convergence, comparison, parametric,
                  interface, pinn_seeds, make_figures   # estudos (rodam os métodos)
outputs/tables/   CSVs gerados pelos scripts
notebooks/        seminario_colab.py              # notebook Colab original (proveniência)
```

> Importar de fora: `from src.metodos.mdf import solve_fdm`,
> `from src.problema3.geometry import OFFICIAL`, `from src.comum.metrics import field_errors`.

---

# Como executar

Núcleo (MDF/MEF, métricas, figuras) precisa só de **numpy + scipy + matplotlib**.
A **PINN** (`pinn_seeds.py`) precisa de **torch** — roda em CPU (~15 min) ou GPU.
Todos os scripts usam caminhos relativos ao próprio arquivo, então **rodam de
qualquer diretório**.

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
py scripts\comparison.py
py scripts\validate.py
py scripts\pinn_seeds.py --seeds 0,1,2,3,4 --epochs 4000
```

*(Sem venv, é só trocar por `py -m pip install ...` global e `py scripts\...`.)*

## Opção 2 — Linux / WSL

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip
cd /mnt/c/workspace/seminario-final-problema-3-metodo-numerico-edo-edp

# venv no HOME do Linux (evita problemas de venv no /mnt/c)
python3 -m venv ~/venv-seminario
source ~/venv-seminario/bin/activate      # a cada novo terminal

pip install -r requirements.txt           # numpy, scipy, matplotlib, torch
python scripts/comparison.py
python scripts/pinn_seeds.py --seeds 0,1,2,3,4 --epochs 4000
```

## Opção 3 — GPU (WSL + CUDA) para a PINN

Pré-requisitos: driver NVIDIA no **Windows** (expõe a GPU ao WSL) e a build **CUDA**
do torch (é a padrão de `pip install torch` no Linux). Verifique:

```bash
nvidia-smi                                                   # a GPU aparece?
python -c "import torch; print('GPU:', torch.cuda.is_available())"
```

Se `True`, o `pinn_seeds.py` **usa a GPU automaticamente** (detecta o device) —
muito mais rápido que os ~15 min de CPU. Para forçar: `--device cuda` (ou `cpu`).

## Opção 4 — Google Colab (notebook original)

`notebooks/seminario_colab.py` é autocontido: abra no Colab e **Ambiente de execução
→ Executar tudo**. As células `%%writefile` recriam os módulos `src/` sozinhas; o
Colab já traz numpy/scipy/matplotlib/torch. (Ative GPU em *Alterar tipo de ambiente*
para acelerar a PINN.)

## O que cada script faz

| Script | O que faz | Tempo (CPU) | Precisa torch? |
|---|---|---|---|
| `validate.py` | Validação vs analítica circular + convergência (BC mista vs Dirichlet exato) | ~10 s | não |
| `convergence.py` | Convergência circular (MDF) e quadrada (MDF+MEF, em `w` e `τ`) | ~20 s | não |
| `comparison.py` | Caso oficial: `G_ef`, erros L² MEF×MDF, métricas de tensão robustas | ~3 s | não |
| `parametric.py` | Estudo de contraste `Gᵢ/Gₘ ∈ {1,2,5,10,50,100}` | ~3 s | não |
| `pinn_seeds.py` | PINN decomposta, **múltiplas sementes** (média±desvio, custo) | ~15 min | **sim** |
| `make_figures.py` | Gera as figuras do artigo em `outputs/figures/` | ~5 s | não\* |

\* `make_figures.py` inclui as figuras da PINN se existir `outputs/pinn_sol_seed*.npz`
(gerado por `pinn_seeds.py`); sem ele, gera só MDF/MEF.

## Opções e variáveis de ambiente

- `pinn_seeds.py --seeds 0,1,2,3,4 --epochs 4000 --device cpu|cuda`
- **Cores:** `NO_COLOR=1` desliga · `FORCE_COLOR=1` força (útil ao redirecionar)

**Saída colorida (Windows + Linux).** Os scripts imprimem de forma visual e
consistente, com cor fixa por método — **MDF azul, MEF verde, PINN amarelo**. As
cores ligam sozinhas no terminal e desligam quando a saída é redirecionada para
arquivo (logs limpos). Sem dependência extra: o `src/console.py` habilita ANSI no
Windows (`SetConsoleMode`) e usa ANSI nativo no Linux.

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