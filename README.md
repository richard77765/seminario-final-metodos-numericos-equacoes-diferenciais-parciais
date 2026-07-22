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

```
src/          geometry, analytic, fdm, fem, pinn, metrics, viz   (núcleo)
scripts/      validate.py  (+ convergence/parametric/pinn_seeds a incluir)
outputs/      figures/  tables/                                  (gerados)
paper/        LaTeX + .bib
notebooks/    seminario_colab.py     (notebook Colab original, para proveniência)
docs/         correcoes.md  (rastro das correções vs. parecer)
```

## Como reproduzir

```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt                     # torch é opcional (só p/ PINN)

python scripts/validate.py        # validação + convergência (gera outputs/tables/…)
```

Os solvers MDF/MEF usam apenas `numpy`/`scipy`. A PINN requer `torch` e **roda em
CPU** (~9 min por semente nesta máquina); GPU é opcional (apenas acelera).
As figuras do artigo saem de `python scripts/make_figures.py`.

**Saída colorida (Windows + Linux).** Os scripts imprimem de forma visual e
consistente, com cor fixa por método — **MDF azul, MEF verde, PINN amarelo**. As
cores ligam sozinhas no terminal e desligam quando a saída é redirecionada para
arquivo. Force com `FORCE_COLOR=1` ou desligue com `NO_COLOR=1`.

## Resultados principais (apurados do próprio código)

Caso oficial, malha `N=80` (`81×81` nós):

| Grandeza | Valor |
|---|---|
| `G_ef` (MDF) | **1,139769** |
| `G_ef` (MEF) | **1,139824** |
| `G_ef` (PINN, 5 sementes) | **0,982 ± 0,002** ⚠️ abaixo de Reuss |
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

