# Rastro de correções — revisão pós-parecer

**Alvo editorial:** REMAT (seção Matemática), com documentação completa das
alterações e conformidade de template.
**Base:** parecer técnico-editorial de 12/07/2026 (revisão maior).
**Última atualização:** 2026-07-21.

Legenda: ✅ feito · 🔧 em andamento · ⏳ a fazer · 🧭 decidido (aguarda execução)
· 📝 documentado em [`paper/CORRECOES_LATEX.md`](../paper/CORRECOES_LATEX.md) (aplicar no Overleaf)

---

## 1. Situação dos pontos do parecer

| # | Ponto do parecer | Gravidade | Status |
|---|---|---|---|
| 2.1 | Validação circular não valida o problema finito (viola BCs) | Crítico | ✅ código corrigido; ⏳ texto §7.1 |
| 2.2 | `G_ef` sem rigor / sanidade Voigt–Reuss | Crítico | ✅ verificado; ⏳ definição operacional no texto |
| 2.3 | `max|τ|` nas quinas não é métrica robusta | Alto | ⏳ trocar por L²/percentil/erro de fluxo |
| 2.4 | PINN com 1 semente; sem custo computacional | Alto | ✅ rodado (CPU): G_ef 0,982±0,002 (abaixo de Reuss); ~15 min |
| 2.5 | "MEF superior" não demonstrado (MDF é a referência) | Crítico | ✅ evidência levantada; ⏳ reescrever afirmações |
| 3.x | Formulação, tabelas, figuras, editorial | Médio/Alto | ⏳ ver seções abaixo |

---

## 2. Correções já aplicadas

### 2.0 Engenharia reversa do Colab → repositório
Os 7 módulos (`geometry, analytic, fdm, fem, metrics, viz, pinn`) foram extraídos
fielmente dos blocos `%%writefile` do notebook original para `src/`, preservando
docstrings e indentação. Adicionados `README.md`, `requirements.txt`, `.gitignore`
e este documento. Atende aos itens de **reprodutibilidade** do parecer (3.8).

### 2.1 Validação consistente — Dirichlet exato (parecer 2.1) ✅ (código)
**Problema.** A solução analítica de inclusão circular é de **meio infinito** e
não satisfaz `w(0,y)=0` nem `w(L,y)=γL`. Verificação numérica: a analítica viola
as BCs do quadrado em **3,7% da faixa `γL`** em ambas as laterais. Logo a
"validação" original comparava **dois problemas diferentes**.

**Correção.** `src/fdm.py::solve_fdm` recebeu o parâmetro opcional
`bc_func(x,y,case)`. Quando fornecido, impõe **Dirichlet exato** dessa função em
**todo o contorno** — o solver passa a resolver o *mesmo* BVP da referência. O
caso oficial (BC mista) permanece inalterado (retrocompatível).

**Evidência** (`scripts/validate.py`, erro L² interior, inclusão circular `Gᵢ=10`):

| N | BC mista (antes) | Dirichlet exato (depois) |
|---|---|---|
| 20 | 1,84×10⁻² | 9,83×10⁻³ |
| 40 | 1,50×10⁻² | 3,44×10⁻³ |
| 80 | 1,61×10⁻² | 2,50×10⁻³ |
| 160 | 1,49×10⁻² (**estagna**, ordem≈0) | 8,34×10⁻⁴ (**converge**, ordem~1,5) |

**Consequência para o texto (§7.1):**
- Remover "confirmando a correta implementação" associado à validação antiga.
- A afirmação "ordem aproximadamente 2" é **falsa**: mesmo corrigido, a ordem é
  **~1,5**, limitada pela representação "em escada" da interface circular na malha
  cartesiana (exatamente o alerta do parecer 3.4). Reportar a tabela de taxas.

---

## 3. Descobertas ao executar o código (contradizem o texto atual)

Rodando os solvers reais (não estavam tabulados no artigo):

1. **Erros MEF×MDF nas tensões são grandes.** `w`: 3,4×10⁻³ (ok), mas
   `τ_xz`: 1,1×10⁻¹ e `τ_yz`: 2,8×10⁻¹. A conclusão do artigo ("10⁻³ ou inferiores
   em **todos** os campos") é **incorreta**: MDF e MEF divergem 10–30% nas tensões,
   dominadas por artefatos na interface (as tensões saem de `np.gradient` do campo
   nodal com `G` nodal, que oscila no salto). Liga-se ao parecer 2.3 e 2.5.

2. **`G_ef` preciso:** MDF = 1,139769, MEF = 1,139824 (o "≈1,14" está certo, mas o
   texto deve trazer dígitos e a incerteza da PINN).

3. **Sanidade Voigt–Reuss:** `f=(2a)²=0,09`, limites [1,0776 ; 1,3600]; `G_ef`
   dentro — confere a aritmética do parecer 2.2.

4. **Ordens de convergência reais** (`scripts/convergence.py`):
   - Circular (MDF, Dirichlet exato) vs analítica: **ordem ~1,5** (não 2).
   - Quadrada, auto-convergência em `w`: **MDF ~0,9** (1ª ordem) e **MEF ~1,7**.
   - Quadrada, `τ_xz` (MDF): **ordem ~0,5** — a tensão converge muito devagar por
     causa da singularidade de quina (parecer 2.3).
   - Nuance para §7.3/Conclusão: MEF converge mais rápido em `w`, mas isso **não**
     é o mesmo que "capturar melhor as tensões de interface" — afirmar só o que
     foi medido.

5. **Paramétrico `Gᵢ/Gₘ`** (`scripts/parametric.py`): `G_ef` = 1,00 → 1,54
   (ratios 1..100), **cresce e desacelera** (dizer "satura" é forte para ratio 100).
   `max|τ|` cresce ~100× mas `p99` só ~15× e `max_far` ~47× → o pico pontual é
   dominado pela quina, confirmando 2.3.

6. **PINN multi-seed** (`scripts/pinn_seeds.py`, 5 sementes × 4000 épocas, CPU,
   ~15 min): `G_ef` = **0,982 ± 0,002** (a semente 42, a do artigo, dá **0,981**).
   **Contradiz a Tabela `tab:gef`**, que reporta PINN ≈ 1,14. Pior: 0,98 fica
   **abaixo do limite de Reuss (1,078)** → estimativa fisicamente inconsistente.
   Erros vs MDF: `w` 4,3% ± 0,1%, `τxz` 30%, `τyz` 99%. Variância entre sementes é
   mínima → **viés sistemático**, não instabilidade: o painel de `w` mostra a PINN
   quase ignorando a inclusão (isolinhas quase verticais, campo ≈ homogêneo `γx`).
   Reforça a conclusão qualitativa (PINN sofre na interface), mas **derruba o "os
   três concordam em 1,14"**.

---

## 4. A fazer (roadmap de execução)

| Prioridade | Tarefa | Entrega | Parecer |
|---|---|---|---|
| Essencial | ✅ Convergência (circular **e** quadrada, MDF+MEF) | `scripts/convergence.py` + CSVs | 2.1, 3.4, 3.7 |
| Essencial | ✅ Tabela de erros L² (w, τ_xz, τ_yz) MEF×MDF | `scripts/comparison.py` + CSV | 2.2, 3.7 |
| Essencial | ✅ Métricas de tensão robustas (max, p99, max_far, L²) | `robust_stress_metrics` | 2.3 |
| Essencial | ⏳ Erro de fluxo normal na interface | script + tabela | 2.3, 3.3 |
| Essencial | ✅ PINN multi-seed (5 sementes, CPU): G_ef 0,982±0,002, ~15 min | `scripts/pinn_seeds.py` + CSV | 2.4 |
| Forte | ✅ Figuras do artigo (campos, convergência, paramétrico) | `scripts/make_figures.py` | 3.7 |
| Forte | ✅ Paramétrico `Gᵢ/Gₘ` (tabela); ⏳ figura | `scripts/parametric.py` + CSV | 3.7 |
| Forte | Sensibilidade aos pesos λ da perda | script PINN | 2.4 |
| Texto | ✅ Abstract/conclusão reescritos (PINN≠1,14; sem "MEF superior") | `paper/artigo.tex` | 3.1, 2.5 |
| Texto | ✅ §7.1 validação Dirichlet + ordem ~1,5; §7.3 tensões 10⁻¹; +Tabelas conv/erros/param/gef | `paper/artigo.tex` | 2.1, 2.5 |
| Texto | ✅ Condições de interface `[[·]]=0`; ⏳ espaços `V₀`/levantamento Dirichlet | `paper/artigo.tex` §3 | 3.3 |
| Texto | ⏳ Q4/Gauss no corpo do MEF; pós-proc. de tensões | `paper/artigo.tex` §5 | 3.5 |
| Editorial | ✅ comparison, Kronecker, "seção", autograd, "PINN, tem-se" | `paper/artigo.tex` | 5 |
| Editorial | ✅ "Dados" em PT + repo versionado/DOI | `paper/artigo.tex` | 3.8 |
| Editorial | ✅ Bibliografia 7 → **19** (interface/PINN/homogeneização), todas citadas; ⏳ chegar a ~25–40 | `paper/TCAM_bibliography.bib` | 5 |
| Figuras | ✅ +3 no artigo (convergência, paramétrico, painel 3 métodos); legenda erro.png corrigida | `paper/` | 5 |
| Template | Converter TCAM.cls → template da REMAT | `paper/` | — |

---

## 5. Decisões tomadas

- **Validação:** Dirichlet exato da analítica (não MMS, não benchmark MEF). 🧭→✅
- **Alvo:** REMAT, documentando tudo e seguindo o template exigido. ⚠️ O `.tex`
  atual usa `TCAM.cls`; a conversão para o modelo da REMAT fica mapeada no roadmap.
- **Repositório:** `C:\workspace\seminario-final-problema-3-metodo-numerico-edo-edp`.
