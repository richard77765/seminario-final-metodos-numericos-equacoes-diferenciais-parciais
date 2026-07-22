# Rastro de correções — revisão pós-parecer

**Alvo editorial:** REMAT (seção Matemática), com documentação completa das
alterações e conformidade de template.
**Base:** parecer técnico-editorial de 12/07/2026 (revisão maior).
**Última atualização:** 2026-07-21.

Legenda: ✅ feito · 🔧 em andamento · ⏳ a fazer · 🧭 decidido (aguarda execução)

---

## 1. Situação dos pontos do parecer

| # | Ponto do parecer | Gravidade | Status |
|---|---|---|---|
| 2.1 | Validação circular não valida o problema finito (viola BCs) | Crítico | ✅ código corrigido; ⏳ texto §7.1 |
| 2.2 | `G_ef` sem rigor / sanidade Voigt–Reuss | Crítico | ✅ verificado; ⏳ definição operacional no texto |
| 2.3 | `max|τ|` nas quinas não é métrica robusta | Alto | ⏳ trocar por L²/percentil/erro de fluxo |
| 2.4 | PINN com 1 semente; sem custo computacional | Alto | 🧭 script multi-seed a criar (roda na GPU) |
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

---

## 4. A fazer (roadmap de execução)

| Prioridade | Tarefa | Entrega | Parecer |
|---|---|---|---|
| Essencial | Tabela de convergência (circular **e** quadrada) | `scripts/convergence.py` + CSV | 2.1, 3.4, 3.7 |
| Essencial | Tabela de erros L² (w, τ_xz, τ_yz) MEF×MDF | `outputs/tables/` | 2.2, 3.7 |
| Essencial | Métricas de interface (salto de `w`, erro de fluxo normal) | script + tabela | 2.3, 3.3 |
| Essencial | PINN multi-seed (5–10) média±desvio + custo | `scripts/pinn_seeds.py` (GPU) | 2.4 |
| Forte | Estudo paramétrico `Gᵢ/Gₘ` com figura+tabela | `scripts/parametric.py` | 3.7 |
| Forte | Sensibilidade aos pesos λ da perda | script PINN | 2.4 |
| Texto | Reescrever abstract/conclusão (moderar "convergência", "MEF superior") | `paper/` | 3.1, 2.5 |
| Texto | Formulação: espaços `V₀`, levantamento Dirichlet, saltos de interface | `paper/` §3 | 3.3 |
| Texto | Q4/Gauss no corpo do MEF; pós-proc. de tensões | `paper/` §5 | 3.5 |
| Editorial | `comparasion`→`comparison`; `Kroenecker`→`Kronecker`; "capítulo"→"seção" | `paper/` | 5 |
| Editorial | "Dados": repo versionado + DOI (Zenodo), no idioma do artigo | `paper/` | 3.8 |
| Editorial | Bibliografia 7 → ~25–40 refs (interface/PINN) | `paper/.bib` | 5 |
| Template | Converter TCAM.cls → template da REMAT | `paper/` | — |

---

## 5. Decisões tomadas

- **Validação:** Dirichlet exato da analítica (não MMS, não benchmark MEF). 🧭→✅
- **Alvo:** REMAT, documentando tudo e seguindo o template exigido. ⚠️ O `.tex`
  atual usa `TCAM.cls`; a conversão para o modelo da REMAT fica mapeada no roadmap.
- **Repositório:** `C:\workspace\seminario-final-problema-3-metodo-numerico-edo-edp`.
