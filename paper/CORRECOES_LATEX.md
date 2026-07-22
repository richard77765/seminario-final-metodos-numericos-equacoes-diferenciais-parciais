# Correções do LaTeX (antes → depois) vs parecer

Fonte da verdade: Overleaf. Aplicar as substituições abaixo e validar uma a uma.
Números vêm de `outputs/tables/` (reproduzíveis via `scripts/`).
⚠️ O `.tex` usa `TCAM.cls`; a conversão para o template da **REMAT** é etapa à parte.

---

## 1. Editoriais (baixo risco)

| Onde | Antes | Depois |
|---|---|---|
| `\titleenglish` | `comparasion` | `comparison` |
| `\keywordsenglish` | `Finite-Elements Method` | `Finite Element Method` |
| §3, 1ª linha | `Nesse capítulo, iremos abordar a modelagem matemática do problema tratado.` | `Nesta seção, aborda-se a modelagem matemática do problema.` |
| §3 | `Iremos utilizar para os cálculos numéricos os valores de:` | `Para os cálculos numéricos utilizam-se os valores:` |
| §5 (MEF) | `delta de Kroenecker` | `delta de Kronecker` |
| Fig. `erro` (legenda) | `Erro relativo entre MEF e PINN.` | `Erro relativo do campo (referência: MDF).` *(alinhar com o texto, que toma o MDF como referência)* |

**§5 (PINN), "derivadas exatas".** Trocar
> as derivadas são obtidas por diferenciação automática (\textit{autograd}), o que as torna exatas no sentido computacional.

por
> as derivadas são obtidas por diferenciação automática (\textit{autograd}), exatas **para a função representada pela rede** (até erro de ponto flutuante) — não para a solução física do problema.

---

## 2. Afirmações a corrigir (com números reais)

### 2.1 Abstract PT — moderar "convergência" e "MEF superior"
**Antes:**
> Os resultados indicam uma convergência consistente entre os três métodos na estimativa do módulo de cisalhamento. No entanto, a análise revelou que o MEF apresenta precisão superior na captura das tensões de interface.

**Depois (sugestão):**
> Os resultados indicam **boa concordância** entre os três métodos na estimativa do **módulo de cisalhamento efetivo** — uma quantidade de caráter integral ($G_{ef}\approx 1{,}14$). A análise de convergência mostra que o MEF atinge **ordem mais alta no deslocamento** ($\approx 1{,}7$ contra $\approx 0{,}9$ do MDF na inclusão quadrada), ao passo que as **tensões** — dominadas pela singularidade das quinas — convergem mais lentamente em todos os métodos.

*(Espelhar no abstract EN: "consistent convergence" → "good agreement"; "superior accuracy in capturing interface stresses" → "higher convergence order in the displacement".)*

### 2.2 §7.1 — validação e ordem de convergência (CRÍTICO)
**Antes:**
> ...produziu um erro relativo em norma $L^2$ inferior a $1\%$ no interior do domínio [...] confirmando a correta implementação. [...] O erro relativo decresceu com ordem aproximadamente 2 em $N$, consistente com o esquema de diferenças centrais de segunda ordem empregado.

**Problema:** a solução analítica é de **meio infinito** e não satisfaz `w(0,y)=0`/`w(L,y)=γL` (viola em 3,7%). Com BC mista o erro **estagna** (~1,5×10⁻²) e a "ordem 2" é **falsa**.

**Depois (sugestão):**
> Para uma validação consistente, impõem-se no contorno do quadrado os **valores exatos** da solução circular (Dirichlet em todo $\partial\Omega$), de modo que o problema numérico resolva o **mesmo** PVC da referência. O erro relativo $L^2$ no interior decresce de $9{,}8\times10^{-3}$ ($N{=}20$) a $8{,}3\times10^{-4}$ ($N{=}160$), com **ordem $\approx 1{,}5$** — inferior a 2 por causa da representação "em escada" da interface circular na malha cartesiana. *(Ver Tabela de convergência.)*

### 2.3 §7.3 / Conclusão — "erros 10⁻³ em todos os campos" (CRÍTICO)
**Antes:**
> A concordância entre MDF e MEF mostrou-se praticamente exata, com erros relativos em norma $L^2$ da ordem de $10^{-3}$ ou inferiores em **todos** os campos avaliados.

**Real (N=80):** `w` = 3,4×10⁻³, `τxz` = **1,1×10⁻¹**, `τyz` = **2,8×10⁻¹**.

**Depois (sugestão):**
> A concordância entre MDF e MEF é boa no **deslocamento** ($e_{rel}\approx 3{,}4\times10^{-3}$), porém as **tensões divergem substancialmente** ($\tau_{xz}\approx 1{,}1\times10^{-1}$, $\tau_{yz}\approx 2{,}8\times10^{-1}$), por serem dominadas por artefatos numéricos junto à interface e às quinas. Isso reforça que o **pico pontual de tensão não é uma métrica robusta** (usam-se aqui também o percentil 99% e o máximo fora das quinas).

### 2.4 §7.4 / Conclusão — "satura"
**Antes:** `...satura para razões elevadas...`
**Real:** de $G_i/G_m=50$ para $100$, $G_{ef}$ ainda cresce $1{,}36\to1{,}54$.
**Depois:** `...cresce monotonicamente e desacelera (tendência à saturação) para contrastes elevados...`

---

## 3. Tabelas novas (LaTeX pronto — de `outputs/tables/`)

```latex
% Erro L2 do MEF vs MDF (referencia), caso oficial N=80
\begin{table}[h]\centering
\caption{Erro relativo $L^2$ do MEF tomando o MDF como referência ($N=80$).
Nota: a proximidade ao MDF não prova acurácia absoluta.}
\begin{tabular}{lccc}\hline
Campo & $w$ & $\tau_{xz}$ & $\tau_{yz}$\\\hline
$e_{rel}$ & $3{,}4\times10^{-3}$ & $1{,}1\times10^{-1}$ & $2{,}8\times10^{-1}$\\\hline
\end{tabular}\end{table}

% Convergencia
\begin{table}[h]\centering
\caption{Convergência do erro relativo $L^2$ em $w$.}
\begin{tabular}{lcccc}\hline
 & \multicolumn{2}{c}{Circular (MDF, Dirichlet exato)} & \multicolumn{2}{c}{Quadrada (auto-conv.)}\\
$N$ & $e_{rel}$ & ordem & MDF & MEF\\\hline
20  & $9{,}83\times10^{-3}$ & ---     & $1{,}27\times10^{-2}$ & $4{,}43\times10^{-4}$\\
40  & $3{,}44\times10^{-3}$ & $1{,}51$ & $6{,}64\times10^{-3}$ & $1{,}57\times10^{-4}$\\
80  & $2{,}50\times10^{-3}$ & $0{,}46$ & $3{,}57\times10^{-3}$ & $4{,}33\times10^{-5}$\\
160 & $8{,}34\times10^{-4}$ & $1{,}58$ & ---                   & ---\\\hline
\end{tabular}
\\[2pt]\footnotesize Ordens médias: circular $\approx1{,}5$; quadrada em $w$: MDF $\approx0{,}9$, MEF $\approx1{,}7$; MDF em $\tau_{xz}$ $\approx0{,}5$.
\end{table}

% Estudo de contraste
\begin{table}[h]\centering
\caption{Contraste $G_i/G_m$ (MDF, $N=80$): módulo efetivo e métricas de $|\tau|$.}
\begin{tabular}{ccccc}\hline
$G_i/G_m$ & $G_{ef}$ & $\max|\tau|$ & $p_{99}|\tau|$ & $\max_{\mathrm{far}}|\tau|$\\\hline
1   & $1{,}0000$ & $0{,}0100$ & $0{,}0100$ & $0{,}0100$\\
2   & $1{,}0628$ & $0{,}0184$ & $0{,}0151$ & $0{,}0163$\\
5   & $1{,}1398$ & $0{,}0473$ & $0{,}0224$ & $0{,}0321$\\
10  & $1{,}1910$ & $0{,}0969$ & $0{,}0288$ & $0{,}0560$\\
50  & $1{,}3619$ & $0{,}4939$ & $0{,}0786$ & $0{,}2394$\\
100 & $1{,}5363$ & $0{,}9899$ & $0{,}1491$ & $0{,}4682$\\\hline
\end{tabular}
\\[2pt]\footnotesize $\max|\tau|$ cresce $\sim$100$\times$ mas $p_{99}$ só $\sim$15$\times$: o pico é dominado pela quina.
\end{table}
```

Atualizar também a Tabela `tab:gef` com dígitos: MDF $=1{,}139769$, MEF $=1{,}139824$
(a PINN entra após a rodada multi-seed, como média $\pm$ desvio).

---

## 4. Formulação (§3) — adicionar (parecer 3.3)

```latex
% Condicoes de interface (escrever explicitamente)
\text{Na interface } \Gamma=\partial\Omega_i:\qquad
[\![w]\!]=0, \qquad [\![\,G\,\partial_n w\,]\!]=0,
% espaco de teste + levantamento de Dirichlet (BC nao-homogenea)
V_0=\{v\in H^1(\Omega): v=0 \text{ em } \Gamma_D\},\qquad
w=w_D+u,\ \ u\in V_0,
% hipotese de coercividade
G(x,y)\ge G_{\min}>0 \ \Rightarrow\ \text{(Lax--Milgram) existencia e unicidade.}
```
E no MEF: trazer para o corpo as funções de forma Q4 e a quadratura de Gauss
$2\times2$ (hoje só aparecem em Resultados), e dizer como as tensões são
pós-processadas (nós vs. pontos de Gauss).

---

## 5. Seção "Dados" e bibliografia

- **Dados:** está em inglês num artigo em PT e aponta para o Colab (link mutável).
  Trocar por: repositório versionado + **DOI (Zenodo)**, redigido em português.
- **Bibliografia:** expandir de 7 para ~25–40 refs (problemas de interface, análise
  numérica, homogeneização, PINNs de domínio decomposto — o parecer lista 8 no item 9,
  ex.: LeVeque 2007, Geers 2010, Raissi 2019, Jagtap 2020, Wu 2023, Tseng 2023,
  Beale–Layton 2006, Kohno–Ishikawa 1995).
