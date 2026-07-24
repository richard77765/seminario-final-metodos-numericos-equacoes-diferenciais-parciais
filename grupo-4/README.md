# Escoamento de Darcy em Meio Poroso Compósito Unidimensional

Implementação computacional em Python — comparação entre Método de Diferenças Finitas (MDF), Método de Elementos Finitos (MEF), Homogeneização Assintótica e Redes Neurais com Física Incorporada (PINNs) para o problema de escoamento de Darcy 1D em um meio poroso compósito periódico.

Este repositório acompanha o artigo submetido à revista **TCAM** (UFF), na disciplina de Métodos Numéricos para Equações Diferenciais Parciais, sob orientação do **Prof. Reinaldo Rodríguez Ramos**.

**Autoras/Autores:** Claudineia Moreira de Oliveira, Paulo Cesar Madeira de Oliveira — Programa de Pós-Graduação em Modelagem Computacional (PPGCM/MCCT), UFF Campus Volta Redonda.

## Descrição do problema

Escoamento de Darcy unidimensional em uma barra porosa de comprimento $L=1$, composta por duas fases periódicas com permeabilidades $k_1=1{,}0$ e $k_2=0{,}01$. A equação governante possui um coeficiente rapidamente oscilante $k^\varepsilon(x)$, e o objetivo é comparar quatro abordagens complementares:

1. **MDF** — discretização conservativa por diferenças finitas.
2. **MEF** — elementos lineares deduzidos via método dos resíduos ponderados.
3. **Homogeneização assintótica** — permeabilidade efetiva $k_{ef} \approx 0{,}019802$ (média harmônica).
4. **PINNs** — três estratégias de treinamento (coeficiente suave, coeficiente oscilante, e coeficiente suave com força externa), incluindo uma variante por decomposição de subdomínios de fase.

## Estrutura do repositório

```
.
├── src/
│   ├── projeto4_mdf.py                 # MDF conservativo + homogeneização (Figuras 1–4)
│   ├── projeto4_mef.py                 # MEF com elementos lineares (Figuras 5–7)
│   ├── projeto4_pinn.py                # PINN Estratégias 1, 2 (monolítica) e 3 (Figuras 8–10, 12)
│   └── projeto4_pinn_subdominios.py    # PINN por subdomínios de fase (Figura 11)
├── figuras/                            # (opcional) figuras finais usadas no artigo
├── docs/                               # (opcional) artigo em PDF/LaTeX
├── requirements.txt
└── README.md
```

## Como executar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

Testado com Python 3.14.5, numpy 2.5.0, scipy 1.18.0, matplotlib 3.11.0, torch 2.12.1 (CPU).

### 2. Rodar os scripts

Cada script é independente e salva suas figuras em uma subpasta `outputs/` (ou `outputs_final/`, no caso da PINN por subdomínios), criada automaticamente no diretório de execução:

```bash
cd src
python projeto4_mdf.py                 # gera Figuras 1–4
python projeto4_mef.py                 # gera Figuras 5–7 (chama funções do MDF internamente)
python projeto4_pinn.py                # gera Figuras 8–10 e 12 (treina 3 redes; mais lento)
python projeto4_pinn_subdominios.py    # gera Figura 11 (treina 20 sub-redes; usa checkpoint)
```

**Nota sobre `projeto4_pinn_subdominios.py`:** o treinamento é feito por checkpoint (`checkpoint_subdominios.pt`), em blocos de 2.000 épocas por execução. Rode o script várias vezes seguidas para acumular épocas de treinamento . Para recomeçar do zero, apague o arquivo `checkpoint_subdominios.pt`.

Nenhum script depende de GPU — todos rodam em CPU.

## Principais resultados

| Método | Referência | Erro relativo ($\varepsilon=0{,}1$) |
|---|---|---|
| MDF (N=400) | $p^0(x)$ | 4,89e-02 |
| MEF (N=400) | $p^0(x)$ | 4,89e-02 |
| PINN Estratégia 1 ($k_{ef}$) | $p^0(x)$ | 7,26e-02 |
| PINN Estratégia 2 ($k^\varepsilon$, monolítica) | MDF | 5,05e-02 |
| PINN Estratégia 2 (por subdomínios) | MDF | 1,45e-02 |
| PINN Estratégia 3 ($k_{ef}$, com $f(x)$) | MDF | 1,67e-01 |

MDF e MEF produzem resultados praticamente idênticos na mesma malha. A PINN monolítica treinada com o coeficiente oscilante $k^\varepsilon(x)$ converge para uma solução suave que não resolve a microestrutura (viés espectral); a decomposição por subdomínios de fase corrige essa limitação e recupera corretamente o padrão em degraus.

## Licença

Uso acadêmico, no âmbito da disciplina de Métodos Numéricos para Equações Diferenciais Parciais (PPGCM/MCCT — UFF).
