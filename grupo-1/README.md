# Projeto 1 – Barra Compósita Unidimensional Submetida à Tração Axial

Este repositório apresenta a implementação computacional do problema de uma **barra compósita unidimensional submetida à tração axial**, utilizando diferentes metodologias numéricas para obtenção da solução.

As implementações foram desenvolvidas com o objetivo de comparar abordagens clássicas e modernas para a resolução de problemas governados por equações diferenciais, abrangendo desde métodos numéricos tradicionais até técnicas baseadas em aprendizado de máquina.

---

## Autores

- **Heitor Salles de Araujo**
- **Marcelo Borges dos Reis**

---

## Objetivo

Implementar e analisar a solução de uma barra compósita unidimensional submetida à tração axial por meio das seguintes abordagens:

- Método dos Elementos Finitos (MEF);
- Método das Diferenças Finitas (MDF);
- Physics-Informed Neural Networks (PINNs).

O projeto possibilita comparar diferentes estratégias de solução, destacando suas características, aplicações e desempenho.

---

## Métodos Implementados

### Método dos Elementos Finitos (MEF)

Discretiza o domínio da barra em elementos finitos, aproximando o campo de deslocamentos por funções de interpolação e obtendo a solução por meio da formulação variacional.

### Método das Diferenças Finitas (MDF)

Aproxima as derivadas presentes na equação diferencial utilizando diferenças entre pontos da malha discretizada, transformando o problema contínuo em um sistema algébrico.

### Physics-Informed Neural Networks (PINNs)

Emprega redes neurais artificiais treinadas para satisfazer simultaneamente as condições de contorno e a equação diferencial governante, incorporando as leis físicas diretamente na função de perda.

---

## Estrutura do Repositório

O projeto disponibiliza diferentes formatos de arquivos para facilitar sua utilização.

- **Arquivos `.py`**: códigos-fonte em **Python**, destinados à execução local (VS Code, PyCharm, terminal, entre outros).
- **Arquivos `.ipynb`**: notebooks compatíveis com **Google Colab** e **Jupyter Notebook**, permitindo execução interativa.
- **Arquivo `.pdf`**: artigo técnico contendo a fundamentação teórica, modelagem matemática, metodologia e resultados.
- **Arquivo `.pptx`**: apresentação utilizada na exposição do projeto.

### Arquivos

| Arquivo | Descrição |
|---------|-----------|
| `barra_composita_mef.py` | Implementação do Método dos Elementos Finitos (MEF) em Python. |
| `barra_composita_mef.ipynb` | Notebook Google Colab contendo a implementação do MEF. |
| `p1_barra_mdf.py` | Implementação do Método das Diferenças Finitas (MDF) em Python. |
| `p1_barra_mdf.ipynb` | Notebook Google Colab contendo a implementação do MDF. |
| `p1_pinn_colab.py` | Implementação das Physics-Informed Neural Networks (PINNs) em Python. |
| `p1_pinn_colab.ipynb` | Notebook Google Colab contendo a implementação das PINNs. |
| `Projeto 1 Barra compósita unidimensional submetida à tração axial.pdf` | Artigo técnico contendo a fundamentação teórica, desenvolvimento matemático, metodologia, resultados e conclusões do projeto. |
| `Projeto 1_ Barra compósita unidimensional submetida à tração axial.pptx-1` | Apresentação utilizada na exposição do projeto. |

---

## Tecnologias Utilizadas

- Python 3
- NumPy
- Matplotlib
- TensorFlow
- Google Colab
- Jupyter Notebook

---

## Como Executar

### Execução Local

Execute qualquer um dos arquivos Python utilizando:

```bash
python barra_composita_mef.py
```

ou

```bash
python p1_barra_mdf.py
```

ou

```bash
python p1_pinn_colab.py
```

### Execução no Google Colab

Abra qualquer arquivo com extensão **`.ipynb`** diretamente no Google Colab e execute as células em sequência.

---

## Documentação

O repositório também disponibiliza toda a documentação do projeto:

- **Artigo Técnico (PDF):** apresenta a fundamentação teórica, modelagem matemática, metodologia, implementações computacionais, resultados obtidos e conclusões.
- **Apresentação (PowerPoint):** utilizada para apresentação do projeto, contendo uma visão geral do problema, métodos empregados e principais resultados.

---

## Licença

Este projeto foi desenvolvido exclusivamente para fins acadêmicos e educacionais.
