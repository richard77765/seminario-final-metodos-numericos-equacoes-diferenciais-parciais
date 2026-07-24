# Projeto 1 – Barra Compósita Unidimensional Submetida à Tração Axial

Este repositório apresenta a implementação computacional de uma **barra compósita unidimensional submetida à tração axial**, utilizando três abordagens numéricas distintas para a solução do problema:

- **Método dos Elementos Finitos (MEF)**
- **Método das Diferenças Finitas (MDF)**
- **Physics-Informed Neural Networks (PINNs)**

O projeto foi desenvolvido com fins acadêmicos, permitindo comparar diferentes metodologias para a solução de problemas de mecânica estrutural.

---

## Autores

- **Heitor Salles de Araujo**
- **Marcelo Borges dos Reis**

---

## Objetivo

Implementar e comparar diferentes métodos computacionais para a análise de uma barra compósita unidimensional submetida à tração axial, avaliando sua precisão, facilidade de implementação e potencial de aplicação.

---

## Estrutura do Repositório

O repositório disponibiliza diferentes formatos de arquivos para facilitar a utilização e o estudo das implementações.

- **Arquivos `.py`**: códigos-fonte em **Python**, destinados à execução em ambientes locais (VS Code, PyCharm, terminal, entre outros).
- **Arquivos `.ipynb`**: notebooks compatíveis com **Google Colab** e **Jupyter Notebook**, permitindo execução interativa em células.
- **Arquivo `.pptx`**: apresentação do projeto contendo fundamentação teórica, metodologia e resultados.

### Arquivos

| Arquivo | Descrição |
|---------|-----------|
| `barra_composita_mef.py` | Implementação do Método dos Elementos Finitos (MEF) em Python. |
| `barra_composita_mef.ipynb` | Notebook Google Colab contendo a implementação do MEF. |
| `p1_barra_mdf.py` | Implementação do Método das Diferenças Finitas (MDF) em Python. |
| `p1_barra_mdf.ipynb` | Notebook Google Colab contendo a implementação do MDF. |
| `p1_pinn_colab.py` | Implementação utilizando Physics-Informed Neural Networks (PINNs) em Python. |
| `p1_pinn_colab.ipynb` | Notebook Google Colab contendo a implementação das PINNs. |
| `Projeto 1_ Barra compósita unidimensional submetida à tração axial.pptx-1` | Apresentação do projeto contendo o problema proposto, metodologia utilizada e resultados obtidos. |

---

## Métodos Implementados

### Método dos Elementos Finitos (MEF)

O MEF discretiza o domínio da barra em elementos finitos, permitindo obter uma aproximação numérica do deslocamento ao longo da estrutura.

---

### Método das Diferenças Finitas (MDF)

O MDF aproxima as derivadas diferenciais por diferenças entre pontos discretizados da barra, transformando o problema diferencial em um sistema algébrico.

---

### Physics-Informed Neural Networks (PINNs)

As PINNs utilizam redes neurais treinadas para satisfazer simultaneamente os dados do problema e as leis físicas descritas pela equação diferencial governante.

---

## Tecnologias Utilizadas

- Python 3
- NumPy
- Matplotlib
- TensorFlow (PINNs)
- Google Colab
- Jupyter Notebook

---

## Como executar

### Execução local

Execute qualquer um dos arquivos Python:

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

---

### Google Colab

Abra qualquer arquivo com extensão `.ipynb` diretamente no Google Colab e execute as células sequencialmente.

---

## Resultados

Cada implementação permite obter a solução do problema utilizando uma abordagem numérica distinta, possibilitando comparar:

- Distribuição de deslocamentos;
- Precisão das soluções;
- Facilidade de implementação;
- Desempenho computacional;
- Diferenças entre métodos clássicos e aprendizado de máquina baseado em física.

---

## Licença

Este projeto foi desenvolvido exclusivamente para fins acadêmicos.
```
