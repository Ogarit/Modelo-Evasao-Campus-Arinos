# Modelo Preditivo de Evasão Acadêmica — IFNMG Campus Arinos

> Trabalho de Conclusão de Curso — Bacharelado em Sistemas de Informação  
> Instituto Federal do Norte de Minas Gerais (IFNMG) — Campus Arinos  
> Autor: **Tiago Marques Lima** | Orientador: Prof. Danilo Silveira Martins

---

## Visão Geral

Este projeto investiga os fatores determinantes da evasão discente no curso de Bacharelado em Sistemas de Informação (BSI) do IFNMG Campus Arinos e desenvolve um **modelo preditivo de aprendizado de máquina** para identificação precoce de estudantes em risco.

A pesquisa analisou dados institucionais e socioeconômicos de estudantes ingressantes entre **2016 e 2023**, utilizando microdados do **INEP** e da **Plataforma Nilo Pecanha (PNP)**. O resultado é um pipeline de ML capaz de classificar estudantes como *Evadido* ou *Não Evadido* com base em variáveis acadêmicas e socioeconômicas.

**Contexto do problema:** dos 35 alunos que ingressaram em 2019 no curso de BSI do campus, apenas 13 permaneceram matriculados — uma taxa de evasão local de **66,7%**, superior à média nacional de 59% (INEP, 2021).

---

## Stack Técnica

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-red)
![SHAP](https://img.shields.io/badge/SHAP-Explicabilidade-lightgrey)
![Pandas](https://img.shields.io/badge/Pandas-2.x-green?logo=pandas)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)

| Categoria | Ferramentas |
|---|---|
| Linguagem | Python 3.11+ |
| Manipulação de dados | Pandas, NumPy |
| Modelagem | Scikit-learn, XGBoost |
| Explicabilidade | SHAP (TreeExplainer) |
| Visualização | Matplotlib, Seaborn |
| Serialização | Joblib |
| Ambiente | Jupyter Notebook, Ubuntu 22.04 (WSL) |

---

## Metodologia

### 1. Construção da Base de Alunos

A base original contém dados agregados por grupo de características (raça, renda, sexo, faixa etária, fonte de financiamento). O pré-processamento desagrega esses registros em **linhas individuais de alunos**, atribuindo a cada um a situação de `Evadido`, `Matriculado` ou `Concluído`, resultando em **296 registros**.

A variável-alvo é definida como:

$$y_i = \begin{cases} 1 & \text{se } \text{Situação}_i = \text{Evadido} \\\ 0 & \text{caso contrário} \end{cases}$$

### 2. Features Utilizadas

| Feature | Tipo | Transformação |
|---|---|---|
| `ClassificacaoRacial` | Categórica | OneHotEncoder (drop='first') |
| `Sexo` | Categórica | OneHotEncoder (drop='first') |
| `FonteFinanciamento` | Categórica | OneHotEncoder (drop='first') |
| `RendaFamiliarNum` | Numérica | StandardScaler |
| `FaixaEtariaNum` | Numérica | StandardScaler |

### 3. Modelos Avaliados

Cinco algoritmos foram treinados e comparados via pipeline Scikit-learn com `train_test_split` (80/20, `stratify=y`, `random_state=42`):

| Modelo | Acurácia | Precisão | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 0.70 | 0.75 | 0.75 | 0.75 |
| Random Forest | 0.73 | 0.76 | 0.81 | **0.78** |
| **Gradient Boosting** | **0.75** | **0.80** | 0.78 | **0.79** |
| XGBoost | 0.73 | 0.78 | 0.78 | 0.78 |
| Decision Tree | 0.73 | 0.78 | 0.78 | 0.78 |

### 4. Modelo Selecionado

O **Gradient Boosting** foi selecionado por apresentar o melhor equilíbrio entre precisão e recall, com **F1-Score de 0.79** para a classe positiva (Evadido). O modelo foi interpretado com **SHAP values**, revelando que `FonteFinanciamento` e `RendaFamiliarNum` são as variáveis de maior impacto preditivo.

---

## Principais Resultados da Análise Exploratória

- A **fonte de financiamento** (Sem Programa Associado vs. Recursos Orçamentários) é o fator de maior importância no modelo, com peso de ~0.65 na impurity-based feature importance do Gradient Boosting.
- A **renda familiar** é o segundo fator mais relevante, indicando que estudantes de menor renda apresentam maior risco de evasão.
- O **sexo masculino** aparece como terceiro fator em importância, alinhado com literatura nacional sobre evasão diferenciada por gênero.
- Variáveis de **classificação racial** apresentam baixo poder preditivo isolado, mas interagem significativamente com renda.
- Aproximadamente **50% das evasões** ocorrem nos primeiros semestres, consistente com o modelo teórico de Tinto (1993).

---

## Como Reproduzir

**Pré-requisitos:**

```bash
pip install -r requirements.txt
```

**Executar a análise exploratória:**

```bash
jupyter notebook notebooks/01_analise_exploratoria.ipynb
```

**Treinar o modelo:**

```bash
jupyter notebook notebooks/02_modelo_preditivo.ipynb
```

**Usar o modelo serializado:**

```python
import joblib
import pandas as pd

modelo = joblib.load('modelo_evasao.pkl')

novo_aluno = pd.DataFrame([{
    'ClassificacaoRacial': 'Parda',
    'Sexo': 'Masculino',
    'FonteFinanciamento': 'Sem Programa Associado',
    'RendaFamiliarNum': 375,
    'FaixaEtariaNum': 21
}])

probabilidade = modelo.predict_proba(novo_aluno)[0][1]
print(f"Probabilidade de evasão: {probabilidade:.2%}")
```

---

## Limitações

- **Tamanho da amostra:** 296 registros reconstruídos de dados agregados. O processo de desagregação preserva proporções, mas não trajetórias individuais reais.
- **Features disponíveis:** variáveis acadêmicas intra-semestre (notas, frequência, engajamento no LMS) não estavam disponíveis na base utilizada e poderiam aumentar substancialmente a performance preditiva.
- **Escopo institucional:** o modelo foi treinado exclusivamente com dados do BSI/IFNMG-Arinos. Generalização para outros cursos ou campi requer retreinamento.
- O modelo deve ser entendido como **suporte à decisão institucional**, não como solução definitiva para o fenômeno da evasão.

---

## Referências Principais

- INEP. *Censo da Educação Superior*, 2021 e 2024.
- TINTO, V. *Dropout from higher education: A theoretical synthesis of recent research*. Review of Educational Research, 1975.
- TINTO, V. *Leaving College: Rethinking the Causes and Cures of Student Attrition*. 2. ed. University of Chicago Press, 1993.
- Plataforma Nilo Pecanha (PNP) — Ministério da Educação.

---

## Contato

**Tiago Marques Lima**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Tiago_Marques_Lima-blue?logo=linkedin)](https://www.linkedin.com/in/tiago-marques-lima)
[![GitHub](https://img.shields.io/badge/GitHub-Ogarit-black?logo=github)](https://github.com/Ogarit)
