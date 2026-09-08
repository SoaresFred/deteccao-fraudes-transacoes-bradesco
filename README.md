# Detecção de Fraudes em Transações Bancárias

Projeto de análise de dados e classificação para identificar transações fraudulentas em um dataset público de cartões de crédito. A implementação acompanha os conceitos das videoaulas, mas inclui uma estrutura modular, prevenção de vazamento de dados, comparação de modelos e avaliação orientada ao recall da classe fraude.

## Objetivo

Detectar a classe rara (`Class = 1`) sem avaliar o modelo apenas pela acurácia. Em bases altamente desbalanceadas, um modelo que sempre prevê transação normal pode ter acurácia alta e ainda assim ser inútil. Por isso, o projeto acompanha precision, recall, F1-score, ROC-AUC, PR-AUC e matriz de confusão.

## Fonte dos dados

O projeto utiliza o dataset público **Credit Card Fraud Detection**, originalmente disponibilizado no Kaggle pela ULB Machine Learning Group e também espelhado em repositórios públicos. O arquivo esperado é `creditcard.csv`, com as colunas `Time`, `V1`–`V28`, `Amount` e `Class`.

Para executar, baixe o arquivo da [página do dataset no Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) e coloque-o em `data/creditcard.csv`. O arquivo de dados não é versionado neste repositório por ser grande e conter dados de terceiros.

## Como executar

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
python -m src.main --data data/creditcard.csv
```

Os gráficos e tabelas são gravados em `outputs/`.

## Abordagem

O pipeline realiza inspeção da distribuição da variável-alvo, separação estratificada de treino e teste, padronização de `Amount` dentro de um `Pipeline` para evitar data leakage, comparação entre Regressão Logística e Random Forest com `class_weight="balanced"`, avaliação com métricas adequadas e ajuste opcional do limiar de decisão para priorizar o recall.

O conjunto de teste permanece isolado. O `StandardScaler` é ajustado somente no treinamento, em vez de calcular média e desvio padrão usando todos os dados.

## Estrutura

```text
├── README.md
├── requirements.txt
├── .gitignore
├── data/                  # arquivo local, não versionado
├── outputs/               # gráficos gerados, não versionados
└── src/
    ├── __init__.py
    ├── evaluation.py
    ├── models.py
    └── main.py
```

## Interpretação

O principal critério de negócio é o **recall da classe fraude**: entre todas as fraudes reais, quantas foram identificadas. Precision também é importante para controlar falsos positivos. O limiar de 0,30 é apresentado como experimento: ele tende a capturar mais fraudes, mas pode sinalizar mais transações legítimas para investigação manual. O melhor limiar deve ser escolhido de acordo com o custo de falsos negativos e falsos positivos.

Este projeto é educacional e não deve ser utilizado sozinho para decisões financeiras reais.
