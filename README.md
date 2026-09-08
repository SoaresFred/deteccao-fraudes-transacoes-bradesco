# Detecção de Fraudes Bancárias com Machine Learning

Projeto modular de análise de dados e classificação para identificar transações fraudulentas em um dataset público de cartões de crédito. A solução foi estruturada para demonstrar boas práticas de ciência de dados: baseline ingênuo, prevenção de vazamento, separação treino/validação/teste, comparação de modelos, seleção de threshold em validação e avaliação final com métricas apropriadas para uma classe rara.

## Objetivo

Detectar a classe rara (`Class = 1`) sem depender apenas da acurácia. Em bases altamente desbalanceadas, um modelo que sempre prevê transação normal pode apresentar acurácia alta e ainda ser inútil. Por isso, o projeto acompanha precision, recall, F1-score, ROC-AUC, PR-AUC, falsos positivos e falsos negativos.

## Fonte dos dados

O projeto utiliza o dataset público [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), originalmente disponibilizado pela ULB Machine Learning Group. O arquivo esperado é `creditcard.csv`, com as colunas `Time`, `V1`–`V28`, `Amount` e `Class`.

Baixe o arquivo no Kaggle e coloque-o em `data/creditcard.csv`. O CSV não é versionado porque é grande e contém dados de terceiros.

## Como executar

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
python -m src.main --data data/creditcard.csv --output outputs
```

Também é possível executar diretamente:

```bash
python src/main.py --data data/creditcard.csv --output outputs
```

Para rodar os testes:

```bash
pytest -q
```

Os gráficos e a tabela `model_metrics.csv` são gravados em `outputs/`.

## Metodologia

O fluxo valida as colunas e a variável-alvo, extrai uma feature aproximada de hora do dia a partir de `Time` e remove o contador original. Em seguida, separa os dados em 60% treino, 20% validação e 20% teste usando `stratify`, mantendo o teste isolado até a avaliação final.

A Regressão Logística usa `StandardScaler` dentro de um `Pipeline`, enquanto o Random Forest usa `class_weight="balanced"`. Um `DummyClassifier` com estratégia `prior` serve como baseline ingênuo. O threshold alternativo é escolhido somente no conjunto de validação, respeitando uma precisão mínima, e depois aplicado uma única vez no conjunto de teste.

## Modelos e métricas

Os modelos comparados são:

- **Baseline (Dummy):** referência simples baseada na distribuição das classes;
- **Regressão Logística:** baseline interpretável para classificação;
- **Random Forest:** conjunto de árvores com ponderação para a classe minoritária.

O principal critério de negócio é o **recall da fraude**, pois falsos negativos representam fraudes que não foram detectadas. A precision controla o volume de falsos positivos encaminhados para investigação manual. PR-AUC é acompanhada porque costuma ser mais informativa que ROC-AUC em problemas extremamente desbalanceados.

Após o treinamento, o projeto salva a importância das variáveis do Random Forest e os coeficientes absolutos da Regressão Logística em CSV e PNG. Essas explicações são associativas, não causais, especialmente para as variáveis anonimizadas `V1`–`V28`.

## Estrutura

```text
├── README.md
├── requirements.txt
├── .gitignore
├── data/                  # CSV local, não versionado
├── outputs/               # resultados gerados, não versionados
├── tests/
│   └── test_evaluation.py
└── src/
    ├── __init__.py
    ├── evaluation.py
    ├── models.py
    └── main.py
```

## Limitações

As variáveis `V1`–`V28` são anonimizadas e transformadas, portanto sua importância não deve ser interpretada como causalidade. A feature `Hour` é uma aproximação derivada do contador `Time`, não uma data real de negócio, o que limita validações temporais. Os resultados numéricos devem ser preenchidos após a execução com o CSV original; este repositório não inventa métricas.

Este projeto é educacional e não deve ser utilizado sozinho para decisões financeiras reais.
