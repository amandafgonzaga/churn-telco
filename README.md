# Churn Telco: pipeline preditivo de churn com API REST

Tech Challenge da Fase 1 (12MLET). Uma operadora de telecomunicações está perdendo clientes. Este projeto
vai da análise exploratória até um modelo de propensão a churn servido via **FastAPI**,
comparando modelos **linear**, **baseado em árvores** e **rede neural (MLP)** no Scikit-Learn.

> 🎬 [Vídeo STAR (YouTube)](https://youtu.be/6kZNn47uVPQ) · 📄 [Model Card](docs/model_card.md) · 🧭 [ML Canvas](docs/ml_canvas.md)

## Arquitetura

```
data/raw/*.csv ─► churn.data.clean ─► Pipeline[ColumnTransformer ─► modelo] ─► models/champion.joblib
                                                                               │
                                     cliente (JSON cru) ─► FastAPI /predict ───┘─► probabilidade + decisão
```

O pré-processamento (imputação, escala, one-hot) fica **dentro** do `Pipeline` salvo. Assim a
validação cruzada não vaza dados e a API recebe os dados crus do cliente, sem reimplementar transformações.

## Estrutura

```
├── data/raw/            # dataset Telco (IBM)
├── notebooks/           # experimentação: EDA, baseline, comparação de modelos
├── src/churn/           # código de produção
│   ├── config.py        # seed, caminhos, features, premissas de negócio
│   ├── data.py          # carga e limpeza
│   ├── features.py      # ColumnTransformer
│   ├── models.py        # LogReg, RandomForest, MLP
│   ├── evaluate.py      # métricas técnicas + custo de negócio
│   ├── train.py         # CV, comparação, threshold, salva o campeão
│   ├── predict.py       # carga do modelo e inferência
│   └── api/             # FastAPI (main.py, schemas.py)
├── models/              # champion.joblib + champion.json (métricas, threshold, versão)
├── tests/               # pytest (dados sintéticos; não depende do CSV real)
└── docs/                # ML Canvas, Model Card, log de experimentos
```

## Setup

Pré-requisitos: Python 3.11+ e [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Como executar

1. Baixar o dataset (Telco Customer Churn, IBM) para `data/raw/`:

```bash
uv run churn-download
```

2. Treinar e comparar os modelos. O script salva o campeão em `models/` e acrescenta uma linha por modelo em `docs/experimentos.csv`:

```bash
uv run churn-train
```

3. Subir a API:

```bash
uv run uvicorn churn.api.main:app --reload
```

A documentação interativa (Swagger) fica em http://localhost:8000/docs.

4. Chamar a API:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":3,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":94.5,"TotalCharges":283.5}'
```

Resposta:

```json
{"churn_probability": 0.895, "churn_prediction": true, "threshold": 0.18, "model_version": "logistic_regression-20261103-091333"}
```

| Endpoint | Método | Descrição |
|---|---|---|
| `/health` | GET | Status da API e se o modelo está carregado |
| `/predict` | POST | Propensão de churn de um cliente |
| `/predict/batch` | POST | Propensão de churn de uma lista de clientes |

## Testes e qualidade

```bash
uv run pytest
```

```bash
uv run ruff check .
```

## Reprodutibilidade

- `SEED = 42` em [`src/churn/config.py`](src/churn/config.py), usado em split, `StratifiedKFold`, modelos e `np.random`.
- Dependências travadas em `uv.lock`.
- Split estratificado 80/20. O conjunto de teste só é usado na avaliação final do campeão.
- O threshold de decisão é escolhido com predições out-of-fold do treino, minimizando o custo de negócio.

## Resultados

Validação cruzada estratificada de 5 folds no treino (5.634 clientes), com threshold 0,5. Fonte: `docs/experimentos.csv`.

| Modelo | ROC-AUC | PR-AUC | F1 | Recall | Precision |
|---|---|---|---|---|---|
| **Regressão Logística (baseline)** | **0,846 ± 0,012** | **0,660 ± 0,020** | 0,628 ± 0,023 | **0,801 ± 0,038** | 0,517 ± 0,019 |
| Random Forest | 0,845 ± 0,009 | 0,653 ± 0,022 | **0,631 ± 0,014** | 0,761 ± 0,026 | 0,539 ± 0,014 |
| MLP | 0,842 ± 0,010 | 0,647 ± 0,020 | 0,581 ± 0,019 | 0,522 ± 0,026 | **0,656 ± 0,024** |

**Campeão: Regressão Logística.** Ela tem a maior ROC-AUC e o modelo mais simples e interpretável.
Os três modelos ficaram dentro de 1 desvio padrão entre si.

**Conjunto de teste** (1.409 clientes, threshold de custo = 0,18): ROC-AUC 0,842 · PR-AUC 0,633 ·
Recall 0,973 · Precision 0,382.

| Política | Custo de negócio (teste) |
|---|---|
| Sem modelo (nenhuma oferta) | R$ 326.579 |
| Modelo com threshold 0,5 | R$ 86.649 |
| **Modelo com threshold de custo (0,18)** | **R$ 57.021** (−83% vs. sem modelo) |

> O threshold baixo reflete as premissas de custo: perder um cliente custa 12 mensalidades,
> muito mais que uma oferta de R$ 50. Ver limitações no [Model Card](docs/model_card.md).

## Equipe

Projeto individual.

| Nome | Responsabilidades |
|---|---|
| Amanda Ferreira Gonzaga | EDA, modelagem, API e documentação |
