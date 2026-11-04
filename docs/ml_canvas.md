# ML Canvas: churn de clientes Telco

| Bloco | Conteúdo |
|---|---|
| **Proposta de valor** | Identificar com antecedência clientes com alto risco de cancelamento para direcionar ações de retenção. |
| **Stakeholders** | Diretoria (patrocinadora), time de Retenção/CRM (usuário da predição), Marketing, Financeiro, time de Dados. |
| **Decisão suportada** | Quais clientes recebem uma oferta de retenção neste ciclo. |
| **Tarefa de ML** | Classificação binária: `Churn` (1 = cancelou) a partir de perfil, serviços contratados e faturamento. |
| **Fontes de dados** | Telco Customer Churn (IBM): 7.043 clientes, 21 colunas, um snapshot. |
| **Features** | Demográficas (gender, SeniorCitizen, Partner, Dependents), serviços (Phone, Internet, add-ons, Streaming), conta (tenure, Contract, PaymentMethod, PaperlessBilling, MonthlyCharges, TotalCharges). |
| **Métrica técnica** | ROC-AUC (principal, independe de threshold), com PR-AUC, F1 e Recall da classe churn. |
| **Métrica de negócio** | Custo total = ofertas enviadas × R$ 50 + churns não detectados × 12 × mensalidade. Comparar com o cenário sem modelo. |
| **Baseline** | Regressão Logística com `class_weight="balanced"`. |
| **Predições / uso** | API REST (`/predict`, `/predict/batch`) consultada pelo CRM; execução em lote mensal. |
| **Avaliação offline** | Validação cruzada estratificada (5 folds) no treino, com teste hold-out de 20%. |
| **Monitoramento (futuro)** | Drift das features, taxa de churn real versus prevista, efetividade das ofertas. |

## Premissas a validar com o negócio

- Custo da oferta de retenção: **R$ 50** (`COST_RETENTION_OFFER`).
- Receita perdida por churn: **12 meses** de mensalidade (`LOST_REVENUE_MONTHS`).
- A oferta é suposta 100% efetiva. Na prática, multiplicar pelo take-rate da campanha.
