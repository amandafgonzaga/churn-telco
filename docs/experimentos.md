# Registro de experimentos

Cada execução de `uv run churn-train` acrescenta uma linha por modelo em
[`experimentos.csv`](experimentos.csv), com `run_id`, `seed`, e média e desvio padrão de cada
métrica da validação cruzada (ROC-AUC, PR-AUC, F1, Recall, Precision).

As métricas do campeão no conjunto de teste, o threshold e a versão ficam em
[`../models/champion.json`](../models/champion.json).

## Diário

| Data | run_id | Mudança testada | Resultado / decisão |
|---|---|---|---|
| 03/11/2026 | 20261103-091333 | Baseline: LogReg com `class_weight="balanced"` | ROC-AUC 0,846 ± 0,012 em CV, a maior entre os três. Virou o campeão. |
| 03/11/2026 | 20261103-091333 | RandomForest e MLP com parâmetros padrão do projeto | RandomForest ficou tecnicamente empatado com a LogReg (0,845 ± 0,009) e o MLP um pouco abaixo (0,842 ± 0,010). Como as três ficaram dentro de 1 desvio padrão, fiquei com a LogReg por ser mais simples e interpretável. |
| 03/11/2026 | 20261103-091333 | Threshold de decisão: 0,5 (padrão) vs. custo de negócio | O threshold 0,5 ignora o desbalanceamento e deixa passar muito churn (custo R$ 86.649 no teste). Busquei na grade o threshold que minimiza o custo de negócio e cheguei em 0,18, que derrubou o custo para R$ 57.021 (−83% vs. não usar modelo). Virou o threshold padrão do campeão. |
