# Model Card: Churn Telco

## Detalhes do modelo

- **Versão:** `logistic_regression-20261103-091333` (ver `models/champion.json`)
- **Tipo:** Regressão Logística dentro de um `sklearn.Pipeline` com imputação, `StandardScaler` e `OneHotEncoder`.
- **Saída:** probabilidade de churn e decisão binária com o threshold `threshold`, escolhido para minimizar o custo de negócio.
- **Treino:** 03/11/2026, `SEED = 42`, split estratificado 80/20, CV estratificada de 5 folds.
- **Responsáveis:** Amanda Ferreira Gonzaga (projeto individual)

## Uso pretendido

- **Para:** priorizar clientes em campanhas de retenção.
- **Não usar para:** decisões automáticas que prejudiquem o cliente (negar serviço, alterar preço individualmente), nem para populações muito diferentes da base de treino.

## Dados

- Telco Customer Churn (IBM): 7.043 clientes, cerca de 26,5% de churn.
- Limpeza: `TotalCharges` convertido para número (11 brancos com `tenure = 0` preenchidos com 0) e `customerID` removido.

## Métricas

| Métrica | CV (média ± dp) | Teste |
|---|---|---|
Campeão: Regressão Logística (`class_weight="balanced"`). A CV usa threshold 0,5; o teste usa o threshold de custo, 0,18.

| Métrica | CV (média ± dp) | Teste |
|---|---|---|
| ROC-AUC | 0,846 ± 0,012 | 0,842 |
| PR-AUC | 0,660 ± 0,020 | 0,633 |
| F1 | 0,628 ± 0,023 | 0,549 |
| Recall | 0,801 ± 0,038 | 0,973 |
| Precision | 0,517 ± 0,019 | 0,382 |
| Custo de negócio (R$) | — | 57.021 |
| Custo sem modelo (R$) | — | 326.579 |

## Análise de vieses

Comparar as métricas do teste por subgrupo:

| Subgrupo | n | Taxa de churn real | Recall | Precision |
|---|---|---|---|---|
| gender = Female | 687 | 28,1% | 0,969 | 0,398 |
| gender = Male | 722 | 25,1% | 0,978 | 0,366 |
| SeniorCitizen = 0 | 1.187 | 23,3% | 0,967 | 0,350 |
| SeniorCitizen = 1 | 222 | 44,1% | 0,990 | 0,511 |

Por gênero o modelo se comporta de forma bem parecida — recall e precision ficam a menos de 3 pontos
percentuais de distância entre Female e Male, o que não me parece uma diferença preocupante.

Já por idade (SeniorCitizen) o quadro é outro: clientes idosos cancelam quase o dobro
(44,1% vs. 23,3%) e, como o modelo aprende esse padrão, acaba sinalizando esse grupo com mais
frequência — por isso a precision sobe para eles (0,511 vs. 0,350). Não é um erro do modelo, é
reflexo de um comportamento real da base, mas tem uma implicação prática: se a retenção é tratada
como recurso limitado, idosos acabam concentrando uma fatia desproporcional das ofertas. Vale o time
de negócio decidir se isso é aceitável ou se faz sentido balancear a alocação por outro critério além
do score puro.

`gender` entrou nas features porque é o que o dataset oferece e não parece introduzir viés indevido
pelos números acima. `SeniorCitizen` eu mantive de propósito — é um preditor real de churn e o
objetivo aqui é ranquear risco, não decidir quem recebe atendimento; a ressalva é só sobre como a
lista de clientes sinalizados é usada depois.

## Limitações

- O dataset é pequeno, de um único snapshot no tempo e de uma operadora dos EUA. A generalização para o mercado brasileiro não foi validada.
- Não há variáveis de interação com o cliente (chamados, NPS, uso), que costumam ser fortes preditores.
- A probabilidade não foi calibrada formalmente. Usar como ranking de risco.
- As premissas de custo (R$ 50 por oferta, 12 meses de receita) são hipóteses e afetam diretamente o threshold.
- O `MLPClassifier` não suporta `class_weight`. O desbalanceamento é compensado apenas pelo threshold.

## Considerações éticas

- Evitar que a decisão de retenção resulte em tratamento desigual por gênero ou idade.
- Revisar o modelo periodicamente (drift) e retreinar com dados recentes.
