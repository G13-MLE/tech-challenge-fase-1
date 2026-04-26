# Resultados dos Modelos Baseline

## Resumo Executivo

Este documento consolida os resultados dos modelos baseline treinados no dataset Telco Customer Churn (IBM). O objetivo e estabelecer um benchmark inicial para a tarefa de classificacao binaria de churn, contra o qual serao comparados modelos futuros. Dois modelos foram avaliados:

- `DummyClassifier` (baselines ingenuos)
- `MLP` (rede neural fully-connected)

## Configuracao do Experimento

- Dataset: Telco Customer Churn (IBM)
- Target: Churn (binario - Yes/No)
- Split: 80/20 estratificado, seed=42
- MLflow tracking: Docker stack local

## DummyClassifier Baseline

### Estrategias Avaliadas

Tres estrategias do `DummyClassifier` foram avaliadas:

- `most_frequent`: sempre prediz a classe majoritaria (No Churn).
- `stratified`: prediz de acordo com a proporcao das classes no treino.
- `uniform`: prediz cada classe com probabilidade uniforme (50/50).

### Tabela de Metricas

| Estrategia    | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
|---------------|----------|-----------|--------|----------|---------|--------|
| most_frequent | 0.7346   | 0.0000    | 0.0000 | 0.0000   | 0.5000  | 0.2654 |
| stratified    | 0.6217   | 0.2891    | 0.2914 | 0.2903   | 0.5163  | 0.2723 |
| uniform       | 0.4869   | 0.2525    | 0.4759 | 0.3299   | 0.5000  | 0.2654 |

### Analise

- A estrategia `most_frequent` alcanca alta acuracia (0.7346) simplesmente prevendo a classe majoritaria, mas falha completamente em identificar churners (recall e precision zero). ROC-AUC e PR-AUC estao no nivel de acaso.
- `stratified` e `uniform` apresentam metricas baixas, refletindo a ausencia de poder preditivo. Notavelmente, `uniform` possui o maior recall (0.4759) porque adivinha metade dos casos como positivos, ao custo de baixa precisao.
- Todos os resultados confirmam a alta taxa de desbalanceamento do dataset (~26% de churn).

## MLP Baseline

### Configuracao do Modelo

- Arquitetura: 3 camadas ocultas (128, 64, 32)
- Dropout: 0.3
- Batch Normalization: Sim
- Otimizador: Adam
- Learning Rate: 0.001
- Weight Decay: 1e-05
- Batch Size: 64
- Max Epochs: 100
- Early Stopping Patience: 5 epochs
- Pre-processamento: One-hot encoding + StandardScaler (ajustado apenas no treino)
- Numero de features de entrada: 30

### Tabela de Metricas (Teste)

| Metrica   | Valor  |
|-----------|--------|
| Accuracy  | 0.7885 |
| Precision | 0.6073 |
| Recall    | 0.5749 |
| F1 Score  | 0.5907 |
| ROC-AUC   | 0.8378 |
| PR-AUC    | 0.6301 |

### Analise

- O MLP supera consistentemente todas as versoes Dummy, com destaque para o ROC-AUC de 0.8378 (muito acima de 0.5).
- O recall de 0.5749 indica que o modelo consegue capturar aproximadamente 57% dos clientes que realmente fazem churn.
- A precisao de 0.6073 significa que, quando o modelo prediz churn, esta correto em cerca de 61% das vezes.
- O PR-AUC de 0.6301 reflete desafios inerentes ao desbalanceamento de classes, embora seja um ganho significativo sobre os baselines.

## Tabela Comparativa Geral

| Modelo                | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
|-----------------------|----------|-----------|--------|----------|---------|--------|
| Dummy (most_frequent) | 0.7346   | 0.0000    | 0.0000 | 0.0000   | 0.5000  | 0.2654 |
| Dummy (stratified)    | 0.6217   | 0.2891    | 0.2914 | 0.2903   | 0.5163  | 0.2723 |
| Dummy (uniform)       | 0.4869   | 0.2525    | 0.4759 | 0.3299   | 0.5000  | 0.2654 |
| MLP (baseline)        | 0.7885   | 0.6073    | 0.5749 | 0.5907   | 0.8378  | 0.6301 |
| Logistic Regression   | Pendente | -         | -      | -        | -       | -      |

## Insights e Conclusoes

1. **Desbalanceamento severo**: A classe majoritaria representa ~74% dos dados, permitindo que `most_frequent` alcance alta acuracia sem valor preditivo real.
2. **Recall e crucial para churn**: Perder um cliente que fara churn e mais custoso do que uma falsa positivo. O MLP conseguiu recall de ~57%, deixando espaco para melhorias futuras.
3. **ROC-AUC como metrica chave**: Com 0.8378, o MLP demonstra boa capacidade de separacao entre as classes, servindo como benchmark solido.
4. **PR-AUC reflete a dificuldade real**: Diferente do ROC-AUC, o PR-AUC (0.6301) e sensivel ao desbalanceamento e sera um indicador mais valioso para comparar modelos futuros.
5. **O potencial esta no F1**: O F1 Score de 0.5907 indica um equilibrio razoavel, mas ha espaco para otimizacao, especialmente com tecnicas de balanceamento ou custo de erro assimetrico.

## Benchmark para Modelos Futuros

Para que um modelo seja considerado superior ao baseline atual, os seguintes patamares minimos devem ser superados:

- **ROC-AUC**: > 0.85
- **PR-AUC**: > 0.65
- **F1 Score**: > 0.62
- **Recall**: > 0.60 (prioridade para churn)
- **Precision**: > 0.62

## Referencias MLflow

Abaixo estao os nomes dos experimentos e os Run IDs utilizados para extrair as metricas acima:

### Experimento: tech-challenge-dummy-baseline

| Estrategia    | Run ID                               |
|---------------|--------------------------------------|
| most_frequent | f7bf58acdb9d4a62955ecd05228a78dd     |
| stratified    | 7f88b5cb31d746adb00ceeae056fc87e     |
| uniform       | afb444bc69394cff84418a3d3cc3b623     |

### Experimento: tech-challenge-mlp

| Descricao        | Run ID                               |
|------------------|--------------------------------------|
| MLP (baseline)   | 8cf933a209c84586ade1231bfb5f9549     |

### Acesso a Interface MLflow

1. Certifique-se de que os containers MLflow estejam ativos (`make docker-up`).
2. Abra o navegador e acesse o endereco configurado no seu ambiente (verifique `.env` para o MLFLOW_TRACKING_URI correto).
3. Navegue pelos experimentos listados acima para inspecionar metricas, parametros e artefatos.

## Nota sobre Logistic Regression

A documentacao de resultados para o modelo de Regressao Logistica esta pendente, pois a Issue #21 (Implementar e Executar Baseline de Regressao Logistica) ainda nao foi concluida. Este documento sera atualizado quando os dados de execucao do modelo estiverem disponiveis no MLflow.
