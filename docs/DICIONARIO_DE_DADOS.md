# Telco Customer Churn (IBM) — Dicionário de Dados

## Fonte

- **Dataset:** Telco Customer Churn
- **Origem:** IBM Sample Data Sets
- **Referência pública:**
  - IBM Community: https://community.ibm.com/community/user/businessanalytics/blogs/monil-shah/2019/07/31/how-to-calculate-customer-churn-rate-in-spss-modeler
  - Kaggle (espelho amplamente utilizado):
    https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## Licença e uso

Este dataset é amplamente distribuído para fins educacionais e de demonstração.
Antes de uso em produção/comercial, validar os termos da fonte escolhida
(IBM/Kaggle) e as políticas internas do projeto.

## Arquivo versionado neste projeto

- `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Linhas: `7043`
- Colunas: `21`
- SHA256:
  `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a`

## Dicionário de campos

| Coluna | Tipo (esperado) | Descrição |
|---|---|---|
| `customerID` | string | Identificador único do cliente |
| `gender` | categórica | Gênero (`Female`, `Male`) |
| `SeniorCitizen` | inteiro/binária | Indica se é idoso (`0`/`1`) |
| `Partner` | categórica | Possui parceiro(a) (`Yes`/`No`) |
| `Dependents` | categórica | Possui dependentes (`Yes`/`No`) |
| `tenure` | inteiro | Tempo de permanência (meses) |
| `PhoneService` | categórica | Serviço de telefone (`Yes`/`No`) |
| `MultipleLines` | categórica | Múltiplas linhas (`Yes`/`No`/`No phone service`) |
| `InternetService` | categórica | Tipo de internet (`DSL`, `Fiber optic`, `No`) |
| `OnlineSecurity` | categórica | Serviço de segurança online |
| `OnlineBackup` | categórica | Serviço de backup online |
| `DeviceProtection` | categórica | Proteção de dispositivo |
| `TechSupport` | categórica | Suporte técnico |
| `StreamingTV` | categórica | Streaming de TV |
| `StreamingMovies` | categórica | Streaming de filmes |
| `Contract` | categórica | Tipo de contrato (`Month-to-month`, `One year`, `Two year`) |
| `PaperlessBilling` | categórica/binária | Fatura sem papel (`Yes`/`No`) |
| `PaymentMethod` | categórica | Método de pagamento |
| `MonthlyCharges` | numérica | Cobrança mensal |
| `TotalCharges` | numérica (com possíveis vazios) | Cobrança total acumulada |
| `Churn` | alvo categórico/binário | Cancelamento (`Yes`/`No`) |
