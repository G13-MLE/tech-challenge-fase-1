# tech-challenge-fase-1

Estrutura inicial de pastas do projeto.

## Estrutura

```text
tech-challenge-fase-1/
├── src/
│   ├── data/
│   ├── features/
│   ├── eda/
│   ├── training/
│   ├── inference/
│   ├── schemas/
│   ├── api/
│   └── pipelines/
├── data/
├── models/
├── tests/
├── notebooks/
└── docs/
```

## Configuração de Variáveis de Ambiente

O projeto utiliza variáveis de ambiente para configuração. Siga os passos abaixo:

### 1. Criar o arquivo .env

Copie o arquivo `.env.example` para `.env`:

```bash
cp .env.example .env
```

### 2. Configurar as variáveis

Edite o arquivo `.env` e preencha os valores conforme seu ambiente:

```bash
# Exemplo de configuração mínima
PORT=8000
ENVIRONMENT=development
DEBUG=true
API_TOKEN=seu_token_seguro_aqui
MODEL_PATH=./models/model.pkl
DATABASE_URL=postgresql://user:pass@localhost:5432/dbname
```

### 3. Importante - Segurança

- **NUNCA** faça commit do arquivo `.env` (já está no `.gitignore`)
- Em produção, use valores seguros para tokens e secrets
- Para gerar tokens seguros: `openssl rand -hex 32`

## Regra de execução (pipelines)

- Não gerar artefatos finais (modelos treinados, bases finais, tracking de experimento) a partir de notebooks.
- Notebooks são para exploração.
- Artefatos finais devem ser gerados por scripts parametrizáveis em `src/pipelines/`, executados via terminal.
