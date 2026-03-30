#!/bin/bash

# Script de inicializacao do MLflow UI
# Uso: ./scripts/start_mlflow.sh [porta]
#
# Recomendado: Use 'make mlflow' ou 'uv run mlflow ui' diretamente

set -e

# Porta padrao
PORT=${1:-5000}
BACKEND_STORE="file:./mlruns"

echo "Iniciando MLflow UI..."
echo "   Porta: $PORT"
echo "   Backend: $BACKEND_STORE"

# Verificar se uv esta disponivel
if ! command -v uv &> /dev/null; then
    echo "[ERRO] uv nao encontrado."
    echo "   Instale com: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Verificar se a porta esta em uso (Linux/Mac)
if command -v lsof &> /dev/null; then
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "[AVISO] Porta $PORT ja esta em uso!"
        echo "   Encerrando processo existente..."
        kill $(lsof -t -i:$PORT) 2>/dev/null || true
        sleep 2
    fi
fi

# Criar diretorio mlruns se nao existir
mkdir -p mlruns

# Iniciar MLflow UI
echo "Acesse: http://localhost:$PORT"
echo "MLflow UI iniciado!"
echo ""

uv run mlflow ui --port $PORT --backend-store-uri $BACKEND_STORE
