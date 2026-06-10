#!/bin/bash

set -e

COMPETITION="ir-20261-rc"
DATA_DIR="data"

echo "Criando pasta de dados..."
mkdir -p "$DATA_DIR"

echo "Baixando dados da competição Kaggle..."
kaggle competitions download -c "$COMPETITION" -p "$DATA_DIR"

echo "Descompactando arquivos..."
unzip -o "$DATA_DIR/$COMPETITION.zip" -d "$DATA_DIR"

echo "Arquivos baixados:"
ls -lh "$DATA_DIR"

echo "Download concluído."