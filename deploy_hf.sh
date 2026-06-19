#!/usr/bin/env bash
# Deploy do app para o Hugging Face Space (SDK docker).
#
# Monta uma pasta isolada (_hf_deploy/) com SO os arquivos que o Space precisa,
# num repositorio git separado do seu GitHub — assim o deploy nao mexe no repo
# principal nem arrisca empurrar algo errado para o origin do GitHub.
#
# USO:
#   1. Gere um token WRITE em https://huggingface.co/settings/tokens
#   2. Rode:   bash deploy_hf.sh
#   3. Quando o git pedir senha no push, COLE O TOKEN (nao a senha da conta).
#
# Variaveis (com defaults para a conta do Herick):
HF_USER="${HF_USER:-HerickHuggingFaceAccount}"
HF_SPACE="${HF_SPACE:-cartas}"

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
DEPLOY="$ROOT/_hf_deploy"
SPACE_URL="https://huggingface.co/spaces/$HF_USER/$HF_SPACE"

echo ">> Space: $SPACE_URL"
echo ">> Montando pasta de deploy isolada em: $DEPLOY"

# 1) Pasta limpa
rm -rf "$DEPLOY"
mkdir -p "$DEPLOY"

# 2) Copia SO o necessario para a inferencia web
cp -r "$ROOT/src" "$DEPLOY/src"
cp -r "$ROOT/reports" "$DEPLOY/reports"
cp "$ROOT/Dockerfile" "$DEPLOY/Dockerfile"
cp "$ROOT/requirements-deploy.txt" "$DEPLOY/requirements-deploy.txt"
cp "$ROOT/.dockerignore" "$DEPLOY/.dockerignore"
# O README do Space PRECISA do YAML header do HF (sdk: docker, app_port).
cp "$ROOT/README_HF.md" "$DEPLOY/README.md"

# Modelos: SO o checkpoint principal (o app carrega apenas este). Os modelos de
# experimento (exp_fe/exp_ft/exp_noaug, ~48 MB) nao sao usados em producao.
mkdir -p "$DEPLOY/models"
cp "$ROOT/models/efficientnet_b0_best.pt" "$DEPLOY/models/efficientnet_b0_best.pt"

# reports/: o app le APENAS os CSVs (texto). Os PNGs das matrizes de confusao nao
# sao usados pelo frontend e o HF rejeita binarios fora do LFS — entao removemos.
rm -f "$DEPLOY"/reports/*.png

# Nao leva codigo de treino/baseline (inferencia nao usa) nem caches.
rm -f "$DEPLOY"/src/baseline.py "$DEPLOY"/src/train.py "$DEPLOY"/src/ood_design.py
find "$DEPLOY" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true

# 3) Repo git do Space + Git LFS para os pesos
cd "$DEPLOY"
git init -q
git lfs install --local
git lfs track "*.pt" >/dev/null
git add .gitattributes

# 4) Commit
git add -A
git -c user.email="deploy@local" -c user.name="deploy" commit -qm "Deploy: classificador de cartas (Flask + EfficientNet-B0, CPU)"
git branch -M main

# 5) Remote do Space e push
git remote remove space 2>/dev/null || true
git remote add space "$SPACE_URL"
echo ""
echo ">> Tudo pronto. Vou fazer o push para o Space."
echo ">> Quando pedir:  Username = $HF_USER   |   Password = SEU TOKEN (write)"
echo ""
git push -u space main

echo ""
echo ">> Pronto! Acompanhe o build em: $SPACE_URL"
echo ">> A primeira build leva alguns minutos (instala torch CPU)."
