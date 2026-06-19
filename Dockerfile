# Imagem de deploy para Hugging Face Spaces (SDK: docker).
# Roda o frontend Flask (src/app.py) com o modelo ja treinado (models/*.pt) em CPU.
#
# O HF Spaces espera o app na porta de app_port (7860, no README). Fixamos
# PORT=7860 e HOST=0.0.0.0 abaixo; o app.py le ambos do ambiente.
FROM python:3.11-slim

# Evita prompts e deixa o stdout sem buffer (logs aparecem na hora no Space).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=7860
# HF Spaces (docker) NAO injeta PORT — ele espera o app na porta de app_port
# (7860, declarada no README). Por isso fixamos PORT=7860 aqui no container.

WORKDIR /app

# 1) Dependencias primeiro (cache de camada). Usamos um requirements enxuto de
#    deploy: torch/torchvision CPU-only (muito menores que a versao CUDA) + flask
#    + pillow. NAO instala scikit-learn/matplotlib/etc. — a inferencia nao precisa.
COPY requirements-deploy.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements-deploy.txt

# 2) Codigo + modelo + frontend.
COPY src/ ./src/
COPY models/ ./models/
COPY reports/ ./reports/

EXPOSE 7860

# PYTHONPATH=. para que "python -m src.app" ache o pacote src.
ENV PYTHONPATH=.
CMD ["python", "-m", "src.app"]
