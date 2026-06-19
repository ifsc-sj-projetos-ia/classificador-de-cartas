# Guia: reduzir o gap de design (OOD)

> Objetivo: subir a **acurácia OOD** (hoje **59,3%**) sem inflar artificialmente o
> teste in-distribution (já em 94,7%). O problema não é a acurácia de teste — é o
> modelo **decorar o design** do baralho de treino. Atacamos isso por dois eixos,
> ambos já implementados no código.

## Por que a augmentation atual não bastava

A augmentation original (rotação, brilho, *erasing*) simula variação de **captura**
(como a carta é fotografada). O OOD mede variação de **design** (arte/fontes/cor).
São gaps diferentes — por isso o relatório mostrou que ela **empatou** no OOD.

## As duas alavancas (implementadas)

### 1. Segundo design no treino  ⭐ (maior impacto)

Misturar um baralho de **design diferente** no treino força o modelo a aprender o
conceito (valor + naipe), não o estilo. **Não usar o mesmo design do OOD** (Vector
Playing Cards), senão a métrica perde o sentido.

```bash
# 1) Baixe o pacote Kenney Playing Cards (CC0/domínio público): https://kenney.nl
#    -> descompacte; os PNGs ficam num diretório (ex.: .../PNG-cards/).
# 2) Monte as 53 classes a partir desses PNGs:
python -m src.extra_designs --assets-dir /caminho/PNG-cards --out data/raw/extra_kenney
```

### 2. Augmentation de aparência/design

Acrescenta transformações que atacam a aparência: variação forte de matiz,
*grayscale* ocasional, posterização e nitidez. Liga com `--design-aug`.

## Como treinar (Colab, GPU T4)

> O treino **não roda** no Python local (3.14, libs pinadas incompatíveis). Use o
> notebook do Colab. As flags abaixo valem igual via `python -m src.train`.

```bash
# Modelo principal ATUAL (baseline a comparar): só captura, 1 design
python -m src.train

# NOVO: 2º design no treino + augmentation de design
python -m src.train --design-aug --extra-train-dir data/raw/extra_kenney \
    --out-dir models_ood

# (pode repetir --extra-train-dir para somar mais designs)
```

## Como medir (o teste honesto)

O ganho aparece no **OOD**, não no teste. Avalie sempre o OOD de design (Vector
Playing Cards), que continua **fora** do treino:

```bash
python -m src.ood_design --out data/raw/ood_design_web      # monta o OOD (se ainda não tiver)
python -m src.evaluate --ckpt models_ood/efficientnet_b0_best.pt \
    --ood-dir data/raw/ood_design_web
```

Compare a **acurácia OOD** antes (59,3%) e depois. Espera-se subida com o 2º design;
a augmentation de design sozinha ajuda menos (some-a ao 2º design).

## Cuidados (para a defesa)

- **Não vazar:** o design do OOD (Vector Cards) **nunca** entra no treino.
- **valid/test não mudam:** continuam no design original — então o teste
  in-distribution segue comparável aos experimentos anteriores.
- **Reprodutibilidade:** `set_seed(42)` mantido; só os dados/augmentation mudam.
- **Honestidade:** reportar OOD de **design** (imagens limpas). Fotos reais (gap de
  **captura**) seguem como trabalho futuro (`docs/guia_coleta_baralho_real.md`).
