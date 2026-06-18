# Dados e preparação

## Dataset principal (Kaggle)
- **Nome:** Cards Image Dataset-Classification (autor *gpiosenka*).
- **Classes:** 53 (52 cartas + 1 coringa). **Imagens:** 224×224×3 RGB, recortadas.
- **Split:** treino/validação/teste = **7.624 / 265 / 265** (5 img/classe em val/test).
- **Balanceamento:** ~144 img/classe em média; F1-macro = accuracy no teste,
  indicando desempenho homogêneo (sem necessidade de *class weights*).
- **Licença:** "Other" (Kaggle) → **não redistribuído** aqui (ver
  [`../data/README.md`](../data/README.md)).

## Conjunto OOD (avaliação de generalização)
- **Origem:** *Vector Playing Cards* (domínio público) via *playing-cards-assets*
  (Howard Yeh, MIT). 53 classes / 54 imagens (renders limpos).
- **Mede:** gap de **design** (estilo diferente do treino).
- **Não mede:** gap de **captura** (luz, sombra, ângulo de fotos reais) → o valor
  observado é um **limite inferior** do gap real. Uso **só para avaliação**.

## Pré-processamento e augmentation
- Resize 224×224 + normalização ImageNet (médias/desvios padrão).
- Augmentation (apenas no experimento principal): flip horizontal, pequenas
  rotações, *color jitter*. Validação/teste sem augmentation.

## Vieses e limitações
- Treino em **um único design** → baixa generalização para outros estilos
  (gap OOD ~35 pp).
- Conjuntos de val/test pequenos (5 img/classe) → métricas sensíveis a poucos erros.
