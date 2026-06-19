# Dados

> As imagens **não** são versionadas no Git (ver `.gitignore`). Esta pasta guarda
> apenas as instruções de download, seguindo o princípio de ciência aberta
> ("dados sensíveis ou pesados não devem ir direto ao repositório").

## Dataset principal — Classificação (Kaggle)

**Cards Image Dataset-Classification** (autor: gpiosenka)
<https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification>

- **53 classes** = 52 cartas (13 valores × 4 naipes) + 1 coringa (*joker*).
- Imagens **224×224×3** (JPG), já recortadas (carta ocupa > 50% da imagem).
- Split pronto: **7.624 treino / 265 validação / 265 teste** (5 imagens por classe em val/teste).
- Estrutura: `train/`, `valid/`, `test/`, cada uma com 53 subpastas (uma por classe), + `cards.csv`.
- Licença: marcada como *"Other"* no Kaggle — **confirme a licença na página** antes de redistribuir.

### Como baixar

**Opção A — kagglehub (recomendada no Colab):**
```python
import kagglehub
path = kagglehub.dataset_download("gpiosenka/cards-image-datasetclassification")
print(path)  # use este caminho como data_dir
```

**Opção B — Kaggle API (kaggle.json):**
1. Em <https://www.kaggle.com/settings> → *Create New API Token* (baixa `kaggle.json`).
2. No Colab, faça upload do `kaggle.json` e rode:
```bash
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d gpiosenka/cards-image-datasetclassification -p data/raw --unzip
```

Após baixar, aponte `data_dir` para a pasta que contém `train/`, `valid/` e `test/`
(por padrão o projeto usa `data/raw/cards`).

## Conjunto de teste OOD — baralho de "design diferente" (web)

Para o experimento de generalização (out-of-distribution) desta entrega, usamos um
**baralho de design diferente** do de treino, montado a partir de **imagens limpas de
licença livre**. Ele é gerado de forma reprodutível (não versionamos as imagens):

```bash
python -m src.ood_design --out data/raw/ood_design_web
```

Isso cria `data/raw/ood_design_web/` com **uma subpasta por classe** usando **exatamente
os mesmos nomes** das subpastas do Kaggle (ex.: `ace of spades/`, `king of hearts/`, `joker/`),
prontas para `src.evaluate.evaluate_ood`.

- **O que mede:** o **gap de design** (estilo de carta diferente do treino) — não o gap de
  condições de captura (luz/sombra/fundo de fotos reais). Detalhes: [`../docs/guia_ood_design_web.md`](../docs/guia_ood_design_web.md).
- **Fonte/licença:** baralho derivado de *Vector Playing Cards* (domínio público), via
  *playing-cards-assets* (Howard Yeh, MIT). **Não** são fotos dos autores.

> **Trabalho futuro — OOD com fotos reais (gap de captura):** fotografar um baralho físico
> próprio é o padrão-ouro e mede um gap que as imagens da web não cobrem. Guia:
> [`../docs/guia_coleta_baralho_real.md`](../docs/guia_coleta_baralho_real.md). Nesse caso, use
> a pasta `data/raw/ood_baralho_real/` e ajuste `OOD_DIR` no notebook.
