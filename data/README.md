# Dados

⚠️ **As imagens NÃO são redistribuídas neste repositório.** A licença do dataset
principal está marcada como **"Other"** no Kaggle; redistribuir as imagens
exigiria confirmação explícita dessa licença. Por isso, esta pasta contém
**apenas instruções de acesso**.

## Dataset principal — Cards Image Dataset-Classification (Kaggle)

- **Autor:** gpiosenka
- **Conteúdo:** 53 classes (52 cartas + 1 coringa), imagens 224×224×3 RGB já
  recortadas.
- **Split pronto:** treino/validação/teste = **7.624 / 265 / 265**
  (5 imagens por classe em validação e teste).
- **Download:** https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification

### Como baixar

Pela API do Kaggle (requer `~/.kaggle/kaggle.json` configurado):

```bash
pip install kaggle
kaggle datasets download -d gpiosenka/cards-image-datasetclassification
unzip cards-image-datasetclassification.zip -d data/cards/
```

Estrutura esperada após extrair (ajuste conforme `src/config.py`):

```
data/cards/
├── train/   # 7.624 imagens, subpastas por classe
├── valid/   # 265 imagens
└── test/    # 265 imagens
```

## Conjunto OOD — baralho de design diferente (apenas avaliação)

- **Origem:** *Vector Playing Cards* (domínio público), via repositório
  *playing-cards-assets* (Howard Yeh, licença MIT) —
  https://github.com/notpeter/playing-cards-assets
- **Conteúdo:** 53 classes / 54 imagens (1 por carta + 2 coringas na classe
  `joker`). São **renders limpos**, não fotos reais.
- **Uso:** **exclusivamente** para avaliar generalização (gap de design).
  **Nunca** usar para treino.

O script de montagem do conjunto OOD é o `src/ood_design.py`.
