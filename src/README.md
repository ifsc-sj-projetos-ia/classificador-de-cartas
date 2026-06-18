# src/ — código-fonte

Adicione aqui os módulos do projeto (vindos do Colab). Estrutura esperada,
conforme o relatório e o `README.md` da raiz:

| Arquivo | Responsabilidade |
|---|---|
| `config.py` | caminhos, hiperparâmetros, dispositivo (CPU/GPU) |
| `seed.py` | `set_seed(42)` para reprodutibilidade |
| `data.py` | datasets, transforms, normalização ImageNet e data augmentation |
| `baseline.py` | HOG + Regressão Logística (`python -m src.baseline`) |
| `train.py` | treino da EfficientNet-B0 em 2 fases (`python -m src.train`) |
| `evaluate.py` | métricas e matrizes de confusão em `reports/` (`python -m src.evaluate`) |
| `predict.py` | predição em imagem única (`python -m src.predict --img <caminho>`) |
| `ood_design.py` | montagem do conjunto OOD (baralho de design diferente) |

> Inclua um `__init__.py` (pode ser vazio) para permitir `python -m src.<modulo>`.

Quando colar os arquivos aqui, apague este README ou substitua-o pela
documentação real dos módulos.
