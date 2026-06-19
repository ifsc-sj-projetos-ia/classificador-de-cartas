"""Pacote de codigo-fonte do projeto de reconhecimento de cartas de baralho.

Modulos:
- config:   configuracao central (caminhos, hiperparametros).
- seed:     reprodutibilidade (set_seed).
- data:     carregamento e transformacoes do dataset (PyTorch ImageFolder).
- model:    construcao dos modelos de transfer learning (EfficientNet/MobileNet/ResNet).
- baseline: baseline classico (HOG + Regressao Logistica) com scikit-learn.
- train:    laco de treino (feature extraction + fine-tuning) com early stopping.
- evaluate: metricas, matriz de confusao e avaliacao OOD.
- predict:  inferencia em uma imagem (uso assistivo/educacional).
"""

__version__ = "0.1.0"
