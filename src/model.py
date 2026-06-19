"""Construcao dos modelos de transfer learning.

Usa backbones pre-treinados no ImageNet (torchvision) e troca a camada de
classificacao final por uma nova cabeca para as N classes de cartas.

Estrategia (recomendada para datasets pequenos/medios):
  Fase 1 - feature extraction: backbone congelado, treina so a cabeca.
  Fase 2 - fine-tuning: descongela o backbone com learning rate baixo.
"""

from __future__ import annotations

import torch.nn as nn

from .config import SUPPORTED_BACKBONES


def _replace_head(model: nn.Module, backbone: str, num_classes: int, dropout: float) -> nn.Module:
    """Substitui a camada de classificacao final conforme a arquitetura."""
    if backbone == "efficientnet_b0":
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )
    elif backbone in ("mobilenet_v3_small", "mobilenet_v3_large"):
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif backbone == "resnet18":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:  # pragma: no cover - validado antes
        raise ValueError(f"Backbone nao suportado: {backbone}")
    return model


def build_model(
    backbone: str = "efficientnet_b0",
    num_classes: int = 53,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    """Cria o modelo com pesos pre-treinados e nova cabeca de classificacao.

    Args:
        backbone: ver SUPPORTED_BACKBONES.
        num_classes: numero de classes (53 = 52 cartas + coringa).
        pretrained: usa pesos do ImageNet.
        freeze_backbone: se True, congela tudo menos a cabeca (feature extraction).
        dropout: dropout antes da camada linear final.
    """
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(f"Backbone '{backbone}' invalido. Use um de {SUPPORTED_BACKBONES}.")

    # Import tardio para nao exigir torchvision em quem so roda o baseline.
    from torchvision import models

    weights_arg = "IMAGENET1K_V1" if pretrained else None
    factory = {
        "efficientnet_b0": models.efficientnet_b0,
        "mobilenet_v3_small": models.mobilenet_v3_small,
        "mobilenet_v3_large": models.mobilenet_v3_large,
        "resnet18": models.resnet18,
    }[backbone]
    model = factory(weights=weights_arg)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model = _replace_head(model, backbone, num_classes, dropout)
    # A nova cabeca sempre treina (requires_grad=True por padrao em modulos novos).
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool = True) -> None:
    """(Des)congela todos os parametros do modelo para a fase de fine-tuning."""
    for param in model.parameters():
        param.requires_grad = trainable


def count_trainable_params(model: nn.Module) -> int:
    """Numero de parametros treinaveis (util para logar nos experimentos)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
