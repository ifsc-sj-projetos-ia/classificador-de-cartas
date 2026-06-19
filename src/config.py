"""Configuracao central do projeto.

Centraliza caminhos e hiperparametros para garantir reprodutibilidade e
facilitar a variacao controlada nos experimentos (item 2.4 do enunciado).
Os scripts (train.py, evaluate.py) aceitam sobrescrever estes valores via CLI.
"""

from __future__ import annotations

from dataclasses import dataclass

# Estatisticas do ImageNet (modelos pre-treinados esperam esta normalizacao).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Backbones suportados em model.py.
SUPPORTED_BACKBONES = ("efficientnet_b0", "mobilenet_v3_small", "mobilenet_v3_large", "resnet18")


@dataclass
class Config:
    """Hiperparametros e caminhos do experimento."""

    # ----- Dados -----
    # Pasta com subpastas train/ valid/ test/, cada uma com 1 subpasta por classe.
    # Estrutura do dataset gpiosenka "Cards Image Dataset-Classification".
    data_dir: str = "data/raw/cards"
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    augment: bool = True  # data augmentation (de CAPTURA) no conjunto de treino
    # Augmentation de APARENCIA/DESIGN (hue/grayscale/posterize/sharpness): ataca o
    # gap de design que move a OOD. Compoe com 'augment'. Ver src/data.build_transforms.
    design_aug: bool = False
    # Pastas de designs ADICIONAIS misturadas SO no treino (reduz o gap de design).
    # Cada uma com 1 subpasta por classe (mesmos nomes). Ex.: ["data/raw/extra_kenney"].
    extra_train_dirs: tuple[str, ...] = ()

    # ----- Modelo -----
    backbone: str = "efficientnet_b0"
    pretrained: bool = True

    # ----- Treino em duas fases -----
    # Fase 1 (feature extraction): backbone congelado, treina so a cabeca.
    epochs_head: int = 8
    lr_head: float = 1e-3
    # Fase 2 (fine-tuning): descongela o backbone. LR de pico com cosine annealing
    # (decai ate ~0). 1e-5/5ep sub-treina; 3e-4/20ep converge perto do teto (~95%).
    finetune: bool = True
    epochs_finetune: int = 20
    lr_finetune: float = 3e-4

    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    early_stop_patience: int = 6  # epocas sem melhora na val antes de parar
    #   (val tem so 5 img/classe -> ruidosa; paciencia maior evita parada precoce)

    # ----- Geral -----
    seed: int = 42
    out_dir: str = "models"
    device: str = "auto"  # "auto" | "cpu" | "cuda"

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
