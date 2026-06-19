"""Carregamento de dados e transformacoes (PyTorch).

Espera a estrutura do dataset gpiosenka "Cards Image Dataset-Classification":

    data_dir/
      train/   <53 subpastas, 1 por classe>
      valid/   (ou 'validation' / 'val')
      test/

Cada subpasta contem as imagens daquela classe. Os nomes das classes sao
derivados automaticamente das subpastas (nao ha rotulos hardcoded).
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms

from .config import IMAGENET_MEAN, IMAGENET_STD

# Nomes alternativos aceitos para cada split (datasets variam a nomenclatura).
_SPLIT_ALIASES = {
    "train": ("train", "training"),
    "valid": ("valid", "validation", "val"),
    "test": ("test", "testing"),
}


def _resolve_split_dir(data_dir: str | Path, split: str) -> Path:
    """Encontra a pasta do split aceitando nomes alternativos."""
    base = Path(data_dir)
    for alias in _SPLIT_ALIASES[split]:
        candidate = base / alias
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Nao encontrei a pasta do split '{split}' em {base}. "
        f"Esperado um destes nomes: {_SPLIT_ALIASES[split]}."
    )


def build_transforms(img_size: int = 224, augment: bool = True, design_aug: bool = False):
    """Cria as transformacoes de treino (com/sem augmentation) e de avaliacao.

    Args:
        augment: augmentation de CAPTURA (rotacao, brilho, translacao, oclusao).
            Simula variacao de como a carta e fotografada. E a augmentation original.
        design_aug: augmentation de APARENCIA/DESIGN. Acrescenta transformacoes que
            atacam o gap de DESIGN (arte/cor/fonte) — o que move a OOD de design:
            variacao forte de matiz, grayscale ocasional, posterizacao e nitidez.
            So tem efeito quando augment=True (compoe com a de captura).
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    eval_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    if augment:
        pre = [
            transforms.Resize((img_size, img_size)),
            # --- Augmentation de CAPTURA (original) ---
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        ]
        if design_aug:
            # --- Augmentation de APARENCIA/DESIGN ---
            # Forca invariancia a cor/textura de arte: a carta deve ser reconhecida
            # pelo VALOR+NAIPE, nao pela paleta/estilo do baralho de treino.
            pre += [
                transforms.ColorJitter(hue=0.5),                 # remapeia matizes
                transforms.RandomGrayscale(p=0.20),              # ignora a cor
                transforms.RandomPosterize(bits=4, p=0.30),      # reduz gradacoes (arte "chapada")
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.30),
            ]
        post = [
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.10)),
        ]
        train_tf = transforms.Compose(pre + post)
    else:
        train_tf = eval_tf

    return train_tf, eval_tf


def build_dataloaders(
    data_dir: str | Path,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
    augment: bool = True,
    design_aug: bool = False,
    extra_train_dirs: list[str | Path] | None = None,
):
    """Constroi os DataLoaders de train/valid/test.

    Args:
        design_aug: liga a augmentation de APARENCIA/DESIGN no treino (ver
            build_transforms) — ataca o gap de design que move a OOD.
        extra_train_dirs: pastas de designs ADICIONAIS para misturar SO no treino
            (cada uma com 1 subpasta por classe, mesmos nomes). Reduz o gap de
            design ao ensinar o modelo com mais de um estilo. valid/test NAO mudam
            (continuam medindo o design original), e o OOD segue separado.

    Returns:
        (loaders, class_names) onde loaders e um dict com chaves
        'train', 'valid', 'test' e class_names e a lista ordenada de classes.
    """
    train_tf, eval_tf = build_transforms(img_size=img_size, augment=augment, design_aug=design_aug)

    train_ds = datasets.ImageFolder(_resolve_split_dir(data_dir, "train"), transform=train_tf)
    valid_ds = datasets.ImageFolder(_resolve_split_dir(data_dir, "valid"), transform=eval_tf)
    test_ds = datasets.ImageFolder(_resolve_split_dir(data_dir, "test"), transform=eval_tf)

    # Sanidade: as classes devem ser as mesmas (mesma ordem) nos tres splits.
    if not (train_ds.classes == valid_ds.classes == test_ds.classes):
        raise ValueError(
            "As classes diferem entre os splits train/valid/test. "
            "Verifique se as subpastas (uma por classe) sao identicas em cada split."
        )

    class_names = train_ds.classes
    train_dataset = train_ds
    # Mistura designs extras no TREINO (mesma augmentation), validando as classes.
    if extra_train_dirs:
        parts = [train_ds]
        for extra in extra_train_dirs:
            extra_ds = datasets.ImageFolder(extra, transform=train_tf)
            if extra_ds.classes != class_names:
                raise ValueError(
                    f"As classes do design extra em '{extra}' diferem do treino.\n"
                    f"  treino : {class_names}\n  extra  : {extra_ds.classes}\n"
                    "Garanta 1 subpasta por classe com EXATAMENTE os mesmos nomes."
                )
            parts.append(extra_ds)
            print(f"  + design extra: {extra} ({len(extra_ds)} imagens)")
        train_dataset = ConcatDataset(parts)

    loaders = {
        "train": DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        ),
        "valid": DataLoader(
            valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        ),
        "test": DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        ),
    }
    return loaders, class_names


def build_eval_loader(
    data_dir: str | Path,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
):
    """DataLoader de avaliacao para uma pasta arbitraria (ex.: teste OOD).

    A pasta deve conter 1 subpasta por classe (mesmos nomes do treino).
    Returns:
        (loader, dataset) — o dataset expoe .classes e .samples.
    """
    _, eval_tf = build_transforms(img_size=img_size, augment=False)
    ds = datasets.ImageFolder(data_dir, transform=eval_tf)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return loader, ds
