"""Inferencia em uma unica imagem (uso educacional/assistivo).

Carrega um checkpoint treinado e preve a carta de uma foto, retornando as
top-k classes com probabilidade. Pensado para o cenario de uso: o usuario
aponta a camera para UMA carta e o sistema diz qual e (poderia ser lido em
voz alta por um leitor de tela).
"""

from __future__ import annotations

from pathlib import Path

import torch

from .config import Config
from .data import build_transforms
from .model import build_model


def load_model(checkpoint_path: str | Path, device: str = "auto"):
    """Carrega modelo + nomes de classes a partir de um checkpoint salvo por train.py.

    Returns:
        (model, class_names, cfg) prontos para inferencia.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_names = ckpt["class_names"]
    cfg = Config(**ckpt["config"]) if isinstance(ckpt.get("config"), dict) else Config()
    dev = cfg.resolve_device() if device == "auto" else device

    model = build_model(
        backbone=cfg.backbone,
        num_classes=len(class_names),
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(dev)
    return model, class_names, cfg


@torch.no_grad()
def predict_image(model, image_path, class_names, cfg: Config, device: str = "auto", topk: int = 3):
    """Preve a carta de uma imagem. Retorna lista de (classe, probabilidade)."""
    from PIL import Image

    dev = cfg.resolve_device() if device == "auto" else device
    _, eval_tf = build_transforms(img_size=cfg.img_size, augment=False)
    img = Image.open(image_path).convert("RGB")
    x = eval_tf(img).unsqueeze(0).to(dev)

    probs = torch.softmax(model(x), dim=1).squeeze(0)
    k = min(topk, len(class_names))
    top_p, top_i = probs.topk(k)
    return [(class_names[i], float(p)) for p, i in zip(top_p.tolist(), top_i.tolist())]


def main():
    import argparse

    p = argparse.ArgumentParser(description="Preve a carta de uma imagem.")
    p.add_argument("image", help="caminho da imagem da carta")
    p.add_argument("--checkpoint", default="models/efficientnet_b0_best.pt")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    model, class_names, cfg = load_model(args.checkpoint, device=args.device)
    preds = predict_image(model, args.image, class_names, cfg, device=args.device, topk=args.topk)
    print(f"Imagem: {args.image}")
    for rank, (name, prob) in enumerate(preds, 1):
        print(f"  {rank}. {name:<20} {prob * 100:5.1f}%")


if __name__ == "__main__":
    main()
