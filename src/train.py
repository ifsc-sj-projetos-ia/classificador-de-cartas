"""Laco de treino: feature extraction + fine-tuning, com early stopping.

Salva o melhor checkpoint (pelo accuracy de validacao) contendo os pesos,
os nomes das classes, a configuracao e o historico de metricas — tudo o que
e preciso para reproduzir e avaliar depois.

Uso (linha de comando, ex. no Colab):
    python -m src.train --data-dir data/raw/cards --backbone efficientnet_b0 \
        --epochs-head 8 --epochs-finetune 5

Uso (programatico):
    from src.config import Config
    from src.train import train_model
    result = train_model(Config(data_dir="data/raw/cards"))
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

from .config import Config
from .data import build_dataloaders
from .model import build_model, count_trainable_params, set_backbone_trainable
from .seed import set_seed


def _run_epoch(model, loader, criterion, device, optimizer=None):
    """Roda uma epoca. Se optimizer for None, e modo avaliacao (sem grad)."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, correct, total = 0.0, 0, 0
    torch.set_grad_enabled(is_train)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if is_train:
            optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        if is_train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
    torch.set_grad_enabled(True)
    return total_loss / total, correct / total


def _train_phase(model, loaders, criterion, optimizer, device, epochs, patience, history, phase_name,
                 scheduler=None):
    """Treina uma fase com early stopping; retorna o melhor state_dict e acc.

    Se scheduler for fornecido, faz scheduler.step() ao fim de cada epoca.
    """
    best_acc, best_state, epochs_no_improve = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = _run_epoch(model, loaders["train"], criterion, device, optimizer)
        va_loss, va_acc = _run_epoch(model, loaders["valid"], criterion, device, optimizer=None)
        if scheduler is not None:
            scheduler.step()
        dt = time.time() - t0
        history.append(
            {
                "phase": phase_name,
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": va_loss,
                "val_acc": va_acc,
                "seconds": dt,
            }
        )
        print(
            f"[{phase_name}] epoca {epoch:02d}/{epochs} | "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} | {dt:.1f}s"
        )
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[{phase_name}] early stopping (sem melhora ha {patience} epocas).")
                break
    return best_state, best_acc


def train_model(cfg: Config):
    """Treina o modelo conforme a configuracao e salva o melhor checkpoint.

    Returns:
        dict com 'history', 'best_val_acc', 'class_names', 'checkpoint_path' e
        'model' (o nn.Module treinado, ja com os melhores pesos carregados).
    """
    set_seed(cfg.seed)
    device = cfg.resolve_device()
    print(f"Dispositivo: {device}")

    loaders, class_names = build_dataloaders(
        data_dir=cfg.data_dir,
        img_size=cfg.img_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        augment=cfg.augment,
        design_aug=cfg.design_aug,
        extra_train_dirs=list(cfg.extra_train_dirs),
    )
    num_classes = len(class_names)
    print(f"Classes: {num_classes} | treino={len(loaders['train'].dataset)} "
          f"val={len(loaders['valid'].dataset)} teste={len(loaders['test'].dataset)}")

    model = build_model(
        backbone=cfg.backbone,
        num_classes=num_classes,
        pretrained=cfg.pretrained,
        freeze_backbone=True,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    history: list[dict] = []

    # ---- Fase 1: feature extraction (backbone congelado) ----
    print(f"\nFase 1 - feature extraction | params treinaveis: {count_trainable_params(model):,}")
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.lr_head,
        weight_decay=cfg.weight_decay,
    )
    best_state, best_acc = _train_phase(
        model, loaders, criterion, optimizer, device,
        cfg.epochs_head, cfg.early_stop_patience, history, "head",
    )
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Fase 2: fine-tuning (descongela o backbone, LR baixo) ----
    if cfg.finetune and cfg.epochs_finetune > 0:
        set_backbone_trainable(model, True)
        print(f"\nFase 2 - fine-tuning | params treinaveis: {count_trainable_params(model):,}")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr_finetune, weight_decay=cfg.weight_decay
        )
        # Cosine annealing: LR alto no inicio (convergencia rapida) decaindo
        # suavemente ate ~0, para um ajuste fino estavel no fim do treino.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs_finetune)
        ft_state, ft_acc = _train_phase(
            model, loaders, criterion, optimizer, device,
            cfg.epochs_finetune, cfg.early_stop_patience, history, "finetune",
            scheduler=scheduler,
        )
        if ft_state is not None and ft_acc >= best_acc:
            best_state, best_acc = ft_state, ft_acc
        if best_state is not None:
            model.load_state_dict(best_state)

    # ---- Salva o melhor checkpoint + metadados ----
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{cfg.backbone}_best.pt"
    torch.save(
        {
            "state_dict": best_state if best_state is not None else model.state_dict(),
            "class_names": class_names,
            "config": asdict(cfg),
            "best_val_acc": best_acc,
        },
        ckpt_path,
    )
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"\nMelhor val_acc={best_acc:.4f} | checkpoint salvo em {ckpt_path}")

    return {
        "history": history,
        "best_val_acc": best_acc,
        "class_names": class_names,
        "checkpoint_path": str(ckpt_path),
        "model": model,
    }


def _build_cfg_from_args(args) -> Config:
    return Config(
        data_dir=args.data_dir,
        backbone=args.backbone,
        epochs_head=args.epochs_head,
        epochs_finetune=args.epochs_finetune,
        finetune=not args.no_finetune,
        augment=not args.no_augment,
        design_aug=args.design_aug,
        extra_train_dirs=tuple(args.extra_train_dir),
        batch_size=args.batch_size,
        img_size=args.img_size,
        lr_head=args.lr_head,
        lr_finetune=args.lr_finetune,
        seed=args.seed,
        out_dir=args.out_dir,
        device=args.device,
        num_workers=args.num_workers,
    )


def main():
    import argparse

    from .config import Config, SUPPORTED_BACKBONES

    _d = Config()  # defaults da CLI seguem o Config (evita divergir e re-introduzir undertraining)
    p = argparse.ArgumentParser(description="Treino do classificador de cartas (transfer learning).")
    p.add_argument("--data-dir", default=_d.data_dir)
    p.add_argument("--backbone", choices=SUPPORTED_BACKBONES, default=_d.backbone)
    p.add_argument("--epochs-head", type=int, default=_d.epochs_head)
    p.add_argument("--epochs-finetune", type=int, default=_d.epochs_finetune)
    p.add_argument("--no-finetune", action="store_true", help="treina so a cabeca (experimento 1).")
    p.add_argument("--no-augment", action="store_true", help="desliga data augmentation (experimento 2).")
    p.add_argument("--design-aug", action="store_true",
                   help="liga a augmentation de APARENCIA/DESIGN (hue/grayscale/posterize) p/ reduzir o gap OOD.")
    p.add_argument("--extra-train-dir", action="append", default=[], metavar="DIR",
                   help="pasta de um design EXTRA p/ misturar no treino (pode repetir). Ex.: data/raw/extra_kenney")
    p.add_argument("--batch-size", type=int, default=_d.batch_size)
    p.add_argument("--img-size", type=int, default=_d.img_size)
    p.add_argument("--lr-head", type=float, default=_d.lr_head)
    p.add_argument("--lr-finetune", type=float, default=_d.lr_finetune)
    p.add_argument("--seed", type=int, default=_d.seed)
    p.add_argument("--out-dir", default=_d.out_dir)
    p.add_argument("--device", default=_d.device, choices=["auto", "cpu", "cuda"])
    p.add_argument("--num-workers", type=int, default=_d.num_workers)
    args = p.parse_args()

    cfg = _build_cfg_from_args(args)
    train_model(cfg)


if __name__ == "__main__":
    main()
