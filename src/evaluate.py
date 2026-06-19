"""Avaliacao: metricas, matriz de confusao e avaliacao OOD.

Reune as funcoes usadas tanto no treino (validacao) quanto na analise final
(item 2.4): accuracy, F1 macro, relatorio por classe e matriz de confusao.
Inclui suporte a avaliacao out-of-distribution (OOD) nas fotos do baralho real.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Roda o modelo sobre um loader e retorna (y_true, y_pred, y_prob)."""
    model.eval()
    model.to(device)
    y_true, y_pred, y_prob = [], [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        y_true.append(labels.numpy())
        y_pred.append(preds.cpu().numpy())
        y_prob.append(probs.cpu().numpy())
    return (
        np.concatenate(y_true),
        np.concatenate(y_pred),
        np.concatenate(y_prob),
    )


def compute_metrics(y_true, y_pred, class_names):
    """Calcula accuracy, F1 macro/ponderado e relatorio por classe."""
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    # labels fixos no espaco completo de classes: evita ValueError caso um split
    # nao contenha todas as classes (numero de labels precisa casar com target_names).
    labels = list(range(len(class_names)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "report": classification_report(
            y_true, y_pred, labels=labels, target_names=class_names, zero_division=0, output_dict=True
        ),
    }


def plot_confusion_matrix(y_true, y_pred, class_names, out_path=None, normalize=True, figsize=(16, 14)):
    """Plota (e opcionalmente salva) a matriz de confusao.

    Returns:
        Caminho salvo (str) ou None se nao salvou.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    # labels fixos no espaco completo de classes (robusto quando nem todas
    # as classes aparecem, como na avaliacao OOD).
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true" if normalize else None)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=figsize)
    disp.plot(ax=ax, xticks_rotation="vertical", colorbar=False, values_format=".2f" if normalize else "d")
    ax.set_title("Matriz de confusao" + (" (normalizada)" if normalize else ""))
    plt.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)
    return None


def top_confusions(y_true, y_pred, class_names, top_k=15):
    """Lista os pares (verdadeiro -> previsto) mais confundidos.

    Util para discutir 'confusoes perigosas' (trocar naipe/valor), relevante
    para o uso educacional onde uma leitura errada muda a regra do jogo.
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                pairs.append((class_names[i], class_names[j], int(cm[i, j])))
    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs[:top_k]


def evaluate_loader(model, loader, class_names, device, cm_path=None):
    """Avaliacao completa de um loader: metricas + matriz de confusao + confusoes."""
    y_true, y_pred, _ = collect_predictions(model, loader, device)
    metrics = compute_metrics(y_true, y_pred, class_names)
    if cm_path is not None:
        metrics["confusion_matrix_path"] = plot_confusion_matrix(
            y_true, y_pred, class_names, out_path=cm_path
        )
    metrics["top_confusions"] = top_confusions(y_true, y_pred, class_names)
    return metrics


def evaluate_ood(model, ood_dir, class_names, device, img_size=224, batch_size=32, cm_path=None):
    """Avalia o modelo num conjunto OOD (fotos do baralho real do usuario).

    A pasta ood_dir deve ter 1 subpasta por classe, com nomes IGUAIS aos do
    treino. NAO precisa ter todas as 53 classes: os rotulos sao remapeados para
    o espaco de classes do treino pelo NOME da subpasta, e as classes ausentes
    simplesmente nao entram nas metricas.
    """
    from sklearn.metrics import accuracy_score, f1_score

    from .data import build_eval_loader

    loader, ds = build_eval_loader(ood_dir, img_size=img_size, batch_size=batch_size, num_workers=0)

    name_to_train_idx = {name: i for i, name in enumerate(class_names)}
    unknown = [c for c in ds.classes if c not in name_to_train_idx]
    if unknown:
        raise ValueError(
            f"Subpastas OOD sem correspondencia nas classes de treino: {unknown}. "
            "Use exatamente os mesmos nomes de subpasta do dataset de treino."
        )
    ood_to_train = {ood_i: name_to_train_idx[name] for ood_i, name in enumerate(ds.classes)}

    y_true_ood, y_pred, _ = collect_predictions(model, loader, device)
    y_true = np.array([ood_to_train[int(y)] for y in y_true_ood])
    present = sorted(set(y_true.tolist()))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
        "n_images": int(len(y_true)),
        "n_classes_present": len(present),
        "top_confusions": top_confusions(y_true, y_pred, class_names),
    }
    if cm_path is not None:
        metrics["confusion_matrix_path"] = plot_confusion_matrix(
            y_true, y_pred, class_names, out_path=cm_path
        )
    return metrics
