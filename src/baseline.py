"""Baseline classico: HOG + Regressao Logistica (scikit-learn).

Serve como linha de base barata (so CPU) para comparar com o transfer learning,
cumprindo o requisito de "baseline simples" do enunciado (item 2.4).

HOG (Histogram of Oriented Gradients) descreve a forma/bordas da carta;
a Regressao Logistica (ou SVM linear) classifica entre as 53 classes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def _list_samples(split_dir: str | Path):
    """Lista (caminho, indice_da_classe) e a lista ordenada de classes."""
    split_dir = Path(split_dir)
    classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    samples = []
    for c in classes:
        for p in sorted((split_dir / c).iterdir()):
            if p.suffix.lower() in _IMG_EXTS:
                samples.append((p, class_to_idx[c]))
    return samples, classes


def extract_hog_features(image_paths, img_size: int = 128):
    """Extrai descritores HOG de uma lista de caminhos de imagem.

    Returns:
        np.ndarray de shape (n_imagens, n_features).
    """
    from PIL import Image
    from skimage.feature import hog

    feats = []
    for path in image_paths:
        img = Image.open(path).convert("L").resize((img_size, img_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        f = hog(
            arr,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )
        feats.append(f)
    return np.asarray(feats, dtype=np.float32)


def run_baseline(
    data_dir: str | Path,
    img_size: int = 128,
    classifier: str = "logreg",
    max_per_class: int | None = None,
    seed: int = 42,
):
    """Treina e avalia o baseline HOG + classificador classico.

    Args:
        data_dir: pasta com train/ e test/ (uma subpasta por classe).
        img_size: tamanho para redimensionar antes do HOG.
        classifier: "logreg" (Regressao Logistica) ou "svm" (SVM linear).
        max_per_class: limita amostras por classe (acelera testes rapidos).
        seed: semente do classificador.

    Returns:
        dict com acuracia, F1 macro, classes e o objeto do classificador treinado.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    from .data import _resolve_split_dir  # reaproveita resolucao de nomes de split

    train_dir = _resolve_split_dir(data_dir, "train")
    test_dir = _resolve_split_dir(data_dir, "test")

    train_samples, classes = _list_samples(train_dir)
    test_samples, test_classes = _list_samples(test_dir)
    if classes != test_classes:
        raise ValueError("Classes diferentes entre train e test no baseline.")

    if max_per_class is not None:
        rng = np.random.default_rng(seed)
        by_class: dict[int, list] = {}
        for p, y in train_samples:
            by_class.setdefault(y, []).append((p, y))
        train_samples = []
        for y, items in by_class.items():
            idx = rng.permutation(len(items))[:max_per_class]
            train_samples.extend(items[i] for i in idx)

    X_train = extract_hog_features([p for p, _ in train_samples], img_size=img_size)
    y_train = np.array([y for _, y in train_samples])
    X_test = extract_hog_features([p for p, _ in test_samples], img_size=img_size)
    y_test = np.array([y for _, y in test_samples])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if classifier == "logreg":
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed, n_jobs=-1)
    elif classifier == "svm":
        clf = LinearSVC(C=1.0, random_state=seed)
    else:
        raise ValueError("classifier deve ser 'logreg' ou 'svm'.")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "classes": classes,
        "classifier": clf,
        "scaler": scaler,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Baseline HOG + classificador classico.")
    parser.add_argument("--data-dir", default="data/raw/cards")
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--classifier", choices=["logreg", "svm"], default="logreg")
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    res = run_baseline(
        data_dir=args.data_dir,
        img_size=args.img_size,
        classifier=args.classifier,
        max_per_class=args.max_per_class,
        seed=args.seed,
    )
    print(
        f"[baseline {args.classifier}] treino={res['n_train']} teste={res['n_test']} | "
        f"accuracy={res['accuracy']:.4f} | macro-F1={res['macro_f1']:.4f}"
    )
