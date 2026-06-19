"""Servidor web (Flask) do classificador de cartas.

Serve um frontend estatico (src/frontend/) e expoe uma API que reaproveita o
mesmo pipeline de inferencia da linha de comando (src/predict.py). O modelo ja
treinado (models/efficientnet_b0_best.pt) e carregado uma vez, numa thread, ao
subir o servidor — entao a primeira predicao nao espera o load.

Uso:
    python -m src.app
    # abra http://localhost:5000

Endpoints:
    GET  /              -> frontend (index.html)
    GET  /api/status    -> {ready, message, n_classes}
    GET  /api/stats     -> metricas do modelo/dataset (experimentos, por classe)
    POST /api/predict   -> recebe imagem (multipart 'image'), devolve top-k
"""

from __future__ import annotations

import csv
import io
import threading
from pathlib import Path

from flask import Flask, jsonify, request

from .config import Config
from .predict import load_model, predict_image

# Raiz do projeto (este arquivo fica em src/). Usado para achar models/ e reports/.
ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "src" / "frontend"
DEFAULT_CHECKPOINT = ROOT / "models" / "efficientnet_b0_best.pt"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")

# Estado global do modelo, preenchido pela thread de init.
_state: dict = {"model": None, "class_names": None, "cfg": None}
_init_status = {"ready": False, "message": "Carregando modelo..."}


def _run_init():
    """Carrega o checkpoint uma vez, em background, ao subir o servidor."""
    try:
        if not DEFAULT_CHECKPOINT.exists():
            _init_status["message"] = (
                f"Checkpoint nao encontrado em {DEFAULT_CHECKPOINT}. "
                "Treine o modelo (ver README) ou copie o .pt para models/."
            )
            return
        model, class_names, cfg = load_model(DEFAULT_CHECKPOINT, device="cpu")
        _state.update(model=model, class_names=class_names, cfg=cfg)
        _init_status["ready"] = True
        _init_status["message"] = f"Pronto — {len(class_names)} classes carregadas."
        print(_init_status["message"])
    except Exception as e:  # noqa: BLE001 - queremos reportar qualquer erro de init na UI
        _init_status["message"] = f"Erro ao carregar modelo: {e}"
        print(_init_status["message"])


threading.Thread(target=_run_init, daemon=True).start()


@app.route("/")
def serve_index():
    return app.send_static_file("index.html")


@app.route("/api/status", methods=["GET"])
def status():
    n = len(_state["class_names"]) if _state["class_names"] else 0
    return jsonify(
        {"ready": _init_status["ready"], "message": _init_status["message"], "n_classes": n}
    )


def _read_experiments() -> list[dict]:
    """Le reports/experimentos.csv (val/test acc por experimento)."""
    path = ROOT / "reports" / "experimentos.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [
            {
                "experimento": r["experimento"],
                "val_acc": float(r["val_acc"]),
                "test_acc": float(r["test_acc"]),
                "test_macroF1": float(r["test_macroF1"]),
            }
            for r in csv.DictReader(f)
        ]


def _worst_classes(top_k: int = 5) -> list[dict]:
    """Classes com menor F1 no teste (de reports/classification_report_test.csv)."""
    path = ROOT / "reports" / "classification_report_test.csv"
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            name = r[""]
            if name in ("accuracy", "macro avg", "weighted avg", ""):
                continue
            rows.append({"classe": name, "f1": float(r["f1-score"]), "support": float(r["support"])})
    rows.sort(key=lambda x: x["f1"])
    return rows[:top_k]


@app.route("/api/stats", methods=["GET"])
def stats():
    """Metricas para a pagina inicial: experimentos + piores classes + modelo principal."""
    experiments = _read_experiments()
    # O modelo principal e o "FE + fine-tuning" (com augmentation).
    main = next((e for e in experiments if e["experimento"] == "FE + fine-tuning"), None)
    return jsonify(
        {
            "n_classes": len(_state["class_names"]) if _state["class_names"] else 53,
            "backbone": _state["cfg"].backbone if _state["cfg"] else "efficientnet_b0",
            "train_size": 7624,
            "val_size": 265,
            "test_size": 265,
            "main_model": main,
            "experiments": experiments,
            "worst_classes": _worst_classes(),
            "ood_acc": 0.593,  # baralho de design diferente (gap de design ~35pp)
        }
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    if not _init_status["ready"]:
        return jsonify({"error": _init_status["message"]}), 503
    if "image" not in request.files:
        return jsonify({"error": "Nenhuma imagem enviada (campo 'image')."}), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Arquivo vazio."}), 400

    try:
        # predict_image abre via PIL; passamos o stream em memoria (BytesIO).
        raw = file.read()
        cfg: Config = _state["cfg"]
        preds = predict_image(
            _state["model"], io.BytesIO(raw), _state["class_names"], cfg, device="cpu", topk=5
        )
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"Falha na predicao: {e}"}), 400

    return jsonify(
        {
            "top": [{"classe": name, "prob": prob} for name, prob in preds],
            "best": {"classe": preds[0][0], "prob": preds[0][1]},
        }
    )


def main():
    import os

    # PORT/HOST via env: local usa 5000; Hugging Face Spaces injeta PORT=7860.
    port = int(os.environ.get("PORT", "5000"))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"Servidor em http://{host}:{port}  (Ctrl+C para parar)")
    app.run(debug=False, host=host, port=port, use_reloader=False)


if __name__ == "__main__":
    main()
