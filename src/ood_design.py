"""Monta o conjunto de avaliacao OOD "design diferente" (imagens limpas da web).

Este conjunto NAO sao fotos de um baralho real do mundo real. Sao renders
digitais limpos de um baralho de **design diferente** do dataset de treino
(gpiosenka), usados para medir um tipo especifico de generalizacao: o
**gap de design** (o modelo aprendeu o conceito da carta ou decorou o estilo
visual do dataset de treino?). Ele NAO mede o gap de condicoes de captura
(iluminacao, sombra, fundo, angulo) — para isso so fotos reais servem, e isso
fica registrado como trabalho futuro (ver docs/guia_coleta_baralho_real.md).

Fonte dos assets: repositorio "playing-cards-assets" (Howard Yeh, licenca MIT),
cujas cartas derivam do projeto "Vector Playing Cards" (dominio publico).
  https://github.com/hayeah/playing-cards-assets

As 53 classes resultantes usam EXATAMENTE os mesmos nomes de pasta do dataset
de treino, para que src.evaluate.evaluate_ood case as predicoes com os rotulos.

Uso (linha de comando):
    python -m src.ood_design --out data/raw/ood_design_web
    # ou, se ja tiver os PNGs localmente (pula o download):
    python -m src.ood_design --out data/raw/ood_design_web --assets-dir /caminho/png
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

# URL base dos PNGs individuais (branch master do repo de assets).
RAW_BASE = "https://raw.githubusercontent.com/hayeah/playing-cards-assets/master/png"

# Mapeia o valor no nome do arquivo de origem -> palavra usada nas classes de treino.
VALUE_MAP = {
    "ace": "ace", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
    "jack": "jack", "queen": "queen", "king": "king",
}
SUITS = ("clubs", "diamonds", "hearts", "spades")


def _planned_files():
    """Lista (arquivo_origem, nome_da_classe, nome_do_arquivo_destino)."""
    plan = []
    for src_val, word in VALUE_MAP.items():
        for suit in SUITS:
            src_name = f"{src_val}_of_{suit}.png"
            class_name = f"{word} of {suit}"
            dst_name = f"{word}_of_{suit}_web01.png"
            plan.append((src_name, class_name, dst_name))
    # O dataset de treino tem UMA classe coringa ("joker"); aproveitamos os dois
    # coringas (preto e vermelho) como 2 imagens dentro da mesma pasta joker/.
    plan.append(("black_joker.png", "joker", "joker_web01.png"))
    plan.append(("red_joker.png", "joker", "joker_web02.png"))
    return plan


def _fetch_bytes(src_name: str, assets_dir: Path | None) -> bytes:
    if assets_dir is not None:
        return (assets_dir / src_name).read_bytes()
    url = f"{RAW_BASE}/{src_name}"
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (URL fixa, confiavel)
        return resp.read()


def build_ood_design_set(out_dir: str = "data/raw/ood_design_web",
                         assets_dir: str | None = None) -> str:
    """Baixa/copia o baralho de design diferente e monta as 53 pastas de classe.

    Args:
        out_dir: pasta de saida (raiz do conjunto OOD).
        assets_dir: se informado, le os PNGs deste diretorio local em vez de baixar.

    Returns:
        Caminho absoluto da pasta de saida.
    """
    out = Path(out_dir)
    src_assets = Path(assets_dir) if assets_dir else None
    plan = _planned_files()

    n_ok = 0
    classes = set()
    for src_name, class_name, dst_name in plan:
        data = _fetch_bytes(src_name, src_assets)
        class_dir = out / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        (class_dir / dst_name).write_bytes(data)
        classes.add(class_name)
        n_ok += 1

    # Registra a procedencia/atribuicao junto das imagens.
    (out / "_FONTE.txt").write_text(
        "Conjunto OOD 'design diferente' (imagens limpas da web) — NAO sao fotos reais.\n"
        "Fonte: https://github.com/hayeah/playing-cards-assets (MIT, Howard Yeh)\n"
        "Arte das cartas: projeto 'Vector Playing Cards' (dominio publico).\n"
        "Uso estritamente academico/diagnostico (mede o gap de DESIGN, nao o de captura).\n",
        encoding="utf-8",
    )

    print(f"OOD 'design diferente' montado em: {out.resolve()}")
    print(f"  {len(classes)} classes | {n_ok} imagens copiadas")
    return str(out.resolve())


def main(argv=None):
    p = argparse.ArgumentParser(description="Monta o conjunto OOD 'design diferente' (web).")
    p.add_argument("--out", default="data/raw/ood_design_web", help="pasta de saida")
    p.add_argument("--assets-dir", default=None,
                   help="diretorio local com os PNGs (pula o download)")
    args = p.parse_args(argv)
    try:
        build_ood_design_set(args.out, args.assets_dir)
    except Exception as e:  # noqa: BLE001 — mensagem amigavel para uso em sala
        print(f"ERRO ao montar o conjunto OOD: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
