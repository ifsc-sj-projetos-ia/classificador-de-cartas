"""Monta um SEGUNDO design de baralho para ADICIONAR AO TREINO.

Objetivo: reduzir o gap de DESIGN (OOD). Hoje o modelo treina num unico estilo
(gpiosenka) e por isso decora a arte em vez de aprender o conceito da carta —
acerta 94,7% no teste (mesmo design) mas so 59,3% num baralho de design
diferente. Misturar um segundo design no treino força o modelo a aprender o que
e invariante (valor + naipe), nao o estilo visual.

IMPORTANTE — nao "vazar" a avaliacao:
  - Este segundo design vai para o TREINO.
  - O baralho "Vector Playing Cards" (src/ood_design.py) continua sendo APENAS
    avaliacao OOD. Sao designs DIFERENTES entre si; nao reaproveitar um no outro,
    senao a metrica OOD perde o sentido.

Fonte deste segundo design: "Kenney Playing Cards Pack" (Kenney.nl), licenca
**CC0 / dominio publico** — uso livre, inclusive comercial, sem atribuicao
obrigatoria. As cartas vem como PNG individuais, design plano e limpo,
visualmente distinto tanto do gpiosenka quanto do Vector Playing Cards.

As 53 classes resultantes usam EXATAMENTE os mesmos nomes de pasta do dataset
de treino, para casar com data.ImageFolder.

Uso (linha de comando):
    # assets-dir = a pasta com os PNGs do baralho (ex.: "PNG/Cards (large)").
    # Arquivos esperados: card_<naipe>_<valor>.png (card_clubs_02.png, card_hearts_A.png...).
    python -m src.extra_designs --assets-dir "PNG/Cards (large)" --out data/raw/extra_kenney

    # depois, no treino, aponte o loader multi-design para ESTA pasta
    # (ver build_dataloaders(..., extra_train_dirs=[...]) em src/data.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Valor no nome do arquivo de origem (Kenney) -> palavra usada nas classes de treino.
# O pacote Kenney usa numeros com zero a esquerda (02..10) e letras A/J/Q/K.
VALUE_MAP = {
    "A": "ace", "02": "two", "03": "three", "04": "four", "05": "five",
    "06": "six", "07": "seven", "08": "eight", "09": "nine", "10": "ten",
    "J": "jack", "Q": "queen", "K": "king",
}
SUITS = ("clubs", "diamonds", "hearts", "spades")


def _planned_files():
    """Lista (arquivo_origem, nome_da_classe, nome_do_arquivo_destino).

    O pacote Kenney nomeia os arquivos como "card_<naipe>_<valor>.png",
    ex.: card_clubs_02.png, card_hearts_A.png, card_spades_K.png.
    """
    plan = []
    for val, word in VALUE_MAP.items():
        for suit in SUITS:
            src_name = f"card_{suit}_{val}.png"
            class_name = f"{word} of {suit}"
            dst_name = f"{word}_of_{suit}_kenney01.png"
            plan.append((src_name, class_name, dst_name))
    # O dataset de treino tem UMA classe coringa. O Kenney nao traz coringa em
    # todas as versoes; se faltar, o build apenas avisa (52 cartas ja bastam).
    plan.append(("card_joker_red.png", "joker", "joker_kenney01.png"))
    plan.append(("card_joker_black.png", "joker", "joker_kenney02.png"))
    return plan


def build_extra_design_set(out_dir: str, assets_dir: str) -> str:
    """Copia o segundo design para 53 pastas de classe (mesmos nomes do treino).

    Args:
        out_dir: pasta de saida (raiz do conjunto extra de treino).
        assets_dir: diretorio local com os PNGs do pacote (obrigatorio — sem
            download automatico, pois a fonte distribui em .zip).

    Returns:
        Caminho absoluto da pasta de saida.
    """
    src_assets = Path(assets_dir)
    if not src_assets.is_dir():
        raise FileNotFoundError(f"assets-dir nao encontrado: {src_assets}")

    out = Path(out_dir)
    plan = _planned_files()

    n_ok, n_missing = 0, []
    classes = set()
    for src_name, class_name, dst_name in plan:
        src_path = src_assets / src_name
        if not src_path.is_file():
            n_missing.append(src_name)
            continue
        class_dir = out / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        (class_dir / dst_name).write_bytes(src_path.read_bytes())
        classes.add(class_name)
        n_ok += 1

    (out / "_FONTE.txt").write_text(
        "Segundo design de baralho para TREINO (reduzir gap de design/OOD).\n"
        "Fonte: Kenney Playing Cards Pack (https://kenney.nl) — licenca CC0 / dominio publico.\n"
        "NAO confundir com o OOD 'Vector Playing Cards' (src/ood_design.py), que e so avaliacao.\n",
        encoding="utf-8",
    )

    print(f"Segundo design montado em: {out.resolve()}")
    print(f"  {len(classes)} classes | {n_ok} imagens copiadas")
    if n_missing:
        print(f"  AVISO: {len(n_missing)} arquivos nao encontrados em {src_assets}:")
        print("    " + ", ".join(n_missing[:8]) + (" ..." if len(n_missing) > 8 else ""))
        print("  Confira os nomes dos PNGs do pacote (ex.: card_clubs_02.png, card_hearts_A.png).")
    return str(out.resolve())


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Monta um 2o design de baralho (Kenney/CC0) para adicionar ao treino."
    )
    p.add_argument("--out", default="data/raw/extra_kenney", help="pasta de saida")
    p.add_argument("--assets-dir", required=True,
                   help="diretorio local com os PNGs do pacote Kenney")
    args = p.parse_args(argv)
    try:
        build_extra_design_set(args.out, args.assets_dir)
    except Exception as e:  # noqa: BLE001 — mensagem amigavel para uso em sala
        print(f"ERRO ao montar o segundo design: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
