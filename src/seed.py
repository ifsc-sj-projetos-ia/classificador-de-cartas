"""Reprodutibilidade: fixa as sementes de todas as fontes de aleatoriedade.

Chamar set_seed() ANTES de criar dataloaders e modelos. Em GPU, o modo
deterministico pode reduzir levemente a velocidade, mas garante resultados
reproduziveis (requisito de ciencia aberta do enunciado).
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Fixa as seeds de random, numpy e torch (CPU e GPU).

    Args:
        seed: valor da semente.
        deterministic: se True, forca operacoes deterministicas no cuDNN.

    Returns:
        A propria seed (conveniencia para logar/documentar).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        # torch pode nao estar instalado no ambiente que so roda o baseline.
        pass

    return seed
