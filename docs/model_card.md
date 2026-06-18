# Model Card — EfficientNet-B0 (classificador de cartas)

## Visão geral
- **Arquitetura:** EfficientNet-B0 (torchvision), pré-treinada no ImageNet.
- **Tarefa:** classificação multiclasse de 1 carta recortada em 53 classes.
- **Entrada:** imagem RGB 224×224, normalização ImageNet.
- **Saída:** rótulo único entre 53 classes (52 cartas + coringa).

## Treino
- **Transfer learning em 2 fases:**
  - Fase 1 (*feature extraction*): backbone congelado, treina só a cabeça —
    8 épocas, AdamW, LR 1e-3.
  - Fase 2 (*fine-tuning*): backbone descongelado, AdamW com LR de pico 3e-4 +
    *cosine annealing*, até 20 épocas com early stopping (paciência 6); parou na
    época 12.
- **Infra:** Google Colab (GPU T4), batch size 32, `set_seed(42)`.

## Desempenho
| Métrica | Valor |
|---|---|
| Acurácia (teste Kaggle) | 0,947 |
| F1-macro (teste Kaggle) | 0,947 |
| Acurácia OOD (design diferente) | 0,593 |
| Acertos no teste | 251 / 265 |

- **Gap de design** (teste − OOD) ≈ 35 pp.
- Erros predominam em **valor dentro do mesmo naipe**; **sem** trocas de naipe/cor.
- Ponto fraco isolado: classe `joker` (recall 0,60).

## Limitações
- Um único design de baralho; depende de recorte prévio; val/test pequenos
  (5 img/classe); OOD de design, não de captura real.

## Uso responsável
Ver [`etica.md`](etica.md). **Uso proibido:** assistência em tempo real (RTA)
em jogos de azar/apostas.
