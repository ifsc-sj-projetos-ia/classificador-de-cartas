# Classificador de Cartas de Baralho por Visão Computacional

Classificação supervisionada de uma **única carta de baralho já recortada** em
**53 classes** (52 cartas + 1 coringa), usando *transfer learning* com
**EfficientNet-B0** e comparação com um *baseline* clássico (HOG + Regressão
Logística).

> Projeto da disciplina **Introdução à Ciência de Dados (ICD)** — Curso de
> **Análise e Desenvolvimento de Sistemas (ADS)**. Autores: Nicolas; Herick.

---

## Objetivo

Reconhecer automaticamente qual carta aparece em uma imagem RGB de 224×224 px.
A motivação principal é **educacional** (ensino de probabilidade, regras de
jogos e matemática), com aplicações secundárias em **acessibilidade** (leitura
da carta em voz alta) e em **pesquisa/benchmark** de visão computacional.

A **detecção** de várias cartas em uma mesma cena está **fora do escopo** e é
tratada como trabalho futuro.

## Estrutura do repositório

```
.
├── src/            # código-fonte (data, train, evaluate, baseline, predict, config, seed)
├── notebooks/      # notebook de treino no Google Colab (GPU T4)
├── models/         # checkpoints treinados (.pt) e históricos de treino (history.json)
├── reports/        # métricas, matrizes de confusão e relatório
├── data/           # SOMENTE instruções de acesso aos dados (ver data/README.md)
├── docs/           # documentação de apoio (definição, dados, ética, model card)
├── requirements.txt
├── LICENSE         # MIT
└── README.md
```

> **Nota sobre o código:** os scripts de `src/` e o notebook de `notebooks/`
> devem ser adicionados aqui. As pastas já estão preparadas para recebê-los.

## Dependências

Ambiente de referência: **Google Colab** (GPU T4 gratuita), Python 3.

```bash
pip install -r requirements.txt
```

As versões estão pinadas em [`requirements.txt`](requirements.txt) para garantir
reprodutibilidade.

## Dados

- **Dataset principal:** *Cards Image Dataset-Classification* (autor *gpiosenka*,
  Kaggle) — 53 classes, imagens 224×224×3 já recortadas, split pronto
  **treino/validação/teste = 7.624 / 265 / 265** (5 imagens por classe em
  validação e teste).
- **Licença:** marcada como **"Other"** no Kaggle. Por isso **as imagens não são
  redistribuídas** neste repositório — veja [`data/README.md`](data/README.md)
  para as instruções de download.
- **Conjunto OOD** (avaliação de generalização): baralho de **design diferente**,
  derivado do *Vector Playing Cards* (domínio público) via *playing-cards-assets*
  (Howard Yeh, MIT). 53 classes / 54 imagens. Usado **apenas para avaliação**.

## Como reproduzir

```bash
# 1. Clonar e instalar
git clone <URL_DO_REPOSITORIO>
cd <repo>
pip install -r requirements.txt

# 2. Obter os dados conforme data/README.md (o dataset NÃO é redistribuído)

# 3. Garantir reprodutibilidade: set_seed(42) e caminhos em src/config.py

# 4. Baseline clássico
python -m src.baseline

# 5. Treino (EfficientNet-B0, transfer learning em 2 fases)
python -m src.train          # ou rode notebooks/ na GPU T4 do Colab

# 6. Avaliação (gera métricas e matrizes de confusão em reports/)
python -m src.evaluate

# 7. Predição em uma imagem única
python -m src.predict --img <caminho_da_imagem>
```

Todos os experimentos usam **semente fixa** (`set_seed(42)`), batch size 32,
imagem 224×224 e normalização ImageNet.

## Métricas principais

| Configuração | Acc. teste | F1-macro teste | Acc. OOD (design) |
|---|---|---|---|
| Baseline: HOG + Reg. Logística | 0,706 | 0,698 | — |
| EfficientNet-B0 — *feature extraction* (congelado) | 0,385 | 0,363 | — |
| **EfficientNet-B0 — *fine-tuning* (com aug.)** ⭐ | **0,947** | **0,947** | **0,593** |
| EfficientNet-B0 — *fine-tuning* sem augmentation | 0,974 | 0,973 | *(a medir)* |

> Os **checkpoints** (`models/**/*.pt`) e os **históricos de treino**
> (`history.json`) estão versionados no repositório para reprodução direta.

- O *fine-tuning* superou a *feature extraction* em **+56,2 pontos percentuais**.
- **Gap de design** (teste − OOD) ≈ **35 pontos percentuais**.
- No teste, o modelo acertou **251 de 265** imagens; erros são quase todos de
  **valor dentro do mesmo naipe** (nenhuma troca de naipe/cor).

Detalhes por classe em
[`reports/classification_report_test.csv`](reports/classification_report_test.csv)
e matrizes em
[`reports/confusion_matrix_test.png`](reports/confusion_matrix_test.png) /
[`reports/confusion_matrix_ood.png`](reports/confusion_matrix_ood.png).

## Limitações

- Escopo apenas de **classificação**; depende de a carta já estar recortada.
- Treinado em **um único design** de baralho → generaliza mal para outros
  estilos (gap OOD de ~35 pp).
- Validação e teste pequenos (**5 imagens/classe**), métricas sensíveis a poucos
  erros.
- O OOD avaliado é de **design** (imagens limpas), não de **captura real** — o
  gap real tende a ser maior.

## Uso responsável (importante)

⚠️ É um **uso PROIBIDO** deste projeto empregá-lo para assistência em tempo real
(*Real-Time Assistance*, RTA) em jogos de azar/apostas — prática vedada por
plataformas como PokerStars e GGPoker. Ver [`docs/etica.md`](docs/etica.md).

## Trabalhos futuros

- **Detecção com YOLO** (várias cartas em cena, sem recorte manual).
- Ampliar o OOD com mais designs e com **fotos de baralhos físicos** (gap de
  captura).
- Medir a acurácia OOD da variante **sem augmentation**.
- Empacotar o modo educacional/acessibilidade (leitura em voz alta, exercícios
  de probabilidade).

## Licença

Código sob licença **MIT** (ver [`LICENSE`](LICENSE)). Os **dados** seguem a
licença original do Kaggle ("Other") e **não** são redistribuídos aqui.
