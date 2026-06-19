# Classificador de Cartas de Baralho por Visão Computacional

Classificação supervisionada de uma **única carta de baralho já recortada** em
**53 classes** (52 cartas + 1 coringa), usando *transfer learning* com
**EfficientNet-B0** e comparação com um *baseline* clássico (HOG + Regressão
Logística).

> Projeto da disciplina **Introdução à Ciência de Dados (ICD)** — Curso de
> **Análise e Desenvolvimento de Sistemas (ADS)**. Autores: Nicolas; Herick.

🔗 **Demonstração ao vivo:** <https://herickhuggingfaceaccount-cartas.hf.space>
(hospedada no Hugging Face Spaces; abra no computador ou no celular)

---

## Objetivo

Reconhecer automaticamente qual carta aparece em uma imagem RGB de 224×224 px.
A motivação principal é **educacional** (ensino de probabilidade, regras de
jogos e matemática), com aplicações secundárias potenciais em **acessibilidade**
(ex.: leitura da carta em voz alta) e em **pesquisa/benchmark** de visão
computacional.

A **detecção** de várias cartas em uma mesma cena está **fora do escopo** e é
tratada como trabalho futuro.

## Estrutura do repositório

```
.
├── src/            # código-fonte (data, train, evaluate, baseline, predict, config, seed)
│   ├── app.py      #   servidor Flask da demonstração web
│   └── frontend/   #   interface estática (HTML/CSS/JS, sem build)
├── notebooks/      # notebook de treino no Google Colab (GPU T4)
├── models/         # checkpoints treinados (.pt) e históricos de treino (history.json)
├── reports/        # métricas, matrizes de confusão e relatório
├── data/           # SOMENTE instruções de acesso aos dados (ver data/README.md)
├── docs/           # documentação de apoio (definição, dados, ética, model card)
├── Dockerfile      # imagem de deploy (Hugging Face Spaces, CPU)
├── requirements.txt
├── LICENSE         # MIT
└── README.md
```

> **Documentação completa** em [`docs/`](docs/): definição do problema, dados,
> ética/impacto, [Model Card](docs/MODEL_CARD.md) e
> [Relatório de desenvolvimento](docs/RELATORIO_DESENVOLVIMENTO.md).

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

## Aplicação web (demonstração)

Interface web simples (Flask + HTML/CSS/JS, **sem build**) que reaproveita o
modelo já treinado e versionado (`models/efficientnet_b0_best.pt`) — roda em
CPU, sem GPU nem dataset.

**Online (sem instalar nada):**
<https://herickhuggingfaceaccount-cartas.hf.space>

**Local:**

```bash
pip install flask          # ou: pip install -r requirements.txt
python -m src.app          # abra http://localhost:5000
```

Funcionalidades:

- **Reconhecer:** envie a foto de uma carta recortada → predição com **top-5** e
  barras de probabilidade. Aceita **upload**, **arrastar-soltar** e, no celular,
  os botões **“Tirar foto”** (abre a câmera) e **“Galeria”**.
- **Leitura em voz alta (acessibilidade):** opcional, **desligada por padrão**;
  ao ligar (botão "Voz"), o resultado é falado em PT-BR (*Web Speech API*) e a
  escolha fica salva no navegador.
- **Sobre o modelo:** página com as métricas reais (de `reports/`): experimentos,
  acurácia/F1 do modelo principal, acurácia OOD e as classes mais difíceis.
- **Responsivo:** layout adaptado para celular.

> A API tem 3 rotas: `GET /api/status`, `GET /api/stats`, `POST /api/predict`
> (campo `image`, multipart). Veja [`src/app.py`](src/app.py).

### Deploy (Hugging Face Spaces)

A demo é publicada no **Hugging Face Spaces** (SDK *docker*), que builda e hospeda
a partir do [`Dockerfile`](Dockerfile) — torch CPU-only, sem GPU. O checkpoint
sobe via **Git LFS**. Como a Vercel não comporta o tamanho do PyTorch em função
*serverless*, o HF Spaces é a opção adequada para servir o modelo gratuitamente.

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
plataformas como PokerStars e GGPoker. Ver [`docs/03_etica_impacto.md`](docs/03_etica_impacto.md).

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
