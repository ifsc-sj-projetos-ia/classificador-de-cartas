# Relatório Final — Classificador de Cartas de Baralho por Visão Computacional

> **Disciplina:** Introdução à Ciência de Dados (ICD) — Curso de Análise e Desenvolvimento de Sistemas (ADS)
> **Item do projeto:** 2.6 (Relatório final, 4–8 páginas)
> **Autores:** Nicolas; Herick
> **Data:** _(preencher)_
> **Repositório:** _(preencher link do GitHub — ver Seção 7)_
>
> **Como usar este template:** cada seção traz, em blocos `> Orientação`, o que você deve preencher e quais documentos do repositório consultar. Substitua todos os campos marcados como **(preencher após o treino)** ou **(preencher)**. Não invente métricas: rode o treino no Colab e cole os números reais. Ao final, apague estes blocos de orientação se desejar uma versão "limpa".

---

## 1. Introdução e definição do problema

> **Orientação — preencher 2–4 parágrafos curtos:**
> - Apresente o problema: **classificação** de uma única carta de baralho já recortada em **53 classes** (52 cartas + 1 coringa/joker). Deixe claro que **detecção** (várias cartas em cena) é trabalho futuro, não faz parte do escopo.
> - Justifique a relevância com o **ângulo de impacto humano primário**: ferramenta **educacional** (ensino de probabilidade, regras de jogos e matemática para crianças e idosos; apoio à aprendizagem; também serve para ensinar visão computacional). Cite os públicos secundários: acessibilidade para pessoas com deficiência visual (ler a carta em voz alta) e pesquisa/benchmark em VC.
> - Enuncie o objetivo geral e os objetivos específicos (treinar um modelo de transfer learning, comparar com um baseline clássico, avaliar generalização OOD).
> - Referencie o documento de **definição do problema** (`docs/definicao.md`) para detalhes de motivação e requisitos.

**Contexto e motivação.** _(preencher)_

**Definição formal do problema.** Trata-se de um problema de **classificação supervisionada multiclasse** com 53 classes mutuamente exclusivas, onde a entrada é uma imagem RGB de 224×224 px de uma única carta já recortada e a saída é um único rótulo. _(preencher detalhes)_

**Objetivo geral.** _(preencher)_

**Objetivos específicos.**
- _(preencher — ex.: treinar EfficientNet-B0 via transfer learning)_
- _(preencher — ex.: comparar contra baseline HOG + Regressão Logística)_
- _(preencher — ex.: medir o gap de domínio em dados OOD)_

---

## 2. Dados e metodologia

> **Orientação — preencher e referenciar `docs/dados.md`:**
> - Descreva o **dataset principal**: "Cards Image Dataset-Classification" (autor *gpiosenka*, Kaggle), 53 classes, imagens 224×224×3 já recortadas, split pronto **train/valid/test = 7.624 / 265 / 265** (5 imagens por classe em validação e teste). Mencione a licença marcada como **"Other"** no Kaggle e a necessidade de **confirmar antes de redistribuir** (por isso `data/` contém apenas instruções, não as imagens).
> - Descreva o **conjunto OOD**: um **baralho de design diferente** montado a partir de imagens limpas de licença livre (web), via `src/ood_design.py`, usado apenas para avaliação de generalização. Deixe claro que mede o **gap de design** (estilo de carta diferente), e **não** o gap de condições de captura (fotos reais) — este último é trabalho futuro. Documente a fonte/licença das imagens e quantas classes/imagens.
> - Detalhe o **pré-processamento e data augmentation**: normalização ImageNet, resize/center-crop, e (no experimento de augmentation) flips/rotações/jitter de cor — descreva exatamente o que `src/data.py` aplica.
> - Detalhe os **modelos**: EfficientNet-B0 (Fase 1: backbone congelado / *feature extraction*; Fase 2: *fine-tuning* do topo com LR baixo) e o baseline HOG + Regressão Logística (scikit-learn). Referencie o **model card** (`docs/model_card.md`).

### 2.1 Dataset principal (Kaggle)

| Item | Valor |
|---|---|
| Nome | Cards Image Dataset-Classification (gpiosenka) |
| Nº de classes | 53 (52 cartas + 1 coringa) |
| Dimensão das imagens | 224 × 224 × 3 (RGB, já recortadas) |
| Split train / valid / test | 7.624 / 265 / 265 |
| Imagens por classe (val/test) | 5 / 5 |
| Licença | "Other" (Kaggle) — _(confirmar antes de redistribuir)_ |

O dataset é aproximadamente balanceado (~144 imagens/classe em média, 7.624/53). No teste, o F1-macro (0,947) ficou igual à accuracy (0,947), confirmando desempenho homogêneo entre as 53 classes — não foi necessário usar *class weights*.

### 2.2 Conjunto OOD (baralho de design diferente — web)

> **Orientação:** descreva o conjunto OOD usado nesta entrega. Ele é um baralho de **design diferente** do de treino, montado a partir de **imagens limpas de licença livre** (`src/ood_design.py`), não de fotos do mundo real. Seja honesto sobre o que ele mede (gap de design) e o que não mede (gap de captura). Reforce que serve **só para avaliação**, nunca para treino.

- Origem das imagens: baralho derivado do projeto *Vector Playing Cards* (domínio público), via repositório *playing-cards-assets* (Howard Yeh, MIT). **Não** são fotos tiradas pelos autores.
- Nº de classes / imagens: **53 classes / 54 imagens** (1 por carta + 2 coringas na classe `joker`).
- O que mede: **gap de design** (estilo de carta diferente do treino).
- O que **não** mede: gap de **condições de captura** (luz, sombra, fundo, ângulo de fotos reais). Como são renders limpos, o gap medido é um **limite inferior** do esperado em uso real.
- Trabalho futuro: repetir com **fotos de um baralho físico** (`docs/guia_coleta_baralho_real.md`) para medir o gap de captura.

### 2.3 Pré-processamento e data augmentation

- Normalização: médias/desvios do ImageNet. _(preencher confirmação)_
- Transformações base: _(preencher — resize/center-crop)_
- Augmentation (apenas no experimento 2): _(preencher — flip horizontal, rotação, color jitter, etc.)_

### 2.4 Modelos e treino

- **Modelo principal:** EfficientNet-B0 (torchvision), transfer learning em duas fases.
  - Fase 1 — *feature extraction*: backbone congelado, treina só a cabeça classificadora. **8 épocas, AdamW, LR 1e-3.**
  - Fase 2 — *fine-tuning*: descongela o backbone. **AdamW com LR de pico 3e-4 + cosine annealing, até 20 épocas, early stopping (paciência 6) — parou na época 12.**
- **Baseline:** HOG (orientações=9, células 16×16 px, blocos 2×2, `L2-Hys`, imagem 128×128 em escala de cinza) + Regressão Logística (`max_iter=2000`, `C=1.0`); até 80 imagens/classe.
- **Infra e reprodutibilidade:** Google Colab (GPU T4 gratuita), `set_seed(42)`, `requirements.txt` com versões pinadas. **batch size 32, AdamW, imagem 224×224, normalização ImageNet; 8 épocas de FE + 12 de fine-tuning (de até 20, com early stopping).**

---

## 3. Experimentos e resultados

> **Orientação — 3 experimentos planejados. NÃO invente números; rode no Colab e preencha:**
> - **Experimento 1 — Feature extraction (FE) vs Fine-tuning (FT):** mede o ganho de descongelar o topo.
> - **Experimento 2 — Com vs Sem data augmentation:** mede o impacto da augmentation na generalização.
> - **Experimento 3 — Avaliação OOD:** compara o desempenho no teste Kaggle contra um **baralho de design diferente** (imagens limpas da web), medindo o **gap de design**. Registre que o gap de **captura** (fotos reais) não é medido aqui e fica como trabalho futuro.
> - Para cada experimento, preencha a tabela abaixo e comente os resultados. A accuracy **esperada** do transfer learning em cartas é ~93–95% (referência da literatura, **não** é resultado seu — substitua pelos números reais).
> - Inclua **matriz de confusão** e a análise de **"confusões perigosas"** (trocar naipe ou valor).

### 3.1 Tabela consolidada de resultados

> Métricas: **Accuracy** e **F1 macro** no teste Kaggle (in-distribution) + **Accuracy OOD** no baralho de design diferente (web). Valores **medidos** no Colab (GPU T4, `set_seed(42)`).

| Configuração | Acc. (teste Kaggle) | F1 macro (teste Kaggle) | Acc. OOD (design diferente) | Observações |
|---|---|---|---|---|
| Baseline: HOG + Reg. Logística | 0,706 | 0,698 | — | Referência clássica |
| EfficientNet-B0 — Feature Extraction (FE) | 0,385 | 0,363 | — | Backbone congelado |
| EfficientNet-B0 — Fine-Tuning (FT, com aug) | 0,947 | 0,947 | 0,593 | **Modelo principal** |
| EfficientNet-B0 — FT **sem** augmentation | 0,974 | 0,973 | *(a medir na célula 8b)* | Maior no teste limpo |

> _Referência da literatura (não é resultado deste trabalho): transfer learning em cartas tende a ~93–95% de accuracy in-distribution._

### 3.2 Experimento 1 — Feature extraction vs Fine-tuning

- Resultado: **FE (congelado) 0,385** vs **FT 0,947** de acurácia no teste (**+56,2 pp**); F1-macro 0,363 → 0,947.
- Interpretação: _(preencher — o fine-tuning valeu o custo? quanto ganhou?)_

### 3.3 Experimento 2 — Com vs sem data augmentation

- Resultado: **com aug 0,947** vs **sem aug 0,974** no teste limpo (sem aug **+2,6 pp** in-distribution). OOD (design) com aug = **0,593**; sem aug *(a medir na célula 8b)*.
- Interpretação: _(preencher — a augmentation ajudou na generalização e/ou no OOD?)_

### 3.4 Experimento 3 — Avaliação OOD (gap de design)

- Gap de design medido (Acc. Kaggle − Acc. OOD design diferente): **0,947 − 0,593 = 0,354 (≈ 35 pp)**.
- Interpretação: _(preencher — quanto o desempenho cai ao trocar o estilo do baralho; o modelo aprendeu o conceito da carta ou decorou o design do treino?)_
- Ressalva (honestidade): este OOD usa **imagens limpas**, então o gap é um **limite inferior**; com **fotos reais** (gap de captura) a queda tende a ser maior. Medi-lo é trabalho futuro.

### 3.5 Matriz de confusão e confusões perigosas

> **Orientação:** insira a figura da matriz de confusão (exportada por `src/evaluate.py`) e discuta os erros mais frequentes. Classifique as **confusões perigosas**: trocar **naipe** (♠♥♦♣) ou **valor** (ex.: 6 vs 9, valete vs dama), que em um uso educacional ou de acessibilidade transmitiriam informação errada.

- Figura: `reports/confusion_matrix_test.png` (teste) e `reports/confusion_matrix_ood.png` (OOD); relatório por classe em `reports/classification_report_test.csv`.
- Confusões mais frequentes: 14 erros em 265 (251 corretos). Ex.: `nine of diamonds`→`eight of diamonds`, `seven of clubs`→`eight of clubs`, `five of diamonds`→`three of diamonds`, `nine of spades`→`six of spades`; classe `joker` com 2 de 5 imagens erradas.
- Confusões perigosas (naipe/valor): predominam trocas de **valor dentro do mesmo naipe** (contagem de pips em cartas numéricas próximas); **não** houve troca de **naipe/cor** entre as listadas — positivo, pois erro de naipe (copas↔ouros) seria o mais crítico no uso educacional/assistivo. Ponto fraco isolado: `joker`.

---

## 4. Discussão: limitações, ética e impacto humano

> **Orientação — referencie `docs/etica.md` e o model card. Cubra explicitamente os riscos levantados na pesquisa:**
> - **(a) Dual-use / trapaça:** a mesma tecnologia pode virar *Real-Time Assistance* (RTA) — proibido por plataformas como PokerStars e GGPoker — em jogos de azar e apostas. Declare que este é um **uso proibido** do projeto.
> - **(b) Jogo problemático:** risco de facilitar comportamento de jogo com dano social documentado.
> - **(c) Viés/limitação do dataset:** treinado em **um** tipo de baralho → tende a falhar em outros designs, iluminações e oclusão (conecte com o resultado do Experimento 3/OOD).
> - **(d) Privacidade (LGPD):** a câmera pode captar o ambiente e terceiros → minimização de dados, processamento **on-device**, sem armazenar imagens desnecessárias.
> - **(e) Documentação ética insuficiente:** publicar sem declarar usos proibidos é um risco em si → este relatório, o `docs/etica.md` e o model card mitigam isso.

### 4.1 Limitações técnicas

- _(preencher — escopo só de classificação, dependência de recorte prévio, dataset de 1 baralho, val/test pequenos com 5 img/classe)_
- _(preencher — ligar à queda observada no OOD)_

### 4.2 Considerações éticas

- **Dual-use e trapaça:** _(preencher — declarar uso proibido; citar RTA / PokerStars / GGPoker)_
- **Jogo problemático:** _(preencher)_
- **Viés do dataset:** _(preencher — ligar ao Experimento 3)_
- **Privacidade / LGPD:** _(preencher — minimização, on-device)_

### 4.3 Impacto humano

- **Primário — educacional:** _(preencher — ensino de probabilidade, regras, matemática; público infantil/idoso; ensino de VC)_
- **Secundário — acessibilidade:** _(preencher — leitura da carta em voz alta para deficientes visuais)_
- **Secundário — pesquisa/benchmark:** _(preencher)_

---

## 5. Conclusão e trabalhos futuros

> **Orientação — 1–2 parágrafos + bullets:**
> - Retome os objetivos e diga, com os números reais, se foram atingidos.
> - Resuma a principal lição de cada experimento (FE vs FT, augmentation, gap OOD).
> - Liste extensões futuras. **Cite explicitamente detecção com YOLO** (várias cartas em cena, do recorte único para a cena completa) como a evolução natural do escopo.

**Conclusão.** _(preencher — síntese com base nos resultados reais)_

**Trabalhos futuros.**
- **Detecção com YOLO:** estender de classificação de carta única para **detecção** de múltiplas cartas em cena (ex.: YOLOv8), removendo a dependência do recorte manual. _(preencher)_
- Ampliar o conjunto OOD com mais baralhos/designs e condições de captura. _(preencher)_
- Empacotar o modo educacional/acessibilidade (leitura em voz alta, exercícios de probabilidade). _(preencher)_
- _(preencher — outras extensões)_

---

## 6. Reprodutibilidade

> **Orientação:** descreva como reproduzir do zero. Referencie o `README.md` e a estrutura do `src/`. Reforce `set_seed(42)` e `requirements.txt` pinado.

1. Clonar o repositório: _(preencher comando, ver Seção 7)_
2. Criar ambiente e instalar dependências pinadas: `pip install -r requirements.txt`.
3. Obter os dados conforme `data/` (instruções) e `docs/dados.md` — o dataset **não** é redistribuído (licença "Other").
4. Configurar `src/config.py` (caminhos, hiperparâmetros) e garantir `set_seed(42)` (`src/seed.py`).
5. Baseline: `python -m src.baseline`. _(preencher)_
6. Treino: `python -m src.train` (ou o notebook `notebooks/treino_cartas_colab.ipynb` na GPU T4 do Colab).
7. Avaliação: `python -m src.evaluate` (gera métricas e matriz de confusão em `reports/`).
8. Predição em imagem única: `python -m src.predict --img <caminho>`. _(preencher)_

---

## 7. Repositório e instruções de acesso

- **Link do GitHub:** _(preencher — ex.: https://github.com/usuario/classificador-cartas)_
- **Licença do código:** MIT (ver `LICENSE`).
- **Documentos de apoio no repositório:**
  - `docs/definicao.md` — definição do problema (Seção 1)
  - `docs/dados.md` — dataset, OOD e licença (Seção 2)
  - `docs/etica.md` — riscos éticos e usos proibidos (Seção 4)
  - `docs/model_card.md` — model card do EfficientNet-B0 (Seções 2 e 4)
  - `README.md` — visão geral e reprodução (Seção 6)

---

## Apêndice A — Roteiro da apresentação (item 2.7, 10–15 min)

> **Orientação:** roteiro sugerido com tempos. Ajuste à duração-alvo e ensaie. Use as figuras de `reports/`.

| Bloco | Tempo | Conteúdo |
|---|---|---|
| 1. Abertura e problema | 1–2 min | O que é: classificar 1 carta recortada em 53 classes. Por que importa: ferramenta **educacional** (probabilidade/regras/matemática); acessibilidade e benchmark como secundários. |
| 2. Dados e método | 2–3 min | Dataset Kaggle (53 classes, 224×224, split 7.624/265/265); conjunto **OOD** (baralho de **design diferente**, imagens limpas da web); EfficientNet-B0 (FE → FT) vs baseline HOG + Reg. Logística. |
| 3. Experimentos | 3–4 min | Os 3 experimentos: FE vs FT; com vs sem augmentation; **gap OOD**. Mostrar a tabela consolidada (números reais) e a **matriz de confusão**. |
| 4. Confusões perigosas e limitações | 1–2 min | Erros de naipe/valor; limitação de 1 baralho; queda no OOD de design. Ser honesto: o OOD é de design (web), não fotos reais → gap de captura é limite inferior/trabalho futuro. |
| 5. Ética e impacto | 2 min | Dual-use/RTA (uso **proibido**), jogo problemático, viés do dataset, privacidade/LGPD (on-device). Impacto educacional positivo. |
| 6. Conclusão e futuro | 1–2 min | O que foi atingido; **detecção com YOLO** como próximo passo. |
| 7. Perguntas | restante | Q&A. |

**Dicas de apresentação:** _(preencher — 1 slide por bloco, no máx. ~2 ideias por slide, mostrar 1 exemplo de predição correta e 1 confusão perigosa, ensaiar para fechar dentro de 15 min)._