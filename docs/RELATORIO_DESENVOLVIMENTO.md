# Relatório de Desenvolvimento — Classificador de Cartas de Baralho por Visão Computacional

> **Disciplina:** Introdução à Ciência de Dados (ICD) — Análise e Desenvolvimento de Sistemas (ADS)
> **Autores:** Nicolas e Herick
> **Tarefa:** classificação de uma única carta de baralho já recortada em **53 classes** (52 cartas + coringa/*joker*)
> **Stack:** PyTorch + torchvision (EfficientNet-B0), scikit-learn/scikit-image (baseline), kagglehub; treino no Google Colab (GPU T4)
> **Data:** 2026-06

**Propósito deste documento.** Este é um **relatório de desenvolvimento técnico**, mais detalhado que o relatório final. Ele serve a dois objetivos: (a) ser a base de conteúdo para escrever o **relatório final** (`reports/relatorio_final_outline.md`) e os **slides**; e (b) ser **material de estudo para a defesa oral** (ver Seção 13, "Banco de perguntas e respostas"). O tom é técnico-acadêmico e **honesto**: limitações, incertezas e itens pendentes são declarados explicitamente.

> **Nota de integridade dos números.** Todas as métricas citadas foram **medidas no Colab (GPU T4), `set_seed(42)`** — incluindo a OOD do modelo sem augmentation (célula 8b) e o Experimento 4 (2º design + aug de aparência). Onde a referência da literatura aparece (~93–95%), ela é **expectativa**, não resultado deste trabalho.

---

## Sumário

1. [Visão geral e objetivo (impacto humano)](#1-visão-geral-e-objetivo-impacto-humano)
2. [Definição do problema e escopo](#2-definição-do-problema-e-escopo)
3. [Dados](#3-dados)
4. [Arquitetura do código](#4-arquitetura-do-código)
5. [Modelos e técnicas](#5-modelos-e-técnicas)
6. [Pipeline de treino e reprodutibilidade](#6-pipeline-de-treino-e-reprodutibilidade)
7. [Experimentos (1, 2, 3)](#7-experimentos-1-2-3)
8. [Resultados consolidados e análise de erros](#8-resultados-consolidados-e-análise-de-erros)
9. [Jornada de desenvolvimento e lições aprendidas](#9-jornada-de-desenvolvimento-e-lições-aprendidas)
10. [Ética, impacto e LGPD](#10-ética-impacto-e-lgpd)
11. [Limitações e ameaças à validade](#11-limitações-e-ameaças-à-validade)
12. [Trabalhos futuros](#12-trabalhos-futuros)
13. [Banco de perguntas e respostas para a defesa](#13-banco-de-perguntas-e-respostas-para-a-defesa)
14. [Glossário rápido](#14-glossário-rápido)

---

## 1. Visão geral e objetivo (impacto humano)

O projeto constrói um **classificador de cartas de baralho** por visão computacional: dada **uma imagem RGB de uma única carta já recortada**, o sistema devolve **um rótulo entre 53 classes** (as 52 cartas de um baralho francês padrão — 13 valores × 4 naipes — mais o coringa/*joker*). Tecnicamente, é um problema de **classificação multiclasse de imagem única**, resolvido por **transfer learning** com **EfficientNet-B0** (PyTorch/torchvision), comparado a um **baseline clássico** (HOG + Regressão Logística, scikit-learn).

**Eixo de impacto humano (por ordem de prioridade):**

- **Educacional (primário).** Apoiar o ensino de **probabilidade, raciocínio combinatório, regras de jogos e matemática elementar** para crianças e idosos, tornando conceitos abstratos tangíveis a partir de cartas físicas reconhecidas pela câmera. O próprio pipeline também é **material didático para ensinar visão computacional** (transfer learning, métricas, avaliação de generalização) — por ser um problema visualmente intuitivo.
- **Acessibilidade (secundário).** Ler a carta identificada **em voz alta** (ex.: "Dama de Copas"), apoiando a autonomia de pessoas com deficiência visual em contextos não monetários.
- **Pesquisa / benchmark (secundário).** Um pipeline reprodutível (dataset público, seed fixa, dependências pinadas) serve como base comparável para estudar transfer learning, *data augmentation* e robustez a domínio.

**Usos proibidos / fora de escopo (declarados).** O projeto **não** se destina a jogos de azar a dinheiro, apostas, cassinos online, nem a assistência em tempo real (RTA) durante partidas valendo dinheiro — ver Seção 10. (Fonte: `README.md`, `docs/01_definicao_problema.md`, `docs/03_etica_impacto.md`.)

---

## 2. Definição do problema e escopo

- **Entrada (X):** 1 imagem RGB de uma carta já recortada, redimensionada internamente para **224×224×3** e normalizada com a média/desvio do ImageNet.
- **Saída (y):** distribuição de probabilidade (softmax) sobre **53 classes mutuamente exclusivas**; rótulo previsto = `argmax`. Disponível também em **top-k** (padrão k=3) com probabilidades.
- **Taxonomia:** `53 classes = 13 valores × 4 naipes + 1 coringa`. Valores: ás, 2–10, valete, dama, rei. Naipes: copas (*hearts*), ouros (*diamonds*), paus (*clubs*), espadas (*spades*). Cada carta carrega dois atributos latentes — **valor** e **naipe** — usados na análise de erros (Seção 8).

**Escopo.** A entrega é deliberadamente restrita a **classificação** (1 carta recortada → 1 rótulo). A **detecção** de múltiplas cartas em cena (localização + recorte automático, ex.: YOLO) está **explicitamente fora do escopo** e é tratada como **trabalho futuro** (Seção 12). Essa restrição também é uma escolha **ética**: a varredura de mesa em tempo real é justamente o que tornaria um RTA prático, e o projeto opta por **não construí-la nem facilitá-la** (Seção 10).

(Fonte: `docs/01_definicao_problema.md`, requisitos funcionais RF01–RF09 e não-funcionais RNF01–RNF13.)

---

## 3. Dados

### 3.1 Dataset principal (in-distribution)

| Item | Valor |
|---|---|
| Nome | **Cards Image Dataset-Classification** (autor **gpiosenka**, Kaggle) |
| URL | https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification |
| Nº de classes | 53 (52 cartas + coringa) |
| Dimensão | 224×224×3 (RGB, **já recortadas**, uma carta por imagem) |
| Split (pronto) | **treino 7.624 / validação 265 / teste 265** |
| Imagens por classe (val/test) | **5 / 5** |
| Treino por classe | ~144 em média (7.624 / 53), aproximadamente equilibrado |
| Licença | marcada como **"Other"** no Kaggle — **a confirmar antes de redistribuir** |

O split oficial é adotado **sem modificação**, para preservar comparabilidade com outros trabalhos. O repositório **não redistribui** as imagens: a pasta `data/` contém apenas instruções de download (`data/README.md`), evitando redistribuição não autorizada enquanto a licença "Other" não é confirmada.

**Por que val/teste pequenos importam.** Com apenas **5 imagens por classe** em validação e teste, as métricas são **estatisticamente ruidosas**: o acerto/erro de **uma única imagem por classe** muda a acurácia daquela classe em **~20 pontos percentuais**. Consequências assumidas no projeto: (i) ler a acurácia com **intervalo de incerteza amplo**; (ii) **privilegiar F1-macro e a matriz de confusão** sobre a acurácia isolada; (iii) **evitar overclaiming** — diferenças pequenas entre configurações (ex.: com vs. sem augmentation) podem **não ser significativas**.

### 3.2 Pré-processamento

- **Verificação de integridade:** abertura/decodificação de cada arquivo, conversão para **RGB** (3 canais), conferência das 53 pastas por split.
- **Resize:** as imagens já vêm em 224×224, mas mantém-se um passo explícito de `Resize((224,224))` por robustez (uniformiza o OOD, cujas imagens têm proporções variadas).
- **Normalização ImageNet:** `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]` (constantes em `src/config.py`). Casa a distribuição de entrada com a vista no pré-treino — **a mesma normalização** vale para treino, validação, teste e OOD.

### 3.3 Conjunto OOD de "design diferente" — decisão honesta e importante

Por **falta de tempo para fotografar um baralho físico** antes da entrega, o conjunto OOD foi montado a partir de um **baralho de design diferente** com **imagens limpas de licença livre da web**, geradas de forma **reprodutível** por `src/ood_design.py`:

```bash
python -m src.ood_design --out data/raw/ood_design_web
```

- **Procedência/licença:** assets do repositório `hayeah/playing-cards-assets` (licença **MIT**, Howard Yeh), cuja arte deriva do projeto **Vector Playing Cards** (**domínio público**).
- **Conteúdo:** 53 subpastas com **nomes idênticos** às classes do treino, totalizando **54 imagens** (1 por carta + **2 coringas** na pasta `joker/`, aproveitando os dois jokers preto/vermelho). Confirmado no repositório em `data/raw/ood_design_web/`.

**O que este conjunto mede — e o que NÃO mede:**

| Tipo de gap | Mede aqui? | Por quê |
|---|---|---|
| **Gap de design** (arte/fontes/cores diferentes do treino) | **Sim** | As cartas têm desenho e tipografia distintos do dataset gpiosenka. |
| **Gap de captura** (luz, sombra, fundo real, ângulo, reflexo, oclusão, desfoque) | **Não** | São renders digitais limpos sobre fundo uniforme — sem condições do mundo real. |

**Implicação metodológica (declarada para evitar *overclaiming*).** Como as imagens são "limpas", em vários aspectos elas são **mais próximas** do domínio de treino do que fotos reais seriam. Portanto, o gap medido aqui é um **LIMITE INFERIOR** do gap esperado em uso real. A pergunta que ele responde é estreita e legítima: *o modelo aprendeu o conceito de cada carta ou decorou o estilo visual do dataset?* Fotografar um baralho físico (gap de captura) fica como **trabalho futuro**, com guia completo em `docs/guia_coleta_baralho_real.md`.

**Honestidade de rotulagem.** Em nenhum momento se apresentou imagem da web **como se fosse foto real**: o conjunto é declarado como "design diferente / imagens limpas", e o arquivo `_FONTE.txt` registra a procedência junto às imagens. O OOD **não** participa de treino nem de validação — é reservado exclusivamente ao Experimento 3.

(Fonte: `docs/02_dados.md`, `docs/guia_ood_design_web.md`, `src/ood_design.py`.)

---

## 4. Arquitetura do código

O código-fonte é um pacote Python modular em `src/`, com responsabilidades separadas por arquivo, mais o notebook de orquestração.

| Módulo | Responsabilidade |
|---|---|
| **`src/config.py`** | Configuração central (`@dataclass Config`): caminhos, hiperparâmetros das duas fases, seed, device. Centraliza valores para reprodutibilidade e variação controlada nos experimentos. Define também `IMAGENET_MEAN/STD` e `SUPPORTED_BACKBONES`. |
| **`src/seed.py`** | `set_seed(42)` — fixa as sementes de `random`, `numpy` e `torch` (CPU/GPU); ativa modo determinístico do cuDNN. Chamado **antes** de criar dataloaders e modelos. |
| **`src/data.py`** | Dataloaders e transformações (torchvision). `build_transforms` cria os pipelines de treino (com/sem augmentation) e de avaliação; `build_dataloaders` monta train/valid/test via `ImageFolder` (com checagem de que as classes coincidem entre splits); `build_eval_loader` serve uma pasta arbitrária (ex.: OOD). Resolve nomes alternativos de split (`valid`/`validation`/`val`). |
| **`src/model.py`** | Construção dos modelos. `build_model` instancia o backbone pré-treinado (EfficientNet-B0 por padrão; também MobileNet-V3 small/large e ResNet18) e **substitui a cabeça** por uma camada para N classes (com `Dropout(0.3)` no caso da EfficientNet). `set_backbone_trainable` (des)congela o backbone para o fine-tuning; `count_trainable_params` loga o nº de parâmetros treináveis por fase. |
| **`src/baseline.py`** | Baseline clássico **HOG + classificador linear** (Regressão Logística por padrão; SVM linear opcional). Extrai HOG, padroniza com `StandardScaler` e treina/avalia, retornando accuracy e F1-macro. |
| **`src/train.py`** | Laço de treino em duas fases com **early stopping**: `_run_epoch` (treina/avalia uma época), `_train_phase` (treina uma fase, guarda o melhor `state_dict` pela acurácia de validação), `train_model` (orquestra Fase 1 → Fase 2, salva o melhor checkpoint com pesos + nomes de classe + config + histórico). Também expõe CLI. |
| **`src/evaluate.py`** | Métricas e artefatos: `collect_predictions`, `compute_metrics` (accuracy, F1-macro/ponderado, *classification report*), `plot_confusion_matrix`, `top_confusions` (pares verdadeiro→previsto mais confundidos, para "confusões perigosas"), `evaluate_loader` (avaliação completa) e `evaluate_ood` (remapeia rótulos OOD pelo **nome** da pasta para o espaço de classes do treino). |
| **`src/predict.py`** | Inferência em **uma** imagem: `load_model` (recarrega checkpoint + nomes de classe + config) e `predict_image` (retorna top-k classes com probabilidade). É o módulo do cenário de uso educacional/assistivo. CPU-friendly (RF04). |
| **`src/ood_design.py`** | Monta o conjunto OOD de "design diferente": baixa (ou copia de um diretório local) os PNGs do baralho de domínio público e cria 53 pastas com os nomes exatos das classes de treino; grava `_FONTE.txt` com a atribuição/licença. |

**Notebook `notebooks/treino_cartas_colab.ipynb`.** Orquestra o pipeline de ponta a ponta no Colab, em 10 seções: (1) setup do ambiente e **backup no Google Drive**; (2) download do dataset via `kagglehub`; (3) EDA; (4) baseline HOG+LogReg; (5) treino do modelo principal (FE→FT, cosine annealing); (6) avaliação no teste + relatório por classe; (7) **Experimentos 1 e 2** (FE vs FT; com vs sem aug) gerando `reports/experimentos.csv`; (8) **Experimento 3** (OOD de design) + **célula 8b** (gap OOD com vs sem aug, reaproveitando checkpoints); (9) demo de predição; (10) exportação do modelo e das figuras. As células de backup (`backup_para_drive`) protegem o trabalho contra reset do runtime efêmero.

---

## 5. Modelos e técnicas

### 5.1 Baseline clássico — HOG + Regressão Logística

Cumpre o requisito de **piso de comparação** barato (roda em CPU). Justificativa: HOG (*Histogram of Oriented Gradients*) descreve **forma/bordas** da carta, e um classificador linear separa as 53 classes; é uma técnica tradicional de extração de características, anterior ao aprendizado profundo, que dá um referencial honesto do "quanto a tarefa é fácil sem rede neural".

Hiperparâmetros (confirmados em `src/baseline.py`):

- **HOG:** `orientations=9`, `pixels_per_cell=(16,16)`, `cells_per_block=(2,2)`, `block_norm="L2-Hys"`, imagem **128×128 em escala de cinza**.
- **Padronização:** `StandardScaler` (fit no treino, transform no teste).
- **Classificador:** `LogisticRegression(max_iter=2000, C=1.0, random_state=42)`.
- **Amostragem:** até **80 imagens/classe** no treino (`--max-per-class 80`) para acelerar (HOG+LogReg em CPU).

### 5.2 Modelo principal — EfficientNet-B0 (transfer learning)

- **Backbone:** EfficientNet-B0 **pré-treinado no ImageNet** (`weights="IMAGENET1K_V1"`).
- **Cabeça:** substituída por `Sequential(Dropout(p=0.3), Linear(in_features, 53))` (em `src/model.py`, `_replace_head`).
- **Normalização ImageNet** na entrada (Seção 3.2).

**Por que EfficientNet-B0 (e não ResNet/MobileNet).** É um bom **equilíbrio acurácia/custo**: leve o suficiente para treinar na GPU T4 grátis do Colab e para inferência razoável em CPU (RF04/RNF07), com forte desempenho via *compound scaling*. O código suporta ResNet18 e MobileNet-V3 como alternativas (`SUPPORTED_BACKBONES`), mas a EfficientNet-B0 foi a escolhida para o modelo principal.

### 5.3 Transfer learning em duas fases

- **Fase 1 — *feature extraction* (backbone congelado).** Congela todos os parâmetros do backbone e treina **apenas a cabeça**. **8 épocas, AdamW, LR 1e-3.** Objetivo: ajustar rapidamente o classificador às features genéricas do ImageNet, sem mexer no backbone.
- **Fase 2 — *fine-tuning* (descongela o backbone).** `set_backbone_trainable(True)` e re-otimiza a rede inteira com **AdamW, LR de pico 3e-4 + cosine annealing**, até **20 épocas**, com **early stopping** (paciência 6). Na prática, o melhor checkpoint ficou em torno da **época 12**. Objetivo: adaptar as features ao **domínio de cartas** (que o ImageNet não cobre bem) — esta fase é **decisiva** (ver Experimento 1).

### 5.4 Cosine annealing e early stopping

- **Cosine annealing (`CosineAnnealingLR`, `T_max = epochs_finetune`).** O learning rate parte do pico (3e-4) e **decai suavemente seguindo uma cosseno** até ~0 ao longo das épocas. Vantagem: LR alto no início acelera a convergência; LR baixo no fim permite um **ajuste fino estável**, sem oscilar em torno do mínimo. Foi a peça que destravou a convergência (ver Seção 9, "saga do undertraining").
- **Early stopping (paciência 6).** Interrompe a fase se a **acurácia de validação** não melhora por 6 épocas seguidas, guardando sempre o **melhor** `state_dict`. A paciência relativamente alta (6) é proposital: a validação tem só 5 img/classe (ruidosa), então uma paciência pequena causaria parada precoce.

### 5.5 Data augmentation (apenas no treino)

Aplicada **somente ao conjunto de treino**; validação, teste e OOD usam só resize + normalização, para que as métricas reflitam imagens não perturbadas. Transformações (em `src/data.py`):

- `RandomRotation(15)` — cartas levemente desalinhadas;
- `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)` — iluminação/balanço de branco distintos;
- `RandomAffine(degrees=0, translate=(0.05,0.05), scale=(0.95,1.05))` — enquadramento imperfeito;
- `RandomErasing(p=0.25, scale=(0.02,0.10))` — oclusão parcial.

**Flips foram EVITADOS de propósito:** índices e símbolos das cartas têm **orientação**, e espelhar horizontal/verticalmente geraria imagens irreais ou ambíguas. As intensidades são **moderadas**, para não destruir a leitura de naipe/valor. O impacto da augmentation é objeto do **Experimento 2**.

### 5.6 Métricas de avaliação

- **Accuracy** (acurácia global) no teste Kaggle.
- **F1-macro** (média não ponderada por classe) — **métrica de destaque**, porque trata todas as 53 classes igualmente (não deixa as frequentes mascararem as raras).
- **Matriz de confusão** 53×53 (normalizada por linha) para inspeção visual.
- **Análise de "confusões perigosas":** distinguir erros de **valor** (ex.: 8↔9) de erros de **naipe/cor** (ex.: copas↔ouros), por terem implicações distintas no uso educacional/assistivo.

**Por que ênfase em F1-macro e matriz de confusão.** Com val/teste de 5 img/classe, a acurácia pontual é **ruidosa** (±20 pp por imagem na classe). F1-macro e a matriz revelam **padrões de erro** (quais pares se confundem, quais classes são frágeis) — informação mais robusta e mais útil do que o número isolado de acurácia.

---

## 6. Pipeline de treino e reprodutibilidade

**Fluxo de ponta a ponta:** dados → baseline → treino (FE → FT) → avaliação (teste) → experimentos (1, 2) → OOD (3) → predição → exportação.

**Reprodutibilidade (RNF11–RNF13):**

- **Seed fixa:** `set_seed(42)` controla `random`, `numpy`, `torch` (CPU/GPU) e ativa o modo determinístico do cuDNN. Chamada antes de criar dataloaders/modelos.
- **Ambiente pinado:** `requirements.txt` fixa scikit-learn 1.5.2, scikit-image 0.24.0, numpy 1.26.4, pandas 2.2.3, pillow 10.4.0, matplotlib 3.9.2, seaborn 0.13.2, tqdm 4.66.5, kagglehub 0.3.4. **`torch`/`torchvision` ficam sem pin rígido** (`>=2.2` / `>=0.17`) **de propósito**: no Colab eles já vêm com a versão de CUDA correta, e fixar versões poderia quebrar a GPU. Para travar o ambiente exato após rodar, sugere-se `pip freeze > requirements-lock.txt`.
- **Split fixo** (o oficial do dataset) e **checkpoint autocontido:** `train.py` salva pesos + nomes de classe + config + histórico em `models/<backbone>_best.pt`, permitindo recarregar e avaliar sem retreino.

**Infra de execução:**

- **Treino:** Google Colab, **GPU T4 gratuita**; alternativa **Kaggle** (quando a cota grátis de GPU do Colab esgota).
- **Backup no Google Drive:** o notebook monta o Drive e copia `models/` e `reports/` (função `backup_para_drive`), porque o runtime do Colab é **efêmero** — um reset apaga tudo em `/content`. Detalhes da motivação na Seção 9.

> **Divergência repo × briefing (documentada).** Os **defaults da CLI** em `src/train.py::main()` ainda são os **antigos** (`--lr-finetune 1e-5`, `--epochs-finetune 5`), valores que **subtreinam** (ver Seção 9). Os defaults **corretos/atuais** vivem na `@dataclass Config` (`lr_finetune=3e-4`, `epochs_finetune=20`) e são os que o **notebook usa** (a célula da Seção 5 passa `epochs_finetune=20` e herda `lr_finetune=3e-4` do `Config`). **Os resultados reportados vêm do notebook**, portanto refletem 3e-4/20ep + cosine annealing. Recomendação de saneamento: alinhar os defaults do `argparse` aos do `Config` para evitar que uma execução por linha de comando reproduza acidentalmente o regime de undertraining.

---

## 7. Experimentos (1, 2, 3)

Os três experimentos são controlados (variando um fator por vez) e rodam no notebook (Seções 7 e 8), com `set_seed(42)`.

### 7.1 Experimento 1 — Feature extraction (congelado) vs Fine-tuning

- **Objetivo:** medir o ganho de **descongelar o backbone**.
- **Método:** mesma arquitetura e dados; (a) só a cabeça treina (`finetune=False`); (b) FE seguido de FT (`finetune=True`, 20 épocas, cosine annealing).
- **Resultado:** teste **0,3849 → 0,9472** de acurácia (**+56 pp**); F1-macro **0,3630 → 0,9472**.
- **Interpretação:** as features ImageNet, congeladas, são **fracas para cartas** (objetos finos, simbólicos, com tipografia e *pips* — distantes das classes naturais do ImageNet). O fine-tuning **adapta** essas features ao domínio e é **decisivo**: sem ele, a EfficientNet-B0 fica **abaixo** até do baseline HOG (0,706). Conclusão: para este domínio, **fine-tuning não é opcional**.

### 7.2 Experimento 2 — Com vs Sem data augmentation

- **Objetivo:** medir o efeito da augmentation na generalização.
- **Método:** melhor configuração (FE+FT, 20 épocas), ligando/desligando a augmentation de treino.
- **Resultado (in-distribution, teste limpo):** **sem aug 0,9736 / F1 0,9734** vs **com aug 0,9472 / F1 0,9472** — o **sem aug é ~2,6 pp MAIOR** no teste limpo.
- **Resultado (OOD design):** com aug = **0,5926 / F1 0,5741**; sem aug = **0,5926 / F1 0,5678** — **mesma acurácia OOD** (32/54), F1 quase igual (diferença dentro do ruído).
- **Interpretação honesta:** como o teste é **quase igual ao treino** (imagens limpas, mesmo design), a augmentation **regulariza demais** e custa ~2,6 pp *in-distribution*. **NÃO** se deve afirmar que "a augmentation melhorou os resultados": no teste limpo ela **piorou**, e no **OOD de design** ela **empatou** (mesma acurácia; F1 +0,6 pp, dentro do ruído). Isso é **esperado**: a augmentation usada (rotação, *color jitter*, *affine*, *erasing*) simula **variação de captura**, mas este OOD testa **variação de design** (arte/fontes) — gaps de tipos diferentes. O modelo principal adotado é o **com augmentation** como escolha de **deploy** (no uso real, com fotos, espera-se robustez a captura), declarando com transparência que esse benefício **não foi medido aqui** — exigiria um OOD de fotos reais (trabalho futuro).

### 7.3 Experimento 3 — Avaliação OOD (gap de design)

- **Objetivo:** quantificar o **gap de design** entre o teste Kaggle e um baralho de **estilo diferente**.
- **Método:** avaliar o modelo principal (com aug) no conjunto `data/raw/ood_design_web` (54 imagens, imagens limpas da web), remapeando rótulos pelo nome da pasta (`evaluate_ood`).
- **Resultado:** teste **0,9472** → **OOD design 0,5926 / F1 0,5741**; **gap de design ≈ 0,3546 (≈ 35 pp)**.
- **Interpretação:** a queda de ~35 pp mostra que o modelo **decorou parcialmente o estilo visual** do dataset de treino — não generaliza perfeitamente para outro design. **Ressalva crucial:** como as imagens OOD são **limpas**, este gap é um **LIMITE INFERIOR**; com **fotos reais** (gap de captura: luz, sombra, fundo, ângulo) a queda tende a ser **maior**. Medir isso é trabalho futuro (`docs/guia_coleta_baralho_real.md`).

### 7.4 Experimento 4 — Tentativa de reduzir o gap de design (**resultado negativo**)

- **Hipótese:** se o gap de ~35 pp vem de o modelo **decorar o estilo** do baralho de treino, então **diversificar o estilo** durante o treino deveria forçá-lo a aprender o conceito invariante (valor + naipe) e **subir a OOD**.
- **Método (duas alavancas combinadas):**
  1. **Segundo design no treino** — misturou-se um baralho de design distinto (**Kenney Playing Cards Pack**, licença **CC0**, 54 imagens) ao conjunto de treino via `ConcatDataset` (`src/extra_designs.py`, `extra_train_dirs` em `src/data.py`). Treino passou de 7.624 → **7.678** imagens. O OOD (Vector Cards) **permaneceu fora** do treino — sem vazamento.
  2. **Augmentation de aparência** (`design_aug=True`) — `ColorJitter(hue=0.5)`, `RandomGrayscale(p=0.20)`, `RandomPosterize(bits=4, p=0.30)`, `RandomAdjustSharpness(p=0.30)`, atacando especificamente cor/textura de arte (`src/data.build_transforms`).
- **Resultado:** treino **converge igual** (val **0,9962**, até ligeiramente acima do principal), mas a **OOD não se moveu**: **0,5926 / F1 0,5710** — **idêntica** ao baseline (mesmas **32/54** cartas certas).
- **Interpretação (por que empatou):** o número idêntico nas **três** configurações (com aug, sem aug, e agora com 2º design + aug de aparência) indica que os erros são **estruturais**, não de margem — o modelo acerta as **mesmas 32** cartas e erra as **mesmas 22**, independentemente do treino. Diversidade de estilo **genérica** (um baralho a mais + perturbação de cor) **não** aproxima o modelo do design-alvo específico do OOD. A val_acc subir para 99,6% mostra que o Kenney foi **absorvido sem esforço** — ou seja, não gerou a pressão de invariância pretendida.
- **Conclusão metodológica:** reduzir o gap de design de fato exigiria **(a)** treinar com um design **próximo do alvo**, ou **(b)** diversidade de designs **muito maior** (dezenas de baralhos), ou **(c)** atacar o gap de **captura** com fotos reais — não uma única intervenção de variação genérica. **O modelo entregue continua o principal** (`models/`); o checkpoint deste experimento (`models_ood/`) **não** o substitui, por não trazer ganho. Este é um **resultado negativo honesto** que **reforça** o achado do Experimento 3: o gap mede distância de design real, não falta de regularização.

---

## 8. Resultados consolidados e análise de erros

### 8.1 Tabela consolidada

| Configuração | Acc. teste (Kaggle) | F1-macro teste | Acc. OOD (design) | F1-macro OOD | Observações |
|---|---|---|---|---|---|
| Baseline: HOG + Reg. Logística | 0,7057 | 0,6977 | — | — | Piso de comparação clássico |
| EfficientNet-B0 — FE (congelado) | 0,3849 | 0,3630 | — | — | Features ImageNet fracas p/ cartas |
| **EfficientNet-B0 — FE+FT com aug** ← **principal** | **0,9472** | **0,9472** | **0,5926** | **0,5741** | Modelo entregue; gap de design ≈ 35 pp |
| EfficientNet-B0 — FE+FT sem aug | 0,9736 | 0,9734 | 0,5926 | 0,5678 | Maior no teste limpo; **mesma acc. OOD** (gap maior, ≈ 38 pp) |
| EfficientNet-B0 — 2º design (Kenney) + aug de aparência | — | — | 0,5926 | 0,5710 | **Exp. 4**: tentativa de reduzir o gap; OOD **inalterada** (resultado negativo) |

> Valores de validação relevantes: FE val 0,4528; FE+FT com aug val 0,9925; FE+FT sem aug val 0,9887; Exp. 4 (2º design + aug aparência) val **0,9962**. Referência da literatura (não é resultado deste trabalho): transfer learning em cartas tende a ~93–95% in-distribution — o modelo principal (94,7%) está alinhado.

**Leituras-chave:**

- O modelo principal **supera o baseline com folga** (94,7% vs 70,6%), evidenciando o valor do transfer learning (meta de "superar o baseline" atingida; metas de >90% acc e >0,90 F1 atingidas).
- **F1-macro = accuracy (0,9472)** no teste indica desempenho **homogêneo** entre as 53 classes — por isso **não** foi necessário usar *class weights* nem amostragem balanceada.
- A queda no OOD (≈35 pp) é o principal achado de **limitação de generalização** — e **três** intervenções distintas (com/sem aug de captura; 2º design + aug de aparência) a deixaram **inalterada** (sempre 0,5926), o que indica que o gap reflete **distância de design real**, não falta de regularização (ver Exp. 4).

### 8.2 Análise de erros e "confusões perigosas" (teste Kaggle)

- **Total de erros:** **14 em 265** (251 corretos).
- **Padrão dominante:** **trocas de VALOR dentro do mesmo naipe** — exemplos: `nine of diamonds → eight of diamonds`, `seven of clubs → eight of clubs`, `five of diamonds → three of diamonds`, `nine of spades → six of spades`. Ou seja, o modelo erra na **contagem de *pips*** de cartas numéricas próximas.
- **Nenhuma troca de naipe/cor** entre as confusões listadas — o modelo **lê bem o naipe**. Isso é um **ponto positivo de segurança**: erros de naipe (copas↔ouros, paus↔espadas) seriam os **mais críticos** no uso educacional/assistivo (mudam a regra do jogo / a leitura em voz alta), e quase não ocorreram.
- **Classe mais frágil: `joker`** — 2 de 5 imagens erradas (previstas como `ten`/`five of clubs`), provavelmente por **design atípico** e **poucas amostras**.

Artefatos gerados pelo notebook (a commitar em `reports/`): `confusion_matrix_test.png`, `confusion_matrix_ood.png`, `classification_report_test.csv`, `experimentos.csv`.

---

## 9. Jornada de desenvolvimento e lições aprendidas

Esta seção registra honestamente o **percurso** (não só o resultado final) — é especialmente valiosa para a defesa oral.

### 9.1 A saga do *undertraining*

- **Sintoma inicial:** a primeira configuração de fine-tuning (`lr_finetune=1e-5`, **5 épocas**) deixou o modelo **sub-treinado** — teste **~0,566**, **perdendo** para o baseline (0,706). Resultado embaraçoso para um modelo de deep learning.
- **Diagnóstico (a parte difícil):** as perdas de treino **e** de validação caíam de forma **monótona até a última época**. Esse é o assinatura de **undertraining** (o modelo ainda estava aprendendo quando o treino parou), **não** de um bug — o pipeline estava correto. A lição central foi **diferenciar "modelo ruim" de "treino interrompido cedo"**: a curva de perda diz qual dos dois é.
- **Correções iterativas:**
  1. `lr 1e-5 → 1e-4` e `5 → 12 épocas`: teste subiu para ~0,74 (ainda crescendo).
  2. **LR de pico 3e-4 + cosine annealing + 20 épocas:** teste **0,947** — convergência perto do teto.
- **Lição técnica:** com LR pequeno demais e poucas épocas, mesmo um pipeline correto produz um modelo aparentemente fraco. Aumentar o LR (com agendamento por cosseno para estabilidade no fim) e dar épocas suficientes destravou a tarefa.

### 9.2 Infraestrutura do Colab

- **Cota de GPU grátis:** esgota após horas de uso; contornado com uma **segunda conta Google** ou migração para **Kaggle**.
- **Runtime efêmero:** um reset apaga **tudo** em `/content` (repo clonado, modelos, figuras). **Lição: montar o Google Drive e fazer backup** de checkpoints e figuras — o notebook foi **blindado** para isso (`backup_para_drive` é chamada após cada etapa cara).
- **`files.download` instável:** pode **travar** quando o popup do navegador está bloqueado; preferir **salvar no Drive** (a célula 10b zipa `reports/` e copia para o Drive como alternativa confiável).

### 9.3 Decisões metodológicas honestas

- **Reportar o gap OOD** (e declará-lo como **limite inferior**) em vez de só a acurácia limpa.
- Montar um **OOD de design honestamente rotulado** (imagens limpas da web) em vez de **passar imagem da web como se fosse foto real**.
- **Não afirmar** que a augmentation "melhorou" os resultados: ela **piorou** ~2,6 pp in-distribution e **empatou** no OOD de design (mesma acurácia). Justificar a adoção do com-aug como **escolha de deploy** (robustez a captura no uso real), deixando explícito que esse benefício **não foi medido** (exige um OOD de fotos reais).

---

## 10. Ética, impacto e LGPD

A análise ética adota uma narrativa **defesa-vs-ataque**: para cada risco (ofensiva), uma defesa correspondente em escopo, técnica, dados ou documentação. (Fonte: `docs/03_etica_impacto.md`, `docs/MODEL_CARD.md`.)

- **Dual-use / trapaça (RTA).** Um classificador de cartas é o componente perceptivo de um sistema de **Real-Time Assistance** — **proibido** por plataformas como **PokerStars** e **GGPoker** (configura trapaça; em jogos presenciais a dinheiro pode ser fraude/crime). **Defesa:** declarar o uso como **PROIBIDO** (README, model card); **manter o escopo em classificação**, não em detecção de cena (não construir a varredura de mesa que tornaria o RTA prático); não embutir em dispositivos discretos.
- **Jogo problemático (dano social).** Ferramentas que reduzem o atrito de apostar podem agravar o *problem gambling*. **Defesa:** modo "**leitura assistiva carta por carta**", não "scan de mesa ao vivo"; enquadramento **educacional explícito**, sem sugestão de jogada sobre partidas a dinheiro.
- **Privacidade / LGPD (Lei 13.709/2018).** A câmera capta o **ambiente e terceiros** (rostos, crianças, documentos). **Defesa:** **processamento on-device** (a inferência roda localmente; nenhuma imagem enviada à nuvem), **não persistir imagens**, **minimização de dados** (enquadrar só a carta). Argumento de deploy relevante: a **NPU Intel** poderia rodar a inferência **on-device via OpenVINO**, reforçando privacidade e latência. Atenção redobrada por envolver **crianças e idosos**.
- **Viés do dataset (1 design de baralho).** O modelo pode ter alta acurácia in-distribution e **falhar fora** dela. **Defesa:** medir e reportar o **gap OOD** (conecta diretamente com o Experimento 3) em vez de só a acurácia limpa; *data augmentation*; análise de confusões perigosas.
- **Documentação ética como mitigação.** Publicar o artefato **sem declarar usos proibidos** é, em si, um risco. **Defesa:** `MODEL_CARD.md` com usos proibidos, limitações e métricas; licença **MIT** no código + declaração de uso aceitável; confirmar a licença "Other" do dataset antes de qualquer redistribuição.

---

## 11. Limitações e ameaças à validade

- **Val/teste pequenos (5 img/classe):** métricas **ruidosas** (±~20 pp por imagem na classe); diferenças finas entre configurações (ex.: com vs sem aug) podem **não ser significativas**. Por isso prioriza-se F1-macro e a matriz de confusão.
- **Treino em um único design de baralho:** viés de domínio forte; tende a falhar em outros designs, baralhos regionais, naipes estilizados — **medido** no Experimento 3.
- **OOD é limite inferior:** o conjunto de "design diferente" usa **imagens limpas**, então mede o **gap de design**, **não** o gap de **captura** (luz/sombra/fundo/ângulo de fotos reais). O gap real esperado é **maior** que os ~35 pp medidos.
- **Benefício da augmentation não comprovado:** na comparação com-vs-sem-aug, a augmentation **não melhorou** o OOD de design (mesma acurácia 0,5926; F1 0,574 vs 0,568). Como ela visa variação de **captura** e o OOD testa **design**, o ganho só seria observável num OOD de **fotos reais** — ainda não coletado.
- **Escopo restrito a classificação:** exige **recorte prévio** de uma carta; não detecta nem segmenta múltiplas cartas.
- **Fundos uniformes/padronização do dataset:** o modelo pode depender do contexto limpo e falhar com fundos complexos.
- **Sem garantia de calibração:** as probabilidades de softmax podem ser mal calibradas — não devem ser lidas como confiança real sem avaliação adicional.
- **Viés de pré-treino (ImageNet):** o backbone carrega vieses de um dataset não pensado para cartas (visível no FE congelado).
- **Divergência de defaults da CLI** (Seção 6): uma execução via `python -m src.train` com os defaults do `argparse` reproduziria o regime de **undertraining** (1e-5/5ep), diferente dos resultados reportados (que vêm do notebook, 3e-4/20ep).

---

## 12. Trabalhos futuros

- **Detecção com YOLO** (ex.: YOLOv8/YOLO11): estender de "1 carta recortada" para **detecção de múltiplas cartas em cena**, removendo a dependência do recorte manual — a evolução natural do escopo.
- **OOD com fotos reais (gap de captura):** fotografar um baralho físico sob luz/fundo/ângulo variados, seguindo `docs/guia_coleta_baralho_real.md` (com cuidados de LGPD), e medir o gap que o OOD de design não cobre.
- **OOD de fotos reais:** medir o gap de **captura** (fotografar um baralho físico) — é onde a augmentation deve mostrar valor, já que no OOD de **design** ela empatou.
- **Inferência on-device na NPU via OpenVINO:** exportar o modelo para rodar localmente na NPU Intel, reforçando privacidade (LGPD) e latência — bom argumento de deploy.
- **Mais designs no treino:** incorporar vários baralhos/estilos para reduzir o gap de design medido.
- **App educacional/assistivo:** evoluir a demo web (abaixo) com **leitura em voz alta** e exercícios de probabilidade, idealmente com **processamento local** (on-device).

---

## 12b. Demonstração web entregue

Além do código de treino/avaliação, o modelo principal foi **publicado como aplicação web** para tornar o projeto utilizável e demonstrável sem instalar nada:

- **URL pública:** <https://herickhuggingfaceaccount-cartas.hf.space>
- **Stack:** servidor **Flask** (`src/app.py`) servindo um *frontend* estático (HTML/CSS/JS, **sem build**); reaproveita `src.predict` e o checkpoint versionado `models/efficientnet_b0_best.pt`. Roda em **CPU**.
- **Funções:** envio por *upload*/arrastar-soltar e, no celular, **“Tirar foto”** (câmera) e **“Galeria”**; retorna o **top-5** com probabilidades; página **“Sobre o modelo”** lê as métricas reais de `reports/` (experimentos, acurácia/F1, OOD, classes mais difíceis). Layout **responsivo**.
- **Deploy:** **Hugging Face Spaces** (SDK *docker*), que builda a partir do `Dockerfile` (torch **CPU-only**); o checkpoint sobe via **Git LFS**. A Vercel foi descartada por não comportar o tamanho do PyTorch em função *serverless*.

> A demo serve para **ilustrar o uso**; as **métricas oficiais** do projeto são as medidas no Colab (Seções anteriores), não estatísticas de uso da página.

---

## 13. Banco de perguntas e respostas para a defesa

> Respostas curtas e corretas para as perguntas mais prováveis na banca.

**1. Por que o modelo *sem* augmentation deu MAIOR que o *com* augmentation?**
Porque o teste é **quase igual ao treino** (imagens limpas, mesmo design): a augmentation **regulariza** e custa um pouco de acurácia quando o teste é "fácil"/in-distribution (−2,6 pp). E no **OOD de design** ela **empatou** (mesma acurácia 0,5926). Motivo: a augmentation simula variação de **captura** (rotação/brilho/oclusão), não de **design** — por isso não ajuda neste OOD específico. Mantivemos o com-aug como escolha de **deploy** (no uso real, com fotos, espera-se ganho de robustez), declarando que isso **não foi medido** aqui.

**2. Por que o feature extraction (backbone congelado) vai tão mal (38%)?**
Porque as features pré-treinadas do **ImageNet** são **genéricas para objetos naturais** e fracas para cartas (símbolos finos, *pips*, tipografia). Congeladas, elas não capturam o que distingue uma carta da outra. O **fine-tuning** adapta essas features ao domínio e leva a acurácia de 0,385 para 0,947 (+56 pp).

**3. O que é cosine annealing e por que usar?**
É um **agendador de learning rate** que faz o LR decair do pico (3e-4) até ~0 seguindo uma curva cosseno ao longo das épocas. LR alto no início **acelera a convergência**; LR baixo no fim dá um **ajuste fino estável** (não fica oscilando no mínimo). Foi o que destravou a convergência após a saga de undertraining.

**4. Seu conjunto OOD é foto real?**
**Não.** São **imagens limpas de licença livre da web** (baralho de **design diferente**, domínio público). Ele mede o **gap de design** — se o modelo aprendeu o conceito da carta ou decorou o estilo do dataset. **Não** mede o gap de **captura** (luz/sombra/fundo de fotos reais). Por serem limpas, o gap medido (~35 pp) é um **limite inferior** do esperado no mundo real. Foi rotulado honestamente, nunca apresentado como foto real.

**5. Por que F1-macro e não só accuracy?**
Porque val/teste têm **5 img/classe** → a accuracy é **ruidosa** (cada imagem vale ~20 pp na classe). O F1-macro trata **todas as 53 classes igualmente** (não deixa classes frequentes mascararem as raras) e, junto da matriz de confusão, revela **padrões de erro** mais robustos que o número isolado.

**6. Por que o `joker` erra mais?**
Porque tem **design atípico** (não segue o padrão valor+naipe) e o modelo viu **poucas amostras** desse tipo de imagem. Foi a classe mais frágil: 2 de 5 imagens erradas no teste.

**7. Quais são as "confusões perigosas" e o modelo as comete?**
São erros que trocam **naipe/cor** (copas↔ouros) — os mais graves no uso educacional/assistivo, pois mudam a regra/leitura. **Quase não ocorreram**: dos 14 erros, predominam **trocas de valor dentro do mesmo naipe** (contagem de *pips*), e **nenhuma troca de naipe** entre as confusões listadas. O modelo **lê bem o naipe**.

**8. Como garantem reprodutibilidade?**
`set_seed(42)` (random/numpy/torch + cuDNN determinístico), `requirements.txt` com versões pinadas (exceto torch/torchvision, que o Colab já fornece com a CUDA certa), **split oficial fixo** e **checkpoint autocontido** (pesos + classes + config + histórico). Treino documentado em GPU T4 no Colab.

**9. Por que EfficientNet-B0 e não ResNet?**
Por **equilíbrio acurácia/custo**: leve para a GPU T4 grátis e para inferência em CPU, com forte desempenho via *compound scaling*. O código suporta ResNet18 e MobileNet-V3 como alternativas, mas a B0 foi a escolhida.

**10. O modelo serve para apostas / pôquer online?**
**Não — é uso PROIBIDO.** A tecnologia é dual-use e poderia virar Real-Time Assistance (RTA), banido por PokerStars/GGPoker e potencialmente fraudulento. Mantivemos o escopo em **classificação** (não detecção de cena), declaramos o uso proibido no model card e desenhamos o uso como **leitura assistiva carta por carta**, não scan de mesa ao vivo.

**11. Por que não usaram flip na augmentation?**
Porque cartas têm **orientação** (índices e símbolos). Espelhar geraria imagens **irreais ou ambíguas**, prejudicando o aprendizado de naipe/valor.

**12. Qual o gap de design e o que ele significa?**
Teste 0,9472 → OOD design 0,5926, **gap ≈ 35 pp**. Significa que o modelo **decorou parcialmente o estilo** do dataset de treino e não generaliza perfeitamente para outro design. É a principal limitação de generalização medida.

**13. Por que treino em 2 fases em vez de fine-tuning direto?**
Para **estabilizar**: a Fase 1 (cabeça nova, backbone congelado) ajusta o classificador a features prontas, com LR alto (1e-3) e sem risco de destruir o backbone; só então a Fase 2 descongela e ajusta tudo com LR menor (3e-4) e cosine annealing. Reduz o risco de "estragar" os pesos pré-treinados logo no início.

**14. Por que early stopping com paciência 6 (e não 2 ou 3)?**
Porque a validação tem só 5 img/classe (**ruidosa**): uma paciência pequena pararia o treino numa flutuação ruim. Paciência 6 evita parada precoce, guardando sempre o melhor checkpoint por acurácia de validação.

**15. A diferença de 2,6 pp (com vs sem aug) é significativa?**
Provavelmente **não**, dado o tamanho do teste (5 img/classe, ±~20 pp por imagem na classe). Por isso evitamos overclaiming e não tratamos essa diferença como conclusiva — é mais um indício de regularização que custa pouco in-distribution.

**16. Como o baseline (HOG+LogReg) funciona e por que incluí-lo?**
HOG descreve **forma/bordas** (gradientes orientados) da carta em 128×128 cinza; a Regressão Logística separa as 53 classes. É um **piso de comparação** barato (CPU) que mostra o ganho do deep learning: 70,6% (baseline) vs 94,7% (modelo principal).

**17. O que está pendente no projeto?**
O **OOD do modelo sem augmentation** (célula 8b) e o **commit das figuras** de `reports/` (matrizes de confusão, `experimentos.csv`, `classification_report_test.csv`). Opcional/futuro: fotos reais (gap de captura) e confirmar a licença "Other" do dataset.

**18. Por que normalizar com média/desvio do ImageNet?**
Porque o backbone foi **pré-treinado** com essa normalização; casar a distribuição de entrada com a do pré-treino é necessário para as features se comportarem como esperado.

**19. Por que 224×224?**
É a **resolução de entrada nativa** da EfficientNet-B0 (e o tamanho em que o dataset já vem). Mantemos um resize explícito por robustez (especialmente para o OOD, de proporções variadas).

**20. O que é o "limite inferior" do gap OOD?**
Como o OOD usa **imagens limpas** (mais parecidas com o treino do que fotos reais seriam), o gap medido (~35 pp) é o **menor** que esperaríamos. Com fotos reais (luz, sombra, fundo, ângulo), o gap tende a ser **maior** — por isso reportamos ~35 pp como **piso**, não como o número final do mundo real.

---

## 14. Glossário rápido

- **Transfer learning:** reaproveitar um modelo pré-treinado (aqui, no ImageNet) e adaptá-lo a uma nova tarefa, em vez de treinar do zero.
- **Feature extraction (extração de características):** usar o backbone **congelado** como extrator fixo de features e treinar apenas a cabeça classificadora.
- **Fine-tuning (ajuste fino):** **descongelar** o backbone e re-treinar a rede inteira (com LR baixo) para adaptar as features ao novo domínio.
- **Backbone / cabeça (head):** o "corpo" convolucional que extrai features vs. a camada final que mapeia features → classes.
- **OOD (out-of-distribution):** dados de **distribuição diferente** da de treino; aqui, um baralho de design distinto, para medir generalização.
- **Gap de design vs gap de captura:** queda de desempenho por **estilo de carta** diferente (medido) vs por **condições de foto** reais — luz/sombra/fundo/ângulo (não medido; trabalho futuro).
- **F1-macro:** média **não ponderada** dos F1 por classe (média harmônica de precisão e recall); trata todas as classes igualmente.
- **Matriz de confusão:** tabela classe-verdadeira × classe-prevista; revela quais pares de classes o modelo confunde.
- **Data augmentation:** transformações aleatórias no treino (rotação, jitter de cor, affine, *erasing*) para aumentar diversidade e robustez. **Flips evitados** (cartas têm orientação).
- **HOG (Histogram of Oriented Gradients):** descritor clássico que resume a forma/bordas pela distribuição de gradientes orientados.
- **Cosine annealing:** agendamento que decai o learning rate do pico até ~0 seguindo uma curva cosseno.
- **Early stopping:** parar o treino quando a métrica de validação para de melhorar (paciência = nº de épocas toleradas sem melhora), guardando o melhor checkpoint.
- **AdamW:** otimizador Adam com *weight decay* desacoplado (regularização).
- **Undertraining (sub-treino):** o modelo para de treinar **antes** de convergir (perdas ainda caindo) — diferente de um modelo intrinsecamente ruim ou de um bug.
- **RTA (Real-Time Assistance):** assistência em tempo real durante o jogo; **proibida** por plataformas de pôquer — uso vedado deste projeto.
- **LGPD:** Lei Geral de Proteção de Dados (Lei 13.709/2018); motiva minimização de dados e processamento on-device.
- **On-device / NPU / OpenVINO:** rodar a inferência **localmente** no dispositivo; NPU é a unidade de processamento neural (ex.: Intel), e OpenVINO um runtime que a aproveita.
