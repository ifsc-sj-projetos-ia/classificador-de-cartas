# 2.2 Dados e Preparação

## 1. Fonte dos dados

### 1.1 Dataset principal (in-distribution)

O dataset principal do projeto é o **"Cards Image Dataset-Classification"**, de autoria de **gpiosenka**, disponibilizado publicamente na plataforma Kaggle.

- **URL:** https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification
- **Natureza:** conjunto de imagens de cartas de baralho já recortadas (uma carta por imagem), padronizadas em **224 × 224 × 3** (RGB), organizadas em **53 classes** e com **split pronto** de treino/validação/teste.
- **Volume:** 7.624 imagens de treino, 265 de validação e 265 de teste (5 imagens por classe em validação e em teste).
- **Licença:** marcada como **"Other"** no Kaggle. Por isso, **antes de qualquer redistribuição** das imagens (por exemplo, anexar o dataset ao repositório), é necessário **confirmar os termos de licença** junto à página oficial do dataset. No repositório deste projeto, o diretório `data/` contém **apenas instruções de download**, não as imagens em si, justamente para evitar redistribuição não autorizada.

Este dataset cumpre o escopo definido para o projeto: **classificação** (1 carta recortada → 1 rótulo). A detecção de múltiplas cartas em cena não faz parte do escopo e está prevista como trabalho futuro.

### 1.2 Conjunto out-of-distribution (OOD)

Para avaliar a **capacidade de generalização** do modelo além do design de baralho visto no treino, é construído um **conjunto OOD de "design diferente"**, composto por **imagens limpas de um baralho de estilo distinto**, de licença livre, baixadas da web (script reprodutível `src/ood_design.py`).

- **O que este conjunto mede (e o que NÃO mede):** ele mede o **gap de design** — se o modelo aprendeu o *conceito* de cada carta (ex.: "rei de espadas") ou apenas decorou o estilo visual específico do dataset gpiosenka. Ele **não** mede o **gap de condições de captura** (iluminação variável, sombras, fundos reais, ângulos, reflexos, oclusão), porque são renders digitais limpos, e não fotos do mundo real. Esse segundo gap exige fotografar um baralho físico e fica registrado como **trabalho futuro** (ver `docs/guia_coleta_baralho_real.md`).
- **Transparência:** estas imagens **não** são fotos tiradas pelos autores; são assets públicos. Por serem "limpas", em alguns aspectos são até mais parecidas com o domínio de treino do que fotos reais seriam — portanto o *gap* medido é um **limite inferior** do gap esperado em uso real. Isso é declarado explicitamente no relatório para evitar *overclaiming*.
- **Procedência/licença:** baralho derivado do projeto *Vector Playing Cards* (domínio público), via repositório *playing-cards-assets* (Howard Yeh, licença MIT). Uso estritamente acadêmico/diagnóstico.
- **Uso:** este conjunto **não** participa de treino nem de validação. É reservado exclusivamente para o **Experimento 3** (avaliação OOD: teste Kaggle vs. baralho de design diferente).
- **Trabalho futuro — OOD com fotos reais:** o caminho padrão-ouro continua sendo fotografar um baralho físico próprio sob condições reais, com os cuidados de **minimização de dados / LGPD** (capturar **apenas as cartas**, sem rostos, terceiros ou ambientes identificáveis; processamento on-device). Guia completo em `docs/guia_coleta_baralho_real.md`.

## 2. Descrição das variáveis e dos rótulos

### 2.1 Variável de entrada (X)

A entrada do modelo é uma **imagem RGB** de dimensão **224 × 224 × 3**:

- 3 canais de cor (vermelho, verde, azul);
- valores de pixel originalmente no intervalo inteiro [0, 255], convertidos para ponto flutuante e normalizados na etapa de preparação (ver Seção 3).

### 2.2 Variável-alvo (y) e taxonomia das classes

O alvo é um **rótulo categórico único** (problema de **classificação multiclasse**, com classes mutuamente exclusivas) entre **53 classes**, definidas por:

> **53 classes = 13 valores × 4 naipes + 1 coringa (joker)**

| Componente | Categorias |
|---|---|
| **Valores (13)** | ás (A), 2, 3, 4, 5, 6, 7, 8, 9, 10, valete (J), dama (Q), rei (K) |
| **Naipes (4)** | copas (hearts), ouros (diamonds), paus (clubs), espadas (spades) |
| **Especial (1)** | coringa / joker |

A combinação 13 × 4 = 52 cartas mais o coringa totaliza as 53 classes. Cada classe é representada por um diretório próprio no dataset (estrutura "uma pasta por classe", padrão `ImageFolder` do torchvision), o que permite derivar o mapeamento `classe → índice` de forma automática e reprodutível.

### 2.3 Estrutura semântica dos rótulos (relevante para a análise de erros)

Embora o modelo trate o problema como 53 classes "planas", cada carta (exceto o coringa) carrega **dois atributos latentes**: **valor** e **naipe**. Essa decomposição é usada na avaliação para distinguir **confusões perigosas** (Seção 5 do documento de avaliação): trocar o **valor** (ex.: 8 ↔ 9) ou o **naipe** (ex.: copas ↔ ouros, ambos vermelhos; paus ↔ espadas, ambos pretos) tem implicações distintas em uma ferramenta educacional de ensino de regras e probabilidade.

## 3. Etapas de limpeza e preparação

### 3.1 Verificação de integridade

Antes do treino, todas as imagens passam por verificação de integridade:

- abertura/decodificação de cada arquivo (descartar arquivos corrompidos ou ilegíveis);
- confirmação do formato e do modo de cor (conversão para **RGB**, garantindo 3 canais — imagens em escala de cinza ou com canal alfa são convertidas);
- conferência da contagem de imagens por classe e da existência das 53 pastas em cada split (treino/validação/teste);
- checagem das dimensões esperadas (224 × 224); imagens fora do padrão são redimensionadas (ver 3.2).

### 3.2 Redimensionamento (resize)

As imagens já são fornecidas em **224 × 224**, que é exatamente a resolução de entrada esperada pela **EfficientNet-B0**. Ainda assim, mantém-se um passo explícito de **resize para 224 × 224** no pipeline, por robustez (uniformizar eventuais imagens fora do padrão e padronizar também o conjunto OOD, cujas fotos têm resoluções e proporções variadas).

### 3.3 Normalização (média/desvio do ImageNet)

Por se tratar de **transfer learning** a partir de pesos pré-treinados no **ImageNet**, os tensores de entrada são normalizados com a **média e o desvio-padrão do ImageNet**, casando a distribuição de entrada com aquela vista durante o pré-treinamento:

- **média (mean):** `[0.485, 0.456, 0.406]`
- **desvio-padrão (std):** `[0.229, 0.224, 0.225]`

O pipeline aplica, na ordem: conversão para tensor (escalando [0, 255] → [0, 1]) e, em seguida, a normalização canal a canal com os valores acima. **A mesma normalização** é aplicada a treino, validação, teste e ao conjunto OOD.

### 3.4 Data augmentation (apenas no treino)

A *data augmentation* é aplicada **somente ao conjunto de treino** — validação, teste e OOD usam apenas resize + normalização, para que as métricas reflitam o desempenho sobre imagens não perturbadas. As transformações foram escolhidas para simular variações plausíveis de captura sem destruir a informação discriminante (valor e naipe da carta):

- **rotação** leve (ex.: ±10° a ±15°), simulando cartas não perfeitamente alinhadas;
- **color jitter** (brilho, contraste, saturação e leve matiz), simulando iluminação e balanço de branco distintos — relevante para reduzir o gap em relação às fotos OOD;
- **transformação affine leve** (pequenas translações/escala), simulando enquadramentos imperfeitos;
- **random erasing**, ocluindo aleatoriamente pequenas regiões para induzir robustez a oclusões parciais.

Decisões de cuidado na augmentation:

- **evitar flip horizontal/vertical agressivo**: índices e símbolos de carta têm orientação, e espelhamentos podem gerar imagens não realistas ou ambíguas;
- intensidades **moderadas**, para não inviabilizar a leitura do naipe/valor.

O **impacto da augmentation é objeto do Experimento 2** (com vs. sem data augmentation), portanto o pipeline é parametrizado para ligar/desligar essas transformações via configuração (`src/config.py`).

### 3.5 Tratamento de classes

- **Mapeamento de rótulos:** as 53 classes são mapeadas para índices inteiros de forma determinística (ordem alfabética das pastas via `ImageFolder`), e esse mapeamento é registrado para garantir consistência entre treino, avaliação e inferência (`src/predict.py`).
- **Balanceamento:** o conjunto de treino (~144 imagens/classe em média, 7.624/53) é aproximadamente equilibrado entre as 53 classes. **Decisão registrada:** **não** foi necessário usar *class weights* nem amostragem balanceada — no teste o **F1-macro (0,947) ficou igual à accuracy (0,947)**, indicando desempenho homogêneo entre as classes (nenhuma classe rara dominando o erro).
- **Reprodutibilidade:** toda a montagem dos *dataloaders* (embaralhamento, ordem, divisão) é feita sob `set_seed(42)` (`src/seed.py`), com `requirements.txt` pinado, garantindo execução reprodutível no Google Colab (GPU T4).

## 4. Estratégia de divisão treino / validação / teste

### 4.1 Split pronto do dataset

O dataset gpiosenka já vem com **divisão oficial pronta**, que será adotada sem modificação para preservar a comparabilidade com outros trabalhos que usam a mesma base:

| Split | Imagens | Imagens por classe (aprox.) | Uso |
|---|---|---|---|
| **Treino** | 7.624 | ~144 (média) | ajuste dos pesos do modelo |
| **Validação** | 265 | 5 | seleção de hiperparâmetros / *early stopping* |
| **Teste** | 265 | 5 | avaliação final in-distribution |

### 4.2 Papel de cada conjunto

- **Treino:** usado nas duas fases do transfer learning (fase 1 — backbone congelado/*feature extraction*; fase 2 — *fine-tuning* do topo com LR baixo) e também no baseline HOG + Regressão Logística.
- **Validação:** usado para escolha de hiperparâmetros, monitoramento de *overfitting* e critério de parada. **Não** é usado para reportar o resultado final.
- **Teste:** usado **uma única vez**, ao final, para reportar as métricas in-distribution (accuracy, F1 macro, matriz de confusão).

### 4.3 Ressalva: validação e teste pequenos

Com apenas **5 imagens por classe** em validação e em teste (265 no total), as **métricas são estatisticamente ruidosas**: o erro/acerto de **uma única imagem por classe altera a accuracy daquela classe em 20 pontos percentuais**. Implicações assumidas no projeto:

- a accuracy **medida** (**94,7%** no teste, modelo principal) deve ser lida com **intervalo de incerteza amplo** (5 img/classe → cada imagem vale ~20 pp na classe);
- privilegia-se o **F1 macro** e a **matriz de confusão** sobre a accuracy isolada, e a análise enfatiza **padrões de confusão** (quais pares de classes se confundem) mais do que o valor pontual da métrica;
- evita-se overclaiming: diferenças pequenas entre configurações (ex.: com vs. sem augmentation) podem **não ser significativas** dado o tamanho dos conjuntos.

### 4.4 Conjunto OOD separado

O **conjunto OOD** (baralho de design diferente, imagens limpas da web) é mantido **totalmente separado** dos três splits acima. Ele **não** influencia treino nem validação e serve unicamente para medir **generalização out-of-distribution** no Experimento 3, quantificando o **gap de design** entre o teste Kaggle e um baralho de estilo distinto. O gap de **condições de captura** (fotos reais) não é coberto por este conjunto e fica como trabalho futuro.

## 5. Vieses e limitações da base

O dataset de origem viabiliza o projeto, mas impõe limitações que **enviesam o modelo** e que devem ser declaradas explicitamente (documentação ética é um requisito do projeto):

- **Um único tipo/estilo de baralho:** o modelo é treinado essencialmente sobre **um design de cartas**. Baralhos com **fontes, símbolos, cores, tamanhos ou ilustrações diferentes** (ex.: baralhos regionais, infantis, de pôquer premium, naipes em quatro cores) tendem a ser mal classificados. Isso é **especialmente crítico para uma ferramenta educacional**, que pode ser usada com qualquer baralho disponível em casa ou na escola.
- **Fundos uniformes e enquadramento padronizado:** as imagens têm cartas centralizadas sobre fundos limpos. O modelo pode aprender a depender desse contexto e falhar com **fundos complexos** (mesa, mãos, padrões).
- **Sinteticidade / padronização excessiva:** a aparência muito uniforme das imagens reduz a diversidade visual. O modelo pode ter **boa métrica no teste Kaggle e desempenho substancialmente pior em fotos reais** — exatamente a hipótese investigada no Experimento 3.
- **Ausência de variação realista de iluminação e oclusão:** não há sombras fortes, reflexos, baixa luz, desfoque de movimento, nem cartas parcialmente cobertas. A *data augmentation* (color jitter, random erasing, affine) **mitiga parcialmente**, mas não substitui dados reais.
- **Validação e teste muito pequenos (5 img/classe):** conforme a Seção 4.3, limita a confiabilidade estatística das métricas e a capacidade de detectar diferenças finas entre configurações.
- **Generalização limitada por construção:** como o treino reflete uma distribuição estreita, **não se deve assumir** que a accuracy in-distribution se transfira para uso no mundo real. Esta limitação será **medida** (gap OOD) e **declarada** na documentação e no README, junto aos **usos não previstos/proibidos** (ex.: assistência em jogos de azar/apostas — risco de *dual-use* / RTA).
- **Possível desbalanceamento residual / qualidade de rótulos:** o treino é aproximadamente equilibrado e a verificação de integridade do pipeline (conversão p/ RGB, descarte de arquivos ilegíveis) não revelou impedimentos — o modelo atingiu 94,7% no teste. Eventual ruído residual de rótulos não foi significativo o bastante para limitar esse resultado in-distribution.

> **Síntese:** o dataset é adequado como **prova de conceito acadêmica** de classificação de cartas via transfer learning, mas suas limitações (baralho único, fundos uniformes, conjuntos de avaliação pequenos) **restringem a generalização**. Por isso o projeto inclui um conjunto **OOD próprio** e adota uma postura de **reporte honesto das métricas e dos riscos**, em vez de prometer desempenho garantido em qualquer baralho ou condição de captura.