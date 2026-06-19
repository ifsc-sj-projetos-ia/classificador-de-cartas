# Definição do Problema e Requisitos

**Projeto:** Classificador de Cartas de Baralho por Visão Computacional
**Disciplina:** Introdução à Ciência de Dados (ICD) / ADS
**Documento:** Item 2.1 — Definição do Problema e Requisitos
**Data:** 2026-06-01

---

## 1. Descrição do Problema e Motivação

### 1.1 O problema técnico

O objetivo deste projeto é construir um sistema de **classificação de cartas de baralho** por visão computacional. Dada **uma única carta já recortada** (imagem RGB), o sistema deve retornar um **rótulo entre 53 classes possíveis** — as 52 cartas de um baralho francês padrão (4 naipes × 13 valores) mais 1 coringa (*joker*).

Trata-se, portanto, de um problema de **classificação multiclasse de imagem única**. A **detecção** de múltiplas cartas em uma cena (localização + recorte automático de várias cartas simultaneamente) está **explicitamente fora do escopo** desta entrega e é tratada como trabalho futuro.

A abordagem adotada combina **transfer learning com EfficientNet-B0** (PyTorch + torchvision) como modelo principal e um **baseline clássico HOG + Regressão Logística** (scikit-learn) para comparação, permitindo evidenciar o ganho do aprendizado profundo sobre uma técnica tradicional de extração de características.

### 1.2 Motivação e impacto humano/social

O eixo de impacto primário do projeto é o de **ferramenta educacional**. Um classificador confiável de cartas pode servir como base para recursos didáticos voltados ao **ensino de probabilidade, raciocínio combinatório, regras de jogos e matemática elementar**, com público potencial entre **crianças e idosos**. Exemplos de uso pedagógico:

- Apoiar atividades de **ensino de probabilidade** (calcular chances de obter determinada carta/mão), tornando conceitos abstratos tangíveis a partir de cartas físicas reconhecidas pela câmera.
- Auxiliar no **aprendizado de regras de jogos de carta** e na verificação de jogadas em contextos de estudo.
- Funcionar como **objeto de ensino da própria visão computacional**: por ser um problema visualmente intuitivo (qualquer pessoa sabe ler uma carta), é um excelente exemplo didático para introduzir transfer learning, métricas de classificação e avaliação de generalização.

Como impactos **secundários**, o sistema abre caminho para:

- **Acessibilidade para pessoas com deficiência visual**: a leitura em voz alta da carta identificada (ex.: "Dama de Copas") pode apoiar a autonomia em jogos e atividades cotidianas.
- **Pesquisa e benchmark em visão computacional**: o pipeline reprodutível (dataset público, seed fixa, requisitos pinados) serve como base comparável para estudos de transfer learning e robustez a domínio.

### 1.3 Considerações de risco que motivam requisitos

O mesmo tipo de tecnologia tem **potencial de uso indevido** (*dual-use*), notadamente em **trapaça em jogos de azar e apostas** (sistemas de *Real-Time Assistance*, proibidos por plataformas como PokerStars e GGPoker) e na **facilitação do jogo problemático**, cujo dano social é documentado. Há ainda riscos de **viés/limitação do dataset** (treinado em um único design de baralho) e de **privacidade** (a câmera pode capturar o ambiente e terceiros). Esses riscos não são apenas observações éticas: eles **originam requisitos não-funcionais concretos** (Seção 4), incluindo declaração explícita de usos proibidos, minimização de dados e processamento *on-device*.

---

## 2. Público-Alvo e Usuários

| Público | Perfil | Necessidade principal |
|---|---|---|
| **Educadores e estudantes** (primário) | Professores, monitores, alunos de ensino fundamental e educação de adultos/idosos | Ferramenta didática confiável para ensinar probabilidade, regras de jogos e matemática |
| **Aprendizes de visão computacional** (primário) | Estudantes de ICD/ADS e áreas afins | Exemplo reprodutível e bem documentado de transfer learning aplicado à classificação de imagens |
| **Pessoas com deficiência visual** (secundário) | Usuários que se beneficiam de leitura em voz alta | Identificação auditiva da carta com autonomia |
| **Pesquisadores/comunidade técnica** (secundário) | Quem precisa de baseline ou benchmark | Pipeline comparável e código aberto (licença MIT) |

**Usos explicitamente NÃO endossados:** assistência em tempo real para apostas/jogos de azar a dinheiro, ou qualquer aplicação que facilite trapaça ou jogo problemático.

---

## 3. Requisitos Funcionais (RF)

| ID | Requisito | Critério de aceitação |
|---|---|---|
| **RF01** | Receber como entrada **uma imagem de uma única carta recortada** (RGB). | O sistema aceita formatos comuns (JPG/PNG) e redimensiona internamente para 224×224×3. |
| **RF02** | Retornar **a classe predita entre as 53 classes** (52 cartas + coringa). | A saída inclui o rótulo legível (ex.: "Ás de Espadas") e o índice da classe. |
| **RF03** | Fornecer predição **top-k** com probabilidades associadas. | Para um *k* configurável (padrão *k*=3), retorna as classes mais prováveis e suas confianças (softmax). |
| **RF04** | Realizar **inferência em CPU**, sem exigir GPU. | `predict.py` executa a classificação de uma imagem em máquina sem GPU em tempo razoável (ordem de poucos segundos). |
| **RF05** | Oferecer **modo de treino reprodutível** (fase 1: backbone congelado / *feature extraction*; fase 2: *fine-tuning* do topo com LR baixo). | `train.py` reproduz os resultados com `set_seed(42)`; notebook Colab documentado. |
| **RF06** | Treinar e avaliar o **baseline HOG + Regressão Logística** para comparação. | `baseline.py` gera métricas comparáveis às do modelo principal. |
| **RF07** | Avaliar o modelo gerando **métricas e artefatos** (accuracy, F1 macro, matriz de confusão). | `evaluate.py` produz relatórios e figuras em `reports/`. |
| **RF08** | Suportar **avaliação OOD** sobre um conjunto externo rotulado (na entrega: baralho de **design diferente**, imagens da web; fotos reais como trabalho futuro). | O pipeline aceita um diretório externo de imagens rotuladas e reporta as métricas separadamente do teste Kaggle. |
| **RF09** | Permitir **carregar um modelo treinado salvo** para inferência. | Pesos persistidos em `models/` são carregados sem necessidade de re-treino. |

---

## 4. Requisitos Não-Funcionais (RNF)

### 4.1 Privacidade e LGPD
- **RNF01 — Processamento on-device:** a inferência deve poder rodar **localmente, sem envio de imagens a servidores externos**, reduzindo exposição de dados pessoais.
- **RNF02 — Minimização de dados:** o sistema deve operar sobre a **carta recortada**, não exigindo nem armazenando imagens do ambiente ou de terceiros. Nenhuma imagem é retida além do necessário para a predição.
- **RNF03 — Conformidade LGPD:** a documentação deve orientar que, caso o usuário capture imagens com câmera, o tratamento siga princípios de finalidade, necessidade e minimização (Lei nº 13.709/2018).

### 4.2 Ética e uso responsável
- **RNF04 — Declaração de usos proibidos:** o `README.md` e a documentação ética em `docs/` devem **declarar explicitamente** que o projeto **não se destina** a assistência em apostas/jogos de azar a dinheiro e que tal uso é desencorajado/proibido.
- **RNF05 — Transparência de limitações:** documentar que o modelo é treinado em **um único design de baralho** e pode falhar com outros designs, iluminação adversa, oclusão ou cartas fora de distribuição.
- **RNF06 — Licenciamento responsável:** confirmar a licença do dataset (marcada como "Other" no Kaggle) **antes de qualquer redistribuição**; o código do projeto é licenciado sob **MIT**.

### 4.3 Desempenho
- **RNF07 — Latência de inferência:** classificação de uma carta em **CPU** em tempo interativo (poucos segundos por imagem).
- **RNF08 — Viabilidade de treino:** o treino completo deve caber em **Google Colab com GPU T4 gratuita**.

### 4.4 Usabilidade
- **RNF09 — Saída legível:** rótulos apresentados em linguagem natural (ex.: "Dama de Copas"), adequados a leitura em voz alta (apoio à acessibilidade).
- **RNF10 — Interface simples:** uso por linha de comando direto (`predict.py <imagem>`), com instruções claras no README.

### 4.5 Reprodutibilidade
- **RNF11 — Seed fixa:** `set_seed(42)` aplicada a todas as fontes de aleatoriedade relevantes.
- **RNF12 — Ambiente pinado:** `requirements.txt` com versões fixadas; notebook Colab versionado em `notebooks/`.
- **RNF13 — Organização do repositório:** estrutura modular (`src/`, `models/`, `data/`, `docs/`, `reports/`) com responsabilidades separadas por arquivo.

---

## 5. Métricas de Sucesso

A avaliação combina métricas quantitativas e análise qualitativa de erros. **Resultados medidos (Colab, GPU T4, `set_seed(42)`):** **94,7%** de acurácia e **0,947** de F1-macro no teste Kaggle, superando o baseline (70,6%) e atendendo às metas abaixo; no OOD de design a acurácia cai para **~59%** (gap ≈ 35 pp). As metas abaixo definiram os critérios de aceitação.

### 5.1 Métricas adotadas
- **Accuracy** (acurácia global) no conjunto de teste Kaggle.
- **F1 macro** (média não ponderada por classe), importante por haver poucas amostras por classe na validação/teste.
- **Matriz de confusão** completa (53 × 53) para inspeção visual de erros.
- **Análise de "confusões perigosas":** identificação de erros que trocam **naipe** ou **valor** (ex.: confundir 6 com 9, ou Copas com Ouros), por serem os mais críticos em aplicações de leitura/educação.
- **Accuracy OOD:** desempenho num **baralho de design diferente** (imagens limpas da web), medindo o **gap de design** em relação ao teste Kaggle. O gap de **condições de captura** (fotos reais) é trabalho futuro.

### 5.2 Metas concretas

| Métrica | Meta | Justificativa |
|---|---|---|
| Accuracy no teste Kaggle | **> 90%** (esperado ~93–95%) | Limiar de utilidade prática para o caso educacional |
| F1 macro no teste Kaggle | **> 0,90** | Garante bom desempenho em todas as 53 classes, não só nas frequentes |
| Comparação com baseline | Modelo principal **supera** o HOG + Regressão Logística em accuracy e F1 macro | Evidencia o valor do transfer learning |
| Confusões perigosas | **Minimizar** trocas de naipe/valor; documentar e analisar as remanescentes | Erros críticos para leitura assistiva e ensino |
| Accuracy OOD | **Reportar o gap** entre teste Kaggle e o baralho de design diferente | A meta é mensurar e discutir a queda ao trocar o design, não necessariamente atingir o mesmo patamar (gap de captura fica como trabalho futuro) |

### 5.3 Experimentos que sustentam as métricas
1. **Feature extraction (backbone congelado) vs. fine-tuning do topo** — impacto da estratégia de transfer learning.
2. **Com vs. sem data augmentation** — impacto da aumentação de dados na generalização.
3. **Avaliação OOD** — teste Kaggle vs. baralho de **design diferente** (web), quantificando o *gap de design* (fotos reais / gap de captura: trabalho futuro).

---

> **Nota de integridade:** todos os números de desempenho citados (ex.: ~93–95%) são **expectativas/placeholders**. Os resultados reais serão inseridos em `reports/` após a execução do treino e da avaliação no Google Colab.