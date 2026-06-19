# Model Card — Classificador de Cartas de Baralho (EfficientNet-B0)

> Documento no estilo Hugging Face / Google Model Cards. Projeto acadêmico (ICD/ADS). **Métricas medidas no Google Colab (GPU T4), `set_seed(42)`** e preenchidas abaixo. Modelo principal: EfficientNet-B0 (feature extraction + fine-tuning, **com** data augmentation).

---

## 1. Detalhes do Modelo

| Campo | Descrição |
|---|---|
| **Nome do modelo** | Classificador de Cartas de Baralho (Card Image Classifier) |
| **Versão** | 1.0.0 (treinado e avaliado) |
| **Tipo de tarefa** | Classificação de imagem — *single-label*, multiclasse |
| **Arquitetura** | Transfer learning com **EfficientNet-B0** (backbone pré-treinado em ImageNet) + cabeça de classificação linear adaptada para **53 classes** |
| **Framework** | PyTorch + torchvision |
| **Entrada** | 1 imagem RGB de **uma carta já recortada**, redimensionada para 224×224×3, normalizada com a média/desvio do ImageNet |
| **Saída** | Distribuição de probabilidade sobre 53 classes (52 cartas + 1 coringa/*joker*); rótulo previsto = `argmax` |
| **Classes (53)** | 13 valores (Ás, 2–10, Valete, Dama, Rei) × 4 naipes (copas, ouros, paus, espadas) + coringa/*joker* |
| **Estratégia de treino** | **Fase 1:** *feature extraction* (backbone congelado, treino apenas da cabeça). **Fase 2:** *fine-tuning* do topo da rede com *learning rate* baixo |
| **Modelo de baseline** | HOG + Regressão Logística (scikit-learn) — referência clássica para comparação |
| **Escopo** | **Apenas classificação** (1 carta recortada → 1 rótulo). **Detecção** (várias cartas em cena, localização) é trabalho futuro e **não** é suportada |
| **Autor / responsável** | Nícolas Pereira (nicolas.pereira@way2.com.br) — projeto acadêmico ICD/ADS |
| **Licença do código** | **MIT** (ver arquivo `LICENSE` do repositório) |
| **Licença dos dados** | Marcada como **"Other"** no Kaggle — **a confirmar antes de qualquer redistribuição** (ver Seção 4) |
| **Idioma da documentação** | Português do Brasil |
| **Data do documento** | 2026-06-12 (atualizado com métricas reais) |

---

## 2. Usos Pretendidos

O modelo foi concebido com **ângulo de impacto humano primariamente educacional**. Os usos a seguir são os endossados pelos autores.

### 2.1 Uso primário — Ferramenta educacional
- **Ensino de probabilidade e estatística:** reconhecer cartas para demonstrar cálculo de chances, combinações e raciocínio probabilístico de forma concreta e visual.
- **Ensino de regras de jogos e de matemática** para crianças e pessoas idosas, como apoio lúdico à aprendizagem.
- **Material didático para ensinar visão computacional:** o próprio pipeline (transfer learning, *data augmentation*, avaliação) serve de exemplo pedagógico de um projeto de classificação de imagens de ponta a ponta.

### 2.2 Usos secundários
- **Acessibilidade para pessoas com deficiência visual:** identificar uma carta e **lê-la em voz alta**, como apoio à autonomia em contextos não monetários (ex.: jogos recreativos, atividades educativas).
- **Pesquisa e *benchmark* em visão computacional:** estudar transfer learning, robustez a *out-of-distribution* (OOD) e *gaps* de domínio em um conjunto de classes bem definido.

### 2.3 Usuários previstos
Educadores, estudantes, pesquisadores de VC, desenvolvedores de ferramentas de acessibilidade e entusiastas de aprendizado de máquina.

---

## 3. Usos Fora de Escopo e Usos Proibidos

> Esta seção é **normativa**. O uso do modelo nas situações abaixo **viola a intenção dos autores** e, em vários casos, regras de plataformas e/ou legislação.

### 3.1 Usos PROIBIDOS (dual-use / dano social)
- **Apostas e jogos de azar com dinheiro real**, presencial ou *online*.
- **Real-Time Assistance (RTA)** — qualquer assistência em tempo real durante partidas com valor monetário. Ferramentas equivalentes são **expressamente proibidas** por operadoras como **PokerStars** e **GGPoker**, e seu uso configura **trapaça**.
- **Integração discreta / oculta** (formatos discretos: óculos, câmeras escondidas, *overlays* furtivos) destinada a obter vantagem indevida em jogos.
- **Contagem de cartas assistida por máquina** ou qualquer vantagem mecânica em cassinos e casas de aposta.
- Qualquer uso que **facilite ou agrave o jogo problemático** (dano social documentado associado ao comportamento de apostas).

### 3.2 Usos fora de escopo (não suportados tecnicamente)
- **Detecção/contagem de múltiplas cartas em uma cena** — o modelo só classifica **uma carta já recortada**.
- Leitura de cartas em **baralhos com design muito diferente** do conjunto de treino (ver Limitações, Seção 6).
- Decisões **automatizadas de alto risco** sem supervisão humana.
- Reconhecimento de objetos que **não sejam cartas de baralho**.

---

## 4. Dados de Treino

| Campo | Descrição |
|---|---|
| **Dataset** | *Cards Image Dataset-Classification* |
| **Autor** | **gpiosenka** (Kaggle) |
| **Link** | https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification |
| **Classes** | 53 (52 cartas + coringa/*joker*) |
| **Formato** | Imagens RGB **224×224×3, já recortadas** (uma carta por imagem) |
| **Split (pré-definido)** | **train: 7.624** / **valid: 265** / **test: 265** imagens |
| **Tamanho de val/test** | **5 imagens por classe** em validação e em teste (conjuntos pequenos — ver Limitações) |
| **Licença** | Marcada como **"Other"** no Kaggle. **Necessário confirmar os termos antes de redistribuir** os dados ou um modelo derivado. Este repositório **não redistribui** o dataset; a pasta `data/` contém **apenas instruções de download** |

### 4.1 Pré-processamento
- Redimensionamento para 224×224 (quando aplicável) e normalização com média/desvio do ImageNet.
- **Data augmentation** (experimento 2): rotação leve, *color jitter* (brilho/contraste/saturação), *affine* leve (translação/escala) e *random erasing* — aplicadas **somente no treino**. *Flips* horizontais/verticais foram **evitados de propósito** (cartas têm orientação; espelhar geraria imagens irreais/ambíguas).

---

## 5. Métricas de Avaliação

**Métricas adotadas:** *Accuracy* + **F1 macro** + **matriz de confusão** + **análise de "confusões perigosas"** (trocas de naipe ou de valor entre classes próximas).

> Valores **medidos** no Colab (GPU T4), `set_seed(42)`. Conjuntos de validação/teste pequenos (5 img/classe) → métricas com ruído estatístico; priorize **F1-macro** e a **matriz de confusão**. O modelo principal é o EfficientNet-B0 FE+FT **com** augmentation.

### 5.1 Resultados por experimento

| Experimento | Configuração | Accuracy | F1 macro | Observações |
|---|---|---|---|---|
| **Baseline** | HOG + Regressão Logística | 0,706 | 0,698 | Referência clássica (superada com folga) |
| **Exp. 1a** | EfficientNet-B0, *feature extraction* (backbone congelado) | 0,385 | 0,363 | Features ImageNet são fracas p/ cartas |
| **Exp. 1b** | EfficientNet-B0, *fine-tuning* do topo | 0,947 | 0,947 | +56 pp sobre 1a → fine-tuning é essencial |
| **Exp. 2a** | Melhor modelo, **sem** *data augmentation* | 0,974 | 0,973 | Maior no teste limpo (in-distribution) |
| **Exp. 2b** | Melhor modelo, **com** *data augmentation* ← principal | 0,947 | 0,947 | −2,6 pp in-dist; **mesma acc. OOD** que sem-aug (ver 5.2) |

### 5.2 Avaliação OOD (Experimento 3)

| Conjunto de avaliação | Origem | Accuracy | F1 macro | *Gap* de domínio |
|---|---|---|---|---|
| **Teste Kaggle (in-distribution)** | Mesmo domínio do treino | 0,947 | 0,947 | — |
| **OOD design — modelo COM aug (principal)** | Imagens limpas de licença livre (web); ver `src/ood_design.py` | 0,593 | 0,574 | gap de design ≈ 35 pp |
| **OOD design — modelo SEM aug** | idem | 0,593 | 0,568 | gap ≈ 38 pp |

> **Conclusão do Experimento 2 no OOD:** a *data augmentation* **não melhorou** a acurácia OOD (ambos **0,593**); a diferença está só no F1-macro (0,574 vs 0,568, dentro do ruído). Esperado: a augmentation usada simula **condições de captura** (rotação/brilho/oclusão), enquanto este OOD testa **design** — gaps de tipos diferentes. O benefício da augmentation tende a aparecer num OOD de **fotos reais** (gap de captura), que é trabalho futuro.

> **O que esta linha OOD mede:** o **gap de design** (generalização para outro estilo de carta), **não** o gap de condições de captura (luz/sombra/fundo/ângulo de fotos reais). Como as imagens são renders limpos, o gap aqui é um **limite inferior** do esperado com fotos do mundo real — medir isso com um baralho fotografado é trabalho futuro (`docs/guia_coleta_baralho_real.md`).

### 5.3 Análise qualitativa
- **Matriz de confusão:** ver `reports/confusion_matrix_test.png` (teste Kaggle) e `reports/confusion_matrix_ood.png` (OOD de design), geradas pelo notebook (seções 6 e 8). Relatório por classe em `reports/classification_report_test.csv`.
- **Confusões perigosas:** dos **14 erros** no teste (251/265 corretos), **quase todos são trocas de _valor_ dentro do mesmo naipe** — ex.: `nine of diamonds`→`eight of diamonds`, `seven of clubs`→`eight of clubs`, `five of diamonds`→`three of diamonds`, `nine of spades`→`six of spades` — e **nenhuma troca de naipe/cor** entre as confusões listadas. Ou seja: o modelo **lê bem o naipe** e erra na **contagem de pips** de cartas numéricas próximas. A classe **`joker` é a mais frágil** (2 de 5 imagens erradas, previstas como `ten`/`five of clubs`), provavelmente por design atípico e poucas amostras. O fato de **erros de naipe (copas↔ouros) — os mais críticos no uso educacional/assistivo — praticamente não ocorrerem** é um ponto positivo de segurança. Em F1 por classe (`reports/classification_report_test.csv`), as piores são **`joker` (F1 0,75; recall 0,60)** e **`three of diamonds` (F1 0,77; precisão 0,625)** — esta última funcionando como "ímã" de falsos positivos (recebe predições de `five`/`queen of diamonds`); as demais 51 classes ficam entre **0,89 e 1,00**.

---

## 6. Limitações e Vieses

- **Treinado em um único tipo de baralho:** o dataset reflete **um design** de cartas. O desempenho tende a **degradar** em baralhos com estilos artísticos diferentes, cartas regionais, decks personalizados ou marcas distintas.
- ***Gap* de design (OOD medido):** treinado em **um** design de cartas, o modelo tende a errar em baralhos de **estilo diferente**. O Experimento 3 quantifica isso com um baralho de design distinto (imagens limpas da web).
- ***Gap* de captura (NÃO medido aqui):** em fotos do mundo real (iluminação variável, fundo, sombras, perspectiva, oclusão parcial, desfoque, reflexos), espera-se queda **adicional** de acurácia. O OOD atual usa renders limpos, então **não** captura esse efeito; medi-lo exige fotos reais (trabalho futuro). Por isso o gap reportado é um **limite inferior**.
- **Conjuntos de validação e teste pequenos:** apenas **5 imagens por classe** (265 no total cada). Isso torna as estimativas de métrica **ruidosas** e sensíveis a poucos exemplos; intervalos de confiança são largos.
- **Escopo restrito a classificação:** não detecta nem segmenta múltiplas cartas; **exige recorte prévio** de uma única carta.
- **Viés de pré-treino (ImageNet):** o backbone carrega vieses do ImageNet, que não foi pensado para cartas.
- **Sem garantia de calibração:** as probabilidades de saída podem ser **mal calibradas**; não devem ser interpretadas como confiança real sem avaliação adicional.
- **Casos não cobertos:** cartas dobradas, sujas, parcialmente cobertas, em ângulos extremos ou sob iluminação atípica podem produzir **erros silenciosos**.

---

## 7. Considerações Éticas (resumo)

- **Dual-use / trapaça:** a mesma tecnologia pode ser usada para **RTA proibido** em pôquer/cassinos (PokerStars, GGPoker). Os usos proibidos estão declarados na Seção 3.
- **Jogo problemático:** há **dano social documentado** ligado a apostas; o projeto **não** endossa nem facilita uso em contextos de jogo por dinheiro.
- **Viés e limitação do dataset:** treinado em **um tipo de baralho** → risco de falha em outros designs, iluminações e oclusões (Seção 6). Resultados não devem ser generalizados além do domínio avaliado.
- **Privacidade (LGPD):** em uso com câmera, o dispositivo pode captar **o ambiente e terceiros**. Recomenda-se **minimização de dados**, **processamento *on-device*** sempre que possível, ausência de armazenamento desnecessário de imagens e **consentimento** quando houver pessoas identificáveis.
- **Transparência / documentação ética:** publicar o modelo **sem declarar os usos proibidos** é em si um risco. Este *model card* cumpre o papel de documentar explicitamente intenção, limites e restrições.
- **Supervisão humana:** ferramenta de **apoio**; decisões devem permanecer sob responsabilidade de uma pessoa, especialmente em contextos educacionais e de acessibilidade.

---

## 8. Como Reproduzir

### 8.1 Ambiente
- **Treino:** Google Colab com **GPU T4 gratuita**.
- **Dependências:** versões **fixadas** em `requirements.txt` (PyTorch, torchvision, scikit-learn, etc.).
- **Reprodutibilidade:** semente global fixa via `set_seed(42)` (`src/seed.py`) — controla `random`, `numpy` e `torch` (CPU/GPU).

### 8.2 Passos
1. **Obter os dados:** baixar o dataset *Cards Image Dataset-Classification* (gpiosenka) seguindo as instruções em `data/` (o repositório **não** redistribui as imagens).
2. **Instalar dependências:** `pip install -r requirements.txt`.
3. **Configurar:** ajustar caminhos e hiperparâmetros em `src/config.py`.
4. **Baseline:** treinar HOG + Regressão Logística com `src/baseline.py`.
5. **Treino principal:** executar `src/train.py` (ou o notebook `notebooks/treino_cartas_colab.ipynb`) — Fase 1 (*feature extraction*) e Fase 2 (*fine-tuning*).
6. **Avaliar:** `src/evaluate.py` gera accuracy, F1 macro, matriz de confusão e relatório de confusões perigosas em `reports/`.
7. **Inferência:** `src/predict.py` classifica uma carta recortada.
8. **OOD:** montar o **baralho de design diferente** (`python -m src.ood_design`) e comparar com o teste Kaggle (Experimento 3). Fotos reais de um baralho físico (gap de captura) ficam como trabalho futuro.

### 8.3 Estrutura do repositório
```
src/        config.py, seed.py, data.py, model.py, baseline.py, train.py, evaluate.py, predict.py
notebooks/  treino_cartas_colab.ipynb
models/     pesos treinados (gerados)
data/       instruções de download (sem dados redistribuídos)
docs/       documentação
reports/    métricas, figuras, matriz de confusão
README.md   visão geral
requirements.txt  dependências fixadas
LICENSE     MIT (código)
```

---

## 9. Citação

```
@misc{classificador_cartas_2026,
  title  = {Classificador de Cartas de Baralho com EfficientNet-B0 (Transfer Learning)},
  author = {Pereira, Nicolas},
  year   = {2026},
  note   = {Projeto acadêmico ICD/ADS. Código sob licença MIT.
            Dataset: Cards Image Dataset-Classification (gpiosenka, Kaggle).}
}
```

---

*Métricas medidas no Colab (GPU T4), `set_seed(42)`, em 2026-06. Em caso de dúvida sobre usos permitidos, prevalece a Seção 3 (Usos Fora de Escopo e Proibidos).*