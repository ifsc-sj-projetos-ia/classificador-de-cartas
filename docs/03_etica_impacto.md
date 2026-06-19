# Avaliação Ética e Impacto

> **Item 2.5 — Classificador de Cartas de Baralho por Visão Computacional**
> Documento de avaliação ética, análise de riscos e definição de contextos de uso e não-uso.
> Tom: acadêmico. Idioma: português (Brasil).

---

## Introdução

Esta seção apresenta uma avaliação ética estruturada do projeto, cujo escopo técnico é restrito à **classificação** de uma única carta de baralho previamente recortada (entrada: 1 imagem 224×224×3 → saída: 1 rótulo entre 53 classes, sendo 52 cartas mais o coringa/*joker*). A detecção de múltiplas cartas em cena é explicitamente tratada como trabalho futuro e **não** integra a entrega atual.

Embora o artefato seja modesto em escopo, a tecnologia subjacente — reconhecimento automático de cartas por visão computacional — é **dual-use**: a mesma capacidade que apoia o objetivo de impacto primário (uma **ferramenta educacional** para ensino de probabilidade, regras de jogos e matemática a crianças e idosos, além de servir como artefato didático para ensinar a própria visão computacional) pode, sem salvaguardas, ser desviada para usos socialmente nocivos. Reconhecer essa tensão desde a concepção é parte da maturidade ética que este documento busca demonstrar.

Para evidenciar essa maturidade, adota-se ao longo do texto uma **narrativa defesa-vs-ataque**: para cada risco identificado (a "ofensiva" — como a tecnologia poderia ser explorada de forma indevida), apresenta-se a correspondente "defesa" (decisões de projeto, restrições técnicas e documentais que reduzem a viabilidade ou o incentivo a esse uso). O objetivo não é alcançar risco zero — algo inatingível para qualquer tecnologia de propósito geral — mas tornar o **uso indevido mais custoso, menos provável e claramente desautorizado**, ao mesmo tempo em que o uso legítimo permanece acessível.

---

## 1. Riscos Potenciais

A seguir, detalham-se os cinco riscos centrais levantados na pesquisa. Cada risco é descrito sob a ótica do atacante (como o sistema poderia ser explorado ou falhar) para fundamentar, na Seção 2, as defesas correspondentes.

### 1.1. Dual-use e trapaça em jogos de azar e apostas

O risco mais grave é o **dual-use**: um classificador de cartas é precisamente o componente perceptivo de um sistema de assistência em tempo real (*Real-Time Assistance*, RTA) — ferramentas que leem o estado do jogo e sugerem a jogada ótima. RTAs são **explicitamente proibidas** pelas principais plataformas de pôquer online (por exemplo, PokerStars e GGPoker), que as classificam como trapaça e baniram contas e confiscaram fundos por seu uso. Em jogos presenciais valendo dinheiro, a leitura automática de cartas por uma câmera discreta constitui fraude e, a depender da jurisdição, crime.

**Vetor de ataque:** um terceiro mal-intencionado poderia tomar o modelo treinado (ou o pipeline) e acoplá-lo a um detector de cartas em cena, alimentando um *solver* de estratégia para obter vantagem injusta em mesas reais ou online. O componente de classificação, embora seja apenas uma peça, é a "porta de entrada" perceptiva que viabiliza a cadeia completa.

### 1.2. Facilitação de jogo problemático (dano social)

Mesmo sem trapaça, ferramentas que reduzem o atrito de jogar a dinheiro podem **agravar o jogo problemático** (*problem gambling*), um dano social documentado, associado a endividamento, sofrimento psíquico e ruptura familiar. Uma ferramenta que automatiza a leitura de cartas e oferece probabilidades em tempo real pode funcionar como um facilitador que normaliza, acelera ou intensifica o comportamento de aposta, especialmente em populações vulneráveis.

**Vetor de ataque / falha de design:** apresentar probabilidades "ao vivo" sobre cartas reais durante uma partida valendo dinheiro confunde a fronteira entre **ensinar probabilidade** (objetivo legítimo) e **assistir a aposta** (objetivo nocivo). Sem delimitação explícita, o mesmo recurso educacional pode ser reposicionado como acelerador de aposta.

### 1.3. Viés e limitação de generalização do dataset

O modelo é treinado predominantemente sobre o *Cards Image Dataset-Classification* (autor *gpiosenka*, Kaggle): imagens já recortadas, padronizadas e, em larga medida, de **um único estilo de baralho**, sob condições controladas. Isso cria um forte risco de **viés de domínio**: o sistema pode atingir alta acurácia no conjunto de teste do próprio dataset e, ainda assim, **falhar sistematicamente** diante de:

- designs de baralho diferentes (cartas regionais, baralhos comemorativos, *naipes* estilizados, baralhos espanhóis/alemães);
- condições de iluminação adversas (sombra, reflexo, luz amarelada);
- oclusão parcial, cartas amassadas, ângulo oblíquo, desgaste;
- variações de resolução e qualidade de câmera.

**Vetor de ataque / falha:** uma falha de classificação em contexto educacional gera, na pior hipótese, um erro didático corrigível. Porém, o problema mais sério é a **falsa sensação de robustez**: relatar apenas a acurácia *in-distribution* (no teste do Kaggle) e omitir a queda *out-of-distribution* (OOD) leva usuários a confiar no sistema fora de seu envelope de validade. A análise de **"confusões perigosas"** (trocar naipe ou valor — por exemplo, confundir Dama com Valete, ou copas com ouros) é especialmente relevante: nem todo erro tem o mesmo peso.

### 1.4. Privacidade e conformidade com a LGPD

O sistema opera por **câmera**. Mesmo destinado a fotografar cartas, o sensor inevitavelmente capta o **ambiente ao redor e, potencialmente, terceiros** (rostos, crianças, documentos, interior de residências). Isso aciona a **Lei Geral de Proteção de Dados (LGPD, Lei nº 13.709/2018)**: imagens de pessoas são dados pessoais, e o tratamento sem base legal, transparência ou minimização é uma violação.

**Vetor de ataque / falha:** um design ingênuo que **persiste imagens**, as envia a servidores na nuvem para inferência ou as acumula para "melhorar o modelo" cria um repositório de dados pessoais sensíveis — atrativo para vazamento, uso secundário não consentido e responsabilização legal. O público-alvo educacional inclui **crianças e idosos**, populações cujos dados exigem proteção reforçada, ampliando a gravidade.

### 1.5. Documentação ética insuficiente

Um risco frequentemente subestimado é **publicar o artefato sem documentar suas fronteiras éticas**. Liberar código e pesos de modelo em repositório aberto, sem declarar usos proibidos, limitações conhecidas e contextos de não-uso, transfere ao acaso a responsabilidade pelo uso indevido.

**Vetor de ataque:** a ausência de um *model card* e de uma declaração de uso aceitável funciona como uma **permissão implícita**: quem encontra o repositório não tem sinal algum de que o uso em apostas é desautorizado, de que o modelo não foi validado fora do dataset, ou de que a embutição em dispositivos discretos é inadmissível. A omissão documental é, ela própria, uma falha ética — não apenas uma lacuna de comunicação.

---

## 2. Estratégias de Mitigação Adotadas e Sugeridas

Para cada risco da Seção 1, apresenta-se a **defesa** correspondente. As mitigações combinam restrições de **escopo** (o que o sistema faz), restrições **técnicas** (como faz), restrições **de dados** (com o quê) e restrições **documentais** (como é comunicado e licenciado).

### 2.1. Defesa contra dual-use e trapaça (responde a 1.1)

- **Declarar escopo e usos proibidos explicitamente.** O projeto declara, no README, no *model card* e na declaração de uso aceitável, que o uso em jogos de azar com dinheiro real, apostas e assistência em tempo real valendo dinheiro é **proibido e fora do propósito** do artefato (ver Seção 3).
- **Manter o escopo em classificação, não em detecção de cena.** A entrega é deliberadamente restrita a "1 carta recortada → 1 rótulo". A capacidade de **varredura de mesa em tempo real** (*scan* de múltiplas cartas em cena), que é o que torna um RTA prático, **não é construída nem facilitada** — é remetida a "trabalho futuro" sem implementação.
- **Não embutir o sistema em formato discreto.** Recomenda-se explicitamente não distribuir nem projetar o sistema para dispositivos ocultos/vestíveis (óculos, câmeras escondidas), formato típico de fraude presencial.

### 2.2. Defesa contra facilitação de jogo problemático (responde a 1.2)

- **Modo "leitura assistiva carta por carta", não "scan de mesa em tempo real".** A interação preconizada é a leitura de **uma carta de cada vez**, em contexto de estudo/acessibilidade, e **não** o monitoramento contínuo de uma partida ao vivo. Essa escolha de design separa "ensinar probabilidade sobre uma carta/baralho" de "assistir apostas em tempo real".
- **Enquadramento educacional explícito.** O material e a interface posicionam o uso como **didático** (aprender regras, valores, naipes e noções de probabilidade), sem recurso a aposta a dinheiro real e sem sugestão de jogada ótima sobre partidas valendo dinheiro.

### 2.3. Defesa contra viés e limitação de generalização (responde a 1.3)

- **Diversidade de dados e *data augmentation*.** Sugere-se ampliar a robustez por meio de aumento de dados (rotação, variação de brilho/contraste, *crop*, ruído) e, quando possível, incorporar baralhos e condições adicionais. O **Experimento 2** (com vs. sem *data augmentation*) mede diretamente o efeito dessa mitigação.
- **Avaliação honesta de generalização (OOD).** O **Experimento 3** compara o teste *in-distribution* (Kaggle) com um **baralho de design diferente** (imagens limpas de licença livre, da web), quantificando o **gap de design**. Em nome da própria honestidade que esta seção defende, declara-se que estas **não** são fotos de um baralho real: o conjunto mede o gap de *design*, não o de *condições de captura* (luz, sombra, fundo, ângulo), e por usar imagens limpas o gap reportado é um **limite inferior**. Medir o gap de captura com fotos reais é **trabalho futuro**. Reportar isso é a defesa contra a "falsa sensação de robustez".
- **Métricas por subgrupo/condição e análise de confusões perigosas.** Além de *accuracy* e **F1 macro**, reporta-se a **matriz de confusão** e uma análise das **confusões perigosas** (trocas de naipe/valor), com desempenho recortado por condição (iluminação, ângulo, design), evitando que uma média alta mascare falhas concentradas.

> **Nota sobre números:** os resultados quantitativos foram **medidos no Colab (GPU T4)**: **94,7%** de acurácia / **0,947** de F1-macro no teste Kaggle e **~59%** no OOD de design (gap de domínio ≈ 35 pp). Detalhes no README e no MODEL_CARD.

### 2.4. Defesa de privacidade e conformidade com a LGPD (responde a 1.4)

- **Processamento *on-device*.** Recomenda-se que a inferência ocorra **localmente no dispositivo**, sem envio de imagens a servidores externos, reduzindo a superfície de exposição.
- **Não persistir imagens.** Por padrão, **as imagens capturadas não são armazenadas**: são processadas em memória para gerar o rótulo e descartadas. Não há acúmulo de dados para "treino futuro" sem consentimento explícito e base legal.
- **Minimização de dados.** Coleta-se apenas o estritamente necessário para a tarefa (a carta), com orientação ao usuário para enquadrar somente a carta e evitar captar pessoas e ambiente — princípio de **minimização** previsto na LGPD. Atenção reforçada por envolver potencialmente **crianças e idosos**.

### 2.5. Defesa por documentação ética (responde a 1.5)

- ***Model card* com usos proibidos.** O projeto inclui um *model card* declarando: finalidade, dados de treino e seus vieses, limitações conhecidas, métricas (com placeholders), e uma lista explícita de **usos proibidos e contextos de não-uso**.
- **Licença aberta + *statement* de uso aceitável.** O código é distribuído sob licença **MIT** (aberta, favorecendo o uso educacional e de pesquisa), acompanhada de uma **declaração de uso aceitável** que torna explícito que a permissividade da licença **não autoriza** os usos vedados na Seção 3. A licença libera o *software*; a declaração comunica a **expectativa ética** de uso.
- **Confirmação de licenciamento do dataset.** Como a licença do *Cards Image Dataset-Classification* está marcada como "Other" no Kaggle, qualquer **redistribuição** de dados será precedida de **confirmação das condições de licença**; na dúvida, distribui-se apenas código e instruções de download, não as imagens.

---

## 3. Limitações do Sistema e Contextos de Não-Uso

Esta seção delimita, de forma explícita e inequívoca, **onde o sistema não deve ser usado** e **o que ele não garante**. A clareza aqui é, ela própria, uma medida de mitigação (responde diretamente ao risco 1.5).

### 3.1. Contextos de NÃO-USO (usos proibidos)

O artefato **não deve ser utilizado**, sob nenhuma circunstância, para:

- **Jogos de azar com dinheiro real** — qualquer uso que envolva apostas em valores monetários.
- **Apostas** — em ambiente presencial ou digital.
- **Cassinos online** e plataformas de jogo a dinheiro.
- **Assistência em tempo real (RTA) valendo dinheiro** — leitura de cartas e/ou sugestão de jogada durante partidas a dinheiro real, prática proibida pelas principais plataformas de pôquer (PokerStars, GGPoker) e potencialmente fraudulenta/ilícita em jogos presenciais.
- **Embutição em dispositivos discretos ou ocultos** (câmeras escondidas, *wearables* dissimulados) destinados a obter vantagem indevida.

Esses usos são incompatíveis com o propósito declarado do projeto (ferramenta educacional, acessibilidade e benchmark de pesquisa) e violam a declaração de uso aceitável que acompanha a licença.

### 3.2. Limitações técnicas e envelope de validade

O sistema **não foi validado fora dos baralhos e condições do dataset**. Em particular:

- O modelo foi treinado sobre um **conjunto predominantemente de um estilo de baralho**, em imagens **já recortadas** e padronizadas; **não há garantia** de desempenho sobre outros designs, baralhos regionais ou cartas estilizadas.
- O desempenho pode degradar significativamente sob **iluminação adversa, oclusão, ângulo oblíquo, desgaste físico** das cartas e **baixa qualidade de câmera** — condições medidas, mas não eliminadas, pelo Experimento 3 (OOD).
- O escopo é **classificação de uma carta isolada já recortada**; o sistema **não detecta** múltiplas cartas em cena nem segmenta cartas automaticamente.
- As métricas de desempenho foram **medidas**: **94,7%** no teste Kaggle e **~59%** no OOD de design (gap de domínio explícito, ≈ 35 pp), coerentes com a literatura de transfer learning (~93–95% in-distribution). O reporte do gap, e não só da acurácia limpa, é parte da postura de honestidade do projeto.
- Como toda classificação probabilística, o sistema **comete erros**, incluindo **confusões perigosas** (troca de naipe ou valor). Em contextos sensíveis (por exemplo, acessibilidade), recomenda-se **confirmação humana** e não dependência exclusiva da saída automática.

### 3.3. Usos legítimos previstos

Para evitar ambiguidade por exclusão, registram-se os usos **legítimos e incentivados**:

- **Educacional (primário):** ensino de probabilidade, de regras e da matemática de jogos de cartas para crianças e idosos; apoio à aprendizagem; e como artefato didático para ensinar **visão computacional** e *transfer learning*.
- **Acessibilidade (secundário):** leitura assistiva, **carta por carta**, em voz alta, para apoio a pessoas com deficiência visual — sempre com a ressalva de confirmação humana acima.
- **Pesquisa/benchmark em visão computacional (secundário):** estudo de transfer learning, *data augmentation* e robustez OOD em um domínio controlado.

---

## Conclusão

A análise demonstra que, embora o escopo técnico do projeto seja deliberadamente restrito à classificação de uma carta isolada, a tecnologia carrega tensões éticas reais — sobretudo o caráter **dual-use** e o potencial de dano social associado ao jogo. A postura adotada não é negar esses riscos, mas **antecipá-los** por meio da narrativa defesa-vs-ataque: para cada vetor de uso indevido ou de falha, há uma defesa correspondente em escopo, técnica, dados ou documentação.

As escolhas centrais — manter-se na classificação e não na varredura de mesa, privilegiar a **leitura assistiva carta por carta** em vez do *scan* em tempo real, processar **on-device** sem persistir imagens, medir explicitamente o **gap OOD**, e publicar com ***model card*, usos proibidos e declaração de uso aceitável** — convergem para um único princípio: **maximizar o valor educacional e de pesquisa enquanto se torna o uso indevido mais custoso, menos provável e claramente desautorizado**. Reconhece-se que nenhuma salvaguarda elimina o risco por completo; a contribuição ética do projeto está em torná-lo transparente, delimitado e responsavelmente comunicado.