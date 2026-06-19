# Guia Prático: Fotografando um Baralho Real para o Conjunto de Teste OOD

> **Status nesta entrega:** o conjunto OOD usado atualmente é o de **"design diferente"**
> (imagens limpas da web — ver [`guia_ood_design_web.md`](guia_ood_design_web.md)), que mede o
> **gap de design**. Este guia descreve o caminho **padrão-ouro** — fotografar um baralho físico —
> que mede o **gap de condições de captura** (luz, sombra, fundo, ângulo) e está marcado como
> **trabalho futuro**. Se/quando você fotografar o baralho, ponha as imagens em
> `data/raw/ood_baralho_real/` e aponte `OOD_DIR` no notebook para essa pasta.

Este documento explica, passo a passo, como você deve fotografar **1 baralho físico próprio** e organizar as imagens para formar o conjunto de avaliação **OOD** (*out-of-distribution* — fora da distribuição de treino) do projeto de classificação de cartas.

---

## 1. Objetivo

O modelo principal (EfficientNet-B0 via *transfer learning*) será treinado e avaliado no dataset **"Cards Image Dataset-Classification"** (autor *gpiosenka*, Kaggle), cujas imagens já vêm recortadas em 224×224, com fundo controlado e um único tipo de design de carta.

O problema: um modelo que vai muito bem no teste do Kaggle pode falhar quando recebe fotos de um baralho **real, diferente, em condições de iluminação e fundo do mundo real**. Esse fenômeno é o **gap de domínio** (*domain gap*).

O conjunto que você vai montar serve para o **Experimento 3** do projeto:

- **Mede a generalização**: o modelo aprendeu o conceito de "rei de espadas" ou apenas decorou o design específico do dataset?
- **Quantifica o gap de domínio**: comparamos a *accuracy* / F1-macro no teste do Kaggle (in-distribution) contra a *accuracy* / F1-macro nas suas fotos (OOD).
- **Expõe limitações reais**: design de baralho diferente, iluminação, sombras, fundo e oclusão — exatamente os riscos de viés/limitação do dataset levantados na pesquisa.

> **Importante:** estas fotos são **apenas para teste/avaliação**. Elas **não** entram no treino. Não há necessidade de centenas de imagens — o objetivo é diagnóstico, não treinar mais.

---

## 2. Quantas fotos por carta

Tire **2 a 3 fotos por carta**, variando **de propósito** as condições para estressar a robustez do modelo:

| Foto | Variação sugerida |
|------|-------------------|
| Foto 1 | Iluminação boa, fundo neutro/claro, carta reta — condição "fácil" (referência). |
| Foto 2 | Ângulo levemente inclinado e/ou iluminação mais fraca/lateral (sombra). |
| Foto 3 (opcional) | Fundo diferente/texturizado (mesa de madeira, tecido, fundo escuro) e/ou outro horário de luz. |

São **53 classes** (52 cartas + 1 coringa). Com 2–3 fotos por classe, o conjunto OOD terá aproximadamente **106 a 159 imagens** — suficiente para uma avaliação diagnóstica.

> Variar de propósito é o ponto central: se todas as fotos forem perfeitas e idênticas, o teste não revela o gap de domínio real.

---

## 3. Padrão de nomes das classes

As subpastas precisam ter **exatamente os mesmos nomes de classe do dataset gpiosenka**, senão a avaliação não consegue casar a predição do modelo com o rótulo verdadeiro.

### Regra do padrão

```
<valor> of <naipe>
```

- Tudo em **letras minúsculas**, em **inglês**, separado por espaços (a palavra `of` no meio).
- **Valores** (13): `ace`, `two`, `three`, `four`, `five`, `six`, `seven`, `eight`, `nine`, `ten`, `jack`, `queen`, `king`
- **Naipes** (4): `clubs` (paus), `diamonds` (ouros), `hearts` (copas), `spades` (espadas)
- **Exceção**: o coringa é apenas `joker` (sem "of").

Isso dá 13 × 4 = **52 cartas + `joker` = 53 classes**.

### Lista completa das 53 classes (use estes nomes literais como nome de pasta)

**Paus (clubs):**
`ace of clubs`, `two of clubs`, `three of clubs`, `four of clubs`, `five of clubs`, `six of clubs`, `seven of clubs`, `eight of clubs`, `nine of clubs`, `ten of clubs`, `jack of clubs`, `queen of clubs`, `king of clubs`

**Ouros (diamonds):**
`ace of diamonds`, `two of diamonds`, `three of diamonds`, `four of diamonds`, `five of diamonds`, `six of diamonds`, `seven of diamonds`, `eight of diamonds`, `nine of diamonds`, `ten of diamonds`, `jack of diamonds`, `queen of diamonds`, `king of diamonds`

**Copas (hearts):**
`ace of hearts`, `two of hearts`, `three of hearts`, `four of hearts`, `five of hearts`, `six of hearts`, `seven of hearts`, `eight of hearts`, `nine of hearts`, `ten of hearts`, `jack of hearts`, `queen of hearts`, `king of hearts`

**Espadas (spades):**
`ace of spades`, `two of spades`, `three of spades`, `four of spades`, `five of spades`, `six of spades`, `seven of spades`, `eight of spades`, `nine of spades`, `ten of spades`, `jack of spades`, `queen of spades`, `king of spades`

**Coringa:**
`joker`

> **Atenção a erros comuns:** use `ace` (não "1" nem "a"), `ten` (não "10"), `jack/queen/king` por extenso (não "J/Q/K"). Não use plural nos valores, não use português, não use maiúsculas. Se o nome da pasta diferir de um único caractere, a classe será tratada como desconhecida.

---

## 4. Onde colocar as imagens (estrutura de pastas)

Crie a árvore abaixo dentro do repositório. **Uma subpasta por classe**, e dentro dela as 2–3 fotos daquela carta:

```
data/
└── raw/
    └── ood_baralho_real/
        ├── ace of clubs/
        │   ├── ace_of_clubs_01.jpg
        │   ├── ace_of_clubs_02.jpg
        │   └── ace_of_clubs_03.jpg
        ├── two of clubs/
        │   ├── two_of_clubs_01.jpg
        │   └── two_of_clubs_02.jpg
        ├── ...
        ├── king of spades/
        │   ├── king_of_spades_01.jpg
        │   └── king_of_spades_02.jpg
        └── joker/
            ├── joker_01.jpg
            └── joker_02.jpg
```

- Caminho-raiz do conjunto OOD: **`data/raw/ood_baralho_real/<classe>/`**
- O **nome da pasta** é o rótulo (precisa bater com a Seção 3). O **nome do arquivo** é livre — use algo legível como `ace_of_clubs_01.jpg`. Recomenda-se substituir os espaços por `_` apenas **no nome do arquivo** (não na pasta).
- Formato de imagem: `.jpg` ou `.png`. Você não precisa recortar nem redimensionar para 224×224 manualmente — o pipeline de pré-processamento (`src/data.py` / `src/predict.py`) aplica o *resize* e a normalização do mesmo modo que no treino.

> A pasta `data/` do repositório contém **apenas instruções** (não versionamos imagens). As fotos do baralho real ficam **localmente** na sua máquina / no seu Google Drive ao rodar no Colab. Não faça *commit* das imagens.

---

## 5. Dicas de qualidade das fotos

- **Foco nítido:** toque na carta na tela do celular antes de disparar para travar o foco. Descarte qualquer foto borrada.
- **Carta centralizada e grande:** a carta deve ocupar boa parte do quadro (ocupando ~60–80% da imagem), totalmente visível, sem cortar cantos. Evite oclusão (dedos/objetos cobrindo o naipe ou o valor).
- **Uma carta por foto:** o escopo do projeto é classificação de **1 carta recortada por imagem**. Não coloque várias cartas no mesmo quadro.
- **Fundo variado (de propósito):** alterne entre fundo neutro (mesa clara, folha branca) e fundos mais difíceis (madeira, tecido, fundo escuro). Isso é o que testa a robustez ao mundo real.
- **Iluminação variada:** misture luz natural e artificial; inclua ao menos uma foto com luz lateral (gera sombra). Evite reflexos fortes/brilho especular sobre a face plastificada da carta — incline levemente a carta ou a fonte de luz se houver brilho.
- **Carta plana e legível:** evite cartas amassadas/dobradas; mantenha a face de frente para a câmera (pequena inclinação é desejável na Foto 2, mas a carta deve continuar reconhecível).
- **Sem filtros/edição:** não aplique filtros, recortes automáticos nem realce de IA do celular — queremos a imagem mais "crua" possível, representando o uso real.
- **Resolução suficiente:** a câmera padrão do celular já basta; não há ganho em fotos gigantes. O importante é nitidez e enquadramento.

---

## 6. Nota de privacidade (LGPD)

A câmera capta tudo o que estiver no enquadramento — não apenas a carta. Para respeitar a **minimização de dados** e a **LGPD** (Lei Geral de Proteção de Dados), e em coerência com os riscos éticos de privacidade levantados no projeto:

- **Não capture rostos** seus nem de terceiros nas fotos.
- **Não fotografe outras pessoas** ao fundo, nem em reflexos (espelhos, telas, vidros).
- **Não inclua documentos, telas com dados, etiquetas com nome/endereço, comprovantes** ou qualquer informação pessoal identificável no fundo.
- Prefira fundos **neutros ou genéricos** (mesa, tecido, parede) que não revelem local específico nem dados de terceiros.
- Estas imagens são de uso **acadêmico/diagnóstico**; ainda assim, trate-as como dados a serem minimizados: **não** publique as fotos no repositório nem as redistribua.
- Coerente com o princípio de **processamento on-device** defendido no projeto: as fotos ficam sob seu controle, não são enviadas a serviços de terceiros além do estritamente necessário para o treino/avaliação (Colab/Drive, que você controla).

---

## 7. Checklist final

Antes de considerar o conjunto OOD pronto:

- [ ] Tirei **2–3 fotos** de **cada uma das 53 classes** (52 cartas + `joker`).
- [ ] Variei **de propósito** ângulo, iluminação e fundo entre as fotos.
- [ ] Criei **53 subpastas**, uma por classe, dentro de `data/raw/ood_baralho_real/`.
- [ ] Os **nomes das pastas** batem **exatamente** com o padrão `<valor> of <naipe>` em inglês minúsculo (e `joker`), conforme a Seção 3.
- [ ] Cada foto tem **1 carta só**, nítida, centralizada e sem oclusão.
- [ ] Nenhuma foto contém **rostos, terceiros, documentos ou dados pessoais** no fundo (LGPD).
- [ ] **Não** apliquei filtros nem edição automática.
- [ ] As imagens estão **apenas localmente / no meu Drive**, sem *commit* no repositório.
- [ ] Confiro que tenho `joker` (e não "joker of ..." nem em português).

Concluído o checklist, o conjunto está pronto para ser usado em `src/evaluate.py` na avaliação OOD e na comparação do **gap de domínio** (teste Kaggle vs. baralho real) do Experimento 3.

> Os resultados do OOD **de design** (web) já constam no README/MODEL_CARD: teste 94,7% vs OOD 59,3% (gap ≈ 35 pp). Os números de um OOD com **fotos reais** (este guia) seriam preenchidos se/quando esse conjunto for coletado.