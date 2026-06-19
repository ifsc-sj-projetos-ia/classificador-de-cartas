# Guia: Conjunto OOD de "Design Diferente" (imagens limpas da web)

Este documento descreve o conjunto de avaliação **OOD** (*out-of-distribution*) usado na
entrega atual do projeto e, principalmente, **o que ele mede e o que NÃO mede** — para que o
relatório e a apresentação façam afirmações honestas.

---

## 1. O que é

Em vez de fotografar um baralho físico, montamos um conjunto OOD a partir de um **baralho de
design diferente** do dataset de treino (gpiosenka), usando **imagens limpas de licença livre**
baixadas da web. O conjunto é gerado de forma **reprodutível** por um script:

```bash
python -m src.ood_design --out data/raw/ood_design_web
```

Resultado: `data/raw/ood_design_web/` com **53 subpastas** (uma por classe), nomeadas
**exatamente** como as classes de treino (`ace of clubs`, ..., `king of spades`, `joker`),
totalizando **54 imagens** (1 por carta + 2 coringas na pasta `joker/`). Esse formato é o que
`src.evaluate.evaluate_ood` espera, então a avaliação roda sem ajustes.

> Por que isso foi feito: não houve tempo de fotografar um baralho real antes da entrega. Em vez
> de apresentar imagens da web **como se fossem** fotos reais (o que seria incorreto), optamos por
> um experimento OOD **honesto e bem rotulado**: testar a generalização para outro **design**.

---

## 2. O que este conjunto MEDE — e o que NÃO mede

| Tipo de gap | Mede aqui? | Por quê |
|---|---|---|
| **Gap de design** (estilo/arte/fontes da carta diferentes do treino) | ✅ Sim | As cartas têm desenho, cor e tipografia distintos do dataset gpiosenka. |
| **Gap de captura** (iluminação, sombra, fundo real, ângulo, reflexo, oclusão, desfoque) | ❌ Não | São renders digitais limpos sobre fundo uniforme — não há condições do mundo real. |

**Implicação para o relatório (importante):** como as imagens são "limpas", em vários aspectos
elas são **mais próximas** do domínio de treino do que fotos reais seriam. Portanto, o *gap*
medido aqui é um **limite inferior** do gap esperado em uso real. **Não** afirme que o modelo
"funciona com qualquer baralho no mundo real" com base neste experimento. A pergunta que ele
responde é mais estreita: *o modelo aprendeu o conceito de cada carta ou decorou o design do
dataset de treino?*

---

## 3. Procedência e licença

- **Assets:** repositório [`hayeah/playing-cards-assets`](https://github.com/hayeah/playing-cards-assets) — licença **MIT** (Howard Yeh).
- **Arte das cartas:** projeto *Vector Playing Cards* — **domínio público**.
- **Uso:** estritamente **acadêmico/diagnóstico**. As imagens não são versionadas no Git
  (`data/raw/*` está no `.gitignore`); o script as regenera quando necessário (inclusive no Colab).
- As imagens **não** são fotos tiradas pelos autores — isso deve ficar explícito no relatório.

---

## 4. Como reproduzir

**No Colab / com Python e internet:**
```bash
python -m src.ood_design --out data/raw/ood_design_web
```
O script baixa os PNGs individuais e monta as 53 pastas. A célula 8 do notebook
(`notebooks/treino_cartas_colab.ipynb`) já chama esse script e roda a avaliação OOD.

**Se você já tem os PNGs localmente** (ex.: clonou o repositório de assets), pule o download:
```bash
python -m src.ood_design --out data/raw/ood_design_web --assets-dir CAMINHO/para/png
```

**Mapeamento de nomes** (origem → classe de treino): valores numéricos viram palavras
(`2 → two`, ..., `10 → ten`), `ace/jack/queen/king` ficam iguais, naipes ficam iguais
(`clubs/diamonds/hearts/spades`), e os dois coringas (`black_joker`, `red_joker`) vão para a
única classe `joker`.

---

## 5. Trabalho futuro — OOD com fotos reais (gap de captura)

O caminho **padrão-ouro** continua sendo fotografar um **baralho físico** sob condições reais,
o que mede o gap que este conjunto não cobre. O passo a passo (com cuidados de privacidade/LGPD)
está em [`guia_coleta_baralho_real.md`](guia_coleta_baralho_real.md). Nesse caso, ponha as fotos
em `data/raw/ood_baralho_real/` e aponte `OOD_DIR` no notebook para essa pasta — o restante do
pipeline é idêntico.
