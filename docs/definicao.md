# Definição do problema e requisitos

## Problema
Classificação supervisionada multiclasse de **uma única carta de baralho já
recortada** em **53 classes** (52 cartas + 1 coringa). Entrada: imagem RGB
224×224. Saída: um rótulo. **Detecção** de várias cartas em cena está fora do
escopo (trabalho futuro).

## Motivação e impacto humano
- **Primário — educacional:** ensino de probabilidade, regras de jogos e
  matemática (crianças, idosos); também ensino de visão computacional.
- **Secundário — acessibilidade:** leitura da carta em voz alta para pessoas
  com deficiência visual.
- **Secundário — pesquisa/benchmark:** problema-modelo bem delimitado de VC.

## Público-alvo
Estudantes e educadores; pessoas com deficiência visual; pesquisadores/estudantes
de visão computacional.

## Requisitos funcionais
- Receber uma imagem de carta recortada e retornar a classe prevista.
- Gerar métricas (accuracy, F1-macro) e matriz de confusão.
- Permitir predição em imagem única via linha de comando.

## Requisitos não funcionais
- **Reprodutibilidade:** semente fixa (42), dependências pinadas.
- **Privacidade (LGPD):** processamento on-device, sem armazenar imagens
  desnecessárias.
- **Ética:** declarar usos proibidos (ver `etica.md`).
- **Desempenho:** rodar em GPU gratuita (Colab T4).

## Métricas de sucesso
**Accuracy** e **F1-macro** no teste (in-distribution) e **accuracy OOD**
(generalização para outro design). Resultados em [`../README.md`](../README.md).
