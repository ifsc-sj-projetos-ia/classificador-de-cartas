---
title: Classificador de Cartas
emoji: 🃏
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Classificador de Cartas de Baralho

Demo de visão computacional: envie a foto de **uma carta de baralho recortada**
e o modelo (EfficientNet-B0, *transfer learning*) diz qual é, entre 53 classes
(52 cartas + coringa). Inclui leitura em voz alta (acessibilidade) e uma página
com as métricas reais do modelo.

> Projeto acadêmico ICD/ADS — Nicolas & Herick. Uso educacional/acessibilidade.
> **Não** se destina a apostas em tempo real.

O código-fonte completo, treino e relatório estão no repositório do projeto.
Esta é apenas a aplicação de demonstração (inferência em CPU).
