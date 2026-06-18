# Avaliação ética e usos proibidos

## Riscos e mitigações

### (a) Uso dual / trapaça — **USO PROIBIDO**
A mesma tecnologia pode ser convertida em **assistência em tempo real**
(*Real-Time Assistance*, RTA) em jogos de azar e apostas, prática
**expressamente proibida** por plataformas como **PokerStars** e **GGPoker**.
Declara-se que empregar este projeto para esse fim é um **uso proibido**.

### (b) Jogo problemático
Ferramentas que facilitem apostas podem contribuir para comportamento de jogo
com dano social documentado. O projeto **não deve** ser aplicado a esse fim.

### (c) Viés e limitação do dataset
Treinado em **um único design** de baralho → tende a falhar em outros estilos,
iluminações e oclusões. Evidenciado pelo **gap OOD de ~35 pontos percentuais**
(Experimento 3). Mitigação: ampliar os dados de treino e declarar claramente os
contextos de uso.

### (d) Privacidade (LGPD)
A câmera pode captar o ambiente e terceiros. Mitigações recomendadas:
**minimização de dados**, **processamento on-device** e **não armazenamento**
de imagens desnecessárias.

### (e) Documentação ética
Publicar o sistema sem declarar os usos proibidos é, em si, um risco. Este
documento, o `model_card.md` e o relatório final mitigam isso.

## Contextos em que NÃO deve ser usado
- Assistência em jogos de azar/apostas (RTA).
- Qualquer decisão crítica que dependa de 100% de acerto sem revisão humana.
- Designs de baralho, iluminações ou enquadramentos muito distintos do treino,
  sem reavaliação prévia (ver gap OOD).
