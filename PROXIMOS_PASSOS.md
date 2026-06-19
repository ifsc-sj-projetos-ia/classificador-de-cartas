# ✅ Guia de Passos Humanos

Checklist das etapas que precisam de ação humana para concluir o projeto.
(As etapas de código/estrutura já estão prontas no repositório.)

🔗 Repositório: <https://github.com/nicolasfvp/reconhecimento-cartas> (público)
👥 Grupo: Nicolas e Herick

---

- [x] **1. Criar o repositório no GitHub e dar push**
  Repositório criado e código + documentação enviados.

- [x] **2. Tornar o repositório público**
  Necessário para clonar no Colab sem token e compartilhar o link.

- [x] **3. Configurar acesso ao Kaggle (`kaggle.json`)**
  Token criado para baixar o dataset *Cards Image Dataset-Classification* (gpiosenka).

- [x] **4. Rodar o notebook no Google Colab (GPU T4)**
  `notebooks/treino_cartas_colab.ipynb` executado: setup → dataset → EDA → baseline → treino → avaliação → experimentos.

- [x] **5. Montar o conjunto de teste OOD ("design diferente", web)**
  Em vez de fotografar (sem tempo), usamos um baralho de **design diferente** com imagens limpas
  de licença livre, gerado por `python -m src.ood_design --out data/raw/ood_design_web`
  (53 classes, nomes de pasta exatos). Mede o **gap de design**. A célula 8 do notebook já o monta e avalia.
  Guia: `docs/guia_ood_design_web.md`.

- [ ] **5b. (Opcional / trabalho futuro) Fotografar 1 baralho real — gap de captura**
  Padrão-ouro: seguir `docs/guia_coleta_baralho_real.md` (nomes de pasta exatos + LGPD), pôr em
  `data/raw/ood_baralho_real/`, ajustar `OOD_DIR` no notebook e medir o gap de **condições de captura**
  (que o OOD de design não cobre).

- [x] **6. Preencher as métricas reais**
  Números do treino (Colab T4) preenchidos em `README.md`, `docs/MODEL_CARD.md`, `docs/02_dados.md`,
  `docs/01_definicao_problema.md`, `docs/03_etica_impacto.md` e na tabela do `relatorio_final_outline.md`:
  teste **94,7%** / F1-macro **0,947**, baseline 70,6%, OOD de design **59,3%** (gap ≈ 35 pp).
  Figuras/CSVs commitados em `reports/` (matrizes de confusão do teste e do OOD, `experimentos.csv`,
  `classification_report_test.csv`). OOD do modelo **sem** aug medido (0,593 — mesma acurácia do com-aug).
  **Lado técnico 100% concluído** — resta apenas escrever o relatório/apresentação (passos 7 e 8).

- [ ] **7. Confirmar a licença exata do dataset gpiosenka**
  No Kaggle a licença está marcada como *"Other"* — confirmar antes de redistribuir imagens.

- [ ] **8. Escrever o relatório final (PDF) e a apresentação (10-15 min)**
  Usar o template `reports/relatorio_final_outline.md` (já traz a estrutura + roteiro da apresentação).
  Exportar o relatório em PDF para a entrega.

---

> Última atualização: 2026-06-12 — passos 1–6 concluídos (métricas reais preenchidas); resta gerar/commitar as figuras (4) e, opcional, fotos reais (5b).
