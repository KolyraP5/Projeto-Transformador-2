AZT1D TREND-ARROW PROJECT — ARTIFACT INVENTORY

Cloud folder:
https://github.com/KolyraP5/Projeto-Transformador-2

This repository documents the team's transition from the initial glucose
forecasting and fine-tuning proposal to a continuous CGM trend-rate arrow.
Paths below are relative to the repository root.

1. DATA

AZT1D 2025/CGM Records/
  Original subject CSV files, Subject 1 through Subject 25. These files contain
  the source records used by the current pipeline. The data are not modified.
  Subject 14 is present in the source but excluded from full_v1 because the
  "Readings (CGM / BGM)" channel is not confirmed as CGM.

AZT1D 2025/Visual Statistics/
  Auxiliary figures distributed with the dataset.

AZT1D 2025/Manuscript.pdf
AZT1D 2025/README.txt
  Dataset reference manuscript and the dataset's included description.

Dataset source: AZT1D v1, Mendeley Data, DOI 10.17632/gk9m674wcx.1, CC BY 4.0.

2. CURRENT SOURCE AND EXECUTION ENTRY POINT

notebooks/atual/AZT1D_Setas_e_Tendencia.ipynb
  Self-contained current notebook. It performs data preparation and audit,
  baseline evaluation, LSTM training at +5 and +15 minutes, horizon selection
  from validation, test evaluation, plots, and report generation.

src/seta_tendencia/azt1d_core.py
  Readable source for the version-1 shared data, modeling, evaluation, and
  plotting functions. The full_v1 archived source hash is recorded in its run
  manifest and execution record.

scripts/executar_experimento.py
  Runs the notebook and archives its outputs, source, environment, and log.
scripts/verificar_experimento.py
  Checks run integrity, partition separation, weights, predictions, and metrics.
scripts/analisar_resultados.py
  Recomputes result analyses without retraining.
scripts/empacotar_resultados.py
  Creates and integrity-checks the full_v1 results archive.

requirements.txt
  Python package requirements for execution. Exact installed versions are
  preserved separately in the full_v1 run.

3. COMPLETED REFERENCE RUN: full_v1

results/azt1d_seta/runs/full_v1/
  Complete run record: preparation audits, temporal splits, dataset manifest,
  frozen horizon-selection record, verification report, execution metadata and
  log, executed notebook, exact run source, and environment freeze.

results/azt1d_seta/runs/full_v1/lstm_h5/
  +5-minute LSTM configuration, validation metrics, training history, plot,
  environment record, completion record, and trained weights:
  results/azt1d_seta/runs/full_v1/lstm_h5/best_weights.pt.

results/azt1d_seta/runs/full_v1/lstm_h15/
  +15-minute LSTM configuration, validation metrics, training history, plot,
  environment record, completion record, and trained weights:
  results/azt1d_seta/runs/full_v1/lstm_h15/best_weights.pt.

results/azt1d_seta/runs/full_v1/baselines/
  Validation metrics and predictions for persistence, last variation, and
  linear-trend baselines.

results/azt1d_seta/runs/full_v1/final/
  Final validation/test metrics, per-model gains, test predictions, test
  completion record, inference timing, summary, and auxiliary figures:
  comparacao_temporal.png, primeiro_episodio_rapido.png,
  setas_continuas.png, and taxa_real_vs_prevista.png.

results/azt1d_seta/runs/full_v1/analise/
  Per-participant comparison, intensity diagnostics, numerical analysis, and
  comparison figure.

results/azt1d_seta/runs/full_v1_resultados.zip
  Integrity-checked archive of the full_v1 run, including compressed
  predictions and figures. Regenerable prepared-array caches are excluded.

docs/analises/ANALISE_FULL_V1.md
  Interpretation of the complete run, selection, outcomes, and limitations.
docs/analises/ANALISE_DATASET_AZT1D.md
  Dataset structure, quality, and preprocessing considerations.
docs/PROTOCOLO_ATUAL.md
  Current experiment definition, targets, splits, baselines, and metrics.
docs/HISTORICO.md
  Project stages, evidence, and documented changes of direction.
docs/PLANO_APRIMORAMENTO.md
  Proposed next development steps; these changes are not yet implemented.
docs/EXECUCAO.md
  Reproduction commands, environment requirements, artifact locations, and
  verification instructions.
docs/templates/REGISTRO_TENTATIVA.md
  Template for recording future experimental attempts.

4. QUICK CHECK AND HISTORICAL WORK

results/azt1d_seta/runs/quick_v1/
  Reduced three-participant, up-to-three-epoch pipeline check. Its summary and
  artifacts are exploratory, not full-dataset evidence.
notebooks/arquivo/03_seta_quick_v1_executado.ipynb
  Preserved executed quick-run notebook.

notebooks/arquivo/01_auditoria_azt1d_e_protocolo_v0.ipynb
notebooks/arquivo/02_preprocessamento_e_baseline_persistencia.ipynb
src/preprocessing/
results/data_audit/
results/preprocessing_baseline/
  Archived v0 audit and +30-minute persistence-baseline artifacts. They use a
  different task and protocol from the current trend-arrow experiment.

docs/planejamento-inicial/
  Preserved initial project proposal, bibliography, figures, and original
  description. It documents the earlier plan, not the current implementation.

notebooks/README.md
  Notebook index. notebooks/exploracao/ is reserved for ongoing studies.
README.md
  Main repository entry point and navigation.

5. full_v1 EXECUTION PARAMETERS AND INTERPRETATION

Run identifier: full_v1
Mode: full (QUICK=False)
Input: 12 historical CGM readings on a nominal 5-minute grid
Targets: mean future CGM rate (G(t+h)-G(t))/h, for h = 5 and 15 minutes
Eligible participants: 24 of 25; Subject 14 excluded for the unconfirmed alias
Temporal split: 60% train, 20% validation, 20% test, per participant
Baselines: persistence, last variation, and fitted linear trend
Models: independent one-layer LSTM per horizon, 32 hidden units, linear output
Training: seed 42, Adam learning rate 0.001, batch size 64, standardized MSE,
          maximum 15 epochs and patience 3; both models ran 15 epochs
Device and environment: CPU; Python 3.14.8, PyTorch 2.15.0, NumPy 2.5.3,
                        pandas 2.3.3; see requirements-freeze.txt for full list

The +15-minute horizon was selected using validation before test metrics were
calculated. On this run, its test macro MAE was 0.5444 mg/dL/min versus 0.6640
for persistence, an 18.02% reduction, with lower MAE for all 24 participants.
The test period had previously been accessed in quick_v1; the full_v1 record
marks this evaluation as exploratory. Participants are split by time, so this
run does not establish generalization to unseen people or clinical validity.
The continuous arrow describes average future change, not the full trajectory.

To reproduce with a new run identifier, follow docs/EXECUCAO.md. Do not overwrite
full_v1. Dataset citation and license information are recorded above and in the
project documentation.
