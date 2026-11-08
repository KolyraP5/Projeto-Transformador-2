# Teste pessoal zero-shot e tendência dos últimos 30 minutos

Foi realizada uma avaliação externa exploratória, sem novo treinamento, no
[histórico pessoal](../../data/pessoal/Stream2026-11-07-23_44_52.tsv). A execução
foi assistida por IA a pedido da equipe. A inferência levou **7,9 segundos** e
a comparação adicional com a regra de 30 minutos levou cerca de **2 segundos**.

## Dados e preparação

- 228.733 leituras, 14 sensores e intervalo nativo mediano de **1 minuto**.
- Alinhamento causal para a grade de **5 minutos** esperada pela LSTM: última
  leitura disponível até cada instante, sem interpolação.
- Intervalos com mais de um sensor foram invalidados; nenhuma janela atravessa
  troca de sensor ou lacuna.
- **38.055 janelas válidas**, com 12 leituras de entrada e previsão a **+15 min**.
- As colunas `Rate` e `ChangeLabel` do arquivo não entraram como atributos ou alvos.

O modelo `combined_B3_s42` foi escolhido pela triagem no AZT1D **antes** desta
avaliação. Seus pesos e escalas permaneceram congelados. Sua confirmação ampliada
foi interrompida por prazo: ele é uma candidata provisória, sem aprovação final.
Também foram avaliados a LSTM de `full_v1` e os baselines nas mesmas janelas.

## Comparação com a regra informada do site

A regra descrita pela equipe usa a variação nos últimos 30 minutos:

`taxa_passada = (G(t) - G(t-30)) / 30`.

Ela descreve uma tendência **retrospectiva**. Para comparar sua utilidade como
previsão, extrapolou-se essa taxa para o mesmo horizonte futuro dos modelos:

`glicemia_prevista(t+15) = G(t) + 15 × taxa_passada`.

Não foram presumidos outros filtros ou regras da implementação do site.

| Método | MAE taxa (mg/dL/min) | MAE glicemia +15 min (mg/dL) | RMSE glicemia (mg/dL) | Acerto de direção |
|---|---:|---:|---:|---:|
| Tendência dos últimos 30 min | 1,0380 | 15,570 | 21,370 | 51,83% |
| Combinado zero-shot, provisório | **0,8649** | **12,973** | **17,359** | 58,13% |
| LSTM `full_v1` | 0,8720 | 13,079 | 17,540 | 58,31% |
| Persistência | 0,8652 | 12,978 | 17,809 | 0,00% |

O combinado reduziu o MAE em aproximadamente **16,7%** frente à extrapolação da
tendência de 30 minutos. Entretanto, seu MAE ficou **praticamente empatado com
a persistência**: o ganho relativo foi de apenas cerca de 0,03%. O ganho agregado
observado no AZT1D não se transferiu de maneira relevante para esse histórico.

Em mudanças rápidas, o MAE da taxa foi **1,5085** no combinado e **1,7539** na
regra de 30 minutos. O acerto direcional nesse subconjunto foi **67,32%** e
**52,55%**, respectivamente. O acerto de direção exclui mudanças reais exatamente
zero; a persistência prevê zero e não antecipa a direção de mudanças não nulas.

## Interpretação e reprodução

Esta é uma avaliação de uma pessoa e de seu dispositivo, com janelas sobrepostas.
Não demonstra generalização populacional nem validade clínica. A regra do site
foi comparada como previsão por extrapolação; isso não avalia a qualidade de sua
descrição do passado nem reproduz toda a implementação do site. Os resultados
pessoais não foram usados para reajustar ou selecionar o modelo.

Scripts: [zero-shot](../../scripts/testar_pessoal_zeroshot.py) e
[comparação de 30 minutos](../../scripts/comparar_tendencia_30min.py).
Resultados locais em `results/pessoal/zeroshot_2026_11_07/`: `RELATORIO.md`,
`COMPARACAO_SITE_30MIN.md`, `metricas.csv`, `comparacao_site_30min.csv`, auditoria,
previsões e gráfico. Esses resultados detalhados são ignorados pelo Git.
