# AZT1D — Seta contínua: resultados executados

- Execução: full_v1
- Modo rápido: False
- Participantes: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
- Horizonte escolhido antes do teste: 15
- Teste já consultado em trabalhos anteriores: True
- Taxa média futura em mg/dL/min; sem classes e sem simulação de doses.

| Partição | Horizonte | Modelo | MAE taxa (média por pessoa) | RMSE glicemia (média por pessoa) |
|---|---:|---|---:|---:|
| val | 5 | linear_15min | 0.7473 | 5.582 |
| val | 5 | lstm | 0.6196 | 4.601 |
| val | 5 | persistencia | 0.8095 | 5.782 |
| val | 5 | ultima_variacao | 0.7000 | 5.356 |
| val | 15 | linear_15min | 0.6982 | 15.396 |
| val | 15 | lstm | 0.5429 | 11.607 |
| val | 15 | persistencia | 0.6683 | 13.918 |
| val | 15 | ultima_variacao | 0.7361 | 16.529 |
| test | 5 | linear_15min | 0.7585 | 5.630 |
| test | 5 | lstm | 0.6282 | 4.664 |
| test | 5 | persistencia | 0.8093 | 5.797 |
| test | 5 | ultima_variacao | 0.7148 | 5.435 |
| test | 15 | linear_15min | 0.7020 | 15.455 |
| test | 15 | lstm | 0.5444 | 11.701 |
| test | 15 | persistencia | 0.6640 | 13.902 |
| test | 15 | ultima_variacao | 0.7475 | 16.692 |

## Interpretação

A seleção usa validação, ganho sobre baseline e comportamento nas mudanças rápidas.
Consulte selecao_horizonte.json para a regra e a decisão congelada.
Se o horizonte escolhido for null, nenhum apresentou evidência suficiente pela regra operacional.
As métricas de teste confirmam ou contradizem a decisão; não reajuste os modelos neste teste.

## Limitações

Janelas sobrepostas não são amostras independentes. Avaliação temporal nas mesmas pessoas.
Agrupamento em 5 minutos; diferenças dos timestamps reais estão na auditoria.
Bins conflitantes e lacunas foram descartados, podendo mudar a população avaliada.
O modo rápido verifica execução; não sustenta conclusão sobre toda a base.
A seta resume a mudança média futura do CGM e não detecta todos os picos intermediários.
A escala angular é uma escolha visual do projeto, sem validação clínica.

## Fonte

Khamesian et al. AZT1D, versão 1. https://doi.org/10.17632/gk9m674wcx.1
