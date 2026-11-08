# AZT1D — Seta contínua: resultados executados

- Execução: quick_v1
- Modo rápido: True
- Participantes: [1, 2, 3]
- Horizonte escolhido antes do teste: 15
- Teste já consultado em trabalhos anteriores: False
- Taxa média futura em mg/dL/min; sem classes e sem simulação de doses.

| Partição | Horizonte | Modelo | MAE taxa (média por pessoa) | RMSE glicemia (média por pessoa) |
|---|---:|---|---:|---:|
| val | 5 | linear_15min | 0.6615 | 5.255 |
| val | 5 | lstm | 0.6256 | 4.753 |
| val | 5 | persistencia | 0.7490 | 5.465 |
| val | 5 | ultima_variacao | 0.6184 | 5.267 |
| val | 15 | linear_15min | 0.6326 | 14.519 |
| val | 15 | lstm | 0.5557 | 11.773 |
| val | 15 | persistencia | 0.6253 | 13.112 |
| val | 15 | ultima_variacao | 0.6550 | 15.684 |
| test | 5 | linear_15min | 0.7586 | 5.771 |
| test | 5 | lstm | 0.6912 | 5.139 |
| test | 5 | persistencia | 0.7896 | 5.709 |
| test | 5 | ultima_variacao | 0.7277 | 5.896 |
| test | 15 | linear_15min | 0.6981 | 15.468 |
| test | 15 | lstm | 0.5821 | 12.301 |
| test | 15 | persistencia | 0.6345 | 13.361 |
| test | 15 | ultima_variacao | 0.7511 | 17.268 |

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
