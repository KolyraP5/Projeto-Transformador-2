# Tentativas anteriores recuperadas

## Contexto e procedência

Cinco notebooks recebidos em `Downloads` foram copiados sem alteração para
[`notebooks/arquivo/tentativas_recebidas/`](../../notebooks/arquivo/tentativas_recebidas/).
Os nomes originais foram preservados. Os arquivos não contêm saídas de células:
as células de código estão sem contagem de execução, sem tabelas resultantes e
sem erros registrados. Assim, eles permitem descrever perguntas, métodos e
parâmetros escritos no código, mas não validar quantitativamente o relato da
equipe de que as tentativas tiveram resultados pouco animadores ou não
funcionaram.

Datas, responsáveis individuais, versões do ambiente e resultados numéricos
dessas tentativas não estão registrados nos notebooks recebidos. Não foram
inferidos. Nenhum notebook foi executado como parte desta organização.

## Inventário

| Notebook arquivado | Pergunta e abordagem documentadas | Configuração identificável | Evidência disponível |
|---|---|---|---|
| [Treino_Simples_Glicose.ipynb](../../notebooks/arquivo/tentativas_recebidas/Treino_Simples_Glicose.ipynb) | Se uma regressão linear sobre histórico de CGM e somas recentes de insulina/carboidratos prevê a próxima leitura. | Subjects 1–3; 12 leituras anteriores; próxima leitura na grade. A descrição propõe somas de insulina em 4 h e carboidratos em 2 h. | Notebook-fonte sem execução. Um texto explicativo antecipa contribuição pequena dos eventos no horizonte curto, mas não há métricas salvas para confirmar essa observação. |
| [01_Arvores_Previsao_Glicemica.ipynb](../../notebooks/arquivo/tentativas_recebidas/01_Arvores_Previsao_Glicemica.ipynb) | Prever CGM a +30 min com persistência, árvores com CGM e árvores com CGM mais resumos de carboidratos/bolus. | `QUICK=True`, semente 42 e Subjects 1–3; o modo rápido limita o treino a até 20.000 exemplos. Divisão temporal por participante 60/20/20. | Notebook-fonte sem execução; não há métricas, modelos treinados nem arquivos de resultado associados nesta entrega. |
| [02_LSTM_Acionada_por_Refeicao.ipynb](../../notebooks/arquivo/tentativas_recebidas/02_LSTM_Acionada_por_Refeicao.ipynb) | Previsão acionada por intervalos com registro de carboidrato, usando LSTM de CGM e vetor contextual de eventos, contra persistência. | `QUICK=True`, semente 42 e Subjects 1–3; alvos a +30 e +60 min. Os exemplos representam refeições/intervalos registrados, não todos os instantes. | Notebook-fonte sem execução nem métricas salvas. A população de exemplos é específica a eventos e não é diretamente comparável a avaliações sobre todas as origens. |
| [03_LSTM_Seta_Futura_15min.ipynb](../../notebooks/arquivo/tentativas_recebidas/03_LSTM_Seta_Futura_15min.ipynb) | Converter a mudança de glicemia a +15 min em três classes: descida, estável e subida. | `QUICK=True`, semente 42 e Subjects 1–3; LSTM com 12 valores de CGM, 32 unidades, Adam 0,001, lote 64, até 3 épocas no modo rápido; limites de classe de ±15 mg/dL. | Notebook-fonte sem execução nem matriz de confusão. O alvo categórico e os limites fixos diferem da taxa contínua usada pelo algoritmo atual. |
| [Previsao_Glicemica_AZT1D_TrilhaA.ipynb](../../notebooks/arquivo/tentativas_recebidas/Previsao_Glicemica_AZT1D_TrilhaA.ipynb) | Comparar uma escada de atributos tabulares: CGM, impulsos de eventos, recências e estados de kernels causais para insulina/carboidratos. | Subjects 1–3; horizontes +30, +60 e +120 min; baseline de persistência e divisão temporal descrita no notebook. | Notebook-fonte sem execução nem tabela de resultados. A cópia não permite concluir se os atributos adicionais ajudaram ou prejudicaram. |

## Interpretação e limites

- O relato de resultados insatisfatórios e de ideias que não avançaram é da
  equipe. Os artefatos recebidos registram tentativas e desenhos, mas não as
  execuções que permitiriam quantificar esse relato.
- O texto do notebook de regressão simples comenta que, no horizonte de uma
  leitura, eventos podem acrescentar pouco ao erro. Sem a saída executada, essa
  observação permanece uma justificativa/expectativa escrita, não um resultado
  medido que possa ser reproduzido aqui.
- Os notebooks usam tarefas e populações de avaliação diferentes: próxima
  leitura, eventos de refeição, alvos de +30/+60/+120 min e classes de seta a
  +15 min. Não se devem comparar métricas entre essas tentativas e `full_v1`
  sem recuperar as saídas e alinhar protocolo, participantes, horizonte e
  métrica.
- O protótipo categórico de seta é histórico. O protocolo vigente prediz a taxa
  média futura de forma contínua e seleciona +15 min por validação.

## Material não incorporado

O arquivo recebido com nome `Histórico2026-11-07-23_23_22.tsv` contém uma série
temporal de medições de glicose com identificador de sensor, e não um histórico
de mentoria ou de execução dos notebooks. Ele não foi copiado nem incluído na
documentação pública, para evitar divulgar dados de saúde que não são
necessários para organizar estas tentativas. Não foi usado nos experimentos
AZT1D.

## Próximas evidências, se recuperadas

Para transformar o relato retrospectivo em resultados verificáveis, ainda
seriam necessários os notebooks executados ou, ao menos, logs, configurações,
previsões, métricas e pesos correspondentes. A autoria deve ser atribuída pela
equipe conforme seus registros; não pode ser inferida apenas pelo arquivo.
