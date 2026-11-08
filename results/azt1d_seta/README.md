# Execuções da seta de tendência

| Execução | Papel | Evidência |
|---|---|---|
| `quick_v1` | Conferência reduzida, três participantes | [Resumo original](runs/quick_v1/final/resumo.md), com saídas preservadas |
| `full_v1` | Rodada completa da versão 1, 24 participantes | [Notebook executado](runs/full_v1/notebook_executado.ipynb), [métricas](runs/full_v1/final/metricas.csv) e [análise](../../docs/analises/ANALISE_FULL_V1.md) |

No `full_v1`, +15 min foi escolhido na validação; a LSTM reduziu o MAE macro
de teste em 18,02% frente à persistência. Há limitações por intensidade, descritas
na análise. A execução é exploratória devido ao acesso anterior a parte do teste.

Cada pasta preserva configuração, auditoria, pesos, ambiente e métricas.
`full_v1` inclui também cópias das fontes, notebook executado, verificação e
diagnósticos adicionais. Caches `prepared/`, previsões compactadas e ZIPs
permanecem locais e são ignorados pelo Git. Para reproduzir, siga o
[guia](../../docs/EXECUCAO.md).
