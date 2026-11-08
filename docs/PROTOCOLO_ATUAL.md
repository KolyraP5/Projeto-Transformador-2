# Protocolo atual: seta contínua de tendência

Este documento descreve a implementação usada na rodada `full_v1`. A direção
do projeto é aprimorar este algoritmo de seta de tendência. A decisão de manter
esse tema foi informada pela equipe; os resultados de outras tentativas devem
ser anexados ao [histórico](HISTORICO.md) para sustentar comparações entre elas.

Fontes do método: [notebook atual](../notebooks/atual/AZT1D_Setas_e_Tendencia.ipynb)
e [código compartilhado](../src/seta_tendencia/azt1d_core.py), versão `1.0.0`.
Os parâmetros efetivamente usados são registrados no `manifest.json` e nos
`config.json` de cada execução.

## Pergunta e alcance

O algoritmo estima a taxa média futura de variação do CGM a partir de leituras
recentes e a representa por uma seta contínua. Para o horizonte `h`, o alvo é
`r = (G(t+h) - G(t))/h`, em mg/dL/min. Os horizontes implementados são 5 e
15 minutos, treinados em duas LSTMs independentes.

A avaliação atual verifica previsão em períodos posteriores **das mesmas
pessoas presentes no treino**. Ela não verifica generalização para uma pessoa
nova, nem implementa o transfer learning/fine-tuning da proposta inicial.
Essa mudança de pergunta e de divisão dos dados impede comparar diretamente
as métricas desta rodada com as do baseline histórico de +30 minutos.

## Dados e preparação

- Fonte: CSVs de `AZT1D 2025/CGM Records/`, identificados por `Subject N`.
- Colunas exigidas: `EventDateTime` e `CGM`. O alias
  `Readings (CGM / BGM)` permanece desabilitado (`ALLOW_MIXED_ALIAS=False`),
  excluindo o Subject 14 desta cópia da base. Sua inclusão exige confirmar
  que as leituras relevantes são CGM e registrar nova configuração.
- Datas: `DATE_FORMAT="mixed"`, `DAYFIRST=False`; o leitor não converte fuso.
  As amostras de datas e os períodos devem ser conferidos na auditoria.
- Duplicatas completas são removidas. Glicemias não numéricas, não finitas
  ou menores ou iguais a zero são invalidadas. Valores 40 e 400 permanecem
  na análise e são contados na auditoria.
- A grade tem passo de cinco minutos. Cada leitura é atribuída ao fechamento
  seguinte do intervalo (`ceil("5min")`), de modo que uma origem não recebe
  uma leitura com timestamp posterior. Intervalos com valores diferentes
  conflitantes são invalidados.
- Não há interpolação nem preenchimento de lacunas. Uma janela somente é
  aceita se as 12 entradas e os três pontos futuros forem finitos.
- Os dois horizontes usam as mesmas origens. Isso permite compará-los na mesma
  amostra, mas exige suporte até +15 minutos também para avaliar +5 minutos;
  a cobertura de +5 minutos fica mais restrita do que poderia ser isoladamente.

Os tempos dos alvos são definidos na grade. A diferença entre os timestamps
reais pode não ser exatamente de 5 ou 15 minutos. O arquivo
`auditoria_horizontes_reais.csv` registra esses desvios; eles precisam acompanhar
a interpretação dos resultados.

## Divisão temporal e entradas

Para cada pessoa, a grade é dividida cronologicamente em 60% para treino, 20%
para validação e 20% para teste. **A divisão acontece antes da construção das
janelas**, mantendo entradas e alvos inteiramente dentro da respectiva partição.
Só entram participantes com pelo menos uma janela válida em cada partição.

Cada entrada contém 12 leituras de CGM, da posição −55 minutos até a origem
em zero. O histórico é chamado de aproximadamente 60 minutos porque são
12 posições em uma grade de cinco minutos. Insulina, carboidratos, identidade
da pessoa e horário do dia não são entradas do modelo atual.

`QUICK=True` usa os três primeiros participantes elegíveis, limita o treino a
20.000 janelas quando necessário e permite até três épocas. `QUICK=False`
usa todos os participantes e todas as janelas elegíveis e permite até 15 épocas.
A seleção de participantes e os descartes constam do manifesto e das auditorias;
“rodada completa” não significa aproveitamento de todas as linhas brutas.

## Modelo e treinamento

Cada LSTM possui uma camada recorrente com 32 unidades e uma camada linear com
saída escalar. O estado recorrente é reiniciado a cada janela. A saída representa
a taxa média futura padronizada, convertida de volta para mg/dL/min na inferência.

As médias e os desvios usados para padronizar entrada e alvo são calculados
somente no treino. O treinamento usa Adam, taxa de aprendizado `0.001`, MSE,
lotes de 64 e semente 42. A validação usa lotes de até 1.024. A melhor perda
de validação salva os pesos; três épocas consecutivas sem melhora encerram o
treinamento. Os melhores pesos são restaurados para a avaliação.

O dispositivo é CUDA quando disponível ou CPU. A semente e as versões ficam
registradas, mas não garantem identidade numérica entre dispositivos ou versões.
Treino interrompido recomeça do início; não há retomada do estado do otimizador.
Um treino concluído é reutilizado quando sua identidade é compatível.

## Referências de comparação e métricas

Os três baselines usam somente o histórico disponível na origem:

| Baseline | Taxa prevista |
|---|---|
| `persistencia` | Zero; equivale a manter a glicemia atual |
| `ultima_variacao` | Diferença das duas últimas leituras dividida por cinco minutos |
| `linear_15min` | Inclinação da reta ajustada às quatro últimas leituras |

As previsões são avaliadas nas mesmas origens por modelo e horizonte. São
registrados MAE, RMSE e viés da taxa, MAE/RMSE de glicemia reconstruída e
acerto de direção. A glicemia reconstruída é `G(t) + h * taxa_prevista`.

Há três formas de agregação: por participante (`subject`), todas as janelas
reunidas (`pooled`) e média das métricas por participante (`macro`). A regra
principal usa `macro`, dando o mesmo peso a cada pessoa avaliável. A coluna
`n` da linha macro soma as janelas; ela não muda esse peso. Não se deve
interpretar o RMSE macro como o RMSE calculado sobre todas as janelas juntas.

O subconjunto de mudanças rápidas contém `|taxa_real| >= 1 mg/dL/min` e serve
como diagnóstico. O acerto de direção considera apenas alvos com taxa real
diferente de zero. Prever exatamente zero conta como erro de direção nesses
casos; sinal oposto é registrado separadamente. Assim, persistência não tem
sinais opostos, mas também não acerta a direção de mudanças não nulas.

## Escolha do horizonte antes do teste

A implementação aplica esta regra operacional somente à validação:

1. Identificar o baseline com menor MAE macro da taxa em cada horizonte.
2. Exigir que a LSTM tenha MAE macro menor e melhore em mais da metade dos
   participantes em relação a esse baseline.
3. Exigir que a LSTM não piore o MAE macro nem o acerto de sinal nas mudanças
   rápidas. Sem suporte válido nesse diagnóstico, o horizonte não é candidato.
4. Entre candidatos, escolher o maior ganho relativo no MAE. Diferença de até
   `0.02` no ganho relativo é empate operacional e favorece 15 minutos.
5. Sem candidato, registrar `chosen_horizon_min=null` e relatar ausência de
   evidência suficiente pela regra atual.

O ganho é `1 - MAE_LSTM/MAE_baseline`. A regra não exige superioridade em
todos os critérios contra todos os baselines: o diagnóstico rápido é comparado
ao baseline escolhido pelo MAE geral daquele horizonte. Essa distinção deve
ser preservada na análise, sobretudo para a última variação.

A seleção, o hash das métricas de validação e os hashes dos pesos ficam em
`selecao_horizonte.json` antes do teste. O teste recebe ambos os modelos para
diagnóstico, mas seus resultados não devem trocar retroativamente a escolha.
Depois de congelada a seleção, outro treinamento exige novo `RUN_NAME`.

## Acesso anterior ao teste

`quick_v1` já produziu previsões e métricas de teste. As pessoas dessa execução
também participam da rodada completa. Portanto, `full_v1` deve ser registrada
com `exploratory_prior_test_access=true`, mesmo mantendo a escolha do horizonte
restrita à validação nesta rodada. Não há novo conjunto externo intocado por
simplesmente trocar `RUN_NAME` ou mudar de computador.

As configurações anteriores, incluindo o valor originalmente salvo em
`quick_v1`, devem permanecer como registro histórico. A informação de acesso
prévio precisa acompanhar a análise de `full_v1` e as próximas decisões.

## Conversão em seta e limites

Com referência visual fixa de 1 mg/dL/min, calcula-se `q=taxa_prevista/1`,
`ângulo=arctan(q)` e vetor unitário `(1,q)/sqrt(1+q²)`. Não há três classes
discretas. Essa transformação visual não acrescenta poder preditivo ao modelo.

A seta resume a mudança média até o horizonte; não garante trajetória monotônica,
não mostra todos os picos intermediários e não representa a derivada instantânea
no final do intervalo. As convenções visuais e a regra de escolha do horizonte
não possuem validação clínica neste projeto.

Outros limites da avaliação são janelas sobrepostas, dependência temporal,
avaliação nas mesmas pessoas, exclusão de lacunas/conflitos e de uma coluna
ambígua, uma única semente na rodada de referência e consulta prévia ao teste.
Esses resultados descrevem este conjunto e protocolo e não sustentam, sozinhos,
uso clínico ou recomendação de dose.

## Regras para aprimorar o algoritmo final

As próximas tentativas devem manter a tarefa da seta contínua e comparar cada
alteração a `full_v1`, registrando hipótese, configuração, dados, custo e resultado.
O plano específico depende da [análise da rodada completa](analises/ANALISE_FULL_V1.md).

Antes de ajustar parâmetros, registrar um protocolo de desenvolvimento temporal:
usar apenas treino e validação para escolha e reservar dados prospectivos ainda
não consultados para uma futura avaliação final. O teste atual pode ser usado
como diagnóstico exploratório declarado; não volta a ser intocado após nova
divisão. Se não houver dados novos, toda avaliação derivada deve explicitar
essa limitação, sem alegar confirmação externa.

Para comparar ajustes, manter origens e suporte iguais, variar um fator por vez
e usar as mesmas sementes para as alternativas. Uma sequência inicial de sementes
42, 43 e 44 permite medir sensibilidade à inicialização. A validação em blocos
cronológicos sucessivos deve construir janelas dentro de cada partição e calcular
a normalização no treino de cada bloco. Janelas individuais sobrepostas não
devem ser tratadas como observações independentes para estimar incerteza.

Uma mudança só deve substituir a versão de referência quando a validação mostrar:

- ganho no MAE macro da taxa frente à versão de referência e ao melhor baseline;
- melhora na maioria das pessoas e resultado consistente entre sementes/blocos;
- ausência de piora no MAE e no acerto direcional das mudanças rápidas, com
  tamanho do subconjunto e sinais opostos publicados;
- desempenho por participante e cobertura publicados, inclusive falhas e descartes;
- custo de treino e inferência compatível com o uso pretendido.

Os limiares de aceitação, os fatores testados e o orçamento de tentativas devem
ser fixados antes da comparação correspondente. A rodada `full_v1` é uma
referência reproduzível para esse processo, sem autorização para reajustar o
modelo até obter melhora no teste já visto.
