# Aprimoramento da seta de tendência de +15 minutos

Este documento preserva o plano de desenvolvimento após `full_v1`, elaborado
com assistência de IA. **As propostas passaram à execução na bateria v2**, cujo
[relatório](analises/ANALISE_APRIMORAMENTO_V2.md) informa o que foi efetivamente
testado e o resultado, e cujo [guia](EXECUCAO_APRIMORAMENTO.md) descreve a execução
por etapas. O orçamento abaixo registra a proposta inicial; a execução utiliza
triagem das oito configurações e confirmação da finalista, com protocolo próprio
congelado antes dos treinos. O trabalho segue com a previsão da taxa média
futura do CGM e sua seta contínua de +15 minutos. As linhas anteriores ficam
preservadas no histórico; não fazem parte deste plano.

## Referência e prioridades

O horizonte de +15 minutos foi escolhido na validação antes da avaliação de
teste. O MAE macro da taxa foi 0,542927 contra 0,668332 da persistência na
validação, ganho de 18,76%. No teste, foi 0,544375 contra 0,664030, ganho de
18,02%, com melhora sobre persistência em 24/24 participantes. Nas mudanças
rápidas do teste, o MAE foi 1,142430 contra 1,196431 da última variação; o
acerto de sinal foi 82,3% e os sinais opostos foram 17,7%.

Essas médias favoráveis não encerram o desenvolvimento. A análise exploratória
por intensidade identificou dois problemas a investigar:

| Situação no teste | Evidência observada | Objetivo do próximo desenvolvimento |
|---|---|---|
| `\|taxa real\| < 0,5` | 29.115 janelas, 52,9% do teste; MAE LSTM 0,296704 contra 0,227262 da persistência | Reduzir movimento previsto desnecessário em períodos estáveis |
| `\|taxa real\| >= 2` | 2.834 janelas, 5,15% do teste; MAE LSTM 1,946065 contra 1,765758 da última variação | Melhorar a magnitude prevista nas mudanças extremas, preservando a direção |

As métricas desta tabela são médias por pessoa (macro), calculadas separadamente
em cada faixa, com suporte nos 24 participantes; não são o MAE macro geral.
Os valores do teste geram hipóteses de desenvolvimento, **não metas a perseguir
repetindo ajustes no mesmo teste**. Os cortes de intensidade devem ser fixados
antes das novas comparações e aplicados também à validação.

O modelo de +15 minutos terminou as 15 épocas com sua melhor perda de validação
na última época. Isso justifica testar um orçamento maior; não demonstra que
mais épocas vão melhorar a generalização.

Fontes: [análise completa](analises/ANALISE_FULL_V1.md),
[seleção congelada](../results/azt1d_seta/runs/full_v1/selecao_horizonte.json),
[métricas](../results/azt1d_seta/runs/full_v1/final/metricas.csv) e
[histórico do treino de +15 minutos](../results/azt1d_seta/runs/full_v1/lstm_h15/historico.csv).

## 1. Fixar a avaliação antes de alterar o modelo

Prazo sugerido: primeira sessão de desenvolvimento, antes dos novos treinos.
Preservar `full_v1` e registrar este plano, seus critérios, os índices temporais
e o orçamento em um arquivo de configuração versionado.

Usar somente os primeiros 80% da grade temporal de cada participante para
desenvolvimento. A proposta inicial de validação em três blocos é:

| Bloco | Treino | Validação para escolher época | Bloco para comparar configurações |
|---|---|---|---|
| B1 | 0–40% | 40–50% | 50–60% |
| B2 | 0–50% | 50–60% | 60–70% |
| B3 | 0–60% | 60–70% | 70–80% |

Os percentuais se referem à grade completa original de cada pessoa, não à
concatenação de pessoas nem à contagem de janelas após descarte. A informação
sempre caminha do passado para o futuro. O período comparado em B1 integra a
validação de época de B2 e o treino de B3, como em uma avaliação temporal com
origem expansiva; isso não torna as medidas dos blocos independentes.

Construir janelas separadamente dentro de cada intervalo. Recalcular todas as
escalas somente no treino do bloco, sem usar as escalas salvas de `full_v1`.
Treinar a referência v1 e cada candidata **do zero em cada bloco**, sem reaproveitar
pesos de `full_v1`, de outro bloco ou de outra semente. Escolher checkpoints
apenas na validação para época; o terceiro intervalo compara configurações.

Usar as sementes **42, 43 e 44**, iguais para referência e candidata. Comparar
as mesmas origens; se uma feature reduzir cobertura, publicar a perda de
cobertura e apresentar a comparação pareada no suporte comum. Não descartar
pessoas porque a candidata foi pior nelas.

O teste atual de 80–100% já foi consultado, inclusive nesta análise. Não será
usado para selecionar features, perda, época ou limiar. Os novos blocos também
não constituem uma confirmação externa intocada: a equipe já examinou resultados
da base. Sua função é disciplinar o desenvolvimento daqui em diante.

## 2. Reproduzir a referência nos blocos

Execuções: `dev_v1_b<B>_s<S>`, total de nove treinos, apenas +15 minutos.
Manter a preparação v1, 12 entradas CGM, LSTM de 32 unidades, Adam `0.001`,
lote 64, MSE, máximo de 15 épocas e paciência 3.

Publicar resultados por bloco, semente, participante e intensidade antes de
avaliar candidatas. Essa referência refeita é o comparador correto das novas
rodadas; os números agregados originais de `full_v1` não substituem a referência
de cada bloco.

## 3. Testar ablações de um fator por vez

Executar nesta ordem. Cada linha usa os mesmos três blocos e três sementes;
registrar resultados sem esconder uma tentativa que falhou.

| Família de execução | Alteração controlada | Hipótese e ponto de atenção |
|---|---|---|
| `dev_ep30_b<B>_s<S>` | Aumentar somente o máximo para 30 épocas; paciência 3 | A queda da perda até a época 15 pode justificar mais treino; verificar sobreajuste e custo |
| `dev_pat5_b<B>_s<S>` | Sobre `ep30`, aumentar somente a paciência para 5 | Executar apenas se `ep30` justificar investigar a parada; comparar também a `ep30`, sem atribuir o ganho à arquitetura |
| `dev_delta_b<B>_s<S>` | Adicionar diferenças consecutivas de CGM ao histórico absoluto da v1 | Tornar a dinâmica recente explícita; diferenças usam somente leituras disponíveis até a origem |
| `dev_relativo_b<B>_s<S>` | Adicionar canal `G(t-k)-G(t)` ao histórico absoluto da v1 | Separar forma recente e nível absoluto; conservar o CGM absoluto e normalizar cada canal apenas no treino |
| `dev_peso15_b<B>_s<S>` | Na v1, peso de perda 1,5 para alvos de treino com `\|taxa real\| >= 1`, peso 1 nos demais | Dar atenção moderada às mudanças rápidas; pode piorar estabilidade e precisa passar nos dois diagnósticos |

Para a perda ponderada, normalizar a soma das perdas pela soma dos pesos.
A taxa futura real pode definir pesos de exemplos **somente no treinamento**,
pois é o alvo supervisionado. Ela não pode entrar como feature nem decidir
qual previsão emitir em produção. Qualquer regra para reconhecer estabilidade
ou extremo em execução deve depender exclusivamente do passado disponível ou
da previsão, nunca da intensidade futura verdadeira.

Não fazer busca livre por pesos, janelas ou tamanhos de rede nesta fase. O peso
1,5, os limites de épocas e as três sementes são propostas fixas para o primeiro
ciclo. Se forem alterados após resultados, registrar um segundo ciclo e sua
justificativa antes de treiná-lo.

## 4. Aplicar critérios de aceitação registrados previamente

Proposta para este primeiro ciclo, a ser congelada antes dos treinos:

- Redução de pelo menos **2% no MAE macro da taxa** frente à v1 refeita nos
  mesmos blocos/sementes, mantendo ganho sobre o melhor baseline geral.
- Melhora no MAE em mais da metade dos participantes na agregação pareada;
  ganho também em pelo menos dois dos três blocos e duas das três sementes.
- No subconjunto `|taxa real| >= 1`, não piorar MAE macro, acerto de sinal nem
  proporção de sinais opostos em relação à referência; publicar as contagens.
- Na faixa `|taxa real| < 0,5`, não piorar MAE em relação à referência e reduzir
  a proporção de previsões com `|taxa prevista| >= 0,5`. Reportar também a
  persistência e os casos de taxa real exatamente zero, sem contabilizar zero
  como acerto de direção de uma mudança não nula.
- Na faixa `|taxa real| >= 2`, não piorar MAE nem direção frente à referência.
  Para declarar esse problema resolvido, exigir também MAE menor ou igual ao
  da última variação na mesma agregação; melhora parcial deve ser chamada
  de melhora parcial.
- Conservar a cobertura e publicar todas as exclusões, custo de treino e
  latência de inferência. Uma versão que melhora apenas retirando casos
  difíceis não substitui a referência por este critério.

Calcular primeiro métricas por pessoa em cada bloco/semente. Agregar com peso
igual por pessoa e mostrar os resultados separados por bloco e semente; não
tratar nove treinos ou milhares de janelas sobrepostas como observações
independentes. Reportar o número de pessoas avaliáveis em cada faixa. Se o
suporte de extremos for insuficiente em algum bloco, identificar o resultado
como inconclusivo nessa faixa, sem completar valores artificialmente.

O ganho mínimo de 2% é um critério de desenvolvimento proposto, não uma margem
clínica nem uma meta calculada dos números do teste. Estimar incerteza com
reanálise pareada por participante e, se necessário, blocos temporais contíguos;
não fazer reamostragem independente de janelas sobrepostas.

## 5. Consolidar somente alterações que passaram

Se houver vencedoras, testar **uma combinação final previamente registrada**
das alterações aprovadas em `dev_combinado_b<B>_s<S>`, com mais nove treinos.
A combinação precisa passar pelos mesmos critérios: ganhos de ablações
isoladas não garantem ganho conjunto. Caso nada passe, manter a v1 e registrar
o diagnóstico que orientará o próximo ciclo.

Após escolher a configuração, fixar código, escalas, horizonte, sementes,
treinamento e interpretação da seta. Uma nova execução consolidada pode usar
os primeiros 80% para desenvolvimento, com divisão interna explícita, mas
qualquer nova medição no teste atual continuará sendo diagnóstico exploratório.
Não substituir nem renomear os artefatos de `full_v1`.

Para confirmação final, reservar **dados prospectivos ainda não consultados**,
com período, origem e critérios de inclusão registrados antes da coleta ou
abertura. Congelar o algoritmo antes de acessar esse conjunto. Se esses dados
não estiverem disponíveis, encerrar o ciclo como aprimoramento exploratório
com essa limitação declarada.

## Orçamento, entregas e custo

O primeiro ciclo admite até **63 treinos de +15 minutos**: nove da referência,
até 45 das cinco ablações e nove da combinação final, quando justificada.
A ablação de paciência é opcional e não deve ser executada apenas para consumir
o orçamento. O escopo deste ciclo se limita a essas configurações do algoritmo
da seta, com registro explícito caso um segundo ciclo seja necessário.

O treino de +15 minutos de `full_v1` levou aproximadamente 154 segundos na CPU
registrada. Esse é um referencial local, não uma previsão de duração: os blocos
têm tamanhos diferentes e candidatos de 30 épocas podem custar mais. Registrar
tempo, pico de memória e latência por janela/lote. Trabalhar por fases permite
encerrar candidatas sem benefício antes de montar a combinação.

Cada execução deve entregar configuração, identificação/hash dos dados e código,
limites dos blocos, sementes, versões do ambiente, pesos, escalas, curva de treino,
previsões e métricas por pessoa/intensidade, contagens, descartes e tempos.
Cada família deve entregar um registro curto de hipótese, mudança, resultado,
decisão e responsável pela revisão. Usar o
[modelo de registro](templates/REGISTRO_TENTATIVA.md) e atualizar o
[histórico](HISTORICO.md), inclusive quando a decisão for manter a versão anterior.

O prazo é medido por essas entregas: protocolo congelado; referência reproduzida;
ablações comparadas; uma versão consolidada ou decisão documentada de manter
v1; e confirmação futura condicionada à disponibilidade de dados novos.
