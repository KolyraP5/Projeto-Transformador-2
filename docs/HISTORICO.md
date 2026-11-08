# Evolução da pesquisa

Este registro permite acompanhar o que foi proposto, o que foi implementado e por que o foco mudou. A ordem abaixo representa a evolução documentada; datas de execução só são atribuídas quando há registro verificável.

## Percurso documentado

| Etapa | Pergunta ou objetivo | Evidência preservada | Situação e aprendizado |
|---|---|---|---|
| Proposta inicial | Prever glicemia a +30 min e melhorar a previsão por paciente com transfer learning e fine-tuning. | [Planejamento original](planejamento-inicial/planning.tex), [README original](planejamento-inicial/README_original.md), bibliografia e figuras na mesma pasta. | Proposta histórica. O texto descreve trabalho planejado; não comprova que todas as etapas tenham sido executadas. |
| Auditoria do AZT1D | Verificar esquema, qualidade temporal e disponibilidade de janelas para histórico de 12 leituras e alvo a +30 min. | [Notebook 01](../notebooks/arquivo/01_auditoria_azt1d_e_protocolo_v0.ipynb), [análise dos dados](analises/ANALISE_DATASET_AZT1D.md) e `results/data_audit/`. | Implementação de auditoria. O notebook preservado não contém saídas de execução. A análise registra duplicatas, conflitos, lacunas e incertezas na semântica de eventos. |
| Pré-processamento e persistência v0 | Criar janelas e estabelecer uma referência antes de treinar modelos. | [Notebook 02](../notebooks/arquivo/02_preprocessamento_e_baseline_persistencia.ipynb), [código de preparação](../src/preprocessing/azt1d.py) e `results/preprocessing_baseline/`. | Notebook com saídas de execução. Há baseline de persistência e janelas multivariadas candidatas; este notebook não treina uma LSTM nem executa fine-tuning. |
| Outras tentativas e reformulação | Segundo relato da equipe, resultados insatisfatórios de outras ideias levaram à proposta de seta de tendência. | Relato registrado em 07/11/2026. Notebooks, configurações e métricas dessas tentativas ainda devem ser incorporados ao acervo. | Decisão de percurso informada pela equipe. Evitar reconstruir experimentos ou resultados que ainda não estão documentados no repositório. |
| Seta de tendência: verificação rápida | Estimar a taxa média futura do CGM em +5 e +15 min e representá-la por uma seta contínua. | [Cópia preservada do notebook executado em modo rápido](../notebooks/arquivo/03_seta_quick_v1_executado.ipynb), [notebook atual](../notebooks/atual/AZT1D_Setas_e_Tendencia.ipynb) e `results/azt1d_seta/runs/quick_v1/`. | Fluxo com baselines, duas LSTMs, seleção por validação e avaliação posterior no teste. `quick_v1` é uma verificação reduzida de execução. A cópia arquivada conserva as saídas recebidas antes da reorganização. |
| Seta de tendência: rodada completa `full_v1` | Executar o mesmo protocolo com `QUICK=False` e toda a população elegível, preservando a execução rápida. | [Execução completa](../results/azt1d_seta/runs/full_v1/execucao.json), [notebook executado](../results/azt1d_seta/runs/full_v1/notebook_executado.ipynb) e [análise](analises/ANALISE_FULL_V1.md). | Concluída localmente em 07/11/2026 por automação do assistente a pedido da equipe. +15 min escolhido na validação; redução de 18,02% no MAE macro de teste frente à persistência, ganho em 24/24 pessoas e 37 verificações aprovadas. Avaliação exploratória; plano seguinte focado no aprimoramento da seta. |
| Direção adotada em 07/11/2026 | Concentrar os próximos experimentos no aperfeiçoamento da seta de tendência. | Instrução da equipe e planejamento atual no repositório. | A escolha do tema está consolidada. As tentativas anteriores permanecem como histórico; novos experimentos devem melhorar o algoritmo final e medir o efeito das mudanças. |

Os commits de 14/09/2026 registram a inclusão dos notebooks 01 e 02. A data do commit indica quando o arquivo entrou no histórico Git, não necessariamente quando cada análise foi executada.

## O que os resultados antigos permitem afirmar

O notebook 02 preserva uma execução com 25 participantes e divisão por pessoa: 18 de treino, 2 de validação e 5 de teste externo. A persistência a +30 min foi avaliada em 58.326 janelas do teste. As médias por pessoa registradas são MAE de **17,737 mg/dL** e RMSE de **24,062 mg/dL**. Esses valores são do protocolo v0 e não devem ser apresentados como métricas da seta.

A preparação multivariada era candidata: bolus e carboidratos tinham regras provisórias de agregação e valores ausentes. A análise dos dados explica por que essa representação exigia cautela. O Subject 14 era incluído mediante renomeação de `Readings (CGM / BGM)`; o fluxo atual o exclui por padrão enquanto a modalidade das leituras não estiver confirmada. A população e o tratamento dos dados mudaram.

## Como apresentar a mudança ao professor

A trajetória documentada começa com previsão personalizada de glicemia, passa pela inspeção dos dados e por uma baseline a +30 min, e chega à estimação da taxa futura para uma seta contínua. Segundo a equipe, outras tentativas também contribuíram para essa decisão. Essas tentativas devem ser adicionadas com suas próprias configurações, limitações e resultados, mesmo quando não avançaram.

A escolha da seta como direção final foi informada pela equipe. A alegação de superioridade quantitativa sobre todas as tentativas anteriores exige os respectivos artefatos e uma comparação compatível. Em particular, o protocolo antigo avaliava glicemia a +30 min em pessoas separadas do treino; o atual avalia taxa a +5/+15 min em partições temporais das mesmas pessoas. Comparar diretamente os números desses protocolos confundiria mudança de tarefa, horizonte e população com melhoria do modelo.

Não há necessidade de retomar as linhas descartadas para prosseguir. Documentá-las serve para tornar a evolução compreensível. A avaliação do algoritmo final deve usar suas próprias referências, critérios de seleção e limitações.

## Como acrescentar material ao acervo

1. Coloque o notebook da tentativa em `notebooks/arquivo/`, com nome que descreva a pergunta ou abordagem. Use uma data no nome somente se for conhecida.
2. Crie um registro a partir do [modelo de tentativa](templates/REGISTRO_TENTATIVA.md), salve-o em `docs/analises/` e acrescente uma linha à tabela deste documento.
3. Preserve resultados disponíveis em uma pasta própria de `results/`, incluindo a configuração e a versão do código usadas. Registre explicitamente quando pesos, logs ou dados não estiverem mais disponíveis.
4. Diferencie observação, interpretação e decisão. Informe quem realizou a execução quando isso for conhecido; não atribua automaticamente todas as etapas a um integrante.
5. Mantenha experimentos concluídos sem substituir seus resultados por novas execuções. Alterações de modelo, preparação ou avaliação devem receber outro identificador.

## Lacunas a preencher pela equipe

- Identificação, ordem e autoria das demais tentativas mencionadas; datas apenas quando recuperáveis.
- Notebooks e saídas que sustentam a avaliação de que as alternativas tiveram resultados insatisfatórios.
- Configurações, divisões dos dados e métricas dessas alternativas, para saber quais comparações são possíveis.
- Confirmação da origem e do estado dos relatórios históricos citados pela análise do dataset.

Essas lacunas limitam a reconstrução retrospectiva; não impedem a execução e o aperfeiçoamento do fluxo atual.
