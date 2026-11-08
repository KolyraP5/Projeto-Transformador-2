> Registro da etapa inicial: as regras descritas aqui pertencem ao protocolo v0. Para o algoritmo atual, consulte [o protocolo da seta](../PROTOCOLO_ATUAL.md).

# Análise da estrutura do dataset AZT1D

## Objetivo e escopo

Este documento reúne a análise estrutural dos dados do AZT1D disponíveis neste repositório. O foco é descrever como os registros estão organizados, quais variáveis estão presentes, sua disponibilidade e as principais irregularidades temporais que devem ser consideradas antes de pesquisar e escolher métodos de previsão glicêmica.

Esta análise **não compara algoritmos nem define uma metodologia final**. Ela separa:

- **Fatos observados:** contagens, nomes de campos, intervalos e resultados dos relatórios de auditoria.
- **Decisões provisórias:** regras de alinhamento e preparação implementadas no pipeline atual.
- **Questões em aberto:** significados e convenções que precisam ser confirmados antes de interpretar os registros como variáveis clínicas.

## 1. Organização e dimensão

A pasta `AZT1D 2025/CGM Records/` contém 25 participantes anonimizados, cada um em uma pasta própria (`Subject 1` a `Subject 25`) com um CSV principal.

| Característica observada | Resultado |
|---|---:|
| Participantes | 25 |
| Linhas brutas somadas | 306.712 |
| Linhas por participante | 8.139–15.825 |
| Período coberto por participante | aproximadamente 28–49 dias |
| Intervalo global das datas | 08/12/2023 a 07/04/2024 |
| Timestamps inválidos identificados na auditoria | 0 |
| Linhas completamente duplicadas identificadas | 5.189 |

O README do dataset informa que os horários são locais do Arizona. Os timestamps nos arquivos não trazem um offset explícito de fuso horário. Os participantes cobrem períodos diferentes; as 25 séries não estão sincronizadas em um mesmo calendário.

Os dados visuais e relatórios existentes em `AZT1D 2025/Visual Statistics/` são materiais de visualização. Os CSVs principais usados nesta análise estão em `CGM Records`.

## 2. Esquema dos CSVs

O esquema canônico contém nove campos. O relatório de auditoria indica que todos os arquivos apresentam as colunas esperadas após a padronização. A exceção de nomenclatura é o **Subject 14**, cujo campo de glicose se chama `Readings (CGM / BGM)` e é tratado pelo pipeline como equivalente a `CGM`.

| Campo | Tipo aparente | Observações |
|---|---|---|
| `EventDateTime` | Data e hora | Eixo temporal. Há timestamps com segundos, não apenas instantes alinhados exatamente a cada cinco minutos. |
| `DeviceMode` | Categoria/texto | Há registros como `sleep` e `exercise`, além de `0` e `sleepsleep`. As categorias precisam ser entendidas e normalizadas antes de uso. |
| `BolusType` | Categoria/texto | Inclui `Standard`, `Standard/Correction`, `Automatic Bolus/Correction` e `BLE Standard Bolus`, além de diversas categorias `Extended...` menos frequentes e registros `0`. |
| `Basal` | Numérico | Preenchido em 200.889 linhas. Os valores observados vão de 0 a 4.127; a auditoria identificou 6.458 valores acima de 10. A escala e a unidade não estão confirmadas. |
| `CorrectionDelivered` | Numérico | 18.934 células numéricas: 6.451 positivas e 12.483 iguais a zero. Máximo observado: 25. |
| `TotalBolusInsulinDelivered` | Numérico | 18.934 células numéricas: 11.118 positivas e 7.816 iguais a zero. Máximo observado: 25. |
| `FoodDelivered` | Numérico | 18.934 células numéricas; 4.400 positivas. O significado exato e a relação com `CarbSize` precisam ser confirmados. |
| `CarbSize` | Numérico | 18.934 células numéricas: 4.471 positivas e 14.463 iguais a zero. Máximo observado: 120. O Subject 23 não tem valor positivo registrado. |
| `CGM` | Numérico | Campo de glicose usado como histórico e alvo nas análises existentes. Após normalizar o alias do Subject 14, está preenchido em todas as linhas brutas. Valores observados: 40–400. |

As contagens dos campos numéricos são de **células numéricas preenchidas**, não necessariamente de eventos clínicos distintos. Um zero explícito e uma célula vazia são estados diferentes no arquivo. Não se deve interpretar automaticamente nenhum dos dois como “nenhum evento” sem validar a convenção de cada campo e participante.

### Ressalva sobre o Subject 14

O nome `Readings (CGM / BGM)` sugere que o campo pode conter leituras CGM e BGM. O pipeline renomeia a coluna para `CGM` para padronizar o esquema, mas isso **não demonstra que todas as observações sejam do mesmo tipo ou diretamente comparáveis às dos demais participantes**. Nos arquivos consultados, não foi identificado um campo que indique explicitamente a modalidade de cada leitura.

Também não foi identificado um dicionário de dados que confirme unidades e definições clínicas detalhadas de todos os campos. Os nomes sugerem a finalidade de várias variáveis, mas não bastam para confirmar, por exemplo, unidades de insulina, unidade de carboidratos ou a relação entre `FoodDelivered` e `CarbSize`.

## 3. Glicose regular e eventos esparsos

A auditoria encontra uma leitura de glicose em cada linha bruta após padronizar o nome do campo do Subject 14. Na prática, as linhas formam sobretudo uma série de leituras de glicose com possíveis registros adicionais de eventos.

A cadência mediana observada é de cinco minutos, mas os campos de insulina e alimentação são muito menos densos: cada um de `CorrectionDelivered`, `TotalBolusInsulinDelivered`, `FoodDelivered` e `CarbSize` tem 18.934 células numéricas no conjunto de 306.712 linhas.

A inspeção dos registros brutos mostra que um evento pode compartilhar timestamp com uma leitura de glicose. Por exemplo, no CSV do Subject 14 há bolus e carboidratos registrados em algumas linhas, enquanto muitas outras têm esses campos vazios. Há também diferenças de preenchimento entre participantes: alguns arquivos deixam campos de evento vazios fora de registros pertinentes; outros incluem mais zeros explícitos.

Consequências descritivas:

- Glicose, insulina e carboidratos **não têm a mesma densidade observacional**, mesmo estando no mesmo CSV.
- Ausência de valor e zero explícito são estados diferentes no arquivo; a semântica de cada um precisa ser confirmada.
- Um bolus ou carboidrato registrado em um instante não descreve, por si só, como representar essa variável nos instantes seguintes.
- `CorrectionDelivered`, `TotalBolusInsulinDelivered` e `BolusType` podem ter semânticas relacionadas. Não se deve presumir que sejam canais independentes ou somáveis sem documentação.
- `Basal` é mais frequente, mas sua escala observada é muito heterogênea e precisa ser entendida antes de compará-la entre participantes.

## 4. Qualidade e irregularidades temporais

Os relatórios `results/data_audit/subject_audit.csv` e `results/data_audit/candidate_windows_cgm.csv` registram:

- **5.789 timestamps duplicados.** Isso é diferente de linhas completamente idênticas: registros distintos podem compartilhar o mesmo timestamp.
- **778 bins de cinco minutos com conflito de glicose**, isto é, mais de um valor distinto de glicose no mesmo bin.
- **302 lacunas acima de 15 minutos.**
- **254 leituras exatamente nos limites 40 ou 400**, marcadas pela auditoria como possíveis limites ou censura a investigar, não como valores a excluir automaticamente.
- Uma fração muito variável de intervalos exatamente iguais a cinco minutos entre participantes: aproximadamente **52,9%–99,7%** na auditoria temporal.

Portanto, “glicose presente em todas as linhas brutas” não equivale a “grade temporal regular sem lacunas ou conflitos”. Regularizar o tempo requer resolver duplicatas e leituras conflitantes; não se deve transformar silenciosamente um bin ambíguo em uma leitura válida.

## 5. Pré-processamento atual: decisões provisórias

A rotina em `src/preprocessing/azt1d.py` documenta e implementa a seguinte política provisória:

1. Normalizar a coluna de glicose do Subject 14 pelo nome.
2. Remover somente linhas completamente idênticas; não deduplicar apenas pelo timestamp.
3. Agrupar os registros em bins de cinco minutos.
4. Aceitar o valor de glicose do bin apenas quando não houver mais de um valor distinto; bins conflitantes tornam-se inválidos.
5. Somar bolus total e carboidratos dentro do bin, depois da remoção de duplicatas exatas.
6. Preencher com zero os bins sem valor de bolus ou carboidrato.
7. Não interpolar glicose; bins sem leitura válida, conflitos e lacunas interrompem os segmentos usados na criação de janelas.
8. Manter `Basal` no dado bruto, mas excluí-lo das features iniciais por suspeita de escala mista.
9. Excluir inicialmente `DeviceMode` e `BolusType` das features por heterogeneidade e falta de regra definida.

Essas decisões tornam os dados processáveis, mas **não demonstram que essa seja a interpretação correta dos eventos**. Em particular, preencher com zero um bin sem valor de bolus ou carboidrato adota a hipótese de “nenhum evento registrado nesse bin”; essa hipótese ainda precisa ser validada com documentação e pela convenção de cada participante.

O agrupamento usa `floor` para atribuir os timestamps a bins de cinco minutos. Isso é uma escolha de preparação que pode agrupar juntos eventos ocorridos em segundos diferentes dentro do mesmo intervalo.

## 6. Janelas temporais da configuração atual

O protocolo provisório considera:

- grade-alvo de **5 minutos**;
- histórico de **60 minutos**, representado por 12 pontos;
- alvo de glicose **30 minutos depois** do fim do histórico;
- exclusão de janelas que cruzem lacunas ou bins com glicose inválida.

Cada janela exige 18 bins consecutivos de glicose válida: 12 pontos de entrada mais seis passos até o alvo. O pipeline produz o alvo em `t+30 min`, não uma trajetória completa de glicose para cada passo futuro.

| Construção reportada | Janelas |
|---|---:|
| Candidatas CGM-only | 286.690 |
| Multivariadas com glicose, bolus e carboidratos | 273.373 |

O Subject 23 permanece elegível à análise CGM-only, mas é excluído da construção multivariada atual por não ter valores positivos de carboidratos registrados. Essa ausência é uma limitação do registro disponível, e não prova de que o participante não tenha consumido carboidratos.

As janelas sequenciais se sobrepõem; portanto, a contagem de janelas **não equivale ao número de observações independentes**. Os números acima também refletem as regras de binning, validade e agregação implementadas, e não validam por si mesmos a interpretação clínica dos eventos.

## 7. Questões de dados a resolver antes da pesquisa de métodos

1. **Unidades e definições:** confirmar o significado e a unidade de `Basal`, `TotalBolusInsulinDelivered`, `CorrectionDelivered`, `FoodDelivered` e `CarbSize`.
2. **Ausência, zero e falha de registro:** estabelecer o que significa cada estado de preenchimento e verificar se essa convenção muda entre participantes.
3. **Duplicatas e conflitos:** definir como tratar timestamps repetidos, múltiplos registros no mesmo instante e valores distintos de glicose no mesmo bin.
4. **Modalidade da glicose:** determinar se e como leituras BGM podem ser distinguidas das CGM no Subject 14.
5. **Heterogeneidade entre participantes:** investigar escalas, categorias e disponibilidade diferentes; `Basal` e a ausência de carboidratos positivos no Subject 23 são exemplos evidentes.
6. **Representação temporal dos eventos:** decidir como preservar o instante registrado e qual significado dar à ausência de novo evento nos bins seguintes, sem pressupor que um único ponto represente toda a duração do efeito.
7. **Campos potencialmente relacionados:** documentar como `CorrectionDelivered`, `TotalBolusInsulinDelivered`, `BolusType` e `FoodDelivered` se relacionam com os demais campos.

## Conclusão

O AZT1D disponível neste projeto consiste em séries temporais de glicose com registros associados de eventos, mas insulina e carboidratos **não aparecem como canais densos e uniformes prontos para serem tratados como sinais regulares**. As maiores incertezas para a preparação dos dados estão na semântica de valores ausentes e zeros, na agregação temporal, nas duplicatas e conflitos e na consistência entre participantes.

Para a etapa seguinte, esta descrição permite pesquisar algoritmos e metodologias a partir de um quadro concreto do dataset, mantendo separadas as propriedades medidas dos CSVs e as decisões de codificação que ainda precisam ser justificadas.

## Referências internas

- [README do dataset AZT1D](../../AZT1D%202025/README.txt)
- [Notebook de auditoria do AZT1D](../../notebooks/arquivo/01_auditoria_azt1d_e_protocolo_v0.ipynb)
- [Notebook de pré-processamento e baseline](../../notebooks/arquivo/02_preprocessamento_e_baseline_persistencia.ipynb)
- [Implementação do pré-processamento](../../src/preprocessing/azt1d.py)
- [Auditoria por participante](../../results/data_audit/subject_audit.csv)
- [Auditoria do esquema](../../results/data_audit/schema_audit.csv)
- [Janelas candidatas CGM-only](../../results/data_audit/candidate_windows_cgm.csv)
- [Verificações de janelas multivariadas](../../results/preprocessing_baseline/multivariate_window_checks.csv)
- [Política provisória de features](../../results/preprocessing_baseline/feature_policy_v0.csv)
