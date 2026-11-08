# Executar a bateria de aprimoramento v2

Esta bateria mantém a tarefa da seta contínua de **+15 minutos**, preservando a
v1. As etapas abaixo descrevem o procedimento, sem antecipar seus resultados.
Dependências e configuração local estão no [guia geral](EXECUCAO.md).

## Protocolo e dados

O [protocolo efetivamente executado](../results/azt1d_seta/experiments/aprimoramento_v2/protocolo_execucao.json)
define a triagem, a confirmação e os critérios antes do primeiro treino.
`configs/aprimoramento_v2.json` preserva o rascunho inicial da grade; não deve ser
confundido com a decisão posterior de executar a bateria por etapas.

Somente os primeiros 80% da grade temporal de cada pessoa participam do
desenvolvimento. São 24 participantes, mantendo a exclusão da coluna ambígua do
Subject 14 e a preparação da v1, sem interpolação.

| Bloco | Treino | Escolha de época (`stop`) | Comparação de configurações (`evaluation`) |
|---|---|---|---|
| B1 | 0–40% | 40–50% | 50–60% |
| B2 | 0–50% | 50–60% | 60–70% |
| B3 | 0–60% | 60–70% | 70–80% |

As janelas são criadas dentro de cada segmento. As escalas usam apenas o treino;
o checkpoint usa MSE sem pesos no `stop`. Os blocos de comparação escolhem a
configuração. O teste atual de 80–100% permanece fora dessas escolhas.

## Oito configurações pré-registradas

| Identificador | Entrada | Épocas máximas | Paciência | Peso dos alvos rápidos no treino |
|---|---|---:|---:|---:|
| `baseline15` | CGM absoluto | 15 | 3 | 1 |
| `ep30` | CGM absoluto | 30 | 3 | 1 |
| `pat5` | CGM absoluto | 30 | 5 | 1 |
| `delta` | CGM + diferenças consecutivas | 15 | 3 | 1 |
| `relative` | CGM + histórico relativo à leitura da origem | 15 | 3 | 1 |
| `weighted` | CGM absoluto | 15 | 3 | 1,5 |
| `combined` | CGM + diferenças + histórico relativo | 30 | 5 | 1 |
| `combined_weighted` | Os três canais | 30 | 5 | 1,5 |

As diferenças e a centralização usam somente a entrada conhecida na origem.
O peso 1,5 aplica-se a exemplos de treino com taxa real absoluta ≥1; a perda é
dividida pela soma dos pesos. O alvo futuro não vira feature nem regra de
escolha da previsão na inferência.

O executor usa **dois processos, com duas threads por processo**, lote 64, LSTM
de 32 unidades e Adam a 0,001. `baseline15`, `ep30` e `pat5` compartilham uma
trajetória de otimização, mantendo checkpoints, históricos e regras de parada
próprios. Isso equivale aos respectivos prefixos de treinos separados, pois não
há dropout nem scheduler dependente do limite de épocas. Não some os tempos
dessas três saídas como se fossem três treinos físicos independentes.

A referência é refeita nos mesmos blocos das candidatas. Normalização em
`float64`, duas threads e o `stop` separado diferem do `full_v1`; não se exige
igualdade bit a bit com seus pesos ou métricas antigos.

## Comandos por etapa

Os exemplos são para uma **nova reprodução**, a partir da raiz do repositório.
Use um Python com as dependências. Nesta máquina, o interpretador utilizado é:

```powershell
$pythonAzt = 'C:/Users/gusta/AppData/Local/Python/pythoncore-3.14-64/python.exe'
$experimentoAzt = 'aprimoramento_v2_reproducao'
& $pythonAzt scripts/preparar_aprimoramento.py --experiment-name $experimentoAzt
Copy-Item -LiteralPath 'results/azt1d_seta/experiments/aprimoramento_v2/protocolo_execucao.json' -Destination "results/azt1d_seta/experiments/$experimentoAzt/protocolo_execucao.json"
```

O preparo gera os três blocos e verifica a identidade dos caches ao reutilizá-los.
O protocolo copiado deve permanecer congelado antes dos treinos.

**Triagem:** execute as oito configurações em B3, semente 42, e gere a comparação.

```powershell
& $pythonAzt scripts/executar_aprimoramento.py --experiment-name $experimentoAzt --stage triagem --variants baseline15 ep30 pat5 delta relative weighted combined combined_weighted --blocks B3 --seeds 42 --workers 2 --threads 2
& $pythonAzt scripts/resumir_aprimoramento.py --experiment-name $experimentoAzt --report-name triagem --variants baseline15 ep30 pat5 delta relative weighted combined combined_weighted --blocks B3 --seeds 42
```

**Confirmação:** substitua o marcador abaixo pela candidata definida pela
comparação de desenvolvimento. Uma candidata promissora que ainda falhe nas
salvaguardas pode ser investigada, mas não recebe aprovação por esse motivo.

```powershell
$candidataAzt = 'SUBSTITUA_PELO_NOME_DEFINIDO_NA_TRIAGEM'
& $pythonAzt scripts/executar_aprimoramento.py --experiment-name $experimentoAzt --stage confirmacao --variants baseline15 $candidataAzt --blocks B1 B2 B3 --seeds 42 43 44 --workers 2 --threads 2
& $pythonAzt scripts/resumir_aprimoramento.py --experiment-name $experimentoAzt --report-name confirmacao --variants baseline15 $candidataAzt --blocks B1 B2 B3 --seeds 42 43 44
& $pythonAzt scripts/relatar_aprimoramento.py --experiment-name $experimentoAzt
& $pythonAzt scripts/validar_aprimoramento.py --experiment-name $experimentoAzt --variants baseline15 $candidataAzt --blocks B1 B2 B3 --seeds 42 43 44 --report-name confirmacao
```

Modelos concluídos compatíveis são reutilizados. Etapas registradas não são
substituídas: cada chamada do executor precisa de um `--stage` novo. A confirmação
exige ganho geral mínimo de 2%, melhora na maioria das pessoas, consistência em
dois blocos e duas sementes, preservação de cobertura e as salvaguardas de
estabilidade, mudanças rápidas e extremos. Todas precisam passar. A agregação
primeiro combina blocos/sementes por pessoa e depois dá peso igual às pessoas.

## Congelar antes da avaliação final

Registre `plano_final_congelado.json` na raiz da pasta do experimento com
`candidate` e `approved`, coerentes com a confirmação; `parameters`, quando
presente, precisa coincidir com a variante do protocolo. Não use resultados do
teste para preencher ou alterar essa decisão.

Confirme antes que os três caches de `full_v1/prepared/` estão disponíveis no
disco. Em uma máquina sem esses caches, recrie uma rodada-fonte com nome novo
pelo executor v1 e indique-a em `--source-run`:

```powershell
# Somente se for necessário reconstruir os caches em outra rodada:
& $pythonAzt scripts/executar_experimento.py --run-name full_v1_reproducao --mode full
```

```powershell
$fonteDadosAzt = 'full_v1' # Use 'full_v1_reproducao' se os caches vierem dela.
& $pythonAzt scripts/executar_final_v2.py --experiment-name $experimentoAzt --candidate $candidataAzt --run-name full_v2_reproducao --source-run $fonteDadosAzt
```

O executor final usa a divisão original 60/20/20, semente 42, e registra
`selecao_final.json` **antes de carregar os arrays do teste**. `--freeze` permite
indicar outro plano congelado; `--screening-summary` indica outro resumo da
triagem. `--approved` apenas exige uma aprovação já registrada: não a concede.
Ela depende de confirmação final, candidata selecionada coincidente e decisão
explícita `accepted=true`. Uma candidata sem aprovação permanece experimental;
se nenhuma passar, a versão de referência continua sendo a adotada.

Os caches não são incluídos no Git; o manifesto final registra a execução-fonte
e os hashes utilizados. A avaliação no teste continua exploratória por seu
acesso anterior; nenhum teste final desta base volta a ser intocado.

## Onde conferir a execução

Sob `results/azt1d_seta/experiments/<experimento>/`:

- `manifest.json`, `fronteiras.csv` e `prepared/`: dados, cortes e hashes;
- `protocolo_execucao.json`, `codigo/` e `stages/`: protocolo efetivo, fontes
  congeladas, argumentos, horários e estado de cada etapa;
- `models/<variante>_B<bloco>_s<semente>/`: configuração, escalas, ambiente,
  pesos, checkpoint, histórico, previsões e métricas;
- `analise/triagem/` e `analise/confirmacao/`: `acceptance.json`, `summary.json`,
  `comparison.csv`, `metrics_by_person.csv`, custos e figuras;
- `analise/relatorio.md` e `painel_aprimoramento.png`: leitura gerada das tabelas;
- `verificacoes/`: auditoria dos dados, modelos e previsões de desenvolvimento.

O validador v2 acima verifica o desenvolvimento; ele não acessa previsões de
teste nem valida a rodada final. `scripts/verificar_experimento.py` continua
destinado ao formato da v1. Na rodada final v2, consulte sua seleção, fontes,
manifesto, `model/`, `final/`, ambiente e registro de execução em
`results/azt1d_seta/runs/<run-name>/`. Caches, previsões grandes e ZIPs ficam no
disco conforme o `.gitignore`; tabelas, decisões, pesos e fontes são preservados.

Para conferir a rodada final após sua conclusão:

```powershell
python scripts/verificar_final_v2.py --run-name full_v2
```
