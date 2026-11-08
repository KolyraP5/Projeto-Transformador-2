# Execução e preservação dos resultados

## Ambiente local

Requer Python com NumPy, pandas, Matplotlib, PyTorch e Jupyter. Em um ambiente
Python funcional, instale `requirements.txt`. Exemplo PowerShell a partir da raiz:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/executar_experimento.py --run-name full_v2 --mode full
```

O comando executa **o próprio notebook**, usando o mesmo interpretador do script.
`--mode quick` limita a três participantes e até três épocas; `--mode full` usa
todas as pessoas elegíveis e até quinze épocas por horizonte, com parada antecipada.
`full_v1` é a rodada registrada nesta organização; para uma repetição, escolha um
nome novo. O script rejeita uma execução já registrada para preservar suas saídas.

O dataset é localizado em `AZT1D 2025/CGM Records`. Para outra cópia original:

```powershell
$env:AZT1D_LOCAL_DATA = 'C:\dados\AZT1D 2025\CGM Records'
```

Também é possível abrir `notebooks/atual/AZT1D_Setas_e_Tendencia.ipynb` e executar
as células manualmente. Defina `RUN_NAME` e `QUICK` antes de começar. O notebook
permanece autossuficiente para o Colab; seu `CORE_SOURCE` deve coincidir com
`src/seta_tendencia/azt1d_core.py`. Não altere a versão congelada para reproduzir v1.

Na máquina desta rodada, o Python disponível com as dependências está em
`C:/Users/gusta/AppData/Local/Python/pythoncore-3.14-64/python.exe`; os aliases
`python`/`py` do sistema não funcionaram e a `.venv` inicial estava sem os pacotes.
O ambiente efetivamente utilizado fica registrado em cada execução.

## Colab

Veja o [guia curto](../README_Colab.md). O notebook monta o Drive e recebe a pasta
ou ZIP do AZT1D. Fora do Colab, usa o disco local. A GPU é opcional; a escolha é
automática se PyTorch detectar CUDA.

Use outro nome ao trocar de ambiente. Além das diferenças numéricas de versões
e hardware, a identificação v1 dos dados incorpora separadores de caminhos do
sistema, então a mesma base em Windows e Linux pode gerar fingerprints distintos.

## O que fica em cada rodada

Em `results/azt1d_seta/runs/<RUN_NAME>/`:

| Artefato | Finalidade |
|---|---|
| `manifest.json`, `auditoria_*.csv`, `divisoes_temporais.csv` | Dados, exclusões, janelas e identificação da preparação |
| `lstm_h5/`, `lstm_h15/` | Configuração, ambiente, histórico, pesos e métricas de validação |
| `baselines/` | Referências de comparação na validação |
| `selecao_horizonte.json` | Escolha congelada usando somente validação |
| `final/` | Métricas, ganhos, previsões e figuras de teste |
| `notebook_executado.ipynb` | Células e saídas preservadas pelo executor |
| `codigo/`, `requirements-freeze.txt`, `execucao.json` | Fontes exatas, ambiente e registro da execução |
| `execucao.log`, `analise/` | Saídas de texto e diagnósticos adicionais por pessoa/intensidade |
| `verificacao.json` | Verificações independentes das partições e artefatos |

O `.gitignore` exclui caches `prepared/`, previsões compactadas e ZIPs grandes.
Eles permanecem no disco e são regeneráveis. Métricas, pesos, figuras, notebooks
executados e relatórios ficam disponíveis para revisão. O dataset original já
presente neste checkout é mantido sem reorganizar seus arquivos internos.

O ZIP gerado pelo notebook contém métricas, previsões, figuras, código e pesos.
O executor salva o notebook executado e seu registro **após** essa exportação.
O ZIP local de `full_v1` foi reexportado ao final para incluir também esses
registros e os diagnósticos de `analise/`. Os documentos de discussão e o
histórico ficam no repositório, nos links do README principal.

Após refazer uma análise, use `scripts/empacotar_resultados.py --run-name full_v1`
para atualizar o ZIP completo. Esse empacotador evita entradas duplicadas do
código arquivado e verifica a integridade do arquivo, preservando o core v1.

## Verificar a rodada

```powershell
.\.venv\Scripts\python.exe scripts/verificar_experimento.py --run-name full_v1
.\.venv\Scripts\python.exe scripts/analisar_resultados.py --run-name full_v1
```

As verificações conferem integridade, separação temporal, contagens, normalização,
vínculo entre teste, seleção e pesos congelados, além da conclusão dos treinos e
do teste. Não recalculam a regra de candidatura do horizonte nem substituem a
interpretação dos resultados. Treino interrompido reinicia; a versão v1 não
retoma o estado do otimizador. Uma execução incompleta deve ser investigada e
preservada antes de iniciar uma rodada com outro nome.

O acesso ao teste em `quick_v1` é registrado como acesso prévio no `full_v1`.
Novos ajustes devem ser escolhidos pela validação, seguindo o
[plano de aprimoramento](PLANO_APRIMORAMENTO.md).
