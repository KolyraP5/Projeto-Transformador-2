# AZT1D — Evolução da pesquisa e seta de tendência

Pesquisa da disciplina **Experiência Criativa: Projeto Transformador II**, Ciência
da Computação — PUCPR. Equipe: Gustavo Faria Cardoso, Mateus da Silva Maciel de
Lima e Pedro Costa Lyra.

O foco atual é estimar a **taxa média futura da glicemia** a partir do histórico
de CGM e representá-la como uma seta contínua. A equipe escolheu essa direção
após explorar outras ideias; as próximas versões aprimoram esse mesmo algoritmo.
Este repositório preserva também a proposta inicial e as etapas anteriores para
que seja possível acompanhar a evolução da pesquisa.

## Por onde começar

| Para acompanhar… | Abra |
|---|---|
| O inventário objetivo dos artefatos para entrega | [readme.txt](readme.txt) |
| A evolução, decisões e evidências disponíveis | [Histórico da pesquisa](docs/HISTORICO.md) |
| O algoritmo atual, dados e critérios de avaliação | [Protocolo atual](docs/PROTOCOLO_ATUAL.md) |
| O fluxo executável da seta | [Notebook atual](notebooks/atual/AZT1D_Setas_e_Tendencia.ipynb) |
| A rodada completa e sua interpretação | [Análise de full_v1](docs/analises/ANALISE_FULL_V1.md) |
| Os aprimoramentos seguintes | [Plano de evolução do algoritmo](docs/PLANO_APRIMORAMENTO.md) |
| Como reproduzir localmente ou no Colab | [Execução e resultados](docs/EXECUCAO.md) |

## Organização

```text
notebooks/
  atual/                 # Fluxo de referência da seta
  arquivo/               # Etapas anteriores, preservadas com seus resultados
  exploracao/            # Novas análises e estudos auxiliares
docs/
  HISTORICO.md           # Índice das etapas e das decisões
  PROTOCOLO_ATUAL.md      # O que o experimento atual realmente mede
  analises/              # Leitura do dataset e dos resultados executados
  planejamento-inicial/  # Proposta original e materiais associados
  templates/             # Modelo para registrar outra tentativa
src/
  seta_tendencia/         # Código legível do algoritmo atual
  preprocessing/         # Pipeline da etapa histórica v0
scripts/                 # Execução e verificação reproduzíveis
results/
  azt1d_seta/runs/        # Uma pasta imutável por execução
AZT1D 2025/               # Dataset original neste checkout, sem modificações
```

## Como acrescentar uma tentativa

1. Guarde o notebook em `notebooks/exploracao/` enquanto estiver em andamento.
   Use um nome como `04_descricao_curta.ipynb`.
2. Copie o [modelo de registro](docs/templates/REGISTRO_TENTATIVA.md) para
   `docs/analises/`. Registre hipótese, configuração, responsáveis, resultados e
   decisão, inclusive quando a ideia não funcionar.
3. Ao encerrar a tentativa, mova o notebook para `notebooks/arquivo/`, preserve
   suas saídas relevantes e adicione uma linha no [histórico](docs/HISTORICO.md).
4. Use outro nome em `results/` para cada configuração. Não substitua métricas de
   uma rodada antiga por resultados de uma nova versão.

O [índice dos notebooks](notebooks/README.md) identifica o papel de cada arquivo.
Materiais ainda não recuperados devem ser indicados como pendentes, sem atribuir
resultados que não estejam documentados.

## Estado da pesquisa

`quick_v1` é a verificação pequena do fluxo. `full_v1` executa a configuração
completa, com divisão cronológica por pessoa, duas LSTMs (+5 e +15 minutos) e três
baselines por horizonte. As saídas e a análise ficam nos links acima.

A proposta original de previsão a +30 minutos com transfer learning está
preservada como histórico. Seu protocolo difere do atual; seus números não
estabelecem uma comparação direta entre as abordagens. A escolha de continuar
com a seta é uma decisão da equipe, e os ganhos do preditor atual são avaliados
contra os baselines do próprio experimento.

Fonte dos dados: [AZT1D v1, Mendeley Data](https://doi.org/10.17632/gk9m674wcx.1),
CC BY 4.0. A avaliação atual é exploratória e não constitui validação clínica.
