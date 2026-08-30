# Previsão Personalizada de Glicemia em Diabetes Tipo 1

Projeto desenvolvido para a disciplina *Experiência Criativa: Projeto Transformador II*, do curso de Ciência da Computação da PUCPR.

O objetivo do projeto é investigar se um modelo populacional de previsão glicêmica pode apresentar melhor desempenho para indivíduos não observados após a aplicação de *Transfer Learning* e *Fine-Tuning* específicos por paciente.

## Visão Geral

A proposta utiliza dados de pacientes com Diabetes Mellitus Tipo 1 para prever valores futuros de glicemia.

O modelo principal será baseado em uma rede *Long Short-Term Memory (LSTM)*, inicialmente treinada com dados de múltiplos indivíduos para aprender padrões gerais de comportamento glicêmico.

Posteriormente, o modelo será adaptado para pacientes específicos por meio de Transfer Learning e Fine-Tuning.

Fluxo geral:

text
AZT1D
  ↓
Pré-processamento
  ↓
CGM + Insulina + Carboidratos
  ↓
LSTM populacional
  ↓
Transfer Learning
  ↓
Fine-Tuning por paciente
  ↓
Modelo personalizado
  ↓
Comparação de desempenho


## Dataset

Será utilizado o dataset *AZT1D*, composto por dados reais de 25 indivíduos com Diabetes Mellitus Tipo 1.

Principais dados utilizados:

* Continuous Glucose Monitoring (CGM);
* administração de insulina;
* ingestão de carboidratos;
* modos do dispositivo.

Dataset:

https://doi.org/10.17632/gk9m674wcx.1

Os arquivos originais do AZT1D não serão armazenados diretamente neste repositório.

## Configuração Inicial do Experimento

A configuração inicial prevista é:

* aproximadamente *60 minutos de histórico* como entrada;
* previsão da glicemia *30 minutos à frente*;
* entradas principais:

  * CGM;
  * insulina;
  * carboidratos.

As métricas principais de avaliação serão:

* RMSE;
* MAE.

Como análise adicional, poderá ser utilizada a Clarke Error Grid.

## Estratégia Experimental

Os participantes serão separados em nível de indivíduo para evitar vazamento de dados.

O experimento principal será dividido em duas condições:

1. *Modelo populacional*

   * treinado utilizando dados de múltiplos participantes;
   * avaliado em participantes não utilizados durante o treinamento.

2. *Modelo personalizado*

   * inicia com os pesos do modelo populacional;
   * utiliza parte dos dados do novo paciente para Fine-Tuning;
   * é posteriormente avaliado em dados cronologicamente posteriores do mesmo participante.

Também poderão ser avaliadas diferentes quantidades de dados individuais utilizadas durante o Fine-Tuning.

## Tecnologias

Principais tecnologias previstas:

* Python;
* TensorFlow / Keras;
* pandas;
* NumPy;
* scikit-learn;
* Matplotlib;
* Jupyter Notebook / Google Colab.

## Estrutura Planejada do Repositório

text
.
├── README.md
├── requirements.txt
├── src/
│   ├── preprocessing/
│   ├── models/
│   ├── training/
│   └── evaluation/
│
├── notebooks/
│   ├── data_exploration.ipynb
│   └── baseline_lstm.ipynb
│
├── configs/
├── logs/
├── results/
└── models/


A estrutura poderá ser modificada conforme o desenvolvimento do projeto.

## Planejamento

### Etapa 1 — Revisão e definição do protocolo

* revisar trabalhos relacionados;
* definir entradas e horizonte de previsão;
* definir métricas;
* consolidar protocolo experimental.

*Marco:* protocolo experimental definido.

### Etapa 2 — Preparação do AZT1D

* baixar e inspecionar o dataset;
* identificar os dados de CGM, insulina e carboidratos;
* alinhar registros temporais;
* tratar dados ausentes;
* criar janelas temporais para treinamento.

*Marco:* pipeline inicial de pré-processamento funcional.

### Etapa 3 — Baseline LSTM

* implementar uma LSTM inicial;
* executar treinamento populacional;
* calcular RMSE e MAE;
* armazenar logs e resultados.

*Marco:* baseline populacional executável com resultados iniciais.

### Etapa 4 — Transfer Learning e Fine-Tuning

* carregar os pesos do modelo populacional;
* adaptar o modelo utilizando dados individuais;
* reservar dados posteriores para avaliação.

*Marco:* pipeline de personalização funcional.

### Etapa 5 — Experimentos

* comparar modelo populacional e personalizado;
* avaliar diferentes quantidades de dados de Fine-Tuning;
* testar estratégias adicionais quando viável.

*Marco:* resultados comparativos por participante.

### Etapa 6 — Análise

* consolidar RMSE e MAE;
* analisar ganhos de personalização;
* gerar tabelas e gráficos;
* discutir limitações.

*Marco:* resultados finais organizados.

### Etapa 7 — Artigo e Reprodutibilidade

* organizar código;
* documentar experimentos;
* preservar configurações e logs;
* preparar artigo e pôster.

*Marco:* repositório reproduzível e documentação final.

## Cronograma

| Período        | Atividade principal                                                 |
| -------------- | ------------------------------------------------------------------- |
| 29/08–11/09    | Revisão bibliográfica e início do processamento                     |
| 12/09–25/09    | Processamento do AZT1D e LSTM inicial                               |
| 26/09–09/10    | Modelo populacional e início do Fine-Tuning                         |
| 10/10–23/10    | Transfer Learning e experimentação                                  |
| 24/10–31/10    | Análise e consolidação dos resultados                               |
| *31/10/2026* | *Entrega 02 — Codificação, Metodologia Implementada e Resultados* |
| 01/11–09/11    | Artigo científico e pôster                                          |
| *09/11/2026* | *Entrega 03a — Artigo Científico e Pôster*                        |

## Status Atual

Projeto em fase inicial de planejamento e implementação.

Próximos passos:

* preparar ambiente de desenvolvimento;
* baixar e explorar o AZT1D;
* identificar o formato dos dados;
* implementar o pipeline inicial de pré-processamento;
* criar a primeira versão da LSTM baseline.

## Equipe

* Gustavo Faria Cardoso
* Mateus da Silva Maciel de Lima
* Pedro Costa Lyra
