# Convenções deste repositório

- A direção atual é aprimorar a seta contínua de tendência a +15 minutos. As
  propostas anteriores permanecem como histórico da pesquisa.
- `README.md` é o ponto de entrada; mantenha seus links e o índice
  `notebooks/README.md` coerentes com mudanças de estrutura.
- Use `notebooks/atual/` para a referência vigente, `notebooks/exploracao/` para
  estudos em andamento e `notebooks/arquivo/` para etapas encerradas.
- Registre tentativas em `docs/analises/`, usando o template em `docs/templates/`,
  e atualize `docs/HISTORICO.md`, inclusive quando não houver melhora.
- Diferencie proposta, execução e resultado. Registre responsáveis conhecidos
  e assistência automatizada; não invente autoria, datas ou métricas ausentes.
- Preserve os artefatos concluídos em `results/azt1d_seta/runs/`. Mudanças de
  configuração ou algoritmo exigem novo identificador e fontes arquivadas.
- A v1 em `src/seta_tendencia/azt1d_core.py` e no notebook é congelada por hash.
  Desenvolva outra versão sem invalidar a reprodução das rodadas v1.
- Não modifique os dados originais de `AZT1D 2025/`. Caches, previsões grandes
  e ZIPs ficam no disco, fora do versionamento conforme `.gitignore`.
- Escolha alterações usando treino e validação, nunca otimizando o teste já
  consultado. Informe a natureza exploratória e os limites do protocolo.
- Para verificar uma rodada concluída, use `scripts/verificar_experimento.py`;
  `scripts/analisar_resultados.py` refaz a análise sem treinar novamente.
