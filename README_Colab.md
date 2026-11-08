# Executar no Google Colab

O guia completo está em [docs/EXECUCAO.md](docs/EXECUCAO.md).

1. Envie [o notebook atual](notebooks/atual/AZT1D_Setas_e_Tendencia.ipynb) ao Colab.
2. Execute as células em ordem, monte o Drive e indique o ZIP ou pasta original
   do AZT1D. O notebook contém o código comum; não precisa enviá-lo separadamente.
3. Para conferência, use `QUICK=True` e um nome novo. Para o experimento completo,
   use `QUICK=False` e outro `RUN_NAME`.
4. Preserve `TESTE_JA_CONSULTADO=True`: a execução anterior já acessou o teste.
5. Baixe o ZIP de resultados ou mantenha a pasta da execução no Drive.

O `full_v1` deste repositório foi executado localmente. Para repeti-lo em outro
ambiente, use um nome como `full_v1_colab`; versões e hardware podem alterar
ligeiramente os resultados. Não sobrescreva uma execução registrada.
