"""Compare retrospectiva de 30 min ao alvo futuro já avaliado, sem treino."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.seta_tendencia import azt1d_core as core

out = ROOT / "results/pessoal/zeroshot_2026_11_07"
raw = pd.read_csv(ROOT / "data/pessoal/Stream2026-11-07-23_44_52.tsv", sep="\t",
                  usecols=["Sensorid", "UnixTime", "mg/dL"])
raw["time"] = pd.to_datetime(raw.UnixTime, unit="s", utc=True, errors="coerce")
raw["g"] = pd.to_numeric(raw["mg/dL"], errors="coerce")
raw = raw.dropna(subset=["time", "g", "Sensorid"])
raw = raw[np.isfinite(raw.g) & (raw.g > 0)].sort_values("time")
raw["grid"] = raw.time.dt.ceil("5min")
mixed = raw.groupby("grid").Sensorid.nunique()
grid = raw.groupby("grid").tail(1).set_index("grid")
grid.loc[mixed[mixed > 1].index, "g"] = np.nan
pred = pd.read_csv(out / "previsoes.csv.gz")
base = pred[pred.model == "combined_zeroshot"].copy()
origin = pd.to_datetime(base.origin_ns, utc=True)
past = grid.g.reindex(origin - pd.Timedelta(minutes=30)).to_numpy()
now = grid.g.reindex(origin).to_numpy()
rate = (now - past) / 30
if not np.isfinite(rate).all():
    raise ValueError("Faltam leituras passadas nas janelas de comparação.")
site = base.copy()
site["model"] = "tendencia_passada_30min"
site["rate_pred"] = rate
site["delta_pred"] = 15 * rate
site["glucose_pred"] = now + 15 * rate
site["angle_deg"], site["vector_x"], site["vector_y"] = core.arrow_values(rate)
all_pred = pd.concat([pred, site], ignore_index=True)
metrics = core.metrics_table(all_pred)
metrics.to_csv(out / "comparacao_site_30min.csv", index=False)
table = metrics[(metrics.scope == "pooled") & (metrics.subset == "all")].set_index("model")
rapid = metrics[(metrics.scope == "pooled") & (metrics.subset == "rapid")].set_index("model")
lines = ["# Comparação com a tendência dos últimos 30 minutos", "",
         f"Mesmas {len(base):,} janelas, sem treinamento adicional.",
         "Taxa histórica: (G(t)-G(t-30))/30. Para comparar com o alvo futuro de +15 min, essa taxa é extrapolada como G(t)+15*taxa.",
         "O cálculo reproduz a regra descrita, sem assumir outros filtros do site.", "",
         "| Método | MAE taxa | MAE glicemia +15min | RMSE glicemia | Acerto de direção |",
         "|---|---:|---:|---:|---:|"]
for name in ["tendencia_passada_30min", "combined_zeroshot", "lstm_v1", "persistencia"]:
    r = table.loc[name]
    lines.append(f"| {name} | {r.mae_rate:.4f} | {r.mae_glucose:.3f} | {r.rmse_glucose:.3f} | {100*r.sign_correct:.2f}% |")
lines += ["", "A regra do site descreve o passado; o alvo desta comparação é a mudança futura. São tarefas diferentes. Aqui foi testada sua utilidade como previsão por extrapolação, sem validar o site nem sua implementação completa.",
          "As leituras de um minuto foram alinhadas causalmente a cinco minutos, sem interpolar. O modelo permanece congelado e o resultado é exploratório de uma pessoa."]
(out / "COMPARACAO_SITE_30MIN.md").write_text("\n".join(lines), encoding="utf-8")
print(table.loc[["tendencia_passada_30min", "combined_zeroshot", "lstm_v1", "persistencia"],
                ["mae_rate", "mae_glucose", "rmse_glucose", "sign_correct"]].to_string())
print("RAPID", rapid.loc[["tendencia_passada_30min", "combined_zeroshot"], ["mae_rate", "sign_correct"]].to_string())
print("REPORT", out / "COMPARACAO_SITE_30MIN.md")
