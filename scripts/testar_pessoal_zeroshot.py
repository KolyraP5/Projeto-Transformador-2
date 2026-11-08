"""Avaliação local, sem treino, de CGM pessoal em grade causal de cinco minutos."""
from pathlib import Path
import hashlib
import json
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.seta_tendencia import azt1d_core as core
from src.seta_tendencia import aprimoramento as engine


def main():
    started = time.perf_counter()
    source = ROOT / "data/pessoal/Stream2026-11-07-23_44_52.tsv"
    out = ROOT / "results/pessoal/zeroshot_2026_11_07"
    if out.exists():
        raise SystemExit("Resultado já existe; preserve-o antes de outra avaliação.")
    out.mkdir(parents=True)
    model = ROOT / "results/azt1d_seta/experiments/aprimoramento_v2/models/combined_B3_s42"
    choice = dict(model="combined_B3_s42", reason="Vencedor da triagem AZT1D antes de avaliar dados pessoais",
                  approved_final=False, training_on_personal_data=False,
                  weights_sha256=core.file_sha(model / "best_weights.pt"),
                  scales_sha256=core.file_sha(model / "scales.json"),
                  frozen_utc=datetime.now(timezone.utc).isoformat())
    core.save_json(out / "selecao_previa.json", choice)
    df = pd.read_csv(source, sep="\t", usecols=["Sensorid", "UnixTime", "mg/dL"])
    raw_n = len(df)
    df["timestamp"] = pd.to_datetime(pd.to_numeric(df.UnixTime, errors="coerce"), unit="s", utc=True, errors="coerce")
    df["glucose"] = pd.to_numeric(df["mg/dL"], errors="coerce")
    df = df.dropna(subset=["timestamp", "glucose", "Sensorid"])
    df = df[np.isfinite(df.glucose) & (df.glucose > 0)].copy()
    before = len(df)
    df = df.drop_duplicates(["Sensorid", "timestamp", "glucose"]).sort_values("timestamp")
    # Leituras de um minuto: última observação disponível, nunca média futura.
    df["grid"] = df.timestamp.dt.ceil("5min")
    sensors = df.groupby("grid").Sensorid.nunique()
    grid = df.groupby("grid", sort=True).tail(1).set_index("grid")
    grid.loc[sensors[sensors > 1].index, "glucose"] = np.nan
    grid = grid.reindex(pd.date_range(grid.index.min(), grid.index.max(), freq="5min"))
    codes = pd.Categorical(grid.Sensorid).codes
    g = grid.glucose.to_numpy(dtype=np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(g, 15)
    sensor_windows = np.lib.stride_tricks.sliding_window_view(codes, 15)
    keep = (np.isfinite(windows).all(axis=1) & (sensor_windows[:, 0] >= 0)
            & (sensor_windows == sensor_windows[:, :1]).all(axis=1))
    starts = np.flatnonzero(keep)
    if not len(starts):
        raise ValueError("Sem janelas completas de um mesmo sensor; não houve interpolação.")
    w = windows[starts]
    times = grid.index.as_unit("ns").asi8
    raw_times = pd.DatetimeIndex(grid.timestamp).as_unit("ns").asi8
    origins = starts + 11
    future = w[:, [12, 14]].copy()
    arr = dict(X=w[:, :12, None].copy(), future=future,
               y=((future - w[:, 11, None]) / np.array([5, 15], np.float32)).astype(np.float32),
               subject=np.ones(len(starts), dtype=np.int32), origin_ns=times[origins],
               target_ns=times[origins[:, None] + [1, 3]], input_start_ns=times[starts],
               raw_origin_ns=raw_times[origins], raw_target_ns=raw_times[origins[:, None] + [1, 3]])
    assert np.all(arr["raw_origin_ns"] <= arr["origin_ns"])
    assert np.all(arr["raw_target_ns"] > arr["origin_ns"][:, None])
    torch.set_num_threads(2)
    rates = core.baseline_rates(arr["X"])
    rates["combined_zeroshot"] = engine.predict_variant(arr, model)
    rates["lstm_v1"], _ = core.predict_horizon(ROOT / "results/azt1d_seta/runs/full_v1", 15, arr)
    frames = [core.prediction_frame(arr, 15, name, rate, "personal_external") for name, rate in rates.items()]
    pred = pd.concat(frames, ignore_index=True)
    metrics = core.metrics_table(pred)
    metrics.to_csv(out / "metricas.csv", index=False)
    pred.to_csv(out / "previsoes.csv.gz", index=False)
    diagnostics = []
    for name, rate in rates.items():
        for label, mask in {"low":abs(arr["y"][:,1])<.5,"rapid":abs(arr["y"][:,1])>=1,"extreme":abs(arr["y"][:,1])>=2}.items():
            y,p = arr["y"][mask,1],np.asarray(rate)[mask]
            diagnostics.append(dict(model=name,subset=label,n=len(y),mae_rate=float(np.abs(y-p).mean()) if len(y) else None))
    pd.DataFrame(diagnostics).to_csv(out / "intensidade.csv", index=False)
    intervals = df.timestamp.diff().dt.total_seconds().div(60)
    audit = dict(raw_rows=raw_n, valid_rows=len(df), exact_duplicates_removed=before-len(df),
                 sensors=int(df.Sensorid.nunique()), native_interval_median_min=float(intervals[intervals>0].median()),
                 first_utc=str(df.timestamp.min()), last_utc=str(df.timestamp.max()),
                 grid_minutes=5, horizon_minutes=15, windows=len(starts),
                 missing_bins=int(grid.glucose.isna().sum()), mixed_sensor_bins=int((sensors>1).sum()),
                 interpolation=False, sensor_crossing_windows=False, source_sha256=core.file_sha(source))
    core.save_json(out / "auditoria.json", audit)
    allrows = metrics[(metrics.scope=="pooled") & (metrics.subset=="all")].set_index("model")
    candidate = allrows.loc["combined_zeroshot"]
    lines = ["# Teste pessoal zero-shot — +15 minutos", "",
             "Dados e resultados pessoais mantidos localmente, fora do Git.", "",
             f"- Leituras originais: {raw_n:,}; sensores: {audit['sensors']}; janelas válidas: {len(starts):,}.",
             f"- Intervalo nativo mediano: {audit['native_interval_median_min']:.3f} min; grade causal de 5 min.",
             "- Última leitura disponível por intervalo, sem interpolar e sem atravessar troca de sensor.",
             "- Modelo e escalas congelados no AZT1D; nenhum treino, calibração ou seleção usando este histórico.",
             "- Candidato combinado é vencedor provisório da triagem; confirmação ampliada interrompida por prazo.",
             "", "| Modelo | MAE taxa (mg/dL/min) | MAE glicemia (mg/dL) | RMSE glicemia | Acerto sinal |",
             "|---|---:|---:|---:|---:|"]
    for name,row in allrows.iterrows():
        lines.append(f"| {name} | {row.mae_rate:.4f} | {row.mae_glucose:.3f} | {row.rmse_glucose:.3f} | {100*row.sign_correct:.2f}% |")
    lines += ["", "## Interpretação", "",
              "Avaliação externa exploratória de uma pessoa e de seu dispositivo. As janelas se sobrepõem; não são observações independentes. Não demonstra validade clínica nem desempenho em outras pessoas. A taxa é a mudança média até +15 min; a seta do fabricante e a coluna Rate não foram usadas como alvo ou entrada.",
              "O resultado não é usado para escolher ou reajustar o modelo. Diferenças entre sensores, frequência nativa e população do AZT1D podem afetar o desempenho."]
    (out / "RELATORIO.md").write_text("\n".join(lines), encoding="utf-8")
    n=min(288,len(starts));t=pd.to_datetime(arr["origin_ns"][:n],utc=True).tz_convert("America/Sao_Paulo")
    fig,ax=plt.subplots(figsize=(12,4));ax.plot(t,arr["y"][:n,1],label="Taxa real +15 min",lw=1)
    ax.plot(t,rates["combined_zeroshot"][:n],label="Combinado zero-shot",lw=1)
    ax.plot(t,rates["lstm_v1"][:n],label="LSTM v1",lw=.8,alpha=.7)
    ax.set(ylabel="mg/dL/min",title="Primeiras 288 janelas válidas — escolha cronológica fixa")
    ax.legend();fig.autofmt_xdate();fig.tight_layout();fig.savefig(out/"comparacao.png",dpi=130);plt.close(fig)
    core.save_json(out / "execucao.json", dict(status="completed", elapsed_seconds=time.perf_counter()-started,
                   zero_shot=True, n=len(starts), model_weights_sha256=choice["weights_sha256"]))
    print(allrows[["mae_rate","mae_glucose","rmse_glucose","sign_correct"]].to_string())
    print("RESULT",out,"SECONDS",round(time.perf_counter()-started,2),flush=True)


if __name__ == "__main__":
    main()
