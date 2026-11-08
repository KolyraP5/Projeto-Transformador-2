"""Experimentos reprodutíveis de taxa futura de CGM: AZT1D, 5 e 15 min."""
from pathlib import Path
import hashlib
import json
import platform
import random
import re
import time
import zipfile

import numpy as np
import pandas as pd

VERSION = "1.0.0"
HORIZONS = (5, 15)
MODELS = ("persistencia", "ultima_variacao", "linear_15min", "lstm")
STEP_NS = 5 * 60 * 10**9


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(value), ensure_ascii=False,
                               indent=2, allow_nan=False), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_extract_zip(archive, destination):
    """Extrai somente dentro da pasta indicada; nunca executa conteúdo do ZIP."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("ZIP contém caminho fora da pasta de destino.")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("ZIP com link simbólico: extraia manualmente.")
        z.extractall(destination)
    return destination


def subject_number(path):
    for piece in reversed(Path(path).parts):
        match = re.search(r"subject[\s_-]*(\d+)", piece, re.I)
        if match:
            return int(match.group(1))
    return None


def discover_csvs(data_root):
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Pasta do AZT1D não encontrada: {root}")
    files = sorted(p for p in root.rglob("*")
                   if p.suffix.lower() == ".csv" and "__MACOSX" not in p.parts
                   and subject_number(p) is not None)
    if not files:
        raise ValueError("Nenhum CSV em pastas Subject N. Confira DATA_DIR ou o ZIP.")
    return files


def read_csv_flexible(path):
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            df = pd.read_csv(path, sep=None, engine="python", dtype=str,
                             encoding=encoding)
            df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Não foi possível ler a codificação de {path}")


def clean_subject(frames, subject, dayfirst=False, date_format="mixed"):
    """Uma origem de grade nunca recebe uma leitura com timestamp posterior."""
    raw = pd.concat(frames, ignore_index=True)
    n_raw = len(raw)
    raw = raw.drop_duplicates()
    n_duplicates = n_raw - len(raw)
    date_examples = raw["EventDateTime"].dropna().head(3).tolist()
    ts = pd.to_datetime(raw["EventDateTime"], format=date_format,
                        dayfirst=dayfirst, errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        raise ValueError("Timestamps com fuso: normalize explicitamente antes de usar.")
    values = pd.to_numeric(raw["CGM"].str.replace(",", ".", regex=False),
                           errors="coerce")
    invalid_g = ~np.isfinite(values) | (values <= 0)
    values = values.mask(invalid_g)
    valid = pd.DataFrame({"timestamp_raw": ts, "glucose": values}).dropna(
        subset=["timestamp_raw"]).sort_values("timestamp_raw")
    if valid.empty or valid["glucose"].notna().sum() < 15:
        raise ValueError(f"Subject {subject}: menos de 15 leituras válidas.")
    if valid["glucose"].median() < 30:
        raise ValueError(f"Subject {subject}: verifique unidade; esperado mg/dL.")
    valid["grid_time"] = valid["timestamp_raw"].dt.ceil("5min")
    grouped = valid.groupby("grid_time")["glucose"]
    g = grouped.first()
    distinct = grouped.nunique()
    g[distinct != 1] = np.nan
    raw_time = valid.dropna(subset=["glucose"]).groupby("grid_time")[
        "timestamp_raw"].max()
    index = pd.date_range(g.index.min(), g.index.max(), freq="5min")
    if len(index) > 200_000:
        raise ValueError("Período inesperadamente longo: confira o formato das datas.")
    grid = pd.DataFrame({"glucose": g, "timestamp_raw": raw_time}).reindex(index)
    grid.index.name = "grid_time"
    grid.loc[grid["glucose"].isna(), "timestamp_raw"] = pd.NaT
    good = grid["timestamp_raw"].notna()
    assert (grid.loc[good, "timestamp_raw"].to_numpy() <=
            grid.index[good].to_numpy()).all()
    age = (grid.index[good].to_numpy() -
           grid.loc[good, "timestamp_raw"].to_numpy()) / np.timedelta64(1, "m")
    audit = {
        "subject": subject, "raw_rows": n_raw, "duplicates_removed": n_duplicates,
        "invalid_timestamps": int(ts.isna().sum()),
        "invalid_glucose": int(invalid_g.sum()), "grid_bins": len(grid),
        "conflict_bins": int((distinct > 1).sum()),
        "missing_or_conflict_bins": int(grid.glucose.isna().sum()),
        "valid_bins": int(grid.glucose.notna().sum()),
        "start": str(index.min()), "end": str(index.max()),
        "raw_date_examples": " | ".join(map(str, date_examples)),
        "reading_age_max_min": float(np.max(age)),
        "reading_age_median_min": float(np.median(age)),
        "values_40_or_400": int(grid.glucose.isin([40, 400]).sum()),
    }
    return grid, audit


def windows_from_grid(grid, subject):
    """Divide primeiro, depois constrói suporte completo de 12 + 3 pontos."""
    n = len(grid)
    c1, c2 = int(n * .60), int(n * .80)
    bounds = [(0, c1), (c1, c2), (c2, n)]
    pieces, split_rows = {}, []
    for split, (lo, hi) in zip(("train", "val", "test"), bounds):
        part = grid.iloc[lo:hi]
        g = part.glucose.to_numpy(dtype=np.float32)
        if len(g) < 15:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(g, 15)
        starts = np.flatnonzero(np.isfinite(windows).all(axis=1))
        if len(starts) == 0:
            continue
        w = windows[starts]
        i = starts + 11
        times = part.index.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        raw = part.timestamp_raw.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        origins = times[i]
        targets = times[i[:, None] + np.array([1, 3])]
        future = w[:, [12, 14]].copy()
        X = w[:, :12, None].copy()
        y = (future - X[:, -1, 0, None]) / np.array([5., 15.], np.float32)
        pieces[split] = {
            "X": X, "y": y.astype(np.float32), "future": future,
            "subject": np.full(len(i), subject, dtype=np.int32),
            "origin_ns": origins, "target_ns": targets,
            "input_start_ns": times[starts], "raw_origin_ns": raw[i],
            "raw_target_ns": raw[i[:, None] + np.array([1, 3])],
        }
        assert np.all(targets - origins[:, None] ==
                      np.array([1, 3]) * STEP_NS)
        assert np.all(times[starts] >= times[0]) and np.all(targets <= times[-1])
        np.testing.assert_allclose(X[:, -1, 0, None] + y * [5, 15],
                                   future, atol=3e-5)
        split_rows.append({
            "subject": subject, "split": split, "start": str(part.index[0]),
            "end": str(part.index[-1]), "windows_before_sampling": len(i),
        })
    return pieces, split_rows


def prepare_dataset(data_root, run_dir, quick=True, dayfirst=False,
                    date_format="mixed", allow_mixed_alias=False,
                    seed=42, min_windows=1):
    started = time.perf_counter()
    run_dir, data_root = Path(run_dir), Path(data_root)
    files = discover_csvs(data_root)
    source = [{"path": str(p.relative_to(data_root)), "sha256": file_sha(p),
               "bytes": p.stat().st_size} for p in files]
    cfg = dict(quick=bool(quick), dayfirst=bool(dayfirst), date_format=date_format,
               allow_mixed_alias=bool(allow_mixed_alias), seed=int(seed),
               min_windows=int(min_windows), version=VERSION,
               core_sha256=file_sha(__file__))
    fingerprint = hashlib.sha256(json.dumps(
        [source, cfg], sort_keys=True).encode()).hexdigest()
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        old = read_json(manifest_path)
        if old["fingerprint"] != fingerprint:
            raise ValueError("RUN_NAME já existe com dados/configuração diferentes. "
                             "Use um novo RUN_NAME; não misture experimentos.")
        if not all((run_dir / "prepared" / f"{s}.npz").exists()
                   for s in ("train", "val", "test")):
            raise ValueError("Cache incompleto. Use um novo RUN_NAME.")
        return old
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("Pasta de execução incompleta ou ocupada. Use novo RUN_NAME.")
    frames, file_audit, excluded = {}, [], []
    for p in files:
        df = read_csv_flexible(p)
        sid = subject_number(p)
        names = {c.lower(): c for c in df.columns}
        date_col = names.get("eventdatetime")
        glucose_col = names.get("cgm")
        mixed = names.get("readings (cgm / bgm)")
        if not date_col or not (glucose_col or mixed):
            file_audit.append({"file": str(p.relative_to(data_root)),
                               "subject": sid, "status": "colunas_ausentes",
                               "columns": " | ".join(df.columns)})
            continue
        if not glucose_col and mixed and not allow_mixed_alias:
            excluded.append({"subject": sid, "reason": "alias_CGM_BGM_nao_confirmado"})
            file_audit.append({"file": str(p.relative_to(data_root)),
                               "subject": sid, "status": "alias_excluido"})
            continue
        df = df.rename(columns={date_col: "EventDateTime",
                                 glucose_col or mixed: "CGM"})
        frames.setdefault(sid, []).append(df)
        file_audit.append({"file": str(p.relative_to(data_root)),
                           "subject": sid, "status": "lido", "rows": len(df)})
    if not frames:
        raise ValueError("Não há CSVs elegíveis com EventDateTime e CGM.")
    eligible, audit, splits, collections = [], [], [], []
    for sid in sorted(frames):
        grid, row = clean_subject(frames[sid], sid, dayfirst, date_format)
        audit.append(row)
        pieces, split_rows = windows_from_grid(grid, sid)
        if not all(s in pieces and len(pieces[s]["X"]) >= min_windows
                   for s in ("train", "val", "test")):
            excluded.append({"subject": sid, "reason": "sem_janelas_nas_3_particoes"})
            continue
        if quick and len(eligible) >= 3:
            continue
        eligible.append(sid)
        collections.append(pieces)
        splits.extend(split_rows)
    if not eligible:
        raise ValueError("Nenhum participante tem janelas nas três partições.")
    prepared = run_dir / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    counts, timing_rows, artifact_hashes = {}, [], {}
    for split in ("train", "val", "test"):
        keys = collections[0][split].keys()
        arr = {k: np.concatenate([c[split][k] for c in collections])
               for k in keys}
        before = len(arr["X"])
        if quick and split == "train" and before > 20_000:
            keep = np.sort(np.random.default_rng(seed).choice(
                before, 20_000, replace=False))
            arr = {k: v[keep] for k, v in arr.items()}
        path = prepared / f"{split}.npz"
        np.savez_compressed(path, **arr)
        artifact_hashes[split] = file_sha(path)
        counts[split] = {"before_sampling": before, "used": len(arr["X"])}
        for sid in eligible:
            mask = arr["subject"] == sid
            for j, h in enumerate(HORIZONS):
                intervals = (arr["raw_target_ns"][mask, j] -
                             arr["raw_origin_ns"][mask]) / (60 * 1e9)
                future_after_origin = (arr["raw_target_ns"][mask, j] -
                                       arr["origin_ns"][mask]) / (60 * 1e9)
                timing_rows.append({
                    "split": split, "subject": sid, "horizon_min": h,
                    "n": len(intervals),
                    "raw_spacing_min": float(intervals.min()),
                    "raw_spacing_median": float(np.median(intervals)),
                    "raw_spacing_max": float(intervals.max()),
                    "raw_spacing_p95_abs_deviation": float(
                        np.quantile(np.abs(intervals - h), .95)),
                    "target_after_grid_origin_min": float(future_after_origin.min()),
                })
    pd.DataFrame(audit).to_csv(run_dir / "auditoria_participantes.csv", index=False)
    pd.DataFrame(file_audit).to_csv(run_dir / "auditoria_arquivos.csv", index=False)
    pd.DataFrame(splits).to_csv(run_dir / "divisoes_temporais.csv", index=False)
    pd.DataFrame(timing_rows).to_csv(run_dir / "auditoria_horizontes_reais.csv", index=False)
    manifest = {
        "fingerprint": fingerprint, "config": cfg, "source_files": source,
        "participants": eligible, "excluded": excluded, "counts": counts,
        "prepared_sha256": artifact_hashes,
        "preparation_seconds": time.perf_counter() - started,
        "dataset_doi": "10.17632/gk9m674wcx.1",
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    save_json(manifest_path, manifest)
    return manifest


def load_run(run_dir):
    run_dir = Path(run_dir)
    if not (run_dir / "manifest.json").exists():
        raise FileNotFoundError("Execute primeiro o notebook 01 com este RUN_NAME.")
    m = read_json(run_dir / "manifest.json")
    if m["config"]["core_sha256"] != file_sha(__file__):
        raise ValueError("Código comum diferente daquele usado na preparação.")
    return m


def load_split(run_dir, split):
    m = load_run(run_dir)
    if split not in ("train", "val", "test"):
        raise ValueError(split)
    path = Path(run_dir) / "prepared" / f"{split}.npz"
    if file_sha(path) != m["prepared_sha256"][split]:
        raise ValueError("Cache de dados mudou. Use uma execução nova.")
    with np.load(path, allow_pickle=False) as a:
        return {k: a[k] for k in a.files}


def baseline_rates(X):
    g = X[:, :, 0].astype(np.float64)
    times = np.array([-15., -10., -5., 0.])
    centered = times - times.mean()
    return {
        "persistencia": np.zeros(len(g)),
        "ultima_variacao": (g[:, -1] - g[:, -2]) / 5,
        "linear_15min": (g[:, -4:] @ centered) / (centered @ centered),
    }


def arrow_values(rate):
    q = np.asarray(rate, dtype=float)  # dividido por 1 mg/dL/min
    norm = np.hypot(1, q)
    return np.degrees(np.arctan(q)), 1 / norm, q / norm


def prediction_frame(arr, horizon, model, rate, split):
    j = HORIZONS.index(horizon)
    rate = np.asarray(rate, dtype=float).reshape(-1)
    if len(rate) != len(arr["X"]) or not np.isfinite(rate).all():
        raise ValueError("Previsões ausentes, não finitas ou desalinhadas.")
    angle, vx, vy = arrow_values(rate)
    current = arr["X"][:, -1, 0]
    return pd.DataFrame({
        "subject": arr["subject"], "split": split, "horizon_min": horizon,
        "model": model, "origin_ns": arr["origin_ns"],
        "origin": pd.to_datetime(arr["origin_ns"]),
        "target_time": pd.to_datetime(arr["target_ns"][:, j]),
        "glucose_current": current, "glucose_true": arr["future"][:, j],
        "rate_true": arr["y"][:, j], "rate_pred": rate,
        "delta_pred": horizon * rate,
        "glucose_pred": current + horizon * rate,
        "angle_deg": angle, "vector_x": vx, "vector_y": vy,
    })


def metrics_table(predictions, rapid_threshold=1.0):
    result = []
    groups = predictions.groupby(["split", "horizon_min", "model"], sort=True)
    for (split, horizon, model), df in groups:
        subsets = {"all": df, "rapid": df[df.rate_true.abs() >= rapid_threshold]}
        for subset, data in subsets.items():
            subjects = sorted(df.subject.unique())
            subject_rows = []
            for scope, sid, part in (
                [("pooled", "ALL", data)] +
                [("subject", str(s), data[data.subject == s]) for s in subjects]
            ):
                n = len(part)
                err = part.rate_pred.to_numpy() - part.rate_true.to_numpy()
                ge = part.glucose_pred.to_numpy() - part.glucose_true.to_numpy()
                prod = part.rate_true.to_numpy() * part.rate_pred.to_numpy()
                nonzero = part.rate_true.to_numpy() != 0
                base = {
                    "split": split, "horizon_min": horizon, "model": model,
                    "subset": subset, "scope": scope, "subject": sid, "n": n,
                    "participants_evaluable": int(n > 0) if scope == "subject"
                        else int(part.subject.nunique()),
                    "mae_rate": float(np.mean(np.abs(err))) if n else np.nan,
                    "rmse_rate": float(np.sqrt(np.mean(err**2))) if n else np.nan,
                    "bias_rate": float(np.mean(err)) if n else np.nan,
                    "mae_glucose": float(np.mean(np.abs(ge))) if n else np.nan,
                    "rmse_glucose": float(np.sqrt(np.mean(ge**2))) if n else np.nan,
                    "direction_n": int(nonzero.sum()),
                    "sign_correct": float(np.mean(prod[nonzero] > 0))
                        if nonzero.any() else np.nan,
                    "sign_opposite": float(np.mean(prod[nonzero] < 0))
                        if nonzero.any() else np.nan,
                    "pred_exact_zero": float(np.mean(
                        part.rate_pred.to_numpy()[nonzero] == 0))
                        if nonzero.any() else np.nan,
                }
                result.append(base)
                if scope == "subject":
                    subject_rows.append(base)
            tab = pd.DataFrame(subject_rows)
            macro = dict(split=split, horizon_min=horizon, model=model,
                         subset=subset, scope="macro", subject="MEAN",
                         n=int(tab.n.sum()),
                         participants_evaluable=int((tab.n > 0).sum()),
                         direction_n=int(tab.direction_n.sum()))
            for key in ("mae_rate", "rmse_rate", "bias_rate", "mae_glucose",
                        "rmse_glucose", "sign_correct", "sign_opposite",
                        "pred_exact_zero"):
                macro[key] = float(tab[key].mean())
            result.append(macro)
    return pd.DataFrame(result)


def baseline_validation(run_dir):
    run_dir = Path(run_dir)
    arr = load_split(run_dir, "val")
    frames = [prediction_frame(arr, h, name, rate, "val")
              for h in HORIZONS for name, rate in baseline_rates(arr["X"]).items()]
    predictions = pd.concat(frames, ignore_index=True)
    metrics = metrics_table(predictions)
    out = run_dir / "baselines"
    out.mkdir(exist_ok=True)
    predictions.to_csv(out / "previsoes_validacao.csv.gz", index=False)
    metrics.to_csv(out / "metricas_validacao.csv", index=False)
    return metrics


def make_model(hidden=32):
    import torch
    class RateLSTM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(1, hidden, batch_first=True)
            self.head = torch.nn.Linear(hidden, 1)
        def forward(self, x):
            states, _ = self.lstm(x)  # estado reiniciado por janela
            return self.head(states[:, -1, :]).squeeze(-1)
    return RateLSTM()


def train_horizon(run_dir, horizon, seed=42):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    run_dir = Path(run_dir)
    m = load_run(run_dir)
    if horizon not in HORIZONS:
        raise ValueError("Horizontes permitidos: 5 e 15.")
    out = run_dir / f"lstm_h{horizon}"
    if (out / "completed.json").exists():
        saved = read_json(out / "completed.json")
        if saved["fingerprint"] != m["fingerprint"] or saved["seed"] != seed:
            raise ValueError("Modelo existente incompatível; crie novo RUN_NAME.")
        print(f"Reutilizando treinamento concluído: {out.name}")
        return pd.read_csv(out / "metricas_validacao.csv")
    if (run_dir / "selecao_horizonte.json").exists():
        raise ValueError("Seleção já congelada: novo treinamento exige novo RUN_NAME.")
    out.mkdir(parents=True, exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val = load_split(run_dir, "train"), load_split(run_dir, "val")
    j = HORIZONS.index(horizon)
    scales = {
        "x_mean": float(train["X"].mean()), "x_std": float(train["X"].std()),
        "y_mean": float(train["y"][:, j].mean()),
        "y_std": float(train["y"][:, j].std()),
    }
    scales["x_std"] = max(scales["x_std"], 1e-6)
    scales["y_std"] = max(scales["y_std"], 1e-6)
    def tensors(a):
        x = (a["X"] - scales["x_mean"]) / scales["x_std"]
        y = (a["y"][:, j] - scales["y_mean"]) / scales["y_std"]
        return TensorDataset(torch.from_numpy(x.astype(np.float32)),
                             torch.from_numpy(y.astype(np.float32)))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(tensors(train), batch_size=64, shuffle=True,
                              generator=generator, num_workers=0)
    val_loader = DataLoader(tensors(val), batch_size=1024, shuffle=False, num_workers=0)
    model = make_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loss_fn = torch.nn.MSELoss()
    epochs = 3 if m["config"]["quick"] else 15
    config = dict(horizon_min=horizon, seed=seed, hidden=32, batch_size=64,
                  lr=.001, epochs=epochs, patience=3, device=str(device),
                  fingerprint=m["fingerprint"], scales=scales, version=VERSION)
    save_json(out / "config.json", config)
    save_json(out / "ambiente.json", {
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "pandas": pd.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    })
    history, best, stale = [], float("inf"), 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train(); total = 0.
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            if not torch.isfinite(loss):
                raise RuntimeError("Perda não finita. Confira os dados e escalas.")
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(xb)
        model.eval(); val_total = 0.
        with torch.no_grad():
            for xb, yb in val_loader:
                val_total += float(loss_fn(model(xb.to(device)),
                                          yb.to(device)).cpu()) * len(xb)
        train_loss = total / len(train["X"])
        val_loss = val_total / len(val["X"])
        history.append(dict(epoch=epoch, train_mse_scaled=train_loss,
                            val_mse_scaled=val_loss,
                            elapsed_seconds=time.perf_counter() - started))
        pd.DataFrame(history).to_csv(out / "historico.csv", index=False)
        if val_loss < best:
            best, stale = val_loss, 0
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                       out / "best_weights.pt")
        else:
            stale += 1
        print(f"h={horizon} | época {epoch}/{epochs} | "
              f"treino={train_loss:.5f} | validação={val_loss:.5f}")
        if stale >= 3:
            break
    elapsed = time.perf_counter() - started
    rates, inference_s = predict_horizon(run_dir, horizon, val)
    pred = prediction_frame(val, horizon, "lstm", rates, "val")
    metrics = metrics_table(pred)
    pred.to_csv(out / "previsoes_validacao.csv.gz", index=False)
    metrics.to_csv(out / "metricas_validacao.csv", index=False)
    save_json(out / "completed.json", {
        "fingerprint": m["fingerprint"], "seed": seed,
        "epochs_ran": len(history), "best_val_mse_scaled": best,
        "train_seconds": elapsed, "val_inference_seconds": inference_s,
        "weights_sha256": file_sha(out / "best_weights.pt"),
    })
    return metrics


def predict_horizon(run_dir, horizon, arr):
    import torch
    out = Path(run_dir) / f"lstm_h{horizon}"
    cfg = read_json(out / "config.json")
    m = load_run(run_dir)
    if cfg["fingerprint"] != m["fingerprint"] or cfg["horizon_min"] != horizon:
        raise ValueError("Configuração incompatível com os dados.")
    if (out / "completed.json").exists():
        expected = read_json(out / "completed.json")["weights_sha256"]
        if file_sha(out / "best_weights.pt") != expected:
            raise ValueError("Pesos alterados depois do treino.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(cfg["hidden"])
    model.load_state_dict(torch.load(out / "best_weights.pt",
                                    map_location="cpu", weights_only=True))
    model.to(device).eval()
    s = cfg["scales"]
    x = (arr["X"] - s["x_mean"]) / s["x_std"]
    started = time.perf_counter()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), 2048):
            xb = torch.from_numpy(x[start:start+2048].astype(np.float32)).to(device)
            chunks.append(model(xb).cpu().numpy())
    rate = np.concatenate(chunks) * s["y_std"] + s["y_mean"]
    return rate, time.perf_counter() - started


def validation_results(run_dir):
    run_dir = Path(run_dir)
    baseline_validation(run_dir)
    paths = [run_dir / "baselines" / "previsoes_validacao.csv.gz"]
    for h in HORIZONS:
        if not (run_dir / f"lstm_h{h}" / "completed.json").exists():
            raise FileNotFoundError(f"Conclua o notebook da LSTM de {h} minutos.")
        paths.append(run_dir / f"lstm_h{h}" / "previsoes_validacao.csv.gz")
    predictions = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    expected = set(map(tuple, predictions[predictions.model == "persistencia"]
                       [["subject", "origin_ns", "horizon_min"]].to_numpy()))
    for model in MODELS:
        actual = set(map(tuple, predictions[predictions.model == model]
                         [["subject", "origin_ns", "horizon_min"]].to_numpy()))
        if actual != expected:
            raise ValueError("Modelos com origens diferentes na validação.")
    return predictions, metrics_table(predictions)


def select_horizon(run_dir, val_metrics, exploratory=False):
    """Regra operacional fixa; nunca usa métricas de teste."""
    run_dir = Path(run_dir)
    if set(val_metrics.split) != {"val"}:
        raise ValueError("Seleção recebe somente validação.")
    summaries = []
    for h in HORIZONS:
        m = val_metrics[val_metrics.horizon_min == h]
        macro = m[(m.scope == "macro") & (m.subset == "all")].set_index("model")
        best = macro.loc[[x for x in MODELS if x != "lstm"], "mae_rate"].idxmin()
        lstm, base = float(macro.loc["lstm", "mae_rate"]), float(macro.loc[best, "mae_rate"])
        per = m[(m.scope == "subject") & (m.subset == "all")].pivot(
            index="subject", columns="model", values="mae_rate")
        wins = float((per.lstm < per[best]).mean())
        rapid = m[(m.scope == "macro") & (m.subset == "rapid")].set_index("model")
        rapid_lstm, rapid_base = float(rapid.loc["lstm", "mae_rate"]), float(rapid.loc[best, "mae_rate"])
        opposite_lstm = float(rapid.loc["lstm", "sign_opposite"])
        opposite_base = float(rapid.loc[best, "sign_opposite"])
        correct_lstm = float(rapid.loc["lstm", "sign_correct"])
        correct_base = float(rapid.loc[best, "sign_correct"])
        # Sem suporte em mudanças rápidas, a candidatura fica inconclusiva.
        candidate = bool(base > 0 and lstm < base and wins > .5 and
                         np.isfinite(rapid_lstm) and rapid_lstm <= rapid_base and
                         correct_lstm >= correct_base)
        summaries.append({
            "horizon_min": h, "best_baseline": best, "mae_lstm": lstm,
            "mae_best_baseline": base, "gain": 1-lstm/base if base > 0 else None,
            "fraction_subjects_improved": wins, "rapid_mae_lstm": rapid_lstm,
            "rapid_mae_baseline": rapid_base,
            "rapid_sign_opposite_lstm": opposite_lstm,
            "rapid_sign_opposite_baseline": opposite_base, "candidate": candidate,
            "rapid_sign_correct_lstm": correct_lstm,
            "rapid_sign_correct_baseline": correct_base,
        })
    candidates = [s for s in summaries if s["candidate"]]
    chosen = None
    if candidates:
        max_gain = max(s["gain"] for s in candidates)
        # Empate operacional: diferença de até 0,02 no ganho relativo.
        near = [s for s in candidates if max_gain - s["gain"] <= .02]
        chosen = max(s["horizon_min"] for s in near)
    selection = {
        "fingerprint": load_run(run_dir)["fingerprint"],
        "chosen_horizon_min": chosen, "validation_summary": summaries,
        "exploratory_prior_test_access": bool(exploratory),
        "rule": "Ganho macro >0, maioria dos participantes, sem piora de MAE "
                "ou acerto de sinal em mudanças rápidas. Empate <=0.02 favorece 15. "
                "Prever zero conta como erro direcional quando a taxa real é não nula.",
        "validation_sha256": hashlib.sha256(
            val_metrics.to_csv(index=False).encode()).hexdigest(),
        "model_hashes": {str(h): file_sha(run_dir / f"lstm_h{h}" / "best_weights.pt")
                         for h in HORIZONS},
    }
    path = run_dir / "selecao_horizonte.json"
    selection = clean_json(selection)
    if path.exists():
        if read_json(path) != selection:
            raise ValueError("Seleção já congelada e diferente. Use novo RUN_NAME.")
    else:
        save_json(path, selection)
    return selection


def test_results(run_dir):
    run_dir = Path(run_dir)
    if not (run_dir / "selecao_horizonte.json").exists():
        raise ValueError("Congele a seleção na validação antes do teste.")
    selection = read_json(run_dir / "selecao_horizonte.json")
    if selection["fingerprint"] != load_run(run_dir)["fingerprint"]:
        raise ValueError("Seleção e dados incompatíveis.")
    for h in HORIZONS:
        if selection["model_hashes"][str(h)] != file_sha(
                run_dir / f"lstm_h{h}" / "best_weights.pt"):
            raise ValueError("Modelo mudou depois da escolha do horizonte.")
    arr = load_split(run_dir, "test")
    frames, timings = [], {}
    for h in HORIZONS:
        rates = baseline_rates(arr["X"])
        rates["lstm"], elapsed = predict_horizon(run_dir, h, arr)
        timings[str(h)] = elapsed
        frames.extend(prediction_frame(arr, h, name, rate, "test")
                      for name, rate in rates.items())
    pred = pd.concat(frames, ignore_index=True)
    metrics = metrics_table(pred)
    out = run_dir / "final"
    out.mkdir(exist_ok=True)
    pred.to_csv(out / "previsoes_teste.csv.gz", index=False)
    metrics.to_csv(out / "metricas_teste.csv", index=False)
    save_json(out / "tempos_inferencia_teste.json", timings)
    save_json(out / "test_done.json", {
        "selection_sha256": file_sha(run_dir / "selecao_horizonte.json"),
        "fingerprint": selection["fingerprint"],
    })
    return pred, metrics


def skill_table(metrics):
    rows = []
    for (split, h, subset, scope, sid), g in metrics.groupby(
            ["split", "horizon_min", "subset", "scope", "subject"], dropna=False):
        by = g.set_index("model")
        if "lstm" not in by.index:
            continue
        for b in MODELS[:-1]:
            if b not in by.index:
                continue
            base, value = float(by.loc[b, "mae_rate"]), float(by.loc["lstm", "mae_rate"])
            rows.append(dict(split=split, horizon_min=h, subset=subset, scope=scope,
                             subject=sid, baseline=b,
                             gain=1-value/base if base > 0 and np.isfinite(value) else np.nan))
    return pd.DataFrame(rows)


def plot_history(run_dir, horizon):
    import matplotlib.pyplot as plt
    out = Path(run_dir) / f"lstm_h{horizon}"
    df = pd.read_csv(out / "historico.csv")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df.epoch, df.train_mse_scaled, marker="o", label="Treino")
    ax.plot(df.epoch, df.val_mse_scaled, marker="o", label="Validação")
    ax.set(xlabel="Época", ylabel="MSE padronizado", title=f"LSTM — {horizon} minutos")
    ax.legend(); fig.tight_layout()
    fig.savefig(out / "aprendizado.png", dpi=140)
    return fig


def plot_results(predictions, run_dir):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    out = Path(run_dir) / "final"
    out.mkdir(exist_ok=True)
    figs = []
    d = predictions[predictions.model == "lstm"].copy()
    d["origin"] = pd.to_datetime(d.origin)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, h in zip(axes, HORIZONS):
        part = d[d.horizon_min == h]
        step = max(1, len(part) // 5000)
        ax.scatter(part.rate_true.iloc[::step], part.rate_pred.iloc[::step],
                   s=6, alpha=.25)
        lo = min(part.rate_true.min(), part.rate_pred.min())
        hi = max(part.rate_true.max(), part.rate_pred.max())
        ax.plot([lo, hi], [lo, hi], "--", color="black")
        ax.set(title=f"+{h} min", xlabel="Taxa real (mg/dL/min)",
               ylabel="Taxa prevista (mg/dL/min)")
    fig.tight_layout(); fig.savefig(out / "taxa_real_vs_prevista.png", dpi=150)
    figs.append(fig)
    chosen = None
    for sid, group in d[d.horizon_min == 5].groupby("subject", sort=True):
        group = group.sort_values("origin")
        segments = group.origin.diff().ne(pd.Timedelta(minutes=5)).cumsum()
        for _, segment in group.groupby(segments):
            if len(segment) >= 25:
                chosen = (sid, segment.origin.iloc[0], segment.origin.iloc[24])
                break
        if chosen:
            break
    if chosen:
        sid, start, end = chosen
        panel = d[(d.subject == sid) & d.origin.between(start, end)]
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        g = panel[panel.horizon_min == 5].sort_values("origin")
        axes[0].plot(g.origin, g.glucose_current, color="black")
        axes[0].set(ylabel="CGM (mg/dL)", title=f"Primeira janela elegível — Subject {sid}")
        for ax, h in zip(axes[1:], HORIZONS):
            part = panel[panel.horizon_min == h].sort_values("origin")
            ax.plot(part.origin, part.rate_true, label="Taxa futura real")
            ax.plot(part.origin, part.rate_pred, label="LSTM")
            ax.axhline(0, color="gray", lw=.7)
            ax.set(ylabel=f"+{h} min\nmg/dL/min"); ax.legend()
        axes[-1].set_xlabel("Origem da previsão (taxas alinhadas à origem)")
        fig.tight_layout(); fig.savefig(out / "comparacao_temporal.png", dpi=150)
        figs.append(fig)
    examples = d.sort_values(["subject", "origin", "horizon_min"]).head(6)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, (_, row) in zip(axes.flat, examples.iterrows()):
        ax.add_patch(FancyArrowPatch((-.4*row.vector_x, -.4*row.vector_y),
                                    (.6*row.vector_x, .6*row.vector_y),
                                    arrowstyle="-|>", mutation_scale=25, lw=2))
        ax.set(xlim=(-1, 1), ylim=(-1, 1), aspect="equal",
               title=f"{row.rate_pred:+.2f} mg/dL/min | +{int(row.horizon_min)} min\n"
                     f"Real: {row.rate_true:+.2f} | Subject {int(row.subject)}")
        ax.axis("off")
    fig.suptitle("Primeiras previsões do teste — escala visual fixa")
    fig.tight_layout(); fig.savefig(out / "setas_continuas.png", dpi=150)
    figs.append(fig)
    # Primeiro episódio rápido de cada horizonte, sem escolher o melhor acerto.
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    for ax, h in zip(axes, HORIZONS):
        part = d[d.horizon_min == h].sort_values(["subject", "origin"])
        rapid = part[part.rate_true.abs() >= 1]
        if rapid.empty:
            ax.text(.5, .5, "Sem episódio rápido elegível", ha="center",
                    transform=ax.transAxes)
            continue
        first = rapid.iloc[0]
        window = part[(part.subject == first.subject) &
                      part.origin.between(first.origin-pd.Timedelta(minutes=30),
                                          first.origin+pd.Timedelta(minutes=90))]
        ax.plot(window.origin, window.rate_true, label="Taxa futura real")
        ax.plot(window.origin, window.rate_pred, label="LSTM")
        ax.axvline(first.origin, color="gray", ls="--")
        ax.set(title=f"Primeiro episódio rápido — +{h} min, Subject {int(first.subject)}",
               ylabel="mg/dL/min"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "primeiro_episodio_rapido.png", dpi=150)
    figs.append(fig)
    return figs


def write_report(run_dir, val_metrics, test_metrics):
    run_dir = Path(run_dir)
    selection, manifest = read_json(run_dir / "selecao_horizonte.json"), load_run(run_dir)
    all_metrics = pd.concat([val_metrics, test_metrics], ignore_index=True)
    all_metrics.to_csv(run_dir / "final" / "metricas.csv", index=False)
    skills = skill_table(all_metrics)
    skills.to_csv(run_dir / "final" / "ganhos.csv", index=False)
    rows = all_metrics[(all_metrics.scope == "macro") & (all_metrics.subset == "all")]
    lines = [
        "# AZT1D — Seta contínua: resultados executados",
        "",
        f"- Execução: {run_dir.name}",
        f"- Modo rápido: {manifest['config']['quick']}",
        f"- Participantes: {manifest['participants']}",
        f"- Horizonte escolhido antes do teste: {selection['chosen_horizon_min']}",
        f"- Teste já consultado em trabalhos anteriores: "
        f"{selection['exploratory_prior_test_access']}",
        "- Taxa média futura em mg/dL/min; sem classes e sem simulação de doses.",
        "",
        "| Partição | Horizonte | Modelo | MAE taxa (média por pessoa) | RMSE glicemia (média por pessoa) |",
        "|---|---:|---|---:|---:|",
    ]
    for r in rows.itertuples():
        lines.append(f"| {r.split} | {r.horizon_min} | {r.model} | "
                     f"{r.mae_rate:.4f} | {r.rmse_glucose:.3f} |")
    lines += ["", "## Interpretação", "",
              "A seleção usa validação, ganho sobre baseline e comportamento nas mudanças rápidas.",
              "Consulte selecao_horizonte.json para a regra e a decisão congelada.",
              "Se o horizonte escolhido for null, nenhum apresentou evidência suficiente pela regra operacional.",
              "As métricas de teste confirmam ou contradizem a decisão; não reajuste os modelos neste teste.",
              "", "## Limitações", "",
              "Janelas sobrepostas não são amostras independentes. Avaliação temporal nas mesmas pessoas.",
              "Agrupamento em 5 minutos; diferenças dos timestamps reais estão na auditoria.",
              "Bins conflitantes e lacunas foram descartados, podendo mudar a população avaliada.",
              "O modo rápido verifica execução; não sustenta conclusão sobre toda a base.",
              "A seta resume a mudança média futura do CGM e não detecta todos os picos intermediários.",
              "A escala angular é uma escolha visual do projeto, sem validação clínica.",
              "", "## Fonte", "",
              "Khamesian et al. AZT1D, versão 1. https://doi.org/10.17632/gk9m674wcx.1",
              ""]
    (run_dir / "final" / "resumo.md").write_text("\n".join(lines), encoding="utf-8")
    return run_dir / "final" / "resumo.md"


def export_results(run_dir):
    run_dir = Path(run_dir)
    zip_path = run_dir.parent / f"{run_dir.name}_resultados.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(__file__, arcname=str(Path(run_dir.name) / "codigo" / "azt1d_core.py"))
        for p in sorted(run_dir.rglob("*")):
            if p.is_file() and "prepared" not in p.relative_to(run_dir).parts:
                z.write(p, arcname=str(Path(run_dir.name) / p.relative_to(run_dir)))
    return zip_path

