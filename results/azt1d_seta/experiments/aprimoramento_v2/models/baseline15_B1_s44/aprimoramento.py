"""Motor v2 de ablações causais para a seta de +15 min, sem alterar a v1.

As partições são fornecidas pelo orquestrador. ``stop`` escolhe a época e
``evaluate`` compara configurações depois do treino. Nenhuma escolha depende
das métricas de ``evaluate`` neste módulo.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import platform
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from . import azt1d_core as core


VERSION = "2.0.0"
HORIZON = 15
_TRAIN_FIELDS = ("feature_mode", "rapid_weight", "seed", "hidden", "batch_size",
                 "lr", "threads", "device")


def normalize_config(config):
    """Complete os parâmetros sem retirar a identificação fornecida pelo chamador."""
    cfg = dict(feature_mode="absolute", max_epochs=15, patience=3,
               rapid_weight=1.0, seed=42, threads=1, hidden=32,
               batch_size=64, lr=.001, device="auto", evaluation_split="dev")
    cfg.update(config)
    for name in ("max_epochs", "patience", "seed", "threads", "hidden", "batch_size"):
        cfg[name] = int(cfg[name])
    for name in ("rapid_weight", "lr"):
        cfg[name] = float(cfg[name])
    if cfg["feature_mode"] not in ("absolute", "delta", "relative", "combined"):
        raise ValueError("feature_mode deve ser absolute, delta, relative ou combined.")
    if any(cfg[k] < 1 for k in ("max_epochs", "patience", "threads", "hidden", "batch_size")):
        raise ValueError("Épocas, paciência, threads, unidades e lote precisam ser positivos.")
    if cfg["rapid_weight"] <= 0 or cfg["lr"] <= 0:
        raise ValueError("Peso e taxa de aprendizado precisam ser positivos.")
    return cfg


def make_features(X, feature_mode="absolute"):
    """Todos os canais dependem somente das leituras presentes na entrada."""
    absolute = np.asarray(X, dtype=np.float32)
    if absolute.ndim != 3 or absolute.shape[1:] != (12, 1):
        raise ValueError("Esperadas 12 entradas CGM em X com forma (n, 12, 1).")
    if not np.isfinite(absolute).all():
        raise ValueError("Entradas CGM não finitas.")
    channels = [absolute]
    if feature_mode in ("delta", "combined"):
        channels.append(np.diff(absolute, axis=1, prepend=absolute[:, :1, :]))
    if feature_mode in ("relative", "combined"):
        channels.append(absolute - absolute[:, -1:, :])
    if feature_mode not in ("absolute", "delta", "relative", "combined"):
        raise ValueError(feature_mode)
    return np.concatenate(channels, axis=2)


def _signature(arr):
    digest = hashlib.sha256()
    for name in sorted(arr):
        value = np.ascontiguousarray(arr[name])
        if value.dtype.hasobject:
            raise ValueError("Arrays object não são aceitos nos artefatos de pesquisa.")
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _check_partitions(train, stop, evaluate):
    for name, arr in (("train", train), ("stop", stop), ("evaluate", evaluate)):
        if len(arr["X"]) == 0:
            raise ValueError(f"Partição {name} vazia.")
        make_features(arr["X"])
        if np.asarray(arr["y"]).ndim != 2 or np.asarray(arr["y"]).shape[1] != 2:
            raise ValueError("y deve preservar as duas colunas [5, 15] da v1.")
        if not np.isfinite(arr["y"][:, 1]).all():
            raise ValueError(f"Alvos não finitos em {name}.")
    # A janela inteira de uma partição precede a próxima, para cada pessoa.
    for older, newer in ((train, stop), (stop, evaluate)):
        for subject in set(older["subject"]) & set(newer["subject"]):
            left = older["subject"] == subject
            right = newer["subject"] == subject
            if older["target_ns"][left].max() >= newer["input_start_ns"][right].min():
                raise ValueError(f"Suportes temporais sobrepostos em Subject {subject}.")


def fit_scales(train, feature_mode):
    features = make_features(train["X"], feature_mode)
    target = np.asarray(train["y"][:, 1], dtype=np.float64)
    return dict(x_mean=features.mean(axis=(0, 1), dtype=np.float64).tolist(),
                x_std=np.maximum(features.std(axis=(0, 1), dtype=np.float64), 1e-6).tolist(),
                y_mean=float(target.mean()), y_std=max(float(target.std()), 1e-6))


class RateLSTM(torch.nn.Module):
    def __init__(self, n_features=1, hidden=32):
        super().__init__()
        self.lstm = torch.nn.LSTM(n_features, hidden, batch_first=True)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x):
        states, _ = self.lstm(x)
        return self.head(states[:, -1, :]).squeeze(-1)


def _features_tensor(arr, cfg, scales):
    x = make_features(arr["X"], cfg["feature_mode"])
    # As escalas foram estimadas somente no treino, em precisão dupla.
    x = (x - np.asarray(scales["x_mean"])) / np.asarray(scales["x_std"])
    return torch.from_numpy(x.astype(np.float32))


def _weights_sha(out):
    return core.file_sha(Path(out) / "best_weights.pt")


def predict_variant(arr, out_dir):
    """Prediga sem ajustar escalas, pesos ou limiares."""
    out = Path(out_dir)
    saved = core.read_json(out / "config.json")
    cfg, scales = saved["parameters"], saved["scales"]
    if (out / "completed.json").exists():
        if core.read_json(out / "completed.json")["weights_sha256"] != _weights_sha(out):
            raise ValueError("Pesos modificados após a conclusão.")
    requested = cfg["device"]
    device = torch.device("cuda" if requested == "auto" and torch.cuda.is_available()
                          else "cpu" if requested == "auto" else requested)
    model = RateLSTM(len(scales["x_mean"]), cfg["hidden"])
    model.load_state_dict(torch.load(out / "best_weights.pt", map_location="cpu", weights_only=True))
    model.to(device).eval()
    features = _features_tensor(arr, cfg, scales)
    parts = []
    with torch.no_grad():
        for start in range(0, len(features), 2048):
            parts.append(model(features[start:start + 2048].to(device)).cpu().numpy())
    return np.concatenate(parts) * scales["y_std"] + scales["y_mean"]


def evaluate_variant(arr, out_dir, split="dev"):
    """Retorne (métricas, previsões); a função não decide qual modelo vence."""
    rates = predict_variant(arr, out_dir)
    predictions = core.prediction_frame(arr, HORIZON, "lstm", rates, split)
    return core.metrics_table(predictions), predictions


def train_variant(train, stop, evaluate, config, out_dir):
    """Treine uma configuração e retorne (metrics_df, predictions_df)."""
    return train_shared_trajectory(train, stop, evaluate, [config], [out_dir])[0]


def train_shared_trajectory(train, stop, evaluate, configs, out_dirs):
    """Materialize limites de época/paciência a partir de uma trajetória comum.

    Configurações só podem diferir no limite de épocas, paciência e metadados.
    Cada variante mantém seu próprio histórico e seu checkpoint. Um limite
    menor usa apenas o prefixo que teria sido observado em treino separado.
    A equivalência vale porque não há dropout/scheduler dependente do limite.
    """
    if not configs or len(configs) != len(out_dirs):
        raise ValueError("Forneça uma configuração e uma pasta por variante.")
    configs = [normalize_config(c) for c in configs]
    out_dirs = [Path(p) for p in out_dirs]
    if len(set(p.resolve() for p in out_dirs)) != len(out_dirs):
        raise ValueError("Variantes não podem compartilhar a mesma pasta de saída.")
    first = configs[0]
    if any(any(c[k] != first[k] for k in _TRAIN_FIELDS) for c in configs):
        raise ValueError("Trajetória compartilhada exige mesmos parâmetros de treino e features.")
    _check_partitions(train, stop, evaluate)
    signatures = {name: _signature(arr) for name, arr in
                  (("train", train), ("stop", stop), ("evaluate", evaluate))}
    engine_sha = core.file_sha(__file__)
    results, pending = [None] * len(configs), []
    for index, (cfg, out) in enumerate(zip(configs, out_dirs)):
        if (out / "completed.json").exists():
            saved = core.read_json(out / "config.json")
            if (saved["parameters"] != cfg or saved["data_sha256"] != signatures
                    or saved["engine_sha256"] != engine_sha):
                raise ValueError(f"Artefato concluído incompatível: {out}")
            if _weights_sha(out) != core.read_json(out / "completed.json")["weights_sha256"]:
                raise ValueError(f"Pesos concluídos modificados: {out}")
            results[index] = (pd.read_csv(out / "metrics_evaluation.csv"),
                              pd.read_csv(out / "previsoes_evaluation.csv.gz"))
        else:
            if out.exists() and any(out.iterdir()):
                raise ValueError(f"Execução incompleta ou ocupada; use outra pasta: {out}")
            pending.append(index)
    if not pending:
        return results

    random.seed(first["seed"])
    np.random.seed(first["seed"])
    torch.manual_seed(first["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(first["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(first["threads"])
    requested = first["device"]
    device = torch.device("cuda" if requested == "auto" and torch.cuda.is_available()
                          else "cpu" if requested == "auto" else requested)
    scales = fit_scales(train, first["feature_mode"])
    train_y = np.asarray(train["y"][:, 1])
    weights = np.where(np.abs(train_y) >= 1, first["rapid_weight"], 1).astype(np.float32)
    train_data = TensorDataset(
        _features_tensor(train, first, scales),
        torch.from_numpy(((train_y - scales["y_mean"]) / scales["y_std"]).astype(np.float32)),
        torch.from_numpy(weights))
    stop_data = TensorDataset(
        _features_tensor(stop, first, scales),
        torch.from_numpy(((stop["y"][:, 1] - scales["y_mean"]) / scales["y_std"]).astype(np.float32)))
    loader = DataLoader(train_data, batch_size=first["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(first["seed"]), num_workers=0)
    stop_loader = DataLoader(stop_data, batch_size=1024, shuffle=False, num_workers=0)
    model = RateLSTM(len(scales["x_mean"]), first["hidden"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=first["lr"])
    states = {}
    environment = dict(python=platform.python_version(), torch=torch.__version__,
                       numpy=np.__version__, pandas=pd.__version__, threads=first["threads"],
                       device=str(device), cuda=torch.version.cuda,
                       gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None)
    for index in pending:
        out = out_dirs[index]
        out.mkdir(parents=True, exist_ok=True)
        core.save_json(out / "config.json", dict(parameters=configs[index], scales=scales,
                       data_sha256=signatures, engine_sha256=engine_sha, version=VERSION,
                       train_n=len(train_y), stop_n=len(stop["X"]), evaluation_n=len(evaluate["X"])))
        core.save_json(out / "scales.json", scales)
        core.save_json(out / "ambiente.json", environment)
        (out / "aprimoramento.py").write_bytes(Path(__file__).read_bytes())
        states[index] = dict(best=float("inf"), stale=0, best_epoch=None,
                             epochs_ran=0, finished=False, train_seconds=None)

    history, started = [], time.perf_counter()
    max_epochs = max(configs[i]["max_epochs"] for i in pending)
    for epoch in range(1, max_epochs + 1):
        model.train()
        numerator, denominator = 0., 0.
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            optimizer.zero_grad(set_to_none=True)
            error = (model(xb) - yb).square()
            weighted_sum = (error * wb).sum()
            weight_sum = wb.sum()
            loss = weighted_sum / weight_sum
            if not torch.isfinite(loss):
                raise RuntimeError("Perda de treino não finita.")
            loss.backward()
            optimizer.step()
            numerator += float(weighted_sum.detach().cpu())
            denominator += float(weight_sum.detach().cpu())
        model.eval()
        stop_total = 0.
        with torch.no_grad():
            for xb, yb in stop_loader:
                stop_total += float((model(xb.to(device)) - yb.to(device)).square().sum().cpu())
        stop_loss = stop_total / len(stop_data)
        if not np.isfinite(stop_loss):
            raise RuntimeError("Perda de validação de época não finita.")
        elapsed = time.perf_counter() - started
        history.append(dict(epoch=epoch, train_mse_scaled=numerator / denominator,
                            stop_mse_scaled=stop_loss, elapsed_seconds=elapsed))
        improved = any(not s["finished"] and stop_loss < s["best"] for s in states.values())
        saved_weights = ({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                         if improved else None)
        for index, state in states.items():
            if state["finished"]:
                continue
            out, cfg = out_dirs[index], configs[index]
            if stop_loss < state["best"]:
                state.update(best=stop_loss, stale=0, best_epoch=epoch)
                torch.save(saved_weights, out / "best_weights.pt")
            else:
                state["stale"] += 1
            state["epochs_ran"] = epoch
            pd.DataFrame(history).to_csv(out / "history.csv", index=False)
            if epoch >= cfg["max_epochs"] or state["stale"] >= cfg["patience"]:
                state.update(finished=True, train_seconds=elapsed)
        print(f"{first['feature_mode']} seed={first['seed']} epoch={epoch}/{max_epochs} "
              f"train={numerator / denominator:.6f} stop={stop_loss:.6f}", flush=True)
        if all(s["finished"] for s in states.values()):
            break

    physical_seconds = time.perf_counter() - started
    for index in pending:
        out, cfg, state = out_dirs[index], configs[index], states[index]
        metrics, predictions = evaluate_variant(evaluate, out, cfg["evaluation_split"])
        metrics.to_csv(out / "metrics_evaluation.csv", index=False)
        predictions.to_csv(out / "previsoes_evaluation.csv.gz", index=False)
        core.save_json(out / "completed.json", dict(
            status="completed", version=VERSION, seed=cfg["seed"],
            epochs_ran=state["epochs_ran"], best_epoch=state["best_epoch"],
            best_stop_mse_scaled=state["best"], train_seconds=state["train_seconds"],
            shared_trajectory_seconds=physical_seconds, physical_epochs=len(history),
            shared_trajectory_variants=len(pending),
            checkpoint_selection="unweighted stop MSE; never evaluation metrics",
            weights_sha256=_weights_sha(out), data_sha256=signatures,
            engine_sha256=engine_sha))
        results[index] = (metrics, predictions)
    return results
