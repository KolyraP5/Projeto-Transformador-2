"""Audite blocos e modelos de desenvolvimento concluídos sem treinar novamente."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.seta_tendencia import azt1d_core as core

MODEL_NAME = re.compile(r"^(?P<variant>.+)_(?P<block>B[123])_s(?P<seed>\d+)$")
STEP = 5 * 60 * 10**9


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_signature(arrays):
    """Recalcule a identidade completa sem chamar o motor que está sob auditoria."""
    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        if value.dtype.hasobject:
            raise ValueError("Array object não permitido.")
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(value.shape).encode("utf-8"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def features(X, mode):
    """Reconstrução independente dos canais, sempre a partir de X observado."""
    values = np.asarray(X, dtype=np.float32)
    channels = [values]
    if mode in ("delta", "combined"):
        difference = np.zeros_like(values)
        difference[:, 1:, :] = values[:, 1:, :] - values[:, :-1, :]
        channels.append(difference)
    if mode in ("relative", "combined"):
        channels.append(values - values[:, -1:, :])
    if mode not in ("absolute", "delta", "relative", "combined"):
        raise ValueError(f"Modo desconhecido: {mode}")
    return np.concatenate(channels, axis=2)


class AuditLSTM(torch.nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.lstm = torch.nn.LSTM(channels, hidden, batch_first=True)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, values):
        sequence, _ = self.lstm(values)
        return self.head(sequence[:, -1, :]).squeeze(-1)


class Auditor:
    def __init__(self, experiment):
        self.exp = experiment
        self.checks = []
        self.manifest = read(experiment / "manifest.json")
        self.protocol = read(experiment / "protocolo_execucao.json")
        self.boundaries = pd.read_csv(experiment / "fronteiras.csv")
        self.subject_audit = pd.read_csv(experiment / "auditoria_participantes.csv").set_index("subject")
        self.blocks = {}

    def check(self, condition, scope, description):
        passed = bool(condition)
        self.checks.append(dict(scope=scope, check=description, passed=passed))
        if not passed:
            raise ValueError(f"{scope}: {description}")

    def close(self, actual, expected, scope, description, rtol=1e-6, atol=1e-7):
        self.check(np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True),
                   scope, description)

    def preparation(self):
        manifest = self.manifest
        identity = {key: manifest[key] for key in
                    ("source_files", "protocol", "core_sha256", "preparation_script_sha256")}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.check(digest == manifest["fingerprint"], "preparation", "Fingerprint da preparação íntegro")
        self.check(read(self.exp / "config_preparacao.json") == manifest["protocol"],
                   "preparation", "Protocolo da preparação arquivado sem alteração")
        self.check(manifest["exploratory_prior_test_access"] is True,
                   "preparation", "Natureza exploratória registrada")
        self.check(sha(self.exp / "codigo/azt1d_core.py") == manifest["core_sha256"],
                   "preparation", "Core v1 arquivado íntegro")
        preparation_source = self.exp / "codigo/preparar_aprimoramento.py"
        if not preparation_source.exists():
            preparation_source = ROOT / "scripts/preparar_aprimoramento.py"
        self.check(sha(preparation_source) == manifest["preparation_script_sha256"],
                   "preparation", "Fonte da preparação confere com manifesto")
        raw = Path(os.environ.get("AZT1D_LOCAL_DATA", ROOT / "AZT1D 2025/CGM Records"))
        self.check(all((raw / source["path"]).is_file()
                       and (raw / source["path"]).stat().st_size == source["bytes"]
                       and sha(raw / source["path"]) == source["sha256"]
                       for source in manifest["source_files"]),
                   "preparation", "Bytes dos CSVs originais conferidos, sem extrair alvos do teste")
        self.check(set(manifest["participants"]) == set(range(1, 26)) - {14},
                   "preparation", "População e exclusão do alias CGM/BGM preservadas")

    def block(self, block):
        if block in self.blocks:
            return self.blocks[block]
        manifest = self.manifest
        arrays, signatures = {}, {}
        limits = self.boundaries[self.boundaries.block == block]
        participants = set(manifest["participants"])
        self.check(not limits.duplicated(["subject", "segment"]).any()
                   and len(limits) == len(participants) * 3,
                   block, "Uma fronteira por pessoa e segmento")
        for segment in ("train", "stop", "evaluation"):
            relative = f"prepared/{block}/{segment}.npz"
            path = self.exp / relative
            self.check(path.is_file() and sha(path) == manifest["prepared_sha256"][relative],
                       f"{block}/{segment}", "NPZ íntegro")
            with np.load(path, allow_pickle=False) as cache:
                arr = {key: cache[key] for key in cache.files}
            scope = f"{block}/{segment}"
            n = len(arr["X"])
            expected_keys = {"X", "y", "future", "subject", "origin_ns", "target_ns",
                             "input_start_ns", "raw_origin_ns", "raw_target_ns"}
            self.check(set(arr) == expected_keys, scope, "Todas as nove chaves da preparação presentes")
            self.check(n == manifest["counts"][block][segment] > 0,
                       scope, "Contagem de janelas confere")
            self.check(arr["X"].shape == (n, 12, 1) and arr["y"].shape == (n, 2)
                       and arr["future"].shape == (n, 2), scope, "Formas da entrada e dos dois horizontes corretas")
            self.check(all(np.isfinite(value).all() for value in arr.values()),
                       scope, "Todos os arrays finitos")
            self.check(set(arr["subject"]) == participants, scope, "Mesmas pessoas nas três partições")
            self.check(np.array_equal(arr["target_ns"] - arr["origin_ns"][:, None],
                                      np.tile([STEP, 3 * STEP], (n, 1)))
                       and np.array_equal(arr["origin_ns"] - arr["input_start_ns"],
                                          np.full(n, 11 * STEP)),
                       scope, "Entrada de 12 pontos e alvos +5/+15 conferidos")
            self.close(arr["X"][:, -1, 0, None] + arr["y"] * [5, 15], arr["future"],
                       scope, "Taxas correspondem às glicemias futuras", atol=3e-5)
            self.check(np.all(arr["raw_origin_ns"] <= arr["origin_ns"])
                       and np.all(arr["raw_target_ns"] > arr["origin_ns"][:, None])
                       and np.all(arr["raw_target_ns"] <= arr["target_ns"]),
                       scope, "Leituras disponíveis na origem e alvos estritamente futuros")
            self.check(not pd.DataFrame(dict(subject=arr["subject"], origin=arr["origin_ns"])).duplicated().any(),
                       scope, "Origens únicas por pessoa")
            for sid in sorted(participants):
                boundary = limits[(limits.subject == sid) & (limits.segment == segment)].iloc[0]
                total = int(boundary.full_grid_rows)
                lo, hi = manifest["protocol"]["blocks"][block][segment]
                mask = arr["subject"] == sid
                grid_start = pd.Timestamp(self.subject_audit.loc[sid, "start"]).value
                self.check(int(boundary.start_grid_index) == int(total * lo)
                           and int(boundary.stop_grid_index_exclusive) == int(total * hi)
                           and int(boundary.stop_grid_index_exclusive) <= int(total * .8),
                           scope, f"Subject {sid}: fronteira nominal fica nos primeiros 80%")
                self.check(pd.Timestamp(boundary.start).value == grid_start + int(total * lo) * STEP
                           and pd.Timestamp(boundary.end).value == grid_start + (int(total * hi) - 1) * STEP
                           and int(arr["input_start_ns"][mask].min()) >= pd.Timestamp(boundary.start).value
                           and int(arr["target_ns"][mask].max()) <= pd.Timestamp(boundary.end).value
                           and int(arr["target_ns"][mask].max()) <= grid_start + (int(total * .8) - 1) * STEP
                           and int(mask.sum()) == int(boundary.n_windows),
                           scope, f"Subject {sid}: janelas respeitam a fronteira e não invadem teste")
            arrays[segment] = arr
            signatures["evaluate" if segment == "evaluation" else segment] = array_signature(arr)
        for sid in sorted(participants):
            support = {}
            for segment, arr in arrays.items():
                mask = arr["subject"] == sid
                support[segment] = (arr["input_start_ns"][mask].min(), arr["target_ns"][mask].max())
            self.check(support["train"][1] < support["stop"][0]
                       and support["stop"][1] < support["evaluation"][0],
                       block, f"Subject {sid}: nenhum suporte compartilhado entre partições")
        self.blocks[block] = (arrays, signatures)
        return arrays, signatures

    def model(self, variant, block, seed):
        name = f"{variant}_{block}_s{seed}"
        out = self.exp / "models" / name
        cfg, done = read(out / "config.json"), read(out / "completed.json")
        parameters, scales = cfg["parameters"], cfg["scales"]
        arrays, signatures = self.block(block)
        self.check(done["status"] == "completed", name, "Treino concluído")
        self.check(parameters["variant"] == variant and parameters["block"] == block
                   and parameters["seed"] == seed == done["seed"], name, "Identificação e semente conferidas")
        self.check(all(parameters[key] == value for key, value in self.protocol["variants"][variant].items()),
                   name, "Variante executada corresponde à pré-declarada")
        self.check(parameters["preparation_fingerprint"] == self.manifest["fingerprint"]
                   and cfg["data_sha256"] == signatures == done["data_sha256"],
                   name, "Assinaturas completas das três partições correspondem ao treino")
        self.check(cfg["engine_sha256"] == done["engine_sha256"] == sha(out / "aprimoramento.py")
                   == sha(self.exp / "codigo/aprimoramento.py"),
                   name, "Motor arquivado corresponde aos bytes usados no treino")
        self.check(sha(out / "best_weights.pt") == done["weights_sha256"],
                   name, "Checkpoint íntegro")
        x = features(arrays["train"]["X"], parameters["feature_mode"])
        y = arrays["train"]["y"][:, 1].astype(np.float64)
        expected = dict(x_mean=x.mean(axis=(0, 1), dtype=np.float64).tolist(),
                        x_std=np.maximum(x.std(axis=(0, 1), dtype=np.float64), 1e-6).tolist(),
                        y_mean=float(y.mean()), y_std=max(float(y.std()), 1e-6))
        del x, y
        self.check(read(out / "scales.json") == scales, name, "Escalas arquivadas consistentes")
        for key in expected:
            self.close(scales[key], expected[key], name,
                       f"{key} recalculado somente do treino em float64", rtol=1e-12, atol=1e-12)

        history = pd.read_csv(out / "history.csv")
        self.check(len(history) == done["epochs_ran"] <= parameters["max_epochs"]
                   and np.array_equal(history.epoch, np.arange(1, len(history) + 1))
                   and np.isfinite(history.to_numpy()).all(), name, "Histórico completo, finito e dentro do orçamento")
        best, stale, best_epoch, stop_epoch = float("inf"), 0, None, None
        for row in history.itertuples(index=False):
            if row.stop_mse_scaled < best:
                best, stale, best_epoch = row.stop_mse_scaled, 0, int(row.epoch)
            else:
                stale += 1
            if row.epoch >= parameters["max_epochs"] or stale >= parameters["patience"]:
                stop_epoch = int(row.epoch)
                break
        self.check(stop_epoch == len(history) and best_epoch == done["best_epoch"],
                   name, "Paciência, limite e melhor época conferidos por replay do histórico")
        self.close(best, done["best_stop_mse_scaled"], name, "Melhor perda corresponde ao prefixo próprio", atol=1e-12)
        self.check(done["physical_epochs"] >= len(history)
                   and "unweighted stop MSE" in done["checkpoint_selection"],
                   name, "Trajetória compartilhada não prolongou artificialmente a variante")
        self.close(history.elapsed_seconds.iloc[-1], done["train_seconds"], name,
                   "Custo da variante corresponde ao seu prefixo", atol=1e-6)
        model = AuditLSTM(len(scales["x_mean"]), parameters["hidden"])
        model.load_state_dict(torch.load(out / "best_weights.pt", map_location="cpu", weights_only=True))
        model.eval()
        computed = {}
        for segment in ("stop", "evaluation"):
            arr = arrays[segment]
            raw_x = features(arr["X"], parameters["feature_mode"])
            norm = (raw_x - np.asarray(scales["x_mean"])) / np.asarray(scales["x_std"])
            tensor = torch.from_numpy(norm.astype(np.float32))
            parts = []
            with torch.no_grad():
                for index in range(0, len(tensor), 1024 if segment == "stop" else 2048):
                    parts.append(model(tensor[index:index + (1024 if segment == "stop" else 2048)]).numpy())
            computed[segment] = np.concatenate(parts)
        # O alvo supervisionado pondera treino, mas não a escolha de checkpoint.
        target = ((arrays["stop"]["y"][:, 1] - scales["y_mean"]) / scales["y_std"]).astype(np.float32)
        mse = sum(float(torch.from_numpy((computed["stop"][i:i + 1024] - target[i:i + 1024]) ** 2).sum())
                  for i in range(0, len(target), 1024)) / len(target)
        self.close(mse, done["best_stop_mse_scaled"], name,
                   "MSE de stop não ponderado recalculado do checkpoint", rtol=2e-5, atol=2e-6)

        pred = pd.read_csv(out / "previsoes_evaluation.csv.gz")
        arr = arrays["evaluation"]
        rates = computed["evaluation"] * scales["y_std"] + scales["y_mean"]
        self.check(len(pred) == len(arr["X"]) and set(pred.model) == {"lstm"}
                   and set(pred.horizon_min) == {15} and set(pred.split) == {parameters["evaluation_split"]},
                   name, "Previsões de desenvolvimento completas e apenas +15 min")
        self.check(np.array_equal(pred.subject, arr["subject"])
                   and np.array_equal(pred.origin_ns, arr["origin_ns"])
                   and not pred.duplicated(["subject", "origin_ns"]).any(),
                   name, "Mesmas pessoas e origens, sem duplicação ou escolha de suporte")
        self.check(np.isfinite(pred.select_dtypes(include="number").to_numpy()).all(),
                   name, "Todas as previsões e vetores finitos")
        self.close(pred.rate_pred, rates, name, "Previsões reproduzidas do checkpoint", rtol=2e-5, atol=2e-6)
        self.close(pred.rate_true, arr["y"][:, 1], name, "Alvo de +15 min corresponde à coluna y[:,1]", atol=1e-6)
        self.close(pred.glucose_true, arr["future"][:, 1], name, "Glicemia futura de +15 min preservada")
        self.close(pred.glucose_current, arr["X"][:, -1, 0], name, "Glicemia da origem preservada")
        self.check(np.array_equal(pd.to_datetime(pred.target_time).astype("int64"), arr["target_ns"][:, 1]),
                   name, "Timestamp do alvo é o de +15 min")
        self.close(pred.glucose_pred, pred.glucose_current + 15 * pred.rate_pred,
                   name, "Taxa e glicemia prevista são consistentes")
        self.close(pred.delta_pred, 15 * pred.rate_pred, name, "Delta previsto consistente com horizonte")
        self.close(pred.angle_deg, np.degrees(np.arctan(pred.rate_pred)), name, "Ângulo da seta consistente")
        self.close(pred.vector_x ** 2 + pred.vector_y ** 2, np.ones(len(pred)),
                   name, "Vetor da seta é unitário")
        keys = ["split", "horizon_min", "model", "subset", "scope", "subject"]
        saved = pd.read_csv(out / "metrics_evaluation.csv", dtype={"subject": str}).set_index(keys).sort_index()
        recalculated = core.metrics_table(pred).set_index(keys).sort_index()
        self.check(saved.index.equals(recalculated.index), name, "Métricas publicadas cobrem todos os escopos")
        self.close(saved.to_numpy(), recalculated[saved.columns].to_numpy(), name,
                   "Todas as métricas recalculadas das previsões", rtol=2e-5, atol=2e-6)
        for subset, part in (("all", pred), ("rapid", pred[pred.rate_true.abs() >= 1])):
            own = (part.rate_pred - part.rate_true).abs().groupby(part.subject).mean().mean()
            recorded = saved.xs((parameters["evaluation_split"], 15, "lstm", subset, "macro", "MEAN")).mae_rate
            self.close(own, recorded, name, f"MAE macro {subset} recalculado sem o agregador do core", atol=1e-6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", default="aprimoramento_v2")
    parser.add_argument("--blocks", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--report-name", default="verificacao")
    args = parser.parse_args()
    report_name = args.report_name.removesuffix(".json")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", name)
           for name in (args.experiment_name, report_name)):
        parser.error("Identificadores inválidos.")
    exp = ROOT / "results/azt1d_seta/experiments" / args.experiment_name
    if not (exp / "manifest.json").is_file():
        parser.error("Experimento não preparado.")
    auditor = Auditor(exp)
    available = []
    for path in sorted((exp / "models").glob("*")):
        match = MODEL_NAME.fullmatch(path.name)
        if match and (path / "completed.json").is_file():
            row = (match["variant"], match["block"], int(match["seed"]))
            if (not args.blocks or row[1] in args.blocks) and (not args.seeds or row[2] in args.seeds):
                available.append(row)
    if args.variants:
        blocks = args.blocks or list(auditor.manifest["protocol"]["blocks"])
        seeds = args.seeds or auditor.manifest["protocol"]["seeds"]
        requested = [(variant, block, seed) for variant in args.variants for block in blocks for seed in seeds]
        absent = [f"{v}_{b}_s{s}" for v, b, s in requested if (v, b, s) not in available]
        if absent:
            parser.error("Variantes especificadas incompletas/inexistentes: " + ", ".join(absent))
    else:
        requested = available
    if not requested:
        parser.error("Nenhum modelo concluído corresponde aos filtros.")
    torch.set_num_threads(1)
    result = dict(experiment_name=args.experiment_name, status="running", arguments=vars(args),
                  models=[f"{v}_{b}_s{s}" for v, b, s in requested],
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  validator_sha256=sha(Path(__file__)),
                  scope="development blocks only; no final_v2 or current test predictions accessed")
    failure = None
    try:
        auditor.preparation()
        for variant, block, seed in requested:
            if variant not in auditor.protocol["variants"]:
                raise ValueError(f"Variante não pré-registrada: {variant}")
            auditor.model(variant, block, seed)
        result["status"] = "passed"
    except Exception as error:
        failure = error
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        result.update(finished_utc=datetime.now(timezone.utc).isoformat(), checks=auditor.checks,
                      checks_passed=sum(item["passed"] for item in auditor.checks),
                      checks_failed=sum(not item["passed"] for item in auditor.checks))
        output = exp / "verificacoes" / f"{report_name}.json"
        output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{result['status'].upper()}: {result['checks_passed']} verificacoes aprovadas; "
              f"{result['checks_failed']} falhas; {len(requested)} modelos; {output}")
    if failure is not None:
        raise SystemExit(str(failure))


if __name__ == "__main__":
    main()
