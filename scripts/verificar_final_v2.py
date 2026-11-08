"""Verifique uma rodada final v2 registrada como concluída, sem treinar modelos."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.seta_tendencia import azt1d_core as core

STEP = 5 * 60 * 10**9
METRIC_KEYS = ["split", "horizon_min", "model", "subset", "scope", "subject"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_signature(arr):
    digest = hashlib.sha256()
    for name in sorted(arr):
        value = np.ascontiguousarray(arr[name])
        if value.dtype.hasobject:
            raise ValueError("Arrays object não permitidos.")
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def make_features(values, mode):
    values = np.asarray(values, dtype=np.float32)
    pieces = [values]
    if mode in ("delta", "combined"):
        delta = np.zeros_like(values)
        delta[:, 1:, :] = values[:, 1:, :] - values[:, :-1, :]
        pieces.append(delta)
    if mode in ("relative", "combined"):
        pieces.append(values - values[:, -1:, :])
    if mode not in ("absolute", "delta", "relative", "combined"):
        raise ValueError(f"Canais desconhecidos: {mode}")
    return np.concatenate(pieces, axis=2)


class AuditLSTM(torch.nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.lstm = torch.nn.LSTM(channels, hidden, batch_first=True)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x):
        sequence, _ = self.lstm(x)
        return self.head(sequence[:, -1, :]).squeeze(-1)


class Audit:
    def __init__(self):
        self.checks = []

    def check(self, condition, description):
        passed = bool(condition)
        self.checks.append(dict(check=description, passed=passed))
        if not passed:
            raise ValueError(description)

    def close(self, observed, expected, description, rtol=2e-5, atol=2e-6):
        self.check(np.allclose(observed, expected, rtol=rtol, atol=atol, equal_nan=True), description)


def final_approved(confirmation, candidate):
    """Reproduza a condição de aprovação sem depender da função do executor."""
    if (confirmation.get("stage") != "confirmation"
            or confirmation.get("selection_is_final") is not True
            or confirmation.get("selected_variant") != candidate):
        return False
    decisions = confirmation.get("decisions")
    if not isinstance(decisions, list):
        return False
    matches = [item for item in decisions if isinstance(item, dict) and item.get("variant") == candidate]
    return len(matches) == 1 and matches[0].get("accepted") is True


def metric_frame(path):
    return pd.read_csv(path, dtype={"subject": str}).set_index(METRIC_KEYS).sort_index()


def verify(run, execution, audit):
    selection = read(run / "selecao_final.json")
    selection_sha = sha(run / "selecao_final.json")
    test_done = read(run / "final/test_done.json")
    manifest = read(run / "manifest.json")
    config = read(run / "model/config.json")
    done = read(run / "model/completed.json")
    parameters, scales = config["parameters"], config["scales"]
    code_names = {"core": "azt1d_core.py", "engine": "aprimoramento.py", "runner": "executar_final_v2.py"}
    audit.check(selection_sha == execution["selection_sha256"] == test_done["selection_sha256"],
                "Seleção congelada vinculada ao executor e ao teste")
    audit.check(selection["code_sha256"] == execution["code_sha256"],
                "Identificação das fontes consistente entre seleção e execução")
    for name, filename in code_names.items():
        audit.check(sha(run / "codigo" / filename) == selection["code_sha256"][name],
                    f"Fonte {name} arquivada íntegra")
    for name, expected in selection["source_sha256"].items():
        audit.check(sha(run / "codigo" / f"{name}.json") == expected,
                    f"Documento de decisão {name} arquivado íntegro")
    confirmation = read(run / "codigo/confirmation.json")
    protocol = read(run / "codigo/protocol.json")
    freeze = read(run / "codigo/freeze_plan.json")
    candidate = selection["candidate"]
    approval = final_approved(confirmation, candidate)
    audit.check(all(item.get("approved") is approval for item in
                    (selection, execution, manifest, test_done)),
                "Aprovação deriva exclusivamente da confirmação final correspondente")
    expected_status = "approved_on_development" if approval else "experimental_candidate_not_promoted"
    audit.check(selection["status"] == manifest["status"] == expected_status,
                "Estado publicado corresponde à aprovação, sem promoção pelo teste")
    audit.check(freeze.get("candidate", freeze.get("variant")) == candidate == execution["candidate"],
                "Plano congelado e execução identificam a mesma candidata")
    if "approved" in freeze:
        audit.check(freeze["approved"] is approval, "Aprovação do plano congelado consistente")
    variant = protocol["variants"][candidate]
    audit.check(all(parameters[key] == value for key, value in variant.items()),
                "Parâmetros correspondem à variante pré-declarada")
    frozen_params = freeze.get("parameters", freeze.get("config"))
    if frozen_params is not None:
        audit.check(all(frozen_params.get(key) == value for key, value in variant.items()),
                    "Parâmetros do plano congelado correspondem ao protocolo")
    audit.check(parameters == execution["parameters"] == selection["parameters"]
                == manifest["config"]["parameters"], "Parâmetros consistentes em todos os registros")
    audit.check(parameters["seed"] == selection["seed"] == manifest["config"]["seed"] == 42
                and parameters["evaluation_split"] == "test"
                and manifest["config"]["horizon_min"] == selection["chosen_horizon_min"] == 15,
                "Semente 42 e horizonte final exclusivamente +15 min")
    source = ast.parse((run / "codigo/aprimoramento.py").read_text(encoding="utf-8"))
    horizons = [node.value.value for node in source.body if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "HORIZON" for t in node.targets)
                and isinstance(node.value, ast.Constant)]
    audit.check(horizons == [15], "Motor arquivado possui horizonte fixo +15 min")
    audit.check(selection["exploratory_prior_test_access"] is True
                and manifest["exploratory_prior_test_access"] is True
                and execution["prior_test_access"] is True
                and test_done["exploratory_prior_test_access"] is True,
                "Consulta anterior ao teste declarada em todos os registros")
    audit.check(pd.Timestamp(execution["started_utc"]) <= pd.Timestamp(selection["freeze_utc"])
                <= pd.Timestamp(execution["finished_utc"]),
                "Registro de seleção tem horário compatível com a execução")

    runs_root = (ROOT / "results/azt1d_seta/runs").resolve()
    original = (runs_root / manifest["source_run"]).resolve()
    audit.check(original.parent == runs_root and original != run.resolve(),
                "Preparação original fica em outra rodada dentro de runs/")
    original_manifest = read(original / "manifest.json")
    audit.check(sha(original / "manifest.json") == manifest["source_manifest_sha256"]
                and manifest["data_source_fingerprint"] == original_manifest["fingerprint"],
                "Manifesto da preparação original referenciado sem alteração")
    audit.check(manifest["counts"] == original_manifest["counts"]
                and manifest["participants"] == original_manifest["participants"]
                and manifest["excluded"] == original_manifest["excluded"],
                "População, exclusões e contagens originais preservadas")
    audit.check(manifest["config"]["core_sha256"] == original_manifest["config"]["core_sha256"]
                == selection["code_sha256"]["core"], "Mesmo núcleo v1 de preparação e métricas")
    fingerprint = hashlib.sha256(json.dumps(dict(data=original_manifest["fingerprint"],
        parameters=parameters, code_sha256=selection["code_sha256"]), sort_keys=True).encode()).hexdigest()
    audit.check(fingerprint == manifest["fingerprint"], "Fingerprint da execução final recalculado")
    arrays, signatures = {}, {}
    for split in ("train", "val", "test"):
        path = (run / manifest["prepared_references"][split]).resolve()
        audit.check(path == (original / "prepared" / f"{split}.npz").resolve(),
                    f"Referência {split} aponta apenas ao cache original")
        audit.check(sha(path) == manifest["prepared_sha256"][split]
                    == original_manifest["prepared_sha256"][split], f"Cache {split} íntegro")
        with np.load(path, allow_pickle=False) as cache:
            arr = {key: cache[key] for key in cache.files}
        n = len(arr["X"])
        audit.check(n == manifest["counts"][split]["used"]
                    == manifest["counts"][split]["before_sampling"]
                    and arr["X"].shape == (n, 12, 1) and arr["y"].shape == (n, 2),
                    f"Contagens e suporte completo em {split}")
        audit.check(all(np.isfinite(value).all() for value in arr.values())
                    and set(arr["subject"]) == set(manifest["participants"]),
                    f"Arrays {split} finitos e com todas as pessoas")
        audit.check(np.array_equal(arr["target_ns"] - arr["origin_ns"][:, None],
                                  np.tile([STEP, 3 * STEP], (n, 1))),
                    f"Horários-alvo +5/+15 preservados em {split}")
        audit.close(arr["X"][:, -1, 0, None] + arr["y"] * [5, 15], arr["future"],
                    f"Taxa e glicemia futura consistentes em {split}", atol=3e-5)
        signatures[{"train": "train", "val": "stop", "test": "evaluate"}[split]] = array_signature(arr)
        arrays[split] = arr
    for sid in manifest["participants"]:
        bounds = {}
        for split, arr in arrays.items():
            mask = arr["subject"] == sid
            bounds[split] = (arr["input_start_ns"][mask].min(), arr["target_ns"][mask].max())
        audit.check(bounds["train"][1] < bounds["val"][0] and bounds["val"][1] < bounds["test"][0],
                    f"Subject {sid}: suportes de treino/validação/teste separados")
    audit.check(config["data_sha256"] == done["data_sha256"] == signatures,
                "Modelo vinculado às nove chaves de cada NPZ original")
    audit.check(config["engine_sha256"] == done["engine_sha256"]
                == manifest["config"]["engine_sha256"] == selection["code_sha256"]["engine"]
                == sha(run / "model/aprimoramento.py"), "Motor e snapshots do modelo íntegros")
    audit.check(done["status"] == "completed" and sha(run / "model/best_weights.pt")
                == done["weights_sha256"] == test_done["weights_sha256"],
                "Pesos finais concluídos e vinculados ao diagnóstico de teste")
    audit.check(config["train_n"] == len(arrays["train"]["X"])
                and config["stop_n"] == len(arrays["val"]["X"])
                and config["evaluation_n"] == len(arrays["test"]["X"]),
                "Treino, parada e avaliação usam as partições declaradas")
    train_features = make_features(arrays["train"]["X"], parameters["feature_mode"])
    train_y = arrays["train"]["y"][:, 1].astype(np.float64)
    expected_scales = dict(x_mean=train_features.mean(axis=(0, 1), dtype=np.float64),
                          x_std=np.maximum(train_features.std(axis=(0, 1), dtype=np.float64), 1e-6),
                          y_mean=float(train_y.mean()), y_std=max(float(train_y.std()), 1e-6))
    del train_features, train_y
    audit.check(read(run / "model/scales.json") == scales, "Escalas arquivadas consistentes")
    for key, value in expected_scales.items():
        audit.close(scales[key], value, f"{key} calculado somente do treino em float64", rtol=1e-12, atol=1e-12)
    history = pd.read_csv(run / "model/history.csv")
    audit.check(len(history) == done["epochs_ran"] <= parameters["max_epochs"]
                and np.isfinite(history.to_numpy()).all()
                and np.array_equal(history.epoch, np.arange(1, len(history) + 1)),
                "Histórico final completo, finito e dentro do orçamento")
    best, stale, best_epoch, stop_epoch = float("inf"), 0, None, None
    for row in history.itertuples(index=False):
        if row.stop_mse_scaled < best:
            best, stale, best_epoch = row.stop_mse_scaled, 0, int(row.epoch)
        else:
            stale += 1
        if row.epoch >= parameters["max_epochs"] or stale >= parameters["patience"]:
            stop_epoch = int(row.epoch)
            break
    audit.check(stop_epoch == len(history) and best_epoch == done["best_epoch"],
                "Paciência e melhor época verificadas no histórico final")
    audit.close(best, done["best_stop_mse_scaled"], "Checkpoint corresponde à menor perda de validação", atol=1e-12)
    model = AuditLSTM(len(scales["x_mean"]), parameters["hidden"])
    model.load_state_dict(torch.load(run / "model/best_weights.pt", map_location="cpu", weights_only=True))
    model.eval()
    rates = {}
    stop_total = 0.
    for split in ("val", "test"):
        arr = arrays[split]
        inputs = (make_features(arr["X"], parameters["feature_mode"])
                  - np.asarray(scales["x_mean"])) / np.asarray(scales["x_std"])
        tensor = torch.from_numpy(inputs.astype(np.float32))
        del inputs
        target = ((arr["y"][:, 1] - scales["y_mean"]) / scales["y_std"]).astype(np.float32)
        pieces = []
        size = 1024 if split == "val" else 2048
        with torch.no_grad():
            for start in range(0, len(tensor), size):
                output = model(tensor[start:start + size])
                pieces.append(output.numpy())
                if split == "val":
                    stop_total += float((output - torch.from_numpy(target[start:start + size])).square().sum())
        rates[split] = np.concatenate(pieces) * scales["y_std"] + scales["y_mean"]
    audit.close(stop_total / len(arrays["val"]["X"]), done["best_stop_mse_scaled"],
                "MSE original de validação não ponderado reproduzido dos pesos")
    all_metrics = []
    columns = ["subject", "split", "horizon_min", "model", "origin_ns", "target_time",
               "glucose_current", "glucose_true", "rate_true", "rate_pred", "delta_pred",
               "glucose_pred", "angle_deg", "vector_x", "vector_y"]
    for split, suffix in (("val", "validacao"), ("test", "teste")):
        arr = arrays[split]
        pred = pd.read_csv(run / f"final/previsoes_{suffix}.csv.gz", usecols=columns)
        models = {"lstm", "persistencia", "ultima_variacao", "linear_15min"}
        audit.check(set(pred.model) == models and set(pred.split) == {split}
                    and set(pred.horizon_min) == {15} and len(pred) == len(arr["X"]) * 4,
                    f"{split}: quatro modelos +15 min para cada origem")
        audit.check(not pred.duplicated(["subject", "origin_ns", "model"]).any()
                    and np.isfinite(pred.select_dtypes(include="number").to_numpy()).all(),
                    f"{split}: previsões únicas e finitas")
        x = arr["X"][:, :, 0].astype(np.float64)
        expected_rates = {"lstm": rates[split], "persistencia": np.zeros(len(x)),
                          "ultima_variacao": (x[:, -1] - x[:, -2]) / 5,
                          "linear_15min": x[:, -4:] @ np.array([-.06, -.02, .02, .06])}
        for name in sorted(models):
            part = pred[pred.model == name]
            audit.check(len(part) == len(arr["X"]) and np.array_equal(part.subject, arr["subject"])
                        and np.array_equal(part.origin_ns, arr["origin_ns"]),
                        f"{split}/{name}: mesmas pessoas e origens da preparação")
            audit.close(part.rate_pred, expected_rates[name], f"{split}/{name}: taxas reproduzidas independentemente")
            audit.close(part.rate_true, arr["y"][:, 1], f"{split}/{name}: alvo vem de y[:,1]")
            audit.close(part.glucose_current, x[:, -1], f"{split}/{name}: glicemia na origem preservada")
            audit.close(part.glucose_true, arr["future"][:, 1], f"{split}/{name}: alvo glicêmico +15 min preservado")
            audit.check(np.array_equal(pd.to_datetime(part.target_time).astype("int64"), arr["target_ns"][:, 1]),
                        f"{split}/{name}: timestamp do alvo de +15 min")
            audit.close(part.delta_pred, 15 * part.rate_pred, f"{split}/{name}: delta corresponde à taxa")
            audit.close(part.glucose_pred, part.glucose_current + 15 * part.rate_pred,
                        f"{split}/{name}: glicemia prevista consistente com taxa")
            audit.close(part.angle_deg, np.degrees(np.arctan(part.rate_pred)), f"{split}/{name}: ângulo consistente")
            audit.close(part.vector_x, 1 / np.hypot(1, part.rate_pred), f"{split}/{name}: componente x da seta")
            audit.close(part.vector_y, part.rate_pred / np.hypot(1, part.rate_pred), f"{split}/{name}: componente y da seta")
        recalculated = core.metrics_table(pred).set_index(METRIC_KEYS).sort_index()
        saved = metric_frame(run / f"final/metricas_{suffix}.csv")
        audit.check(saved.index.equals(recalculated.index), f"{split}: todas as métricas publicadas têm suporte")
        audit.close(saved.to_numpy(), recalculated[saved.columns].to_numpy(), f"{split}: métricas recalculadas das previsões")
        for name in sorted(models):
            for subset in ("all", "rapid"):
                part = pred[(pred.model == name) & ((pred.rate_true.abs() >= 1) if subset == "rapid" else True)]
                independent = (part.rate_pred - part.rate_true).abs().groupby(part.subject).mean().mean()
                recorded = saved.xs((split, 15, name, subset, "macro", "MEAN")).mae_rate
                audit.close(independent, recorded, f"{split}/{name}: MAE macro {subset} recalculado sem core")
        if split == "test":
            model_pred = pd.read_csv(run / "model/previsoes_evaluation.csv.gz",
                                     usecols=["subject", "origin_ns", "rate_pred"])
            audit.check(np.array_equal(model_pred.subject, arr["subject"])
                        and np.array_equal(model_pred.origin_ns, arr["origin_ns"]),
                        "Previsões diretas do motor têm o mesmo suporte de teste")
            audit.close(model_pred.rate_pred, rates[split], "Previsões diretas do motor iguais às publicadas")
            model_metrics = metric_frame(run / "model/metrics_evaluation.csv")
            final_lstm_metrics = saved[saved.index.get_level_values("model") == "lstm"]
            audit.check(model_metrics.index.equals(final_lstm_metrics.index), "Métricas diretas do motor têm os mesmos escopos")
            audit.close(model_metrics.to_numpy(), final_lstm_metrics[model_metrics.columns].to_numpy(),
                        "Métricas do motor iguais às do relatório final")
        all_metrics.append(recalculated.reset_index())
        del pred
    merged = pd.concat(all_metrics, ignore_index=True).set_index(METRIC_KEYS).sort_index()
    saved = metric_frame(run / "final/metricas.csv")
    audit.check(saved.index.equals(merged.index), "Tabela consolidada contém validação e teste completos")
    audit.close(saved.to_numpy(), merged[saved.columns].to_numpy(), "Tabela consolidada corresponde aos cálculos por partição")
    return dict(candidate=candidate, approved=approval, selection_sha256=selection_sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="full_v2")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        parser.error("Identificador inválido.")
    run = ROOT / "results/azt1d_seta/runs" / args.run_name
    # Este guard antecede toda abertura de manifesto, decisões, NPZ ou previsões.
    if not (run / "execucao.json").is_file():
        parser.error("Rodada final ainda não registrada; nenhum dado de teste foi aberto.")
    execution = read(run / "execucao.json")
    if execution.get("status") != "completed":
        parser.error("Aguarde execucao.json com status completed; nenhum dado de teste foi aberto.")
    torch.set_num_threads(1)
    audit = Audit()
    result = dict(run_name=args.run_name, status="running", validator_sha256=sha(Path(__file__)),
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  guard="Dataset/test predictions opened only after execution status completed")
    failure = None
    try:
        result.update(verify(run, execution, audit))
        result["status"] = "passed"
    except Exception as error:
        failure = error
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        result.update(finished_utc=datetime.now(timezone.utc).isoformat(), checks=audit.checks,
                      checks_passed=sum(item["passed"] for item in audit.checks),
                      checks_failed=sum(not item["passed"] for item in audit.checks))
        output = run / "verificacao.json"
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{result['status'].upper()}: {result['checks_passed']} verificacoes aprovadas; "
              f"{result['checks_failed']} falhas; {output}")
    if failure is not None:
        raise SystemExit(str(failure))


if __name__ == "__main__":
    main()
