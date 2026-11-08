"""Execute uma candidata de +15 min congelada antes de carregar o teste."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.seta_tendencia import azt1d_core as core
from src.seta_tendencia import aprimoramento as engine


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def approval_decision(confirmation, candidate):
    """Aprovação exige decisão final explícita para a candidata selecionada."""
    if (not isinstance(confirmation, dict)
            or confirmation.get("stage") != "confirmation"
            or confirmation.get("selection_is_final") is not True
            or confirmation.get("selected_variant") != candidate):
        return False
    decisions = confirmation.get("decisions", [])
    if not isinstance(decisions, list):
        return False
    matching = [decision for decision in decisions
                if isinstance(decision, dict) and decision.get("variant") == candidate]
    return len(matching) == 1 and matching[0].get("accepted") is True


def find_screening(exp, explicit=None):
    if explicit:
        return explicit.resolve()
    for name in ("triagem", "screening"):
        path = exp / "analise" / name / "acceptance.json"
        if path.is_file():
            return path
    raise FileNotFoundError("Informe --screening-summary ou gere analise/triagem/acceptance.json.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", default="aprimoramento_v2")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--run-name", default="full_v2")
    parser.add_argument("--source-run", default="full_v1",
                        help="Execução v1 completa com caches locais disponíveis.")
    parser.add_argument("--approved", action="store_true",
                        help="Exige que a confirmação tenha aprovado; não concede aprovação.")
    parser.add_argument("--freeze", type=Path,
                        help="Padrão: experiment/plano_final_congelado.json.")
    parser.add_argument("--screening-summary", type=Path)
    args = parser.parse_args()
    for name in (args.experiment_name, args.candidate, args.run_name, args.source_run):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            parser.error("Identificadores devem usar letras, números, hífen ou sublinhado.")
    exp = ROOT / "results/azt1d_seta/experiments" / args.experiment_name
    run = ROOT / "results/azt1d_seta/runs" / args.run_name
    if run.exists() and any(run.iterdir()):
        parser.error("A pasta da execução já contém artefatos; preserve-a e use outro nome.")
    protocol_path = exp / "protocolo_execucao.json"
    confirmation_path = exp / "analise/confirmacao/acceptance.json"
    freeze_path = args.freeze.resolve() if args.freeze else exp / "plano_final_congelado.json"
    screening_path = find_screening(exp, args.screening_summary)
    protocol, confirmation, freeze = read(protocol_path), read(confirmation_path), read(freeze_path)
    if args.candidate not in protocol["variants"]:
        parser.error("A candidata não está pré-registrada no protocolo.")
    frozen_candidate = freeze.get("candidate", freeze.get("variant"))
    if frozen_candidate != args.candidate:
        parser.error("O plano congelado precisa identificar exatamente esta candidata.")
    approved = approval_decision(confirmation, args.candidate)
    if args.approved and not approved:
        parser.error("A confirmação não aprovou a candidata; --approved não pode promovê-la.")
    if "approved" in freeze and freeze["approved"] is not approved:
        parser.error("A decisão no plano congelado diverge da confirmação.")
    variant = protocol["variants"][args.candidate]
    frozen_params = freeze.get("parameters", freeze.get("config"))
    if frozen_params is not None and any(frozen_params.get(k) != v for k, v in variant.items()):
        parser.error("A configuração do plano congelado diverge do protocolo.")
    cfg = engine.normalize_config(dict(variant, seed=42, threads=2, hidden=32,
                                      batch_size=64, lr=.001, variant=args.candidate,
                                      block="original_train60_val20_test20", evaluation_split="test"))
    source_paths = dict(core=Path(core.__file__), engine=Path(engine.__file__), runner=Path(__file__))
    source_bytes = {name: path.read_bytes() for name, path in source_paths.items()}
    source_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in source_bytes.items()}
    for name in ("core", "engine"):
        frozen_source = exp / "codigo" / source_paths[name].name
        if not frozen_source.is_file() or frozen_source.read_bytes() != source_bytes[name]:
            raise ValueError(f"Fonte {name} diferente daquela congelada no desenvolvimento.")

    started, clock_start = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    run.mkdir(parents=True, exist_ok=True)
    code = run / "codigo"
    code.mkdir(exist_ok=True)
    for name, data in source_bytes.items():
        (code / source_paths[name].name).write_bytes(data)
    copied = {"protocol": protocol_path, "confirmation": confirmation_path,
              "screening": screening_path, "freeze_plan": freeze_path}
    for name, path in copied.items():
        (code / f"{name}.json").write_bytes(path.read_bytes())
    selection = dict(candidate=args.candidate, parameters=cfg, seed=42, approved=approved,
                     status="approved_on_development" if approved else "experimental_candidate_not_promoted",
                     chosen_horizon_min=15, exploratory_prior_test_access=True,
                     freeze_utc=datetime.now(timezone.utc).isoformat(),
                     source_sha256={name: core.file_sha(path) for name, path in copied.items()},
                     code_sha256=source_hashes,
                     selection_rule="Decisão de desenvolvimento congelada; teste nunca promove a candidata.")
    # Nenhum NPZ de teste foi carregado antes deste registro persistente.
    core.save_json(run / "selecao_final.json", selection)
    selection_sha = core.file_sha(run / "selecao_final.json")
    execution = dict(run_name=args.run_name, candidate=args.candidate, started_utc=started,
                     status="running", approved=approved, prior_test_access=True,
                     selection_sha256=selection_sha, python=sys.executable,
                     parameters=cfg, code_sha256=source_hashes)
    core.save_json(run / "execucao.json", execution)
    print(f"Executando {args.run_name}: {args.candidate}; aprovado no desenvolvimento={approved}", flush=True)
    try:
        original = ROOT / "results/azt1d_seta/runs" / args.source_run
        original_manifest = core.load_run(original)
        development_manifest = read(exp / "manifest.json")
        def source_identity(manifest):
            return {f["path"].replace("\\", "/"): f["sha256"] for f in manifest["source_files"]}
        if (source_identity(original_manifest) != source_identity(development_manifest)
                or original_manifest["participants"] != development_manifest["participants"]):
            raise ValueError("A execução fonte precisa usar a mesma base e população do desenvolvimento.")
        arrays = {split: core.load_split(original, split) for split in ("train", "val", "test")}
        with np.load(exp / "prepared/B3/train.npz", allow_pickle=False) as cache:
            same_train = engine._signature({k: cache[k] for k in cache.files})
        if engine._signature(arrays["train"]) != same_train:
            raise ValueError("O treino final não coincide com a preparação B3 congelada.")
        fingerprint = hashlib.sha256(json.dumps(dict(data=original_manifest["fingerprint"],
            parameters=cfg, code_sha256=source_hashes), sort_keys=True).encode()).hexdigest()
        manifest = dict(version=engine.VERSION, fingerprint=fingerprint,
                        source_run=args.source_run, source_manifest_sha256=core.file_sha(original / "manifest.json"),
                        data_source_fingerprint=original_manifest["fingerprint"],
                        participants=original_manifest["participants"], excluded=original_manifest["excluded"],
                        counts=original_manifest["counts"], prepared_sha256=original_manifest["prepared_sha256"],
                        prepared_references={s: f"../{args.source_run}/prepared/{s}.npz" for s in arrays},
                        config=dict(quick=False, seed=42, horizon_min=15,
                                    core_sha256=source_hashes["core"], engine_sha256=source_hashes["engine"],
                                    parameters=cfg),
                        exploratory_prior_test_access=True, approved=approved,
                        status=selection["status"], dataset_doi=original_manifest["dataset_doi"])
        core.save_json(run / "manifest.json", manifest)
        with (run / "execucao.log").open("w", encoding="utf-8") as stream:
            with redirect_stdout(stream), redirect_stderr(stream):
                _, test_lstm = engine.train_variant(arrays["train"], arrays["val"], arrays["test"], cfg, run / "model")
                _, val_lstm = engine.evaluate_variant(arrays["val"], run / "model", split="val")
                out = run / "final"
                out.mkdir(exist_ok=True)
                all_metrics = []
                for split, arr, lstm in (("val", arrays["val"], val_lstm), ("test", arrays["test"], test_lstm)):
                    parts = [lstm] + [core.prediction_frame(arr, 15, name, rates, split)
                                     for name, rates in core.baseline_rates(arr["X"]).items()]
                    predictions = pd.concat(parts, ignore_index=True)
                    metrics = core.metrics_table(predictions)
                    suffix = "validacao" if split == "val" else "teste"
                    predictions.to_csv(out / f"previsoes_{suffix}.csv.gz", index=False)
                    metrics.to_csv(out / f"metricas_{suffix}.csv", index=False)
                    all_metrics.append(metrics)
                pd.concat(all_metrics, ignore_index=True).to_csv(out / "metricas.csv", index=False)
        if core.file_sha(run / "selecao_final.json") != selection_sha:
            raise ValueError("Seleção foi alterada durante a execução.")
        core.save_json(run / "final/test_done.json", dict(selection_sha256=selection_sha,
                       weights_sha256=core.file_sha(run / "model/best_weights.pt"),
                       exploratory_prior_test_access=True, approved=approved))
        execution["status"] = "completed"
    except BaseException as exc:
        execution.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        freeze_env = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
        (run / "requirements-freeze.txt").write_text(freeze_env.stdout, encoding="utf-8")
        execution.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                         elapsed_seconds=time.perf_counter() - clock_start,
                         environment_freeze_exit_code=freeze_env.returncode)
        core.save_json(run / "execucao.json", execution)
    print(f"Concluído: {run}", flush=True)


if __name__ == "__main__":
    main()
