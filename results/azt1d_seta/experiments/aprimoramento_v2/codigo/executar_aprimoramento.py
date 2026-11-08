"""Execute uma etapa pré-registrada das melhorias de +15 min, sem acessar o teste."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(task):
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from src.seta_tendencia.aprimoramento import train_shared_trajectory
    started = time.perf_counter()
    exp = Path(task["experiment"])
    arrays = {}
    for split in ("train", "stop", "evaluation"):
        with np.load(exp / "prepared" / task["block"] / f"{split}.npz", allow_pickle=False) as cache:
            arrays[split] = {key: cache[key] for key in cache.files}
    label = "+".join(task["variants"]) + f"_{task['block']}_s{task['seed']}"
    logs = exp / "logs"
    logs.mkdir(exist_ok=True)
    with (logs / f"{label}.log").open("w", encoding="utf-8") as stream:
        with redirect_stdout(stream), redirect_stderr(stream):
            outputs = [exp / "models" / f"{v}_{task['block']}_s{task['seed']}"
                       for v in task["variants"]]
            train_shared_trajectory(arrays["train"], arrays["stop"], arrays["evaluation"],
                                    task["configs"], outputs)
    return dict(task=label, seconds=time.perf_counter() - started, variants=task["variants"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", default="aprimoramento_v2")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--blocks", nargs="+", default=["B3"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if any(not name.replace("_", "").isalnum() for name in (args.experiment_name, args.stage)):
        parser.error("Identificadores inválidos.")
    exp = ROOT / "results/azt1d_seta/experiments" / args.experiment_name
    protocol = json.loads((exp / "protocolo_execucao.json").read_text(encoding="utf-8"))
    manifest = json.loads((exp / "manifest.json").read_text(encoding="utf-8"))
    for variant in args.variants:
        if variant not in protocol["variants"]:
            parser.error(f"Variante não pré-registrada: {variant}")
    for block in args.blocks:
        if block not in ("B1", "B2", "B3"):
            parser.error("Bloco inválido.")
    stage_dir = exp / "stages"
    stage_dir.mkdir(exist_ok=True)
    path = stage_dir / f"{args.stage}.json"
    if path.exists():
        parser.error("Etapa já registrada. Preserve o registro e use outro --stage.")
    tasks = []
    for block in args.blocks:
        for seed in args.seeds:
            grouped = [v for v in args.variants if v in ("baseline15", "ep30", "pat5")]
            groups = ([grouped] if grouped else []) + [[v] for v in args.variants if v not in grouped]
            for group in groups:
                configs = [dict(protocol["variants"][v], seed=seed, threads=args.threads,
                                hidden=32, batch_size=64, lr=.001, variant=v, block=block,
                                preparation_fingerprint=manifest["fingerprint"])
                           for v in group]
                tasks.append(dict(experiment=str(exp), block=block, seed=seed,
                                  variants=group, configs=configs))
    metadata = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                    arguments=vars(args), tasks=tasks, completed_tasks=[],
                    protocol_sha256=digest(exp / "protocolo_execucao.json"),
                    engine_sha256=digest(ROOT / "src/seta_tendencia/aprimoramento.py"),
                    script_sha256=digest(Path(__file__)))
    source = exp / "codigo"
    source.mkdir(exist_ok=True)
    for file in (Path(__file__), ROOT / "src/seta_tendencia/aprimoramento.py",
                 ROOT / "src/seta_tendencia/azt1d_core.py"):
        target = source / file.name
        if target.exists() and target.read_bytes() != file.read_bytes():
            raise RuntimeError(f"Fonte já congelada e diferente: {target}")
        target.write_bytes(file.read_bytes())
    def save():
        path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    save()
    print(f"Etapa {args.stage}: {len(tasks)} trajetorias, {len(args.variants)} variantes; "
          f"{args.workers} processos x {args.threads} threads", flush=True)
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(worker, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                metadata["completed_tasks"].append(result)
                save()
                print(f"Concluido {len(metadata['completed_tasks'])}/{len(tasks)}: "
                      f"{result['task']} ({result['seconds']:.1f}s)", flush=True)
        metadata["status"] = "completed"
    except BaseException as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
