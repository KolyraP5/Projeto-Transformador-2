"""Prepare blocos causais de desenvolvimento, preservando o núcleo v1."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.seta_tendencia import azt1d_core as core


def segment_windows(part, subject):
    """Crie janelas dentro de um único segmento, sem dividir novamente."""
    if len(part) < 15:
        raise ValueError(f"Subject {subject}: segmento menor que o suporte.")
    g = part.glucose.to_numpy(dtype=np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(g, 15)
    starts = np.flatnonzero(np.isfinite(windows).all(axis=1))
    if not len(starts):
        raise ValueError(f"Subject {subject}: segmento sem janelas completas.")
    w = windows[starts]
    i = starts + 11
    times = part.index.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    raw = part.timestamp_raw.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    origin = times[i]
    targets = times[i[:, None] + [1, 3]]
    future = w[:, [12, 14]].copy()
    x = w[:, :12, None].copy()
    y = ((future - x[:, -1, 0, None]) / np.array([5., 15.], dtype=np.float32)).astype(np.float32)
    arr = dict(X=x, y=y, future=future,
               subject=np.full(len(i), subject, dtype=np.int32),
               origin_ns=origin, target_ns=targets, input_start_ns=times[starts],
               raw_origin_ns=raw[i], raw_target_ns=raw[i[:, None] + [1, 3]])
    if not (np.all(arr["input_start_ns"] >= times[0]) and
            np.all(targets <= times[-1]) and np.all(arr["raw_origin_ns"] <= origin) and
            np.all(arr["raw_target_ns"] > origin[:, None])):
        raise ValueError("Suporte temporal inválido na preparação.")
    np.testing.assert_array_equal(targets - origin[:, None],
                                  np.tile(np.array([1, 3]) * core.STEP_NS, (len(i), 1)))
    np.testing.assert_allclose(x[:, -1, 0, None] + y * [5, 15], future, atol=3e-5)
    return arr


def prepare(experiment_name, data_root, config_path):
    started = time.perf_counter()
    config = core.read_json(config_path)
    files = core.discover_csvs(data_root)
    source = [dict(path=p.relative_to(data_root).as_posix(), sha256=core.file_sha(p),
                   bytes=p.stat().st_size) for p in files]
    identity = dict(source_files=source, protocol=config,
                    core_sha256=core.file_sha(Path(core.__file__)),
                    preparation_script_sha256=core.file_sha(Path(__file__)))
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    experiment = ROOT / "results/azt1d_seta/experiments" / experiment_name
    manifest_path = experiment / "manifest.json"
    prepared = experiment / "prepared"
    if manifest_path.exists():
        old = core.read_json(manifest_path)
        if old["fingerprint"] != fingerprint:
            raise ValueError("Preparação existente com outra identidade; use outro experiment-name.")
        for relative, expected in old["prepared_sha256"].items():
            path = experiment / relative
            if not path.exists() or core.file_sha(path) != expected:
                raise ValueError(f"Cache incompleto ou alterado: {path}")
        print("Preparação existente verificada:", manifest_path)
        return old
    if prepared.exists() and any(prepared.iterdir()):
        raise ValueError("Preparação parcial já existe; preserve-a e use outro identificador.")

    frames, excluded, file_audit = {}, [], []
    for path in files:
        df = core.read_csv_flexible(path)
        sid = core.subject_number(path)
        names = {c.lower(): c for c in df.columns}
        date, glucose, mixed = names.get("eventdatetime"), names.get("cgm"), names.get("readings (cgm / bgm)")
        if not date or not (glucose or mixed):
            excluded.append(dict(subject=sid, reason="colunas_ausentes"))
            continue
        if not glucose:
            excluded.append(dict(subject=sid, reason="alias_CGM_BGM_nao_confirmado"))
            file_audit.append(dict(subject=sid, file=path.relative_to(data_root).as_posix(), status="alias_excluido"))
            continue
        df = df.rename(columns={date: "EventDateTime", glucose: "CGM"})
        frames.setdefault(sid, []).append(df)
        file_audit.append(dict(subject=sid, file=path.relative_to(data_root).as_posix(), status="lido", rows=len(df)))
    grids, subject_audit = {}, []
    cfg = config["preparation"]
    for sid in sorted(frames):
        grids[sid], audit = core.clean_subject(frames[sid], sid, cfg["dayfirst"], cfg["date_format"])
        subject_audit.append(audit)
    del frames
    if set(grids) != set(range(1, 26)) - {14}:
        raise ValueError(f"População diferente da rodada v1: {sorted(grids)}")

    experiment.mkdir(parents=True, exist_ok=True)
    counts, hashes, boundaries, temporal_checks = {}, {}, [], []
    for block, segments in config["blocks"].items():
        counts[block] = {}
        prior_end = {}
        for name, (lo_fraction, hi_fraction) in segments.items():
            if not (0 <= lo_fraction < hi_fraction <= .8):
                raise ValueError("Bloco invade o teste ou contém limites inválidos.")
            collections = []
            for sid, grid in grids.items():
                lo, hi = int(len(grid) * lo_fraction), int(len(grid) * hi_fraction)
                part = grid.iloc[lo:hi]
                arr = segment_windows(part, sid)
                if sid in prior_end and not prior_end[sid] < int(arr["input_start_ns"].min()):
                    raise ValueError(f"Suporte compartilhado entre segmentos: {block}/{sid}")
                prior_end[sid] = int(arr["target_ns"].max())
                collections.append(arr)
                boundaries.append(dict(block=block, segment=name, subject=sid,
                                       start_grid_index=lo, stop_grid_index_exclusive=hi,
                                       start=str(part.index[0]), end=str(part.index[-1]),
                                       n_windows=len(arr["X"]), full_grid_rows=len(grid)))
            arrays = {key: np.concatenate([a[key] for a in collections]) for key in collections[0]}
            path = prepared / block / f"{name}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **arrays)
            relative = path.relative_to(experiment).as_posix()
            hashes[relative] = core.file_sha(path)
            counts[block][name] = len(arrays["X"])
            temporal_checks.append(dict(block=block, segment=name, finite_inputs=True,
                                        finite_targets=True, no_cross_segment_support=True,
                                        no_current_test_windows=True, participants=len(grids)))
    pd.DataFrame(boundaries).to_csv(experiment / "fronteiras.csv", index=False)
    pd.DataFrame(subject_audit).to_csv(experiment / "auditoria_participantes.csv", index=False)
    pd.DataFrame(file_audit).to_csv(experiment / "auditoria_arquivos.csv", index=False)
    (experiment / "config_preparacao.json").write_bytes(config_path.read_bytes())
    manifest = dict(fingerprint=fingerprint, **identity, participants=sorted(grids), excluded=excluded,
                    counts=counts, prepared_sha256=hashes, temporal_checks=temporal_checks,
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    preparation_seconds=time.perf_counter() - started,
                    dataset_doi="10.17632/gk9m674wcx.1", exploratory_prior_test_access=True,
                    test_policy="Nenhuma janela de teste participa da comparação; leitura bruta apenas para grade original.")
    core.save_json(manifest_path, manifest)
    print("Preparação concluída:", manifest_path)
    print(json.dumps(counts, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", default="aprimoramento_v2")
    parser.add_argument("--data-root", type=Path,
                        default=Path(os.environ.get("AZT1D_LOCAL_DATA", ROOT / "AZT1D 2025/CGM Records")))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/aprimoramento_v2.json")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.experiment_name):
        parser.error("Nome de experimento inválido.")
    prepare(args.experiment_name, args.data_root.resolve(), args.config.resolve())


if __name__ == "__main__":
    main()
