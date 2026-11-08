"""Verifique os artefatos reais de uma execução da seta, sem treinar novamente."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(run):
    checks = []
    def check(condition, description):
        if not condition:
            raise ValueError(description)
        checks.append(description)

    manifest = read(run / "manifest.json")
    execution = read(run / "execucao.json")
    check(execution["status"] == "completed", "Executor e notebook concluídos")
    check(sha(run / "codigo/azt1d_core.py") == manifest["config"]["core_sha256"],
          "Código arquivado idêntico ao usado na preparação")
    check(sha(run / "codigo/notebook_fonte.ipynb") == execution["notebook_sha256"],
          "Notebook fonte arquivado íntegro")
    notebook = read(run / "notebook_executado.ipynb")
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    check(all(c["execution_count"] is not None and
              not any(o["output_type"] == "error" for o in c["outputs"])
              for c in code_cells), "Todas as células executadas sem erro")

    boundaries = {}
    for split in ("train", "val", "test"):
        path = run / f"prepared/{split}.npz"
        check(sha(path) == manifest["prepared_sha256"][split], f"Cache {split} íntegro")
        with np.load(path, allow_pickle=False) as arr:
            n = len(arr["X"])
            check(n == manifest["counts"][split]["used"], f"Contagens de {split} conferidas")
            check(arr["X"].shape == (n, 12, 1) and np.isfinite(arr["X"]).all()
                  and np.isfinite(arr["y"]).all(), f"Suporte e valores finitos em {split}")
            np.testing.assert_array_equal(arr["target_ns"] - arr["origin_ns"][:, None],
                                          np.tile([5 * 60 * 10**9, 15 * 60 * 10**9], (n, 1)))
            np.testing.assert_allclose(arr["X"][:, -1, 0, None] + arr["y"] * [5, 15],
                                       arr["future"], atol=3e-5)
            check(np.all(arr["raw_origin_ns"] <= arr["origin_ns"])
                  and np.all(arr["raw_target_ns"] > arr["origin_ns"][:, None]),
                  f"Leituras disponíveis na origem e alvos futuros em {split}")
            check(set(arr["subject"].tolist()) == set(manifest["participants"]),
                  f"Participantes declarados presentes em {split}")
            for sid in manifest["participants"]:
                mask = arr["subject"] == sid
                boundaries[sid, split] = (int(arr["input_start_ns"][mask].min()),
                                          int(arr["target_ns"][mask].max()))
            if split == "train":
                # Redução independente em float64 evita acumular arredondamento
                # ao somar centenas de milhares de alvos float32 por coluna.
                scales = (float(arr["X"].mean(dtype=np.float64)),
                          float(arr["X"].std(dtype=np.float64)),
                          arr["y"].mean(axis=0, dtype=np.float64),
                          arr["y"].std(axis=0, dtype=np.float64))
        if not manifest["config"]["quick"]:
            check(manifest["counts"][split]["before_sampling"] == n,
                  f"Sem amostragem reduzida em {split}")
    check(all(boundaries[s, "train"][1] < boundaries[s, "val"][0]
              and boundaries[s, "val"][1] < boundaries[s, "test"][0]
              for s in manifest["participants"]),
          "Nenhum suporte de entrada/alvo compartilhado entre partições por pessoa")

    selection = read(run / "selecao_horizonte.json")
    check(selection["exploratory_prior_test_access"] is True, "Acesso prévio ao teste declarado")
    for j, h in enumerate((5, 15)):
        out = run / f"lstm_h{h}"
        done, config = read(out / "completed.json"), read(out / "config.json")
        check(sha(out / "best_weights.pt") == done["weights_sha256"]
              == selection["model_hashes"][str(h)], f"Pesos h{h} íntegros e congelados")
        actual = config["scales"]
        np.testing.assert_allclose([actual["x_mean"], actual["x_std"],
                                    actual["y_mean"], actual["y_std"]],
                                   [scales[0], scales[1], scales[2][j], scales[3][j]],
                                   rtol=2e-5, atol=2e-5)
        check(done["fingerprint"] == manifest["fingerprint"], f"Treino h{h} vinculado aos dados")
        history = pd.read_csv(out / "historico.csv")
        check(len(history) == done["epochs_ran"] <= config["epochs"]
              and np.isclose(history.val_mse_scaled.min(), done["best_val_mse_scaled"]),
              f"Histórico e melhor estado h{h} conferidos")
    check(read(run / "final/test_done.json")["selection_sha256"]
          == sha(run / "selecao_horizonte.json"), "Teste vinculado à seleção congelada")

    metrics = pd.read_csv(run / "final/metricas.csv")
    for split, paths in (
        ("val", [run / "baselines/previsoes_validacao.csv.gz",
                 run / "lstm_h5/previsoes_validacao.csv.gz", run / "lstm_h15/previsoes_validacao.csv.gz"]),
        ("test", [run / "final/previsoes_teste.csv.gz"]),
    ):
        # Carrega só as colunas necessárias para manter baixo o uso de memória.
        cols = ["subject", "horizon_min", "model", "origin_ns", "rate_true", "rate_pred"]
        pred = pd.concat([pd.read_csv(p, usecols=cols) for p in paths], ignore_index=True)
        n = manifest["counts"][split]["used"]
        check(len(pred) == n * 8 and not pred.duplicated(
              ["subject", "horizon_min", "model", "origin_ns"]).any(),
              f"Oito previsões únicas por origem em {split}")
        check(np.isfinite(pred[["rate_true", "rate_pred"]]).all().all(),
              f"Previsões finitas em {split}")
        pred["abs_error"] = (pred.rate_pred - pred.rate_true).abs()
        for subset in ("all", "rapid"):
            part = pred if subset == "all" else pred[pred.rate_true.abs() >= 1]
            macro = part.groupby(["horizon_min", "model", "subject"]).abs_error.mean().groupby(
                ["horizon_min", "model"]).mean().sort_index()
            saved = metrics[(metrics.split == split) & (metrics.subset == subset)
                            & (metrics.scope == "macro")].set_index(["horizon_min", "model"]).sort_index()
            np.testing.assert_allclose(macro.to_numpy(), saved.mae_rate.to_numpy(), atol=1e-7)
        checks.append(f"MAE macro geral e rápido recalculados das previsões de {split}")
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = root / "results/azt1d_seta/runs" / args.run_name
    if run.parent.resolve() != (root / "results/azt1d_seta/runs").resolve() or not run.is_dir():
        parser.error("Execução inexistente ou nome inválido.")
    checks = verify(run)
    result = dict(status="passed", checks_passed=len(checks), checks=checks)
    (run / "verificacao.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"PASS: {len(checks)} verificações; {run / 'verificacao.json'}")


if __name__ == "__main__":
    main()
