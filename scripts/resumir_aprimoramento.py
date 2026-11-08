"""Compare ablações em blocos de desenvolvimento, sem consultar o teste final."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.seta_tendencia import azt1d_core as core


DEFAULT_VARIANTS = ("baseline15", "ep30", "pat5", "delta", "relative", "weighted")
FOLDS = (1, 2, 3)
SEEDS = (42, 43, 44)
SUBSETS = ("all", "rapid", "low", "extreme", "zero")
MEASURES = ("mae_rate", "rmse_rate", "sign_correct", "sign_opposite", "false_movement")
BASELINES = ("persistencia", "ultima_variacao", "linear_15min")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, value):
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")


def origin_signature(subject, origin):
    keys = np.column_stack((np.asarray(subject, dtype=np.int64),
                            np.asarray(origin, dtype=np.int64)))
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    return hashlib.sha256(np.ascontiguousarray(keys[order]).tobytes()).hexdigest(), order


def person_metrics(predictions, variant, fold, seed):
    rows = []
    for subject, group in predictions.groupby("subject", sort=True):
        real, predicted = group.rate_true.to_numpy(), group.rate_pred.to_numpy()
        masks = dict(all=np.ones(len(group), dtype=bool), rapid=np.abs(real) >= 1,
                     low=np.abs(real) < .5, extreme=np.abs(real) >= 2, zero=real == 0)
        for subset, mask in masks.items():
            target, output = real[mask], predicted[mask]
            n = len(target)
            nonzero = target != 0
            prod = target[nonzero] * output[nonzero]
            err = output - target
            rows.append(dict(
                variant=variant, fold=fold, seed=seed, subject=int(subject), subset=subset,
                n=n, direction_n=int(nonzero.sum()),
                mae_rate=float(np.abs(err).mean()) if n else np.nan,
                rmse_rate=float(np.sqrt(np.square(err).mean())) if n else np.nan,
                sign_correct=float((prod > 0).mean()) if len(prod) else np.nan,
                sign_opposite=float((prod < 0).mean()) if len(prod) else np.nan,
                false_movement=float((np.abs(output) >= .5).mean()) if n else np.nan))
    return pd.DataFrame(rows)


def read_model(path, variant, fold, seed, expected, expected_signature):
    config = read_json(path / "config.json")
    done = read_json(path / "completed.json")
    parameters = config.get("parameters", config)
    if int(parameters["seed"]) != seed:
        raise ValueError(f"Semente incompatível: {path}")
    if done.get("status", "completed") != "completed":
        raise ValueError(f"Treino não concluído: {path}")
    weights = path / "best_weights.pt"
    if "weights_sha256" in done and sha(weights) != done["weights_sha256"]:
        raise ValueError(f"Pesos alterados: {path}")
    columns = {"subject", "origin_ns", "rate_true", "rate_pred", "horizon_min", "model", "split"}
    pred = pd.read_csv(path / "previsoes_evaluation.csv.gz", usecols=lambda c: c in columns)
    if not columns.issubset(pred.columns):
        raise ValueError(f"Previsões sem o esquema esperado: {path}")
    if set(pred.horizon_min) != {15} or set(pred.model) != {"lstm"} or set(pred.split) != {"dev"}:
        raise ValueError(f"Mistura de modelos, horizontes ou partições: {path}")
    if pred.duplicated(["subject", "origin_ns"]).any():
        raise ValueError(f"Origens duplicadas: {path}")
    if not np.isfinite(pred[["rate_true", "rate_pred"]].to_numpy()).all():
        raise ValueError(f"Previsões não finitas: {path}")
    signature, order = origin_signature(pred.subject, pred.origin_ns)
    if len(pred) != len(expected["X"]) or signature != expected_signature[0]:
        raise ValueError(f"Cobertura/origens diferentes do bloco preparado: {path}")
    np.testing.assert_allclose(pred.rate_true.to_numpy()[order],
                               expected["y"][expected_signature[1], 1], atol=1e-6, rtol=1e-6)
    metrics = person_metrics(pred, variant, fold, seed)
    # O recálculo a partir das previsões é independente da tabela produzida no treino.
    saved = pd.read_csv(path / "metrics_evaluation.csv", dtype={"subject": str})
    saved = saved[(saved.scope == "subject") & saved.subset.isin(["all", "rapid"])].copy()
    saved["subject"] = pd.to_numeric(saved.subject).astype(int)
    index = ["subject", "subset"]
    own = metrics[metrics.subset.isin(["all", "rapid"])].set_index(index).sort_index()
    saved = saved.set_index(index).sort_index()
    if not own.index.equals(saved.index):
        raise ValueError(f"Pessoas/subconjuntos diferentes na tabela salva: {path}")
    for key in ("mae_rate", "sign_correct", "sign_opposite", "n"):
        np.testing.assert_allclose(own[key], saved[key], atol=1e-6, rtol=1e-6, equal_nan=True)
    files = ("config.json", "completed.json", "metrics_evaluation.csv", "previsoes_evaluation.csv.gz")
    provenance = {name: sha(path / name) for name in files}
    provenance["origin_signature"] = signature
    history = path / "history.csv"
    if not history.exists():
        history = path / "historico.csv"
    train_info = dict(variant=variant, fold=fold, seed=seed,
                      epochs_ran=done.get("epochs_ran"), train_seconds=done.get("train_seconds"),
                      best_epoch=done.get("best_epoch"),
                      feature_mode=parameters.get("feature_mode"),
                      max_epochs=parameters.get("max_epochs"),
                      patience=parameters.get("patience"),
                      rapid_weight=parameters.get("rapid_weight"),
                      shared_trajectory_seconds=done.get("shared_trajectory_seconds"))
    if history.exists():
        provenance[history.name] = sha(history)
    return metrics, provenance, train_info


def aggregate(person):
    """Primeiro média dos blocos/sementes por pessoa; depois peso igual por pessoa."""
    summaries = []
    for level, keys in (("overall", []), ("fold", ["fold"]), ("seed", ["seed"])):
        identity = ["variant", "subset", *keys]
        valid = person[person.n > 0]
        subjects = valid.groupby([*identity, "subject"], dropna=False)[list(MEASURES)].mean()
        result = subjects.groupby(identity, dropna=False).mean().reset_index()
        support = valid.groupby(identity, dropna=False).agg(
            n_predictions=("n", "sum"), participants=("subject", "nunique"))
        result = result.merge(support.reset_index(), on=identity, validate="one_to_one")
        result["level"] = level
        if "fold" not in result:
            result["fold"] = "ALL"
        if "seed" not in result:
            result["seed"] = "ALL"
        summaries.append(result)
    return pd.concat(summaries, ignore_index=True)


def evaluate_acceptance(person, comparison, variants, folds, seeds):
    overall = comparison[comparison.level == "overall"].set_index(["variant", "subset"])
    reference = overall.loc["baseline15"]
    best_simple = overall.loc[(list(BASELINES), "all"), "mae_rate"].min()
    rows = []
    eps = 1e-10
    confirmed = set(folds) == set(FOLDS) and set(seeds) == set(SEEDS)
    for variant in variants:
        if variant == "baseline15":
            continue
        candidate = overall.loc[variant]
        gain = 1 - candidate.loc["all", "mae_rate"] / reference.loc["all", "mae_rate"]
        paired = person[(person.variant.isin(["baseline15", variant])) & (person.subset == "all")]
        per_subject = paired.groupby(["variant", "subject"]).mae_rate.mean().unstack("variant")
        fraction = float((per_subject[variant] < per_subject.baseline15).mean())
        blocks = comparison[(comparison.level == "fold") & (comparison.subset == "all")]
        by_block = blocks.pivot(index="fold", columns="variant", values="mae_rate")
        seeds = comparison[(comparison.level == "seed") & (comparison.subset == "all")]
        by_seed = seeds.pivot(index="seed", columns="variant", values="mae_rate")
        blocks_won = int((by_block[variant] < by_block.baseline15).sum())
        seeds_won = int((by_seed[variant] < by_seed.baseline15).sum())

        def le(subset, metric):
            return bool(np.isfinite(candidate.loc[subset, metric]) and
                        candidate.loc[subset, metric] <= reference.loc[subset, metric] + eps)

        def ge(subset, metric):
            return bool(np.isfinite(candidate.loc[subset, metric]) and
                        candidate.loc[subset, metric] >= reference.loc[subset, metric] - eps)

        # Cobertura já foi verificada origem por origem na leitura dos nove modelos.
        criteria = dict(
            gain_at_least_2_percent=bool(gain >= .02),
            better_than_best_simple_baseline=bool(candidate.loc["all", "mae_rate"] < best_simple),
            majority_participants=bool(fraction > .5),
            rapid_mae_not_worse=le("rapid", "mae_rate"),
            rapid_direction_not_worse=ge("rapid", "sign_correct"),
            rapid_opposites_not_worse=le("rapid", "sign_opposite"),
            low_mae_not_worse=le("low", "mae_rate"),
            low_false_movement_improves=bool(candidate.loc["low", "false_movement"] <
                                            reference.loc["low", "false_movement"]),
            extreme_mae_not_worse=le("extreme", "mae_rate"),
            extreme_direction_not_worse=ge("extreme", "sign_correct"),
            same_origin_coverage=True)
        if confirmed:
            criteria.update(at_least_two_folds=blocks_won >= 2,
                            at_least_two_seeds=seeds_won >= 2)
        rows.append(dict(variant=variant, gain_vs_refit_v1=float(gain),
                         fraction_participants_improved=fraction, folds_improved=blocks_won,
                         seeds_improved=seeds_won, criteria=criteria,
                         accepted=bool(confirmed and all(criteria.values())),
                         provisionally_accepted=all(criteria.values()),
                         extreme_matches_last_variation=bool(candidate.loc["extreme", "mae_rate"] <=
                             overall.loc[("ultima_variacao", "extreme"), "mae_rate"])))
    accepted = [r for r in rows if r["provisionally_accepted"]]
    winner = max(accepted, key=lambda r: r["gain_vs_refit_v1"])["variant"] if accepted else "baseline15"
    return dict(stage="confirmation" if confirmed else "screening",
                selection_is_final=confirmed, selected_variant=winner,
                provisional_selected_variant=winner if not confirmed else None,
                baseline_variant="baseline15", decisions=rows,
                rule="Critérios fixados no plano; todas as condições devem passar. Sem candidata, mantém v1.",
                screening_note="Em triagem, consenso entre três blocos e três sementes ainda não avaliado; aprovação apenas provisória." if not confirmed else None,
                selection_data="Somente blocos de comparação do desenvolvimento, nos primeiros 80%.")


def figure(comparison, variants, destination, folds, seeds):
    data = comparison[comparison.level == "overall"].set_index(["variant", "subset"])
    labels = [*variants, *BASELINES]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    for ax, subset, title in zip(axes, ("all", "low", "extreme"),
                                ("Todas as janelas", "Baixa variação: |r| < 0,5", "Extremos: |r| ≥ 2")):
        y = [data.loc[(name, subset), "mae_rate"] for name in labels]
        ax.bar(np.arange(len(labels)), y,
               color=["#305eaa" if n not in BASELINES else "#9a9a9a" for n in labels])
        ax.set(xticks=np.arange(len(labels)), xticklabels=labels, title=title,
               ylabel="MAE taxa (mg/dL/min), média por pessoa")
        ax.tick_params(axis="x", rotation=55, labelsize=8)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle(f"Desenvolvimento temporal: {len(folds)} bloco(s) e {len(seeds)} semente(s)\n"
                 "Sem acesso ao teste para escolher as melhorias")
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", default="aprimoramento_v2")
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--blocks", nargs="+", choices=("B1", "B2", "B3"), default=["B1", "B2", "B3"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--report-name", help="Subpasta de analise para preservar triagem e confirmação separadas.")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.experiment_name):
        parser.error("Nome do experimento inválido.")
    if args.report_name and not re.fullmatch(r"[A-Za-z0-9_-]+", args.report_name):
        parser.error("Nome do relatório inválido.")
    variants = list(dict.fromkeys(["baseline15", *args.variants]))
    folds = tuple(dict.fromkeys(int(block[1:]) for block in args.blocks))
    seeds = tuple(dict.fromkeys(args.seeds))
    if any(seed < 0 for seed in seeds):
        parser.error("Sementes devem ser não negativas.")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", v) for v in variants):
        parser.error("Identificador de variante inválido.")
    experiment = ROOT / "results/azt1d_seta/experiments" / args.experiment_name
    out = experiment / "analise"
    if args.report_name:
        out /= args.report_name
    out.mkdir(parents=True, exist_ok=True)
    required = ("config.json", "completed.json", "metrics_evaluation.csv", "previsoes_evaluation.csv.gz")
    missing = []
    for variant in variants:
        for fold in folds:
            for seed in seeds:
                model = experiment / "models" / f"{variant}_B{fold}_s{seed}"
                absent = [name for name in required if not (model / name).is_file()]
                if absent:
                    missing.append(dict(model=model.name, missing_files=absent))
    if missing:
        save_json(out / "incomplete.json", dict(status="incomplete", variants=variants,
                                                missing_models=missing,
                                                selection_performed=False))
        raise SystemExit(f"Grade incompleta: {len(missing)} modelos; seleção recusada. Veja {out / 'incomplete.json'}")

    rows, source_hashes, train_rows = [], {}, []
    for fold in folds:
        cache = experiment / "prepared" / f"B{fold}" / "evaluation.npz"
        with np.load(cache, allow_pickle=False) as prepared:
            arr = {key: prepared[key] for key in prepared.files}
        source_hashes[str(cache.relative_to(experiment))] = sha(cache)
        signature = origin_signature(arr["subject"], arr["origin_ns"])
        for variant in variants:
            for seed in seeds:
                model = experiment / "models" / f"{variant}_B{fold}_s{seed}"
                own, sources, training = read_model(model, variant, fold, seed, arr, signature)
                rows.append(own)
                source_hashes[str(model.relative_to(experiment))] = sources
                train_rows.append(training)
        for baseline, rates in core.baseline_rates(arr["X"]).items():
            pred = pd.DataFrame(dict(subject=arr["subject"], rate_true=arr["y"][:, 1], rate_pred=rates))
            # Baselines são determinísticos; repetidos por semente para o mesmo peso.
            rows.extend(person_metrics(pred, baseline, fold, seed) for seed in seeds)
    people = pd.concat(rows, ignore_index=True)
    comparison = aggregate(people)
    decision = evaluate_acceptance(people, comparison, variants, folds, seeds)
    people.to_csv(out / "metrics_by_person.csv", index=False)
    comparison.to_csv(out / "comparison.csv", index=False)
    pd.DataFrame(train_rows).to_csv(out / "training_cost.csv", index=False)
    save_json(out / "acceptance.json", decision)
    figure(comparison, variants, out / "comparacao_aprimoramento.png", folds, seeds)
    save_json(out / "summary.json", dict(
        status="completed", experiment=args.experiment_name, variants=variants,
        n_trainings=len(variants) * len(folds) * len(seeds), folds=folds, seeds=seeds,
        stage=decision["stage"], selection_is_final=decision["selection_is_final"],
        selected_variant=decision["selected_variant"], horizon_min=15,
        aggregation="Média dos blocos/sementes por pessoa, seguida de média com peso igual por pessoa.",
        subsets=dict(all="Todas", rapid="|taxa real| >= 1", low="|taxa real| < 0.5",
                     extreme="|taxa real| >= 2", zero="taxa real exatamente zero"),
        false_movement="Proporção de |taxa prevista| >= 0.5 na faixa indicada; critério usado na faixa low.",
        interpretation="Comparação exploratória no desenvolvimento temporal; blocos sobrepostos e sementes não são unidades independentes. Teste final não acessado por este script.",
        source_hashes=source_hashes, script_sha256=sha(Path(__file__))))
    if (out / "incomplete.json").exists():
        (out / "incomplete.json").unlink()
    print(comparison[(comparison.level == "overall") & (comparison.subset == "all")]
          [["variant", "mae_rate", "sign_correct"]].to_string(index=False))
    print(f"Selecionada: {decision['selected_variant']}; artefatos em {out}")


if __name__ == "__main__":
    main()
