"""Analise artefatos existentes sem treinar modelos nem alterar a selecao."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 42
INTENSITY_LABELS = ("<0.5", "0.5-<1", "1-<2", ">=2")
MODEL_LABELS = {
    "persistencia": "Persistência",
    "ultima_variacao": "Última variação",
    "linear_15min": "Linear 15 min",
    "lstm": "LSTM",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def comparison_by_person(metrics, baselines):
    """Use o mesmo baseline da validacao tambem nos diagnosticos do teste."""
    keys = ["split", "horizon_min", "subset", "subject"]
    cols = ["n", "mae_rate", "rmse_rate", "rmse_glucose", "sign_correct",
            "sign_opposite", "direction_n"]
    rows = []
    for horizon, baseline in baselines.items():
        data = metrics[(metrics.scope == "subject") &
                       (metrics.horizon_min == horizon)]
        left = data[data.model == baseline][keys + cols]
        right = data[data.model == "lstm"][keys + cols]
        paired = left.merge(right, on=keys, how="outer", validate="one_to_one",
                            suffixes=("_baseline", "_lstm"), indicator=True)
        if not paired._merge.eq("both").all():
            raise ValueError("LSTM e baseline nao possuem os mesmos participantes.")
        if not paired.n_baseline.eq(paired.n_lstm).all():
            raise ValueError("LSTM e baseline nao possuem as mesmas contagens.")
        paired = paired.drop(columns="_merge")
        paired.insert(4, "baseline_escolhido_validacao", baseline)
        paired["diferenca_mae_baseline_menos_lstm"] = (
            paired.mae_rate_baseline - paired.mae_rate_lstm)
        paired["ganho_relativo_mae"] = np.where(
            paired.mae_rate_baseline > 0,
            1 - paired.mae_rate_lstm / paired.mae_rate_baseline, np.nan)
        paired["lstm_melhor_mae"] = paired.mae_rate_lstm < paired.mae_rate_baseline
        rows.append(paired)
    return pd.concat(rows, ignore_index=True).sort_values(keys).reset_index(drop=True)


def intensity_diagnostics(predictions):
    """Media das metricas por pessoa; zeros reais nao entram na direcao."""
    data = predictions.copy()
    data["intensidade"] = pd.cut(
        data.rate_true.abs(), bins=[0, .5, 1, 2, np.inf], right=False,
        labels=INTENSITY_LABELS)
    rows = []
    for (horizon, model), group in data.groupby(["horizon_min", "model"], sort=True):
        for label in INTENSITY_LABELS:
            part = group[group.intensidade == label]
            per_person = []
            for subject, samples in part.groupby("subject", sort=True):
                real = samples.rate_true.to_numpy()
                predicted = samples.rate_pred.to_numpy()
                nonzero = real != 0
                product = real[nonzero] * predicted[nonzero]
                per_person.append(dict(
                    subject=subject, n=len(samples), direction_n=int(nonzero.sum()),
                    mae_rate=float(np.abs(predicted - real).mean()),
                    bias_rate=float((predicted - real).mean()),
                    sign_correct=float((product > 0).mean()) if nonzero.any() else np.nan,
                    sign_opposite=float((product < 0).mean()) if nonzero.any() else np.nan,
                    pred_exact_zero=float((predicted[nonzero] == 0).mean())
                    if nonzero.any() else np.nan))
            by_person = pd.DataFrame(per_person)
            row = dict(split="test", horizon_min=int(horizon), model=model,
                       intensidade_abs_taxa_real=label, n_janelas=len(part),
                       n_participantes=len(per_person),
                       n_direcao=int(by_person.direction_n.sum()) if len(by_person) else 0,
                       n_participantes_direcao=int((by_person.direction_n > 0).sum())
                       if len(by_person) else 0)
            for metric in ("mae_rate", "bias_rate", "sign_correct", "sign_opposite",
                           "pred_exact_zero"):
                row[f"{metric}_macro"] = float(by_person[metric].mean()) \
                    if len(by_person) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_by_person(comparison):
    """IC percentil descritivo da media das diferencas pareadas por pessoa."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    overall = comparison[comparison.subset == "all"]
    for (split, horizon), part in overall.groupby(["split", "horizon_min"], sort=True):
        part = part.sort_values("subject")
        differences = part.diferenca_mae_baseline_menos_lstm.to_numpy(dtype=float)
        if not len(differences) or not np.isfinite(differences).all():
            raise ValueError("Diferencas pareadas invalidas para o bootstrap.")
        indices = rng.integers(0, len(differences),
                              size=(BOOTSTRAP_REPETITIONS, len(differences)))
        estimates = differences[indices].mean(axis=1)
        low, high = np.quantile(estimates, [.025, .975])
        rows.append(dict(
            split=split, horizon_min=int(horizon),
            baseline_escolhido_validacao=part.baseline_escolhido_validacao.iloc[0],
            n_participantes=len(part), n_janelas=int(part.n_lstm.sum()),
            diferenca_media_mae_baseline_menos_lstm=float(differences.mean()),
            ic95_percentil_inferior=float(low), ic95_percentil_superior=float(high),
            pessoas_com_ganho=int((differences > 0).sum()),
            fracao_pessoas_com_ganho=float((differences > 0).mean()),
            repeticoes=BOOTSTRAP_REPETITIONS, seed=BOOTSTRAP_SEED,
            unidade_reamostrada="participante; par baseline/LSTM preservado"))
    return rows


def macro_summaries(metrics, baselines):
    rows = []
    macro = metrics[metrics.scope == "macro"]
    for (split, horizon), group in macro.groupby(["split", "horizon_min"], sort=True):
        baseline = baselines[int(horizon)]
        all_rows = group[group.subset == "all"].set_index("model")
        rapid = group[group.subset == "rapid"].set_index("model")
        alternatives = [model for model in all_rows.index if model != "lstm"]
        best_diagnostic = all_rows.loc[alternatives, "mae_rate"].idxmin()
        best_rapid_diagnostic = rapid.loc[alternatives, "mae_rate"].idxmin()
        lstm, base = all_rows.loc["lstm"], all_rows.loc[baseline]
        fast_lstm, fast_base = rapid.loc["lstm"], rapid.loc[baseline]
        rows.append(dict(
            split=split, horizon_min=int(horizon),
            baseline_escolhido_validacao=baseline,
            melhor_baseline_geral_nesta_particao_apenas_diagnostico=best_diagnostic,
            melhor_baseline_rapido_nesta_particao_apenas_diagnostico=best_rapid_diagnostic,
            n_janelas=int(lstm.n), n_participantes=int(lstm.participants_evaluable),
            mae_rate_macro_lstm=float(lstm.mae_rate),
            mae_rate_macro_baseline=float(base.mae_rate),
            ganho_relativo_macro=float(1 - lstm.mae_rate / base.mae_rate),
            rmse_glucose_macro_lstm=float(lstm.rmse_glucose),
            rmse_glucose_macro_baseline=float(base.rmse_glucose),
            sign_correct_macro_lstm=float(lstm.sign_correct),
            sign_correct_macro_baseline=float(base.sign_correct),
            sign_opposite_macro_lstm=float(lstm.sign_opposite),
            sign_opposite_macro_baseline=float(base.sign_opposite),
            n_janelas_rapidas=int(fast_lstm.n),
            mae_rate_macro_rapido_lstm=float(fast_lstm.mae_rate),
            mae_rate_macro_rapido_baseline=float(fast_base.mae_rate),
            mae_rate_macro_melhor_baseline_rapido=float(
                rapid.loc[best_rapid_diagnostic, "mae_rate"]),
            ganho_relativo_macro_rapido=float(1 - fast_lstm.mae_rate / fast_base.mae_rate),
            sign_correct_macro_rapido_lstm=float(fast_lstm.sign_correct),
            sign_correct_macro_rapido_baseline=float(fast_base.sign_correct),
            sign_opposite_macro_rapido_lstm=float(fast_lstm.sign_opposite),
            sign_opposite_macro_rapido_baseline=float(fast_base.sign_opposite)))
    return rows


def comparison_figure(metrics, comparison, baselines, output):
    horizons = sorted(baselines)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), layout="constrained")
    colors = {"persistencia": "#a5a5a5", "ultima_variacao": "#d68a36",
              "linear_15min": "#78a393", "lstm": "#305eaa"}
    models = list(MODEL_LABELS)
    for ax, subset, title in zip(axes[0], ("all", "rapid"),
                                 ("Todas as janelas", "Mudanças rápidas (|taxa real| ≥ 1)")):
        x = np.arange(len(horizons))
        width = .19
        for i, model in enumerate(models):
            data = metrics[(metrics.split == "test") & (metrics.scope == "macro") &
                           (metrics.subset == subset) & (metrics.model == model)]
            values = data.set_index("horizon_min").loc[horizons, "mae_rate"]
            ax.bar(x + (i - 1.5) * width, values, width=width,
                   color=colors[model], label=MODEL_LABELS[model])
        ax.set(xticks=x, xticklabels=[f"+{h} min" for h in horizons],
               ylabel="MAE da taxa (mg/dL/min), média por pessoa", title=title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    for ax, horizon in zip(axes[1], horizons):
        data = comparison[(comparison.split == "test") &
                          (comparison.horizon_min == horizon) &
                          (comparison.subset == "all")].copy()
        data["subject_number"] = pd.to_numeric(data.subject)
        data = data.sort_values("subject_number")
        gains = 100 * data.ganho_relativo_mae.to_numpy()
        ax.bar(np.arange(len(data)), gains,
               color=np.where(gains > 0, "#305eaa", "#c85858"))
        ax.axhline(0, color="black", linewidth=.8)
        ax.set(xticks=np.arange(len(data)), xticklabels=data.subject.tolist(),
               xlabel="Participante", ylabel="Ganho relativo do MAE (%)",
               title=f"+{horizon} min: LSTM vs. {MODEL_LABELS[baselines[horizon]]}")
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("Teste exploratório: erro por horizonte e ganho por pessoa\n"
                 "Baseline das comparações por pessoa fixado na validação", fontsize=14)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        parser.error("Use letras, numeros, hifen ou sublinhado em --run-name.")
    root = Path(__file__).resolve().parents[1]
    run = root / "results/azt1d_seta/runs" / args.run_name
    paths = {
        "metricas": run / "final/metricas.csv",
        "previsoes_teste": run / "final/previsoes_teste.csv.gz",
        "selecao": run / "selecao_horizonte.json",
        "manifesto": run / "manifest.json",
        "execucao": run / "execucao.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise SystemExit("Aguarde os artefatos completos da execucao: " + ", ".join(missing))
    execution = read_json(paths["execucao"])
    if execution["status"] != "completed":
        raise SystemExit("A execucao ainda nao esta registrada como concluida.")
    selection, manifest = read_json(paths["selecao"]), read_json(paths["manifesto"])
    metrics = pd.read_csv(paths["metricas"], dtype={"subject": str})
    if set(metrics.split) != {"val", "test"}:
        raise ValueError("O arquivo deve conter validacao e teste.")
    baselines = {int(row["horizon_min"]): row["best_baseline"]
                 for row in selection["validation_summary"]}
    for horizon, baseline in baselines.items():
        val = metrics[(metrics.split == "val") & (metrics.scope == "macro") &
                      (metrics.subset == "all") & (metrics.horizon_min == horizon) &
                      (metrics.model != "lstm")]
        if val.set_index("model").mae_rate.idxmin() != baseline:
            raise ValueError("Baseline salvo difere do vencedor na validacao.")
    predictions = pd.read_csv(paths["previsoes_teste"],
                              usecols=["subject", "split", "horizon_min", "model",
                                       "rate_true", "rate_pred"])
    if set(predictions.split) != {"test"}:
        raise ValueError("Diagnostico de intensidade recebe apenas teste.")
    if not np.isfinite(predictions[["rate_true", "rate_pred"]].to_numpy()).all():
        raise ValueError("Previsoes nao finitas.")
    comparison = comparison_by_person(metrics, baselines)
    diagnostics = intensity_diagnostics(predictions)
    bootstrap = bootstrap_by_person(comparison)
    macro = macro_summaries(metrics, baselines)
    out = run / "analise"
    out.mkdir(exist_ok=True)
    comparison.to_csv(out / "comparacao_por_pessoa.csv", index=False)
    diagnostics.to_csv(out / "diagnostico_intensidade.csv", index=False)
    comparison_figure(metrics, comparison, baselines, out / "comparacao_mae_e_ganhos.png")
    summary = dict(
        run_name=args.run_name, chosen_horizon_min=selection["chosen_horizon_min"],
        exploratory_prior_test_access=selection["exploratory_prior_test_access"],
        participants=manifest["participants"], counts=manifest["counts"],
        baselines_escolhidos_somente_validacao=baselines,
        destaques_por_particao_e_horizonte=macro,
        bootstrap_pareado_por_pessoa=bootstrap,
        definicao_ganho="1 - MAE_LSTM/MAE_baseline; positivo favorece LSTM",
        definicao_diferenca="MAE_baseline - MAE_LSTM; positivo favorece LSTM",
        interpretacao_ic=("IC percentil de 95% descritivo da media das diferencas de MAE "
                          "por pessoa. Reamostragem pareada de pessoas, nao de janelas. "
                          "Condicionado aos modelos, amostra e particionamento desta rodada; "
                          "nao inclui variabilidade de treino nem confirma generalizacao "
                          "para pacientes novos. Avaliacao exploratoria, sem inferencia clinica."),
        interpretacao_intensidade=("Faixas definidas pela taxa real absoluta no teste. "
                                  "MAE e direcao sao medias por pessoa com suporte na faixa. "
                                  "Acerto/opostos excluem taxa real exatamente zero; contagens "
                                  "de suporte de direcao sao publicadas separadamente."),
        fontes_sha256={name: file_sha(path) for name, path in paths.items()},
        script_sha256=file_sha(Path(__file__)))
    (out / "resumo_analitico.json").write_text(
        json.dumps(clean_json(summary), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    print(f"Analise salva em {out}")
    for row in macro:
        print(f"{row['split']} +{row['horizon_min']} min: "
              f"MAE LSTM={row['mae_rate_macro_lstm']:.4f}; "
              f"baseline={row['baseline_escolhido_validacao']} "
              f"MAE={row['mae_rate_macro_baseline']:.4f}; "
              f"ganho={row['ganho_relativo_macro']:.2%}")


if __name__ == "__main__":
    main()
