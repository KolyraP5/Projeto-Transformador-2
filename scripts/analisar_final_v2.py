"""Compare v1 e v2 concluídas; o diagnóstico de teste nunca promove a candidata."""
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


ROOT = Path(__file__).resolve().parents[1]
SUBSETS = ("all", "rapid", "low", "extreme", "zero")
ALGORITHMS = ("v1_lstm", "v2_lstm", "persistencia", "ultima_variacao", "linear_15min")
LABELS = dict(v1_lstm="LSTM v1", v2_lstm="LSTM v2", persistencia="Persistência",
              ultima_variacao="Última variação", linear_15min="Tendência linear")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def guard_completed(run, reference):
    """Não leia métricas/previsões antes de conferir estado e decisão congelada."""
    execution_path, selection_path = run / "execucao.json", run / "selecao_final.json"
    if not execution_path.is_file() or read_json(execution_path).get("status") != "completed":
        raise ValueError("Aguarde full_v2 concluída: nenhuma previsão foi aberta.")
    if not (reference / "execucao.json").is_file() or read_json(
            reference / "execucao.json").get("status") != "completed":
        raise ValueError("A referência v1 precisa estar concluída.")
    execution, selection = read_json(execution_path), read_json(selection_path)
    done = read_json(run / "final/test_done.json")
    selection_sha = sha(selection_path)
    if execution.get("selection_sha256") != selection_sha or done.get("selection_sha256") != selection_sha:
        raise ValueError("Teste e execução não correspondem à seleção congelada.")
    confirmation_path = run / "codigo/confirmation.json"
    confirmation = read_json(confirmation_path)
    candidate = selection["candidate"]
    matching = [entry for entry in confirmation.get("decisions", [])
                if entry.get("variant") == candidate]
    approved = bool(confirmation.get("stage") == "confirmation"
                    and confirmation.get("selection_is_final") is True
                    and confirmation.get("selected_variant") == candidate
                    and len(matching) == 1 and matching[0].get("accepted") is True)
    if any(value is not approved for value in
           (selection.get("approved"), execution.get("approved"), done.get("approved"))):
        raise ValueError("Aprovação diverge da confirmação de desenvolvimento congelada.")
    expected_confirmation_sha = selection.get("source_sha256", {}).get("confirmation")
    if expected_confirmation_sha != sha(confirmation_path):
        raise ValueError("A confirmação copiada foi alterada após a seleção.")
    weight_path = run / "model/best_weights.pt"
    model_done = read_json(run / "model/completed.json")
    if sha(weight_path) != done.get("weights_sha256") or sha(weight_path) != model_done.get("weights_sha256"):
        raise ValueError("Pesos finais alterados.")
    manifest, reference_manifest = read_json(run / "manifest.json"), read_json(reference / "manifest.json")
    if (manifest.get("data_source_fingerprint") != reference_manifest.get("fingerprint")
            or manifest["participants"] != reference_manifest["participants"]):
        raise ValueError("v1 e v2 precisam compartilhar base e população.")
    return execution, selection, manifest, approved


def load_predictions(path):
    columns = ["subject", "origin_ns", "horizon_min", "split", "model", "rate_true", "rate_pred",
               "glucose_true", "glucose_pred"]
    predictions = pd.read_csv(path, usecols=columns)
    predictions = predictions[predictions.horizon_min == 15].copy()
    if set(predictions.split) != {"test"}:
        raise ValueError("A comparação recebe somente previsões de teste em +15 minutos.")
    if not np.isfinite(predictions[["rate_true", "rate_pred", "glucose_true", "glucose_pred"]]).all().all():
        raise ValueError("Valores não finitos nas previsões finais.")
    if predictions.duplicated(["subject", "origin_ns", "model"]).any():
        raise ValueError("Previsões duplicadas para a mesma origem/modelo.")
    return predictions


def aligned_frames(v1, v2, participants, expected_n):
    base = v1[v1.model == "lstm"].sort_values(["subject", "origin_ns"]).reset_index(drop=True)
    if len(base) != expected_n or set(base.subject) != set(participants):
        raise ValueError("Contagem ou população diferente da registrada no manifesto.")
    keys = base[["subject", "origin_ns"]].to_numpy(dtype=np.int64)
    results = []
    for algorithm, source, model in (("v1_lstm", v1, "lstm"), ("v2_lstm", v2, "lstm"),
                                     ("persistencia", v1, "persistencia"),
                                     ("ultima_variacao", v1, "ultima_variacao"),
                                     ("linear_15min", v1, "linear_15min")):
        frame = source[source.model == model].sort_values(["subject", "origin_ns"]).reset_index(drop=True)
        if not np.array_equal(frame[["subject", "origin_ns"]].to_numpy(dtype=np.int64), keys):
            raise ValueError(f"Origens diferentes: {algorithm}")
        np.testing.assert_allclose(frame.rate_true, base.rate_true, atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(frame.glucose_true, base.glucose_true, atol=1e-7, rtol=1e-7)
        frame["algorithm"] = algorithm
        results.append(frame)
    # Os baselines recalculados no runner v2 precisam coincidir com os da v1.
    for model in ("persistencia", "ultima_variacao", "linear_15min"):
        old = v1[v1.model == model].sort_values(["subject", "origin_ns"])
        new = v2[v2.model == model].sort_values(["subject", "origin_ns"])
        if not np.array_equal(old[["subject", "origin_ns"]].to_numpy(),
                              new[["subject", "origin_ns"]].to_numpy()):
            raise ValueError(f"Origens do baseline {model} mudaram na v2.")
        np.testing.assert_allclose(old.rate_pred, new.rate_pred, atol=1e-7, rtol=1e-7)
    return pd.concat(results, ignore_index=True)


def person_metrics(predictions):
    rows = []
    for (algorithm, subject), group in predictions.groupby(["algorithm", "subject"], sort=True):
        true, pred = group.rate_true.to_numpy(), group.rate_pred.to_numpy()
        glucose_error = group.glucose_pred.to_numpy() - group.glucose_true.to_numpy()
        masks = dict(all=np.ones(len(group), dtype=bool), rapid=np.abs(true) >= 1,
                     low=np.abs(true) < .5, extreme=np.abs(true) >= 2, zero=true == 0)
        for subset, mask in masks.items():
            actual, output, ge = true[mask], pred[mask], glucose_error[mask]
            n = len(actual)
            nonzero = actual != 0
            product = actual[nonzero] * output[nonzero]
            error = output - actual
            rows.append(dict(algorithm=algorithm, subject=int(subject), subset=subset, n=n,
                direction_n=int(nonzero.sum()),
                mae_rate=float(np.abs(error).mean()) if n else np.nan,
                rmse_rate=float(np.sqrt(np.square(error).mean())) if n else np.nan,
                bias_rate=float(error.mean()) if n else np.nan,
                mae_glucose=float(np.abs(ge).mean()) if n else np.nan,
                rmse_glucose=float(np.sqrt(np.square(ge).mean())) if n else np.nan,
                sign_correct=float((product > 0).mean()) if len(product) else np.nan,
                sign_opposite=float((product < 0).mean()) if len(product) else np.nan,
                false_movement=float((np.abs(output) >= .5).mean()) if n and subset in ("low", "zero") else np.nan,
                mean_abs_true=float(np.abs(actual).mean()) if n else np.nan,
                mean_abs_prediction=float(np.abs(output).mean()) if n else np.nan))
    return pd.DataFrame(rows)


def validate_saved_metrics(people, metrics, algorithm, model):
    own = people[(people.algorithm == algorithm) & people.subset.isin(["all", "rapid"])].copy()
    saved = metrics[(metrics.split == "test") & (metrics.horizon_min == 15)
                    & (metrics.model == model) & (metrics.scope == "subject")].copy()
    saved["subject"] = pd.to_numeric(saved.subject).astype(int)
    index = ["subject", "subset"]
    own, saved = own.set_index(index).sort_index(), saved.set_index(index).sort_index()
    if not own.index.equals(saved.index):
        raise ValueError(f"Suporte diferente na tabela salva de {algorithm}.")
    for column in ("n", "mae_rate", "rmse_glucose", "sign_correct", "sign_opposite"):
        np.testing.assert_allclose(own[column], saved[column], rtol=1e-6, atol=1e-6, equal_nan=True)


def compare_people(people):
    fields = ["n", "direction_n", "mae_rate", "rmse_glucose", "sign_correct", "sign_opposite", "false_movement"]
    result = None
    for algorithm in ALGORITHMS:
        selected = people[people.algorithm == algorithm][["subject", "subset", *fields]].copy()
        selected = selected.rename(columns={key: f"{key}_{algorithm}" for key in fields})
        result = selected if result is None else result.merge(selected, on=["subject", "subset"],
                                                              validate="one_to_one", how="outer")
    result["difference_mae_v1_minus_v2"] = result.mae_rate_v1_lstm - result.mae_rate_v2_lstm
    result["gain_v2_vs_v1"] = np.where(result.mae_rate_v1_lstm > 0,
                                         1 - result.mae_rate_v2_lstm / result.mae_rate_v1_lstm, np.nan)
    result["gain_v2_vs_persistence"] = np.where(result.mae_rate_persistencia > 0,
                                                 1-result.mae_rate_v2_lstm/result.mae_rate_persistencia, np.nan)
    return result.sort_values(["subject", "subset"]).reset_index(drop=True)


def bootstrap(comparison):
    data = comparison[comparison.subset == "all"].sort_values("subject")
    differences = data.difference_mae_v1_minus_v2.to_numpy()
    if not np.isfinite(differences).all() or not len(differences):
        raise ValueError("Diferenças pareadas inválidas.")
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(differences), size=(10_000, len(differences)))
    estimates = differences[indices].mean(axis=1)
    low, high = np.quantile(estimates, [.025, .975])
    return dict(difference_mae_v1_minus_v2=float(differences.mean()),
                ci95_percentile_low=float(low), ci95_percentile_high=float(high),
                participants=len(differences), participants_improved=int((differences > 0).sum()),
                repetitions=10_000, seed=42, unit="participante, par v1/v2 preservado",
                interpretation="Intervalo descritivo condicionado aos modelos, pessoas e partição. Não inclui novas sementes nem generalização para pacientes novos.")


def macro_metrics(people):
    columns = [key for key in people if key not in ("algorithm", "subject", "subset", "n", "direction_n")]
    grouped = people.groupby(["algorithm", "subset"], sort=True)
    means = grouped[columns].mean()
    support = grouped.agg(n=("n", "sum"), participants=("n", lambda x: int((x > 0).sum())),
                          direction_n=("direction_n", "sum"))
    return means.join(support).reset_index()


def plot(macro, comparison, target):
    table = macro.set_index(["algorithm", "subset"])
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.7), layout="constrained")
    x, width = np.arange(4), .16
    subsets = ("all", "rapid", "low", "extreme")
    colors = ("#929292", "#305eaa", "#ba9a60", "#7caa98", "#b9b9c5")
    for i, algorithm in enumerate(ALGORITHMS):
        values = [table.loc[(algorithm, subset), "mae_rate"] for subset in subsets]
        axes[0].bar(x + (i - 2) * width, values, width, label=LABELS[algorithm], color=colors[i])
    axes[0].set(xticks=x, xticklabels=["Todas", "Rápidas", "Baixa variação", "Extremos"],
                ylabel="MAE da taxa (mg/dL/min), média por pessoa", title="Diagnóstico exploratório de teste")
    axes[0].legend(fontsize=8)
    part = comparison[comparison.subset == "all"].sort_values("subject")
    values = 100 * part.gain_v2_vs_v1.to_numpy()
    axes[1].bar(np.arange(len(part)), values, color=np.where(values >= 0, "#305eaa", "#c85858"))
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].set(xticks=np.arange(len(part)), xticklabels=part.subject.tolist(),
                xlabel="Participante", ylabel="Ganho de MAE v2 frente à v1 (%)", title="Comparação pareada por pessoa")
    axes[1].tick_params(axis="x", labelsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    figure.suptitle("Seta de +15 minutos: v1 e candidata v2 nas mesmas origens\n"
                   "A decisão de aprovação foi congelada antes deste diagnóstico")
    figure.savefig(target, dpi=160)
    plt.close(figure)


def fmt(value, digits=4):
    return f"{value:.{digits}f}".replace(".", ",") if pd.notna(value) else "sem suporte"


def report(summary, macro, output):
    data = macro.set_index(["algorithm", "subset"])
    approved = summary["approved_on_development"]
    ci = summary["bootstrap_paired_participants"]
    lines = ["# Diagnóstico final: LSTM v1 e candidata v2", "",
             f"Candidata: `{summary['candidate']}`. **Aprovada pelos critérios do desenvolvimento: "
             f"{'sim' if approved else 'não'}.**", "",
             ("A aprovação acima veio da confirmação de desenvolvimento congelada antes de abrir o teste."
              if approved else "A candidata permanece experimental e não substitui automaticamente a v1. Um ganho observado neste teste não altera essa decisão."), "",
             f"As execuções usam as mesmas **{summary['test_windows']:,} origens** e "
             f"**{summary['participants']} participantes**, no horizonte fixo de +15 minutos. "
             "Os resultados abaixo são médias das métricas por pessoa.", "",
             "| Método | MAE geral | MAE rápido | MAE baixa variação | MAE extremos | RMSE glicemia geral |",
             "|---|---:|---:|---:|---:|---:|"]
    for algorithm in ALGORITHMS:
        values = [data.loc[(algorithm, subset), "mae_rate"] for subset in ("all", "rapid", "low", "extreme")]
        lines.append(f"| {LABELS[algorithm]} | " + " | ".join(fmt(v) for v in values)
                     + f" | {fmt(data.loc[(algorithm,'all'),'rmse_glucose'],3)} |")
    v1, v2, persistence = (data.loc[(algorithm, "all"), "mae_rate"]
                            for algorithm in ("v1_lstm", "v2_lstm", "persistencia"))
    lines += ["", "MAE da taxa em mg/dL/min; RMSE da glicemia em mg/dL. Rápido: magnitude "
              "real ≥1; baixa variação: <0,5; extremo: ≥2.", "",
              f"No agregado, o ganho relativo da v2 frente à v1 foi **{fmt(100*(1-v2/v1),2)}%**; "
              f"frente à persistência, **{fmt(100*(1-v2/persistence),2)}%**. Houve melhora de MAE "
              f"em **{ci['participants_improved']}/{ci['participants']} pessoas**.", "",
              f"A diferença média pareada `MAE_v1 − MAE_v2` foi **{fmt(ci['difference_mae_v1_minus_v2'])}**, "
              f"com intervalo percentil descritivo de 95% **[{fmt(ci['ci95_percentile_low'])}; "
              f"{fmt(ci['ci95_percentile_high'])}] mg/dL/min**, usando 10.000 reamostragens de "
              "pessoas e semente 42. Valores positivos favorecem v2. A unidade reamostrada é "
              "a pessoa, não a janela sobreposta.", "", "## Direção e estabilidade", "",
              "| Método | Acerto rápido | Sinais opostos rápidos | Movimento em baixa variação |",
              "|---|---:|---:|---:|"]
    for algorithm in ("v1_lstm", "v2_lstm", "persistencia", "ultima_variacao"):
        rapid, low = data.loc[(algorithm, "rapid")], data.loc[(algorithm, "low")]
        lines.append(f"| {LABELS[algorithm]} | {fmt(100*rapid.sign_correct,2)}% | "
                     f"{fmt(100*rapid.sign_opposite,2)}% | {fmt(100*low.false_movement,2)}% |")
    lines += ["", "Direção exclui taxa real exatamente zero. Movimento em baixa variação é "
              "a proporção de previsões com magnitude ≥0,5 quando a taxa real tem magnitude <0,5. "
              "Os casos de taxa real zero estão publicados separadamente no diagnóstico CSV.", "",
              "## Limites e decisão", "", "Este é um diagnóstico exploratório: parte do teste já foi "
              "consultada anteriormente, as mesmas pessoas aparecem em treino e avaliação, "
              "e as janelas se sobrepõem. O intervalo é condicionado aos modelos e à amostra "
              "desta rodada; não mede variação entre novos treinamentos nem valida uso clínico.", "",
              "As faixas usam a taxa futura real somente para avaliar. Toda feature ou regra "
              "de inferência deve depender de dados disponíveis até a origem. O resultado "
              "de teste não promove uma candidata que falhou os critérios de desenvolvimento.", "",
              "![Comparação v1 e v2](comparacao_v1_v2.png)", "",
              "Fontes: [execução](../execucao.json), [decisão congelada](../selecao_final.json), "
              "[métricas salvas](../final/metricas.csv), [comparação por pessoa](comparacao_v1_v2.csv), "
              "[intensidade](diagnostico_intensidade.csv) e [resumo com hashes](resumo_analitico.json).", ""]
    output.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="full_v2")
    parser.add_argument("--reference-run", default="full_v1")
    args = parser.parse_args()
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in (args.run_name, args.reference_run)):
        parser.error("Identificadores inválidos.")
    parent = ROOT / "results/azt1d_seta/runs"
    run, reference = parent / args.run_name, parent / args.reference_run
    execution, selection, manifest, approved = guard_completed(run, reference)
    # O guard acima precede toda abertura de métricas e de previsões.
    paths = dict(v1_metrics=reference / "final/metricas.csv", v2_metrics=run / "final/metricas.csv",
                 v1_predictions=reference / "final/previsoes_teste.csv.gz",
                 v2_predictions=run / "final/previsoes_teste.csv.gz",
                 execution=run / "execucao.json", selection=run / "selecao_final.json",
                 confirmation=run / "codigo/confirmation.json", test_done=run / "final/test_done.json")
    saved1 = pd.read_csv(paths["v1_metrics"], dtype={"subject": str})
    saved2 = pd.read_csv(paths["v2_metrics"], dtype={"subject": str})
    v1, v2 = load_predictions(paths["v1_predictions"]), load_predictions(paths["v2_predictions"])
    n = manifest["counts"]["test"]["used"]
    predictions = aligned_frames(v1, v2, manifest["participants"], n)
    people = person_metrics(predictions)
    validate_saved_metrics(people, saved1, "v1_lstm", "lstm")
    validate_saved_metrics(people, saved2, "v2_lstm", "lstm")
    validate_saved_metrics(people, saved1, "persistencia", "persistencia")
    comparison = compare_people(people)
    macro = macro_metrics(people)
    interval = bootstrap(comparison)
    summary = dict(run_name=args.run_name, reference_run=args.reference_run,
                   candidate=selection["candidate"], approved_on_development=approved,
                   promotion_from_test=False,
                   status="approved_on_development" if approved else "experimental_candidate_not_promoted",
                   exploratory_prior_test_access=True, horizon_min=15, test_windows=n,
                   participants=len(manifest["participants"]), parameters=execution.get("parameters"),
                   macro_metrics=macro.to_dict(orient="records"), bootstrap_paired_participants=interval,
                   comparison_support="Mesmos participantes, origens subject/origin_ns e alvos reais em v1 e v2.",
                   definitions=dict(gain="1 - MAE_v2/MAE_comparador; positivo favorece v2",
                                    difference="MAE_v1 - MAE_v2; positivo favorece v2",
                                    false_movement="|previsão| >= 0.5 em baixa variação ou taxa real zero"),
                   source_sha256={key: sha(path) for key, path in paths.items()},
                   script_sha256=sha(Path(__file__)))
    output = run / "analise"
    output.mkdir(exist_ok=True)
    comparison.to_csv(output / "comparacao_v1_v2.csv", index=False)
    macro.to_csv(output / "diagnostico_intensidade.csv", index=False)
    (output / "resumo_analitico.json").write_text(json.dumps(clean(summary), indent=2,
                                                 ensure_ascii=False, allow_nan=False), encoding="utf-8")
    plot(macro, comparison, output / "comparacao_v1_v2.png")
    report(summary, macro, output / "relatorio.md")
    (run / "codigo/analisar_final_v2.py").write_bytes(Path(__file__).read_bytes())
    print(f"Análise final salva em {output}; aprovação no desenvolvimento={approved}")


if __name__ == "__main__":
    main()
