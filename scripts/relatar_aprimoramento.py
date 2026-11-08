"""Gere uma leitura estática das ablações já resumidas, sem treinar ou usar teste."""
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
LABELS = dict(baseline15="Referência 15 épocas", ep30="Até 30 épocas", pat5="Paciência 5",
              delta="CGM + diferenças", relative="CGM + canal relativo",
              weighted="Perda ponderada", combined="Canais combinados",
              combined_weighted="Canais + perda ponderada", persistencia="Persistência",
              ultima_variacao="Última variação", linear_15min="Tendência linear")
CRITERIA = dict(
    gain_at_least_2_percent="ganho mínimo de 2% no MAE geral",
    better_than_best_simple_baseline="ganho sobre o melhor baseline simples",
    majority_participants="ganho na maioria das pessoas",
    at_least_two_folds="consistência em pelo menos dois blocos",
    at_least_two_seeds="consistência em pelo menos duas sementes",
    rapid_mae_not_worse="MAE nas mudanças rápidas",
    rapid_direction_not_worse="acerto direcional nas mudanças rápidas",
    rapid_opposites_not_worse="sinais opostos nas mudanças rápidas",
    low_mae_not_worse="MAE em baixa variação",
    low_false_movement_improves="redução de movimento previsto em baixa variação",
    extreme_mae_not_worse="MAE nos extremos",
    extreme_direction_not_worse="direção nos extremos",
    same_origin_coverage="cobertura comparável")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value, digits=4):
    if pd.isna(value):
        return "sem suporte"
    return f"{value:.{digits}f}".replace(".", ",")


def percent(value, digits=2):
    return number(100 * value, digits) + "%" if pd.notna(value) else "sem suporte"


def load_report(directory):
    required = [directory / name for name in ("comparison.csv", "acceptance.json", "summary.json")]
    if not all(path.is_file() for path in required):
        return None
    comparison = pd.read_csv(required[0])
    acceptance = read_json(required[1])
    summary = read_json(required[2])
    if summary.get("status") != "completed":
        raise ValueError(f"Resumo ainda não concluído: {directory}")
    overall = comparison[comparison.level == "overall"].set_index(["variant", "subset"])
    variants = summary["variants"]
    for variant in variants:
        for subset in ("all", "rapid", "low", "extreme"):
            if (variant, subset) not in overall.index:
                raise ValueError(f"Métrica ausente para {variant}/{subset}: {directory}")
    costs = directory / "training_cost.csv"
    return dict(directory=directory, comparison=comparison, overall=overall,
                acceptance=acceptance, summary=summary, variants=variants,
                costs=pd.read_csv(costs) if costs.exists() else None,
                files=[*required, *([costs] if costs.exists() else [])])


def stage_lines(report, title, root_output):
    data, acceptance, summary = report["overall"], report["acceptance"], report["summary"]
    reference = data.loc[("baseline15", "all"), "mae_rate"]
    relative = report["directory"].relative_to(root_output).as_posix()
    lines = [f"## {title}", "",
             f"Blocos: {', '.join('B'+str(b) for b in summary['folds'])}; sementes: "
             f"{', '.join(map(str, summary['seeds']))}. Configurações materializadas: "
             f"{summary['n_trainings']}. Horizonte fixo: +15 minutos.", "",
             "As métricas são calculadas por pessoa em cada bloco/semente. O resumo faz "
             "a média dos blocos/sementes por pessoa e depois atribui peso igual às pessoas.", "",
             "| Configuração | MAE geral | Ganho frente à referência | MAE rápido | MAE baixa variação | MAE extremos | Movimento em baixa variação |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for variant in report["variants"]:
        general = data.loc[(variant, "all"), "mae_rate"]
        low = data.loc[(variant, "low")]
        lines.append(f"| {LABELS.get(variant, variant)} (`{variant}`) | {number(general)} | "
                     f"{percent(1-general/reference)} | {number(data.loc[(variant,'rapid'),'mae_rate'])} | "
                     f"{number(low.mae_rate)} | {number(data.loc[(variant,'extreme'),'mae_rate'])} | "
                     f"{percent(low.false_movement)} |")
    lines += ["", "Ganho positivo favorece a candidata. MAE em mg/dL/min. Movimento em baixa "
              "variação é a proporção de previsões com magnitude ≥0,5 quando a taxa real tem "
              "magnitude <0,5. A taxa real define a análise retrospectiva, não uma feature disponível em execução.", ""]

    simple = [v for v in ("persistencia", "ultima_variacao", "linear_15min")
              if (v, "all") in data.index]
    if simple:
        best = min(simple, key=lambda v: data.loc[(v, "all"), "mae_rate"])
        lines += [f"O melhor baseline simples no desenvolvimento foi **{LABELS[best]}**, com "
                  f"MAE geral **{number(data.loc[(best,'all'),'mae_rate'])}**. Este comparador "
                  "foi calculado nos mesmos blocos; não foi escolhido no teste.", ""]
    best_variant = min(report["variants"], key=lambda v: data.loc[(v, "all"), "mae_rate"])
    lines += [f"O menor MAE geral entre as configurações foi de **{LABELS.get(best_variant,best_variant)}** "
              f"(`{best_variant}`), **{number(data.loc[(best_variant,'all'),'mae_rate'])}**, "
              f"ganho de **{percent(1-data.loc[(best_variant,'all'),'mae_rate']/reference)}** frente "
              "à referência. O menor erro geral, isoladamente, não determina a aprovação.", ""]

    selected = acceptance["selected_variant"]
    if acceptance.get("selection_is_final", False):
        lines += [f"**Decisão no desenvolvimento:** `{selected}`. "
                  + ("Nenhuma candidata passou por todas as condições; a referência foi mantida."
                     if selected == "baseline15" else "A candidata passou por todas as condições registradas."), ""]
    else:
        lines += [f"**Decisão provisória da triagem:** `{selected}`. "
                  "Esta etapa não mede consistência em três blocos e três sementes; não aprova "
                  "definitivamente uma mudança.", ""]
    lines += ["### Condições e trade-offs observados", ""]
    for decision in acceptance["decisions"]:
        variant = decision["variant"]
        failed = [CRITERIA.get(k, k) for k, passed in decision["criteria"].items() if not passed]
        if failed:
            explanation = "condições não satisfeitas: " + "; ".join(failed) + "."
        else:
            explanation = "passou por todas as condições avaliadas nesta etapa."
        lines.append(f"- **{LABELS.get(variant,variant)}** (`{variant}`): ganho geral "
                     f"{percent(decision['gain_vs_refit_v1'])}; melhora em "
                     f"{percent(decision['fraction_participants_improved'])} das pessoas; {explanation}")
    lines += ["", "### Direção e suporte", "",
              "| Configuração | Acerto rápido | Sinais opostos rápidos | Acerto nos extremos | Pessoas avaliáveis nos extremos |",
              "|---|---:|---:|---:|---:|"]
    for variant in report["variants"]:
        rapid, extreme = data.loc[(variant, "rapid")], data.loc[(variant, "extreme")]
        lines.append(f"| `{variant}` | {percent(rapid.sign_correct)} | {percent(rapid.sign_opposite)} | "
                     f"{percent(extreme.sign_correct)} | {int(extreme.participants)} |")
    lines += ["", "A direção exclui taxa real exatamente zero. Prever zero não é acerto quando "
              "a mudança real é não nula. As contagens de previsões no CSV incluem repetições "
              "dos mesmos períodos entre sementes; não são novas observações independentes.", "",
              f"Fontes: [comparação]({relative}/comparison.csv), [decisões]({relative}/acceptance.json), "
              f"[métricas por pessoa]({relative}/metrics_by_person.csv) e "
              f"[configuração do resumo]({relative}/summary.json).", ""]
    if report["costs"] is not None:
        costs = report["costs"]
        if costs.train_seconds.notna().any():
            lines += ["### Custo registrado", "",
                      "| Configuração | Épocas mínimas–máximas | Tempo médio registrado (s) |",
                      "|---|---:|---:|"]
            for variant, part in costs.groupby("variant", sort=False):
                lines.append(f"| `{variant}` | {int(part.epochs_ran.min())}–{int(part.epochs_ran.max())} | "
                             f"{number(part.train_seconds.mean(),1)} |")
            lines += ["", "Limites de época e paciência podem compartilhar a mesma trajetória "
                      "de otimização. Os tempos por artefato não devem ser somados como custo "
                      "físico independente.", ""]
    return lines


def panel(report, destination):
    data = report["overall"]
    variants = report["variants"]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, subset, title in zip(axes.flat, ("all", "rapid", "low", "extreme"),
                                ("Todas as janelas", "Mudanças rápidas: |r| ≥ 1",
                                 "Baixa variação: |r| < 0,5", "Extremos: |r| ≥ 2")):
        values = [data.loc[(variant, subset), "mae_rate"] for variant in variants]
        x = np.arange(len(variants))
        ax.bar(x, values, color=["#929292" if v == "baseline15" else "#305eaa" for v in variants])
        ax.set(xticks=x, xticklabels=variants, title=title,
               ylabel="MAE taxa, média por pessoa (mg/dL/min)")
        ax.tick_params(axis="x", rotation=40, labelsize=8)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    stage = "Confirmação no desenvolvimento" if report["acceptance"].get("selection_is_final") else "Triagem exploratória"
    figure.suptitle(f"{stage}: seta de +15 minutos\nMudanças comparadas às mesmas origens")
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", default="aprimoramento_v2")
    parser.add_argument("--screening-report", default="triagem")
    parser.add_argument("--confirmation-report", default="confirmacao")
    args = parser.parse_args()
    for name in (args.experiment_name, args.screening_report, args.confirmation_report):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            parser.error("Identificadores devem usar letras, números, hífen ou sublinhado.")
    output = ROOT / "results/azt1d_seta/experiments" / args.experiment_name / "analise"
    screening = load_report(output / args.screening_report)
    confirmation = load_report(output / args.confirmation_report)
    if screening is None:
        parser.error("Relatório de triagem completo ainda não existe.")
    lines = ["# Aprimoramento da seta de +15 minutos", "",
             "Relatório gerado automaticamente dos artefatos reais de desenvolvimento. "
             "O algoritmo v1 e seus resultados permanecem preservados. Nenhum resultado "
             "de teste final é lido por este gerador.", ""]
    lines += stage_lines(screening, "Triagem das ideias", output)
    if confirmation:
        lines += stage_lines(confirmation, "Confirmação da finalista", output)
    else:
        lines += ["## Confirmação", "", "Ainda não há relatório confirmatório completo. "
                  "Os resultados acima permanecem provisórios.", ""]
    lines += ["## Leitura conjunta", "",
              "A taxa prevista é convertida em uma seta contínua; melhorar sua representação "
              "visual não altera o erro preditivo. O objetivo das ablações é reduzir erro "
              "geral sem transferir a piora para estabilidade, intensidade extrema ou direção.", "",
              "As faixas usam a taxa real futura somente para avaliar. Em execução, toda "
              "feature e qualquer regra de ajuste dependem exclusivamente do passado disponível "
              "e da própria previsão.", "",
              "Esta pesquisa já examinou dados e resultados anteriores da mesma base. "
              "A validação temporal organiza o desenvolvimento, mas não recupera um teste "
              "externo intocado. Há janelas sobrepostas e períodos compartilhados entre "
              "blocos; sementes e janelas não devem ser contadas como indivíduos independentes.", "",
              "![Comparação das configurações](painel_aprimoramento.png)", ""]
    panel(confirmation or screening, output / "painel_aprimoramento.png")
    (output / "relatorio.md").write_text("\n".join(lines), encoding="utf-8")
    reports = [screening, *([confirmation] if confirmation else [])]
    sources = {str(path.relative_to(output)): sha(path) for report in reports for path in report["files"]}
    (output / "relatorio_fontes.json").write_text(json.dumps(dict(
        script_sha256=sha(Path(__file__)), source_hashes=sources,
        final_test_accessed=False), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Relatório e painel salvos em {output}")


if __name__ == "__main__":
    main()
