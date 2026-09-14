"""Pré-processamento rastreável do dataset AZT1D.

Regras desta versão:
- a coluna de glicose do Subject 14 é normalizada por nome;
- somente linhas completamente idênticas são removidas;
- CGM divergente no mesmo bin de 5 min torna o bin ambíguo;
- lacunas e bins ambíguos nunca recebem interpolação;
- basal e modo do dispositivo não entram nas features iniciais.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import pandas as pd


CANONICAL_COLUMNS = (
    "EventDateTime",
    "DeviceMode",
    "BolusType",
    "Basal",
    "CorrectionDelivered",
    "TotalBolusInsulinDelivered",
    "FoodDelivered",
    "CarbSize",
    "CGM",
)
NUMERIC_COLUMNS = (
    "Basal",
    "CorrectionDelivered",
    "TotalBolusInsulinDelivered",
    "FoodDelivered",
    "CarbSize",
    "CGM",
)
COLUMN_ALIASES = {"Readings (CGM / BGM)": "CGM"}

# Basal permanece no dado cru, mas é excluído da v0 por possível escala mista.
FEATURE_COLUMNS = ("CGM", "bolus_units", "carbs_g")


@dataclass(frozen=True)
class WindowSet:
    """Janelas sequenciais de um ou mais participantes."""

    inputs: np.ndarray
    targets: np.ndarray
    subjects: np.ndarray
    input_start_times: pd.DatetimeIndex
    input_end_times: pd.DatetimeIndex
    target_times: pd.DatetimeIndex
    segment_ids: np.ndarray
    feature_names: tuple[str, ...] = FEATURE_COLUMNS

    @property
    def size(self) -> int:
        return int(self.targets.shape[0])

    def metadata_frame(self) -> pd.DataFrame:
        """Retorna metadados para auditoria e cálculo de métricas."""
        return pd.DataFrame(
            {
                "subject": self.subjects,
                "segment_id": self.segment_ids,
                "input_start_time": self.input_start_times,
                "input_end_time": self.input_end_times,
                "target_time": self.target_times,
                "target_cgm": self.targets,
            }
        )


def _subject_number(subject_dir: Path) -> int:
    match = re.search(r"\d+", subject_dir.name)
    if match is None:
        raise ValueError(f"Não foi possível extrair o identificador de {subject_dir.name!r}.")
    return int(match.group())


def discover_subject_files(cgm_records_dir: Path | str) -> dict[int, Path]:
    """Retorna os CSVs principais ordenados pelo identificador de participante."""
    root = Path(cgm_records_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Pasta de dados não encontrada: {root}")

    subject_files: dict[int, Path] = {}
    for subject_dir in sorted(root.glob("Subject *"), key=_subject_number):
        csv_files = list(subject_dir.glob("*.csv"))
        if len(csv_files) != 1:
            raise ValueError(
                f"{subject_dir} deveria conter exatamente um CSV principal; encontrados: {csv_files}"
            )
        subject = _subject_number(subject_dir)
        if subject in subject_files:
            raise ValueError(f"Identificador de participante repetido: {subject}")
        subject_files[subject] = csv_files[0]
    return subject_files


def load_subject(subject: int, csv_path: Path | str) -> tuple[pd.DataFrame, dict[str, object]]:
    """Carrega um CSV e o normaliza, sem imputar nem corrigir escalas clínicas."""
    path = Path(csv_path)
    raw = pd.read_csv(path)
    raw.columns = raw.columns.str.strip()
    original_columns = raw.columns.tolist()
    raw = raw.rename(columns=COLUMN_ALIASES)

    missing_required = [
        column for column in ("EventDateTime", "CGM") if column not in raw.columns
    ]
    if missing_required:
        raise ValueError(f"Subject {subject}: colunas obrigatórias ausentes: {missing_required}")

    for column in CANONICAL_COLUMNS:
        if column not in raw.columns:
            raw[column] = np.nan
    frame = raw.loc[:, CANONICAL_COLUMNS].copy()
    frame["EventDateTime"] = pd.to_datetime(frame["EventDateTime"], errors="coerce")

    parse_errors: dict[str, int] = {}
    for column in NUMERIC_COLUMNS:
        original = frame[column]
        converted = pd.to_numeric(original, errors="coerce")
        parse_errors[column] = int(
            (original.notna() & original.astype(str).str.strip().ne("") & converted.isna()).sum()
        )
        frame[column] = converted

    raw_rows = len(frame)
    # A deduplicação por timestamp seria insegura: eventos diferentes podem ter o mesmo horário.
    frame = frame.drop_duplicates().copy()
    exact_duplicates_removed = raw_rows - len(frame)
    frame["Basal_raw"] = frame["Basal"]
    frame["subject"] = int(subject)

    report: dict[str, object] = {
        "subject": int(subject),
        "source_file": str(path),
        "raw_rows": int(raw_rows),
        "exact_duplicates_removed": int(exact_duplicates_removed),
        "invalid_timestamps": int(frame["EventDateTime"].isna().sum()),
        "used_cgm_alias": "Readings (CGM / BGM)" in original_columns,
        "numeric_parse_errors": {
            column: count for column, count in parse_errors.items() if count
        },
    }
    return frame, report


def make_5min_grid(frame: pd.DataFrame, frequency: str = "5min") -> pd.DataFrame:
    """Alinha um participante a uma grade e preserva conflitos de CGM.

    Uma leitura de CGM é aceita apenas se houver um único valor distinto não nulo
    no bin. Bolus total e carboidrato são somados depois da deduplicação exata.
    """
    required = {"EventDateTime", "CGM", "TotalBolusInsulinDelivered", "CarbSize"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Frame sem colunas necessárias: {sorted(missing)}")

    ordered = (
        frame.dropna(subset=["EventDateTime"])
        .sort_values("EventDateTime", kind="stable")
        .copy()
    )
    if ordered.empty:
        empty = pd.DataFrame(
            columns=[
                "raw_rows_in_bin",
                "cgm_distinct_values",
                "CGM",
                "bolus_units",
                "carbs_g",
                "bolus_records",
                "carb_records",
                "cgm_conflict",
                "valid_cgm",
                "clean_segment_id",
            ],
            index=pd.DatetimeIndex([], name="EventDateTime"),
        )
        return empty

    ordered["grid_time"] = ordered["EventDateTime"].dt.floor(frequency)
    grouped = ordered.groupby("grid_time", sort=True)
    grouped_cgm = grouped["CGM"]

    binned = pd.concat(
        [
            grouped.size().rename("raw_rows_in_bin"),
            grouped_cgm.nunique(dropna=True).rename("cgm_distinct_values"),
            grouped_cgm.first().rename("CGM"),
            grouped["TotalBolusInsulinDelivered"].sum(min_count=1).rename("bolus_units"),
            grouped["CarbSize"].sum(min_count=1).rename("carbs_g"),
            grouped["TotalBolusInsulinDelivered"].count().rename("bolus_records"),
            grouped["CarbSize"].count().rename("carb_records"),
        ],
        axis=1,
    )
    binned.loc[binned["cgm_distinct_values"].gt(1), "CGM"] = np.nan

    full_index = pd.date_range(binned.index.min(), binned.index.max(), freq=frequency)
    grid = binned.reindex(full_index)
    grid.index.name = "EventDateTime"
    for column in ("raw_rows_in_bin", "cgm_distinct_values", "bolus_records", "carb_records"):
        grid[column] = grid[column].fillna(0).astype(int)
    # Zero para eventos significa apenas que não há evento registrado naquele bin.
    grid["bolus_units"] = grid["bolus_units"].fillna(0.0)
    grid["carbs_g"] = grid["carbs_g"].fillna(0.0)
    grid["cgm_conflict"] = grid["cgm_distinct_values"].gt(1)
    grid["valid_cgm"] = grid["CGM"].notna() & ~grid["cgm_conflict"]

    segment_change = grid["valid_cgm"].ne(grid["valid_cgm"].shift(fill_value=False))
    segment_number = segment_change.cumsum()
    grid["clean_segment_id"] = segment_number.where(grid["valid_cgm"], pd.NA).astype("Int64")
    return grid


def _empty_window_set(history_steps: int, feature_names: tuple[str, ...]) -> WindowSet:
    empty_times = pd.DatetimeIndex([], dtype="datetime64[ns]")
    return WindowSet(
        inputs=np.empty((0, history_steps, len(feature_names)), dtype=np.float32),
        targets=np.empty((0,), dtype=np.float32),
        subjects=np.empty((0,), dtype=np.int64),
        input_start_times=empty_times,
        input_end_times=empty_times,
        target_times=empty_times,
        segment_ids=np.empty((0,), dtype=np.int64),
        feature_names=feature_names,
    )


def build_cgm_windows(
    grid: pd.DataFrame,
    subject: int,
    history_steps: int = 12,
    horizon_steps: int = 6,
    feature_names: tuple[str, ...] = FEATURE_COLUMNS,
) -> WindowSet:
    """Cria janelas sem atravessar lacunas ou conflitos de CGM.

    Para 12 entradas e horizonte 6, a janela usa entradas i até i+11 e alvo i+17.
    Portanto, são exigidos 18 bins consecutivos de CGM válido.
    """
    if history_steps < 1 or horizon_steps < 1:
        raise ValueError("history_steps e horizon_steps devem ser positivos.")
    required = {"valid_cgm", "clean_segment_id", *feature_names}
    missing = required - set(grid.columns)
    if missing:
        raise ValueError(f"Grade sem colunas necessárias: {sorted(missing)}")

    span = history_steps + horizon_steps
    input_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    subject_parts: list[np.ndarray] = []
    start_parts: list[pd.DatetimeIndex] = []
    end_parts: list[pd.DatetimeIndex] = []
    target_time_parts: list[pd.DatetimeIndex] = []
    segment_parts: list[np.ndarray] = []

    valid_grid = grid.loc[grid["valid_cgm"]].copy()
    for segment_id, segment in valid_grid.groupby("clean_segment_id", sort=False):
        if len(segment) < span:
            continue
        values = segment.loc[:, feature_names].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(
                f"Subject {subject}, segmento {segment_id}: feature ausente em janela válida."
            )

        number_of_windows = len(segment) - span + 1
        raw_windows = np.lib.stride_tricks.sliding_window_view(
            values, window_shape=history_steps, axis=0
        )[:number_of_windows]
        # A saída bruta é (n, features, histórico); a rede recebe (n, histórico, features).
        inputs = np.moveaxis(raw_windows, -1, -2).copy()
        targets = values[span - 1 :, 0].copy()
        times = segment.index

        input_parts.append(inputs)
        target_parts.append(targets)
        subject_parts.append(np.full(number_of_windows, subject, dtype=np.int64))
        start_parts.append(times[:number_of_windows])
        end_parts.append(times[history_steps - 1 : history_steps - 1 + number_of_windows])
        target_time_parts.append(times[span - 1 :])
        segment_parts.append(np.full(number_of_windows, int(segment_id), dtype=np.int64))

    if not input_parts:
        return _empty_window_set(history_steps, feature_names)
    return WindowSet(
        inputs=np.concatenate(input_parts, axis=0),
        targets=np.concatenate(target_parts, axis=0),
        subjects=np.concatenate(subject_parts),
        input_start_times=pd.DatetimeIndex(np.concatenate(start_parts)),
        input_end_times=pd.DatetimeIndex(np.concatenate(end_parts)),
        target_times=pd.DatetimeIndex(np.concatenate(target_time_parts)),
        segment_ids=np.concatenate(segment_parts),
        feature_names=feature_names,
    )


def combine_window_sets(window_sets: Sequence[WindowSet]) -> WindowSet:
    """Combina janelas de vários participantes, preservando o identificador de origem."""
    non_empty = [window_set for window_set in window_sets if window_set.size]
    if not non_empty:
        history_steps = window_sets[0].inputs.shape[1] if window_sets else 12
        features = window_sets[0].feature_names if window_sets else FEATURE_COLUMNS
        return _empty_window_set(history_steps, features)

    features = non_empty[0].feature_names
    if any(window_set.feature_names != features for window_set in non_empty):
        raise ValueError("Não é possível combinar conjuntos com features diferentes.")
    return WindowSet(
        inputs=np.concatenate([window_set.inputs for window_set in non_empty], axis=0),
        targets=np.concatenate([window_set.targets for window_set in non_empty], axis=0),
        subjects=np.concatenate([window_set.subjects for window_set in non_empty]),
        input_start_times=pd.DatetimeIndex(
            np.concatenate([window_set.input_start_times for window_set in non_empty])
        ),
        input_end_times=pd.DatetimeIndex(
            np.concatenate([window_set.input_end_times for window_set in non_empty])
        ),
        target_times=pd.DatetimeIndex(
            np.concatenate([window_set.target_times for window_set in non_empty])
        ),
        segment_ids=np.concatenate([window_set.segment_ids for window_set in non_empty]),
        feature_names=features,
    )


def make_subject_split(
    subjects: Sequence[int],
    train_size: int = 18,
    validation_size: int = 2,
    test_size: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Cria uma divisão populacional reprodutível e sem interseção entre pacientes."""
    unique_subjects = np.array(sorted(set(int(subject) for subject in subjects)))
    expected_total = train_size + validation_size + test_size
    if len(unique_subjects) != expected_total:
        raise ValueError(
            f"A divisão solicita {expected_total} participantes, mas recebeu {len(unique_subjects)}."
        )

    shuffled = unique_subjects.copy()
    np.random.default_rng(seed).shuffle(shuffled)
    roles = ["train"] * train_size + ["validation"] * validation_size + ["test"] * test_size
    split = pd.DataFrame({"subject": shuffled, "role": roles}).sort_values("subject")
    split = split.reset_index(drop=True)
    if split["subject"].duplicated().any():
        raise AssertionError("A divisão contém participantes repetidos.")
    return split


def compute_persistence_baseline(window_set: WindowSet) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Avalia a previsão y(t+30) = CGM(t)."""
    if window_set.size == 0:
        raise ValueError("Não há janelas para avaliar a baseline.")

    result = window_set.metadata_frame()
    result["prediction_cgm"] = window_set.inputs[:, -1, 0]
    result["absolute_error"] = np.abs(result["target_cgm"] - result["prediction_cgm"])
    result["squared_error"] = (result["target_cgm"] - result["prediction_cgm"]) ** 2

    grouped = result.groupby("subject", sort=True)
    per_subject = grouped.agg(
        windows=("target_cgm", "size"),
        mae=("absolute_error", "mean"),
        mse=("squared_error", "mean"),
    ).reset_index()
    per_subject["rmse"] = np.sqrt(per_subject.pop("mse"))

    overall = pd.DataFrame(
        {
            "subject": ["overall_pooled"],
            "windows": [len(result)],
            "mae": [result["absolute_error"].mean()],
            "rmse": [np.sqrt(result["squared_error"].mean())],
        }
    )
    macro = pd.DataFrame(
        {
            "subject": ["overall_macro"],
            "windows": [len(per_subject)],
            "mae": [per_subject["mae"].mean()],
            "rmse": [per_subject["rmse"].mean()],
        }
    )
    metrics = pd.concat([per_subject, overall, macro], ignore_index=True)
    return result, metrics


def validate_window_set(
    window_set: WindowSet,
    history_steps: int,
    horizon_steps: int,
    frequency: str = "5min",
) -> pd.DataFrame:
    """Executa invariantes básicos antes do treinamento."""
    expected_shape = (
        window_set.inputs.ndim == 3
        and window_set.inputs.shape[1] == history_steps
        and window_set.inputs.shape[2] == len(window_set.feature_names)
    )
    step = pd.Timedelta(frequency)
    expected_input_duration = step * (history_steps - 1)
    expected_target_offset = step * horizon_steps
    input_durations = window_set.input_end_times - window_set.input_start_times
    target_offsets = window_set.target_times - window_set.input_end_times

    return pd.DataFrame(
        [
            {
                "check": "shape de X é (n, histórico, features)",
                "passed": bool(expected_shape),
                "detail": str(window_set.inputs.shape),
            },
            {
                "check": "X e y não possuem valores ausentes ou infinitos",
                "passed": bool(
                    np.isfinite(window_set.inputs).all() and np.isfinite(window_set.targets).all()
                ),
                "detail": f"janelas={window_set.size}",
            },
            {
                "check": "cada entrada cobre o histórico configurado",
                "passed": bool((input_durations == expected_input_duration).all()),
                "detail": f"duração esperada={expected_input_duration}",
            },
            {
                "check": "alvo ocorre exatamente no horizonte configurado",
                "passed": bool((target_offsets == expected_target_offset).all()),
                "detail": f"horizonte esperado={expected_target_offset}",
            },
        ]
    )
