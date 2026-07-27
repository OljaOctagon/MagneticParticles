#!/usr/bin/env python3
"""Append scalar and coarse q4/q6 crystallinity features to order-parameter data."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


KEY_COLUMNS = ["file_id", "lambda", "shift"]
DEFAULT_DATA_PATH = Path("results/MAG2P_order_parameters-2025-12-8-16:13:13.pickle")
DEFAULT_Q4_PATH = Path("results/MAG2P_order_parameters-2026-6-17-10:55:21.pickle")
DEFAULT_Q6_PATH = Path("results/MAG2P_order_parameters-2026-1-21-10:27:28.pickle")
DEFAULT_HISTOGRAM_PATH = Path("results/MAG2P_order_parameters-2026-2-16-16:25:4.pickle")
STABLE_OUTPUT_PATH = Path("results/MAG2P_order_parameters_with_crystallinity.pickle")
FINE_BIN_WIDTH = 0.02
FINE_BIN_COUNT = 50
COARSE_BIN_COUNT = 10
FINE_BINS_PER_COARSE_BIN = 5
Q4_FINE_COLUMNS = [f"q4_{round(FINE_BIN_WIDTH * index, 2)}" for index in range(FINE_BIN_COUNT)]
Q4_COARSE_COLUMNS = [f"q4_hist_{start:02d}_{start + 10:02d}" for start in range(0, 100, 10)]
Q6_COARSE_COLUMNS = [f"q6_hist_{start:02d}_{start + 10:02d}" for start in range(0, 100, 10)]
COARSE_COLUMNS = Q4_COARSE_COLUMNS + Q6_COARSE_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append scalar and coarse q4/q6 crystallinity features using exact metadata keys."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Main order-parameter pickle.")
    parser.add_argument("--q4-path", type=Path, default=DEFAULT_Q4_PATH, help="Source pickle for p_q4.")
    parser.add_argument("--q6-path", type=Path, default=DEFAULT_Q6_PATH, help="Source pickle for p_q6.")
    parser.add_argument(
        "--histogram-path",
        type=Path,
        default=DEFAULT_HISTOGRAM_PATH,
        help="Source pickle containing fine q4/q6 histograms.",
    )
    return parser.parse_args()


def resolve_input_path(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} pickle does not exist: {resolved}")
    return resolved


def normalize_identifier_name(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Normalize the only accepted harmless identifier-name variation."""
    has_file_id = "file_id" in frame.columns
    has_fileid = "fileid" in frame.columns
    if has_file_id and has_fileid:
        raise ValueError(f"{label} contains both 'file_id' and 'fileid'; refusing an ambiguous schema.")
    if has_fileid:
        frame = frame.rename(columns={"fileid": "file_id"})

    missing = [column for column in KEY_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required merge-key columns: {missing}")
    if frame[KEY_COLUMNS].isna().any().any():
        raise ValueError(f"{label} has missing merge-key values: {frame[KEY_COLUMNS].isna().sum().to_dict()}")
    return frame


def inspect_frame(frame: pd.DataFrame, label: str) -> dict[str, object]:
    duplicate_key_count = int(frame.duplicated(KEY_COLUMNS).sum())
    duplicate_file_id_count = int(frame.duplicated("file_id").sum())
    print(
        f"{label}: {len(frame)} rows, {len(frame.columns)} columns, "
        f"duplicate file_id={duplicate_file_id_count}, duplicate exact keys={duplicate_key_count}"
    )
    if duplicate_key_count:
        raise ValueError(f"{label} has {duplicate_key_count} duplicate exact merge keys: {KEY_COLUMNS}")
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "duplicate_file_id_count": duplicate_file_id_count,
        "duplicate_key_count": duplicate_key_count,
        "file_id_globally_unique": duplicate_file_id_count == 0,
    }


def validate_metadata_consistency(main: pd.DataFrame, source: pd.DataFrame, source_label: str) -> dict[str, object]:
    main_by_file = main.set_index("file_id")[["lambda", "shift"]].sort_index()
    source_by_file = source.set_index("file_id")[["lambda", "shift"]].sort_index()
    shared_file_ids = main_by_file.index.intersection(source_by_file.index)
    comparison = main_by_file.loc[shared_file_ids].eq(source_by_file.loc[shared_file_ids])
    lambda_agrees = bool(comparison["lambda"].all())
    shift_agrees = bool(comparison["shift"].all())
    if not lambda_agrees or not shift_agrees:
        raise ValueError(
            f"{source_label} has inconsistent lambda or shift values for matching file_id entries."
        )
    return {
        "shared_file_id_count": int(len(shared_file_ids)),
        "lambda_agrees": lambda_agrees,
        "shift_agrees": shift_agrees,
    }


def source_key_overlap(main: pd.DataFrame, source: pd.DataFrame) -> tuple[int, int]:
    comparison = main[KEY_COLUMNS].merge(source[KEY_COLUMNS], on=KEY_COLUMNS, how="outer", indicator=True)
    return (
        int((comparison["_merge"] == "left_only").sum()),
        int((comparison["_merge"] == "right_only").sum()),
    )


def merge_payload(base: pd.DataFrame, payload: pd.DataFrame, source_label: str) -> tuple[pd.DataFrame, int]:
    payload_columns = [column for column in payload.columns if column not in KEY_COLUMNS]
    collisions = sorted(set(payload_columns) & set(base.columns))
    if collisions:
        raise ValueError(f"{source_label} would overwrite existing columns: {collisions}")
    merged = base.merge(
        payload,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
        indicator="_merge_status",
        sort=False,
    )
    unmatched_base_rows = int((merged["_merge_status"] == "left_only").sum())
    merged = merged.drop(columns="_merge_status")
    if len(merged) != len(base):
        raise RuntimeError(f"{source_label} merge changed row count from {len(base)} to {len(merged)}.")
    return merged, unmatched_base_rows


def numeric_summary(values: np.ndarray) -> dict[str, float]:
    return {"min": float(np.min(values)), "max": float(np.max(values)), "mean": float(np.mean(values))}


def histogram_normalization(values: np.ndarray) -> dict[str, object]:
    sums = values.sum(axis=1)
    integrals = sums * FINE_BIN_WIDTH
    return {
        "stored_value_summary": numeric_summary(values),
        "row_sum_range": [float(sums.min()), float(sums.max())],
        "integral_range": [float(integrals.min()), float(integrals.max())],
        "max_absolute_integral_error_from_one": float(np.max(np.abs(integrals - 1.0))),
        "interpretation": "probability_density" if np.allclose(integrals, 1.0, rtol=0, atol=1e-6) else "unknown",
    }


def validate_histogram_source(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    if len(frame) != 2760:
        raise ValueError(f"Histogram source must contain exactly 2760 rows; found {len(frame)}.")
    missing_q4 = [column for column in Q4_FINE_COLUMNS if column not in frame.columns]
    if missing_q4:
        raise ValueError(f"Histogram source is missing q4 fine-bin columns: {missing_q4}")
    actual_q4 = sorted(
        [column for column in frame.columns if isinstance(column, str) and column.startswith("q4_")],
        key=lambda column: float(column.split("_", maxsplit=1)[1]),
    )
    if actual_q4 != Q4_FINE_COLUMNS:
        raise ValueError("q4 fine-bin columns do not exactly match the expected numeric 0.02 bin layout.")
    for column in ("q6_0.0", "q6_0.02"):
        if column not in frame.columns:
            raise ValueError(f"Histogram source is missing required q6 histogram column '{column}'.")
    if frame[Q4_FINE_COLUMNS].isna().any().any() or frame["q6_0.0"].isna().any() or frame["q6_0.02"].isna().any():
        raise ValueError("Histogram source contains missing q4 or q6 histogram rows.")

    q4_values = frame[Q4_FINE_COLUMNS].to_numpy(dtype=float)
    try:
        q6_values = np.stack(frame["q6_0.0"].to_numpy()).astype(float)
        q6_edges = np.stack(frame["q6_0.02"].to_numpy()).astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("q6 histogram values or edge arrays cannot be converted to numeric arrays.") from error
    if q6_values.shape != (len(frame), FINE_BIN_COUNT):
        raise ValueError(f"Every q6_0.0 entry must have 50 values; found stacked shape {q6_values.shape}.")
    if q6_edges.shape != (len(frame), FINE_BIN_COUNT + 1):
        raise ValueError(f"Every q6_0.02 entry must have 51 edges; found stacked shape {q6_edges.shape}.")
    if not np.all(q6_edges == q6_edges[0]):
        raise ValueError("q6 bin-edge arrays are not identical across rows.")
    expected_edges = np.linspace(0.0, 1.0, FINE_BIN_COUNT + 1)
    if not np.allclose(q6_edges[0], expected_edges, rtol=1e-6, atol=1e-8):
        raise ValueError("q6 bin edges do not span 0-1 with 0.02 spacing.")
    q4_left_edges = np.arange(FINE_BIN_COUNT) * FINE_BIN_WIDTH
    if not np.allclose(q4_left_edges, q6_edges[0, :-1], rtol=1e-6, atol=1e-8):
        raise ValueError("q4 and q6 fine histograms do not share the same bin layout.")
    if not np.isfinite(q4_values).all() or not np.isfinite(q6_values).all():
        raise ValueError("Histogram values must be finite.")
    if (q4_values < 0).any() or (q6_values < 0).any():
        raise ValueError("Histogram values must be non-negative.")

    report = {
        "q4_columns": Q4_FINE_COLUMNS,
        "q6_value_column": "q6_0.0",
        "q6_edge_column": "q6_0.02",
        "q4_shape": list(q4_values.shape),
        "q6_shape": list(q6_values.shape),
        "q6_edges": q6_edges[0].tolist(),
        "q6_edges_identical_across_rows": True,
        "q4_q6_bin_layout_matches": True,
        "finite": True,
        "non_negative": True,
        "missing_histogram_rows": {"q4": 0, "q6": 0},
        "normalization": {"q4": histogram_normalization(q4_values), "q6": histogram_normalization(q6_values)},
    }
    return q4_values, q6_values, q6_edges[0], report


def coarse_grain(fine_values: np.ndarray, names: list[str], fine_names: list[str] | None = None) -> tuple[np.ndarray, dict[str, object]]:
    coarse_values = fine_values.reshape(len(fine_values), COARSE_BIN_COUNT, FINE_BINS_PER_COARSE_BIN).sum(axis=2)
    if not np.allclose(coarse_values.sum(axis=1), fine_values.sum(axis=1), rtol=0, atol=1e-12):
        raise RuntimeError("Coarse histogram sums do not reproduce fine-bin sums.")
    mapping = {}
    for index, name in enumerate(names):
        start = index * FINE_BINS_PER_COARSE_BIN
        mapping[name] = {
            "fine_indices": list(range(start, start + FINE_BINS_PER_COARSE_BIN)),
            "fine_columns": fine_names[start : start + FINE_BINS_PER_COARSE_BIN] if fine_names else None,
            "interval": f"{index / 10:.1f}-{(index + 1) / 10:.1f}",
        }
    return coarse_values, {
        "columns": names,
        "mapping": mapping,
        "max_absolute_sum_difference": float(np.max(np.abs(coarse_values.sum(axis=1) - fine_values.sum(axis=1)))),
        "coarse_value_summary": numeric_summary(coarse_values),
    }


def representative_levels(values: np.ndarray) -> np.ndarray:
    return np.unique(values[np.linspace(0, len(values) - 1, 3, dtype=int)])


def select_representatives(frame: pd.DataFrame, lambdas: np.ndarray, shifts: np.ndarray) -> list[dict[str, object]]:
    representatives = []
    for lambda_value in lambdas:
        for shift_value in shifts:
            matches = frame.loc[(frame["lambda"] == lambda_value) & (frame["shift"] == shift_value)].sort_values("file_id")
            if matches.empty:
                raise ValueError(f"No histogram row found for lambda={lambda_value}, shift={shift_value}.")
            row = matches.iloc[0]
            representatives.append({"file_id": row["file_id"], "lambda": row["lambda"], "shift": row["shift"]})
    return representatives


def histogram_row(frame: pd.DataFrame, file_id: object, q4_values: np.ndarray, q6_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    index = int(frame.index[frame["file_id"] == file_id][0])
    return q4_values[index], q6_values[index]


def save_representative_histograms(
    output_path: Path,
    frame: pd.DataFrame,
    representatives: list[dict[str, object]],
    q4_values: np.ndarray,
    q6_values: np.ndarray,
    edges: np.ndarray,
) -> None:
    columns = 3
    rows = int(np.ceil(len(representatives) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 3.5 * rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    centers = (edges[:-1] + edges[1:]) / 2
    ymax = max(float(q4_values.max()), float(q6_values.max())) * 1.05
    for axis, state in zip(axes, representatives):
        q4_row, q6_row = histogram_row(frame, state["file_id"], q4_values, q6_values)
        axis.step(centers, q4_row, where="mid", color="#2a6fbb", label="q4")
        axis.step(centers, q6_row, where="mid", color="#d95f02", label="q6")
        axis.set_title(f"{state['file_id']}\nshift={state['shift']}, λ*={state['lambda']}", fontsize=9)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, ymax)
        axis.set_xlabel("q_l value")
        axis.set_ylabel("Stored histogram value")
    for axis in axes[len(representatives) :]:
        axis.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2)
    fig.suptitle("Representative fine q4 and q6 histograms", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_fine_coarse_validation(
    output_path: Path,
    frame: pd.DataFrame,
    representatives: list[dict[str, object]],
    q4_values: np.ndarray,
    q6_values: np.ndarray,
    q4_coarse: np.ndarray,
    q6_coarse: np.ndarray,
    edges: np.ndarray,
) -> None:
    selected = representatives[:4]
    fig, axes = plt.subplots(2, len(selected), figsize=(5.0 * len(selected), 6.0), sharex=True)
    centers = (edges[:-1] + edges[1:]) / 2
    coarse_edges = np.linspace(0, 1, COARSE_BIN_COUNT + 1)
    for column, state in enumerate(selected):
        index = int(frame.index[frame["file_id"] == state["file_id"]][0])
        for axis, fine, coarse, label, color in [
            (axes[0, column], q4_values[index], q4_coarse[index], "q4", "#2a6fbb"),
            (axes[1, column], q6_values[index], q6_coarse[index], "q6", "#d95f02"),
        ]:
            axis.step(centers, fine, where="mid", color=color, label=f"fine {label}")
            axis.stairs(coarse, coarse_edges, color="black", linewidth=1.5, label=f"coarse {label} five-bin sum")
            axis.set_xlim(0, 1)
            axis.set_xlabel("q_l value")
            axis.set_ylabel("Stored value")
        axes[0, column].set_title(f"{state['file_id']}\nshift={state['shift']}, λ*={state['lambda']}", fontsize=9)
    q4_handles, q4_labels = axes[0, 0].get_legend_handles_labels()
    q6_handles, q6_labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(q4_handles + q6_handles, q4_labels + q6_labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=4)
    fig.suptitle("Fine-versus-coarse histogram validation", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_state_variation(
    output_path: Path,
    frame: pd.DataFrame,
    shifts: np.ndarray,
    q_values: np.ndarray,
    edges: np.ndarray,
    label: str,
) -> list[dict[str, object]]:
    lambdas = np.sort(frame["lambda"].unique())
    centers = (edges[:-1] + edges[1:]) / 2
    fig, axes = plt.subplots(1, len(shifts), figsize=(5.4 * len(shifts), 4.2), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap("viridis")
    selected = []
    ymax = float(q_values.max()) * 1.05
    for axis, shift_value in zip(axes, shifts):
        for index, lambda_value in enumerate(lambdas):
            row = frame.loc[(frame["lambda"] == lambda_value) & (frame["shift"] == shift_value)].sort_values("file_id").iloc[0]
            row_index = int(frame.index[frame["file_id"] == row["file_id"]][0])
            axis.step(centers, q_values[row_index], where="mid", color=cmap(index / max(len(lambdas) - 1, 1)), alpha=0.7)
            selected.append({"file_id": row["file_id"], "lambda": row["lambda"], "shift": row["shift"]})
        axis.set(title=f"shift={shift_value}", xlim=(0, 1), ylim=(0, ymax), xlabel="q_l value")
    axes[0].set_ylabel("Stored histogram value")
    scalar_map = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(float(lambdas.min()), float(lambdas.max())))
    fig.colorbar(scalar_map, ax=axes, label="λ*")
    fig.suptitle(f"{label} histogram variation across λ*", y=1.02)
    fig.subplots_adjust(left=0.06, right=0.9, bottom=0.16, top=0.82, wspace=0.22)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return selected


def create_diagnostics(
    timestamp: str,
    frame: pd.DataFrame,
    q4_values: np.ndarray,
    q6_values: np.ndarray,
    q4_coarse: np.ndarray,
    q6_coarse: np.ndarray,
    edges: np.ndarray,
) -> tuple[Path, dict[str, Path], dict[str, object]]:
    directory = Path("results") / f"crystallinity_histogram_diagnostics_{timestamp}"
    directory.mkdir(parents=True, exist_ok=False)
    lambdas = representative_levels(np.sort(frame["lambda"].unique()))
    shifts = representative_levels(np.sort(frame["shift"].unique()))
    representatives = select_representatives(frame, lambdas, shifts)
    paths = {
        "representative_paired_histograms": directory / "representative_q4_q6_histograms.pdf",
        "fine_vs_coarse": directory / "fine_vs_coarse_histograms.pdf",
        "q4_state_variation": directory / "q4_state_variation.pdf",
        "q6_state_variation": directory / "q6_state_variation.pdf",
    }
    save_representative_histograms(paths["representative_paired_histograms"], frame, representatives, q4_values, q6_values, edges)
    save_fine_coarse_validation(paths["fine_vs_coarse"], frame, representatives, q4_values, q6_values, q4_coarse, q6_coarse, edges)
    q4_overview = save_state_variation(paths["q4_state_variation"], frame, shifts, q4_values, edges, "q4")
    q6_overview = save_state_variation(paths["q6_state_variation"], frame, shifts, q6_values, edges, "q6")
    return directory, paths, {
        "representative_grid_lambdas": lambdas.tolist(),
        "representative_grid_shifts": shifts.tolist(),
        "representative_state_points": representatives,
        "fine_vs_coarse_state_points": representatives[:4],
        "q4_state_variation_points": q4_overview,
        "q6_state_variation_points": q6_overview,
    }


def json_default(value: object):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> None:
    args = parse_args()
    data_path = resolve_input_path(args.data_path, "Main data")
    q4_path = resolve_input_path(args.q4_path, "q4 scalar source")
    q6_path = resolve_input_path(args.q6_path, "q6 scalar source")
    histogram_path = resolve_input_path(args.histogram_path, "Histogram source")
    main_frame = normalize_identifier_name(pd.read_pickle(data_path), "Main data")
    q4_frame = normalize_identifier_name(pd.read_pickle(q4_path), "q4 scalar source")
    q6_frame = normalize_identifier_name(pd.read_pickle(q6_path), "q6 scalar source")
    histogram_frame = normalize_identifier_name(pd.read_pickle(histogram_path), "Histogram source")

    schemas = {
        "main": inspect_frame(main_frame, "Main data"),
        "q4": inspect_frame(q4_frame, "q4 scalar source"),
        "q6": inspect_frame(q6_frame, "q6 scalar source"),
        "histogram": inspect_frame(histogram_frame, "Histogram source"),
    }
    consistency = {
        "q4": validate_metadata_consistency(main_frame, q4_frame, "q4 scalar source"),
        "q6": validate_metadata_consistency(main_frame, q6_frame, "q6 scalar source"),
        "histogram": validate_metadata_consistency(main_frame, histogram_frame, "Histogram source"),
    }
    overlaps = {
        label: source_key_overlap(main_frame, frame)
        for label, frame in {"q4": q4_frame, "q6": q6_frame, "histogram": histogram_frame}.items()
    }
    q4_values, q6_values, edges, histogram_validation = validate_histogram_source(histogram_frame)
    q4_coarse, q4_coarse_report = coarse_grain(q4_values, Q4_COARSE_COLUMNS, Q4_FINE_COLUMNS)
    q6_coarse, q6_coarse_report = coarse_grain(q6_values, Q6_COARSE_COLUMNS)
    histogram_payload = histogram_frame[KEY_COLUMNS].copy()
    histogram_payload[Q4_COARSE_COLUMNS] = q4_coarse
    histogram_payload[Q6_COARSE_COLUMNS] = q6_coarse

    scalar_q4_payload = q4_frame[KEY_COLUMNS + ["p_q4"]]
    scalar_q6_payload = q6_frame[KEY_COLUMNS + ["p_q6"]]
    after_q4, unmatched_q4 = merge_payload(main_frame, scalar_q4_payload, "q4 scalar source")
    after_q6, unmatched_q6 = merge_payload(after_q4, scalar_q6_payload, "q6 scalar source")
    enriched_frame, unmatched_histogram = merge_payload(after_q6, histogram_payload, "Histogram source")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    stable_output = STABLE_OUTPUT_PATH.resolve()
    archival_output = stable_output.with_name(f"{stable_output.stem}-{timestamp}{stable_output.suffix}")
    report_output = stable_output.with_name(f"{stable_output.stem}_merge_report-{timestamp}.json")
    stable_output.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_directory, diagnostic_paths, selection_report = create_diagnostics(
        timestamp, histogram_frame, q4_values, q6_values, q4_coarse, q6_coarse, edges
    )
    enriched_frame.to_pickle(stable_output)
    enriched_frame.to_pickle(archival_output)

    report = {
        "resolved_input_paths": {"main": data_path, "q4": q4_path, "q6": q6_path, "histogram": histogram_path},
        "stable_output_path": stable_output,
        "archival_output_path": archival_output,
        "merge_report_path": report_output,
        "diagnostic_directory": diagnostic_directory,
        "diagnostic_plot_paths": diagnostic_paths,
        "merge_keys": KEY_COLUMNS,
        "merge_method": "exact left join with validate='one_to_one'; no rounding or approximate matching",
        "schemas": schemas,
        "metadata_consistency": consistency,
        "row_counts": {"main_before_merges": len(main_frame), "after_p_q4_merge": len(after_q4), "after_p_q6_merge": len(after_q6), "after_histogram_merge": len(enriched_frame)},
        "unmatched_rows": {
            "main_rows_without_q4": unmatched_q4,
            "main_rows_without_q6": unmatched_q6,
            "main_rows_without_histograms": unmatched_histogram,
            **{f"{label}_source_rows_not_in_main": overlap[1] for label, overlap in overlaps.items()},
        },
        "final_missing_values": {column: int(enriched_frame[column].isna().sum()) for column in ["p_q4", "p_q6", *COARSE_COLUMNS]},
        "histogram_validation": histogram_validation,
        "coarse_graining": {"operation": "sum five adjacent stored fine-bin values; no renormalization", "q4": q4_coarse_report, "q6": q6_coarse_report},
        "representative_selection": selection_report,
    }
    with report_output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=json_default)
    print(f"Wrote stable enriched pickle: {stable_output}")
    print(f"Wrote archival enriched pickle: {archival_output}")
    print(f"Wrote diagnostics: {diagnostic_directory}")
    print(f"Wrote merge report: {report_output}")
    print(f"Merge summary: rows={len(enriched_frame)}, missing coarse values={sum(report['final_missing_values'][column] for column in COARSE_COLUMNS)}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
