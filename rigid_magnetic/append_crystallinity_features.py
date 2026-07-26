#!/usr/bin/env python3
"""Append scalar p_q4 and p_q6 crystallinity features to order-parameter data."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLUMNS = ["file_id", "lambda", "shift"]
DEFAULT_DATA_PATH = Path("results/MAG2P_order_parameters-2025-12-8-16:13:13.pickle")
DEFAULT_Q4_PATH = Path("results/MAG2P_order_parameters-2026-6-17-10:55:21.pickle")
DEFAULT_Q6_PATH = Path("results/MAG2P_order_parameters-2026-1-21-10:27:28.pickle")
STABLE_OUTPUT_PATH = Path("results/MAG2P_order_parameters_with_p_q4_p_q6.pickle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append p_q4 and p_q6 to an order-parameter pickle using exact metadata keys."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Main order-parameter pickle.")
    parser.add_argument("--q4-path", type=Path, default=DEFAULT_Q4_PATH, help="Source pickle for p_q4.")
    parser.add_argument("--q6-path", type=Path, default=DEFAULT_Q6_PATH, help="Source pickle for p_q6.")
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
        missing_counts = frame[KEY_COLUMNS].isna().sum()
        raise ValueError(f"{label} has missing merge-key values: {missing_counts.to_dict()}")
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
            f"{source_label} has inconsistent lambda or shift values for matching file_id entries. "
            "Exact joins require consistent metadata."
        )
    return {
        "shared_file_id_count": int(len(shared_file_ids)),
        "lambda_agrees": lambda_agrees,
        "shift_agrees": shift_agrees,
    }


def source_key_overlap(main: pd.DataFrame, source: pd.DataFrame) -> tuple[int, int]:
    comparison = main[KEY_COLUMNS].merge(source[KEY_COLUMNS], on=KEY_COLUMNS, how="outer", indicator=True)
    unmatched_main = int((comparison["_merge"] == "left_only").sum())
    unmatched_source = int((comparison["_merge"] == "right_only").sum())
    return unmatched_main, unmatched_source


def merge_feature(
    base: pd.DataFrame,
    source: pd.DataFrame,
    feature_column: str,
    source_label: str,
) -> tuple[pd.DataFrame, int]:
    if feature_column not in source.columns:
        raise ValueError(f"{source_label} does not contain required crystallinity column '{feature_column}'.")
    if feature_column in base.columns:
        raise ValueError(
            f"Base dataframe already contains '{feature_column}'; refusing to overwrite an existing column."
        )
    merged = base.merge(
        source[KEY_COLUMNS + [feature_column]],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
        indicator=f"_{feature_column}_merge",
        sort=False,
    )
    unmatched_base_rows = int((merged[f"_{feature_column}_merge"] == "left_only").sum())
    merged = merged.drop(columns=f"_{feature_column}_merge")
    if len(merged) != len(base):
        raise RuntimeError(
            f"{source_label} merge changed the main dataframe row count from {len(base)} to {len(merged)}."
        )
    return merged, unmatched_base_rows


def json_default(value: object):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> None:
    args = parse_args()
    data_path = resolve_input_path(args.data_path, "Main data")
    q4_path = resolve_input_path(args.q4_path, "q4 source")
    q6_path = resolve_input_path(args.q6_path, "q6 source")
    print(f"Loading main data: {data_path}")
    main_frame = normalize_identifier_name(pd.read_pickle(data_path), "Main data")
    print(f"Loading q4 source: {q4_path}")
    q4_frame = normalize_identifier_name(pd.read_pickle(q4_path), "q4 source")
    print(f"Loading q6 source: {q6_path}")
    q6_frame = normalize_identifier_name(pd.read_pickle(q6_path), "q6 source")

    main_schema = inspect_frame(main_frame, "Main data")
    q4_schema = inspect_frame(q4_frame, "q4 source")
    q6_schema = inspect_frame(q6_frame, "q6 source")
    q4_consistency = validate_metadata_consistency(main_frame, q4_frame, "q4 source")
    q6_consistency = validate_metadata_consistency(main_frame, q6_frame, "q6 source")
    q4_unmatched_main, q4_unmatched_source = source_key_overlap(main_frame, q4_frame)
    q6_unmatched_main, q6_unmatched_source = source_key_overlap(main_frame, q6_frame)

    after_q4, unmatched_q4_rows = merge_feature(main_frame, q4_frame, "p_q4", "q4 source")
    enriched_frame, unmatched_q6_rows = merge_feature(after_q4, q6_frame, "p_q6", "q6 source")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    stable_output = STABLE_OUTPUT_PATH.resolve()
    archival_output = stable_output.with_name(f"{stable_output.stem}-{timestamp}{stable_output.suffix}")
    report_output = stable_output.with_name(f"{stable_output.stem}_merge_report-{timestamp}.json")
    stable_output.parent.mkdir(parents=True, exist_ok=True)
    enriched_frame.to_pickle(stable_output)
    enriched_frame.to_pickle(archival_output)

    report = {
        "resolved_input_paths": {"main": data_path, "q4": q4_path, "q6": q6_path},
        "stable_output_path": stable_output,
        "archival_output_path": archival_output,
        "merge_report_path": report_output,
        "merge_keys": KEY_COLUMNS,
        "merge_method": "exact left join with validate='one_to_one'; no rounding or approximate matching",
        "schemas": {"main": main_schema, "q4": q4_schema, "q6": q6_schema},
        "metadata_consistency": {"q4": q4_consistency, "q6": q6_consistency},
        "row_counts": {
            "main_before_merges": len(main_frame),
            "after_p_q4_merge": len(after_q4),
            "after_p_q6_merge": len(enriched_frame),
        },
        "unmatched_rows": {
            "main_rows_without_q4": unmatched_q4_rows,
            "main_rows_without_q6": unmatched_q6_rows,
            "q4_source_rows_not_in_main": q4_unmatched_source,
            "q6_source_rows_not_in_main": q6_unmatched_source,
            "q4_key_overlap_main_rows_unmatched": q4_unmatched_main,
            "q6_key_overlap_main_rows_unmatched": q6_unmatched_main,
        },
        "final_missing_values": {
            "p_q4": int(enriched_frame["p_q4"].isna().sum()),
            "p_q6": int(enriched_frame["p_q6"].isna().sum()),
        },
        "file_id_shift_lambda_mutually_consistent": bool(
            q4_consistency["lambda_agrees"]
            and q4_consistency["shift_agrees"]
            and q6_consistency["lambda_agrees"]
            and q6_consistency["shift_agrees"]
        ),
    }
    with report_output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=json_default)

    print(f"Wrote stable enriched pickle: {stable_output}")
    print(f"Wrote archival enriched pickle: {archival_output}")
    print(f"Wrote merge report: {report_output}")
    print(
        "Merge summary: "
        f"rows={len(enriched_frame)}, unmatched q4={unmatched_q4_rows}, "
        f"unmatched q6={unmatched_q6_rows}, missing p_q4={report['final_missing_values']['p_q4']}, "
        f"missing p_q6={report['final_missing_values']['p_q6']}"
    )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
