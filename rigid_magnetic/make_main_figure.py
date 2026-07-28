#!/usr/bin/env python3
"""Create the publication main figure for the reduced spectral embedding."""

from __future__ import annotations

import argparse
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, PowerNorm, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch

from feature_schema import (
    CRYSTALLINITY_COARSE_HISTOGRAM_FEATURES,
    CRYSTALLINITY_SCALARS,
    FEATURE_GROUPS as SCHEMA_FEATURE_GROUPS,
    Q4_HISTOGRAM_FEATURES,
    Q6_HISTOGRAM_FEATURES,
)


EXPECTED_K = 32
EXPECTED_SEED = 42
FEATURE_SET = "reduced_no_global"
COORDINATE_COUNT = 3
LAMBDA_COLOR_GAMMA = 0.5
LAMBDA_COLOR_LIMITS = (0.0, 30.0)
MISSING_CELL_COLOR = "#d0d0d0"
ORIENTATION_CORRELATION_EPSILON = 0.05

# Set this to an explicit list to override metadata-driven descriptor icons.
DESCRIPTOR_PANEL_CONFIG: list[dict[str, str]] | None = None

GROUP_DESCRIPTOR_DEFAULTS = {
    "orientation": {"name": "Orientation distribution", "type": "histogram"},
    "gofr": {"name": r"$g(r)$", "type": "radial_distribution"},
    "Rg": {"name": r"$R_g$ distribution", "type": "histogram"},
    "q4_coarse_histogram_features": {"name": r"$q_4$", "type": "histogram"},
    "q6_coarse_histogram_features": {"name": r"$q_6$", "type": "histogram"},
    "crystallinity_scalar_features": {"name": "Crystallinity", "type": "scalar"},
    "crystallinity_coarse_histogram_features": {
        "name": r"$q_4$, $q_6$ distributions",
        "type": "histogram",
    },
    "global": {"name": "Global descriptors", "type": "scalar"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a 2 x 3 publication figure from reduced spectral-embedding results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Analysis run directory containing data/run_metadata.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/main_spectral_embedding"),
        help="Output base path without a file extension.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Pickle input used only when embedding-coordinate CSV output is unavailable.",
    )
    parser.add_argument("--k", type=int, default=EXPECTED_K, help="Expected k value (default: 32).")
    parser.add_argument(
        "--seed", type=int, default=EXPECTED_SEED, help="Expected random seed (default: 42)."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow replacing existing figure outputs."
    )
    return parser.parse_args()


def resolve_data_directory(results_dir: Path) -> Path:
    results_dir = results_dir.expanduser().resolve()
    data_dir = results_dir / "data"
    if data_dir.is_dir():
        return data_dir
    if results_dir.name == "data" and results_dir.is_dir():
        return results_dir
    raise FileNotFoundError(
        f"Results directory must contain a data subdirectory: {results_dir}"
    )


def load_run_metadata(data_dir: Path) -> dict[str, Any]:
    metadata_path = data_dir / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Required run metadata is missing: {metadata_path}")
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    for field in ("selected_k", "random_seed", "feature_columns", "state_lambda_plot_limits"):
        if field not in metadata:
            raise ValueError(f"Run metadata is missing required field '{field}'.")
    return metadata


def validate_run_parameters(metadata: dict[str, Any], args: argparse.Namespace) -> int:
    if args.k != EXPECTED_K:
        raise ValueError(
            f"This main-figure configuration is defined for k={EXPECTED_K}; got --k={args.k}."
        )
    if args.seed != EXPECTED_SEED:
        raise ValueError(
            f"This main-figure configuration is defined for seed {EXPECTED_SEED}; got --seed={args.seed}."
        )
    selected_k = int(metadata["selected_k"])
    random_seed = int(metadata["random_seed"])
    if selected_k != args.k:
        raise ValueError(
            f"Run metadata selected_k={selected_k}, but this figure requires k={args.k}."
        )
    if random_seed != args.seed:
        raise ValueError(
            f"Run metadata random_seed={random_seed}, but this figure requires seed {args.seed}."
        )
    return selected_k


def selected_feature_definition(metadata: dict[str, Any]) -> tuple[list[str], list[str], str]:
    feature_columns = metadata["feature_columns"]
    if FEATURE_SET not in feature_columns or not isinstance(feature_columns[FEATURE_SET], list):
        raise ValueError(
            f"Run metadata does not declare concrete columns for feature set '{FEATURE_SET}'."
        )
    selected_columns = list(feature_columns[FEATURE_SET])
    if not selected_columns:
        raise ValueError(f"Feature set '{FEATURE_SET}' has no selected columns.")

    feature_set_groups = metadata.get("feature_set_groups")
    feature_groups = metadata.get("feature_groups")
    if isinstance(feature_set_groups, dict) and isinstance(feature_groups, dict):
        selected_groups = feature_set_groups.get(FEATURE_SET)
        if not isinstance(selected_groups, list) or not selected_groups:
            raise ValueError(
                f"Run metadata does not declare concrete groups for feature set '{FEATURE_SET}'."
            )
        unknown_groups = [group for group in selected_groups if group not in feature_groups]
        if unknown_groups:
            raise ValueError(f"Run metadata references unknown feature groups: {unknown_groups}")
        flattened_columns = [
            column for group in selected_groups for column in feature_groups[group]
        ]
        if flattened_columns != selected_columns:
            raise ValueError(
                "Run metadata feature groups and selected feature columns do not agree for "
                f"'{FEATURE_SET}'."
            )
        return list(selected_groups), selected_columns, "run_metadata"

    return [], selected_columns, "legacy_columns_only"


def schema_feature_groups() -> dict[str, list[str]]:
    """Return named schema groups for interpreting legacy column-only metadata."""
    return {
        **{name: list(columns) for name, columns in SCHEMA_FEATURE_GROUPS.items()},
        "q4_coarse_histogram_features": list(Q4_HISTOGRAM_FEATURES),
        "q6_coarse_histogram_features": list(Q6_HISTOGRAM_FEATURES),
        "crystallinity_scalar_features": list(CRYSTALLINITY_SCALARS),
        "crystallinity_coarse_histogram_features": list(
            CRYSTALLINITY_COARSE_HISTOGRAM_FEATURES
        ),
    }


def infer_groups_from_columns(selected_columns: list[str]) -> list[str]:
    """Identify complete schema groups from legacy metadata without using its set name."""
    groups = schema_feature_groups()
    matched_groups: list[str] = []
    remaining = set(selected_columns)
    preferred_order = [
        "global",
        "orientation",
        "Rg",
        "gofr",
        "crystallinity_scalar_features",
        "q4_coarse_histogram_features",
        "q6_coarse_histogram_features",
    ]
    for name in preferred_order:
        columns = set(groups[name])
        if columns and columns.issubset(remaining):
            matched_groups.append(name)
            remaining.difference_update(columns)
    if remaining:
        raise ValueError(
            "Legacy metadata contains selected columns that cannot be assigned to complete "
            f"schema groups: {sorted(remaining)}"
        )
    return matched_groups


def descriptor_panels_for_groups(selected_groups: list[str]) -> list[dict[str, str]]:
    if DESCRIPTOR_PANEL_CONFIG is not None:
        return [dict(panel) for panel in DESCRIPTOR_PANEL_CONFIG]
    panels = []
    for group in selected_groups:
        panel = GROUP_DESCRIPTOR_DEFAULTS.get(
            group, {"name": group.replace("_", " "), "type": "histogram"}
        )
        panels.append({**panel, "group": group})
    return panels


def coordinate_file(data_dir: Path, selected_k: int) -> Path:
    return data_dir / f"reduced_embedding_coordinates_k{selected_k}.csv"


def validate_coordinate_table(
    coordinates: pd.DataFrame, selected_k: int, expected_rows: int | None
) -> pd.DataFrame:
    coordinate_columns = [
        "file_id",
        "lambda",
        "shift",
        *[f"spectral_coordinate_{index}" for index in range(1, COORDINATE_COUNT + 1)],
    ]
    missing_columns = [column for column in coordinate_columns if column not in coordinates]
    if missing_columns:
        raise ValueError(f"Embedding-coordinate data is missing columns: {missing_columns}")
    if "k" in coordinates and not coordinates["k"].eq(selected_k).all():
        raise ValueError("Embedding-coordinate data contains a k value different from run metadata.")
    if "feature_set" in coordinates and not coordinates["feature_set"].eq(FEATURE_SET).all():
        raise ValueError(
            f"Embedding-coordinate data contains a feature set other than '{FEATURE_SET}'."
        )
    if expected_rows is not None and len(coordinates) != expected_rows:
        raise ValueError(
            f"Embedding-coordinate data has {len(coordinates)} rows; metadata declares {expected_rows}."
        )
    numeric_columns = [column for column in coordinate_columns if column != "file_id"]
    numeric_values = coordinates[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric_values.to_numpy(dtype=float)).all():
        raise ValueError("Embedding-coordinate data contains missing or non-finite numeric values.")
    validated = coordinates.copy()
    validated[numeric_columns] = numeric_values
    return validated


def reconstruct_coordinates(
    metadata: dict[str, Any],
    data_path: Path,
    selected_k: int,
    selected_groups: list[str],
    selected_columns: list[str],
) -> pd.DataFrame:
    """Recreate coordinates only when the additive analysis CSV is unavailable."""
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Embedding coordinates are unavailable and the source pickle does not exist: {data_path}"
        )
    analysis = importlib.import_module("ML_diffusion_map_testing")
    if analysis.SEED != int(metadata["random_seed"]):
        raise ValueError(
            "The current analysis helper seed differs from the selected run metadata; "
            "cannot reproduce embedding coordinates safely."
        )
    crystallinity_mode = metadata.get("crystallinity_features", {}).get("mode")
    if not isinstance(crystallinity_mode, str):
        raise ValueError(
            "Run metadata lacks crystallinity_features.mode required for coordinate reconstruction."
        )

    df, feature_groups, _ = analysis.load_and_validate_data(data_path, crystallinity_mode)
    if not selected_groups:
        selected_groups = infer_groups_from_columns(selected_columns)
    matrices, reconstructed_columns = analysis.standardized_feature_matrices(
        df, feature_groups, {FEATURE_SET: selected_groups}
    )
    if reconstructed_columns[FEATURE_SET] != selected_columns:
        raise ValueError(
            "Current preprocessing does not reproduce the concrete reduced feature columns "
            "recorded in run metadata."
        )
    _, embeddings, _ = analysis.compute_detailed_embeddings(
        matrices, [selected_k], selected_k
    )
    return analysis.embedding_coordinate_table(
        df[analysis.META_COLUMNS].reset_index(drop=True),
        embeddings[(FEATURE_SET, selected_k)],
        FEATURE_SET,
        selected_k,
    )


def load_coordinates(
    data_dir: Path,
    metadata: dict[str, Any],
    args: argparse.Namespace,
    selected_k: int,
    selected_groups: list[str],
    selected_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    exported_path = coordinate_file(data_dir, selected_k)
    expected_rows = int(metadata.get("n_configurations", 0)) or None
    if exported_path.is_file():
        coordinates = pd.read_csv(exported_path)
        source = {"mode": "exported_csv", "path": str(exported_path.resolve())}
    else:
        configured_path = args.data_path
        metadata_path = metadata.get("input_data_path")
        data_path = configured_path or (Path(metadata_path) if metadata_path else None)
        if data_path is None:
            raise FileNotFoundError(
                f"Missing {exported_path.name}; provide --data-path to reconstruct coordinates."
            )
        data_path = data_path.expanduser().resolve()
        coordinates = reconstruct_coordinates(
            metadata, data_path, selected_k, selected_groups, selected_columns
        )
        source = {"mode": "reconstructed_from_pickle", "path": str(data_path)}
    return validate_coordinate_table(coordinates, selected_k, expected_rows), source


def correlation(values: np.ndarray, metadata_values: np.ndarray) -> float | None:
    if np.std(values) == 0.0 or np.std(metadata_values) == 0.0:
        return None
    value = float(np.corrcoef(values, metadata_values)[0, 1])
    return value if np.isfinite(value) else None


def orient_coordinates(coordinates: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Fix arbitrary spectral-vector signs with a metadata-only convention."""
    oriented = coordinates.copy()
    lambda_values = oriented["lambda"].to_numpy(dtype=float)
    shift_values = oriented["shift"].to_numpy(dtype=float)
    decisions = []
    for index in range(1, COORDINATE_COUNT + 1):
        column = f"spectral_coordinate_{index}"
        values = oriented[column].to_numpy(dtype=float)
        lambda_correlation = correlation(values, lambda_values)
        shift_correlation = correlation(values, shift_values)
        if lambda_correlation is not None and abs(lambda_correlation) > ORIENTATION_CORRELATION_EPSILON:
            sign = 1 if lambda_correlation > 0 else -1
            criterion = "positive_pearson_correlation_with_lambda"
        elif shift_correlation is not None and abs(shift_correlation) > ORIENTATION_CORRELATION_EPSILON:
            sign = 1 if shift_correlation > 0 else -1
            criterion = "positive_pearson_correlation_with_shift"
        else:
            largest_index = int(np.argmax(np.abs(values)))
            sign = 1 if values[largest_index] >= 0 else -1
            criterion = "largest_absolute_coordinate_entry_positive"
        oriented[column] = sign * values
        decisions.append(
            {
                "coordinate": index,
                "applied_sign": sign,
                "criterion": criterion,
                "raw_lambda_pearson_r": lambda_correlation,
                "raw_shift_pearson_r": shift_correlation,
                "correlation_epsilon": ORIENTATION_CORRELATION_EPSILON,
            }
        )
    return oriented, decisions


def choose_colormaps():
    try:
        import cmasher as cmr

        return cmr.rainforest, cmr.pride
    except ImportError:
        return plt.get_cmap("viridis"), plt.get_cmap("RdBu_r")


def add_panel_label(axis: plt.Axes, letter: str) -> None:
    axis.text(
        -0.16,
        1.05,
        letter,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def draw_descriptor_icon(axis: plt.Axes, panel: dict[str, str]) -> None:
    axis.set_axis_off()
    icon_type = panel["type"]
    if icon_type == "radial_distribution":
        x = np.linspace(0.0, 1.0, 120)
        y = 0.18 + 0.58 * np.exp(-3.2 * x) * (0.55 + 0.45 * np.cos(18 * x))
        axis.plot(x, y, color="#1f4e79", linewidth=1.2)
        axis.axhline(0.18, color="#909090", linewidth=0.5)
    elif icon_type == "scalar":
        x = np.array([0.22, 0.5, 0.78])
        axis.plot([0.1, 0.9], [0.45, 0.45], color="#909090", linewidth=0.6)
        axis.scatter(x, [0.36, 0.63, 0.49], s=11, color="#1f4e79", zorder=2)
    else:
        heights = np.array([0.18, 0.32, 0.58, 0.78, 0.62, 0.36, 0.2])
        axis.bar(
            np.arange(len(heights)), heights, width=0.82, color="#5b84b1", edgecolor="none"
        )
        axis.set_xlim(-0.55, len(heights) - 0.45)
    label = panel["name"]
    if label == "Orientation distribution":
        label = "Orientation\ndistribution"
    axis.text(
        0.5,
        -0.18,
        label,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=5.4,
        linespacing=0.85,
    )


def draw_knn_graph(axis: plt.Axes) -> None:
    positions = np.array(
        [[0.12, 0.52], [0.34, 0.8], [0.6, 0.75], [0.84, 0.53], [0.65, 0.22], [0.29, 0.22]]
    )
    edges = [(0, 1), (0, 5), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 5), (5, 2)]
    for start, end in edges:
        axis.plot(
            positions[[start, end], 0], positions[[start, end], 1], color="#7d8790", linewidth=0.65
        )
    axis.scatter(positions[:, 0], positions[:, 1], s=15, color="#244a68", zorder=2)


def arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=axis.transAxes,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.75,
            color="#56616a",
        )
    )


def draw_panel_a(axis: plt.Axes, descriptor_panels: list[dict[str, str]]) -> None:
    axis.set_axis_off()
    add_panel_label(axis, "A")
    if not descriptor_panels:
        raise ValueError("Panel A requires at least one descriptor panel.")
    n_columns = 2 if len(descriptor_panels) > 1 else 1
    n_rows = int(np.ceil(len(descriptor_panels) / n_columns))
    icon_width = 0.22 if n_columns == 2 else 0.28
    icon_height = 0.16 if n_rows == 1 else 0.1
    row_spacing = 0.22 if n_rows > 1 else 0.0
    descriptor_y = 0.84 if n_rows > 1 else 0.79
    for index, panel in enumerate(descriptor_panels):
        row, column = divmod(index, n_columns)
        x0 = (0.11 if n_columns == 2 else 0.36) + column * 0.42
        y0 = descriptor_y - row * row_spacing
        icon_axis = axis.inset_axes([x0, y0, icon_width, icon_height])
        draw_descriptor_icon(icon_axis, panel)

    descriptor_bottom = descriptor_y - (n_rows - 1) * row_spacing
    standard_y = descriptor_bottom - (0.26 if n_rows > 1 else 0.32)
    graph_y = standard_y - (0.19 if n_rows > 1 else 0.22)
    embedding_y = graph_y - 0.12
    standard_x = 0.5
    axis.add_patch(
        plt.Rectangle((standard_x - 0.14, standard_y), 0.28, 0.12, transform=axis.transAxes, fill=False, linewidth=0.7, edgecolor="#56616a")
    )
    axis.text(standard_x, standard_y + 0.06, "standardize", transform=axis.transAxes, ha="center", va="center", fontsize=5.6)
    graph_axis = axis.inset_axes([0.39, graph_y, 0.22, 0.14])
    graph_axis.set_axis_off()
    draw_knn_graph(graph_axis)
    axis.text(0.5, graph_y - 0.025, r"kNN graph ($k=32$)", transform=axis.transAxes, ha="center", va="top", fontsize=5.5)
    axis.text(0.38, embedding_y, "spectral embedding", transform=axis.transAxes, ha="center", va="center", fontsize=5.6)
    output_axis = axis.inset_axes([0.62, embedding_y - 0.075, 0.12, 0.11])
    output_axis.set_axis_off()
    output_axis.plot([0.12, 0.88], [0.2, 0.75], color="#1f4e79", linewidth=0.9)
    output_axis.plot([0.12, 0.88], [0.42, 0.24], color="#b55d3d", linewidth=0.9)
    axis.text(0.8, embedding_y, r"$(\psi_1, \psi_2, \psi_3)$", transform=axis.transAxes, ha="center", va="center", fontsize=5.3)
    for column in range(n_columns):
        descriptor_center = (0.22 if n_columns == 2 else 0.5) + column * 0.42
        arrow(axis, (descriptor_center, descriptor_bottom - 0.015), (0.5, standard_y + 0.12))
    arrow(axis, (0.5, standard_y), (0.5, graph_y + 0.14))
    arrow(axis, (0.5, graph_y), (0.5, embedding_y + 0.025))


def scatter_limits(coordinates: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    limits = []
    for column in ("spectral_coordinate_1", "spectral_coordinate_2"):
        values = coordinates[column].to_numpy(dtype=float)
        lower, upper = float(values.min()), float(values.max())
        padding = max((upper - lower) * 0.04, 1e-12)
        limits.append((lower - padding, upper + padding))
    return limits[0], limits[1]


def coordinate_state_grids(
    coordinates: pd.DataFrame, lambda_limits: tuple[float, float]
) -> tuple[list[pd.DataFrame], np.ndarray, np.ndarray]:
    visible = coordinates.loc[coordinates["lambda"].between(*lambda_limits)].copy()
    if visible.empty:
        raise ValueError("No coordinate data falls within the metadata lambda plot limits.")
    lambda_values = np.sort(visible["lambda"].unique())
    shift_values = np.sort(visible["shift"].unique())
    grids = []
    for index in range(1, COORDINATE_COUNT + 1):
        grid = (
            visible.groupby(["lambda", "shift"], sort=True)[f"spectral_coordinate_{index}"]
            .mean()
            .unstack("shift")
            .reindex(index=lambda_values, columns=shift_values)
        )
        grids.append(grid)
    return grids, lambda_values, shift_values


def cell_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5])
    midpoints = (values[:-1] + values[1:]) / 2.0
    return np.concatenate(
        ([values[0] - (midpoints[0] - values[0])], midpoints, [values[-1] + (values[-1] - midpoints[-1])])
    )


def output_paths(output_base: Path) -> dict[str, Path]:
    if output_base.suffix:
        raise ValueError("--output must be a base path without a file extension.")
    return {
        "pdf": output_base.with_suffix(".pdf"),
        "svg": output_base.with_suffix(".svg"),
        "png": output_base.with_suffix(".png"),
        "json": output_base.with_suffix(".json"),
    }


def build_figure(
    coordinates: pd.DataFrame,
    descriptor_panels: list[dict[str, str]],
    lambda_limits: tuple[float, float],
) -> plt.Figure:
    sequential_cmap, diverging_cmap = choose_colormaps()
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(7.4, 6.0), facecolor="white")
    outer = figure.add_gridspec(2, 1, height_ratios=(1.0, 1.03), hspace=0.48)
    top = outer[0].subgridspec(1, 5, width_ratios=(1.38, 1.0, 0.055, 1.0, 0.055), wspace=0.62)
    bottom = outer[1].subgridspec(1, 4, width_ratios=(1.0, 1.0, 1.0, 0.07), wspace=0.26)
    axis_a = figure.add_subplot(top[0])
    axis_b = figure.add_subplot(top[1])
    lambda_colorbar_axis = figure.add_subplot(top[2])
    axis_c = figure.add_subplot(top[3])
    shift_colorbar_axis = figure.add_subplot(top[4])
    map_axes = [figure.add_subplot(bottom[index]) for index in range(3)]
    map_colorbar_axis = figure.add_subplot(bottom[3])

    draw_panel_a(axis_a, descriptor_panels)
    x_limits, y_limits = scatter_limits(coordinates)
    x_values = coordinates["spectral_coordinate_1"].to_numpy(dtype=float)
    y_values = coordinates["spectral_coordinate_2"].to_numpy(dtype=float)
    lambda_values = coordinates["lambda"].to_numpy(dtype=float)
    shift_values = coordinates["shift"].to_numpy(dtype=float)
    lambda_norm = PowerNorm(gamma=LAMBDA_COLOR_GAMMA, vmin=LAMBDA_COLOR_LIMITS[0], vmax=LAMBDA_COLOR_LIMITS[1])
    lambda_scatter = axis_b.scatter(
        x_values,
        y_values,
        c=np.clip(lambda_values, *LAMBDA_COLOR_LIMITS),
        cmap=sequential_cmap,
        norm=lambda_norm,
        s=4,
        alpha=0.65,
        linewidths=0,
        rasterized=True,
    )
    shift_min, shift_max = float(shift_values.min()), float(shift_values.max())
    if shift_min == shift_max:
        shift_min -= 0.5
        shift_max += 0.5
    shift_scatter = axis_c.scatter(
        x_values,
        y_values,
        c=shift_values,
        cmap=sequential_cmap,
        norm=Normalize(vmin=shift_min, vmax=shift_max),
        s=4,
        alpha=0.65,
        linewidths=0,
        rasterized=True,
    )
    for axis, letter in ((axis_b, "B"), (axis_c, "C")):
        axis.set(xlim=x_limits, ylim=y_limits, xlabel="Spectral coordinate 1", ylabel="Spectral coordinate 2")
        axis.grid(False)
        add_panel_label(axis, letter)
    lambda_colorbar = figure.colorbar(lambda_scatter, cax=lambda_colorbar_axis)
    lambda_colorbar.set_label(r"$\lambda$")
    lambda_colorbar.set_ticks(np.arange(0.0, 31.0, 5.0))
    shift_colorbar = figure.colorbar(shift_scatter, cax=shift_colorbar_axis)
    shift_colorbar.set_label("shift")

    grids, map_lambdas, map_shifts = coordinate_state_grids(coordinates, lambda_limits)
    value_limit = max(
        float(np.nanmax(np.abs(grid.to_numpy(dtype=float)))) for grid in grids
    )
    value_limit = max(value_limit, 1e-12)
    map_norm = TwoSlopeNorm(vmin=-value_limit, vcenter=0.0, vmax=value_limit)
    map_cmap = diverging_cmap.copy()
    map_cmap.set_bad(MISSING_CELL_COLOR)
    shift_edges = cell_edges(map_shifts)
    lambda_edges = cell_edges(map_lambdas)
    map_artist = None
    for index, (axis, grid, letter) in enumerate(zip(map_axes, grids, ("D", "E", "F")), start=1):
        map_artist = axis.pcolormesh(
            shift_edges,
            lambda_edges,
            np.ma.masked_invalid(grid.to_numpy(dtype=float)),
            cmap=map_cmap,
            norm=map_norm,
            shading="flat",
            rasterized=True,
        )
        axis.set(xlim=(shift_edges[0], shift_edges[-1]), ylim=(lambda_edges[0], lambda_edges[-1]), xlabel="shift")
        axis.set_title(f"Coordinate {index}", fontsize=8, pad=3)
        axis.grid(False)
        add_panel_label(axis, letter)
        if index == 1:
            axis.set_ylabel(r"$\lambda$")
        else:
            axis.tick_params(labelleft=False)
    colorbar = figure.colorbar(map_artist, cax=map_colorbar_axis)
    colorbar.set_label("Mean spectral coordinate")
    colorbar.set_ticks([-value_limit, 0.0, value_limit])
    colorbar.set_ticklabels([f"{-value_limit:.2g}", "0", f"{value_limit:.2g}"])
    return figure


def write_sidecar(
    path: Path,
    results_dir: Path,
    metadata: dict[str, Any],
    source: dict[str, str],
    selected_groups: list[str],
    selected_columns: list[str],
    group_source: str,
    descriptor_panels: list[dict[str, str]],
    orientations: list[dict[str, Any]],
    output_paths_map: dict[str, Path],
) -> None:
    contents = {
        "input_results_directory": str(results_dir.resolve()),
        "input_data_file": source["path"] if source["mode"] == "reconstructed_from_pickle" else None,
        "coordinate_source": source,
        "feature_set": FEATURE_SET,
        "selected_feature_groups": selected_groups,
        "selected_feature_columns": selected_columns,
        "feature_group_source": group_source,
        "descriptor_panels": descriptor_panels,
        "k": int(metadata["selected_k"]),
        "random_seed": int(metadata["random_seed"]),
        "spectral_coordinate_indices": list(range(1, COORDINATE_COUNT + 1)),
        "coordinate_orientation": orientations,
        "output_files": {name: str(output.resolve()) for name, output in output_paths_map.items() if name != "json"},
        "output_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(contents, handle, indent=2)


def main() -> None:
    args = parse_args()
    data_dir = resolve_data_directory(args.results_dir)
    results_dir = data_dir.parent if data_dir.name == "data" else data_dir
    metadata = load_run_metadata(data_dir)
    selected_k = validate_run_parameters(metadata, args)
    selected_groups, selected_columns, group_source = selected_feature_definition(metadata)
    panel_groups = selected_groups or infer_groups_from_columns(selected_columns)
    descriptor_panels = descriptor_panels_for_groups(panel_groups)
    coordinates, source = load_coordinates(
        data_dir,
        metadata,
        args,
        selected_k,
        selected_groups,
        selected_columns,
    )
    coordinates, orientations = orient_coordinates(coordinates)
    lambda_limits_raw = metadata["state_lambda_plot_limits"]
    if not isinstance(lambda_limits_raw, list) or len(lambda_limits_raw) != 2:
        raise ValueError("run_metadata.json state_lambda_plot_limits must contain two values.")
    lambda_limits = (float(lambda_limits_raw[0]), float(lambda_limits_raw[1]))
    paths = output_paths(args.output.expanduser().resolve())
    existing_outputs = [path for path in paths.values() if path.exists()]
    if existing_outputs and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing outputs; use --overwrite to replace: "
            + ", ".join(str(path) for path in existing_outputs)
        )
    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    figure = build_figure(coordinates, descriptor_panels, lambda_limits)
    try:
        figure.savefig(paths["pdf"], bbox_inches="tight")
        figure.savefig(paths["svg"], bbox_inches="tight")
        figure.savefig(paths["png"], dpi=400, bbox_inches="tight")
    finally:
        plt.close(figure)
    write_sidecar(
        paths["json"],
        results_dir,
        metadata,
        source,
        panel_groups,
        selected_columns,
        group_source,
        descriptor_panels,
        orientations,
        paths,
    )
    print("Created:")
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
