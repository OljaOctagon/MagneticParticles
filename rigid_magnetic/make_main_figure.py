#!/usr/bin/env python3
"""Build the wide publication figure for the reduced spectral embedding."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as element_tree
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import Normalize, PowerNorm, TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle

import ML_diffusion_map_testing as analysis
from topology_state_map import (
    BEND,
    DEFAULT_TOPOLOGY_DATA_PATH,
    MISSING_TOPOLOGY_COLOR,
    TOPOLOGY_CATEGORIES,
    TOPOLOGY_COLORS,
    TOPOLOGY_LABELS,
    TopologyStateMap,
    build_topology_state_map,
)


EXPECTED_K = 32
EXPECTED_SEED = 42
FEATURE_SET = "reduced_no_global"
DISPLAY_COORDINATES = (1, 2)
ORIENTATION_CORRELATION_EPSILON = 0.05
FIGURE_SIZE = (16.5, 8.0)
MATPLOTLIB_AXIS_LINEWIDTH = 0.8
SVG_NAMESPACE = "http://www.w3.org/2000/svg"
element_tree.register_namespace("", SVG_NAMESPACE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the 2 x 3 publication figure for the reduced spectral embedding."
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("figures/main_spectral_embedding")
    )
    parser.add_argument(
        "--panel-a-svg",
        type=Path,
        default=None,
        help="External workflow SVG for Panel a. It remains vector in SVG/PDF output.",
    )
    parser.add_argument(
        "--panel-a-svg-has-label",
        action="store_true",
        help="Do not add the a.) label outside the external Panel-a SVG.",
    )
    parser.add_argument(
        "--topology-data-path",
        type=Path,
        default=None,
        help="Topology pickle for Panel f; defaults to the state_diagram.ipynb source.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Input pickle used only if the coordinate export is absent.",
    )
    parser.add_argument(
        "--bottom-right",
        choices=("topology", "psi3"),
        default="topology",
        help="Panel f content (default: topology).",
    )
    parser.add_argument("--k", type=int, default=EXPECTED_K)
    parser.add_argument("--seed", type=int, default=EXPECTED_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_data_directory(results_dir: Path) -> tuple[Path, Path]:
    candidate = results_dir.expanduser().resolve()
    data_dir = candidate / "data"
    if data_dir.is_dir():
        return candidate, data_dir
    if candidate.name == "data" and candidate.is_dir():
        return candidate.parent, candidate
    raise FileNotFoundError(f"Results directory must contain data/: {candidate}")


def load_metadata(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "run_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Required run metadata is missing: {path}")
    with path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    required = {
        "input_data_path",
        "selected_k",
        "random_seed",
        "n_configurations",
        "state_lambda_plot_limits",
        "feature_set_groups",
        "feature_groups",
        "feature_columns",
        "crystallinity_features",
    }
    missing = sorted(required.difference(metadata))
    if missing:
        raise ValueError(f"Run metadata is missing required fields: {missing}")
    return metadata


def validate_run(metadata: dict[str, Any], args: argparse.Namespace) -> tuple[list[str], list[str]]:
    if args.k != EXPECTED_K or args.seed != EXPECTED_SEED:
        raise ValueError(
            f"This final figure is configured for k={EXPECTED_K}, seed={EXPECTED_SEED}."
        )
    if int(metadata["selected_k"]) != args.k:
        raise ValueError(
            f"Run metadata has k={metadata['selected_k']}; expected k={args.k}."
        )
    if int(metadata["random_seed"]) != args.seed:
        raise ValueError(
            f"Run metadata has seed={metadata['random_seed']}; expected seed={args.seed}."
        )
    groups = metadata["feature_set_groups"].get(FEATURE_SET)
    columns = metadata["feature_columns"].get(FEATURE_SET)
    if not isinstance(groups, list) or not groups or not isinstance(columns, list) or not columns:
        raise ValueError(f"Metadata does not define concrete groups and columns for {FEATURE_SET}.")
    feature_groups = metadata["feature_groups"]
    unknown = [group for group in groups if group not in feature_groups]
    if unknown:
        raise ValueError(f"Metadata references unknown selected feature groups: {unknown}")
    flattened = [column for group in groups for column in feature_groups[group]]
    if flattened != columns:
        raise ValueError("Selected feature groups do not match the recorded reduced columns.")
    crystallinity = metadata["crystallinity_features"]
    active_crystallinity = crystallinity.get("active_columns", [])
    if crystallinity.get("mode") != "none" or active_crystallinity:
        raise ValueError(
            "The final main figure requires the crystallinity-free reduced run; "
            f"metadata reports mode={crystallinity.get('mode')!r}, active columns={active_crystallinity!r}."
        )
    return list(groups), list(columns)


def coordinate_path(data_dir: Path, selected_k: int) -> Path:
    return data_dir / f"reduced_embedding_coordinates_k{selected_k}.csv"


def validate_coordinate_table(
    table: pd.DataFrame, metadata: dict[str, Any], selected_k: int, data_dir: Path
) -> pd.DataFrame:
    required = [
        "file_id",
        "lambda",
        "shift",
        "k",
        "feature_set",
        *[f"spectral_coordinate_{index}" for index in range(1, 4)],
    ]
    missing = [column for column in required if column not in table]
    if missing:
        raise ValueError(f"Embedding-coordinate CSV is missing columns: {missing}")
    if len(table) != int(metadata["n_configurations"]):
        raise ValueError(
            f"Coordinate CSV has {len(table)} rows; metadata declares {metadata['n_configurations']}."
        )
    if not table["k"].eq(selected_k).all() or not table["feature_set"].eq(FEATURE_SET).all():
        raise ValueError("Coordinate CSV k or feature_set does not match the selected run.")
    numeric = [column for column in required if column not in {"file_id", "feature_set"}]
    validated = table.copy()
    validated[numeric] = validated[numeric].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(validated[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Coordinate CSV contains missing or non-finite numeric values.")
    if validated["file_id"].isna().any() or validated["file_id"].duplicated().any():
        raise ValueError("Coordinate CSV file_id values must be present and unique.")
    counts_path = data_dir / "state_point_counts.csv"
    if counts_path.is_file():
        expected = pd.read_csv(counts_path).query("k == @selected_k")
        actual = (
            validated.groupby(["lambda", "shift"], sort=True)
            .size()
            .rename("count")
            .reset_index()
        )
        comparison = expected[["lambda", "shift", "count"]].merge(
            actual, on=["lambda", "shift"], how="outer", suffixes=("_expected", "_actual")
        )
        if comparison.isna().any().any() or not comparison["count_expected"].eq(comparison["count_actual"]).all():
            raise ValueError("Coordinate CSV state-point counts do not match this run's metadata output.")
    return validated


def reconstruct_coordinates(
    metadata: dict[str, Any],
    groups: list[str],
    columns: list[str],
    data_path: Path,
    selected_k: int,
) -> pd.DataFrame:
    if not data_path.is_file():
        raise FileNotFoundError(f"Recorded input pickle does not exist: {data_path}")
    if analysis.SEED != int(metadata["random_seed"]):
        raise ValueError("Analysis helper seed differs from the selected run metadata.")
    df, feature_groups, _ = analysis.load_and_validate_data(data_path, "none")
    matrices, rebuilt_columns = analysis.standardized_feature_matrices(
        df, feature_groups, {FEATURE_SET: groups}
    )
    if rebuilt_columns[FEATURE_SET] != columns:
        raise ValueError("Current preprocessing does not reproduce the metadata feature columns.")
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
    groups: list[str],
    columns: list[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, str]]:
    selected_k = int(metadata["selected_k"])
    exported = coordinate_path(data_dir, selected_k)
    if exported.is_file():
        coordinates = pd.read_csv(exported)
        source = {"mode": "exported_csv", "path": str(exported.resolve())}
    else:
        data_path = args.data_path or Path(metadata["input_data_path"])
        data_path = data_path.expanduser().resolve()
        coordinates = reconstruct_coordinates(
            metadata, groups, columns, data_path, selected_k
        )
        source = {"mode": "reconstructed_from_pickle", "path": str(data_path)}
    return validate_coordinate_table(coordinates, metadata, selected_k, data_dir), source


def pearson_correlation(values: np.ndarray, metadata_values: np.ndarray) -> float | None:
    if np.std(values) == 0.0 or np.std(metadata_values) == 0.0:
        return None
    value = float(np.corrcoef(values, metadata_values)[0, 1])
    return value if np.isfinite(value) else None


def orient_coordinates(coordinates: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    oriented = coordinates.copy()
    lambda_values = oriented["lambda"].to_numpy(dtype=float)
    shift_values = oriented["shift"].to_numpy(dtype=float)
    decisions = []
    for index in range(1, 4):
        column = f"spectral_coordinate_{index}"
        values = oriented[column].to_numpy(dtype=float)
        lambda_r = pearson_correlation(values, lambda_values)
        shift_r = pearson_correlation(values, shift_values)
        if lambda_r is not None and abs(lambda_r) > ORIENTATION_CORRELATION_EPSILON:
            sign = 1 if lambda_r > 0 else -1
            criterion = "positive_pearson_correlation_with_lambda"
            selected_correlation = lambda_r
        elif shift_r is not None and abs(shift_r) > ORIENTATION_CORRELATION_EPSILON:
            sign = 1 if shift_r > 0 else -1
            criterion = "positive_pearson_correlation_with_shift"
            selected_correlation = shift_r
        else:
            maximum_index = int(np.argmax(np.abs(values)))
            sign = 1 if values[maximum_index] >= 0 else -1
            criterion = "largest_absolute_coordinate_entry_positive"
            selected_correlation = None
        oriented[column] = sign * values
        decisions.append(
            {
                "coordinate": index,
                "original_sign": 1,
                "applied_sign": sign,
                "criterion": criterion,
                "correlation_used": selected_correlation,
                "raw_lambda_pearson_r": lambda_r,
                "raw_shift_pearson_r": shift_r,
                "correlation_epsilon": ORIENTATION_CORRELATION_EPSILON,
            }
        )
    return oriented, decisions


def cell_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5])
    midpoints = (values[:-1] + values[1:]) / 2.0
    return np.concatenate(
        ([values[0] - (midpoints[0] - values[0])], midpoints, [values[-1] + (values[-1] - midpoints[-1])])
    )


def spectral_state_grids(
    coordinates: pd.DataFrame,
    lambda_limits: tuple[float, float],
    indices: tuple[int, ...],
) -> tuple[dict[int, pd.DataFrame], np.ndarray, np.ndarray]:
    visible = coordinates.loc[coordinates["lambda"].between(*lambda_limits)].copy()
    if visible.empty:
        raise ValueError("No configurations fall within the configured state-diagram lambda range.")
    lambda_values = np.sort(visible["lambda"].unique())
    shift_values = np.sort(visible["shift"].unique())
    metadata = visible[["lambda", "shift"]]
    grids = {}
    for index in indices:
        statistics = analysis.state_stat_table(
            metadata, visible[f"spectral_coordinate_{index}"].to_numpy(dtype=float)
        )
        grids[index] = analysis.state_grid(
            statistics, "mean", lambda_values, shift_values
        )
    return grids, lambda_values, shift_values


def choose_font() -> tuple[str, bool]:
    names = {font.name.casefold() for font in font_manager.fontManager.ttflist}
    if "lato" in names:
        return "Lato", True
    return "DejaVu Sans", False


def style_axis(axis: plt.Axes) -> None:
    axis.grid(False)
    axis.tick_params(width=MATPLOTLIB_AXIS_LINEWIDTH, length=4, pad=4)
    for spine in axis.spines.values():
        spine.set_linewidth(MATPLOTLIB_AXIS_LINEWIDTH)


def panel_label(axis: plt.Axes, label: str, *, inside: bool = False) -> None:
    axis.text(
        0.02 if inside else -0.12,
        1.045,
        label,
        transform=axis.transAxes,
        fontweight="bold",
        fontsize=analysis.TITLE_FONT,
        ha="left",
        va="bottom",
        clip_on=False,
    )


def draw_panel_a_placeholder(axis: plt.Axes) -> None:
    axis.set_axis_off()
    axis.add_patch(
        Rectangle(
            (0.04, 0.06), 0.92, 0.87, transform=axis.transAxes,
            fill=False, linewidth=MATPLOTLIB_AXIS_LINEWIDTH, edgecolor="#6b7280"
        )
    )
    axis.text(0.5, 0.52, "Panel a.) SVG", ha="center", va="center", transform=axis.transAxes)
    panel_label(axis, "a.)")


def topology_legend(axis: plt.Axes) -> None:
    axis.set_axis_off()
    handles = [
        Patch(facecolor=TOPOLOGY_COLORS[category], edgecolor="none", label=TOPOLOGY_LABELS[category])
        for category in TOPOLOGY_CATEGORIES
    ]
    axis.legend(
        handles=handles,
        title="Connectivity\ntypes",
        loc="center left",
        frameon=False,
        fontsize=8,
        title_fontsize=8.5,
        handlelength=1.0,
        handletextpad=0.4,
        labelspacing=0.45,
        borderaxespad=0.0,
    )


def draw_topology_map(
    axis: plt.Axes,
    topology: TopologyStateMap,
    shift_edges: np.ndarray,
    lambda_edges: np.ndarray,
) -> None:
    mesh = axis.pcolormesh(
        shift_edges,
        lambda_edges,
        np.zeros(topology.rgba.shape[:2]),
        shading="flat",
        edgecolors="none",
        rasterized=True,
    )
    # Disable scalar-mappable recoloring so each cell keeps its topology RGB mix.
    mesh.set_array(None)
    mesh.set_facecolor(topology.rgba.reshape(-1, 4))


def build_figure(
    coordinates: pd.DataFrame,
    lambda_limits: tuple[float, float],
    topology_path: Path | None,
    bottom_right: str,
    panel_a_external: bool,
) -> tuple[plt.Figure, plt.Axes, dict[str, Any]]:
    font_name, lato_available = choose_font()
    plt.rcParams.update(
        {
            "font.family": font_name,
            "font.size": analysis.TICK_FONT,
            "axes.labelsize": analysis.TICK_FONT,
            "axes.titlesize": analysis.TITLE_FONT,
            "xtick.labelsize": analysis.TICK_FONT - 1,
            "ytick.labelsize": analysis.TICK_FONT - 1,
            "axes.linewidth": MATPLOTLIB_AXIS_LINEWIDTH,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=FIGURE_SIZE, facecolor="white")
    outer = figure.add_gridspec(2, 1, height_ratios=(1.15, 0.9), hspace=0.34)
    top = outer[0].subgridspec(1, 3, width_ratios=(0.98, 1.06, 1.06), wspace=0.18)
    bottom = outer[1].subgridspec(
        1, 5, width_ratios=(1.0, 1.0, 0.055, 1.0, 0.25), wspace=0.2
    )
    axis_a = figure.add_subplot(top[0])
    b_layout = top[1].subgridspec(1, 2, width_ratios=(1.0, 0.045), wspace=0.06)
    c_layout = top[2].subgridspec(1, 2, width_ratios=(1.0, 0.045), wspace=0.06)
    axis_b = figure.add_subplot(b_layout[0])
    lambda_colorbar_axis = figure.add_subplot(b_layout[1])
    axis_c = figure.add_subplot(c_layout[0])
    shift_colorbar_axis = figure.add_subplot(c_layout[1])
    axis_d = figure.add_subplot(bottom[0])
    axis_e = figure.add_subplot(bottom[1])
    spectral_colorbar_axis = figure.add_subplot(bottom[2])
    axis_f = figure.add_subplot(bottom[3])
    topology_legend_axis = figure.add_subplot(bottom[4])
    figure.subplots_adjust(left=0.055, right=0.975, top=0.945, bottom=0.105)

    if panel_a_external:
        axis_a.set_axis_off()
    else:
        draw_panel_a_placeholder(axis_a)

    x_values = coordinates["spectral_coordinate_1"].to_numpy(dtype=float)
    y_values = coordinates["spectral_coordinate_2"].to_numpy(dtype=float)
    lambda_values = coordinates["lambda"].to_numpy(dtype=float)
    shift_values = coordinates["shift"].to_numpy(dtype=float)
    x_padding = max(0.04 * np.ptp(x_values), 1e-12)
    y_padding = max(0.04 * np.ptp(y_values), 1e-12)
    x_limits = (float(x_values.min() - x_padding), float(x_values.max() + x_padding))
    y_limits = (float(y_values.min() - y_padding), float(y_values.max() + y_padding))
    lambda_norm = PowerNorm(
        gamma=analysis.LAMBDA_COLOR_GAMMA,
        vmin=analysis.LAMBDA_COLOR_VMIN,
        vmax=analysis.LAMBDA_COLOR_VMAX,
    )
    lambda_scatter = axis_b.scatter(
        x_values,
        y_values,
        c=np.clip(lambda_values, analysis.LAMBDA_COLOR_VMIN, analysis.LAMBDA_COLOR_VMAX),
        cmap=analysis.cmr.lilac,
        norm=lambda_norm,
        s=analysis.EMBEDDING_MARKER_SIZE,
        alpha=analysis.EMBEDDING_MARKER_OPACITY,
        linewidths=0,
        rasterized=True,
    )
    shift_norm = Normalize(vmin=float(shift_values.min()), vmax=float(shift_values.max()))
    shift_scatter = axis_c.scatter(
        x_values,
        y_values,
        c=shift_values,
        cmap=analysis.cmr.lilac,
        norm=shift_norm,
        s=analysis.EMBEDDING_MARKER_SIZE,
        alpha=analysis.EMBEDDING_MARKER_OPACITY,
        linewidths=0,
        rasterized=True,
    )
    for axis, label in ((axis_b, "b.)"), (axis_c, "c.)")):
        axis.set(xlim=x_limits, ylim=y_limits, xlabel=r"$\psi_1$", ylabel=r"$\psi_2$")
        axis.set_box_aspect(0.78)
        style_axis(axis)
        panel_label(axis, label)
    lambda_colorbar = figure.colorbar(lambda_scatter, cax=lambda_colorbar_axis)
    lambda_colorbar.set_label(r"$\lambda$", labelpad=5)
    lambda_colorbar.set_ticks(np.arange(0.0, 31.0, 5.0))
    lambda_colorbar.ax.tick_params(pad=3, width=MATPLOTLIB_AXIS_LINEWIDTH)
    shift_colorbar = figure.colorbar(shift_scatter, cax=shift_colorbar_axis)
    shift_colorbar.set_label("shift", labelpad=5)
    shift_colorbar.ax.tick_params(pad=3, width=MATPLOTLIB_AXIS_LINEWIDTH)

    indices = (1, 2, 3) if bottom_right == "psi3" else (1, 2)
    grids, state_lambdas, state_shifts = spectral_state_grids(
        coordinates, lambda_limits, indices
    )
    color_limit = analysis.symmetric_limits(*(grids[index] for index in indices))[2]
    map_norm = TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit)
    spectral_cmap = analysis.cmr.pride.copy()
    spectral_cmap.set_bad(analysis.MISSING_CELL_COLOR)
    shift_edges = cell_edges(state_shifts)
    lambda_edges = cell_edges(state_lambdas)
    map_axes = ((axis_d, 1, "d.)"), (axis_e, 2, "e.)"))
    if bottom_right == "psi3":
        map_axes += ((axis_f, 3, "f.)"),)
    image = None
    for axis, index, label in map_axes:
        image = axis.pcolormesh(
            shift_edges,
            lambda_edges,
            np.ma.masked_invalid(grids[index].to_numpy(dtype=float)),
            cmap=spectral_cmap,
            norm=map_norm,
            shading="flat",
            rasterized=True,
        )
        axis.set(
            xlim=(shift_edges[0], shift_edges[-1]),
            ylim=(lambda_edges[0], lambda_edges[-1]),
            xlabel="shift",
            title=rf"$\langle\psi_{index}\rangle$",
        )
        style_axis(axis)
        panel_label(axis, label, inside=axis is axis_f)
    axis_d.set_ylabel(r"$\lambda$")
    axis_e.tick_params(labelleft=False)
    spectral_colorbar = figure.colorbar(image, cax=spectral_colorbar_axis)
    spectral_colorbar.set_label(r"$\langle\psi_i\rangle$", labelpad=5)
    ticks, _ = analysis.symmetric_ticks(color_limit)
    spectral_colorbar.set_ticks(ticks)
    spectral_colorbar.set_ticklabels([f"{value:.2g}" for value in ticks])
    spectral_colorbar.ax.yaxis.set_ticks_position("right")
    spectral_colorbar.ax.yaxis.set_label_position("right")
    spectral_colorbar.ax.tick_params(pad=3, width=MATPLOTLIB_AXIS_LINEWIDTH)

    figure_details: dict[str, Any] = {
        "font_family": font_name,
        "lato_available": lato_available,
        "state_lambdas": state_lambdas.tolist(),
        "state_shifts": state_shifts.tolist(),
        "spectral_color_limit": color_limit,
        "bottom_right": bottom_right,
    }
    if bottom_right == "topology":
        if topology_path is None:
            raise ValueError("A topology data path is required when Panel f is topology.")
        topology = build_topology_state_map(topology_path, state_lambdas, state_shifts)
        draw_topology_map(axis_f, topology, shift_edges, lambda_edges)
        axis_f.set(
            xlim=(shift_edges[0], shift_edges[-1]),
            ylim=(lambda_edges[0], lambda_edges[-1]),
            xlabel="shift",
            title="Topology",
        )
        axis_f.tick_params(labelleft=False)
        style_axis(axis_f)
        panel_label(axis_f, "f.)", inside=True)
        topology_legend(topology_legend_axis)
        figure_details["topology"] = {
            "source": str(topology_path.resolve()),
            "categories": list(TOPOLOGY_CATEGORIES),
            "excluded_categories": ["tree"],
            "colors": TOPOLOGY_COLORS,
            "blend": "top-two fractions with deterministic MD5 pair midpoint and quadratic Bezier mix",
            "bend": BEND,
            "missing_state_color": MISSING_TOPOLOGY_COLOR,
        }
    else:
        topology_legend_axis.set_axis_off()
        axis_f.tick_params(labelleft=False)
    figure.canvas.draw()
    for scatter_axis, colorbar_axis in (
        (axis_b, lambda_colorbar_axis), (axis_c, shift_colorbar_axis)
    ):
        scatter_position = scatter_axis.get_position()
        colorbar_position = colorbar_axis.get_position()
        colorbar_axis.set_position(
            [colorbar_position.x0, scatter_position.y0, colorbar_position.width, scatter_position.height]
        )
    return figure, axis_a, figure_details


def output_paths(output_base: Path) -> dict[str, Path]:
    if output_base.suffix:
        raise ValueError("--output must be a base path without a file extension.")
    return {
        "pdf": output_base.with_suffix(".pdf"),
        "svg": output_base.with_suffix(".svg"),
        "png": output_base.with_suffix(".png"),
        "json": output_base.with_suffix(".json"),
    }


def svg_canvas_size(svg_root: element_tree.Element) -> tuple[float, float]:
    view_box = svg_root.attrib.get("viewBox")
    if not view_box:
        raise ValueError("Matplotlib SVG is missing a viewBox required for Panel-a placement.")
    values = [float(value) for value in view_box.replace(",", " ").split()]
    if len(values) != 4:
        raise ValueError(f"Unexpected SVG viewBox: {view_box}")
    return values[2], values[3]


def compose_panel_a_svg(
    base_svg: Path,
    panel_svg: Path,
    panel_bbox: Any,
    font_name: str,
    add_label: bool,
    output_svg: Path,
) -> None:
    if not panel_svg.is_file():
        raise FileNotFoundError(f"Panel-a SVG does not exist: {panel_svg}")
    base_tree = element_tree.parse(base_svg)
    base_root = base_tree.getroot()
    panel_root = copy.deepcopy(element_tree.parse(panel_svg).getroot())
    canvas_width, canvas_height = svg_canvas_size(base_root)
    inset_x = panel_bbox.x0 + 0.025 * panel_bbox.width
    inset_y = panel_bbox.y0 + 0.02 * panel_bbox.height
    inset_width = 0.95 * panel_bbox.width
    inset_height = 0.95 * panel_bbox.height
    panel_root.attrib.update(
        {
            "x": f"{inset_x * canvas_width:.6f}",
            "y": f"{(1.0 - inset_y - inset_height) * canvas_height:.6f}",
            "width": f"{inset_width * canvas_width:.6f}",
            "height": f"{inset_height * canvas_height:.6f}",
            "overflow": "hidden",
            "preserveAspectRatio": "xMidYMid meet",
        }
    )
    base_root.append(panel_root)
    if add_label:
        text = element_tree.SubElement(
            base_root,
            f"{{{SVG_NAMESPACE}}}text",
            {
                "x": f"{panel_bbox.x0 * canvas_width:.6f}",
                "y": f"{(1.0 - panel_bbox.y1) * canvas_height + 22:.6f}",
                "font-family": font_name,
                "font-size": str(analysis.TITLE_FONT),
                "font-weight": "700",
                "fill": "#000000",
            },
        )
        text.text = "a.)"
    base_tree.write(output_svg, encoding="utf-8", xml_declaration=True)


def export_with_inkscape(svg_path: Path, paths: dict[str, Path]) -> None:
    inkscape = shutil.which("inkscape")
    if inkscape is None:
        raise RuntimeError(
            "Vector Panel-a composition requires Inkscape for PDF/PNG export, but it is unavailable."
        )
    for command in (
        [inkscape, str(svg_path), "--export-type=pdf", f"--export-filename={paths['pdf']}"],
        [inkscape, str(svg_path), "--export-type=png", "--export-dpi=400", f"--export-filename={paths['png']}"],
    ):
        subprocess.run(command, check=True, capture_output=True, text=True)


def write_sidecar(
    path: Path,
    results_dir: Path,
    metadata: dict[str, Any],
    source: dict[str, str],
    groups: list[str],
    columns: list[str],
    orientations: list[dict[str, Any]],
    panel_a_svg: Path | None,
    panel_a_svg_has_label: bool,
    details: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    contents = {
        "input_results_directory": str(results_dir.resolve()),
        "input_data_file": metadata["input_data_path"],
        "coordinate_source": source,
        "feature_set": FEATURE_SET,
        "selected_feature_groups": groups,
        "selected_feature_columns": columns,
        "crystallinity_mode": metadata["crystallinity_features"]["mode"],
        "crystallinity_features_absent": not metadata["crystallinity_features"]["active_columns"],
        "k": int(metadata["selected_k"]),
        "random_seed": int(metadata["random_seed"]),
        "spectral_coordinate_indices": list(DISPLAY_COORDINATES),
        "coordinate_definition": "First non-trivial coordinates of the detailed graph-Laplacian spectral embedding.",
        "coordinate_orientation": orientations,
        "panel_a": {
            "source": str(panel_a_svg.resolve()) if panel_a_svg else None,
            "is_external_svg": panel_a_svg is not None,
            "svg_contains_panel_label": panel_a_svg_has_label,
            "panel_label_added_by_main_figure": not panel_a_svg_has_label,
        },
        "analysis_style": {
            "lambda_gamma": analysis.LAMBDA_COLOR_GAMMA,
            "lambda_limits": [analysis.LAMBDA_COLOR_VMIN, analysis.LAMBDA_COLOR_VMAX],
            "scatter_colormap": "cmr.lilac",
            "spectral_colormap": "cmr.pride",
            "missing_state_color": analysis.MISSING_CELL_COLOR,
            "marker_size": analysis.EMBEDDING_MARKER_SIZE,
            "marker_opacity": analysis.EMBEDDING_MARKER_OPACITY,
            "tick_font_size": analysis.TICK_FONT,
            "title_font_size": analysis.TITLE_FONT,
            "matplotlib_axis_linewidth": MATPLOTLIB_AXIS_LINEWIDTH,
        },
        "figure": details,
        "output_files": {
            name: str(file_path.resolve()) for name, file_path in paths.items() if name != "json"
        },
        "output_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(contents, handle, indent=2)


def main() -> None:
    args = parse_args()
    results_dir, data_dir = resolve_data_directory(args.results_dir)
    metadata = load_metadata(data_dir)
    groups, columns = validate_run(metadata, args)
    input_pickle = Path(metadata["input_data_path"])
    if not input_pickle.is_file():
        raise FileNotFoundError(f"Recorded input pickle does not exist: {input_pickle}")
    coordinates, coordinate_source = load_coordinates(
        data_dir, metadata, groups, columns, args
    )
    coordinates, orientations = orient_coordinates(coordinates)
    lambda_limits = tuple(float(value) for value in metadata["state_lambda_plot_limits"])
    panel_a_svg = args.panel_a_svg.expanduser().resolve() if args.panel_a_svg else None
    topology_path = None
    if args.bottom_right == "topology":
        topology_path = (args.topology_data_path or DEFAULT_TOPOLOGY_DATA_PATH).expanduser().resolve()
        if not topology_path.is_file():
            raise FileNotFoundError(f"Topology data file does not exist: {topology_path}")
    paths = output_paths(args.output.expanduser().resolve())
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing figure outputs; use --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    paths["pdf"].parent.mkdir(parents=True, exist_ok=True)
    figure, axis_a, details = build_figure(
        coordinates,
        lambda_limits,
        topology_path,
        args.bottom_right,
        panel_a_svg is not None,
    )
    try:
        if panel_a_svg is None:
            figure.savefig(paths["svg"], bbox_inches="tight")
            figure.savefig(paths["pdf"], bbox_inches="tight")
            figure.savefig(paths["png"], dpi=400, bbox_inches="tight")
        else:
            with tempfile.TemporaryDirectory(prefix="main_spectral_embedding_") as temporary:
                base_svg = Path(temporary) / "data_panels.svg"
                figure.savefig(base_svg, format="svg")
                compose_panel_a_svg(
                    base_svg,
                    panel_a_svg,
                    axis_a.get_position(),
                    details["font_family"],
                    not args.panel_a_svg_has_label,
                    paths["svg"],
                )
            export_with_inkscape(paths["svg"], paths)
    finally:
        plt.close(figure)
    write_sidecar(
        paths["json"],
        results_dir,
        metadata,
        coordinate_source,
        groups,
        columns,
        orientations,
        panel_a_svg,
        args.panel_a_svg_has_label,
        details,
        paths,
    )
    print("Created:")
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"error: {error}") from error
