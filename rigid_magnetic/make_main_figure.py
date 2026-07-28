#!/usr/bin/env python3
"""Create the aligned publication figure for the reduced spectral embedding."""

from __future__ import annotations

import argparse
import copy
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
FIGURE_SIZE = (16.5, 7.7)
MATPLOTLIB_AXIS_LINEWIDTH = 0.8
PANEL_LABEL_FONT_SIZE = 13
TOP_HEADING_FONT_SIZE = 12
AXIS_LABEL_FONT_SIZE = 13
SCATTER_TICK_FONT_SIZE = 10
MAP_TICK_FONT_SIZE = 11
COLORBAR_TICK_FONT_SIZE = 9
TITLE_FONT_SIZE = 14
TITLE_PAD = 5
PANEL_LABEL_OFFSET = 0.008
# Visible artwork occupies this region of the 1600 x 1000 source SVG canvas.
PANEL_A_VISIBLE_VIEWBOX = (150.0, 5.0, 1220.0, 985.0)
SVG_NAMESPACE = "http://www.w3.org/2000/svg"
element_tree.register_namespace("", SVG_NAMESPACE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("figures/main_spectral_embedding"))
    parser.add_argument("--panel-a-svg", type=Path, default=None)
    parser.add_argument("--panel-a-svg-has-label", action="store_true")
    parser.add_argument("--topology-data-path", type=Path, default=None)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--bottom-right", choices=("topology", "psi3"), default="topology")
    parser.add_argument("--k", type=int, default=EXPECTED_K)
    parser.add_argument("--seed", type=int, default=EXPECTED_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_data_directory(results_dir: Path) -> tuple[Path, Path]:
    candidate = results_dir.expanduser().resolve()
    if (candidate / "data").is_dir():
        return candidate, candidate / "data"
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
        "input_data_path", "selected_k", "random_seed", "n_configurations",
        "state_lambda_plot_limits", "feature_set_groups", "feature_groups",
        "feature_columns", "crystallinity_features",
    }
    missing = sorted(required.difference(metadata))
    if missing:
        raise ValueError(f"Run metadata is missing required fields: {missing}")
    return metadata


def validate_run(metadata: dict[str, Any], args: argparse.Namespace) -> tuple[list[str], list[str]]:
    if args.k != EXPECTED_K or args.seed != EXPECTED_SEED:
        raise ValueError(f"This figure requires k={EXPECTED_K}, seed={EXPECTED_SEED}.")
    if int(metadata["selected_k"]) != args.k or int(metadata["random_seed"]) != args.seed:
        raise ValueError("Run k or seed does not match the requested publication configuration.")
    groups = metadata["feature_set_groups"].get(FEATURE_SET)
    columns = metadata["feature_columns"].get(FEATURE_SET)
    feature_groups = metadata["feature_groups"]
    if not isinstance(groups, list) or not groups or not isinstance(columns, list) or not columns:
        raise ValueError(f"Metadata does not define {FEATURE_SET} concretely.")
    if [column for group in groups for column in feature_groups[group]] != columns:
        raise ValueError("Metadata feature groups and reduced columns disagree.")
    crystallinity = metadata["crystallinity_features"]
    if crystallinity.get("mode") != "none" or crystallinity.get("active_columns"):
        raise ValueError("The final figure requires the crystallinity-free run.")
    return list(groups), list(columns)


def coordinate_path(data_dir: Path, selected_k: int) -> Path:
    return data_dir / f"reduced_embedding_coordinates_k{selected_k}.csv"


def validate_coordinate_table(table: pd.DataFrame, metadata: dict[str, Any], data_dir: Path) -> pd.DataFrame:
    required = ["file_id", "lambda", "shift", "k", "feature_set", *[f"spectral_coordinate_{i}" for i in range(1, 4)]]
    missing = [column for column in required if column not in table]
    if missing:
        raise ValueError(f"Coordinate CSV is missing columns: {missing}")
    if len(table) != int(metadata["n_configurations"]):
        raise ValueError("Coordinate CSV row count does not match run metadata.")
    if not table["k"].eq(int(metadata["selected_k"])).all() or not table["feature_set"].eq(FEATURE_SET).all():
        raise ValueError("Coordinate CSV k or feature set does not match this run.")
    numeric = [column for column in required if column not in {"file_id", "feature_set"}]
    validated = table.copy()
    validated[numeric] = validated[numeric].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(validated[numeric].to_numpy(dtype=float)).all() or validated["file_id"].duplicated().any():
        raise ValueError("Coordinate CSV contains invalid metadata or coordinate values.")
    count_path = data_dir / "state_point_counts.csv"
    if count_path.is_file():
        expected = pd.read_csv(count_path)
        expected = expected.loc[expected["k"].eq(int(metadata["selected_k"]))]
        actual = validated.groupby(["lambda", "shift"], sort=True).size().rename("count").reset_index()
        joined = expected[["lambda", "shift", "count"]].merge(actual, on=["lambda", "shift"], how="outer", suffixes=("_expected", "_actual"))
        if joined.isna().any().any() or not joined["count_expected"].eq(joined["count_actual"]).all():
            raise ValueError("Coordinate CSV state-point counts do not match the selected run.")
    return validated


def reconstruct_coordinates(metadata: dict[str, Any], groups: list[str], columns: list[str], data_path: Path) -> pd.DataFrame:
    if not data_path.is_file():
        raise FileNotFoundError(f"Recorded input pickle does not exist: {data_path}")
    df, feature_groups, _ = analysis.load_and_validate_data(data_path, "none")
    matrices, rebuilt_columns = analysis.standardized_feature_matrices(df, feature_groups, {FEATURE_SET: groups})
    if rebuilt_columns[FEATURE_SET] != columns:
        raise ValueError("Current preprocessing does not reproduce recorded reduced columns.")
    selected_k = int(metadata["selected_k"])
    _, embeddings, _ = analysis.compute_detailed_embeddings(matrices, [selected_k], selected_k)
    return analysis.embedding_coordinate_table(df[analysis.META_COLUMNS].reset_index(drop=True), embeddings[(FEATURE_SET, selected_k)], FEATURE_SET, selected_k)


def load_coordinates(data_dir: Path, metadata: dict[str, Any], groups: list[str], columns: list[str], args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, str]]:
    exported = coordinate_path(data_dir, int(metadata["selected_k"]))
    if exported.is_file():
        table = pd.read_csv(exported)
        source = {"mode": "exported_csv", "path": str(exported.resolve())}
    else:
        path = (args.data_path or Path(metadata["input_data_path"])).expanduser().resolve()
        table = reconstruct_coordinates(metadata, groups, columns, path)
        source = {"mode": "reconstructed_from_pickle", "path": str(path)}
    return validate_coordinate_table(table, metadata, data_dir), source


def correlation(values: np.ndarray, metadata_values: np.ndarray) -> float | None:
    if np.std(values) == 0.0 or np.std(metadata_values) == 0.0:
        return None
    value = float(np.corrcoef(values, metadata_values)[0, 1])
    return value if np.isfinite(value) else None


def orient_coordinates(coordinates: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    oriented = coordinates.copy()
    decisions = []
    for index in range(1, 4):
        column = f"spectral_coordinate_{index}"
        values = oriented[column].to_numpy(dtype=float)
        lambda_r = correlation(values, oriented["lambda"].to_numpy(dtype=float))
        shift_r = correlation(values, oriented["shift"].to_numpy(dtype=float))
        if lambda_r is not None and abs(lambda_r) > ORIENTATION_CORRELATION_EPSILON:
            sign, criterion, used = (1 if lambda_r > 0 else -1), "positive_pearson_correlation_with_lambda", lambda_r
        elif shift_r is not None and abs(shift_r) > ORIENTATION_CORRELATION_EPSILON:
            sign, criterion, used = (1 if shift_r > 0 else -1), "positive_pearson_correlation_with_shift", shift_r
        else:
            sign, criterion, used = (1 if values[np.argmax(np.abs(values))] >= 0 else -1), "largest_absolute_coordinate_entry_positive", None
        oriented[column] = sign * values
        decisions.append({"coordinate": index, "original_sign": 1, "applied_sign": sign, "criterion": criterion, "correlation_used": used, "raw_lambda_pearson_r": lambda_r, "raw_shift_pearson_r": shift_r, "correlation_epsilon": ORIENTATION_CORRELATION_EPSILON})
    return oriented, decisions


def cell_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5])
    middle = (values[:-1] + values[1:]) / 2.0
    return np.concatenate(([values[0] - (middle[0] - values[0])], middle, [values[-1] + (values[-1] - middle[-1])]))


def spectral_state_grids(coordinates: pd.DataFrame, limits: tuple[float, float], indices: tuple[int, ...]) -> tuple[dict[int, pd.DataFrame], np.ndarray, np.ndarray]:
    visible = coordinates.loc[coordinates["lambda"].between(*limits)]
    lambdas, shifts = np.sort(visible["lambda"].unique()), np.sort(visible["shift"].unique())
    grids = {}
    for index in indices:
        stats = analysis.state_stat_table(visible[["lambda", "shift"]], visible[f"spectral_coordinate_{index}"].to_numpy(dtype=float))
        grids[index] = analysis.state_grid(stats, "mean", lambdas, shifts)
    return grids, lambdas, shifts


def choose_font() -> tuple[str, bool]:
    available = {font.name.casefold() for font in font_manager.fontManager.ttflist}
    return ("Lato", True) if "lato" in available else ("DejaVu Sans", False)


def style_axis(axis: plt.Axes, tick_size: float, tick_pad: float = 3) -> None:
    axis.grid(False)
    axis.tick_params(width=MATPLOTLIB_AXIS_LINEWIDTH, length=4, pad=tick_pad, labelsize=tick_size)
    for spine in axis.spines.values():
        spine.set_linewidth(MATPLOTLIB_AXIS_LINEWIDTH)


def bbox_record(bbox: Any) -> dict[str, float]:
    return {key: float(getattr(bbox, key)) for key in ("x0", "y0", "x1", "y1", "width", "height")}


def add_panel_labels(figure: plt.Figure, boxes: dict[str, Any]) -> dict[str, dict[str, float]]:
    text = {"a": "(a)", "b": "(b)", "c": "(c)", "d": "(d)", "e": "(e)", "f": "(f)"}
    top_y, bottom_y = boxes["a"].y1 + PANEL_LABEL_OFFSET, boxes["d"].y1 + PANEL_LABEL_OFFSET
    positions = {}
    for key, label in text.items():
        y = top_y if key in {"a", "b", "c"} else bottom_y
        figure.text(boxes[key].x0, y, label, fontsize=PANEL_LABEL_FONT_SIZE, fontweight="bold", ha="left", va="bottom")
        positions[key] = {"x": float(boxes[key].x0), "y": float(y)}
    return positions


def add_top_headings(figure: plt.Figure, header_boxes: dict[str, Any]) -> dict[str, dict[str, float]]:
    headings = {
        "a": "Spectral embedding method",
        "b": r"$\psi_1$–$\psi_2$ coloured by $\lambda$",
        "c": r"$\psi_1$–$\psi_2$ coloured by shift",
    }
    positions = {}
    for key, heading in headings.items():
        box = header_boxes[key]
        x, y = (box.x0 + box.x1) / 2.0, (box.y0 + box.y1) / 2.0
        figure.text(x, y, heading, fontsize=TOP_HEADING_FONT_SIZE, fontweight="normal", ha="center", va="center")
        positions[key] = {"x": float(x), "y": float(y), "text": heading}
    return positions


def draw_panel_a_placeholder(axis: plt.Axes) -> None:
    axis.set_axis_off()
    axis.add_patch(Rectangle((0.04, 0.06), 0.92, 0.87, transform=axis.transAxes, fill=False, linewidth=MATPLOTLIB_AXIS_LINEWIDTH, edgecolor="#6b7280"))
    axis.text(0.5, 0.52, "Panel A SVG", ha="center", va="center", transform=axis.transAxes)


def draw_topology_map(axis: plt.Axes, topology: TopologyStateMap, shift_edges: np.ndarray, lambda_edges: np.ndarray) -> None:
    mesh = axis.pcolormesh(shift_edges, lambda_edges, np.zeros(topology.rgba.shape[:2]), shading="flat", edgecolors="none", rasterized=True)
    mesh.set_array(None)
    mesh.set_facecolor(topology.rgba.reshape(-1, 4))


def topology_legend(axis: plt.Axes) -> dict[str, float]:
    bounds = {"x": 0.025, "y": 0.64, "width": 0.45, "height": 0.31}
    inset = axis.inset_axes([bounds["x"], bounds["y"], bounds["width"], bounds["height"]])
    inset.set_axis_off()
    handles = [Patch(facecolor=TOPOLOGY_COLORS[item], edgecolor="white" if item == "chain" else "none", linewidth=0.4, label=TOPOLOGY_LABELS[item]) for item in TOPOLOGY_CATEGORIES]
    legend = inset.legend(handles=handles, title="Connectivity", loc="upper left", frameon=True, facecolor="#101820", edgecolor="none", framealpha=0.68, fontsize=8.5, title_fontsize=9, handlelength=0.9, handletextpad=0.35, labelspacing=0.28, borderpad=0.38, borderaxespad=0.0)
    legend.get_title().set_color("white")
    for item in legend.get_texts():
        item.set_color("white")
    return bounds


def layout_checks(figure: plt.Figure, containers: dict[str, Any], axes: dict[str, plt.Axes], labels: dict[str, dict[str, float]], headings: dict[str, dict[str, float]]) -> dict[str, float]:
    figure.canvas.draw()
    width, height = figure.bbox.width, figure.bbox.height
    px = lambda first, second, scale: abs(first - second) * scale
    title_anchor = lambda axis: axis.transAxes.transform((0.0, axis.title.get_position()[1]))[1]
    checks = {
        "top_container_y0_max_difference_px": max(px(containers[key].y0, containers["a"].y0, height) for key in ("b", "c")),
        "top_container_y1_max_difference_px": max(px(containers[key].y1, containers["a"].y1, height) for key in ("b", "c")),
        "bottom_container_y0_max_difference_px": max(px(containers[key].y0, containers["d"].y0, height) for key in ("e", "f")),
        "bottom_container_y1_max_difference_px": max(px(containers[key].y1, containers["d"].y1, height) for key in ("e", "f")),
        "scatter_width_difference_px": px(axes["b"].get_position().width, axes["c"].get_position().width, width),
        "scatter_height_difference_px": px(axes["b"].get_position().height, axes["c"].get_position().height, height),
        "map_width_max_difference_px": max(px(axes[key].get_position().width, axes["d"].get_position().width, width) for key in ("e", "f")),
        "map_height_max_difference_px": max(px(axes[key].get_position().height, axes["d"].get_position().height, height) for key in ("e", "f")),
        "bottom_title_baseline_max_difference_px": max(abs(title_anchor(axes[key]) - title_anchor(axes["d"])) for key in ("e", "f")),
        "top_label_baseline_max_difference_px": max(px(labels[key]["y"], labels["a"]["y"], height) for key in ("b", "c")),
        "bottom_label_baseline_max_difference_px": max(px(labels[key]["y"], labels["d"]["y"], height) for key in ("e", "f")),
        "top_heading_baseline_max_difference_px": max(px(headings[key]["y"], headings["a"]["y"], height) for key in ("b", "c")),
    }
    if any(value > 1.0 for value in checks.values()):
        raise RuntimeError(f"Panel layout alignment exceeds one display pixel: {checks}")
    return checks


def build_figure(coordinates: pd.DataFrame, lambda_limits: tuple[float, float], topology_path: Path | None, bottom_right: str, external_panel_a: bool) -> tuple[plt.Figure, plt.Axes, dict[str, Any]]:
    font_name, lato_available = choose_font()
    plt.rcParams.update({"font.family": font_name, "font.size": AXIS_LABEL_FONT_SIZE, "axes.labelsize": AXIS_LABEL_FONT_SIZE, "axes.titlesize": TITLE_FONT_SIZE, "axes.linewidth": MATPLOTLIB_AXIS_LINEWIDTH, "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})
    figure = plt.figure(figsize=FIGURE_SIZE, facecolor="white")
    outer = figure.add_gridspec(2, 1, height_ratios=(1.16, 0.84), hspace=0.30)
    top = outer[0].subgridspec(1, 3, width_ratios=(1.05, 1.35, 1.35), wspace=0.19)
    bottom = outer[1].subgridspec(1, 3, width_ratios=(1, 1, 1), wspace=0.16)
    a_panel = top[0].subgridspec(2, 1, height_ratios=(0.11, 0.89), hspace=0.0)
    b_panel = top[1].subgridspec(2, 1, height_ratios=(0.11, 0.89), hspace=0.0)
    c_panel = top[2].subgridspec(2, 1, height_ratios=(0.11, 0.89), hspace=0.0)
    axis_a = figure.add_subplot(a_panel[1])
    b_sub = b_panel[1].subgridspec(1, 2, width_ratios=(1, 0.040), wspace=0.055)
    c_sub = c_panel[1].subgridspec(1, 2, width_ratios=(1, 0.040), wspace=0.055)
    axis_b, lambda_cax = figure.add_subplot(b_sub[0]), figure.add_subplot(b_sub[1])
    axis_c, shift_cax = figure.add_subplot(c_sub[0]), figure.add_subplot(c_sub[1])
    axis_d, axis_e, axis_f = figure.add_subplot(bottom[0]), figure.add_subplot(bottom[1]), figure.add_subplot(bottom[2])
    figure.subplots_adjust(left=0.052, right=0.965, top=0.94, bottom=0.105)
    containers = {"a": top[0].get_position(figure), "b": top[1].get_position(figure), "c": top[2].get_position(figure), "d": bottom[0].get_position(figure), "e": bottom[1].get_position(figure), "f": bottom[2].get_position(figure)}
    heading_boxes = {"a": a_panel[0].get_position(figure), "b": b_panel[0].get_position(figure), "c": c_panel[0].get_position(figure)}
    axes = {"a": axis_a, "b": axis_b, "c": axis_c, "d": axis_d, "e": axis_e, "f": axis_f}
    if external_panel_a:
        axis_a.set_axis_off()
    else:
        draw_panel_a_placeholder(axis_a)

    x, y = coordinates["spectral_coordinate_1"].to_numpy(float), coordinates["spectral_coordinate_2"].to_numpy(float)
    lambdas, shifts = coordinates["lambda"].to_numpy(float), coordinates["shift"].to_numpy(float)
    x_limits = (float(x.min() - max(np.ptp(x) * 0.04, 1e-12)), float(x.max() + max(np.ptp(x) * 0.04, 1e-12)))
    y_limits = (float(y.min() - max(np.ptp(y) * 0.04, 1e-12)), float(y.max() + max(np.ptp(y) * 0.04, 1e-12)))
    lambda_scatter = axis_b.scatter(x, y, c=np.clip(lambdas, analysis.LAMBDA_COLOR_VMIN, analysis.LAMBDA_COLOR_VMAX), cmap=analysis.cmr.lilac, norm=PowerNorm(gamma=analysis.LAMBDA_COLOR_GAMMA, vmin=analysis.LAMBDA_COLOR_VMIN, vmax=analysis.LAMBDA_COLOR_VMAX), s=analysis.EMBEDDING_MARKER_SIZE, alpha=analysis.EMBEDDING_MARKER_OPACITY, linewidths=0, rasterized=True)
    shift_scatter = axis_c.scatter(x, y, c=shifts, cmap=analysis.cmr.lilac, norm=Normalize(vmin=float(shifts.min()), vmax=float(shifts.max())), s=analysis.EMBEDDING_MARKER_SIZE, alpha=analysis.EMBEDDING_MARKER_OPACITY, linewidths=0, rasterized=True)
    for axis in (axis_b, axis_c):
        axis.set(xlim=x_limits, ylim=y_limits, xlabel=r"$\psi_1$", ylabel=r"$\psi_2$")
        axis.set_box_aspect(0.78)
        style_axis(axis, SCATTER_TICK_FONT_SIZE, 2)
    lambda_bar = figure.colorbar(lambda_scatter, cax=lambda_cax)
    lambda_bar.set_label(r"$\lambda$", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=3)
    lambda_bar.set_ticks(np.arange(0, 31, 5))
    lambda_bar.ax.yaxis.set_ticks_position("left")
    lambda_bar.ax.yaxis.set_label_position("left")
    lambda_bar.ax.tick_params(labelsize=COLORBAR_TICK_FONT_SIZE, pad=1, width=MATPLOTLIB_AXIS_LINEWIDTH)
    shift_bar = figure.colorbar(shift_scatter, cax=shift_cax)
    shift_bar.set_label("shift", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=3)
    shift_bar.ax.tick_params(labelsize=COLORBAR_TICK_FONT_SIZE, pad=1, width=MATPLOTLIB_AXIS_LINEWIDTH)

    indices = (1, 2, 3) if bottom_right == "psi3" else (1, 2)
    grids, state_lambdas, state_shifts = spectral_state_grids(coordinates, lambda_limits, indices)
    color_limit = analysis.symmetric_limits(*(grids[index] for index in indices))[2]
    spectral_cmap = analysis.cmr.pride.copy()
    spectral_cmap.set_bad(analysis.MISSING_CELL_COLOR)
    x_edges, y_edges = cell_edges(state_shifts), cell_edges(state_lambdas)
    map_axes = ((axis_d, 1), (axis_e, 2)) + (((axis_f, 3),) if bottom_right == "psi3" else ())
    image = None
    for axis, index in map_axes:
        image = axis.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(grids[index].to_numpy(float)), cmap=spectral_cmap, norm=TwoSlopeNorm(vmin=-color_limit, vcenter=0, vmax=color_limit), shading="flat", rasterized=True)
        axis.set(xlim=(x_edges[0], x_edges[-1]), ylim=(y_edges[0], y_edges[-1]), xlabel="shift", title=rf"$\langle\psi_{index}\rangle$")
        axis.set_title(axis.get_title(), fontsize=TITLE_FONT_SIZE, fontweight="normal", pad=TITLE_PAD)
        style_axis(axis, MAP_TICK_FONT_SIZE)
    axis_d.set_ylabel(r"$\lambda$")
    axis_e.tick_params(labelleft=False)

    details: dict[str, Any] = {"font_family": font_name, "lato_available": lato_available, "state_lambdas": state_lambdas.tolist(), "state_shifts": state_shifts.tolist(), "spectral_color_limit": color_limit, "bottom_right": bottom_right}
    if bottom_right == "topology":
        if topology_path is None:
            raise ValueError("Topology data is required for the default Panel f.")
        topology = build_topology_state_map(topology_path, state_lambdas, state_shifts)
        draw_topology_map(axis_f, topology, x_edges, y_edges)
        axis_f.set(xlim=(x_edges[0], x_edges[-1]), ylim=(y_edges[0], y_edges[-1]), xlabel="shift", title="Topology")
        axis_f.set_title(axis_f.get_title(), fontsize=TITLE_FONT_SIZE, fontweight="normal", pad=TITLE_PAD)
        axis_f.tick_params(labelleft=False)
        style_axis(axis_f, MAP_TICK_FONT_SIZE)
        legend_position = topology_legend(axis_f)
        details["topology"] = {"source": str(topology_path.resolve()), "categories": list(TOPOLOGY_CATEGORIES), "excluded_categories": ["tree"], "colors": TOPOLOGY_COLORS, "blend": "top-two fractions with deterministic MD5 pair midpoint and quadratic Bezier mix", "bend": BEND, "missing_state_color": MISSING_TOPOLOGY_COLOR, "legend_inset_axes": legend_position}
    else:
        axis_f.tick_params(labelleft=False)

    figure.canvas.draw()
    for scatter_axis, color_axis in ((axis_b, lambda_cax), (axis_c, shift_cax)):
        scatter_bbox, color_bbox = scatter_axis.get_position(), color_axis.get_position()
        color_axis.set_position([color_bbox.x0, scatter_bbox.y0, color_bbox.width, scatter_bbox.height])
    figure.canvas.draw()
    e_box, f_box = axis_e.get_position(), axis_f.get_position()
    gap = f_box.x0 - e_box.x1
    cbar_width = min(0.011, gap * 0.24)
    spectral_cax = figure.add_axes([e_box.x1 + gap * 0.10, e_box.y0, cbar_width, e_box.height])
    spectral_bar = figure.colorbar(image, cax=spectral_cax)
    spectral_bar.set_label(r"$\langle\psi_i\rangle$", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=3)
    ticks, _ = analysis.symmetric_ticks(color_limit)
    spectral_bar.set_ticks(ticks)
    spectral_bar.set_ticklabels([f"{-round(color_limit, 3):.3f}", "0", f"{round(color_limit, 3):.3f}"])
    spectral_bar.ax.yaxis.set_ticks_position("right")
    spectral_bar.ax.yaxis.set_label_position("right")
    spectral_bar.ax.tick_params(labelsize=COLORBAR_TICK_FONT_SIZE, pad=1, width=MATPLOTLIB_AXIS_LINEWIDTH)
    labels = add_panel_labels(figure, containers)
    headings = add_top_headings(figure, heading_boxes)
    checks = layout_checks(figure, containers, axes, labels, headings)
    details.update({"panel_containers": {name: bbox_record(box) for name, box in containers.items()}, "principal_axes": {name: bbox_record(axis.get_position()) for name, axis in axes.items()}, "panel_label_positions": labels, "top_heading_positions": headings, "spectral_colorbar_axes": bbox_record(spectral_cax.get_position()), "layout_checks_px": checks, "grid_spacing": {"top_row_wspace": 0.19, "bottom_row_wspace": 0.16, "outer_hspace": 0.30, "scatter_colorbar_wspace": 0.055}, "spectral_colorbar": {"actual_limit": color_limit, "displayed_tick_labels": [f"{-round(color_limit, 3):.3f}", "0", f"{round(color_limit, 3):.3f}"]}, "font_sizes": {"panel_labels": PANEL_LABEL_FONT_SIZE, "top_headings": TOP_HEADING_FONT_SIZE, "axis_labels": AXIS_LABEL_FONT_SIZE, "scatter_ticks": SCATTER_TICK_FONT_SIZE, "map_ticks": MAP_TICK_FONT_SIZE, "colorbar_ticks": COLORBAR_TICK_FONT_SIZE, "titles": TITLE_FONT_SIZE}})
    return figure, axis_a, details


def output_paths(base: Path) -> dict[str, Path]:
    if base.suffix:
        raise ValueError("--output must be a base path without an extension.")
    return {"pdf": base.with_suffix(".pdf"), "svg": base.with_suffix(".svg"), "png": base.with_suffix(".png"), "json": base.with_suffix(".json")}


def svg_canvas_size(root: element_tree.Element) -> tuple[float, float]:
    values = [float(value) for value in root.attrib["viewBox"].replace(",", " ").split()]
    return values[2], values[3]


def compose_panel_a_svg(base_svg: Path, panel_svg: Path, panel_box: Any, output_svg: Path) -> dict[str, Any]:
    if not panel_svg.is_file():
        raise FileNotFoundError(f"Panel-a SVG does not exist: {panel_svg}")
    base_tree = element_tree.parse(base_svg)
    base_root = base_tree.getroot()
    panel_root = copy.deepcopy(element_tree.parse(panel_svg).getroot())
    panel_root.attrib["viewBox"] = " ".join(f"{value:g}" for value in PANEL_A_VISIBLE_VIEWBOX)
    width, height = svg_canvas_size(base_root)
    inset_x, inset_y = panel_box.x0 + 0.005 * panel_box.width, panel_box.y0 + 0.01 * panel_box.height
    inset_width, inset_height = 0.99 * panel_box.width, 0.98 * panel_box.height
    panel_root.attrib.update({"x": f"{inset_x * width:.6f}", "y": f"{(1 - inset_y - inset_height) * height:.6f}", "width": f"{inset_width * width:.6f}", "height": f"{inset_height * height:.6f}", "overflow": "hidden", "preserveAspectRatio": "xMidYMid meet"})
    base_root.append(panel_root)
    base_tree.write(output_svg, encoding="utf-8", xml_declaration=True)
    live_text = any(element.tag.endswith("text") for element in panel_root.iter())
    return {"source_viewbox": [0.0, 0.0, 1600.0, 1000.0], "visible_viewbox": list(PANEL_A_VISIBLE_VIEWBOX), "placement_fraction": {"x": inset_x, "y": inset_y, "width": inset_width, "height": inset_height}, "live_text_in_composed_svg": live_text}


def export_with_inkscape(svg_path: Path, paths: dict[str, Path]) -> None:
    inkscape = shutil.which("inkscape")
    if inkscape is None:
        raise RuntimeError("Inkscape is required to export the vector-composed Panel-a SVG.")
    for command in ([inkscape, str(svg_path), "--export-type=pdf", f"--export-filename={paths['pdf']}"], [inkscape, str(svg_path), "--export-type=png", "--export-dpi=400", f"--export-filename={paths['png']}"]):
        subprocess.run(command, check=True, capture_output=True, text=True)


def write_sidecar(path: Path, results_dir: Path, metadata: dict[str, Any], source: dict[str, str], groups: list[str], columns: list[str], orientations: list[dict[str, Any]], panel_a: dict[str, Any], details: dict[str, Any], paths: dict[str, Path]) -> None:
    contents = {"input_results_directory": str(results_dir.resolve()), "input_data_file": metadata["input_data_path"], "coordinate_source": source, "feature_set": FEATURE_SET, "selected_feature_groups": groups, "selected_feature_columns": columns, "crystallinity_mode": metadata["crystallinity_features"]["mode"], "crystallinity_features_absent": not metadata["crystallinity_features"]["active_columns"], "k": int(metadata["selected_k"]), "random_seed": int(metadata["random_seed"]), "spectral_coordinate_indices": list(DISPLAY_COORDINATES), "coordinate_definition": "First non-trivial coordinates of the detailed graph-Laplacian spectral embedding.", "coordinate_orientation": orientations, "panel_a": panel_a, "analysis_style": {"lambda_gamma": analysis.LAMBDA_COLOR_GAMMA, "lambda_limits": [analysis.LAMBDA_COLOR_VMIN, analysis.LAMBDA_COLOR_VMAX], "scatter_colormap": "cmr.lilac", "spectral_colormap": "cmr.pride", "missing_state_color": analysis.MISSING_CELL_COLOR, "marker_size": analysis.EMBEDDING_MARKER_SIZE, "marker_opacity": analysis.EMBEDDING_MARKER_OPACITY}, "figure": details, "output_files": {name: str(value.resolve()) for name, value in paths.items() if name != "json"}, "output_timestamp_utc": datetime.now(timezone.utc).isoformat()}
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
    coordinates, source = load_coordinates(data_dir, metadata, groups, columns, args)
    coordinates, orientations = orient_coordinates(coordinates)
    panel_svg = args.panel_a_svg.expanduser().resolve() if args.panel_a_svg else None
    topology_path = None if args.bottom_right == "psi3" else (args.topology_data_path or DEFAULT_TOPOLOGY_DATA_PATH).expanduser().resolve()
    if topology_path is not None and not topology_path.is_file():
        raise FileNotFoundError(f"Topology data file does not exist: {topology_path}")
    paths = output_paths(args.output.expanduser().resolve())
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Refusing to overwrite existing outputs; use --overwrite: " + ", ".join(map(str, existing)))
    paths["pdf"].parent.mkdir(parents=True, exist_ok=True)
    figure, axis_a, details = build_figure(coordinates, tuple(float(value) for value in metadata["state_lambda_plot_limits"]), topology_path, args.bottom_right, panel_svg is not None)
    panel_a_details: dict[str, Any] = {"source": str(panel_svg) if panel_svg else None, "is_external_svg": panel_svg is not None, "svg_contains_panel_label": args.panel_a_svg_has_label, "panel_label_added_by_main_figure": not args.panel_a_svg_has_label}
    try:
        if panel_svg is None:
            figure.savefig(paths["svg"])
            figure.savefig(paths["pdf"])
            figure.savefig(paths["png"], dpi=400)
        else:
            with tempfile.TemporaryDirectory(prefix="main_spectral_embedding_") as temporary:
                base_svg = Path(temporary) / "data_panels.svg"
                figure.savefig(base_svg, format="svg")
                panel_a_details.update(compose_panel_a_svg(base_svg, panel_svg, axis_a.get_position(), paths["svg"]))
            export_with_inkscape(paths["svg"], paths)
    finally:
        plt.close(figure)
    write_sidecar(paths["json"], results_dir, metadata, source, groups, columns, orientations, panel_a_details, details, paths)
    print("Created:")
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"error: {error}") from error
