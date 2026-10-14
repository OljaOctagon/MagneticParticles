#!/usr/bin/env python3
"""Create the seven-panel spectral-embedding figure with three coordinates."""

from __future__ import annotations

import argparse
import tempfile
import xml.etree.ElementTree as element_tree
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import Normalize, PowerNorm, TwoSlopeNorm
from matplotlib.textpath import TextPath
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import ML_diffusion_map_testing as analysis
import make_main_figure as reference


FIGURE_SIZE = (19.5, 7.7)
OUTPUT_BASE = Path("figures/spectral_embedding_3components")
SCATTER_TICK_FONT_SIZE = 11
INSET_TICK_FONT_SIZE = 8
INSET_COLORBAR_TICK_FONT_SIZE = 7
INSET_COLORBAR_LABEL_FONT_SIZE = 9
MAP_TICK_FONT_SIZE = 12
COLORBAR_TICK_FONT_SIZE = 10
COLORBAR_LABEL_FONT_SIZE = 14


def scientific_formatter(exponent: int = -3) -> ScalarFormatter:
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((exponent, exponent))
    formatter.set_useOffset(False)
    return formatter


def format_spectral_axis(axis: plt.Axes, tick_size: float) -> None:
    axis.xaxis.set_major_formatter(scientific_formatter())
    axis.yaxis.set_major_formatter(scientific_formatter())
    axis.xaxis.get_offset_text().set_fontsize(tick_size)
    axis.yaxis.get_offset_text().set_fontsize(tick_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_BASE)
    parser.add_argument(
        "--panel-a-svg",
        type=Path,
        default=Path("panel_a_spectral_workflow_v5.svg"),
    )
    parser.add_argument("--panel-a-svg-has-label", action="store_true")
    parser.add_argument("--topology-data-path", type=Path, default=None)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--k", type=int, default=reference.EXPECTED_K)
    parser.add_argument("--seed", type=int, default=reference.EXPECTED_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def add_panel_labels(figure: plt.Figure, boxes: dict[str, Any]) -> dict[str, dict[str, float]]:
    labels = {key: f"({key})" for key in "abcdefg"}
    positions = {}
    top_y = boxes["a"].y1 + reference.PANEL_LABEL_OFFSET
    bottom_y = boxes["d"].y1 + reference.PANEL_LABEL_OFFSET
    for key, label in labels.items():
        y = top_y if key in {"a", "b", "c"} else bottom_y
        figure.text(
            boxes[key].x0,
            y,
            label,
            fontsize=reference.PANEL_LABEL_FONT_SIZE,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        positions[key] = {"x": float(boxes[key].x0), "y": float(y)}
    return positions


def add_top_headings(figure: plt.Figure, boxes: dict[str, Any]) -> dict[str, dict[str, float]]:
    headings = {
        "a": "Spectral embedding method",
        "b": r"$\psi_1$–$\psi_2$ coloured by shift",
        "c": r"$\psi_1$–$\psi_3$ coloured by shift",
    }
    positions = {}
    for key, heading in headings.items():
        box = boxes[key]
        x, y = (box.x0 + box.x1) / 2.0, box.y1 + reference.PANEL_LABEL_OFFSET
        figure.text(
            x,
            y,
            heading,
            fontsize=reference.TOP_HEADING_FONT_SIZE,
            fontweight="normal",
            ha="center",
            va="bottom",
        )
        positions[key] = {"x": float(x), "y": float(y), "text": heading}
    return positions


def scatter_with_inset(
    figure: plt.Figure,
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    lambdas: np.ndarray,
    shifts: np.ndarray,
    xlabel: str,
    ylabel: str,
    inset_bounds: tuple[float, float, float, float],
) -> tuple[Any, Any]:
    shift_scatter = axis.scatter(
        x,
        y,
        c=shifts,
        cmap=analysis.cmr.lilac,
        norm=Normalize(vmin=float(shifts.min()), vmax=float(shifts.max())),
        s=analysis.EMBEDDING_MARKER_SIZE,
        alpha=analysis.EMBEDDING_MARKER_OPACITY,
        linewidths=0,
        rasterized=True,
    )
    x_limits = (
        float(x.min() - max(np.ptp(x) * 0.04, 1e-12)),
        float(x.max() + max(np.ptp(x) * 0.04, 1e-12)),
    )
    y_limits = (
        float(y.min() - max(np.ptp(y) * 0.04, 1e-12)),
        float(y.max() + max(np.ptp(y) * 0.04, 1e-12)),
    )
    axis.set(xlim=x_limits, ylim=y_limits, xlabel=xlabel, ylabel=ylabel)
    axis.set_box_aspect(0.78)
    reference.style_axis(axis, SCATTER_TICK_FONT_SIZE, 2)
    format_spectral_axis(axis, SCATTER_TICK_FONT_SIZE)

    inset = axis.inset_axes(inset_bounds)
    lambda_scatter = inset.scatter(
        x,
        y,
        c=np.clip(lambdas, analysis.LAMBDA_COLOR_VMIN, analysis.LAMBDA_COLOR_VMAX),
        cmap=analysis.cmr.lilac,
        norm=PowerNorm(
            gamma=analysis.LAMBDA_COLOR_GAMMA,
            vmin=analysis.LAMBDA_COLOR_VMIN,
            vmax=analysis.LAMBDA_COLOR_VMAX,
        ),
        s=analysis.EMBEDDING_MARKER_SIZE,
        alpha=analysis.EMBEDDING_MARKER_OPACITY,
        linewidths=0,
        rasterized=True,
    )
    inset.set(xlim=x_limits, ylim=y_limits)
    inset.tick_params(
        labelsize=INSET_TICK_FONT_SIZE,
        length=2,
        width=reference.MATPLOTLIB_AXIS_LINEWIDTH,
        pad=1,
    )
    for spine in inset.spines.values():
        spine.set_linewidth(reference.MATPLOTLIB_AXIS_LINEWIDTH)
    format_spectral_axis(inset, INSET_TICK_FONT_SIZE)
    inset.set_facecolor("white")
    inset.set_zorder(5)
    # Keep the lambda scale outside the inset's plotting area so it cannot
    # cover the projected points.
    lambda_cax = axis.inset_axes(
        [
            inset_bounds[0] + inset_bounds[2] + 0.012,
            inset_bounds[1] + 0.05 * inset_bounds[3],
            0.018,
            0.90 * inset_bounds[3],
        ]
    )
    lambda_bar = figure.colorbar(lambda_scatter, cax=lambda_cax)
    lambda_bar.set_label(r"$\lambda$", fontsize=INSET_COLORBAR_LABEL_FONT_SIZE, labelpad=1)
    lambda_bar.set_ticks(np.arange(0, 31, 10))
    lambda_bar.ax.tick_params(
        labelsize=INSET_COLORBAR_TICK_FONT_SIZE,
        pad=1,
        width=reference.MATPLOTLIB_AXIS_LINEWIDTH,
        length=2,
    )
    return shift_scatter, lambda_bar


def build_figure(
    coordinates: pd.DataFrame,
    lambda_limits: tuple[float, float],
    topology_path: Path,
    external_panel_a: bool,
) -> tuple[plt.Figure, plt.Axes, dict[str, Any]]:
    font_name, lato_available = reference.choose_font()
    plt.rcParams.update(
        {
            "font.family": font_name,
            "font.size": reference.AXIS_LABEL_FONT_SIZE,
            "axes.labelsize": reference.AXIS_LABEL_FONT_SIZE,
            "axes.titlesize": reference.TITLE_FONT_SIZE,
            "axes.linewidth": reference.MATPLOTLIB_AXIS_LINEWIDTH,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=FIGURE_SIZE, facecolor="white")
    outer = figure.add_gridspec(2, 1, height_ratios=(1.16, 0.84), hspace=0.30)
    top = outer[0].subgridspec(1, 3, width_ratios=(1.25, 1.0, 1.0), wspace=0.19)
    bottom = outer[1].subgridspec(1, 4, width_ratios=(1, 1, 1, 1), wspace=0.16)
    top_panels = [top[index].subgridspec(2, 1, height_ratios=(0.11, 0.89), hspace=0.0) for index in range(3)]
    axis_a = figure.add_subplot(top_panels[0][1])
    axis_b = figure.add_subplot(top_panels[1][1])
    axis_c = figure.add_subplot(top_panels[2][1])
    map_axes = []
    map_caxes = []
    for panel in bottom:
        subgrid = panel.subgridspec(1, 2, width_ratios=(1, 0.048), wspace=0.18)
        map_axes.append(figure.add_subplot(subgrid[0]))
        map_caxes.append(figure.add_subplot(subgrid[1]))
    axis_d, axis_e, axis_f, axis_g = map_axes
    for cax in map_caxes:
        cax.set_axis_off()
    figure.subplots_adjust(left=0.045, right=0.965, top=0.94, bottom=0.105)

    containers = {
        "a": top[0].get_position(figure),
        "b": top[1].get_position(figure),
        "c": top[2].get_position(figure),
        "d": bottom[0].get_position(figure),
        "e": bottom[1].get_position(figure),
        "f": bottom[2].get_position(figure),
        "g": bottom[3].get_position(figure),
    }
    heading_boxes = {
        "a": top_panels[0][0].get_position(figure),
        "b": top_panels[1][0].get_position(figure),
        "c": top_panels[2][0].get_position(figure),
    }
    if external_panel_a:
        axis_a.set_axis_off()
    else:
        reference.draw_panel_a_placeholder(axis_a)

    lambdas = coordinates["lambda"].to_numpy(float)
    shifts = coordinates["shift"].to_numpy(float)
    psi1 = coordinates["spectral_coordinate_1"].to_numpy(float)
    psi2 = coordinates["spectral_coordinate_2"].to_numpy(float)
    psi3 = coordinates["spectral_coordinate_3"].to_numpy(float)
    shift_scatter_b, _ = scatter_with_inset(
        figure,
        axis_b,
        psi1,
        psi2,
        lambdas,
        shifts,
        r"$\psi_1$",
        r"$\psi_2$",
        (0.32, 0.58, 0.35, 0.35),
    )
    shift_scatter_c, _ = scatter_with_inset(
        figure,
        axis_c,
        psi1,
        psi3,
        lambdas,
        shifts,
        r"$\psi_1$",
        r"$\psi_3$",
        (0.56, 0.58, 0.34, 0.34),
    )

    shift_cax_b = inset_axes(axis_b, width="3.5%", height="100%", loc="lower left", bbox_to_anchor=(1.085, 0.0, 1.0, 1.0), bbox_transform=axis_b.transAxes, borderpad=0)
    shift_cax_c = inset_axes(axis_c, width="3.5%", height="100%", loc="lower left", bbox_to_anchor=(1.085, 0.0, 1.0, 1.0), bbox_transform=axis_c.transAxes, borderpad=0)
    for scatter, cax in ((shift_scatter_b, shift_cax_b), (shift_scatter_c, shift_cax_c)):
        bar = figure.colorbar(scatter, cax=cax)
        bar.set_label("shift", fontsize=reference.AXIS_LABEL_FONT_SIZE, labelpad=3)
        bar.ax.yaxis.set_ticks_position("left")
        bar.ax.yaxis.set_label_position("right")
        bar.ax.tick_params(labelsize=COLORBAR_TICK_FONT_SIZE, pad=1, width=reference.MATPLOTLIB_AXIS_LINEWIDTH)

    grids, state_lambdas, state_shifts = reference.spectral_state_grids(coordinates, lambda_limits, (1, 2, 3))
    # Keep the existing psi1/psi2 scale unchanged and apply it to psi3.
    color_limit = analysis.symmetric_limits(grids[1], grids[2])[2]
    spectral_cmap = analysis.cmr.pride.copy()
    spectral_cmap.set_bad(analysis.MISSING_CELL_COLOR)
    x_edges, y_edges = reference.cell_edges(state_shifts), reference.cell_edges(state_lambdas)
    map_images = {}
    for axis, index in zip((axis_d, axis_e, axis_f), (1, 2, 3)):
        map_images[index] = axis.pcolormesh(
            x_edges,
            y_edges,
            np.ma.masked_invalid(grids[index].to_numpy(float)),
            cmap=spectral_cmap,
            norm=TwoSlopeNorm(vmin=-color_limit, vcenter=0, vmax=color_limit),
            shading="flat",
            rasterized=True,
        )
        axis.set(xlim=(x_edges[0], x_edges[-1]), ylim=(y_edges[0], y_edges[-1]), xlabel="shift", title=rf"$\langle\psi_{index}\rangle$ state diagram")
        axis.set_title(axis.get_title(), fontsize=reference.TITLE_FONT_SIZE, fontweight="normal", pad=reference.TITLE_PAD)
        reference.style_axis(axis, MAP_TICK_FONT_SIZE)
    axis_d.set_ylabel(r"$\lambda$")
    axis_e.tick_params(labelleft=False)
    axis_f.tick_params(labelleft=False)

    topology = reference.build_topology_state_map(topology_path, state_lambdas, state_shifts)
    reference.draw_topology_map(axis_g, topology, x_edges, y_edges)
    axis_g.set(xlim=(x_edges[0], x_edges[-1]), ylim=(y_edges[0], y_edges[-1]), xlabel="shift", title="Connectivity state diagram")
    axis_g.set_title(axis_g.get_title(), fontsize=reference.TITLE_FONT_SIZE, fontweight="normal", pad=reference.TITLE_PAD)
    axis_g.tick_params(labelleft=False)
    reference.style_axis(axis_g, MAP_TICK_FONT_SIZE)
    legend_position = reference.topology_legend(axis_g)

    figure.canvas.draw()
    state_colorbars = []
    displayed_limit = round(color_limit, 3)
    ticks = [-displayed_limit, 0.0, displayed_limit]
    displayed_ticks = [f"{tick / 1e-3:.3g}" for tick in ticks]
    for coordinate, cax, axis in zip((1, 2, 3), map_caxes[:3], (axis_d, axis_e, axis_f)):
        map_box, color_box = axis.get_position(), cax.get_position()
        cax.set_position([color_box.x0 + reference.STATE_COLORBAR_RIGHT_SHIFT, map_box.y0, color_box.width, map_box.height])
        cax.set_axis_on()
        colorbar = figure.colorbar(map_images[coordinate], cax=cax)
        colorbar.set_label(rf"$\langle\psi_{coordinate}\rangle$", fontsize=COLORBAR_LABEL_FONT_SIZE, labelpad=3)
        colorbar.set_ticks(ticks)
        colorbar.formatter = scientific_formatter()
        colorbar.update_ticks()
        colorbar.ax.yaxis.set_ticks_position("left")
        colorbar.ax.yaxis.set_label_position("right")
        colorbar.ax.tick_params(labelsize=COLORBAR_TICK_FONT_SIZE, pad=1, width=reference.MATPLOTLIB_AXIS_LINEWIDTH)
        offset_text = colorbar.ax.yaxis.get_offset_text()
        offset_text.set_fontsize(COLORBAR_TICK_FONT_SIZE)
        offset_text.set_x(0.5)
        offset_text.set_horizontalalignment("center")
        state_colorbars.append(colorbar)

    labels = add_panel_labels(figure, containers)
    headings = add_top_headings(figure, heading_boxes)
    details = {
        "font_family": font_name,
        "lato_available": lato_available,
        "state_lambdas": state_lambdas.tolist(),
        "state_shifts": state_shifts.tolist(),
        "spectral_color_limit": color_limit,
        "spectral_color_limit_source": "psi1_and_psi2_reference_scale",
        "panels": "three_component_embedding_and_three_spectral_state_diagrams",
        "panel_containers": {name: reference.bbox_record(box) for name, box in containers.items()},
        "principal_axes": {name: reference.bbox_record(axis.get_position()) for name, axis in zip("abcdefg", (axis_a, axis_b, axis_c, axis_d, axis_e, axis_f, axis_g))},
        "panel_label_positions": labels,
        "top_heading_positions": headings,
        "topology": {"source": str(topology_path.resolve()), "legend_inset_axes": legend_position},
    }
    return figure, axis_a, details


def enlarged_panel_a_source(panel_svg: Path, output_svg: Path) -> None:
    """Prepare a larger-text copy for composition without changing the source SVG."""
    tree = element_tree.parse(panel_svg)
    root = tree.getroot()
    panel_font, _ = reference.choose_font()
    label_size, small_size, psi_size = 58, 44, 82
    font = font_manager.FontProperties(family=panel_font)
    standardize_width = TextPath((0, 0), "standardize", size=label_size, prop=font).get_extents().width
    standardize_detail_width = TextPath((0, 0), "zero mean, unit variance", size=small_size, prop=font).get_extents().width
    standardize_box_width = max(420.0, standardize_width, standardize_detail_width) + 84.0
    embedding_width = TextPath((0, 0), "spectral embedding", size=label_size, prop=font).get_extents().width
    embedding_box_width = max(450.0, embedding_width + 84.0)
    for element in root.iter():
        if element.tag.endswith("style") and element.text:
            element.text = (
                element.text.replace("font-size:30px", f"font-size:{label_size}px")
                .replace("font-size:22px", f"font-size:{small_size}px")
                .replace("font-size:46px", f"font-size:{psi_size}px")
            )
        if element.tag.endswith("rect") and element.get("class") == "box":
            if element.get("x") == "560" and element.get("y") == "360":
                element.attrib.update({"x": f"{770.0 - standardize_box_width / 2.0:g}", "y": "350", "width": f"{standardize_box_width:g}", "height": "130"})
            elif element.get("x") == "565" and element.get("y") == "750":
                element.attrib.update({"x": f"{790.0 - embedding_box_width / 2.0:g}", "y": "740", "width": f"{embedding_box_width:g}", "height": "110"})
        if element.tag.endswith("text") and element.text:
            if element.text == "standardize":
                element.set("y", "407")
            if element.text == "zero mean, unit variance":
                element.set("y", "459")
            elif element.text == "kNN graph":
                element.set("y", "580")
            elif element.text == "(k = 32)":
                element.set("y", "650")
            elif element.text == "spectral embedding":
                element.set("y", "811")
            elif element.text == "non-trivial":
                element.set("y", "902")
            elif element.text == "spectral coordinates":
                element.set("y", "972")
        if element.tag.endswith("path"):
            path = element.get("d")
            if path == "M 790 470 L 790 500":
                element.set("d", "M 790 495 L 790 515")
            elif path == "M 790 705 L 790 740":
                element.set("d", "M 790 705 L 790 730")
            elif path == "M 790 835 L 790 865":
                element.set("d", "M 790 855 L 790 875")
    tree.write(output_svg, encoding="utf-8", xml_declaration=True)


def main() -> None:
    args = parse_args()
    results_dir, data_dir = reference.resolve_data_directory(args.results_dir)
    metadata = reference.load_metadata(data_dir)
    groups, columns = reference.validate_run(metadata, args)
    coordinates, source = reference.load_coordinates(data_dir, metadata, groups, columns, args)
    coordinates, orientations = reference.orient_coordinates(coordinates)
    panel_svg = args.panel_a_svg.expanduser().resolve() if args.panel_a_svg else None
    topology_path = (args.topology_data_path or reference.DEFAULT_TOPOLOGY_DATA_PATH).expanduser().resolve()
    if not topology_path.is_file():
        raise FileNotFoundError(f"Topology data file does not exist: {topology_path}")
    paths = reference.output_paths(args.output.expanduser().resolve())
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Refusing to overwrite existing outputs; use --overwrite: " + ", ".join(map(str, existing)))
    paths["pdf"].parent.mkdir(parents=True, exist_ok=True)
    figure, axis_a, details = build_figure(
        coordinates,
        tuple(float(value) for value in metadata["state_lambda_plot_limits"]),
        topology_path,
        panel_svg is not None,
    )
    panel_a_details: dict[str, Any] = {"source": str(panel_svg) if panel_svg else None, "is_external_svg": panel_svg is not None, "svg_contains_panel_label": args.panel_a_svg_has_label, "panel_label_added_by_main_figure": not args.panel_a_svg_has_label}
    try:
        if panel_svg is None:
            figure.savefig(paths["svg"])
            figure.savefig(paths["pdf"])
            figure.savefig(paths["png"], dpi=400)
        else:
            with tempfile.TemporaryDirectory(prefix="spectral_embedding_3components_") as temporary:
                base_svg = Path(temporary) / "data_panels.svg"
                enlarged_panel_a = Path(temporary) / "panel_a_enlarged_text.svg"
                enlarged_panel_a_source(panel_svg, enlarged_panel_a)
                figure.savefig(base_svg, format="svg")
                panel_a_details.update(reference.compose_panel_a_svg(base_svg, enlarged_panel_a, axis_a.get_position(), paths["svg"]))
            reference.export_with_inkscape(paths["svg"], paths)
    finally:
        plt.close(figure)
    reference.write_sidecar(paths["json"], results_dir, metadata, source, groups, columns, orientations, panel_a_details, details, paths)
    print("Created:")
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError, OSError) as error:
        raise SystemExit(f"error: {error}") from error
