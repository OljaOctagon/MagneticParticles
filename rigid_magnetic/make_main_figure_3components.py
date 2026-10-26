#!/usr/bin/env python3
"""Create the nine-panel spectral-embedding figure with three coordinates."""

from __future__ import annotations

import argparse
import copy
import tempfile
import xml.etree.ElementTree as element_tree
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import cmasher as cmr
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, PowerNorm, TwoSlopeNorm
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter

import ML_diffusion_map_testing as analysis
import make_main_figure as reference


FIGURE_SIZE = (16.5, 10.0)
OUTPUT_BASE = Path("figures/spectral_embedding_3components")
PANEL_LABEL_FONT_SIZE = 16
TITLE_FONT_SIZE = 14
AXIS_LABEL_FONT_SIZE = 14
SCATTER_TICK_FONT_SIZE = 12
MAP_TICK_FONT_SIZE = 12
COLORBAR_TICK_FONT_SIZE = 11
COLORBAR_LABEL_FONT_SIZE = 13
LEGEND_FONT_SIZE = 11
PANEL_LABEL_OFFSET = 0.004


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


def add_panel_headings(figure: plt.Figure, boxes: dict[str, Any]) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    headings = {
        "a": "Spectral embedding method",
        "b": r"$\psi_1$–$\psi_2$ colored by shift",
        "c": r"$\psi_1$–$\psi_2$ colored by $\lambda$",
        "d": r"$\psi_1$–$\psi_3$ colored by shift",
        "e": r"$\psi_1$–$\psi_3$ colored by $\lambda$",
        "f": r"$\psi_1$ state diagram",
        "g": r"$\psi_2$ state diagram",
        "h": r"$\psi_3$ state diagram",
        "i": "Connectivity state diagram",
    }
    label_positions, heading_positions = {}, {}
    for key, heading in headings.items():
        box = boxes[key]
        heading_y = box.y1 + PANEL_LABEL_OFFSET
        label_y = box.y1 + 0.025
        figure.text(
            box.x0,
            label_y,
            f"({key})",
            fontsize=PANEL_LABEL_FONT_SIZE,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        x = (box.x0 + box.x1) / 2.0
        if key == "i":
            x -= 0.015
        figure.text(
            x,
            heading_y,
            heading,
            fontsize=TITLE_FONT_SIZE,
            fontweight="normal",
            ha="center",
            va="bottom",
        )
        label_positions[key] = {"x": float(box.x0), "y": float(label_y)}
        heading_positions[key] = {"x": float(x), "y": float(heading_y), "text": heading}
    return label_positions, heading_positions


def draw_scatter(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    colors: np.ndarray,
    cmap: Any,
    norm: Normalize,
    xlabel: str,
    ylabel: str,
) -> Any:
    scatter = axis.scatter(
        x,
        y,
        c=colors,
        cmap=cmap,
        norm=norm,
        s=analysis.EMBEDDING_MARKER_SIZE,
        alpha=analysis.EMBEDDING_MARKER_OPACITY,
        linewidths=0,
        rasterized=True,
    )
    axis.set(xlabel=xlabel, ylabel=ylabel)
    reference.style_axis(axis, SCATTER_TICK_FONT_SIZE, 2)
    format_spectral_axis(axis, SCATTER_TICK_FONT_SIZE)
    return scatter


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
            "font.size": AXIS_LABEL_FONT_SIZE,
            "axes.labelsize": AXIS_LABEL_FONT_SIZE,
            "axes.titlesize": TITLE_FONT_SIZE,
            "axes.linewidth": reference.MATPLOTLIB_AXIS_LINEWIDTH,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=FIGURE_SIZE, facecolor="white")
    outer = figure.add_gridspec(2, 1, height_ratios=(2.15, 1.0), hspace=0.30)
    top = outer[0].subgridspec(2, 3, width_ratios=(0.90, 1.12, 1.12), wspace=0.31, hspace=0.34)
    bottom = outer[1].subgridspec(1, 4, wspace=0.30)
    axis_a = figure.add_subplot(top[:, 0])

    scatter_axes, scatter_caxes = [], []
    for panel in (top[0, 1], top[0, 2], top[1, 1], top[1, 2]):
        subgrid = panel.subgridspec(1, 2, width_ratios=(1, 0.045), wspace=0.13)
        scatter_axes.append(figure.add_subplot(subgrid[0]))
        scatter_caxes.append(figure.add_subplot(subgrid[1]))
    axis_b, axis_c, axis_d, axis_e = scatter_axes

    map_axes = []
    map_caxes = []
    for panel in bottom:
        subgrid = panel.subgridspec(1, 2, width_ratios=(1, 0.055), wspace=0.24)
        map_axes.append(figure.add_subplot(subgrid[0]))
        map_caxes.append(figure.add_subplot(subgrid[1]))
    axis_f, axis_g, axis_h, axis_i = map_axes
    map_caxes[-1].set_axis_off()
    figure.subplots_adjust(left=0.050, right=0.955, top=0.940, bottom=0.080)

    containers = {
        "a": top[:, 0].get_position(figure),
        "b": top[0, 1].get_position(figure),
        "c": top[0, 2].get_position(figure),
        "d": top[1, 1].get_position(figure),
        "e": top[1, 2].get_position(figure),
        "f": bottom[0].get_position(figure),
        "g": bottom[1].get_position(figure),
        "h": bottom[2].get_position(figure),
        "i": bottom[3].get_position(figure),
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
    shift_norm = Normalize(vmin=float(shifts.min()), vmax=float(shifts.max()))
    lambda_norm = PowerNorm(
        gamma=analysis.LAMBDA_COLOR_GAMMA,
        vmin=analysis.LAMBDA_COLOR_VMIN,
        vmax=analysis.LAMBDA_COLOR_VMAX,
    )
    clipped_lambdas = np.clip(lambdas, analysis.LAMBDA_COLOR_VMIN, analysis.LAMBDA_COLOR_VMAX)
    scatter_specs = (
        (axis_b, psi2, shifts, analysis.cmr.lilac, shift_norm, r"$\psi_2$", "shift"),
        (axis_c, psi2, clipped_lambdas, cmr.lavender, lambda_norm, r"$\psi_2$", r"$\lambda$"),
        (axis_d, psi3, shifts, analysis.cmr.lilac, shift_norm, r"$\psi_3$", "shift"),
        (axis_e, psi3, clipped_lambdas, cmr.lavender, lambda_norm, r"$\psi_3$", r"$\lambda$"),
    )
    for (axis, y, colors, cmap, norm, ylabel, colorbar_label), cax in zip(scatter_specs, scatter_caxes):
        scatter = draw_scatter(axis, psi1, y, colors, cmap, norm, r"$\psi_1$", ylabel)
        bar = figure.colorbar(scatter, cax=cax)
        bar.set_label(colorbar_label, fontsize=COLORBAR_LABEL_FONT_SIZE, labelpad=3)
        if colorbar_label == r"$\lambda$":
            bar.set_ticks(np.arange(0, 31, 10))
        bar.ax.tick_params(labelsize=COLORBAR_TICK_FONT_SIZE, pad=2, width=reference.MATPLOTLIB_AXIS_LINEWIDTH)

    psi12_limits = (
        (float(psi1.min() - max(np.ptp(psi1) * 0.04, 1e-12)), float(psi1.max() + max(np.ptp(psi1) * 0.04, 1e-12))),
        (float(psi2.min() - max(np.ptp(psi2) * 0.04, 1e-12)), float(psi2.max() + max(np.ptp(psi2) * 0.04, 1e-12))),
    )
    psi13_limits = (
        psi12_limits[0],
        (float(psi3.min() - max(np.ptp(psi3) * 0.04, 1e-12)), float(psi3.max() + max(np.ptp(psi3) * 0.04, 1e-12))),
    )
    for axis in (axis_b, axis_c):
        axis.set(xlim=psi12_limits[0], ylim=psi12_limits[1])
    for axis in (axis_d, axis_e):
        axis.set(xlim=psi13_limits[0], ylim=psi13_limits[1])

    grids, state_lambdas, state_shifts = reference.spectral_state_grids(coordinates, lambda_limits, (1, 2, 3))
    # Keep the existing psi1/psi2 scale unchanged and apply it to psi3.
    color_limit = analysis.symmetric_limits(grids[1], grids[2])[2]
    spectral_cmap = analysis.cmr.pride.copy()
    spectral_cmap.set_bad(analysis.MISSING_CELL_COLOR)
    x_edges, y_edges = reference.cell_edges(state_shifts), reference.cell_edges(state_lambdas)
    map_images = {}
    for axis, index in zip((axis_f, axis_g, axis_h), (1, 2, 3)):
        map_images[index] = axis.pcolormesh(
            x_edges,
            y_edges,
            np.ma.masked_invalid(grids[index].to_numpy(float)),
            cmap=spectral_cmap,
            norm=TwoSlopeNorm(vmin=-color_limit, vcenter=0, vmax=color_limit),
            shading="flat",
            rasterized=True,
        )
        axis.set(xlim=(x_edges[0], x_edges[-1]), ylim=(y_edges[0], y_edges[-1]), xlabel="shift")
        reference.style_axis(axis, MAP_TICK_FONT_SIZE)
    axis_f.set_ylabel(r"$\lambda$")
    axis_g.tick_params(labelleft=False)
    axis_h.tick_params(labelleft=False)

    topology = reference.build_topology_state_map(topology_path, state_lambdas, state_shifts)
    reference.draw_topology_map(axis_i, topology, x_edges, y_edges)
    axis_i.set(xlim=(x_edges[0], x_edges[-1]), ylim=(y_edges[0], y_edges[-1]), xlabel="shift")
    axis_i.tick_params(labelleft=False)
    reference.style_axis(axis_i, MAP_TICK_FONT_SIZE)
    legend_position = {"x": 0.025, "y": 0.57, "width": 0.68, "height": 0.40}
    legend_axis = axis_i.inset_axes(list(legend_position.values()))
    legend_axis.set_axis_off()
    handles = [
        Patch(
            facecolor=reference.TOPOLOGY_COLORS[item],
            edgecolor="white" if item == "chain" else "none",
            linewidth=0.5,
            label=reference.TOPOLOGY_LABELS[item],
        )
        for item in reference.TOPOLOGY_CATEGORIES
    ]
    legend = legend_axis.legend(
        handles=handles,
        title="Connectivity",
        loc="upper left",
        frameon=True,
        facecolor="#101820",
        edgecolor="none",
        framealpha=0.72,
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_FONT_SIZE + 0.5,
        handlelength=0.9,
        handletextpad=0.4,
        labelspacing=0.30,
        borderpad=0.42,
        borderaxespad=0.0,
    )
    legend.get_title().set_color("white")
    for text in legend.get_texts():
        text.set_color("white")

    figure.canvas.draw()
    for axis, cax in zip(scatter_axes, scatter_caxes):
        axis_box, color_box = axis.get_position(), cax.get_position()
        cax.set_position([color_box.x0, axis_box.y0, color_box.width, axis_box.height])
    state_colorbars = []
    displayed_limit = round(color_limit, 3)
    ticks = [-displayed_limit, 0.0, displayed_limit]
    for coordinate, cax, axis in zip((1, 2, 3), map_caxes[:3], (axis_f, axis_g, axis_h)):
        map_box, color_box = axis.get_position(), cax.get_position()
        cax.set_position([color_box.x0, map_box.y0, color_box.width, map_box.height])
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

    labels, headings = add_panel_headings(figure, containers)
    axes = dict(zip("abcdefghi", (axis_a, axis_b, axis_c, axis_d, axis_e, axis_f, axis_g, axis_h, axis_i)))
    details = {
        "font_family": font_name,
        "lato_available": lato_available,
        "state_lambdas": state_lambdas.tolist(),
        "state_shifts": state_shifts.tolist(),
        "spectral_color_limit": color_limit,
        "spectral_color_limit_source": "psi1_and_psi2_reference_scale",
        "panels": "nine_panel_three_component_embedding_and_state_diagrams",
        "panel_containers": {name: reference.bbox_record(box) for name, box in containers.items()},
        "principal_axes": {name: reference.bbox_record(axis.get_position()) for name, axis in axes.items()},
        "panel_label_positions": labels,
        "panel_heading_positions": headings,
        "scatter_colormaps": {"shift": "cmr.lilac", "lambda": "cmr.lavender"},
        "scatter_normalization": {
            "shift": [float(shifts.min()), float(shifts.max())],
            "lambda": [analysis.LAMBDA_COLOR_VMIN, analysis.LAMBDA_COLOR_VMAX, analysis.LAMBDA_COLOR_GAMMA],
        },
        "font_sizes": {
            "panel_labels": PANEL_LABEL_FONT_SIZE,
            "titles": TITLE_FONT_SIZE,
            "axis_labels": AXIS_LABEL_FONT_SIZE,
            "scatter_ticks": SCATTER_TICK_FONT_SIZE,
            "map_ticks": MAP_TICK_FONT_SIZE,
            "colorbar_labels": COLORBAR_LABEL_FONT_SIZE,
            "colorbar_ticks": COLORBAR_TICK_FONT_SIZE,
            "legend": LEGEND_FONT_SIZE,
        },
        "topology": {"source": str(topology_path.resolve()), "legend_inset_axes": legend_position},
    }
    return figure, axis_a, details


def enlarged_panel_a_source(panel_svg: Path, output_svg: Path) -> None:
    """Reflow the landscape workflow into a tall composition for Panel (a)."""
    source_root = element_tree.parse(panel_svg).getroot()
    children = list(source_root)
    if len(children) < 21:
        raise ValueError("Panel-a SVG does not have the expected workflow structure.")

    root = element_tree.Element(
        source_root.tag,
        {"width": "900", "height": "1650", "viewBox": "0 0 900 1650"},
    )
    root.append(copy.deepcopy(children[0]))
    for element in root.iter():
        if element.tag.endswith("style") and element.text:
            element.text = (
                element.text.replace("font-size:46px", "font-size:66px")
                .replace("font-size:30px", "font-size:46px")
                .replace("font-size:22px", "font-size:34px")
            )

    namespace = source_root.tag.split("}")[0].lstrip("{")
    qualified = lambda name: f"{{{namespace}}}{name}"
    element_tree.SubElement(root, qualified("rect"), {"width": "100%", "height": "100%", "fill": "white"})

    orientation = element_tree.SubElement(root, qualified("g"), {"transform": "matrix(1.697619 0 0 1.15 -212.5714 51)"})
    for element in children[2:14]:
        copied = copy.deepcopy(element)
        if copied.get("class") == "axis":
            copied.set("vector-effect", "non-scaling-stroke")
        if copied.tag.endswith("text") and copied.text == "orientation histogram":
            copied.set("transform", "translate(390 0) scale(0.6775 1) translate(-390 0)")
        orientation.append(copied)

    radial = element_tree.SubElement(root, qualified("g"), {"transform": "matrix(1.15 0 0 1.15 -735 430)"})
    for element in children[14:18]:
        copied = copy.deepcopy(element)
        if copied.get("class") == "axis":
            copied.set("vector-effect", "non-scaling-stroke")
        if copied.tag.endswith("text") and copied.text == "g(r)":
            copied.set("x", "1030")
        radial.append(copied)

    element_tree.SubElement(root, qualified("path"), {"class": "arrow", "d": "M 330 755 L 405 785"})
    element_tree.SubElement(root, qualified("path"), {"class": "arrow", "d": "M 570 755 L 495 785"})

    workflow = element_tree.SubElement(root, qualified("g"), {"transform": "matrix(1.2 0 0 1.2 -498 360)"})
    for element in children[20:]:
        copied = copy.deepcopy(element)
        if copied.tag.endswith("text") and copied.text in {"kNN graph", "(k = 32)"}:
            copied.set("x", "530")
        if copied.tag.endswith("text") and copied.text in {"non-trivial", "spectral coordinates"}:
            copied.set("x", "790")
            copied.set("text-anchor", "middle")
            copied.set("y", "1017" if copied.text == "non-trivial" else "1054")
        workflow.append(copied)
    element_tree.ElementTree(root).write(output_svg, encoding="utf-8", xml_declaration=True)


def compose_panel_a_svg(base_svg: Path, panel_svg: Path, panel_box: Any, output_svg: Path) -> dict[str, Any]:
    """Place the tall workflow SVG into the Matplotlib-generated SVG."""
    base_tree = element_tree.parse(base_svg)
    base_root = base_tree.getroot()
    panel_root = copy.deepcopy(element_tree.parse(panel_svg).getroot())
    source_viewbox = [float(value) for value in panel_root.attrib["viewBox"].split()]
    width, height = reference.svg_canvas_size(base_root)
    inset_x = panel_box.x0 + 0.005 * panel_box.width
    inset_y = panel_box.y0 + 0.005 * panel_box.height
    inset_width = 0.99 * panel_box.width
    inset_height = 0.99 * panel_box.height
    panel_root.attrib.update(
        {
            "x": f"{inset_x * width:.6f}",
            "y": f"{(1 - inset_y - inset_height) * height:.6f}",
            "width": f"{inset_width * width:.6f}",
            "height": f"{inset_height * height:.6f}",
            "overflow": "hidden",
            "preserveAspectRatio": "xMidYMid meet",
        }
    )
    base_root.append(panel_root)
    base_tree.write(output_svg, encoding="utf-8", xml_declaration=True)
    return {
        "source_viewbox": source_viewbox,
        "visible_viewbox": source_viewbox,
        "placement_fraction": {"x": inset_x, "y": inset_y, "width": inset_width, "height": inset_height},
        "live_text_in_composed_svg": True,
    }


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
                panel_a_details.update(compose_panel_a_svg(base_svg, enlarged_panel_a, axis_a.get_position(), paths["svg"]))
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
