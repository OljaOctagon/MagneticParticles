#!/usr/bin/env python3
"""Command-line spectral-embedding analysis for magnetic-particle descriptors."""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.linalg import subspace_angles
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.sparse.linalg import eigsh
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr
from sklearn.manifold import SpectralEmbedding
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.preprocessing import StandardScaler


SCRIPT_VERSION = "1.0.0"
SEED = 42
N_COMPONENTS = 10
N_COMPONENTS_TO_ANALYZE = 5
N_MATCHED_COMPONENTS = 3
DEFAULT_GRAPH_K_VALUES = (5, 10, 20, 32, 50)
STATE_LAMBDA_MIN = 0.0
STATE_LAMBDA_MAX = 30.0
LAMBDA_COLOR_GAMMA = 0.5
LAMBDA_COLOR_VMIN = 0.0
LAMBDA_COLOR_VMAX = 30.0
TICK_FONT = 14
TITLE_FONT = 16
EMBEDDING_MARKER_SIZE = 4
EMBEDDING_MARKER_OPACITY = 0.65
MISSING_CELL_COLOR = "#bdbdbd"
META_COLUMNS = ["file_id", "lambda", "shift"]
DROP_COLUMNS = [
    "std_bonds_1_8",
    "std_bonds_1_5",
    "std_size",
    "std_radius_of_gyration",
    "std_second_neighbours",
]
GLOBAL_DESCRIPTOR_COLS = [
    "mean_bonds_1_8",
    "mean_bonds_1_5",
    "mean_second_neighbours",
    "mean_size",
    "largest",
    "mean_radius_of_gyration",
]
FEATURE_GROUP_SIZES = {"global": 6, "orientation": 24, "Rg": 29, "gofr": 25}
FEATURE_SET_GROUPS = {
    "all_features": ["global", "orientation", "Rg", "gofr"],
    "reduced_no_global": ["orientation", "gofr"],
}
EXPLORATORY_SET_GROUPS = {
    "all_features": ["global", "orientation", "Rg", "gofr"],
    "no_functions": ["global"],
    "no_orientation_Rg": ["global", "gofr"],
    "no_orientation_gofr": ["global", "Rg"],
    "no_Rg_gofr": ["global", "orientation"],
    "no_global_Rg": ["orientation", "gofr"],
    "no_orientation": ["global", "Rg", "gofr"],
    "no_Rg": ["global", "orientation", "gofr"],
}
FEATURE_SET_LABELS = {
    "all_features": "All features",
    "no_functions": "Global descriptors only",
    "no_orientation_Rg": "Global descriptors + g(r)",
    "no_orientation_gofr": "Global descriptors + Rg",
    "no_Rg_gofr": "Global descriptors + orientation",
    "no_global_Rg": "Orientation + g(r)",
    "no_orientation": "Global descriptors + Rg + g(r)",
    "no_Rg": "Global descriptors + orientation + g(r)",
    "reduced_no_global": "Reduced specification: orientation + g(r)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare full and reduced spectral embeddings of magnetic-particle descriptors."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=32,
        help="Nearest-neighbour count for detailed analyses (default: 32).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("results/MAG2P_order_parameters-2025-12-8-16:13:13.pickle"),
        help="Input order-parameter pickle file.",
    )
    args = parser.parse_args()
    if args.k <= 0:
        parser.error("--k must be a positive integer.")
    return args


def get_git_commit(script_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(script_dir), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def json_default(value: object):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def create_run_directories(
    data_path: Path, selected_k: int
) -> tuple[Path, dict[str, Path], str]:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    safe_stem = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in data_path.stem
    )
    results_dir = Path("results") / f"{safe_stem}_{timestamp}_k{selected_k}"
    try:
        results_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise RuntimeError(
            f"Refusing to overwrite an existing results directory: {results_dir.resolve()}"
        ) from error

    directories = {
        name: results_dir / name
        for name in ("diagnostics", "scatters", "state_diagrams", "differences", "data")
    }
    for directory in directories.values():
        directory.mkdir()
    return results_dir, directories, timestamp


def matplotlib_cmap_to_plotly(cmap, n_colors: int = 256) -> list[list[float | str]]:
    """Densely sample a Matplotlib-compatible colormap without remapping values."""
    return [
        [
            index / (n_colors - 1),
            f"rgb({round(red * 255)}, {round(green * 255)}, {round(blue * 255)})",
        ]
        for index, (red, green, blue, _) in enumerate(
            cmap(np.linspace(0.0, 1.0, n_colors))
        )
    ]


def nonlinear_lambda_colors(lambda_values: pd.Series | np.ndarray) -> np.ndarray:
    """Apply the shared power-law lambda color normalization, saturating above 30."""
    clipped = np.clip(
        np.asarray(lambda_values, dtype=float), LAMBDA_COLOR_VMIN, LAMBDA_COLOR_VMAX
    )
    return ((clipped - LAMBDA_COLOR_VMIN) / (LAMBDA_COLOR_VMAX - LAMBDA_COLOR_VMIN)) ** LAMBDA_COLOR_GAMMA


def configure_lambda_colorbar(fig: go.Figure) -> None:
    """Display original lambda units for the transformed Plotly color values."""
    tick_values = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    fig.update_coloraxes(
        cmin=LAMBDA_COLOR_VMIN,
        cmax=1.0,
        colorbar=dict(
            title="lambda",
            tickvals=nonlinear_lambda_colors(tick_values),
            ticktext=[f"{value:g}" for value in tick_values],
        ),
    )


def configure_shift_colorbar(fig: go.Figure, shift_min: float, shift_max: float) -> None:
    """Use the complete observed shift range for a linear colorbar."""
    fig.update_coloraxes(
        cmin=shift_min,
        cmax=shift_max,
        colorbar=dict(title="shift"),
    )


def save_plotly_figure(
    fig: go.Figure,
    output_base: Path,
    *,
    rows: int,
    cols: int,
    panel_width: int = 450,
    panel_height: int = 360,
    colorbar_width: int = 85,
) -> None:
    """Write responsive HTML and proportionally sized PDF for a Plotly figure."""
    margins = dict(l=70, r=45 + colorbar_width, t=80, b=60)
    width = cols * panel_width + margins["l"] + margins["r"]
    height = rows * panel_height + margins["t"] + margins["b"]
    fig.update_layout(width=width, height=height, margin=margins)

    html_path = output_base.with_suffix(".html")
    pdf_path = output_base.with_suffix(".pdf")
    logging.info("Writing HTML: %s", html_path)
    fig.write_html(
        html_path,
        include_plotlyjs="directory",
        config={"responsive": True, "displaylogo": False},
        default_width="100%",
        default_height=f"{height}px",
    )
    logging.info("Writing PDF: %s", pdf_path)
    try:
        fig.write_image(pdf_path, width=width, height=height, scale=1)
    except (ImportError, ValueError, OSError) as error:
        raise RuntimeError(
            "Plotly PDF export requires Kaleido and its supported browser runtime. "
            "Install Kaleido, then run `plotly_get_chrome` if Plotly requests Chrome."
        ) from error


def load_and_validate_data(
    data_path: Path,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Input data file does not exist: {data_path}. Use --data-path to specify a valid pickle file."
        )
    logging.info("Loading data from %s", data_path)
    raw_df = pd.read_pickle(data_path)
    missing_metadata = [
        column for column in META_COLUMNS if column not in raw_df.columns
    ]
    missing_dropped = [
        column for column in DROP_COLUMNS if column not in raw_df.columns
    ]
    if missing_metadata:
        raise ValueError(
            f"Input data is missing required metadata columns: {missing_metadata}"
        )
    if missing_dropped:
        raise ValueError(
            "Input data is missing expected descriptor columns required by the current feature schema: "
            f"{missing_dropped}"
        )

    with pd.option_context("future.no_silent_downcasting", True):
        df = raw_df.fillna(0)
    df = df.infer_objects(copy=False).drop(columns=DROP_COLUMNS)
    df = df.loc[:, (df != 0).any(axis=0)]
    if list(df.columns[:3]) != META_COLUMNS:
        raise ValueError(
            "The current feature schema requires the first three processed columns to be "
            f"{META_COLUMNS}, got {list(df.columns[:3])}."
        )
    if list(df.columns[3:9]) != GLOBAL_DESCRIPTOR_COLS:
        raise ValueError(
            "The processed global descriptor block does not match the expected six columns: "
            f"{GLOBAL_DESCRIPTOR_COLS}."
        )

    expected_feature_count = sum(FEATURE_GROUP_SIZES.values())
    actual_feature_count = len(df.columns) - len(META_COLUMNS)
    if actual_feature_count != expected_feature_count:
        raise ValueError(
            "Expected 84 processed feature columns split into global=6, orientation=24, "
            f"Rg=29, and gofr=25; found {actual_feature_count}."
        )

    feature_groups = {
        "global": list(df.columns[3:9]),
        "orientation": list(df.columns[9:33]),
        "Rg": list(df.columns[33:62]),
        "gofr": list(df.columns[62:]),
    }
    invalid_groups = {
        name: len(columns)
        for name, columns in feature_groups.items()
        if len(columns) != FEATURE_GROUP_SIZES[name]
    }
    if invalid_groups:
        raise ValueError(f"Feature-group validation failed: {invalid_groups}")

    feature_columns = [column for group in feature_groups.values() for column in group]
    non_numeric = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]
    if non_numeric:
        raise ValueError(
            f"Feature columns must be numeric; found non-numeric columns: {non_numeric}"
        )
    feature_values = df[feature_columns].to_numpy(dtype=float)
    if not np.isfinite(feature_values).all():
        raise ValueError("Feature data contains non-finite values after preprocessing.")
    if not pd.api.types.is_numeric_dtype(
        df["lambda"]
    ) or not pd.api.types.is_numeric_dtype(df["shift"]):
        raise ValueError(
            "Required metadata columns 'lambda' and 'shift' must be numeric."
        )

    logging.info("Loaded %d rows and %d processed columns", len(df), len(df.columns))
    logging.info(
        "Feature groups: %s",
        {name: len(columns) for name, columns in feature_groups.items()},
    )
    return df, feature_groups


def feature_columns_for_groups(
    feature_groups: dict[str, list[str]], groups: list[str]
) -> list[str]:
    unknown_groups = [group for group in groups if group not in feature_groups]
    if unknown_groups:
        raise ValueError(f"Unknown feature groups requested: {unknown_groups}")
    columns = [column for group in groups for column in feature_groups[group]]
    if len(columns) != len(set(columns)):
        raise ValueError(f"Feature groups contain duplicated columns: {groups}")
    return columns


def state_stat_table(metadata: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "lambda": metadata["lambda"].to_numpy(),
                "shift": metadata["shift"].to_numpy(),
                "value": values,
            }
        )
        .groupby(["lambda", "shift"], sort=True)["value"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )


def state_grid(
    table: pd.DataFrame,
    column: str,
    state_lambdas: np.ndarray,
    state_shifts: np.ndarray,
) -> pd.DataFrame:
    visible = table.loc[table["lambda"].between(STATE_LAMBDA_MIN, STATE_LAMBDA_MAX)]
    return visible.pivot(index="lambda", columns="shift", values=column).reindex(
        index=state_lambdas, columns=state_shifts
    )


def finite_abs_max(*grids: pd.DataFrame) -> float:
    values = np.concatenate([grid.to_numpy(dtype=float).ravel() for grid in grids])
    finite_values = values[np.isfinite(values)]
    return float(np.max(np.abs(finite_values))) if finite_values.size else 1.0


def finite_max(*grids: pd.DataFrame) -> float:
    values = np.concatenate([grid.to_numpy(dtype=float).ravel() for grid in grids])
    finite_values = values[np.isfinite(values)]
    return float(np.max(finite_values)) if finite_values.size else 1.0


def heatmap_trace(
    grid: pd.DataFrame,
    count_grid: pd.DataFrame,
    *,
    coloraxis: str,
    quantity_label: str,
) -> go.Heatmap:
    customdata = np.dstack((count_grid.to_numpy(dtype=float),))
    return go.Heatmap(
        x=grid.columns.to_numpy(),
        y=grid.index.to_numpy(),
        z=grid.to_numpy(dtype=float),
        customdata=customdata,
        coloraxis=coloraxis,
        hoverongaps=False,
        hovertemplate=(
            "shift=%{x}<br>lambda=%{y}<br>"
            f"{quantity_label}=%{{z:.5g}}<br>contributing samples=%{{customdata[0]:.0f}}<extra></extra>"
        ),
    )


def apply_coloraxis(
    fig: go.Figure,
    *,
    coloraxis: str,
    colorscale: list[list[float | str]],
    cmin: float,
    cmax: float,
    title: str,
) -> None:
    fig.update_layout(
        **{
            coloraxis: dict(
                colorscale=colorscale,
                cmin=cmin,
                cmax=cmax,
                colorbar=dict(
                    title=dict(text=title, side="right"),
                    thickness=18,
                    len=0.78,
                    x=1.02,
                    y=0.5,
                    tickfont=dict(size=TICK_FONT),
                ),
            )
        }
    )


def style_state_figure(fig: go.Figure, title: str) -> go.Figure:
    fig.update_xaxes(
        title_text="shift",
        tickfont=dict(size=TICK_FONT),
        title_font=dict(size=TITLE_FONT),
    )
    fig.update_yaxes(
        title_text="lambda",
        tickfont=dict(size=TICK_FONT),
        title_font=dict(size=TITLE_FONT),
    )
    fig.update_annotations(font=dict(size=TITLE_FONT))
    fig.update_layout(
        title=title,
        template="plotly_white",
        plot_bgcolor=MISSING_CELL_COLOR,
        paper_bgcolor="white",
        font=dict(size=TICK_FONT),
    )
    return fig


def build_initial_embedding_plots(
    df: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    metadata: pd.DataFrame,
    selected_k: int,
    state_lambdas: np.ndarray,
    state_shifts: np.ndarray,
    directories: dict[str, Path],
    guppy: list[list[float | str]],
    rainforest: list[list[float | str]],
    shift_limits: tuple[float, float],
) -> None:
    logging.info(
        "Computing exploratory spectral embeddings for %d feature specifications",
        len(EXPLORATORY_SET_GROUPS),
    )
    for set_name, groups in EXPLORATORY_SET_GROUPS.items():
        columns = feature_columns_for_groups(feature_groups, groups)
        embedding = SpectralEmbedding(
            # sklearn returns non-trivial spectral coordinates, excluding the constant mode.
            n_components=3,
            affinity="nearest_neighbors",
            n_neighbors=selected_k,
            random_state=SEED,
        ).fit_transform(
            StandardScaler().fit_transform(df[columns].to_numpy(dtype=float))
        )
        label = FEATURE_SET_LABELS[set_name]

        scatter_df = metadata.copy()
        scatter_df["embedding_1"] = embedding[:, 0]
        scatter_df["embedding_2"] = embedding[:, 1]
        scatter_df["selected_k"] = selected_k
        scatter_df["lambda_color"] = nonlinear_lambda_colors(scatter_df["lambda"])
        for color_variable, color_column, colorscale, color_range, colorbar_config in [
            (
                "lambda",
                "lambda_color",
                rainforest,
                [0.0, 1.0],
                lambda figure: configure_lambda_colorbar(figure),
            ),
            (
                "shift",
                "shift",
                rainforest,
                list(shift_limits),
                lambda figure: configure_shift_colorbar(figure, *shift_limits),
            ),
        ]:
            scatter = px.scatter(
                scatter_df,
                x="embedding_1",
                y="embedding_2",
                color=color_column,
                color_continuous_scale=colorscale,
                range_color=color_range,
                hover_data={
                    "file_id": True,
                    "lambda": ":.5g",
                    "lambda_color": False,
                    "shift": ":.5g",
                    "selected_k": True,
                    "embedding_1": ":.6g",
                    "embedding_2": ":.6g",
                },
                labels={
                    "embedding_1": "Spectral coordinate 1",
                    "embedding_2": "Spectral coordinate 2",
                    "lambda_color": "lambda",
                    "shift": "shift",
                },
                title=f"{label}: 2D spectral embedding, k={selected_k}, coloured by {color_variable}",
                render_mode="svg",
                opacity=EMBEDDING_MARKER_OPACITY,
            )
            scatter.update_traces(marker=dict(size=EMBEDDING_MARKER_SIZE))
            scatter.update_layout(template="plotly_white", font=dict(size=TICK_FONT))
            colorbar_config(scatter)
            scatter.update_xaxes(showgrid=False, zeroline=False)
            scatter.update_yaxes(showgrid=False, zeroline=False)
            save_plotly_figure(
                scatter,
                directories["scatters"] / f"initial_embedding_{set_name}_k{selected_k}_{color_variable}",
                rows=1,
                cols=1,
                panel_width=850,
                panel_height=620,
            )

        vector_grids = []
        vector_counts = []
        for vector_index in range(3):
            stats = state_stat_table(metadata, embedding[:, vector_index])
            vector_grids.append(
                state_grid(stats, "mean", state_lambdas, state_shifts)
            )
            vector_counts.append(
                state_grid(stats, "count", state_lambdas, state_shifts)
            )
        vmax = max(finite_abs_max(*vector_grids), 1e-12)
        state_fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[
                "Non-trivial spectral vector 1",
                "Non-trivial spectral vector 2",
                "Non-trivial spectral vector 3",
            ],
            horizontal_spacing=0.08,
        )
        for column, (mean_grid, count_grid) in enumerate(
            zip(vector_grids, vector_counts), start=1
        ):
            state_fig.add_trace(
                heatmap_trace(
                    mean_grid,
                    count_grid,
                    coloraxis="coloraxis",
                    quantity_label=f"mean non-trivial spectral vector {column}",
                ),
                row=1,
                col=column,
            )
        apply_coloraxis(
            state_fig,
            coloraxis="coloraxis",
            colorscale=guppy,
            cmin=-vmax,
            cmax=vmax,
            title="Mean non-trivial spectral vector",
        )
        style_state_figure(
            state_fig,
            f"{label}: first three non-trivial spectral embedding vectors, k={selected_k}",
        )
        save_plotly_figure(
            state_fig,
            directories["state_diagrams"]
            / f"initial_mean_first_three_nontrivial_vectors_{set_name}_k{selected_k}",
            rows=1,
            cols=3,
            panel_width=470,
            panel_height=460,
        )


def compute_detailed_embeddings(
    matrices: dict[str, np.ndarray],
    graph_k_values: list[int],
    selected_k: int,
) -> tuple[
    dict[tuple[str, int], object],
    dict[tuple[str, int], np.ndarray],
    dict[tuple[str, int], np.ndarray],
]:
    graphs = {}
    embeddings = {}
    eigenvalues = {}
    logging.info("Computing graph sweep for k values: %s", graph_k_values)
    for feature_set, matrix in matrices.items():
        for k in graph_k_values:
            graph = kneighbors_graph(
                matrix, n_neighbors=k, mode="connectivity", include_self=True
            )
            graph = 0.5 * (graph + graph.T)
            graphs[(feature_set, k)] = graph
            if k == selected_k:
                embeddings[(feature_set, k)] = SpectralEmbedding(
                    n_components=N_COMPONENTS,
                    affinity="precomputed",
                    random_state=SEED,
                ).fit_transform(graph)

            n_graph_components, _ = connected_components(graph, directed=False)
            n_eigenvalues = n_graph_components + N_COMPONENTS
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                values, _ = eigsh(
                    laplacian(graph, normed=True),
                    k=n_eigenvalues,
                    which="SM",
                    v0=np.random.default_rng(SEED + k).standard_normal(graph.shape[0]),
                )
            eigenvalues[(feature_set, k)] = np.sort(values)
            logging.info("Computed %s graph and spectrum for k=%d", feature_set, k)
    return graphs, embeddings, eigenvalues


def create_connectivity_table(
    graphs: dict[tuple[str, int], object], graph_k_values: list[int]
) -> pd.DataFrame:
    rows = []
    for feature_set in FEATURE_SET_GROUPS:
        for k in graph_k_values:
            graph = graphs[(feature_set, k)]
            n_components, labels = connected_components(graph, directed=False)
            sizes = np.bincount(labels)
            rows.append(
                {
                    "feature_set": feature_set,
                    "k": k,
                    "n_components": n_components,
                    "largest_component": int(sizes.max()),
                    "fraction_in_largest": float(sizes.max() / graph.shape[0]),
                }
            )
    return pd.DataFrame(rows)


def save_laplacian_spectrum(
    graphs: dict[tuple[str, int], object],
    eigenvalues: dict[tuple[str, int], np.ndarray],
    graph_k_values: list[int],
    diagnostics_dir: Path,
) -> pd.DataFrame:
    logging.info("Generating eigenvalue spectra")
    fig, axes = plt.subplots(
        1, len(graph_k_values), figsize=(4.4 * len(graph_k_values), 4.4), squeeze=False
    )
    spectrum_rows = []
    for axis, k in zip(axes.ravel(), graph_k_values):
        maximum_value = 0.0
        maximum_index = 0
        for feature_set, style in [
            ("all_features", "-o"),
            ("reduced_no_global", "--s"),
        ]:
            values = eigenvalues[(feature_set, k)]
            n_components, _ = connected_components(
                graphs[(feature_set, k)], directed=False
            )
            indices = np.arange(len(values))
            axis.plot(
                indices,
                values,
                style,
                markersize=4,
                label=FEATURE_SET_LABELS[feature_set],
            )
            if n_components > 1:
                axis.axvspan(0, n_components - 0.5, alpha=0.15, color="red")
            maximum_value = max(maximum_value, float(np.max(values)))
            maximum_index = max(maximum_index, len(values) - 1)
            spectrum_rows.extend(
                {
                    "feature_set": feature_set,
                    "k": k,
                    "eigenvalue_index": index,
                    "eigenvalue": value,
                }
                for index, value in enumerate(values)
            )
        axis.set_title(f"k = {k}")
        axis.set_xlabel("Eigenvalue index")
        axis.set_ylabel("Eigenvalue")
        axis.set_xlim(0, maximum_index + 0.5)
        axis.set_ylim(0, max(maximum_value * 1.05, 1e-12))
        axis.legend(fontsize=7)
    fig.suptitle("Laplacian eigenvalue spectrum")
    fig.tight_layout()
    path = diagnostics_dir / "laplacian_spectrum.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(spectrum_rows)


def match_components(
    embeddings: dict[tuple[str, int], np.ndarray],
    selected_k: int,
    diagnostics_dir: Path,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    full_embedding = embeddings[("all_features", selected_k)][
        :, :N_COMPONENTS_TO_ANALYZE
    ]
    reduced_embedding = embeddings[("reduced_no_global", selected_k)][
        :, :N_COMPONENTS_TO_ANALYZE
    ]
    pearson_signed = np.empty((N_COMPONENTS_TO_ANALYZE, N_COMPONENTS_TO_ANALYZE))
    spearman_signed = np.empty_like(pearson_signed)
    for full_component in range(N_COMPONENTS_TO_ANALYZE):
        for reduced_component in range(N_COMPONENTS_TO_ANALYZE):
            pearson_signed[full_component, reduced_component] = pearsonr(
                full_embedding[:, full_component],
                reduced_embedding[:, reduced_component],
            ).statistic
            spearman_signed[full_component, reduced_component] = spearmanr(
                full_embedding[:, full_component],
                reduced_embedding[:, reduced_component],
            ).statistic
    row_indices, column_indices = linear_sum_assignment(-np.abs(pearson_signed))
    matching = pd.DataFrame(
        [
            {
                "k": selected_k,
                "all_feature_comp": full_component + 1,
                "reduced_comp": reduced_component + 1,
                "pearson_r": pearson_signed[full_component, reduced_component],
                "abs_pearson": abs(pearson_signed[full_component, reduced_component]),
                "spearman_r": spearman_signed[full_component, reduced_component],
                "abs_spearman": abs(spearman_signed[full_component, reduced_component]),
                "sign": 1
                if pearson_signed[full_component, reduced_component] >= 0
                else -1,
            }
            for full_component, reduced_component in zip(row_indices, column_indices)
        ]
    ).sort_values("all_feature_comp", ignore_index=True)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    matrices = [
        pearson_signed,
        abs(pearson_signed),
        spearman_signed,
        abs(spearman_signed),
    ]
    titles = ["Pearson", "|Pearson|", "Spearman", "|Spearman|"]
    for axis, matrix, title in zip(axes, matrices, titles):
        image = axis.imshow(
            matrix,
            cmap="RdBu_r",
            vmin=-1 if "|" not in title else 0,
            vmax=1,
            aspect="equal",
        )
        axis.set(title=title, xlabel="Reduced component", ylabel="Full component")
        axis.set_xticks(
            range(N_COMPONENTS_TO_ANALYZE), range(1, N_COMPONENTS_TO_ANALYZE + 1)
        )
        axis.set_yticks(
            range(N_COMPONENTS_TO_ANALYZE), range(1, N_COMPONENTS_TO_ANALYZE + 1)
        )
        for full_component, reduced_component in zip(row_indices, column_indices):
            axis.add_patch(
                plt.Rectangle(
                    (reduced_component - 0.5, full_component - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="black",
                    linewidth=2,
                )
            )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(f"Cross-embedding correlations and Hungarian matching, k={selected_k}")
    fig.tight_layout()
    fig.savefig(
        diagnostics_dir / f"cross_embedding_matching_k{selected_k}.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)
    return matching, {"pearson": pearson_signed, "spearman": spearman_signed}


def create_state_diagrams(
    metadata: pd.DataFrame,
    embeddings: dict[tuple[str, int], np.ndarray],
    matching: pd.DataFrame,
    selected_k: int,
    state_lambdas: np.ndarray,
    state_shifts: np.ndarray,
    directories: dict[str, Path],
    guppy: list[list[float | str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logging.info("Generating state diagrams for k=%d", selected_k)
    full_embedding = embeddings[("all_features", selected_k)]
    reduced_embedding = embeddings[("reduced_no_global", selected_k)]
    matching = matching.set_index("all_feature_comp")
    selected = matching.loc[list(range(1, N_MATCHED_COMPONENTS + 1))]
    all_stats = {
        (feature_set, component): state_stat_table(metadata, embedding[:, component])
        for feature_set, embedding in [
            ("all_features", full_embedding),
            ("reduced_no_global", reduced_embedding),
        ]
        for component in range(N_COMPONENTS_TO_ANALYZE)
    }
    mean_rows: list[dict] = []
    std_rows: list[dict] = []
    difference_rows: list[dict] = []
    count_rows: list[dict] = []

    mean_grids = []
    mean_entries = []
    for component, row in selected.iterrows():
        reduced_component = int(row.reduced_comp) - 1
        sign = int(row.sign)
        full_table = all_stats[("all_features", component - 1)]
        reduced_table = state_stat_table(
            metadata, sign * reduced_embedding[:, reduced_component]
        )
        full_grid = state_grid(full_table, "mean", state_lambdas, state_shifts)
        reduced_grid = state_grid(reduced_table, "mean", state_lambdas, state_shifts)
        full_counts = state_grid(full_table, "count", state_lambdas, state_shifts)
        reduced_counts = state_grid(reduced_table, "count", state_lambdas, state_shifts)
        mean_grids.extend([full_grid, reduced_grid])
        mean_entries.append(
            (
                component,
                row,
                full_table,
                reduced_table,
                full_grid,
                reduced_grid,
                full_counts,
                reduced_counts,
            )
        )
    mean_vmax = max(finite_abs_max(*mean_grids), 1e-12)

    mean_fig = make_subplots(
        rows=2,
        cols=N_MATCHED_COMPONENTS,
        subplot_titles=[
            f"Full Spec-{component}" for component in range(1, N_MATCHED_COMPONENTS + 1)
        ]
        + [
            f"Reduced Spec-{int(row.reduced_comp)} matched to Full Spec-{component} ({'+' if row.sign > 0 else '-'} aligned)"
            for component, row in selected.iterrows()
        ],
        horizontal_spacing=0.075,
        vertical_spacing=0.16,
    )
    for column, (
        component,
        row,
        full_table,
        reduced_table,
        full_grid,
        reduced_grid,
        full_counts,
        reduced_counts,
    ) in enumerate(mean_entries, start=1):
        mean_fig.add_trace(
            heatmap_trace(
                full_grid,
                full_counts,
                coloraxis="coloraxis",
                quantity_label="mean spectral coordinate",
            ),
            row=1,
            col=column,
        )
        mean_fig.add_trace(
            heatmap_trace(
                reduced_grid,
                reduced_counts,
                coloraxis="coloraxis",
                quantity_label="mean spectral coordinate",
            ),
            row=2,
            col=column,
        )
        reduced_component = int(row.reduced_comp)
        for feature_set, original_component, table, sign_applied in [
            ("all_features", component, full_table, 1),
            ("reduced_no_global", reduced_component, reduced_table, int(row.sign)),
        ]:
            for record in table.to_dict("records"):
                mean_rows.append(
                    {
                        "k": selected_k,
                        "feature_set": feature_set,
                        "full_component": component,
                        "original_component": original_component,
                        "sign_applied": sign_applied,
                        **record,
                    }
                )
                std_rows.append(
                    {
                        "k": selected_k,
                        "feature_set": feature_set,
                        "full_component": component,
                        "original_component": original_component,
                        "sign_applied": sign_applied,
                        "lambda": record["lambda"],
                        "shift": record["shift"],
                        "std": record["std"],
                        "count": record["count"],
                    }
                )

        individual_full = go.Figure(
            heatmap_trace(
                full_grid,
                full_counts,
                coloraxis="coloraxis",
                quantity_label="mean spectral coordinate",
            )
        )
        apply_coloraxis(
            individual_full,
            coloraxis="coloraxis",
            colorscale=guppy,
            cmin=-mean_vmax,
            cmax=mean_vmax,
            title="Mean spectral coordinate",
        )
        style_state_figure(
            individual_full,
            f"Full Spec-{component}: mean spectral coordinate, k={selected_k}",
        )
        save_plotly_figure(
            individual_full,
            directories["state_diagrams"] / f"mean_full_spec_{component}_k{selected_k}",
            rows=1,
            cols=1,
            panel_width=600,
            panel_height=450,
        )

        individual_reduced = go.Figure(
            heatmap_trace(
                reduced_grid,
                reduced_counts,
                coloraxis="coloraxis",
                quantity_label="mean spectral coordinate",
            )
        )
        apply_coloraxis(
            individual_reduced,
            coloraxis="coloraxis",
            colorscale=guppy,
            cmin=-mean_vmax,
            cmax=mean_vmax,
            title="Mean spectral coordinate",
        )
        style_state_figure(
            individual_reduced,
            f"Reduced Spec-{reduced_component} aligned to Full Spec-{component}: mean spectral coordinate, k={selected_k}",
        )
        save_plotly_figure(
            individual_reduced,
            directories["state_diagrams"]
            / f"mean_reduced_spec_{reduced_component}_matched_full_{component}_k{selected_k}",
            rows=1,
            cols=1,
            panel_width=600,
            panel_height=450,
        )

    apply_coloraxis(
        mean_fig,
        coloraxis="coloraxis",
        colorscale=guppy,
        cmin=-mean_vmax,
        cmax=mean_vmax,
        title="Mean spectral coordinate",
    )
    style_state_figure(
        mean_fig,
        f"Mean spectral coordinates: full and sign-aligned reduced specifications, k={selected_k}",
    )
    save_plotly_figure(
        mean_fig,
        directories["state_diagrams"] / f"mean_spectral_components_k{selected_k}",
        rows=2,
        cols=N_MATCHED_COMPONENTS,
        panel_width=440,
        panel_height=355,
    )

    std_components = [component for component in (2, 3) if component in selected.index]
    std_entries = []
    std_grids = []
    for component in std_components:
        row = selected.loc[component]
        reduced_component = int(row.reduced_comp) - 1
        full_table = all_stats[("all_features", component - 1)]
        reduced_table = all_stats[("reduced_no_global", reduced_component)]
        full_grid = state_grid(full_table, "std", state_lambdas, state_shifts)
        reduced_grid = state_grid(reduced_table, "std", state_lambdas, state_shifts)
        full_counts = state_grid(full_table, "count", state_lambdas, state_shifts)
        reduced_counts = state_grid(reduced_table, "count", state_lambdas, state_shifts)
        std_grids.extend([full_grid, reduced_grid])
        std_entries.append(
            (component, row, full_grid, reduced_grid, full_counts, reduced_counts)
        )
    std_vmax = max(finite_max(*std_grids), 1e-12)
    std_fig = make_subplots(
        rows=2,
        cols=len(std_components),
        horizontal_spacing=0.11,
        vertical_spacing=0.16,
        subplot_titles=[f"Full Spec-{component}" for component in std_components]
        + [
            f"Reduced Spec-{int(selected.loc[component, 'reduced_comp'])} matched to Full Spec-{component}"
            for component in std_components
        ],
    )
    for column, (
        _,
        _,
        full_grid,
        reduced_grid,
        full_counts,
        reduced_counts,
    ) in enumerate(std_entries, start=1):
        std_fig.add_trace(
            heatmap_trace(
                full_grid,
                full_counts,
                coloraxis="coloraxis",
                quantity_label="within-state sample standard deviation",
            ),
            row=1,
            col=column,
        )
        std_fig.add_trace(
            heatmap_trace(
                reduced_grid,
                reduced_counts,
                coloraxis="coloraxis",
                quantity_label="within-state sample standard deviation",
            ),
            row=2,
            col=column,
        )
    apply_coloraxis(
        std_fig,
        coloraxis="coloraxis",
        colorscale=guppy,
        cmin=0,
        cmax=std_vmax,
        title="Within-state sample standard deviation",
    )
    style_state_figure(
        std_fig,
        f"Within-state spectral-coordinate sample standard deviations, k={selected_k}",
    )
    save_plotly_figure(
        std_fig,
        directories["state_diagrams"] / f"spectral_standard_deviations_k{selected_k}",
        rows=2,
        cols=len(std_components),
        panel_width=500,
        panel_height=355,
    )

    difference_entries = []
    difference_grids = []
    for component, row in selected.iterrows():
        reduced_component = int(row.reduced_comp) - 1
        z_full = (
            full_embedding[:, component - 1] - full_embedding[:, component - 1].mean()
        ) / full_embedding[:, component - 1].std()
        z_reduced = (
            int(row.sign)
            * (
                reduced_embedding[:, reduced_component]
                - reduced_embedding[:, reduced_component].mean()
            )
            / reduced_embedding[:, reduced_component].std()
        )
        difference_table = state_stat_table(metadata, z_full - z_reduced)
        difference_grid = state_grid(
            difference_table, "mean", state_lambdas, state_shifts
        )
        count_grid = state_grid(difference_table, "count", state_lambdas, state_shifts)
        difference_entries.append(
            (component, row, difference_table, difference_grid, count_grid)
        )
        difference_grids.append(difference_grid)
    difference_vmax = max(finite_abs_max(*difference_grids), 1e-12)
    difference_fig = make_subplots(
        rows=1,
        cols=N_MATCHED_COMPONENTS,
        horizontal_spacing=0.08,
        subplot_titles=[
            f"Full Spec-{component} - aligned Reduced Spec-{int(row.reduced_comp)}"
            for component, row, _, _, _ in difference_entries
        ],
    )
    for column, (component, row, table, grid, count_grid) in enumerate(
        difference_entries, start=1
    ):
        difference_fig.add_trace(
            heatmap_trace(
                grid, count_grid, coloraxis="coloraxis", quantity_label="mean delta z"
            ),
            row=1,
            col=column,
        )
        for record in table.to_dict("records"):
            difference_rows.append(
                {
                    "k": selected_k,
                    "full_component": component,
                    "reduced_component": int(row.reduced_comp),
                    "sign_applied": int(row.sign),
                    "mean_delta_z": record["mean"],
                    "count": record["count"],
                    "lambda": record["lambda"],
                    "shift": record["shift"],
                }
            )
    apply_coloraxis(
        difference_fig,
        coloraxis="coloraxis",
        colorscale=guppy,
        cmin=-difference_vmax,
        cmax=difference_vmax,
        title="Mean delta z",
    )
    style_state_figure(
        difference_fig,
        f"Full-minus-reduced sign-aligned spectral differences, k={selected_k}",
    )
    save_plotly_figure(
        difference_fig,
        directories["differences"] / f"mean_delta_z_k{selected_k}",
        rows=1,
        cols=N_MATCHED_COMPONENTS,
        panel_width=470,
        panel_height=430,
    )

    count_table = state_stat_table(metadata, full_embedding[:, 0])[
        ["lambda", "shift", "count"]
    ]
    count_grid = state_grid(count_table, "count", state_lambdas, state_shifts)
    count_fig = go.Figure(
        heatmap_trace(
            count_grid,
            count_grid,
            coloraxis="coloraxis",
            quantity_label="configurations per state point",
        )
    )
    apply_coloraxis(
        count_fig,
        coloraxis="coloraxis",
        colorscale=guppy,
        cmin=0,
        cmax=max(finite_max(count_grid), 1.0),
        title="Configurations",
    )
    style_state_figure(count_fig, f"Configurations per state point, k={selected_k}")
    save_plotly_figure(
        count_fig,
        directories["state_diagrams"] / f"state_point_counts_k{selected_k}",
        rows=1,
        cols=1,
        panel_width=650,
        panel_height=460,
    )
    count_rows.extend(
        {"k": selected_k, **record} for record in count_table.to_dict("records")
    )
    return (
        pd.DataFrame(mean_rows),
        pd.DataFrame(std_rows),
        pd.DataFrame(difference_rows),
        pd.DataFrame(count_rows),
    )


def create_embedding_comparison(
    embeddings: dict[tuple[str, int], np.ndarray], selected_k: int
) -> pd.DataFrame:
    rows = []
    full_embedding = embeddings[("all_features", selected_k)]
    reduced_embedding = embeddings[("reduced_no_global", selected_k)]
    for dimension in (2, 3, 5):
        full_coordinates = full_embedding[:, :dimension]
        reduced_coordinates = reduced_embedding[:, :dimension]
        full_distances, reduced_distances = (
            pdist(full_coordinates),
            pdist(reduced_coordinates),
        )
        _, _, disparity = procrustes(full_coordinates, reduced_coordinates)
        row = {
            "k": selected_k,
            "dim": dimension,
            "pearson_dist": pearsonr(full_distances, reduced_distances).statistic,
            "spearman_dist": spearmanr(full_distances, reduced_distances).statistic,
            "max_principal_angle": float(
                np.max(
                    subspace_angles(
                        np.linalg.qr(full_coordinates)[0],
                        np.linalg.qr(reduced_coordinates)[0],
                    )
                )
            ),
            "procrustes_disparity": float(disparity),
        }
        for neighbour_count in (5, 10, 20, 50):
            full_indices = (
                NearestNeighbors(n_neighbors=neighbour_count)
                .fit(full_coordinates)
                .kneighbors(return_distance=False)
            )
            reduced_indices = (
                NearestNeighbors(n_neighbors=neighbour_count)
                .fit(reduced_coordinates)
                .kneighbors(return_distance=False)
            )
            intersections = sum(
                len(set(full) & set(reduced))
                for full, reduced in zip(full_indices, reduced_indices)
            )
            unions = sum(
                len(set(full) | set(reduced))
                for full, reduced in zip(full_indices, reduced_indices)
            )
            row[f"nn_intersect_{neighbour_count}"] = intersections / (
                len(full_coordinates) * neighbour_count
            )
            row[f"nn_jaccard_{neighbour_count}"] = intersections / unions
        rows.append(row)
    return pd.DataFrame(rows)


def save_detailed_scatters(
    df: pd.DataFrame,
    metadata: pd.DataFrame,
    embeddings: dict[tuple[str, int], np.ndarray],
    matching: pd.DataFrame,
    selected_k: int,
    scatters_dir: Path,
    rainforest: list[list[float | str]],
    shift_limits: tuple[float, float],
) -> None:
    logging.info("Generating detailed embedding scatter plots")
    matching = matching.set_index("all_feature_comp")
    for full_y_component in (2, 3):
        for feature_set, x_component, y_component, x_sign, y_sign in [
            ("all_features", 1, full_y_component, 1, 1),
            (
                "reduced_no_global",
                int(matching.loc[1, "reduced_comp"]),
                int(matching.loc[full_y_component, "reduced_comp"]),
                int(matching.loc[1, "sign"]),
                int(matching.loc[full_y_component, "sign"]),
            ),
        ]:
            embedding = embeddings[(feature_set, selected_k)]
            plot_df = metadata.copy()
            plot_df["x"] = x_sign * embedding[:, x_component - 1]
            plot_df["y"] = y_sign * embedding[:, y_component - 1]
            plot_df["selected_k"] = selected_k
            plot_df["lambda_color"] = nonlinear_lambda_colors(plot_df["lambda"])
            for descriptor in GLOBAL_DESCRIPTOR_COLS:
                if descriptor in df.columns:
                    plot_df[descriptor] = df[descriptor].to_numpy()
            hover_columns = {
                "file_id": True,
                "lambda": ":.5g",
                "lambda_color": False,
                "shift": ":.5g",
                "selected_k": True,
                "x": ":.6g",
                "y": ":.6g",
            }
            hover_columns.update(
                {
                    descriptor: ":.5g"
                    for descriptor in GLOBAL_DESCRIPTOR_COLS
                    if descriptor in plot_df.columns
                }
            )
            title_feature_set = FEATURE_SET_LABELS[feature_set]
            alignment_note = (
                ""
                if feature_set == "all_features"
                else " (sign aligned to full specification)"
            )
            for color_variable, color_column, color_range, colorbar_config in [
                (
                    "lambda",
                    "lambda_color",
                    [0.0, 1.0],
                    lambda figure: configure_lambda_colorbar(figure),
                ),
                (
                    "shift",
                    "shift",
                    list(shift_limits),
                    lambda figure: configure_shift_colorbar(figure, *shift_limits),
                ),
            ]:
                fig = px.scatter(
                    plot_df,
                    x="x",
                    y="y",
                    color=color_column,
                    color_continuous_scale=rainforest,
                    range_color=color_range,
                    labels={
                        "x": f"Spec-{x_component}",
                        "y": f"Spec-{y_component}",
                        "lambda_color": "lambda",
                        "shift": "shift",
                    },
                    hover_data=hover_columns,
                    title=(
                        f"{title_feature_set}, k={selected_k}: Spec-{x_component} vs Spec-{y_component}"
                        f"{alignment_note}, coloured by {color_variable}"
                    ),
                    opacity=EMBEDDING_MARKER_OPACITY,
                    render_mode="svg",
                )
                fig.update_traces(marker=dict(size=EMBEDDING_MARKER_SIZE))
                fig.update_layout(template="plotly_white", font=dict(size=TICK_FONT))
                colorbar_config(fig)
                fig.update_xaxes(showgrid=False, zeroline=False)
                fig.update_yaxes(showgrid=False, zeroline=False)
                save_plotly_figure(
                    fig,
                    scatters_dir
                    / f"scatter_{feature_set}_k{selected_k}_spec{x_component}_vs_spec{y_component}_{color_variable}",
                    rows=1,
                    cols=1,
                    panel_width=850,
                    panel_height=620,
                )


def write_data_outputs(
    directories: dict[str, Path],
    selected_k: int,
    graph_k_values: list[int],
    input_path: Path,
    timestamp: str,
    row_count: int,
    feature_columns: dict[str, list[str]],
    matching: pd.DataFrame,
    connectivity: pd.DataFrame,
    spectrum: pd.DataFrame,
    mean_data: pd.DataFrame,
    std_data: pd.DataFrame,
    difference_data: pd.DataFrame,
    count_data: pd.DataFrame,
    comparison: pd.DataFrame,
    script_dir: Path,
) -> pd.DataFrame:
    data_dir = directories["data"]
    matching.to_csv(data_dir / f"component_matching_k{selected_k}.csv", index=False)
    connectivity.to_csv(data_dir / "connectivity.csv", index=False)
    spectrum.to_csv(data_dir / "laplacian_spectrum.csv", index=False)
    mean_data.to_csv(data_dir / "spectral_means.csv", index=False)
    std_data.to_csv(data_dir / "spectral_standard_deviations.csv", index=False)
    difference_data.to_csv(data_dir / "spectral_differences.csv", index=False)
    count_data.to_csv(data_dir / "state_point_counts.csv", index=False)
    comparison.to_csv(data_dir / "embedding_comparison.csv", index=False)

    summary_rows = []
    for feature_set in FEATURE_SET_GROUPS:
        connection = connectivity.query(
            "feature_set == @feature_set and k == @selected_k"
        ).iloc[0]
        summary_rows.append(
            {
                "feature_set": feature_set,
                "k": selected_k,
                "n_graph_components": connection.n_components,
                "fraction_largest": connection.fraction_in_largest,
                "matched_abs_pearson_mean": matching.abs_pearson.mean()
                if feature_set == "all_features"
                else np.nan,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(data_dir / "spectral_embedding_summary.csv", index=False)
    metadata = {
        "script_version": SCRIPT_VERSION,
        "git_commit": get_git_commit(script_dir),
        "run_timestamp": timestamp,
        "input_data_path": str(input_path),
        "selected_k": selected_k,
        "random_seed": SEED,
        "graph_k_values": graph_k_values,
        "n_configurations": row_count,
        "state_lambda_plot_limits": [STATE_LAMBDA_MIN, STATE_LAMBDA_MAX],
        "feature_columns": feature_columns,
        "state_point_sample_counts": sorted(
            int(count) for count in count_data["count"].unique()
        ),
        "sign_alignment": "Reduced components are matched by Hungarian assignment on absolute Pearson correlation and multiplied by the signed Pearson correlation sign.",
        "standard_deviation": "Within-state sample standard deviation computed by pandas Series.std(ddof=1).",
    }
    with (data_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=json_default)
    return summary


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("choreographer").setLevel(logging.WARNING)
    logging.getLogger("kaleido").setLevel(logging.WARNING)
    random.seed(SEED)
    np.random.seed(SEED)

    input_path = args.data_path.expanduser().resolve()
    script_dir = Path(__file__).resolve().parent
    df, feature_groups = load_and_validate_data(input_path)
    if args.k >= len(df):
        raise ValueError(
            f"--k must be smaller than the number of configurations ({len(df)}); got {args.k}."
        )

    results_dir, directories, timestamp = create_run_directories(input_path, args.k)
    logging.info("Resolved input path: %s", input_path)
    logging.info("Selected k: %d", args.k)
    logging.info("Main output directory: %s", results_dir.resolve())

    guppy = matplotlib_cmap_to_plotly(cmr.guppy_r)
    rainforest = matplotlib_cmap_to_plotly(cmr.rainforest)
    sample_metadata = df[META_COLUMNS].reset_index(drop=True).copy()
    state_lambdas = np.sort(
        sample_metadata.loc[
            sample_metadata["lambda"].between(STATE_LAMBDA_MIN, STATE_LAMBDA_MAX),
            "lambda",
        ].unique()
    )
    state_shifts = np.sort(sample_metadata["shift"].unique())
    shift_limits = (
        float(sample_metadata["shift"].min()),
        float(sample_metadata["shift"].max()),
    )
    if not len(state_lambdas) or not len(state_shifts):
        raise ValueError(
            "No state points are available within the configured lambda plotting range."
        )

    logging.info("Generating initial feature-specification plots")
    build_initial_embedding_plots(
        df,
        feature_groups,
        sample_metadata,
        args.k,
        state_lambdas,
        state_shifts,
        directories,
        guppy,
        rainforest,
        shift_limits,
    )

    matrices = {}
    feature_columns = {}
    for feature_set, groups in FEATURE_SET_GROUPS.items():
        columns = feature_columns_for_groups(feature_groups, groups)
        feature_columns[feature_set] = columns
        matrices[feature_set] = StandardScaler().fit_transform(
            df[columns].to_numpy(dtype=float)
        )
        logging.info("Detailed feature set %s: %d features", feature_set, len(columns))

    graph_k_values = sorted(set(DEFAULT_GRAPH_K_VALUES) | {args.k})
    graphs, embeddings, eigenvalues = compute_detailed_embeddings(
        matrices, graph_k_values, args.k
    )
    connectivity = create_connectivity_table(graphs, graph_k_values)
    spectrum = save_laplacian_spectrum(
        graphs, eigenvalues, graph_k_values, directories["diagnostics"]
    )
    matching, _ = match_components(embeddings, args.k, directories["diagnostics"])
    matching.to_csv(
        directories["data"] / f"component_matching_k{args.k}.csv", index=False
    )

    mean_data, std_data, difference_data, count_data = create_state_diagrams(
        sample_metadata,
        embeddings,
        matching,
        args.k,
        state_lambdas,
        state_shifts,
        directories,
        guppy,
    )
    comparison = create_embedding_comparison(embeddings, args.k)
    save_detailed_scatters(
        df,
        sample_metadata,
        embeddings,
        matching,
        args.k,
        directories["scatters"],
        rainforest,
        shift_limits,
    )
    summary = write_data_outputs(
        directories,
        args.k,
        graph_k_values,
        input_path,
        timestamp,
        len(df),
        feature_columns,
        matching,
        connectivity,
        spectrum,
        mean_data,
        std_data,
        difference_data,
        count_data,
        comparison,
        script_dir,
    )
    logging.info("Summary:\n%s", summary.to_string(index=False))
    logging.info("Results saved to: %s", results_dir.resolve())


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        logging.error("%s", error)
        sys.exit(1)
