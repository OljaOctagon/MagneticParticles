#!/usr/bin/env python3
"""Robustness of benchmark spectral neighborhoods to graph-k variation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.sparse.linalg import eigsh
from scipy.stats import pearsonr, spearmanr
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import StandardScaler

from ML_diffusion_map_testing import (
    ABLATION_GROUPS,
    EMBEDDING_MARKER_OPACITY,
    EMBEDDING_MARKER_SIZE,
    META_COLUMNS,
    SEED,
    STATE_LAMBDA_MAX,
    TICK_FONT,
    TITLE_FONT,
    build_ablation_graph,
    directed_neighborhoods,
    feature_columns_for_groups,
    finite_max,
    json_default,
    load_and_validate_data,
    matplotlib_cmap_to_plotly,
    neighbor_target_consistency,
    neighborhood_overlap,
    save_aligned_spectral_visualization_family,
)
from connectivity_recovery_study import (
    KNN_NEIGHBORS,
    N_SPLITS,
    RIDGE_ALPHA,
    TARGET_LAMBDA_MAX,
    grouped_splits,
    load_and_align_sources,
    run_models,
    state_coordinates,
    topology_composition,
    validate_state_replicas,
)
from topology_state_map import TOPOLOGY_CATEGORIES


SPECTRUM_K_VALUES = (5, 10, 20, 32, 50, 200)
ANALYSIS_K_VALUES = (5, 10, 20, 28, 32, 36, 50, 200)
N_SPECTRUM_VALUES = 20
N_ALIGNMENT_CANDIDATES = 10
ZERO_TOLERANCE = 1e-10
NEAR_DEGENERACY_TOLERANCE = 1e-6

FEATURE_SETS = {
    "gofr": ["gofr"],
    "orientation__gofr": ["orientation", "gofr"],
    "orientation__Rg": ["orientation", "Rg"],
    "full_reference": list(ABLATION_GROUPS),
}
FEATURE_LABELS = {
    "gofr": "g(r)",
    "orientation__gofr": "orientation + g(r)",
    "orientation__Rg": "orientation + Rg",
    "full_reference": "full reference",
}
FEATURE_STYLES = {
    "gofr": ("#2B6CB0", "o"),
    "orientation__gofr": ("#008F7A", "s"),
    "orientation__Rg": ("#B7791F", "^"),
    "full_reference": ("#7B2CBF", "D"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate spectral-neighborhood robustness across graph-k values."
    )
    parser.add_argument(
        "--ablation-root",
        type=Path,
        default=Path(
            "results/MAG2P_order_parameters_with_crystallinity_20260803-174008-847690_k32_cryst-coarse-histograms/ablation"
        ),
        help="Completed full-ablation root used as benchmark provenance.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("results/MAG2P_order_parameters_with_crystallinity.pickle"),
    )
    parser.add_argument(
        "--topology-path",
        type=Path,
        default=Path("results/MAG2P_order_parameters_per_cluster-2026-1-13-18:15:54.pickle"),
    )
    parser.add_argument(
        "--output-name",
        default="spectral_neighbourhood_robustness",
        help="Fresh output directory name below --ablation-root.",
    )
    return parser.parse_args()


def spectral_diagnostics(
    graph, feature_set: str, k_graph: int
) -> tuple[np.ndarray, np.ndarray, dict[str, object], list[dict]]:
    """Compute ordered normalized-Laplacian values with direct components authoritative."""
    n_components, labels = connected_components(graph, directed=False)
    component_sizes = np.bincount(labels)
    requested = min(
        graph.shape[0] - 1,
        max(N_SPECTRUM_VALUES, n_components + 11),
    )
    lap = laplacian(graph, normed=True)
    values, vectors = eigsh(
        lap,
        k=requested,
        which="SM",
        v0=np.random.default_rng(SEED + k_graph).standard_normal(graph.shape[0]),
    )
    order = np.argsort(values)
    values = values[order]
    vectors = vectors[:, order]
    near_zero_count = int(np.count_nonzero(np.abs(values) <= ZERO_TOLERANCE))
    agreement = near_zero_count == n_components
    if not agreement:
        logging.warning(
            "%s k_graph=%d: direct components=%d but %d eigenvalues satisfy |μ| <= %.1e",
            feature_set,
            k_graph,
            n_components,
            near_zero_count,
            ZERO_TOLERANCE,
        )
    nontrivial = values[n_components:]
    gaps = {}
    for count in (2, 3, 5, 10):
        if len(nontrivial) > count:
            absolute_gap = float(nontrivial[count] - nontrivial[count - 1])
            denominator = max(abs(float(nontrivial[count - 1])), 1e-15)
            gaps[f"gap_after_{count}_nontrivial_absolute"] = absolute_gap
            gaps[f"gap_after_{count}_nontrivial_relative"] = absolute_gap / denominator
        else:
            gaps[f"gap_after_{count}_nontrivial_absolute"] = np.nan
            gaps[f"gap_after_{count}_nontrivial_relative"] = np.nan
    diagnostics = {
        "feature_set": feature_set,
        "k_graph": k_graph,
        "n_components": n_components,
        "largest_component": int(component_sizes.max()),
        "fraction_in_largest": float(component_sizes.max() / graph.shape[0]),
        "near_zero_count": near_zero_count,
        "zero_tolerance": ZERO_TOLERANCE,
        "component_count_agrees_with_near_zero_count": agreement,
        "algebraic_connectivity": float(values[1]) if len(values) > 1 else np.nan,
        "first_nontrivial_eigenvalue": float(nontrivial[0]) if len(nontrivial) else np.nan,
        **gaps,
    }
    rows = []
    for index, value in enumerate(values):
        gap_before = float(value - values[index - 1]) if index else np.nan
        rows.append(
            {
                **diagnostics,
                "eigenvalue_index": index,
                "eigenvalue": float(value),
                "adjacent_eigengap_absolute": gap_before,
                "adjacent_eigengap_relative": (
                    gap_before / max(abs(float(values[index - 1])), 1e-15)
                    if index
                    else np.nan
                ),
            }
        )
    return values, vectors, diagnostics, rows


def build_embeddings(
    matrices: dict[str, np.ndarray], k_values: tuple[int, ...]
) -> tuple[
    dict[tuple[str, int], np.ndarray],
    pd.DataFrame,
    dict[tuple[str, int], np.ndarray],
    dict[tuple[str, int], object],
    dict[tuple[str, int], object],
    dict[tuple[str, int], np.ndarray],
]:
    embeddings = {}
    diagnostics = []
    eigenvalues = {}
    graphs = {}
    directed_graphs = {}
    alignment_candidates = {}
    for feature_set, matrix in matrices.items():
        for k_graph in k_values:
            directed_graph, graph = build_ablation_graph(matrix, k_graph)
            values, eigenvectors, graph_diagnostics, rows = spectral_diagnostics(
                graph, feature_set, k_graph
            )
            # Candidate coordinates are the actual sklearn embedding representation.
            embedding = SpectralEmbedding(
                n_components=N_ALIGNMENT_CANDIDATES,
                affinity="precomputed",
                random_state=SEED,
            ).fit_transform(graph)
            graphs[(feature_set, k_graph)] = graph
            directed_graphs[(feature_set, k_graph)] = directed_graph
            # Disconnected sklearn embeddings retain extra component-null modes. For
            # cross-k matching, explicitly use the first ten modes after the direct
            # component-count block; connected graphs retain benchmark coordinates.
            alignment_candidates[(feature_set, k_graph)] = (
                embedding
                if graph_diagnostics["n_components"] == 1
                else eigenvectors[
                    :,
                    graph_diagnostics["n_components"] : graph_diagnostics["n_components"]
                    + N_ALIGNMENT_CANDIDATES,
                ]
            )
            embeddings[(feature_set, k_graph)] = embedding
            eigenvalues[(feature_set, k_graph)] = values
            diagnostics.extend(rows)
            logging.info("Computed %s graph, spectrum, and embedding for k_graph=%d", feature_set, k_graph)
    return (
        embeddings,
        pd.DataFrame(diagnostics),
        eigenvalues,
        graphs,
        directed_graphs,
        alignment_candidates,
    )


def align_coordinates(
    embeddings: dict[tuple[str, int], np.ndarray],
    alignment_candidates: dict[tuple[str, int], np.ndarray],
    eigenvalues: dict[tuple[str, int], np.ndarray],
    component_counts: dict[tuple[str, int], int],
    k_values: tuple[int, ...],
) -> tuple[dict[tuple[str, int], np.ndarray], pd.DataFrame]:
    """Align ψ1–ψ3 to each representation's deterministic k_graph=32 reference."""
    aligned = {}
    rows = []
    for feature_set in FEATURE_SETS:
        reference = embeddings[(feature_set, 32)][:, :3].copy()
        reference_signs = []
        for coordinate in range(3):
            maximum_index = int(np.argmax(np.abs(reference[:, coordinate])))
            sign = 1 if reference[maximum_index, coordinate] >= 0 else -1
            reference[:, coordinate] *= sign
            reference_signs.append(sign)
        for k_graph in k_values:
            candidates = alignment_candidates[(feature_set, k_graph)][:, :N_ALIGNMENT_CANDIDATES]
            if k_graph == 32:
                chosen = reference.copy()
                matched_indices = np.arange(3)
                correlations = np.ones(3)
                signs = np.asarray(reference_signs, dtype=int)
            else:
                correlation_matrix = np.empty((3, N_ALIGNMENT_CANDIDATES))
                for reference_index in range(3):
                    for candidate_index in range(N_ALIGNMENT_CANDIDATES):
                        correlation_matrix[reference_index, candidate_index] = pearsonr(
                            reference[:, reference_index], candidates[:, candidate_index]
                        ).statistic
                reference_indices, matched_indices = linear_sum_assignment(
                    -np.abs(correlation_matrix)
                )
                order = np.argsort(reference_indices)
                matched_indices = matched_indices[order]
                correlations = correlation_matrix[np.arange(3), matched_indices]
                signs = np.where(correlations >= 0, 1, -1)
                chosen = candidates[:, matched_indices] * signs
            aligned[(feature_set, k_graph)] = chosen
            all_values = eigenvalues[(feature_set, k_graph)]
            zero_mode_count = component_counts[(feature_set, k_graph)]
            nontrivial_values = all_values[zero_mode_count:]
            for reference_index, (candidate_index, correlation, sign) in enumerate(
                zip(matched_indices, correlations, signs), start=1
            ):
                mode_index = int(candidate_index)
                gap_before = (
                    float(nontrivial_values[mode_index] - nontrivial_values[mode_index - 1])
                    if mode_index > 0
                    else np.nan
                )
                gap_after = (
                    float(nontrivial_values[mode_index + 1] - nontrivial_values[mode_index])
                    if mode_index + 1 < len(nontrivial_values)
                    else np.nan
                )
                rows.append(
                    {
                        "feature_set": feature_set,
                        "k_graph": k_graph,
                        "aligned_coordinate": reference_index,
                        "matched_raw_coordinate_index": int(candidate_index) + 1,
                        "reference_raw_coordinate_index": reference_index,
                        "alignment_pearson_r": float(correlation),
                        "sign_applied": int(sign),
                        "coordinate_exchange": int(candidate_index) + 1 != reference_index,
                        "adjacent_gap_before_absolute": gap_before,
                        "adjacent_gap_after_absolute": gap_after,
                        "adjacent_gap_before_relative": (
                            gap_before / max(abs(float(nontrivial_values[mode_index - 1])), 1e-15)
                            if np.isfinite(gap_before)
                            else np.nan
                        ),
                        "adjacent_gap_after_relative": (
                            gap_after / max(abs(float(nontrivial_values[mode_index])), 1e-15)
                            if np.isfinite(gap_after)
                            else np.nan
                        ),
                        "near_degenerate": bool(
                            (np.isfinite(gap_before) and abs(gap_before) <= NEAR_DEGENERACY_TOLERANCE)
                            or (np.isfinite(gap_after) and abs(gap_after) <= NEAR_DEGENERACY_TOLERANCE)
                        ),
                        "near_degeneracy_tolerance": NEAR_DEGENERACY_TOLERANCE,
                    }
                )
    return aligned, pd.DataFrame(rows)


def physical_and_graph_scores(
    graphs: dict[tuple[str, int], object],
    directed_graphs: dict[tuple[str, int], object],
    descriptors: pd.DataFrame,
    k_values: tuple[int, ...],
) -> pd.DataFrame:
    rows = []
    normalizations = {
        target: float(np.quantile(descriptors[target], 0.95) - np.quantile(descriptors[target], 0.05))
        for target in ("mean_bonds_1_8", "mean_size")
    }
    for k_graph in k_values:
        reference = directed_neighborhoods(directed_graphs[("full_reference", k_graph)])
        for feature_set in FEATURE_SETS:
            neighborhoods = directed_neighborhoods(directed_graphs[(feature_set, k_graph)])
            overlap = neighborhood_overlap(neighborhoods, reference)
            bond_spearman, bond_nmae, _ = neighbor_target_consistency(
                neighborhoods, descriptors["mean_bonds_1_8"].to_numpy(float), normalizations["mean_bonds_1_8"]
            )
            size_spearman, size_nmae, _ = neighbor_target_consistency(
                neighborhoods, descriptors["mean_size"].to_numpy(float), normalizations["mean_size"]
            )
            n_components, labels = connected_components(graphs[(feature_set, k_graph)], directed=False)
            sizes = np.bincount(labels)
            rows.append(
                {
                    "feature_set": feature_set,
                    "k_graph": k_graph,
                    "is_full_reference": feature_set == "full_reference",
                    "n_components": n_components,
                    "fraction_in_largest": float(sizes.max() / len(descriptors)),
                    "rank_valid": float(sizes.max() / len(descriptors)) >= 0.99,
                    "mean_bonds_1_8_spearman": bond_spearman,
                    "mean_bonds_1_8_normalized_mae": bond_nmae,
                    "mean_size_spearman": size_spearman,
                    "mean_size_normalized_mae": size_nmae,
                    "full_reference_neighborhood_agreement_mean": float(np.mean(overlap)),
                    "full_reference_neighborhood_agreement_median": float(np.median(overlap)),
                    "full_reference_neighborhood_agreement_p10": float(np.quantile(overlap, 0.1)),
                }
            )
    scores = pd.DataFrame(rows)
    scores["physical_rank_bonds"] = np.nan
    scores["physical_rank_size"] = np.nan
    scores["physical_quality_rank"] = np.nan
    scores["full_reference_neighborhood_agreement_rank"] = np.nan
    for k_graph in k_values:
        valid_reduced = (
            (scores["k_graph"] == k_graph)
            & scores["rank_valid"]
            & ~scores["is_full_reference"]
        )
        scores.loc[valid_reduced, "physical_rank_bonds"] = scores.loc[
            valid_reduced, "mean_bonds_1_8_spearman"
        ].rank(ascending=False, method="min")
        scores.loc[valid_reduced, "physical_rank_size"] = scores.loc[
            valid_reduced, "mean_size_spearman"
        ].rank(ascending=False, method="min")
        scores.loc[valid_reduced, "physical_quality_rank"] = scores.loc[
            valid_reduced, ["physical_rank_bonds", "physical_rank_size"]
        ].mean(axis=1)
        scores.loc[valid_reduced, "full_reference_neighborhood_agreement_rank"] = scores.loc[
            valid_reduced, "full_reference_neighborhood_agreement_mean"
        ].rank(ascending=False, method="min")
    return scores


def connectivity_scores(
    descriptors: pd.DataFrame,
    aligned: dict[tuple[str, int], np.ndarray],
    target: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    k_values: tuple[int, ...],
    feature_counts: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    category_rows = []
    prediction_frames = []
    for feature_set, groups in FEATURE_SETS.items():
        for k_graph in k_values:
            coordinates = state_coordinates(descriptors, aligned[(feature_set, k_graph)], target)
            state_table = target.merge(
                coordinates, on=["lambda", "shift"], how="inner", validate="one_to_one"
            ).sort_values(["lambda", "shift"]).reset_index(drop=True)
            if len(state_table) != len(target):
                raise RuntimeError(f"Topology target state mismatch for {feature_set}, k_graph={k_graph}.")
            state_table.attrs["n_input_columns"] = feature_counts[feature_set]
            rows, categories, frames, _ = run_models(
                state_table,
                splits,
                feature_set,
                circular=False,
                provenance_overlap=("Rg" in FEATURE_SETS[feature_set] or feature_set == "full_reference"),
            )
            for row in rows:
                row["k_graph"] = k_graph
            for row in categories:
                row["k_graph"] = k_graph
            for frame in frames:
                frame["k_graph"] = k_graph
            summary_rows.extend(rows)
            category_rows.extend(categories)
            prediction_frames.extend(frames)
    summary = pd.DataFrame(summary_rows)
    summary["composition_error_rank"] = summary.groupby(["k_graph", "model"])[
        "mean_composition_error"
    ].rank(ascending=True, method="min")
    return summary, pd.DataFrame(category_rows), pd.concat(prediction_frames, ignore_index=True)


def correlation_table(
    descriptors: pd.DataFrame,
    aligned: dict[tuple[str, int], np.ndarray],
    target: pd.DataFrame,
    k_values: tuple[int, ...],
) -> pd.DataFrame:
    rows = []
    for feature_set in FEATURE_SETS:
        for k_graph in k_values:
            coordinates = aligned[(feature_set, k_graph)]
            for coordinate in range(3):
                for variable in ("lambda", "shift"):
                    values = descriptors[variable].to_numpy(float)
                    rows.append(
                        {
                            "feature_set": feature_set,
                            "k_graph": k_graph,
                            "domain": "sample",
                            "coordinate": coordinate + 1,
                            "variable": variable,
                            "pearson_r": float(pearsonr(coordinates[:, coordinate], values).statistic),
                            "spearman_r": float(spearmanr(coordinates[:, coordinate], values).statistic),
                        }
                    )
            state_coords = state_coordinates(descriptors, coordinates, target)
            state_table = target.merge(state_coords, on=["lambda", "shift"], validate="one_to_one")
            for coordinate in range(3):
                for category in TOPOLOGY_CATEGORIES:
                    rows.append(
                        {
                            "feature_set": feature_set,
                            "k_graph": k_graph,
                            "domain": "topology_state",
                            "coordinate": coordinate + 1,
                            "variable": category,
                            "pearson_r": float(pearsonr(state_table[f"psi_{coordinate + 1}"], state_table[category]).statistic),
                            "spearman_r": float(spearmanr(state_table[f"psi_{coordinate + 1}"], state_table[category]).statistic),
                        }
                    )
    return pd.DataFrame(rows)


def plot_ordered_spectra(eigenvalues: dict[tuple[str, int], np.ndarray], plots_dir: Path) -> None:
    figure, axes = plt.subplots(1, len(SPECTRUM_K_VALUES), figsize=(4.2 * len(SPECTRUM_K_VALUES), 4.2), sharey=True)
    for axis, k_graph in zip(axes, SPECTRUM_K_VALUES):
        for feature_set in FEATURE_SETS:
            values = eigenvalues[(feature_set, k_graph)][:15]
            color, marker = FEATURE_STYLES[feature_set]
            axis.plot(np.arange(len(values)), values, color=color, marker=marker, markersize=3, label=FEATURE_LABELS[feature_set])
        axis.set(title=f"k_graph={k_graph}", xlabel="Ordered eigenvalue index")
        axis.set_yscale("symlog", linthresh=ZERO_TOLERANCE)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Laplacian eigenvalue μ_j")
    axes[-1].legend(fontsize=7, loc="upper left")
    figure.suptitle("Ordered normalized-Laplacian spectra")
    figure.tight_layout()
    figure.savefig(plots_dir / "ordered_laplacian_spectra.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_score_stability(scores: pd.DataFrame, plots_dir: Path) -> None:
    figure, axes = plt.subplots(1, 5, figsize=(27, 4.5))
    metrics = [
        ("mean_bonds_1_8_spearman", "Bond Spearman"),
        ("mean_size_spearman", "Mean-size Spearman"),
        ("mean_bonds_1_8_normalized_mae", "Bond normalized MAE"),
        ("mean_size_normalized_mae", "Mean-size normalized MAE"),
        ("full_reference_neighborhood_agreement_mean", "Agreement with full-reference neighborhood graph"),
    ]
    for axis, (metric, title) in zip(axes, metrics):
        for feature_set in FEATURE_SETS:
            data = scores.loc[scores["feature_set"] == feature_set]
            color, marker = FEATURE_STYLES[feature_set]
            axis.plot(data["k_graph"], data[metric], color=color, marker=marker, label=FEATURE_LABELS[feature_set])
        axis.set(title=title, xlabel="k_graph")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Raw score")
    axes[-1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(plots_dir / "physical_and_full_reference_agreement_scores.pdf", bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, metric, title in zip(
        axes,
        ("physical_quality_rank", "full_reference_neighborhood_agreement_rank"),
        (
            "Reduced physical-quality rank\n(full reference excluded)",
            "Agreement with full-reference neighborhood graph rank",
        ),
    ):
        for feature_set in FEATURE_SETS:
            data = scores.loc[scores["feature_set"] == feature_set]
            color, marker = FEATURE_STYLES[feature_set]
            axis.plot(data["k_graph"], data[metric], color=color, marker=marker, label=FEATURE_LABELS[feature_set])
        axis.set(title=title, xlabel="k_graph")
        axis.invert_yaxis()
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Rank, lower is better")
    axes[-1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(plots_dir / "physical_and_full_reference_agreement_rank_stability.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_connectivity_stability(scores: pd.DataFrame, plots_dir: Path) -> None:
    for metric, title, filename in [
        ("mean_composition_error", "Mean OOF topology composition error", "connectivity_raw_scores.pdf"),
        ("composition_error_rank", "Topology composition-error rank", "connectivity_rank_stability.pdf"),
    ]:
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        for axis, model in zip(axes, ("ridge", "knn")):
            data = scores.loc[scores["model"] == model]
            for feature_set in FEATURE_SETS:
                series = data.loc[data["feature_set"] == feature_set]
                color, marker = FEATURE_STYLES[feature_set]
                axis.plot(series["k_graph"], series[metric], color=color, marker=marker, label=FEATURE_LABELS[feature_set])
            axis.set(title=f"{title}: {model}", xlabel="k_graph")
            if metric == "composition_error_rank":
                axis.invert_yaxis()
            axis.grid(alpha=0.25)
        axes[0].set_ylabel("Rank, lower is better" if metric == "composition_error_rank" else "Raw score")
        axes[-1].legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(plots_dir / filename, bbox_inches="tight")
        plt.close(figure)


def plot_correlation_heatmaps(correlations: pd.DataFrame, plots_dir: Path) -> None:
    selected = correlations.loc[correlations["domain"] == "sample"]
    for metric in ("pearson_r", "spearman_r"):
        figure, axes = plt.subplots(len(FEATURE_SETS), 2, figsize=(12, 2.7 * len(FEATURE_SETS)), sharex=True, sharey=True)
        for row, feature_set in enumerate(FEATURE_SETS):
            for column, variable in enumerate(("lambda", "shift")):
                table = selected.loc[
                    (selected["feature_set"] == feature_set) & (selected["variable"] == variable)
                ].pivot(index="coordinate", columns="k_graph", values=metric).reindex(index=[1, 2, 3], columns=ANALYSIS_K_VALUES)
                image = axes[row, column].imshow(table.to_numpy(float), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
                axes[row, column].set(title=f"{FEATURE_LABELS[feature_set]}: {variable}", yticks=range(3), yticklabels=["ψ1", "ψ2", "ψ3"], xticks=range(len(ANALYSIS_K_VALUES)), xticklabels=ANALYSIS_K_VALUES)
                axes[row, column].tick_params(axis="x", labelrotation=45)
        figure.colorbar(image, ax=axes.ravel().tolist(), label=metric)
        figure.tight_layout()
        figure.savefig(plots_dir / f"{metric}_coordinate_correlation_stability.pdf", bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ablation_root = args.ablation_root.expanduser().resolve()
    benchmark_manifest_path = ablation_root / "data" / "full_group_ablation_manifest.json"
    if not benchmark_manifest_path.is_file():
        raise FileNotFoundError(f"Completed ablation manifest is missing: {benchmark_manifest_path}")
    benchmark_manifest = json.loads(benchmark_manifest_path.read_text(encoding="utf-8"))
    if benchmark_manifest.get("scope") != "full" or benchmark_manifest.get("n_feature_combinations") != 31:
        raise ValueError("Robustness study requires the completed full 31-combination ablation.")
    if Path(args.output_name).name != args.output_name:
        raise ValueError("--output-name must be a single directory name.")
    output_root = ablation_root / args.output_name
    data_dir = output_root / "data"
    plots_dir = output_root / "plots"
    final_dir = output_root / "final_visualizations"
    if output_root.exists():
        raise RuntimeError(f"Refusing to overwrite an existing robustness study: {output_root}")
    data_dir.mkdir(parents=True)
    plots_dir.mkdir()
    final_dir.mkdir()

    descriptors, feature_groups, topology, alignment = load_and_align_sources(
        args.data_path, args.topology_path
    )
    target = topology_composition(topology).sort_values(["lambda", "shift"]).reset_index(drop=True)
    replica_counts = validate_state_replicas(descriptors, target)
    splits = grouped_splits(target)
    matrices = {
        feature_set: StandardScaler().fit_transform(
            descriptors[feature_columns_for_groups(feature_groups, groups)].to_numpy(dtype=float)
        )
        for feature_set, groups in FEATURE_SETS.items()
    }
    feature_counts = {
        feature_set: len(feature_columns_for_groups(feature_groups, groups))
        for feature_set, groups in FEATURE_SETS.items()
    }
    all_k_values = tuple(sorted(set(SPECTRUM_K_VALUES) | set(ANALYSIS_K_VALUES)))
    (
        embeddings,
        spectrum_table,
        eigenvalues,
        graphs,
        directed_graphs,
        alignment_candidates,
    ) = build_embeddings(matrices, all_k_values)
    component_counts = {
        (feature_set, k_graph): int(
            spectrum_table.loc[
                (spectrum_table["feature_set"] == feature_set)
                & (spectrum_table["k_graph"] == k_graph),
                "n_components",
            ].iloc[0]
        )
        for feature_set in FEATURE_SETS
        for k_graph in all_k_values
    }
    aligned, alignment_table = align_coordinates(
        embeddings, alignment_candidates, eigenvalues, component_counts, all_k_values
    )
    physical_scores = physical_and_graph_scores(
        graphs, directed_graphs, descriptors, ANALYSIS_K_VALUES
    )
    connectivity_summary, connectivity_categories, connectivity_predictions = connectivity_scores(
        descriptors, aligned, target, splits, ANALYSIS_K_VALUES, feature_counts
    )
    connectivity_summary["n_input_columns"] = connectivity_summary["feature_set"].map(feature_counts)
    correlations = correlation_table(descriptors, aligned, target, ANALYSIS_K_VALUES)
    spectrum_table.to_csv(data_dir / "ordered_laplacian_spectra.csv", index=False)
    physical_scores.to_csv(data_dir / "physical_and_graph_scores_and_ranks.csv", index=False)
    connectivity_summary.to_csv(data_dir / "connectivity_recovery_scores_and_ranks.csv", index=False)
    connectivity_categories.to_csv(data_dir / "connectivity_per_category_metrics.csv", index=False)
    connectivity_predictions.to_csv(data_dir / "connectivity_oof_predictions.csv", index=False)
    correlations.to_csv(data_dir / "coordinate_correlations.csv", index=False)
    alignment_table.to_csv(data_dir / "coordinate_alignment_and_degeneracy.csv", index=False)
    replica_counts.to_csv(data_dir / "topology_target_replica_counts.csv", index=False)
    plot_ordered_spectra(eigenvalues, plots_dir)
    plot_score_stability(physical_scores, plots_dir)
    plot_connectivity_stability(connectivity_summary, plots_dir)
    plot_correlation_heatmaps(correlations, plots_dir)

    final_k_graph = 32
    final_metadata = descriptors[META_COLUMNS].reset_index(drop=True)
    final_state_coordinates = []
    lilac = matplotlib_cmap_to_plotly(cmr.lilac)
    pride = matplotlib_cmap_to_plotly(cmr.pride)
    state_lambdas = np.sort(
        final_metadata.loc[final_metadata["lambda"] <= STATE_LAMBDA_MAX, "lambda"].unique()
    )
    state_shifts = np.sort(final_metadata["shift"].unique())
    shift_limits = (float(state_shifts.min()), float(state_shifts.max()))
    for feature_set in FEATURE_SETS:
        coordinates = aligned[(feature_set, final_k_graph)]
        samples = final_metadata.copy()
        samples["feature_set"] = feature_set
        samples["k_graph"] = final_k_graph
        samples[["psi_1", "psi_2", "psi_3"]] = coordinates
        samples.to_csv(data_dir / f"final_sample_coordinates_{feature_set}.csv", index=False)
        state = (
            samples.loc[samples["lambda"] <= STATE_LAMBDA_MAX]
            .groupby(["lambda", "shift", "feature_set", "k_graph"], sort=True)[["psi_1", "psi_2", "psi_3"]]
            .mean()
            .reset_index()
        )
        final_state_coordinates.append(state)
        save_aligned_spectral_visualization_family(
            feature_set,
            FEATURE_LABELS[feature_set],
            coordinates,
            final_metadata,
            final_k_graph,
            state_lambdas,
            state_shifts,
            final_dir,
            lilac,
            pride,
            shift_limits,
        )
    pd.concat(final_state_coordinates, ignore_index=True).to_csv(
        data_dir / "final_state_averaged_coordinates.csv", index=False
    )

    graph_diagnostics = spectrum_table.drop_duplicates(["feature_set", "k_graph"])
    usable_by_k = (
        graph_diagnostics.groupby("k_graph")["n_components"].max().rename("max_components").reset_index()
    )
    manifest = {
        "benchmark_ablation_root": str(ablation_root),
        "benchmark_ablation_manifest": benchmark_manifest,
        "feature_sets": FEATURE_SETS,
        "feature_columns": {
            feature_set: feature_columns_for_groups(feature_groups, groups)
            for feature_set, groups in FEATURE_SETS.items()
        },
        "preprocessing": "StandardScaler fit independently to each representation over all 2760 samples, matching the completed ablation.",
        "graph_construction": "Self-inclusive directed kNN with k_graph - 1 non-self neighbors, then 0.5 * (A + A.T) symmetrization.",
        "laplacian": "scipy.sparse.csgraph.laplacian(graph, normed=True)",
        "eigensolver": "scipy.sparse.linalg.eigsh(which='SM')",
        "zero_tolerance": ZERO_TOLERANCE,
        "near_degeneracy_tolerance": NEAR_DEGENERACY_TOLERANCE,
        "spectrum_k_graph_values": SPECTRUM_K_VALUES,
        "analysis_k_graph_values": ANALYSIS_K_VALUES,
        "coordinate_alignment": "k_graph=32 references oriented by largest-absolute sample entry; other k values matched by Hungarian absolute Pearson correlation against ten candidate coordinates. Connected graphs use benchmark embedding coordinates; disconnected graphs use the first ten Laplacian eigenvectors after the direct component-count zero-mode block. Matched coordinates are oriented by correlation sign.",
        "connectivity_target": {
            "source": str(args.topology_path.expanduser().resolve()),
            "lambda_restriction": f"λ <= {TARGET_LAMBDA_MAX}",
            "target_states": len(target),
            "categories": list(TOPOLOGY_CATEGORIES),
            "tree_excluded": True,
            "exact_alignment": "canonical file_id plus λ, shift, and replica validation",
            "replicas_per_state": 8,
            "folds": f"GroupKFold(n_splits={N_SPLITS}) over unique (λ, shift) states",
            "decoder_knn_neighbors": KNN_NEIGHBORS,
            "ridge_alpha": RIDGE_ALPHA,
        },
        "embedding_scope": "Each representation/k_graph embedding is fit once on all 2760 samples; no spectral embedding is refit inside cross-validation folds.",
        "regression_scope": "Only the downstream state-level topology decoders are cross-validated.",
        "physical_rank_convention": "Full reference raw scores are retained but excluded from reduced-representation physical-quality ranking.",
        "graph_overlap_convention": "Agreement with the full-reference neighborhood graph; not an independent physical or topology quality measure.",
        "common_usable_k_graph": usable_by_k.loc[usable_by_k["max_components"] == 1, "k_graph"].tolist(),
        "selected_final_k_graph": final_k_graph,
        "topology_alignment_matched_runs": len(alignment),
    }
    with (data_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=json_default)
    logging.info("Spectral-neighborhood robustness study saved to: %s", output_root)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        logging.error("%s", error)
        raise SystemExit(1)
