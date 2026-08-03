#!/usr/bin/env python3
"""Recover notebook topology compositions from ablation spectral coordinates."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ML_diffusion_map_testing import (
    ABLATION_GROUPS,
    META_COLUMNS,
    SEED,
    build_ablation_graph,
    feature_columns_for_groups,
    load_and_validate_data,
)
from topology_state_map import TOPOLOGY_CATEGORIES, TOPOLOGY_COLORS


TARGET_LAMBDA_MAX = 20.0
N_SPLITS = 8
RIDGE_ALPHA = 1.0
KNN_NEIGHBORS = 5
RID_PATTERN = re.compile(r"_rid_(?P<rid>[^_]+)$")

FEATURE_SETS = {
    "gofr": ["gofr"],
    "orientation": ["orientation"],
    "Rg": ["Rg"],
    "orientation__gofr": ["orientation", "gofr"],
    "orientation__Rg": ["orientation", "Rg"],
    "Rg__gofr": ["Rg", "gofr"],
    "full_reference": list(ABLATION_GROUPS),
}
PLOT_FEATURE_SETS = (
    "gofr",
    "orientation__gofr",
    "orientation__Rg",
    "Rg__gofr",
    "full_reference",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover state-level topology compositions from ablation embeddings."
    )
    parser.add_argument(
        "--ablation-root",
        type=Path,
        default=Path(
            "results/MAG2P_order_parameters_with_crystallinity_20260803-174008-847690_k32_cryst-coarse-histograms/ablation"
        ),
        help="Completed full-ablation directory.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("results/MAG2P_order_parameters_with_crystallinity.pickle"),
        help="Descriptor pickle used by the completed ablation.",
    )
    parser.add_argument(
        "--topology-path",
        type=Path,
        default=Path("results/MAG2P_order_parameters_per_cluster-2026-1-13-18:15:54.pickle"),
        help="Exact topology source used by state_diagram.ipynb.",
    )
    return parser.parse_args()


def canonical_file_id(values: pd.Series) -> pd.Series:
    return values.astype(str).str.rstrip("/").str.rsplit("/", n=1).str[-1]


def replica_id(values: pd.Series) -> pd.Series:
    extracted = values.str.extract(RID_PATTERN)["rid"]
    if extracted.isna().any():
        raise ValueError("Every canonical file_id must contain a replica suffix '_rid_<id>'.")
    return extracted


def load_and_align_sources(
    data_path: Path, topology_path: Path
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame, pd.DataFrame]:
    descriptors, feature_groups, _ = load_and_validate_data(
        data_path.resolve(), "coarse-histograms"
    )
    topology = pd.read_pickle(topology_path.resolve()).fillna(0)
    required_topology = {
        "file_id",
        "lambda",
        "shift",
        "cluster_size",
        "structure_type",
    }
    missing = sorted(required_topology.difference(topology.columns))
    if missing:
        raise ValueError(f"Topology source is missing required columns: {missing}")

    descriptors = descriptors.copy()
    topology = topology.copy()
    descriptors["canonical_file_id"] = canonical_file_id(descriptors["file_id"])
    topology["canonical_file_id"] = canonical_file_id(topology["file_id"])
    descriptors["replica"] = replica_id(descriptors["canonical_file_id"])
    topology["replica"] = replica_id(topology["canonical_file_id"])

    descriptor_metadata = descriptors[
        ["canonical_file_id", "lambda", "shift", "replica"]
    ].copy()
    topology_metadata = topology[
        ["canonical_file_id", "lambda", "shift", "replica"]
    ].drop_duplicates()
    if descriptor_metadata["canonical_file_id"].duplicated().any():
        raise ValueError("Descriptor canonical file_id values are not unique.")
    if topology_metadata["canonical_file_id"].duplicated().any():
        raise ValueError("Topology canonical file_id metadata are not unique.")
    aligned = descriptor_metadata.merge(
        topology_metadata,
        on="canonical_file_id",
        how="outer",
        suffixes=("_descriptor", "_topology"),
        indicator=True,
        validate="one_to_one",
    )
    missing_descriptor = aligned.loc[aligned["_merge"] == "right_only"]
    missing_topology = aligned.loc[aligned["_merge"] == "left_only"]
    matched = aligned.loc[aligned["_merge"] == "both"]
    if len(missing_descriptor) or len(missing_topology):
        raise ValueError(
            "Canonical file_id alignment is incomplete: "
            f"{len(missing_descriptor)} topology-only and {len(missing_topology)} descriptor-only runs."
        )
    mismatch = (
        (matched["lambda_descriptor"] != matched["lambda_topology"])
        | (matched["shift_descriptor"] != matched["shift_topology"])
        | (matched["replica_descriptor"] != matched["replica_topology"])
    )
    if mismatch.any():
        raise ValueError(
            "Canonical file_id alignment has λ, shift, or replica mismatches for "
            f"{int(mismatch.sum())} runs."
        )
    return descriptors, feature_groups, topology, matched


def topology_composition(topology: pd.DataFrame) -> pd.DataFrame:
    """Reproduce state_diagram.ipynb's retained-category composition exactly."""
    selected = topology.loc[topology["lambda"] <= TARGET_LAMBDA_MAX].copy()
    counts = (
        selected.groupby(["lambda", "shift", "structure_type"], sort=True)["cluster_size"]
        .sum()
        .unstack("structure_type", fill_value=0)
    )
    for category in TOPOLOGY_CATEGORIES:
        if category not in counts:
            counts[category] = 0.0
    counts = counts.loc[:, list(TOPOLOGY_CATEGORIES)]
    total_cluster_size = counts.sum(axis=1)
    fractions = counts.div(total_cluster_size, axis=0).fillna(0.0)
    output = fractions.reset_index()
    output["total_cluster_size"] = total_cluster_size.to_numpy(dtype=float)
    return output


def validate_state_replicas(
    descriptors: pd.DataFrame, target: pd.DataFrame
) -> pd.DataFrame:
    selected = descriptors.loc[descriptors["lambda"] <= TARGET_LAMBDA_MAX].copy()
    counts = selected.groupby(["lambda", "shift"], sort=True).agg(
        contributing_replicas=("canonical_file_id", "size"),
        unique_replicas=("replica", "nunique"),
    )
    target_index = pd.MultiIndex.from_frame(target[["lambda", "shift"]])
    counts = counts.reindex(target_index)
    if counts.isna().any().any():
        raise ValueError("Some topology target states have no aligned descriptor replicas.")
    if not (counts["contributing_replicas"] == 8).all() or not (
        counts["unique_replicas"] == 8
    ).all():
        raise ValueError(
            "Every topology target state must have exactly eight aligned replicas; got "
            f"contributing counts {counts['contributing_replicas'].value_counts().to_dict()} and "
            f"unique replica counts {counts['unique_replicas'].value_counts().to_dict()}."
        )
    return counts.reset_index()


def state_coordinates(
    descriptors: pd.DataFrame, embedding: np.ndarray, target: pd.DataFrame
) -> pd.DataFrame:
    coordinate_columns = ["psi_1", "psi_2", "psi_3"]
    coordinates = descriptors[["canonical_file_id", "lambda", "shift", "replica"]].copy()
    coordinates[coordinate_columns] = embedding
    # Exact run alignment happens before this replica-level state average.
    averaged = (
        coordinates.loc[coordinates["lambda"] <= TARGET_LAMBDA_MAX]
        .groupby(["lambda", "shift"], sort=True)[coordinate_columns]
        .mean()
        .reset_index()
    )
    merged = target[["lambda", "shift"]].merge(
        averaged, on=["lambda", "shift"], how="outer", indicator=True, validate="one_to_one"
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError("State-level spectral coordinate averaging does not match topology targets.")
    return merged.drop(columns="_merge").sort_values(["lambda", "shift"]).reset_index(drop=True)


def grouped_splits(state_table: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = state_table["lambda"].astype(str) + "|" + state_table["shift"].astype(str)
    if groups.nunique() < N_SPLITS:
        raise ValueError(f"Need at least {N_SPLITS} unique states for GroupKFold.")
    splitter = GroupKFold(n_splits=N_SPLITS)
    return list(splitter.split(state_table, groups=groups))


def prediction_metrics(
    observed: np.ndarray, predicted: np.ndarray, ranges: np.ndarray
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    per_category_r2 = r2_score(observed, predicted, multioutput="raw_values")
    normalized_mae = np.mean(np.abs(observed - predicted), axis=0) / ranges
    composition_error = 0.5 * np.abs(observed - predicted).sum(axis=1)
    per_category = pd.DataFrame(
        {
            "category": TOPOLOGY_CATEGORIES,
            "r2": per_category_r2,
            "normalized_mae": normalized_mae,
        }
    )
    summary = {
        "macro_r2": float(np.mean(per_category_r2)),
        "variance_weighted_r2": float(
            r2_score(observed, predicted, multioutput="variance_weighted")
        ),
        "macro_normalized_mae": float(np.mean(normalized_mae)),
        "mean_composition_error": float(np.mean(composition_error)),
        "median_composition_error": float(np.median(composition_error)),
        "p90_composition_error": float(np.quantile(composition_error, 0.90)),
        "fraction_predictions_outside_unit_interval": float(
            np.any((predicted < 0.0) | (predicted > 1.0), axis=1).mean()
        ),
        "fraction_predictions_not_summing_to_one": float(
            (~np.isclose(predicted.sum(axis=1), 1.0, atol=1e-8)).mean()
        ),
        "prediction_sum_min": float(predicted.sum(axis=1).min()),
        "prediction_sum_max": float(predicted.sum(axis=1).max()),
    }
    return summary, per_category, composition_error


def run_models(
    state_table: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    feature_set: str,
    circular: bool,
    provenance_overlap: bool,
) -> tuple[list[dict], list[dict], list[pd.DataFrame], dict[str, np.ndarray]]:
    coordinate_columns = ["psi_1", "psi_2", "psi_3"]
    x_values = state_table[coordinate_columns].to_numpy(dtype=float)
    y_values = state_table[list(TOPOLOGY_CATEGORIES)].to_numpy(dtype=float)
    ranges = np.quantile(y_values, 0.95, axis=0) - np.quantile(y_values, 0.05, axis=0)
    if (ranges <= 0.0).any():
        raise ValueError(f"Topology category has zero q95-q05 range: {ranges.tolist()}")
    model_factories = {
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA)),
        "knn": lambda: make_pipeline(
            StandardScaler(), KNeighborsRegressor(n_neighbors=KNN_NEIGHBORS)
        ),
    }
    summary_rows = []
    category_rows = []
    prediction_frames = []
    predictions_by_model = {}
    for model_name, factory in model_factories.items():
        predictions = np.full_like(y_values, np.nan)
        fold_ids = np.full(len(state_table), -1, dtype=int)
        for fold, (train_index, test_index) in enumerate(splits):
            model = factory()
            model.fit(x_values[train_index], y_values[train_index])
            predictions[test_index] = model.predict(x_values[test_index])
            fold_ids[test_index] = fold
        if np.isnan(predictions).any() or (fold_ids < 0).any():
            raise RuntimeError(f"Missing out-of-fold predictions for {feature_set}/{model_name}.")
        metrics, per_category, composition_error = prediction_metrics(
            y_values, predictions, ranges
        )
        summary_rows.append(
            {
                "feature_set": feature_set,
                "model": model_name,
                "n_input_columns": int(state_table.attrs["n_input_columns"]),
                "n_states": len(state_table),
                "n_splits": len(splits),
                "connectivity_target_circular": circular,
                "connectivity_target_provenance_overlap": provenance_overlap,
                **metrics,
            }
        )
        per_category["feature_set"] = feature_set
        per_category["model"] = model_name
        category_rows.extend(per_category.to_dict("records"))
        prediction = state_table[["lambda", "shift"]].copy()
        prediction["feature_set"] = feature_set
        prediction["model"] = model_name
        prediction["fold"] = fold_ids
        prediction["composition_error"] = composition_error
        for index, category in enumerate(TOPOLOGY_CATEGORIES):
            prediction[f"observed_{category}"] = y_values[:, index]
            prediction[f"predicted_{category}"] = predictions[:, index]
        prediction_frames.append(prediction)
        predictions_by_model[model_name] = predictions
    return summary_rows, category_rows, prediction_frames, predictions_by_model


def plot_ranked_summary(summary: pd.DataFrame, plots_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for axis, (model, data) in zip(axes, summary.groupby("model", sort=True)):
        data = data.sort_values("mean_composition_error", ascending=False)
        axis.barh(data["feature_set"], data["mean_composition_error"], color="#5c4b8a")
        axis.set(title=model, xlabel="Mean out-of-fold composition error")
    figure.suptitle("Connectivity-composition recovery rankings")
    figure.tight_layout()
    figure.savefig(plots_dir / "ranked_summary.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_category_heatmap(per_category: pd.DataFrame, plots_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(16, 7))
    for axis, metric, title in zip(
        axes,
        ("r2", "normalized_mae"),
        ("Per-category R²", "Per-category normalized MAE"),
    ):
        table = per_category.pivot(
            index=["model", "feature_set"], columns="category", values=metric
        ).reindex(columns=TOPOLOGY_CATEGORIES)
        image = axis.imshow(table.to_numpy(dtype=float), aspect="auto", cmap="viridis")
        axis.set(
            title=title,
            yticks=np.arange(len(table)),
            yticklabels=[f"{model}: {feature_set}" for model, feature_set in table.index],
            xticks=np.arange(len(TOPOLOGY_CATEGORIES)),
            xticklabels=TOPOLOGY_CATEGORIES,
        )
        axis.tick_params(axis="x", labelrotation=35)
        figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(plots_dir / "per_category_metric_heatmap.pdf", bbox_inches="tight")
    plt.close(figure)


def state_grid(
    state_table: pd.DataFrame, values: np.ndarray, lambda_values: np.ndarray, shift_values: np.ndarray
) -> np.ndarray:
    table = state_table[["lambda", "shift"]].copy()
    table["value"] = values
    return (
        table.pivot(index="lambda", columns="shift", values="value")
        .reindex(index=lambda_values, columns=shift_values)
        .to_numpy(dtype=float)
    )


def plot_composition_maps(
    state_table: pd.DataFrame,
    values: np.ndarray,
    lambda_values: np.ndarray,
    shift_values: np.ndarray,
    title: str,
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, len(TOPOLOGY_CATEGORIES), figsize=(4 * len(TOPOLOGY_CATEGORIES), 4))
    for axis, category, category_values in zip(axes, TOPOLOGY_CATEGORIES, values.T):
        cmap = LinearSegmentedColormap.from_list(
            f"{category}_composition", ["#ffffff", TOPOLOGY_COLORS[category]]
        )
        grid = state_grid(state_table, category_values, lambda_values, shift_values)
        image = axis.imshow(grid, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=1)
        axis.set(
            title=category.replace("_", " "),
            xticks=np.arange(len(shift_values)),
            xticklabels=[f"{value:g}" for value in shift_values],
            yticks=np.arange(len(lambda_values)),
            yticklabels=[f"{value:g}" for value in lambda_values],
            xlabel="shift",
        )
        axis.tick_params(axis="x", labelrotation=90, labelsize=7)
        axis.tick_params(axis="y", labelsize=7)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("λ")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def plot_error_map(
    state_table: pd.DataFrame,
    composition_error: np.ndarray,
    lambda_values: np.ndarray,
    shift_values: np.ndarray,
    title: str,
    path: Path,
) -> None:
    grid = state_grid(state_table, composition_error, lambda_values, shift_values)
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(grid, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
    axis.set(
        title=title,
        xticks=np.arange(len(shift_values)),
        xticklabels=[f"{value:g}" for value in shift_values],
        yticks=np.arange(len(lambda_values)),
        yticklabels=[f"{value:g}" for value in lambda_values],
        xlabel="shift",
        ylabel="λ",
    )
    axis.tick_params(axis="x", labelrotation=90, labelsize=8)
    figure.colorbar(image, ax=axis, label="Composition error")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ablation_root = args.ablation_root.expanduser().resolve()
    manifest_path = ablation_root / "data" / "full_group_ablation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Completed ablation manifest does not exist: {manifest_path}")
    ablation_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if ablation_manifest.get("scope") != "full" or ablation_manifest.get("n_feature_combinations") != 31:
        raise ValueError("Connectivity recovery requires a completed full 31-combination ablation.")
    output_root = ablation_root / "connectivity_recovery"
    data_dir = output_root / "data"
    plots_dir = output_root / "plots"
    if output_root.exists():
        raise RuntimeError(f"Refusing to overwrite an existing connectivity study: {output_root}")
    data_dir.mkdir(parents=True)
    plots_dir.mkdir()

    descriptors, feature_groups, topology, alignment = load_and_align_sources(
        args.data_path, args.topology_path
    )
    target = topology_composition(topology)
    replica_counts = validate_state_replicas(descriptors, target)
    lambda_values = np.sort(target["lambda"].unique())
    shift_values = np.sort(target["shift"].unique())
    target = target.sort_values(["lambda", "shift"]).reset_index(drop=True)
    splits = grouped_splits(target)
    observed = target[list(TOPOLOGY_CATEGORIES)].to_numpy(dtype=float)
    plot_composition_maps(
        target,
        observed,
        lambda_values,
        shift_values,
        "Observed connectivity-category composition",
        plots_dir / "observed_composition_state_diagrams.pdf",
    )

    summary_rows = []
    category_rows = []
    prediction_frames = []
    plot_predictions: dict[str, dict[str, tuple[pd.DataFrame, np.ndarray]]] = {}
    for feature_set, groups in FEATURE_SETS.items():
        columns = feature_columns_for_groups(feature_groups, groups)
        matrix = StandardScaler().fit_transform(descriptors[columns].to_numpy(dtype=float))
        _, graph = build_ablation_graph(matrix, int(ablation_manifest["selected_k"]))
        # Embeddings are fitted once on all 2,760 samples; CV applies only downstream.
        embedding = SpectralEmbedding(
            n_components=3,
            affinity="precomputed",
            random_state=SEED,
        ).fit_transform(graph)
        coordinates = state_coordinates(descriptors, embedding, target)
        state_table = target.merge(
            coordinates, on=["lambda", "shift"], how="inner", validate="one_to_one"
        ).sort_values(["lambda", "shift"]).reset_index(drop=True)
        if len(state_table) != len(target):
            raise RuntimeError(f"State target mismatch for {feature_set}.")
        state_table.attrs["n_input_columns"] = len(columns)
        provenance_overlap = "Rg" in groups or "global" in groups
        rows, categories, frames, predictions = run_models(
            state_table,
            splits,
            feature_set,
            circular=False,
            provenance_overlap=provenance_overlap,
        )
        summary_rows.extend(rows)
        category_rows.extend(categories)
        prediction_frames.extend(frames)
        plot_predictions[feature_set] = {
            model: (state_table, predicted)
            for model, predicted in predictions.items()
        }
        logging.info("Connectivity recovery complete: %s", feature_set)

    summary = pd.DataFrame(summary_rows)
    summary["composition_error_rank"] = summary.groupby("model")[
        "mean_composition_error"
    ].rank(method="min", ascending=True)
    summary = summary.sort_values(["model", "composition_error_rank", "feature_set"])
    per_category = pd.DataFrame(category_rows).sort_values(["model", "feature_set", "category"])
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary.to_csv(data_dir / "connectivity_recovery_summary.csv", index=False)
    per_category.to_csv(data_dir / "connectivity_recovery_per_category_metrics.csv", index=False)
    predictions.to_csv(data_dir / "connectivity_recovery_oof_predictions.csv", index=False)
    replica_counts.to_csv(data_dir / "aligned_target_state_replica_counts.csv", index=False)
    plot_ranked_summary(summary, plots_dir)
    plot_category_heatmap(per_category, plots_dir)

    best_feature_sets = summary.loc[
        summary.groupby("model")["mean_composition_error"].idxmin(), "feature_set"
    ].tolist()
    selected_for_plots = list(dict.fromkeys([*PLOT_FEATURE_SETS, *best_feature_sets]))
    for feature_set in selected_for_plots:
        for model, (state_table, predicted) in plot_predictions[feature_set].items():
            _, _, composition_error = prediction_metrics(
                state_table[list(TOPOLOGY_CATEGORIES)].to_numpy(dtype=float),
                predicted,
                np.quantile(observed, 0.95, axis=0) - np.quantile(observed, 0.05, axis=0),
            )
            plot_composition_maps(
                state_table,
                predicted,
                lambda_values,
                shift_values,
                f"{feature_set}: {model} predicted composition",
                plots_dir / f"{feature_set}_{model}_predicted_composition_state_diagrams.pdf",
            )
            plot_error_map(
                state_table,
                composition_error,
                lambda_values,
                shift_values,
                f"{feature_set}: {model} composition error",
                plots_dir / f"{feature_set}_{model}_composition_error_state_diagram.pdf",
            )

    manifest = {
        "ablation_root": str(ablation_root),
        "ablation_manifest": ablation_manifest,
        "descriptor_data_path": str(args.data_path.expanduser().resolve()),
        "topology_data_path": str(args.topology_path.expanduser().resolve()),
        "topology_target": {
            "form": "state-level cluster-size-weighted composition",
            "categories": list(TOPOLOGY_CATEGORIES),
            "excluded_category": "tree",
            "lambda_max": TARGET_LAMBDA_MAX,
            "n_target_states": len(target),
        },
        "alignment": {
            "method": "canonical run-directory file_id, followed by λ, shift, and replica validation",
            "matched_runs": len(alignment),
            "contributing_replicas_per_target_state": 8,
            "all_target_states_have_eight_replicas": True,
            "state_level_predictor_averaging": "Replica-level coordinates are averaged only after exact run alignment.",
        },
        "embedding": {
            "scope": "Each spectral embedding is fitted once on the full 2,760-sample dataset.",
            "cross_validated": False,
            "coordinates": ["psi_1", "psi_2", "psi_3"],
        },
        "regression": {
            "scope": "Only downstream state-level regression is cross-validated.",
            "splitter": f"GroupKFold(n_splits={N_SPLITS}) grouped by unique (λ, shift)",
            "predictor_scaling": "StandardScaler fitted inside each training fold only",
            "models": {"ridge_alpha": RIDGE_ALPHA, "knn_neighbors": KNN_NEIGHBORS},
            "predictions": "Raw out-of-fold predictions; no clipping or simplex normalization for primary metrics.",
        },
        "feature_sets": FEATURE_SETS,
        "circularity": {
            "connectivity_target_circular": False,
            "definition": "No tested feature set contains topology category fractions or structure_type labels directly.",
            "provenance_overlap_sets": [
                name for name, groups in FEATURE_SETS.items() if "Rg" in groups or "global" in groups
            ],
        },
        "best_feature_set_by_model": {
            model: summary.loc[summary["model"] == model].iloc[0]["feature_set"]
            for model in sorted(summary["model"].unique())
        },
    }
    with (data_dir / "connectivity_recovery_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    logging.info("Connectivity recovery study saved to: %s", output_root)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        logging.error("%s", error)
        raise SystemExit(1)
