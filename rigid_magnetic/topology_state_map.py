"""Topology-state grid construction shared by publication figures.

This preserves the first panel semantics of ``state_diagram.ipynb``: five
included topology classes, top-two fractions, and deterministic bent RGB mixes.
The source notebook intentionally excludes ``tree`` from the retained fractions.
"""

from __future__ import annotations

import colorsys
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TOPOLOGY_DATA_PATH = Path(
    "results/MAG2P_order_parameters_per_cluster-2026-1-13-18:15:54.pickle"
)
TOPOLOGY_CATEGORIES = (
    "chain",
    "liquid",
    "complex_network",
    "ring",
    "strongly_clustered",
)
TOPOLOGY_LABELS = {
    "chain": "chain",
    "liquid": "liquid",
    "complex_network": "complex network",
    "ring": "ring",
    "strongly_clustered": "strongly clustered",
}
TOPOLOGY_COLORS = {
    "chain": "#000000",
    "liquid": "#86878A",
    "complex_network": "#1CA879",
    "ring": "#591FDF",
    "strongly_clustered": "#EEBF25",
}
BEND = 0.35
MISSING_TOPOLOGY_COLOR = "#ffffff"


@dataclass(frozen=True)
class TopologyStateMap:
    """Per-state RGB mixture and notebook-derived topology metadata."""

    rgba: np.ndarray
    top_primary: np.ndarray
    top_secondary: np.ndarray
    primary_fraction: np.ndarray
    secondary_fraction: np.ndarray
    top_two_mass: np.ndarray
    total_cluster_size: np.ndarray


def _hex_to_rgb(color: str) -> np.ndarray:
    value = color.lstrip("#")
    return np.array([int(value[index : index + 2], 16) for index in (0, 2, 4)])


def _pair_mid_color(first: str, second: str) -> np.ndarray:
    key = "|".join(sorted((first, second))).encode("utf-8")
    hue = (int(hashlib.md5(key).hexdigest()[:8], 16) % 360) / 360.0
    return np.array(colorsys.hsv_to_rgb(hue, 0.65, 0.97)) * 255.0


def _bent_mix(first: np.ndarray, second: np.ndarray, midpoint: np.ndarray, fraction: float) -> np.ndarray:
    straight_midpoint = 0.5 * (first + second)
    control_point = (1.0 - BEND) * straight_midpoint + BEND * midpoint
    return (
        (1.0 - fraction) ** 2 * first
        + 2.0 * (1.0 - fraction) * fraction * control_point
        + fraction**2 * second
    )


def build_topology_state_map(
    data_path: Path,
    lambda_values: np.ndarray,
    shift_values: np.ndarray,
) -> TopologyStateMap:
    """Build the notebook's top-two topology mixtures on a requested state grid."""
    if not data_path.is_file():
        raise FileNotFoundError(f"Topology data file does not exist: {data_path}")
    topology = pd.read_pickle(data_path).fillna(0)
    required = {"lambda", "shift", "structure_type", "cluster_size"}
    missing = sorted(required.difference(topology.columns))
    if missing:
        raise ValueError(f"Topology data is missing required columns: {missing}")

    lambda_values = np.asarray(lambda_values, dtype=float)
    shift_values = np.asarray(shift_values, dtype=float)
    selected = topology.loc[
        topology["lambda"].isin(lambda_values) & topology["shift"].isin(shift_values)
    ].copy()
    state_index = pd.MultiIndex.from_product(
        [lambda_values, shift_values], names=["lambda", "shift"]
    )
    presence = selected.groupby(["lambda", "shift"], sort=True).size().reindex(state_index)
    counts = (
        selected.groupby(["lambda", "shift", "structure_type"], sort=True)["cluster_size"]
        .sum()
        .unstack("structure_type", fill_value=0)
        .reindex(state_index, fill_value=0)
    )
    for category in TOPOLOGY_CATEGORIES:
        if category not in counts:
            counts[category] = 0
    # Deliberately retain the notebook's five-category denominator and tree exclusion.
    counts = counts.loc[:, list(TOPOLOGY_CATEGORIES)]
    totals = counts.sum(axis=1)
    fractions = counts.div(totals, axis=0).fillna(0.0).to_numpy(dtype=float)
    top_indices = np.argsort(-fractions, axis=1)[:, :2]
    top_values = np.take_along_axis(fractions, top_indices, axis=1)
    category_array = np.asarray(TOPOLOGY_CATEGORIES, dtype=object)
    primary = category_array[top_indices[:, 0]]
    secondary = category_array[top_indices[:, 1]]
    primary_fraction = top_values[:, 0]
    secondary_fraction = top_values[:, 1]
    mass = np.maximum(primary_fraction + secondary_fraction, 1e-12)
    blend_fraction = secondary_fraction / mass

    shape = (len(lambda_values), len(shift_values))
    rgba = np.ones((*shape, 4), dtype=float)
    primary_grid = primary.reshape(shape).astype(object)
    secondary_grid = secondary.reshape(shape).astype(object)
    primary_fraction_grid = primary_fraction.reshape(shape)
    secondary_fraction_grid = secondary_fraction.reshape(shape)
    mass_grid = mass.reshape(shape)
    total_grid = totals.to_numpy(dtype=float).reshape(shape)
    present_grid = presence.notna().to_numpy().reshape(shape)
    base_colors = {name: _hex_to_rgb(color) for name, color in TOPOLOGY_COLORS.items()}
    for row in range(shape[0]):
        for column in range(shape[1]):
            if not present_grid[row, column]:
                primary_grid[row, column] = None
                secondary_grid[row, column] = None
                primary_fraction_grid[row, column] = np.nan
                secondary_fraction_grid[row, column] = np.nan
                mass_grid[row, column] = np.nan
                total_grid[row, column] = np.nan
                continue
            first = primary_grid[row, column]
            second = secondary_grid[row, column]
            rgb = _bent_mix(
                base_colors[first],
                base_colors[second],
                _pair_mid_color(first, second),
                float(blend_fraction.reshape(shape)[row, column]),
            )
            rgba[row, column, :3] = np.clip(rgb / 255.0, 0.0, 1.0)

    return TopologyStateMap(
        rgba=rgba,
        top_primary=primary_grid,
        top_secondary=secondary_grid,
        primary_fraction=primary_fraction_grid,
        secondary_fraction=secondary_fraction_grid,
        top_two_mass=mass_grid,
        total_cluster_size=total_grid,
    )
