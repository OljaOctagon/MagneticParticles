#!/usr/bin/env python3
"""Generate quasi-2D structural and crystallinity features from MAG2P dumps."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import multiprocessing as mp
import os
import re
import sys
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import freud
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from feature_schema import (
    CRYSTALLINITY_COARSE_HISTOGRAM_FEATURES,
    CRYSTALLINITY_SCALARS,
    DIAGNOSTIC_FEATURES,
    FEATURE_GROUPS,
    GLOBAL_FEATURES,
    GLOBAL_STD_FEATURES,
    LEGACY_DISTRIBUTION_MAPPING,
    META_COLUMNS,
    MODEL_FEATURES,
    ORIENTATION_BIN_EDGES,
    ORIENTATION_FEATURES,
    OUTPUT_COLUMNS,
    Q4_HISTOGRAM_FEATURES,
    Q6_HISTOGRAM_FEATURES,
    Q_BIN_EDGES,
    RDF_BIN_COUNT,
    RDF_FEATURES,
    RDF_R_MAX,
    RG_CLUSTER_SIZES,
    RG_FEATURES,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path("/media/karner/vsc3_backup_big/vsc4-backup/mag2patch_renormalized/mag2patch_renormalized")
DEFAULT_OUTPUT = Path("results/MAG2P_order_parameters_with_crystallinity.pickle")
VALIDATION_STATES = ((0.1, 10.0), (0.4, 5.0), (0.6, 5.0))
BOX_LENGTH_XY = 270.0
EXPECTED_PARTICLES = 1000
CENTER_TYPE = 1
MOMENT_TYPE = 2
BOND_CUTOFF_15 = 1.5
BOND_CUTOFF_18 = 1.8
SHELL_CUTOFF_MAX = 2.0
PSI_CUTOFF = 1.6
Q_THRESHOLD = 0.35
Q_MIN_NEIGHBORS = 4
Q_MIN_CLUSTER_SIZE = 6

RUN_PATTERN = re.compile(
    r"^mag2p_shift_(?P<shift>[^_]+)_lambda_(?P<lambda>[^_]+)_phi2d_(?P<phi2d>[^_]+)_rid_(?P<rid>[^_]+)$"
)


@dataclass(frozen=True)
class RunInfo:
    path: Path
    shift: float
    lambda_value: float
    phi2d: float
    rid: int


@dataclass
class DumpFrame:
    timestep: int
    fields: list[str]
    rows: list[list[str]]
    bounds: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MAG2P features from final common dump frames.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Read-only simulation-data root.")
    parser.add_argument("--input-glob", default="mag2p_shift_*", help="Directory glob relative to --data-root.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Repository-local output pickle path.")
    parser.add_argument("--workers", type=int, default=None, help="Worker count; defaults to one for limited runs.")
    parser.add_argument("--validate-only", action="store_true", help="Process exactly the three validation configurations.")
    parser.add_argument("--max-files", type=int, help="Process only the first valid sorted directories.")
    parser.add_argument("--full-run", action="store_true", help="Explicitly permit processing every valid directory.")
    args = parser.parse_args()
    modes = int(args.validate_only) + int(args.max_files is not None) + int(args.full_run)
    if modes != 1:
        parser.error("Choose exactly one of --validate-only, --max-files, or --full-run; full processing is never implicit.")
    if args.max_files is not None and args.max_files <= 0:
        parser.error("--max-files must be positive.")
    if args.workers is not None and args.workers <= 0:
        parser.error("--workers must be positive.")
    return args


def repo_path(path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (SCRIPT_DIR / path).resolve()


def make_box() -> freud.box.Box:
    return freud.box.Box(Lx=BOX_LENGTH_XY, Ly=BOX_LENGTH_XY, is2D=True)


def parse_run_directory(path: Path) -> RunInfo | None:
    match = RUN_PATTERN.match(path.name)
    if not match:
        logging.warning("Skipping unparseable directory name: %s", path)
        return None
    try:
        return RunInfo(
            path=path,
            shift=float(match.group("shift")),
            lambda_value=float(match.group("lambda")),
            phi2d=float(match.group("phi2d")),
            rid=int(match.group("rid")),
        )
    except ValueError:
        logging.warning("Skipping directory with non-numeric parameters: %s", path)
        return None


def discover_runs(data_root: Path, input_glob: str) -> tuple[list[RunInfo], list[Path]]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist or is not a directory: {data_root}")
    parsed: list[RunInfo] = []
    failed: list[Path] = []
    for path in sorted(data_root.glob(input_glob)):
        if not path.is_dir():
            continue
        run = parse_run_directory(path)
        if run is None:
            failed.append(path)
        else:
            parsed.append(run)
    return parsed, failed


def has_required_files(run: RunInfo) -> bool:
    missing = [name for name in ("traj.gz", "mu.gz") if not (run.path / name).is_file()]
    if missing:
        logging.warning("Skipping %s; missing required files: %s", run.path, ", ".join(missing))
        return False
    return True


def select_validation_runs(runs: list[RunInfo], failed_parsing: list[Path]) -> list[RunInfo]:
    selected = []
    available = sorted({(run.shift, run.lambda_value) for run in runs})
    for shift, lambda_value in VALIDATION_STATES:
        matches = [run for run in runs if run.shift == shift and run.lambda_value == lambda_value and has_required_files(run)]
        if not matches:
            raise ValueError(
                f"No exact valid directory for shift={shift}, lambda={lambda_value}. "
                f"Available parsed state points: {available}. Failed parses: {[str(path) for path in failed_parsing[:10]]}"
            )
        selected.append(sorted(matches, key=lambda run: str(run.path))[0])
    return selected


def select_runs(args: argparse.Namespace, runs: list[RunInfo], failed_parsing: list[Path]) -> list[RunInfo]:
    if args.validate_only:
        return select_validation_runs(runs, failed_parsing)
    if args.max_files is not None:
        selected = []
        for run in runs:
            if has_required_files(run):
                selected.append(run)
            if len(selected) == args.max_files:
                break
        return selected
    return [run for run in runs if has_required_files(run)]


def iter_dump_frames(path: Path) -> Iterator[DumpFrame]:
    """Yield complete LAMMPS dump frames; stop safely at malformed gzip tails."""
    try:
        with gzip.open(path, "rt") as handle:
            while True:
                marker = handle.readline()
                if not marker:
                    return
                if marker.strip() != "ITEM: TIMESTEP":
                    raise ValueError(f"expected 'ITEM: TIMESTEP', found {marker.strip()!r}")
                timestep_line = handle.readline()
                if not timestep_line:
                    return
                timestep = int(timestep_line.strip())
                if handle.readline().strip() != "ITEM: NUMBER OF ATOMS":
                    raise ValueError("missing 'ITEM: NUMBER OF ATOMS' marker")
                atom_count = int(handle.readline().strip())
                box_marker = handle.readline()
                if not box_marker.startswith("ITEM: BOX BOUNDS"):
                    raise ValueError("missing 'ITEM: BOX BOUNDS' marker")
                bounds_rows = [handle.readline().split() for _ in range(3)]
                if any(len(row) < 2 for row in bounds_rows):
                    return
                bounds = np.array([[float(row[0]), float(row[1])] for row in bounds_rows])
                atom_marker = handle.readline().split()
                if len(atom_marker) < 3 or atom_marker[:2] != ["ITEM:", "ATOMS"]:
                    raise ValueError("missing 'ITEM: ATOMS' marker")
                fields = atom_marker[2:]
                rows = []
                for _ in range(atom_count):
                    line = handle.readline()
                    if not line:
                        return
                    values = line.split()
                    if len(values) != len(fields):
                        raise ValueError("atom row does not match declared dump fields")
                    rows.append(values)
                yield DumpFrame(timestep=timestep, fields=fields, rows=rows, bounds=bounds)
    except (EOFError, OSError, ValueError, gzip.BadGzipFile, zlib.error) as error:
        logging.warning("Stopped reading %s after malformed or corrupted data: %s", path, error)


def latest_common_frames(traj_path: Path, mu_path: Path) -> tuple[DumpFrame, DumpFrame]:
    """Stream both dumps and retain only the latest exact common complete timestep."""
    traj_iter = iter_dump_frames(traj_path)
    mu_iter = iter_dump_frames(mu_path)
    traj_frame = next(traj_iter, None)
    mu_frame = next(mu_iter, None)
    latest: tuple[DumpFrame, DumpFrame] | None = None
    while traj_frame is not None and mu_frame is not None:
        if traj_frame.timestep == mu_frame.timestep:
            latest = (traj_frame, mu_frame)
            traj_frame = next(traj_iter, None)
            mu_frame = next(mu_iter, None)
        elif traj_frame.timestep < mu_frame.timestep:
            traj_frame = next(traj_iter, None)
        else:
            mu_frame = next(mu_iter, None)
    if latest is None:
        raise ValueError(f"No common complete timestep between {traj_path} and {mu_path}")
    return latest


def rows_as_dicts(frame: DumpFrame, required_fields: set[str], label: str) -> list[dict[str, str]]:
    missing = required_fields - set(frame.fields)
    if missing:
        raise ValueError(f"{label} timestep {frame.timestep} is missing fields: {sorted(missing)}")
    return [dict(zip(frame.fields, row)) for row in frame.rows]


def align_positions_and_moments(traj_frame: DumpFrame, mu_frame: DumpFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Join type-1 centers to one validated type-2 moment vector per molecule."""
    if traj_frame.timestep != mu_frame.timestep:
        raise ValueError("Attempted to align position and moment frames from different timesteps.")
    trajectory_rows = rows_as_dicts(traj_frame, {"id", "type", "xs", "ys", "zs"}, "traj.gz")
    moment_rows = rows_as_dicts(mu_frame, {"id", "mol", "type", "mux", "muy", "muz"}, "mu.gz")
    trajectory_centers = [row for row in trajectory_rows if int(row["type"]) == CENTER_TYPE]
    moment_center_rows = [row for row in moment_rows if int(row["type"]) == CENTER_TYPE]
    center_positions = {int(row["id"]): row for row in trajectory_centers}
    moment_centers = {int(row["id"]): int(row["mol"]) for row in moment_center_rows}
    if len(trajectory_centers) != len(center_positions) or len(moment_center_rows) != len(moment_centers):
        raise ValueError("Center particle identifiers are duplicated within a selected frame.")
    if len({int(row["mol"]) for row in moment_center_rows}) != len(moment_center_rows):
        raise ValueError("Moment-file center molecule identifiers are duplicated within a selected frame.")
    if len(center_positions) != EXPECTED_PARTICLES or len(moment_centers) != EXPECTED_PARTICLES:
        raise ValueError(
            f"Expected {EXPECTED_PARTICLES} type-{CENTER_TYPE} centers at timestep {traj_frame.timestep}; "
            f"found traj={len(center_positions)}, mu={len(moment_centers)}."
        )
    if set(center_positions) != set(moment_centers):
        raise ValueError("Trajectory center IDs and moment-file center IDs do not match.")

    patch_vectors: dict[int, list[np.ndarray]] = defaultdict(list)
    for row in moment_rows:
        if int(row["type"]) == MOMENT_TYPE:
            patch_vectors[int(row["mol"])].append(np.array([float(row["mux"]), float(row["muy"]), float(row["muz"])]))
    center_molecules = set(moment_centers.values())
    if set(patch_vectors) != center_molecules:
        raise ValueError("Moment vectors are missing or inconsistent for one or more center molecules.")
    selected_moments: dict[int, np.ndarray] = {}
    for molecule, vectors in patch_vectors.items():
        if len(vectors) != 2:
            raise ValueError(f"Expected two type-{MOMENT_TYPE} patch records for molecule {molecule}; found {len(vectors)}.")
        if not np.allclose(vectors, vectors[0], rtol=1e-6, atol=1e-10):
            raise ValueError(f"Type-{MOMENT_TYPE} patch moments disagree within molecule {molecule}.")
        selected_moments[molecule] = vectors[0]

    records = []
    bounds = traj_frame.bounds
    for center_id, row in center_positions.items():
        molecule = moment_centers[center_id]
        records.append(
            (
                molecule,
                [
                    bounds[0, 0] + float(row["xs"]) * (bounds[0, 1] - bounds[0, 0]),
                    bounds[1, 0] + float(row["ys"]) * (bounds[1, 1] - bounds[1, 0]),
                    0.0,
                ],
                selected_moments[molecule],
            )
        )
    records.sort(key=lambda record: record[0])
    molecule_ids = np.array([record[0] for record in records], dtype=int)
    if len(np.unique(molecule_ids)) != EXPECTED_PARTICLES:
        raise ValueError("Molecule identifiers are duplicated or incomplete after center/moment alignment.")
    return molecule_ids, np.asarray([record[1] for record in records]), np.asarray([record[2] for record in records])


def filtered_neighbor_list(nlist: freud.locality.NeighborList, particle_count: int, mask: np.ndarray) -> freud.locality.NeighborList:
    return freud.locality.NeighborList.from_arrays(
        particle_count,
        particle_count,
        nlist.query_point_indices[mask],
        nlist.point_indices[mask],
        nlist.vectors[mask],
    )


def unique_edges(nlist: freud.locality.NeighborList, mask: np.ndarray) -> np.ndarray:
    pairs = np.column_stack((nlist.query_point_indices[mask], nlist.point_indices[mask]))
    if not len(pairs):
        return np.empty((0, 2), dtype=int)
    pairs.sort(axis=1)
    return np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0)


def degrees_from_mask(nlist: freud.locality.NeighborList, mask: np.ndarray, particle_count: int) -> np.ndarray:
    return np.bincount(nlist.query_point_indices[mask], minlength=particle_count)


def cluster_statistics(box: freud.box.Box, positions: np.ndarray, bonded_nlist: freud.locality.NeighborList) -> tuple[dict[str, float], int]:
    cluster = freud.cluster.Cluster()
    cluster.compute((box, positions), neighbors=bonded_nlist)
    properties = freud.cluster.ClusterProperties()
    properties.compute((box, positions), cluster.cluster_idx)
    sizes = np.asarray(properties.sizes)
    radii = np.asarray(properties.radii_of_gyration)
    bonded = sizes >= 2
    bonded_sizes = sizes[bonded]
    bonded_radii = radii[bonded]
    values = {
        name: 0.0
        for name in [
            "mean_size",
            "std_size",
            "largest",
            "mean_radius_of_gyration",
            "std_radius_of_gyration",
            *RG_FEATURES,
        ]
    }
    if bonded_sizes.size:
        values["mean_size"] = float(np.mean(bonded_sizes))
        values["std_size"] = float(np.std(bonded_sizes))
        values["largest"] = float(np.max(bonded_sizes))
        # Each bonded cluster contributes once, independent of its particle count.
        values["mean_radius_of_gyration"] = float(np.mean(bonded_radii))
        values["std_radius_of_gyration"] = float(np.std(bonded_radii))
        for size, name in zip(RG_CLUSTER_SIZES, RG_FEATURES):
            matches = bonded_radii[bonded_sizes == size]
            values[name] = float(np.mean(matches)) if matches.size else 0.0
    return values, int(bonded_sizes.size)


def orientation_features(edges: np.ndarray, moments: np.ndarray) -> dict[str, float]:
    angles = []
    for first, second in edges:
        first_norm = np.linalg.norm(moments[first])
        second_norm = np.linalg.norm(moments[second])
        if first_norm == 0 or second_norm == 0:
            continue
        cosine = np.clip(np.dot(moments[first], moments[second]) / (first_norm * second_norm), -1.0, 1.0)
        angles.append(np.arccos(cosine))
    if not angles:
        return dict.fromkeys(ORIENTATION_FEATURES, 0.0)
    counts, _ = np.histogram(angles, bins=ORIENTATION_BIN_EDGES)
    fractions = counts / counts.sum()
    return dict(zip(ORIENTATION_FEATURES, fractions.astype(float)))


def rdf_features(box: freud.box.Box, positions: np.ndarray) -> dict[str, float]:
    rdf = freud.density.RDF(bins=RDF_BIN_COUNT, r_max=RDF_R_MAX)
    rdf.compute((box, positions))
    return dict(zip(RDF_FEATURES, rdf.rdf.astype(float)))


def psi_features(box: freud.box.Box, positions: np.ndarray, nlist: freud.locality.NeighborList) -> dict[str, float]:
    mask = nlist.distances < PSI_CUTOFF
    psi_nlist = filtered_neighbor_list(nlist, len(positions), mask)
    psi4 = freud.order.Hexatic(k=4)
    psi6 = freud.order.Hexatic(k=6)
    psi4.compute((box, positions), neighbors=psi_nlist)
    psi6.compute((box, positions), neighbors=psi_nlist)
    return {"mean_Psi_4": float(abs(np.mean(psi4.particle_order))), "mean_Psi_6": float(abs(np.mean(psi6.particle_order)))}


def crystalline_fraction(q_values: np.ndarray, coordination: np.ndarray, q_nlist: freud.locality.NeighborList, threshold: float) -> float:
    selected = np.isfinite(q_values) & (np.abs(q_values) >= threshold) & (coordination >= Q_MIN_NEIGHBORS)
    selected_indices = np.flatnonzero(selected)
    if not selected_indices.size:
        return 0.0
    local_index = np.full(len(q_values), -1, dtype=int)
    local_index[selected_indices] = np.arange(len(selected_indices))
    selected_edges = (selected[q_nlist.query_point_indices] & selected[q_nlist.point_indices])
    rows = local_index[q_nlist.query_point_indices[selected_edges]]
    cols = local_index[q_nlist.point_indices[selected_edges]]
    graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(selected_indices), len(selected_indices)))
    _, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    return float(np.count_nonzero(sizes[labels] >= Q_MIN_CLUSTER_SIZE) / len(q_values))


def q_features(box: freud.box.Box, positions: np.ndarray, q_nlist: freud.locality.NeighborList) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    coordination = np.bincount(q_nlist.query_point_indices, minlength=len(positions))
    steinhardt = freud.order.Steinhardt(l=[4, 6], average=True, wl=False)
    steinhardt.compute((box, positions), neighbors=q_nlist)
    q4, q6 = steinhardt.particle_order.T
    eligible = coordination >= Q_MIN_NEIGHBORS
    n_eligible = int(np.count_nonzero(eligible))
    values: dict[str, float] = {
        "n_q_eligible": float(n_eligible),
        "fraction_q_eligible": float(n_eligible / len(positions)),
        "p_q4": crystalline_fraction(q4, coordination, q_nlist, Q_THRESHOLD),
        "p_q6": crystalline_fraction(q6, coordination, q_nlist, Q_THRESHOLD),
    }
    histograms: dict[str, np.ndarray] = {}
    for order, q_values, names in ((4, q4, Q4_HISTOGRAM_FEATURES), (6, q6, Q6_HISTOGRAM_FEATURES)):
        filtered = q_values[eligible]
        if n_eligible == 0:
            values[f"mean_q{order}"] = np.nan
            fractions = np.zeros(len(names))
        else:
            if not np.isfinite(filtered).all():
                raise ValueError(f"Eligible q{order} values contain NaNs or infinities.")
            values[f"mean_q{order}"] = float(np.mean(filtered))
            counts, _ = np.histogram(np.abs(filtered), bins=Q_BIN_EDGES)
            fractions = counts / n_eligible
        values.update(dict(zip(names, fractions.astype(float))))
        histograms[f"q{order}_all_finite"] = q_values[np.isfinite(q_values)]
        histograms[f"q{order}_eligible"] = filtered[np.isfinite(filtered)]
    return values, {"coordination": coordination, **histograms}


def calculate_features(run: RunInfo, include_validation_data: bool = False) -> tuple[dict[str, object], dict[str, object] | None]:
    traj_frame, mu_frame = latest_common_frames(run.path / "traj.gz", run.path / "mu.gz")
    molecule_ids, positions, moments = align_positions_and_moments(traj_frame, mu_frame)
    box = make_box()
    query = freud.locality.AABBQuery(box, positions)
    nlist = query.query(positions, {"r_max": SHELL_CUTOFF_MAX, "exclude_ii": True}).toNeighborList()
    mask_15 = nlist.distances < BOND_CUTOFF_15
    mask_18 = nlist.distances < BOND_CUTOFF_18
    mask_shell = (nlist.distances >= BOND_CUTOFF_15) & (nlist.distances < SHELL_CUTOFF_MAX)
    degree_15 = degrees_from_mask(nlist, mask_15, len(positions))
    degree_18 = degrees_from_mask(nlist, mask_18, len(positions))
    shell_degree = degrees_from_mask(nlist, mask_shell, len(positions))
    q_nlist = filtered_neighbor_list(nlist, len(positions), mask_18)
    bonded_edges = unique_edges(nlist, mask_18)
    cluster_values, bonded_cluster_count = cluster_statistics(box, positions, q_nlist)

    result: dict[str, object] = {
        "file_id": str(run.path),
        "lambda": run.lambda_value,
        "shift": run.shift,
        "mean_bonds_1_8": float(np.mean(degree_18)),
        "std_bonds_1_8": float(np.std(degree_18)),
        "mean_bonds_1_5": float(np.mean(degree_15)),
        "std_bonds_1_5": float(np.std(degree_15)),
        # Compatibility name: radial-shell coordination count for 1.5 <= r < 2.0.
        "mean_second_neighbours": float(np.mean(shell_degree)),
        "std_second_neighbours": float(np.std(shell_degree)),
        **cluster_values,
        **orientation_features(bonded_edges, moments),
        **rdf_features(box, positions),
        **psi_features(box, positions, nlist),
    }
    q_values, q_validation = q_features(box, positions, q_nlist)
    result.update(q_values)
    missing = [column for column in OUTPUT_COLUMNS if column not in result]
    if missing:
        raise RuntimeError(f"Feature calculation omitted schema columns: {missing}")
    result = {column: result[column] for column in OUTPUT_COLUMNS}
    if not include_validation_data:
        return result, None
    validation = {
        "run": run,
        "timestep": traj_frame.timestep,
        "molecule_ids": molecule_ids,
        "degree_18": degree_18,
        "bonded_cluster_count": bonded_cluster_count,
        **q_validation,
    }
    return result, validation


def worker(run: RunInfo) -> dict[str, object] | None:
    try:
        return calculate_features(run)[0]
    except (EOFError, OSError, ValueError, RuntimeError, zlib.error) as error:
        logging.warning("Skipping %s: %s", run.path, error)
        return None


def validate_feature_frame(frame: pd.DataFrame) -> None:
    missing = [column for column in OUTPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Generated output is missing schema columns: {missing}")
    if not all(np.isscalar(value) and not isinstance(value, (str, bytes)) for value in frame[MODEL_FEATURES].to_numpy().ravel()):
        raise ValueError("Generated ML features contain non-scalar values.")
    for names in (Q4_HISTOGRAM_FEATURES, Q6_HISTOGRAM_FEATURES):
        sums = frame[names].sum(axis=1)
        eligible = frame["n_q_eligible"] > 0
        if not np.allclose(sums[eligible], 1.0, rtol=0, atol=1e-8) or not np.allclose(sums[~eligible], 0.0, rtol=0, atol=1e-12):
            raise ValueError("q histogram fractions do not have the expected eligible-particle normalization.")


def write_metadata(output_path: Path, selected_runs: list[RunInfo], worker_count: int) -> None:
    metadata = {
        "freud_version": freud.__version__,
        "box": {"Lx": BOX_LENGTH_XY, "Ly": BOX_LENGTH_XY, "is2D": True, "positions_z": 0.0},
        "cutoffs": {"bond_1_5": BOND_CUTOFF_15, "bond_1_8": BOND_CUTOFF_18, "second_neighbour_shell": "1.5 <= r < 2.0", "query_max": SHELL_CUTOFF_MAX},
        "radius_of_gyration": "Unweighted mean and population standard deviation over bonded clusters; each cluster contributes once.",
        "q_eligibility": "coordination within r_cut=1.8 is at least 4; q histograms are fractions of eligible particles.",
        "legacy_distribution_mapping": LEGACY_DISTRIBUTION_MAPPING,
        "feature_groups": FEATURE_GROUPS,
        "selected_directories": [str(run.path) for run in selected_runs],
        "workers": worker_count,
    }
    with output_path.with_suffix(".metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def plot_validation(run_directory: Path, result: dict[str, object], validation: dict[str, object]) -> dict[str, Path]:
    run_directory.mkdir(parents=True, exist_ok=True)
    degree = validation["degree_18"]
    categories = [0, 1, 2, 3, "≥4"]
    counts = [int(np.count_nonzero(degree == value)) for value in range(4)] + [int(np.count_nonzero(degree >= 4))]
    coordination_path = run_directory / "coordination.pdf"
    fig, axis = plt.subplots(figsize=(5, 3.5))
    axis.bar([str(category) for category in categories], counts)
    axis.set(xlabel="Coordination within r < 1.8", ylabel="Particle count", title="Coordination distribution")
    fig.tight_layout()
    fig.savefig(coordination_path)
    plt.close(fig)

    q_filter_path = run_directory / "q_filter_comparison.pdf"
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
    for axis, order in zip(axes, (4, 6)):
        before = validation[f"q{order}_all_finite"]
        after = validation[f"q{order}_eligible"]
        axis.hist(before, bins=Q_BIN_EDGES, histtype="step", density=True, label="finite q")
        axis.hist(after, bins=Q_BIN_EDGES, histtype="step", density=True, label="coordination ≥ 4")
        axis.set(xlabel=f"q{order}", title=f"q{order} coordination filter")
        axis.legend()
    axes[0].set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(q_filter_path)
    plt.close(fig)
    return {"coordination_plot": coordination_path, "q_filter_plot": q_filter_path}


def run_validation(selected_runs: list[RunInfo]) -> pd.DataFrame:
    validation_dir = SCRIPT_DIR / "results" / "feature_validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    selected_records = []
    feature_rows = []
    summary_rows = []
    for run in selected_runs:
        logging.info("Validating %s", run.path)
        result, validation = calculate_features(run, include_validation_data=True)
        assert validation is not None
        state_dir = validation_dir / f"shift_{run.shift:g}_lambda_{run.lambda_value:g}"
        plots = plot_validation(state_dir, result, validation)
        degree = validation["degree_18"]
        summary_rows.append(
            {
                "file_id": result["file_id"],
                "timestep": validation["timestep"],
                "n_particles": len(degree),
                "coordination_0": int(np.count_nonzero(degree == 0)),
                "coordination_1": int(np.count_nonzero(degree == 1)),
                "coordination_2": int(np.count_nonzero(degree == 2)),
                "coordination_3": int(np.count_nonzero(degree == 3)),
                "coordination_ge_4": int(np.count_nonzero(degree >= 4)),
                "bonded_cluster_count": validation["bonded_cluster_count"],
                "largest": result["largest"],
                "mean_bonds_1_5": result["mean_bonds_1_5"],
                "mean_bonds_1_8": result["mean_bonds_1_8"],
                "mean_second_neighbours": result["mean_second_neighbours"],
                "n_q_eligible": result["n_q_eligible"],
                "fraction_q_eligible": result["fraction_q_eligible"],
                "mean_q4": result["mean_q4"],
                "mean_q6": result["mean_q6"],
                "q4_histogram_sum": float(sum(result[name] for name in Q4_HISTOGRAM_FEATURES)),
                "q6_histogram_sum": float(sum(result[name] for name in Q6_HISTOGRAM_FEATURES)),
                **{name: str(path) for name, path in plots.items()},
            }
        )
        selected_records.append({"path": str(run.path), "shift": run.shift, "lambda": run.lambda_value, "timestep": validation["timestep"]})
        feature_rows.append(result)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(validation_dir / "validation_summary.csv", index=False)
    (validation_dir / "validation_summary.txt").write_text(summary.to_string(index=False) + "\n", encoding="utf-8")
    with (validation_dir / "selected_directories.json").open("w", encoding="utf-8") as handle:
        json.dump({"freud_version": freud.__version__, "planar_steinhardt": "validated with freud.order.Steinhardt and is2D box", "selected": selected_records}, handle, indent=2)
    features = pd.DataFrame(feature_rows).reindex(columns=OUTPUT_COLUMNS)
    validate_feature_frame(features)
    validation_output = validation_dir / "validation_features.pickle"
    features.to_pickle(validation_output)
    try:
        from ML_diffusion_map_testing import load_and_validate_data

        load_and_validate_data(validation_output, "both")
    except Exception as error:
        raise RuntimeError(f"Validation features are not compatible with ML_diffusion_map_testing.py: {error}") from error
    return summary


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    data_root = args.data_root.expanduser().resolve()
    output_path = repo_path(args.output)
    runs, failed_parsing = discover_runs(data_root, args.input_glob)
    selected_runs = select_runs(args, runs, failed_parsing)
    if not selected_runs:
        raise ValueError("No valid run directories selected.")
    workers = args.workers if args.workers is not None else (min(12, os.cpu_count() or 1) if args.full_run else 1)
    logging.info("Selected %d directories", len(selected_runs))
    for run in selected_runs:
        logging.info("Selected: %s", run.path)
    if args.validate_only:
        summary = run_validation(selected_runs)
        logging.info("Validation complete:\n%s", summary.to_string(index=False))
        return
    if args.full_run:
        logging.info("Full-run mode explicitly selected: candidates=%d valid=%d workers=%d output=%s", len(runs), len(selected_runs), workers, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if workers == 1:
        rows = [worker(run) for run in selected_runs]
    else:
        with mp.Pool(processes=workers) as pool:
            rows = pool.map(worker, selected_runs)
    rows = [row for row in rows if row is not None]
    if not rows:
        raise RuntimeError("No feature rows were generated successfully.")
    frame = pd.DataFrame(rows).reindex(columns=OUTPUT_COLUMNS)
    validate_feature_frame(frame)
    frame.to_pickle(output_path)
    write_metadata(output_path, selected_runs, workers)
    logging.info("Wrote %d feature rows to %s", len(frame), output_path)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        logging.error("%s", error)
        sys.exit(1)
