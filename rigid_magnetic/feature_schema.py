"""Shared feature names and bin definitions for magnetic-particle analysis."""

from __future__ import annotations

import numpy as np


META_COLUMNS = ["file_id", "lambda", "shift"]

GLOBAL_FEATURES = [
    "mean_bonds_1_8",
    "mean_bonds_1_5",
    "mean_second_neighbours",
    "mean_size",
    "largest",
    "mean_radius_of_gyration",
]
GLOBAL_STD_FEATURES = [
    "std_bonds_1_8",
    "std_bonds_1_5",
    "std_second_neighbours",
    "std_size",
    "std_radius_of_gyration",
]

ORIENTATION_BIN_EDGES = np.linspace(0.0, np.pi, 25)
ORIENTATION_FEATURES = [f"orientation_{index:02d}" for index in range(24)]

RG_CLUSTER_SIZES = tuple(range(2, 31))
RG_FEATURES = [f"rg_size_{size:02d}" for size in RG_CLUSTER_SIZES]

RDF_BIN_COUNT = 25
RDF_R_MAX = 6.0
RDF_BIN_EDGES = np.linspace(0.0, RDF_R_MAX, RDF_BIN_COUNT + 1)
RDF_FEATURES = [f"rdf_{index:02d}" for index in range(RDF_BIN_COUNT)]

Q_BIN_EDGES = np.linspace(0.0, 1.0, 11)
Q4_HISTOGRAM_FEATURES = [f"q4_hist_{start:02d}_{start + 10:02d}" for start in range(0, 100, 10)]
Q6_HISTOGRAM_FEATURES = [f"q6_hist_{start:02d}_{start + 10:02d}" for start in range(0, 100, 10)]
CRYSTALLINITY_SCALARS = ["p_q4", "p_q6"]
CRYSTALLINITY_COARSE_HISTOGRAM_FEATURES = Q4_HISTOGRAM_FEATURES + Q6_HISTOGRAM_FEATURES

Q_DIAGNOSTIC_FEATURES = ["mean_q4", "mean_q6", "n_q_eligible", "fraction_q_eligible"]
PSI_DIAGNOSTIC_FEATURES = ["mean_Psi_4", "mean_Psi_6"]
DIAGNOSTIC_FEATURES = Q_DIAGNOSTIC_FEATURES + PSI_DIAGNOSTIC_FEATURES

FEATURE_GROUPS = {
    "global": GLOBAL_FEATURES,
    "orientation": ORIENTATION_FEATURES,
    "Rg": RG_FEATURES,
    "gofr": RDF_FEATURES,
    "crystallinity_scalar_features": CRYSTALLINITY_SCALARS,
    "crystallinity_coarse_histogram_features": CRYSTALLINITY_COARSE_HISTOGRAM_FEATURES,
}

MODEL_FEATURES = [
    *GLOBAL_FEATURES,
    *ORIENTATION_FEATURES,
    *RG_FEATURES,
    *RDF_FEATURES,
    *CRYSTALLINITY_SCALARS,
    *CRYSTALLINITY_COARSE_HISTOGRAM_FEATURES,
]

OUTPUT_COLUMNS = [
    *META_COLUMNS,
    *GLOBAL_FEATURES,
    *GLOBAL_STD_FEATURES,
    *DIAGNOSTIC_FEATURES,
    *ORIENTATION_FEATURES,
    *RG_FEATURES,
    *RDF_FEATURES,
    *CRYSTALLINITY_SCALARS,
    *CRYSTALLINITY_COARSE_HISTOGRAM_FEATURES,
]

# Historical distribution columns were raw numeric bin keys. The revised generator
# uses these stable semantic names; global and scalar crystallinity names are unchanged.
LEGACY_DISTRIBUTION_MAPPING = {
    "orientation": "raw angle-bin centers -> orientation_00...orientation_23",
    "Rg": "raw cluster-size integers -> rg_size_02...rg_size_30",
    "gofr": "raw RDF bin centers -> rdf_00...rdf_24",
}
