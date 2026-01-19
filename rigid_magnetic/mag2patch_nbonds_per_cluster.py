import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import glob
import gzip
import pandas as pd
from datetime import datetime
import numba
import cProfile
from scipy.spatial.distance import squareform
from pathlib import Path
import multiprocessing
import freud


def read_lammpstrj(t):
    Nskip = 9
    Natoms = 3000
    Nparticles = Natoms / 3
    lx_box = 270
    ly_box = 270
    lz_box = 3

    frames = []
    frame_nr_old = -1
    mfile = Path(t)
    if mfile.is_file():
        try:
            with gzip.open(t, "r") as traj_file:
                for i, line in enumerate(traj_file):
                    modulo = i % (Nskip + Natoms)
                    frame_nr = i // (Nskip + Natoms)
                    if frame_nr != frame_nr_old:
                        frames.append([])

                    if modulo >= Nskip:
                        whole_line = np.array(line.split()).astype(float)
                        if whole_line[1] == 1:
                            x = whole_line[2] * lx_box
                            y = whole_line[3] * ly_box
                            z = whole_line[4] * lz_box
                            frames[-1].append(np.array([x, y, z]))

                    frame_nr_old = frame_nr
        except (
            EOFError,
            IndexError,
            ValueError,
            gzip.BadGzipFile,
            gzip.zlib.error,
        ) as er:
            print("Caught error in {}:".format(t), er)

        if frames:
            if len(frames[-1]) != Nparticles:
                del frames[-1]

    frames = np.array(frames)
    return frames


def read_moments(mu):
    Nskip = 9
    Natoms = 3000
    Nparticles = Natoms / 3

    frames = []
    frame_nr_old = -1
    mfile = Path(mu)
    if mfile.is_file():
        try:
            with gzip.open(mu, "r") as traj_file:
                is_first = True
                eval = True
                for i, line in enumerate(traj_file):
                    if is_first == True:
                        eval = True

                    modulo = i % (Nskip + Natoms)
                    frame_nr = i // (Nskip + Natoms)
                    if frame_nr != frame_nr_old:
                        frames.append([])

                    if modulo >= Nskip:
                        whole_line = np.array(line.split()).astype(float)
                        is_first = True
                        if whole_line[2] == 2 and eval == True:
                            mx = whole_line[6]
                            my = whole_line[7]
                            mz = whole_line[8]
                            frames[-1].append(np.array([mx, my, mz]))
                            is_first = False
                            eval = False

                    frame_nr_old = frame_nr

        except (
            EOFError,
            IndexError,
            ValueError,
            gzip.BadGzipFile,
            gzip.zlib.error,
        ) as er:
            print("Caught error in {}:".format(mu), er)

        if frames:
            if len(frames[-1]) != Nparticles:
                del frames[-1]

    frames = np.array(frames)
    return frames


@numba.njit(fastmath=True, parallel=False)
def numba_distances(frame_i):
    lx_box = 270
    ly_box = 270
    lz_box = 3
    N_particles = 1000

    dist_norm = []
    for i, ipos in enumerate(frame_i):
        for j, jpos in enumerate(frame_i):
            if j > i:
                dist = ipos - jpos

                dx = dist[0]
                dy = dist[1]
                dz = dist[2]

                sign_dx = np.sign(dx)
                sign_dy = np.sign(dy)
                sign_dz = np.sign(dz)

                # pbc only for x and y
                dx = sign_dx * (min(np.fabs(dx), lx_box - np.fabs(dx)))
                dy = sign_dy * (min(np.fabs(dy), ly_box - np.fabs(dy)))

                dist_ij = np.sqrt(np.power(dx * dx + dy * dy + dz * dz, 2))
                dist_norm.append(dist_ij)

    return dist_norm


def calculate_neighbours(frame_i, cutoff):
    lx_box = 270
    ly_box = 270
    lz_box = 3

    neighbour_list = []

    for i, ipos in enumerate(frame_i):
        for j, jpos in enumerate(frame_i):
            if i < j:
                dist = ipos - jpos
                dx = dist[0]
                dy = dist[1]
                dz = dist[2]

                sign_dx = np.sign(dx)
                sign_dy = np.sign(dy)
                sign_dz = np.sign(dz)

                # pbc only for x and y
                dx = sign_dx * (min(np.fabs(dx), lx_box - np.fabs(dx)))
                dy = sign_dy * (min(np.fabs(dy), ly_box - np.fabs(dy)))
                dz = dz

                dist_norm = np.sqrt(dx * dx + dy * dy + dz * dz)
                if dist_norm < cutoff:
                    neighbour_list.append([i, j])

    return neighbour_list


def calculate_neighbours_fast(sq_dist, cutoff):
    b = np.where((sq_dist < cutoff) & (sq_dist > 0.01))
    neighbour_list = [[b[0][i], b[1][i]] for i in range(len(b[0]))]
    return neighbour_list


################# Per particle analysis #####################


def generate_graph(cutoff, dist_squareform):
    neighbour_list = calculate_neighbours_fast(dist_squareform, cutoff)
    G = nx.Graph()
    G.add_edges_from(neighbour_list)
    return G


def calculate_degree_per_cluster(G, Nparticles):
    avg_degrees = {}
    for cluster_id, component in enumerate(nx.connected_components(G)):
        subgraph = G.subgraph(component)
        avg_degree = sum(dict(subgraph.degree()).values()) / len(component)
        avg_degrees[cluster_id] = avg_degree
    return avg_degrees


def radius_of_gyration_per_cluster(G, sq_dist):
    radius_of_gyration = {}
    for cluster_id, component in enumerate(nx.connected_components(G)):
        cluster = list(component)
        rg = 0
        N_cluster = len(cluster)
        cluster_sorted = sorted(cluster)
        for ic in cluster_sorted:
            for jc in cluster:
                if ic < jc:
                    rg += sq_dist[ic, jc]

        rg = (1 / (2 * N_cluster * N_cluster)) * rg
        radius_of_gyration[cluster_id] = rg
    return radius_of_gyration


def calculate_size_per_cluster(G):
    cluster_sizes = {}
    for cluster_id, component in enumerate(nx.connected_components(G)):
        cluster_sizes[cluster_id] = len(component)
    return cluster_sizes


def comprehensive_structure_classification(
    G,
    clustering_threshold=0.5,
    liquid_threshold=0.9,
    Nparticles=1000,
):
    """
    Comprehensive classification combining strict ring/chain detection
    with branching analysis and clustering.

    Ring: n edges = n nodes AND all particles have exactly degree 2
    Chain: All particles have degree 1 or 2, at least 2 with degree 1, no cycles
    Liquid: Most particles are singlets (degree 0, unbonded)
    Strongly Clustered: Average clustering coefficient > threshold (high local density)
    Branch: Contains branch points (degree >= 3)
    Tree: No cycles but has branching
    Complex Network: Has cycles and branching
    """
    classification = {}

    for cluster_id, component in enumerate(nx.connected_components(G)):
        subgraph = G.subgraph(component)
        cluster_size = len(component)

        # It is not a liquid as there bonds
        is_liquid = False

        # Get degree distribution
        degrees = dict(subgraph.degree())
        degree_list = list(degrees.values())

        # Basic metrics
        edges = subgraph.number_of_edges()
        nodes = subgraph.number_of_nodes()
        avg_degree = sum(degree_list) / len(degree_list) if degree_list else 0

        # Cyclomatic complexity
        cyclomatic_complexity = edges - nodes + 1

        # Count particles by degree
        degree_0_count = 0
        degree_1_count = sum(1 for d in degree_list if d == 1)
        degree_2_count = sum(1 for d in degree_list if d == 2)
        degree_3_plus_count = sum(1 for d in degree_list if d >= 3)

        # Clustering coefficient
        clustering_coeff = nx.average_clustering(subgraph)

        # STRICT RING: n edges = n nodes AND all degree 2
        is_strict_ring = (
            edges == nodes
            and degree_1_count == 0
            and degree_2_count == cluster_size
            and degree_3_plus_count == 0
        )

        # STRICT CHAIN: all degree 1 or 2, at least 2 with degree 1, no cycles
        is_strict_chain = (
            degree_3_plus_count == 0
            and degree_1_count >= 2
            and cyclomatic_complexity == 0
            and all(d in [1, 2] for d in degree_list)
        )

        # STRONGLY CLUSTERED: high average clustering coefficient
        is_strongly_clustered = clustering_coeff > clustering_threshold

        # BRANCHING metrics
        branch_points = degree_3_plus_count
        is_branched = branch_points > 0
        is_tree = cyclomatic_complexity == 0 and is_branched
        is_complex_network = cyclomatic_complexity > 0 and is_branched

        # Determine structure type (priority order)

        if is_strict_ring:
            structure_type = "ring"
        elif is_strict_chain:
            structure_type = "chain"
        elif is_strongly_clustered:
            structure_type = "strongly_clustered"
        elif is_complex_network:
            structure_type = "complex_network"
        elif is_tree:
            structure_type = "tree"
        elif is_branched:
            structure_type = "branched"
        else:
            structure_type = "other"

        classification[cluster_id] = {
            "cluster_size": cluster_size,
            "structure_type": structure_type,
            "is_liquid": is_liquid,
            "is_strict_ring": is_strict_ring,
            "is_strict_chain": is_strict_chain,
            "is_strongly_clustered": is_strongly_clustered,
            "is_tree": is_tree,
            "is_complex_network": is_complex_network,
            "is_branched": is_branched,
            "avg_degree": avg_degree,
            "avg_clustering_coefficient": clustering_coeff,
            "branch_points": branch_points,
            "degree_0_count": degree_0_count,
            "degree_1_count": degree_1_count,
            "degree_2_count": degree_2_count,
            "degree_3_plus_count": degree_3_plus_count,
            "cyclomatic_complexity": cyclomatic_complexity,
        }

    return classification


def process_files(idir):
    Nparticles = 1000

    frames = read_lammpstrj("{}traj.gz".format(idir))
    frames_mu = read_moments("{}mu.gz".format(idir))

    if frames.size > 0 and frames_mu.size > 0:
        dist = numba_distances(frames[-1])
        dist_squareform = squareform(dist)

        cutoff = 1.8
        G = generate_graph(cutoff, dist_squareform)
        avg_degrees = calculate_degree_per_cluster(G, Nparticles)
        # radius_of_gyration_per_cluster_dict = radius_of_gyration_per_cluster(
        #    G, dist_squareform
        # )
        cluster_sizes = calculate_size_per_cluster(G)
        classification = comprehensive_structure_classification(G)

        # Create one row per cluster
        results_list = []
        for cluster_id in avg_degrees.keys():
            new_results = {}
            new_results["file_id"] = idir
            new_results["lambda"] = float(idir.split("_")[4])
            new_results["shift"] = float(idir.split("_")[2])
            new_results["cluster_id"] = cluster_id

            # Per-cluster metrics
            new_results["cluster_size"] = cluster_sizes[cluster_id]
            new_results["avg_degree"] = avg_degrees[cluster_id]
            # new_results["radius_of_gyration"] = radius_of_gyration_per_cluster_dict[
            #    cluster_id
            # ]

            # Structure classification metrics
            class_info = classification[cluster_id]
            new_results["structure_type"] = class_info["structure_type"]
            new_results["is_liquid"] = class_info["is_liquid"]
            new_results["is_strict_ring"] = class_info["is_strict_ring"]
            new_results["is_strict_chain"] = class_info["is_strict_chain"]
            new_results["is_strongly_clustered"] = class_info["is_strongly_clustered"]
            new_results["is_tree"] = class_info["is_tree"]
            new_results["is_complex_network"] = class_info["is_complex_network"]
            new_results["is_branched"] = class_info["is_branched"]
            new_results["avg_clustering_coefficient"] = class_info[
                "avg_clustering_coefficient"
            ]
            new_results["branch_points"] = class_info["branch_points"]
            new_results["degree_0_count"] = class_info["degree_0_count"]
            new_results["degree_1_count"] = class_info["degree_1_count"]
            new_results["degree_2_count"] = class_info["degree_2_count"]
            new_results["degree_3_plus_count"] = class_info["degree_3_plus_count"]
            new_results["cyclomatic_complexity"] = class_info["cyclomatic_complexity"]

            results_list.append(new_results)

        # Add artificial singleton rows for unbonded particles (label as liquid)
        # Nodes present in G are those with at least one bond; remaining particles
        # are unbonded singlets and should be represented as cluster_size=1
        nodes_in_graph = G.number_of_nodes()
        num_singletons = max(0, Nparticles - nodes_in_graph)

        # Determine starting cluster_id for singletons
        if avg_degrees:
            next_cluster_id = max(avg_degrees.keys()) + 1
        else:
            next_cluster_id = 0

        for i in range(num_singletons):
            singleton_id = next_cluster_id + i
            sres = {}
            sres["file_id"] = idir
            sres["lambda"] = float(idir.split("_")[4])
            sres["shift"] = float(idir.split("_")[2])
            sres["cluster_id"] = singleton_id
            sres["cluster_size"] = 1
            sres["avg_degree"] = 0.0
            sres["structure_type"] = "liquid"
            sres["is_liquid"] = True
            sres["is_strict_ring"] = False
            sres["is_strict_chain"] = False
            sres["is_strongly_clustered"] = False
            sres["is_tree"] = False
            sres["is_complex_network"] = False
            sres["is_branched"] = False
            sres["avg_clustering_coefficient"] = 0.0
            sres["branch_points"] = 0
            sres["degree_0_count"] = 1
            sres["degree_1_count"] = 0
            sres["degree_2_count"] = 0
            sres["degree_3_plus_count"] = 0
            sres["cyclomatic_complexity"] = 0
            results_list.append(sres)

        new_results_df = pd.DataFrame(results_list)

        return new_results_df
    else:
        print("Problem with folder {}. Results not evaluated".format(idir))


if __name__ == "__main__":
    df = pd.DataFrame()
    dirs = glob.glob("mag2p_shift*/")

    with multiprocessing.Pool(processes=12) as pool:
        new_results = pool.map(process_files, dirs)
        pool.close()
        pool.join()
        df = pd.concat(new_results, ignore_index=True)

    currentDateAndTime = datetime.now()
    df.to_pickle(
        "MAG2P_order_parameters_per_cluster-{}-{}-{}-{}:{}:{}.pickle".format(
            currentDateAndTime.year,
            currentDateAndTime.month,
            currentDateAndTime.day,
            currentDateAndTime.hour,
            currentDateAndTime.minute,
            currentDateAndTime.second,
        )
    )
