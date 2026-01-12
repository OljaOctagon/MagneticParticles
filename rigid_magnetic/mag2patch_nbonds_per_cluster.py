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
        except (EOFError, IndexError, ValueError) as er:
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

        except (EOFError, IndexError, ValueError) as er:
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
                sign_dy = np.sign(dz)

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
                sign_dy = np.sign(dz)

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


##################################################################3


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
        radius_of_gyration_per_cluster_dict = radius_of_gyration_per_cluster(
            G, dist_squareform
        )
        cluster_sizes = calculate_size_per_cluster(G)

        # Create one row per cluster
        results_list = []
        for cluster_id in avg_degrees.keys():
            new_results = {}
            new_results["file_id"] = idir
            new_results["lambda"] = float(idir.split("_")[4])
            new_results["shift"] = float(idir.split("_")[2])
            new_results["cluster_id"] = cluster_id
            new_results["avg_degree"] = avg_degrees[cluster_id]
            new_results["radius_of_gyration"] = radius_of_gyration_per_cluster_dict[
                cluster_id
            ]
            new_results["cluster_size"] = cluster_sizes[cluster_id]

            results_list.append(new_results)

        new_results_df = pd.DataFrame(results_list)

        return new_results_df

    else:
        print("Problem with folder {}. Results not evaluated".format(idir))


if __name__ == "__main__":
    df = pd.DataFrame()
    dirs = glob.glob("mag2p_shift*/")

    with multiprocessing.Pool(processes=8) as pool:
        new_results = pool.map(process_files, dirs)
        pool.close()
        pool.join()
        df = pd.concat(new_results, ignore_index=True)

    currentDateAndTime = datetime.now()
    df.to_pickle(
        "MAG2P_order_parameters-{}-{}-{}-{}:{}:{}.pickle".format(
            currentDateAndTime.year,
            currentDateAndTime.month,
            currentDateAndTime.day,
            currentDateAndTime.hour,
            currentDateAndTime.minute,
            currentDateAndTime.second,
        )
    )
