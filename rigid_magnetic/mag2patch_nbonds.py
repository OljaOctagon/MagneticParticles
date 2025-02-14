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

def read_lammpstrj(t):
    Nskip = 9
    Natoms=3000 
    Nparticles = Natoms/3 
    lx_box = 270 
    ly_box = 270
    lz_box = 3

    frames = []
    frame_nr_old = -1 
    with gzip.open(t, "r") as traj_file:
        try: 
            for i,line in enumerate(traj_file):
                modulo = i % (Nskip+Natoms)
                frame_nr = i // (Nskip+Natoms)
                if frame_nr != frame_nr_old:
                    frames.append([])

                if modulo >=Nskip:
                    whole_line = np.array(line.split()).astype(float)
                    if whole_line[1] == 1:
                        x = whole_line[2]*lx_box
                        y = whole_line[3]*ly_box
                        z = whole_line[4]*lz_box 
                        frames[-1].append(np.array([x,y,z])) 

                frame_nr_old = frame_nr
        except EOFError as er:
            print(er)
    
    if len(frames[-1])!=Nparticles:
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
            if j>i: 
                dist = ipos - jpos
                
                dx = dist[0]
                dy = dist[1]
                dz = dist[2]
                
                sign_dx = np.sign(dx)
                sign_dy = np.sign(dy)
                sign_dy = np.sign(dz)
                
                # pbc only for x and y 
                dx = sign_dx*(min(np.fabs(dx),lx_box-np.fabs(dx)))
                dy = sign_dy*(min(np.fabs(dy),ly_box-np.fabs(dy)))
                
                dist_ij = np.sqrt(np.power(dx*dx+dy*dy+dz*dz,2))
                dist_norm.append(dist_ij)
                
    return dist_norm
    

def calculate_neighbours(frame_i,cutoff):

    lx_box = 270 
    ly_box = 270
    lz_box = 3

    neighbour_list = []

    for i, ipos in enumerate(frame_i):
        for j, jpos in enumerate(frame_i):
            if i<j:
                
                dist = ipos - jpos
                dx = dist[0]
                dy = dist[1]
                dz = dist[2]
                
                sign_dx = np.sign(dx)
                sign_dy = np.sign(dy)
                sign_dy = np.sign(dz)

                # pbc only for x and y 
                dx = sign_dx*(min(np.fabs(dx),lx_box-np.fabs(dx)))
                dy = sign_dy*(min(np.fabs(dy),ly_box-np.fabs(dy)))
                dz = dz 

                dist_norm = np.sqrt(np.power(dx*dx+dy*dy+dz*dz,2))
                if dist_norm < cutoff: 
                    neighbour_list.append([i,j])


    return neighbour_list 


files = glob.glob("mag2p_shift*/traj.gz")
df = pd.DataFrame()
Rg_df = pd.DataFrame()
cutoff=1.3

def calculate_neighbours_fast(sq_dist, cutoff):
    b = np.where((sq_dist<cutoff) & (sq_dist>0.01))
    neighbour_list = [[b[0][i],b[1][i]] for i in range(len(b[0]))]
    return neighbour_list

# TODO rewrite in jit
def radius_of_gyration(sq_dist,clusters):

        Rg_result_dict = dict(zip(np.arange(31), np.zeros(31)))
        Rg=[] 
        for cluster in clusters:
            rg=0 
            N_cluster = len(cluster) 
            cluster_sorted = sorted(cluster)
            if(N_cluster<31):
                for ic in cluster_sorted:
                    for jc in cluster:
                        if ic<jc: 
                            rg+=sq_dist[ic,jc]

                rg = (1/(2*N_cluster*N_cluster))*rg
                Rg_result_dict[N_cluster] = rg 
                Rg.append(rg)
            
        mean_Rg = np.mean(np.array(Rg))
        std_Rg = np.std(np.array(Rg))

        return  Rg_result_dict, mean_Rg, std_Rg 
    


for file in files:
    print(file)
    Nparticles = 1000 
    frames = read_lammpstrj(file)

    dist = numba_distances(frames[-1])
    dist_squareform = squareform(dist)
    neighbour_list= calculate_neighbours_fast(dist_squareform,cutoff)
    
    G = nx.Graph() 
    G.add_edges_from(neighbour_list)

    degree = np.array([tuple[1] for tuple in G.degree()])
    number_of_bonded_particles = G.number_of_nodes()
    number_of_unbonded_particles = Nparticles - number_of_bonded_particles
    full_degree = np.append(degree,np.zeros(number_of_unbonded_particles))   
    mean_degree = np.mean(full_degree)
    std_degree = np.std(full_degree)
   
    clusters = [ list(cluster) for cluster in list(nx.connected_components(G))]
    # average/std cluster size 
    cluster_sizes = np.array([ len(c) for c in clusters ])
    mean_cluster_size = np.mean(cluster_sizes)
    std_cluster_size = np.std(cluster_sizes)
    # largest cluster size 
    largest_cc = 0 
    if clusters:
        largest_cc = np.max(cluster_sizes)
    
    # radius of gyration     
    Rg_result_dict, mean_Rg, std_Rg = radius_of_gyration(dist_squareform,clusters)

    print("Radius of gyration", mean_Rg, std_Rg)
    new_results = {}
    new_results["file_id"] = file.split("/")[0]
    new_results["lambda"] = float(file.split("_")[4])
    new_results["shift"] = float(file.split("_")[2])
    new_results["mean_bonds"] = mean_degree
    new_results["std_bonds"] = std_degree 
    new_results["mean_size"] = mean_cluster_size
    new_results["std_size"] = std_cluster_size
    new_results["largest"] = largest_cc 
    new_results["mean_radius_of_gyration"] = mean_Rg
    new_results["std_radius_of_gyration"] = std_Rg 

    new_results = pd.DataFrame.from_dict(new_results, orient="index").T
    df = pd.concat([df, new_results], ignore_index=True)

    Rg_result_dict["file_id"] = file.split("/")[0]
    Rg_result_dict["lambda"] = float(file.split("_")[4])
    Rg_result_dict["shift"] = float(file.split("_")[2])
    Rg_results = pd.DataFrame.from_dict(Rg_result_dict, orient="index").T
    Rg_df = pd.concat([Rg_df,Rg_results])


currentDateAndTime = datetime.now()
df.to_pickle(
        "MAG2P_nbonded-{}-{}-{}-{}:{}:{}.pickle".format(
            currentDateAndTime.year,
            currentDateAndTime.month,
            currentDateAndTime.day,
            currentDateAndTime.hour,
            currentDateAndTime.minute,
            currentDateAndTime.second,
        )
    )

Rg_df.to_pickle(
        "MAG2P_radius_of_gyration-{}-{}-{}-{}:{}:{}.pickle".format(
            currentDateAndTime.year,
            currentDateAndTime.month,
            currentDateAndTime.day,
            currentDateAndTime.hour,
            currentDateAndTime.minute,
            currentDateAndTime.second,
        )
    )

