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
    Natoms=3000 
    Nparticles = Natoms/3 
    lx_box = 270 
    ly_box = 270
    lz_box = 3

    frames = []
    frame_nr_old = -1 
    mfile = Path(t)
    if mfile.is_file():
        try:
            with gzip.open(t, "r") as traj_file:
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
        except (EOFError, IndexError, ValueError) as er:
            print("Caught error in {}:".format(t), er) 
       
        if frames:
            if len(frames[-1])!=Nparticles:
                del frames[-1]

    frames = np.array(frames)
    return frames 


def read_moments(mu):
    Nskip = 9
    Natoms=3000 
    Nparticles = Natoms/3 
    
    frames = []
    frame_nr_old = -1 
    mfile = Path(mu)
    if mfile.is_file():
        try: 
            with gzip.open(mu, "r") as traj_file:
            
                is_first=True 
                eval=True
                for i,line in enumerate(traj_file):

                    if is_first == True:
                         eval = True 

                    modulo = i % (Nskip+Natoms)
                    frame_nr = i // (Nskip+Natoms)
                    if frame_nr != frame_nr_old:
                        frames.append([])

                    if modulo >=Nskip:
                        whole_line = np.array(line.split()).astype(float)
                        is_first = True 
                        if whole_line[2] == 2 and eval==True:
                            mx = whole_line[6]
                            my = whole_line[7]
                            mz = whole_line[8] 
                            frames[-1].append(np.array([mx,my,mz])) 
                            is_first = False 
                            eval=False 
                        

                    frame_nr_old = frame_nr
       
        except (EOFError, IndexError, ValueError) as er:
            print("Caught error in {}:".format(mu), er) 

        if frames:
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

                dist_norm = np.sqrt(dx*dx+dy*dy+dz*dz)
                if dist_norm < cutoff: 
                    neighbour_list.append([i,j])


    return neighbour_list 


def calculate_neighbours_fast(sq_dist, cutoff):
    b = np.where((sq_dist<cutoff) & (sq_dist>0.01))
    neighbour_list = [[b[0][i],b[1][i]] for i in range(len(b[0]))]
    return neighbour_list

def calculate_second_neighbours_fast(sq_dist,cutoff_1,cutoff_2):
    b = np.where((sq_dist>cutoff_1) & (sq_dist<cutoff_2))
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
    
def mu_orientation_distribution(neighbour_list,moment_orientation):
    theta=[]
    for i,j in neighbour_list:
        m1 = moment_orientation[i]
        m2 = moment_orientation[j]
        scalar_product = np.dot(m1,m2)/(np.linalg.norm(m1)*np.linalg.norm(m2))
        theta.append(np.arccos(scalar_product))

    theta=np.array(theta)
    hist, bin_edges = np.histogram(theta,bins=50, range=(0,2*np.pi), density=True)
    moments_dict = dict(zip(bin_edges[:-1], hist))

    return moments_dict

def calculate_degree(cutoff,dist_squareform,Nparticles):
    neighbour_list= calculate_neighbours_fast(dist_squareform,cutoff)
    G = nx.Graph()
    G.add_edges_from(neighbour_list)

    degree = np.array([tuple[1] for tuple in G.degree()])
    number_of_bonded_particles = G.number_of_nodes()
    number_of_unbonded_particles = Nparticles - number_of_bonded_particles
    full_degree = np.append(degree,np.zeros(number_of_unbonded_particles))   
    mean_degree = np.mean(full_degree)
    std_degree = np.std(full_degree)
    
    return mean_degree, std_degree, neighbour_list, G


def calculate_rdf(frame_i,bins,r_max):
    lx_box = 270 
    ly_box = 270
    lz_box = 3

    # Quasi 2D box pos 
    box = freud.box.Box(Lx=lx_box, Ly=ly_box, Lz=lz_box, periodic=[True, True, False])

    # Compute g(r)
    rdf = freud.density.RDF(bins=bins, r_max=r_max)
    rdf.compute(system=(box, frame_i))

    rdf_dict = dict(zip(rdf.bin_centers, rdf.rdf))

    return rdf_dict
    

def process_files(idir):
    Nparticles = 1000 
    
    frames = read_lammpstrj("{}traj.gz".format(idir))
    frames_mu = read_moments("{}mu.gz".format(idir))

    if frames.size>0 and frames_mu.size >0:  

        dist = numba_distances(frames[-1])
        dist_squareform = squareform(dist)
        
        # first neighbour cutoffs
        cutoff_1_5 = 1.5 
        mean_degree_1_5, std_degree_1_5, _, _  = calculate_degree(cutoff_1_5,dist_squareform, Nparticles)
        cutoff_1_8 = 1.8
        mean_degree_1_8, std_degree_1_8, neighbour_list_1_8, G = calculate_degree(cutoff_1_8,dist_squareform, Nparticles)

        # second neighbour cutoff 
        cutoff_2 = 2.0 
        second_neighbour_list = calculate_second_neighbours_fast(dist_squareform,cutoff_1_5,cutoff_2)
        G2 = nx.Graph() 
        G2.add_edges_from(second_neighbour_list)
        degree_2 = np.array([tuple[1] for tuple in G2.degree()])
        number_of_second_neighbours = G2.number_of_nodes()
        mean_degree_2 = np.mean(degree_2)
        std_degree_2 = np.std(degree_2)

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
        Moments_dict = mu_orientation_distribution(neighbour_list_1_8, frames_mu[-1])

        # rdf 
        rdf_dict = calculate_rdf(frames[-1],30,6)
       
        new_results = {}
        new_results["file_id"] = idir
        new_results["lambda"] = float(idir.split("_")[4])
        new_results["shift"] = float(idir.split("_")[2])

        
        new_results["mean_bonds_1_8"] = mean_degree_1_8
        new_results["std_bonds_1_8"] = std_degree_1_8
        new_results["mean_bonds_1_5"] = mean_degree_1_5
        new_results["std_bonds_1_5"] = std_degree_1_5

        new_results["mean_second_neighbours"] = mean_degree_2
        new_results["std_second_neighbours"] = std_degree_2 
        new_results["mean_size"] = mean_cluster_size
        new_results["std_size"] = std_cluster_size
        new_results["largest"] = largest_cc 
        new_results["mean_radius_of_gyration"] = mean_Rg
        new_results["std_radius_of_gyration"] = std_Rg 
        
        new_results = new_results | Moments_dict 
        new_results = new_results | Rg_result_dict 
        new_results = new_results | rdf_dict

        new_results = pd.DataFrame.from_dict(new_results, orient="index").T
         
        return new_results 
    
    else: 
        print("Problem with folder {}. Results not evaluated".format(idir))


if __name__ == "__main__":

    df = pd.DataFrame()
    dirs = glob.glob("mag2p_shift*/")
   
    with multiprocessing.Pool(processes=8) as pool:
        new_results = pool.map(process_files,dirs)
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

                                                                                      