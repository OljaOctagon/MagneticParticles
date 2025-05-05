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
        except EOFError as er:
            print(er)
        
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

        except EOFError as er:
                print(er)
        
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


def process_files(pfile,mfile):
    Nparticles = 1000 
    cutoff = 1.3 
    frames = read_lammpstrj(pfile)
    frames_mu = read_moments(mfile)

    if frames.size>0 and frames_mu.size >0:  

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

        cutoff_2 = 2.0 
        second_neighbour_list = calculate_second_neighbours_fast(dist_squareform,cutoff,cutoff_2)
        G2 = nx.Graph() 
        G2.add_edges_from(second_neighbour_list)
        degree2 = np.array([tuple[1] for tuple in G2.degree()])
        number_of_second_neighbours = G2.number_of_nodes()
        mean_degree2 = np.mean(degree2)
        std_degree2 = np.std(degree2)

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
        #Rg_result_dict, mean_Rg, std_Rg = radius_of_gyration(dist_squareform,clusters)

        Moments_dict = mu_orientation_distribution(neighbour_list, frames_mu[-1])
        
        new_results = {}
        new_results["file_id"] = file.split("/")[0]
        new_results["lambda"] = float(file.split("_")[4])
        new_results["shift"] = float(file.split("_")[2])
        new_results["mean_bonds"] = mean_degree
        new_results["std_bonds"] = std_degree 
        new_results["mean_second_neighbours"] = mean_degree2
        new_results["std_second_neighbours"] = std_degree2 
        new_results["mean_size"] = mean_cluster_size
        new_results["std_size"] = std_cluster_size
        new_results["largest"] = largest_cc 
        new_results["mean_radius_of_gyration"] = mean_Rg
        new_results["std_radius_of_gyration"] = std_Rg 
        
        new_results = new_results | Moments_dict 

        #new_results = pd.DataFrame.from_dict(new_results, orient="index").T
        
        #df = pd.concat([df, new_results], ignore_index=True)

        #Rg_result_dict["file_id"] = file.split("/")[0]
        #Rg_result_dict["lambda"] = float(file.split("_")[4])
        #Rg_result_dict["shift"] = float(file.split("_")[2])
        #Rg_results = pd.DataFrame.from_dict(Rg_result_dict, orient="index").T
        #Rg_df = pd.concat([Rg_df,Rg_results])

        #Moments_dict["file_id"] = file.split("/")[0]
        #Moments_dict["lambda"] = float(file.split("_")[4])
        #Moments_dict["shift"] = float(file.split("_")[2])
        #Moments_results = pd.DataFrame.from_dict(Moments_dict, orient="index").T
        #Moments_df = pd.concat([Moments_df,Moments_results])
            
        return new_results 
    
    else: 
        print("Problem with folder {}. Results not evaluated".format(pfile))


if __name__ == "__main__":

    df = pd.DataFrame()
    pfiles = glob.glob("mag2p_shift*/traj.gz")
    mfiles = glob.glob("mag2p_shift*/mu.gz")
    pfiles.sort()
    mfiles.sort()
   
    with multiprocessing.Pool(processes=8) as pool:
        new_results = pool.map(process_files,pfiles, mfiles)
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

                                                                                      

############################## old 

'''
for file, mfile  in zip(files,mfiles):
    print(file)
    Nparticles = 1000 
    frames = read_lammpstrj(file)
    frames_mu = read_moments(mfile)

    if frames.size>0 and frames_mu.size >0:  

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

        cutoff_2 = 2.0 
        second_neighbour_list = calculate_second_neighbours_fast(dist_squareform,cutoff,cutoff_2)
        G2 = nx.Graph() 
        G2.add_edges_from(second_neighbour_list)
        degree2 = np.array([tuple[1] for tuple in G2.degree()])
        number_of_second_neighbours = G2.number_of_nodes()
        mean_degree2 = np.mean(degree2)
        std_degree2 = np.std(degree2)
    
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

        Moments_dict = mu_orientation_distribution(neighbour_list, frames_mu[-1])
        

        print("Radius of gyration", mean_Rg, std_Rg)
        new_results = {}
        new_results["file_id"] = file.split("/")[0]
        new_results["lambda"] = float(file.split("_")[4])
        new_results["shift"] = float(file.split("_")[2])
        new_results["mean_bonds"] = mean_degree
        new_results["std_bonds"] = std_degree 
        new_results["mean_second_neighbours"] = mean_degree2
        new_results["std_second_neighbours"] = std_degree2 
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

        Moments_dict["file_id"] = file.split("/")[0]
        Moments_dict["lambda"] = float(file.split("_")[4])
        Moments_dict["shift"] = float(file.split("_")[2])
        Moments_results = pd.DataFrame.from_dict(Moments_dict, orient="index").T
        Moments_df = pd.concat([Moments_df,Moments_results])
        


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

Moments_df.to_pickle(
        "MAG2P_moments-{}-{}-{}-{}:{}:{}.pickle".format(
            currentDateAndTime.year,
            currentDateAndTime.month,
            currentDateAndTime.day,
            currentDateAndTime.hour,
            currentDateAndTime.minute,
            currentDateAndTime.second,
        )
    )
'''