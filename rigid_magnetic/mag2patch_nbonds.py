import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt 
import glob 
import gzip 
import pandas as pd 
from datetime import datetime

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

# TODO make c++ routine out of this 
def calculate_neighbours(frame_i):

    lx_box = 270 
    ly_box = 270
    lz_box = 3

    cutoff = 1.3
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
for file in files:
    print(file)
    frames = read_lammpstrj(file)
    neighbour_list = calculate_neighbours(frames[-1])
    G = nx.Graph() 
    G.add_edges_from(neighbour_list)
    degree = np.array([tuple[1] for tuple in G.degree()])
    number_of_bonded_particles = G.number_of_nodes()
    Nparticles=1000 
    number_of_unbonded_particles = Nparticles - number_of_bonded_particles
    print("unbonded", number_of_unbonded_particles)
    full_degree = np.append(degree,np.zeros(number_of_unbonded_particles))   
    mean_degree = np.mean(full_degree)
    std_degree = np.std(full_degree)
    print("degree", mean_degree, std_degree)
    
    new_results = {}
    new_results["file_id"] = file.split("/")[0]
    new_results["lambda"] = float(file.split("_")[4])
    new_results["shift"] = float(file.split("_")[2])
    new_results["mean_bonds"] = mean_degree
    new_results["std_bonds"] = std_degree 

    new_results = pd.DataFrame.from_dict(new_results, orient="index").T

    df = pd.concat([df, new_results], ignore_index=True)


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


