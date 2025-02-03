import numpy as np 

def read_lammpstrj(t):
    Nskip = 9
    Natoms=3000 
    lx_box = 270 
    ly_box = 270
    lz_box = 3

    pos = []
    frames = []
    frame_nr_old = -1 
    with open(t, "r") as traj_file:
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

    frames = np.array(frames)
    return frames 

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


frames = read_lammpstrj("traj.lammpstrj")
neighbour_list = calculate_neighbours(frames[-1])
import networkx as nx 
import matplotlib.pyplot as plt 

G = nx.Graph() 
G.add_edges_from(neighbour_list)
av_degree = np.mean(np.array([tuple[1] for tuple in G.degree()]))
print(av_degree)


#domains = list(nx.connected_components(G))
#N_domains = len(domains)
#av_domain_size = 




