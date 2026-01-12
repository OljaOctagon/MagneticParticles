import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import argparse 
from matplotlib.patches import Wedge, Rectangle
import multiprocessing
import glob
#from lammpstools.tools import read_mag2patch

import numpy as np 
import gzip 
from pathlib import Path
import numba 
from scipy.spatial.distance import squareform

def read_mag2patch(t):
    Nskip = 9 
    Config = []
    Box = []
    frame_nr_old = -1 
    mfile = Path(t)
    Natoms = 3000
    if mfile.is_file():
        try: 
            with gzip.open(t, "r") as traj_file:    
           
                for i,line in enumerate(traj_file):
                    modulo = i % (Nskip+Natoms)
                    frame_nr = i // (Nskip+Natoms)
                    if frame_nr != frame_nr_old:
                        Config.append([])
                        Box.append([])
                        
                    if modulo == 3:
                        Natoms = np.array(line.split()).astype(float)[0]

                    if modulo == 5:
                        whole_line = np.array(line.split()).astype(float)
                        Lstart = whole_line[0]
                        Lend = whole_line[1]
                        Lx = Lend - Lstart
                        Box[-1].extend([Lx])
                        
                    if modulo == 6:
                        whole_line = np.array(line.split()).astype(float)
                        Lstart = whole_line[0]
                        Lend = whole_line[1]
                        Ly = Lend - Lstart
                        Box[-1].extend([Ly])
                        
                    if modulo == 7:
                        whole_line = np.array(line.split()).astype(float)
                        Lstart = whole_line[0]
                        Lend = whole_line[1]
                        Lz = Lend - Lstart
                        Box[-1].extend([Lz])
                

                    if modulo >=Nskip:
                            whole_line = np.array(line.split()).astype(float)
                            x = whole_line[2]
                            y = whole_line[3]
                            z = whole_line[4] 

                            Config[-1].append(np.array([x,y,z])) 

                    frame_nr_old = frame_nr
                    
        except (EOFError, IndexError, ValueError) as er:
            print("Caught error in {}:".format(t), er) 
    
        if Config:
            if len(Config[-1])!=Natoms:
                del Config[-1]
                del Box[-1]

    Config = np.array(Config)
    return Natoms, Config, Box 

def read_moments(mu):
    Nskip = 9
    Natoms=3000 
    
    frames = []
    frame_nr_old = -1 
    mfile = Path(mu)
    if mfile.is_file():
        try: 
            with gzip.open(mu, "r") as traj_file:
            
                for i,line in enumerate(traj_file):

                    modulo = i % (Nskip+Natoms)
                    frame_nr = i // (Nskip+Natoms)
                    if frame_nr != frame_nr_old:
                        frames.append([])

                    if modulo >=Nskip:
                        whole_line = np.array(line.split()).astype(float)
                       
                        mx = whole_line[6]
                        my = whole_line[7]
                        mz = whole_line[8] 
                        frames[-1].append(np.array([mx,my,mz])) 
                         
                    frame_nr_old = frame_nr
       
        except (EOFError, IndexError, ValueError) as er:
            print("Caught error in {}:".format(mu), er) 

        if frames:
            if len(frames[-1])!=Natoms:
                del frames[-1]

    frames = np.array(frames)
    return frames 


def process_files(filen):

    ifile = "{}/traj.gz".format(filen)
    Lambda = filen.split("_")[4]
    Shift = filen.split("_")[2]
    Natoms, frames, Box  = read_mag2patch(ifile)
    mu = read_moments("{}/mu.gz".format(filen))

    if frames.size > 0: 
        freq = 2 
        boxl=Box[0][0]
        radius=0.5/boxl
        
        for j in range(len(frames)-1, len(frames),freq):
            
            frame = np.array(frames[j])
            mu_part =  np.array(mu[j])
         
            core_particles = frame[::3,:2]*boxl
            patch1_particles = frame[1::3,:2]*boxl
            patch2_particles = frame[2::3,:2]*boxl
                
            dist1 = patch1_particles-core_particles
            dist1 = dist1 - boxl*np.rint(dist1/boxl)
            dist1norm = np.linalg.norm(dist1,axis=1)
            
            dist2 = patch2_particles-core_particles
            dist2 = dist2 - boxl*np.rint(dist2/boxl)
            dist2norm = np.linalg.norm(dist2,axis=1)

            scale_mu = 5
            mu_patch1 = mu_part[1::3,:2]*scale_mu
            mu_patch2 = mu_part[2::3,:2]*scale_mu


            fig, ax = plt.subplots(figsize=(20,20))
            ax.set_aspect('equal')
            ax.set_title("shift = {}, $\lambda$ = {}".format(Shift,Lambda), loc="left",fontsize=20)

            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])

            for i, center_i in enumerate(core_particles): 
                center = (center_i[0]/boxl,center_i[1]/boxl)    
                c = plt.Circle(center,
                            radius, 
                            fc="#FFFFFF",
                            ec="k",
                            linewidth=0.01)
                
        
                ax.axis("off")
                ax.add_patch(c)
                
                c1x, c1y, c2x,c2y = (0,0,0,0) 
                d1x, d1y, d2x,d2y = (0,0,0,0) 
                
                '''
                # automatically assume that dipoles are lateral 

                c1x = center_i[0]/boxl + dist1[i,0]/boxl
                c1y = center_i[1]/boxl + dist1[i,1]/boxl
            
                # rotate 90 degrees 
                d1x =  - dist1[i,1]/boxl
                d1y =    dist1[i,0]/boxl
            
                c2x = center_i[0]/boxl + dist2[i,0]/boxl
                c2y = center_i[1]/boxl + dist2[i,1]/boxl
            
                # rotate 90 degrees 
                d2x =    dist2[i,1]/boxl
                d2y =  - dist2[i,0]/boxl
                '''

                c1x = patch1_particles[i,0]/boxl - (mu_patch1[i,0]/boxl)*0.5
                c1y = patch1_particles[i,1]/boxl - (mu_patch1[i,1]/boxl)*0.5

                d1x = mu_patch1[i,0]/boxl
                d1y = mu_patch1[i,1]/boxl

                c2x = patch2_particles[i,0]/boxl - (mu_patch2[i,0]/boxl)*0.5
                c2y = patch2_particles[i,1]/boxl - (mu_patch2[i,1]/boxl)*0.5
                d2x = mu_patch2[i,0]/boxl
                d2y = mu_patch2[i,1]/boxl

                print(c1x, c1y,d1x,d1y)

                width=0.2*radius
                ax.arrow(
                        c1x, c1y, d1x, d1y,
                        width = width,
                        zorder = 10,
                        head_width=width*3,
                        head_length=width*3,
                        linewidth = 0.01,
                        fc = "#3B9DEE",
                        ec="k")
                
                ax.arrow(
                        c2x,c2y,d2x,d2y,  
                        width = width,
                        zorder = 10,
                        head_width=width*3,
                        head_length=width*3,
                        linewidth = 0.01,
                        fc = '#3B9DEE',
                        ec = "k"
                        )
                
            #plt.savefig("all_pngs/frame_{}.png".format(filen),dpi=300)
            plt.savefig("pdfs/frame_{}_{}.pdf".format(j,filen))
            plt.close(fig)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", 
                        type=str, 
                        default="dir_list.txt")
    
    args = parser.parse_args()

    #dirs = pd.read_csv(args.f).values.flatten().tolist()
    dirs = glob.glob("mag2p_shift_*_rid_1")

    with multiprocessing.Pool(processes=8) as pool:
        pool.map(process_files,dirs)
        pool.close()
        pool.join()

 
