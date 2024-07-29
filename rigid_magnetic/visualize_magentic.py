import numpy as np 
import seaborn as sns 
import mdtraj as md 
import matplotlib.pyplot as plt 
import argparse 
from matplotlib.patches import Wedge, Rectangle

def read_file(filen):
    particles = []
    with open(filen, "r") as flmp:
            collect_line =False
            for line in flmp.readlines():
                    
                if line.startswith("ITEM: TIMESTEP"):
                    collect_line=False 
                
                if collect_line == True:
                    entry = np.array([float(x) for x in line.split()])
                    particles[-1].append(entry)
                    
                if line.startswith("ITEM: ATOMS"):
                    collect_line=True
                    particles.append([])
                    
    return particles 


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", 
                        type=str, 
                        default="trajectory.lammpstrj")
    
    parser.add_argument("-dipole", type=str, default="lateral")
    args = parser.parse_args()

    frames = read_file(args.f)
    freq = 10
    boxl = 40 
    radius=0.5/boxl
    
    for j in range(len(frames)-1, len(frames),freq):
        print(j)
        frame = np.array(frames[j])
        core_particles = frame[::2,2:4]*boxl
        patch_particles = frame[1::2,2:4]*boxl
            
        dist = patch_particles-core_particles
        dist = dist - boxl*np.rint(dist/boxl)
        distnorm = np.linalg.norm(dist,axis=1)
            
        fig, ax = plt.subplots(figsize=(20,20))
        ax.set_aspect('equal')

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        for i, center_i in enumerate(core_particles):

            center = (center_i[0]/boxl,center_i[1]/boxl)  
            ax.axis("off")
            c = plt.Circle(center,
                            radius, 
                            color='#C80036')
             
            ax.add_patch(c) 
            
            if args.dipole == "radial": 
                theta1 =  np.arccos(
                    dist[i,1]/distnorm[i])*180/np.pi
            
                theta2 = theta1 - 180   
                w = Wedge(center, 
                        radius, 
                        theta1, 
                        theta2, 
                        fc='#0C1844',
                        edgecolor='black')
                #ax.add_patch(w)
                width=0.004
                
                ax.arrow(center_i[0]/boxl, 
                        center_i[1]/boxl, 
                        dist[i,0]/boxl, dist[i,1]/boxl,
                        width = width,
                        zorder = 10,
                        head_width=width*3,
                        head_length=width*3,
                        fc = "#F8C794", 
                        ec = "k")
                
            if args.dipole == "lateral": 
                
                theta1 =  np.arccos(
                    dist[i,1]/distnorm[i])*180/np.pi
            
                theta2 = theta1 - 180 
                
                w = Wedge(center, 
                        radius, 
                        theta1, 
                        theta2, 
                        fc='#0C1844',
                        edgecolor='black')
                #ax.add_patch(w)
                
                width=0.004     
                ax.arrow(center_i[0]/boxl + dist[i,0]/boxl, 
                        center_i[1]/boxl + dist[i,1]/boxl, 
                        -dist[i,1]/boxl, dist[i,0]/boxl,
                        width = width,
                        zorder = 10,
                        head_width=width*3,
                        head_length=width*3,
                        fc = "#F8C794", 
                        ec = "k")
            
            
        plt.savefig("pngs/frame_{}.png".format(j),dpi=300)
        plt.savefig("pdfs/frame_{}.pdf".format(j))
        plt.close(fig)

  