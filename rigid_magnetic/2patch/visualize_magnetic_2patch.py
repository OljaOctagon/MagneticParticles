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
    args = parser.parse_args()

    frames = read_file(args.f)
    freq = 5
    boxl = 40 
    radius=0.5/boxl
    
    for j in range(0, len(frames),freq):
        print(j)
        frame = np.array(frames[j])
        core_particles = frame[::3,2:4]*boxl
        patch1_particles = frame[1::3,2:4]*boxl
        patch2_particles = frame[2::3,2:4]*boxl
            
        dist1 = patch1_particles-core_particles
        dist1 = dist1 - boxl*np.rint(dist1/boxl)
        dist1norm = np.linalg.norm(dist1,axis=1)
        
        dist2 = patch2_particles-core_particles
        dist2 = dist2 - boxl*np.rint(dist2/boxl)
        dist2norm = np.linalg.norm(dist2,axis=1)
            
        fig, ax = plt.subplots(figsize=(20,20))
        ax.set_aspect('equal')

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        for i, center_i in enumerate(core_particles):

            center = (center_i[0]/boxl,center_i[1]/boxl)    
            c = plt.Circle(center,
                           radius, 
                           fc='#C80036',
                           ec="#0C1844", 
                           lw=2,)  
            
      
            ax.axis("off")
            ax.add_patch(c)
            '''
            theta11 =  90 + np.arccos(
                dist1[i,1]/dist1norm[i])*180/np.pi
            theta12 = theta11 - 180     
            
            w1 = Wedge(center, 
                      radius, 
                      theta11, 
                      theta12, 
                      width = 0.2*radius,
                      fc='y')
            
            theta21 = 150 + np.arccos(
            dist2[i,1]/dist2norm[i])*180/np.pi
            theta22 = theta21 - 10  
            w2 = Wedge(center, 
                      radius, 
                      theta21, 
                      theta22, 
                      width = 0.2*radius,
                      fc='#0C1844')
               
            ax.add_patch(w1)
            ax.add_patch(w2)
            '''
            width=0.0015
            ax.arrow(
                    center_i[0]/boxl + dist1[i,0]/boxl, 
                    center_i[1]/boxl + dist1[i,1]/boxl, 
                    -dist1[i,1]/boxl, 
                    dist1[i,0]/boxl,
                    width = width,
                    zorder = 10,
                    head_width=width*3,
                    head_length=width*2,
                    fc = "#0C1844")
            
            ax.arrow(
                    center_i[0]/boxl + dist2[i,0]/boxl, 
                    center_i[1]/boxl + dist2[i,1]/boxl, 
                    dist2[i,1]/boxl, 
                    -dist2[i,0]/boxl,
                    width = width,
                    zorder = 10,
                    head_width=width*3,
                    head_length=width*2,
                    fc = '#0C1844')
            
        plt.savefig("pngs/frame_{}.png".format(j),dpi=300)
        plt.savefig("pdfs/frame_{}.pdf".format(j))
        plt.close(fig)

  