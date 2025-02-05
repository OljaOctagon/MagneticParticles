import shutil 
import os 
import numpy as np 
import shutil 
import re

def U2_chain(s):
    sigma=1 
    sr=0.5*s 
    theta = np.arctan(2*sr/sigma)
    d = np.sqrt( 1 + np.power(2*sr,2))
    U = -4 + (2/np.power(d,3))*(1-3*np.power(np.cos(theta),2)) 
    return U

def U2_zipper(s):

    r=0.5
    sr = r*s 
    
    cl = np.sqrt(np.power(r,2)-np.power(sr,2)) 
    U1 = -2/np.power(2*cl,3)

    d = np.sqrt(4*np.power(sr,2) + 4*np.power(cl,2))
    theta = np.arctan(sr/cl)
    U2 = 1/(np.power(d,3))*(1-np.power(np.cos(theta),2))        
    
    d = np.sqrt(16*np.power(sr,2) + 4*np.power(cl,2))
    theta = np.arctan((2*sr/cl))
    U3 = 1/(np.power(d,3))*(1-np.power(np.cos(theta),2))        
    
    U = U1 + 2*U2 + U3 
    
    return U

def U2_ap(s):
    sr = 0.5*s 
    U1=-1/np.power(1-2*sr,3)

    U2 = -1 

    U3 = -1/np.power(1+2*sr,3)

    U = U1 + 2*U2 + U3 

    return U

def inverted_renormalize_lbda(lbda,s):
    MIN_lbda = np.min([U2_chain(s), U2_zipper(s), U2_ap(s)])
    INRE_lbda = -2*lbda/MIN_lbda
    return INRE_lbda 

if __name__ == "__main__":
    
    # define lambda 
    Lbdas = np.array([1,1.5,2,2.3,2.5,3,3.5,4,4.5,5,6,7,8,9,10,15,20,25,30,35,40,50,60,70,80,90,100,120, 150,170,200,400])
    Nruns = np.arange(1,8)
    Shifts = np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.375,0.4,0.425,0.45,0.475,0.5,0.525,0.55,0.575,0.6,0.65,0.7])

    timestep_eq = 0.005 
    timestep = 0.005

    for irun in Nruns:
           for shift in Shifts:
                  for lbda in Lbdas:
                         if shift >= 0.7:
                             timestep = 0.001
                        
                         parent_dir = os.getcwd()
                         shift_str = np.round(shift,4)
                         dir = "mag2p_shift_{}_lambda_{}_phi2d_0.0106_rid_{}".format(shift_str,lbda,irun)
                         path = os.path.join(parent_dir, dir)
                         print(path)
                         os.mkdir(path)

                         # 2. Copy run files to new directory 
                         # 2.1 copy lammps run bash script 
                         src = os.path.join(parent_dir,"runlammps.sh")
                         dst = path 
                         shutil.copy(src,dst)
                         # 2.2 copy lammps script 
                         src = os.path.join(parent_dir,"in.mag2patch-quasi-2d")
                         shutil.copy(src,dst)
                         # 2.3 copy lammps molecule input file 
                         src = os.path.join(parent_dir, "2patch.txt")
                         shutil.copy(src,dst)

                         # 3. adjust temperature to yield right and renormalized lambda for lammps run bash script
                         mu_squared = 0.1 
                         RE_lbda = inverted_renormalize_lbda(lbda,shift)
                         temp = mu_squared/RE_lbda

                         src_file = os.path.join(parent_dir, dir, "runlammps.sh")
                         with open(src_file, "r") as sources:
                              lines = sources.readlines()
                        
                         with open(src_file, "w") as sources:
                            for line in lines:
                                 sources.write(re.sub("Temperature", "{}".format(temp), line))

            
                        # 4. change time steps for equilibration and assembly 
                         src_file = os.path.join(parent_dir, dir, "runlammps.sh")
                         with open(src_file, "r") as sources:
                              lines = sources.readlines()
                        
                         with open(src_file, "w") as sources:
                            for line in lines:
                                 sources.write(re.sub("timestep_eq", "{}".format(timestep_eq), line))

                         src_file = os.path.join(parent_dir, dir, "runlammps.sh")
                         with open(src_file, "r") as sources:
                              lines = sources.readlines()
                        
                         with open(src_file, "w") as sources:
                            for line in lines:
                                 sources.write(re.sub("timestep", "{}".format(timestep), line))
                        
                        # 4. set shift in lammps molecule input file 
                         src_file = os.path.join(parent_dir, dir,"2patch.txt")
                         with open(src_file, "r") as sources:
                              lines = sources.readlines()
                        
                         shift1 = np.round(shift/2,4)   
                         dst_file = os.path.join(parent_dir, dir,"2patch.txt")
                         with open(dst_file, "w") as sources:
                            for line in lines:
                                 sources.write(re.sub("s1", "{}".format(shift1), line))
                     
                         src_file = os.path.join(parent_dir, dir,"2patch.txt")
                         with open(src_file, "r") as sources:
                              lines = sources.readlines()

                         shift2 = np.round(-shift/2,4)  
                         dst_file = os.path.join(parent_dir, dir,"2patch.txt")
                         with open(dst_file, "w") as sources:
                            for line in lines:
                                 sources.write(re.sub("s2", "{}".format(shift2), line))
