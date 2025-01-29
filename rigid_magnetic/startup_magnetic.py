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
    Lbdas = np.array([1,1.5,2,2.3,2.5,3,3.5,4,4.5,5,6,7,8,9,10,15,20,25,30,35,40,50,100])
    Nruns = np.arange(1,8)
    Shifts = np.linspace(0,0.7,0.1)

    for irun in Nruns:
           for shift in Shifts:
                  for lbda in Lbdas:
                         
                         # 1. make new directory 
                         parent_dir = os.getcwd()
                         dir = "mag2p_shift_{}_lambda_{}_phi2d_0.0106_rid_{}".format(shift,lbda,irun)
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

                         src_file = os.path.join(parent_dir, "2patch.txt")
                         with open(src_file, "r") as sources:
                              lines = sources.readlines()
                        
                         with open(src_file, "w") as sources:
                            for line in lines:
                                 sources.write(re.sub(r'^# deb', 'deb', line))

                         # SED 


                         # 4 set sfhit in lammps molecule input file 
                         # SED 


'''                                               
mkdir $dir
cp in.mag2patch-quasi-2d $dir
cp 2patch.txt $dir
mu_squared=0.01
temp=$(echo "scale=9; $mu_squared/$li" | bc)
cp runlammps.sh $dir
sed -i "s/Temperature/$temp/" $dir/runlammps.sh
cp 2patch.txt $dir
s1=$(echo "scale=2; $shift/2" | bc)
s2=$(echo "scale=2; -$shift/2" | bc)
sed -i "s/s1/$s1/" $dir/2patch.txt
sed -i "s/s2/$s2/" $dir/2patch.txt
'''