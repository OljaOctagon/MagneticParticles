from jinja2 import Environment, FileSystemLoader
import numpy as np

npart = 100
npatches = 2
natoms_per_mol = npatches + 1 
natoms = npart * natoms_per_mol
nangles = npart 
nbonds = npart*npatches

# diameter of core
sigma = 1

# Box 
lx2 = 40
ly2 = 40
lz2 = 0.5

shift_percent=0.5
zdist_from_center = (shift_percent)*sigma/2*np.array([-1,1])

context = {
    "number_of_atoms": natoms,
    "number_of_bonds": nbonds,
    "number_of_angles": nangles,
    "number_of_atom_types": 2,
    "number_of_bond_types": 1,
    "number_of_angle_types": 1, 
    "xlo": -lx2,
    "xhi": lx2,
    "ylo": -lx2,
    "yhi": lx2,
    "zlo": -lz2,
    "zhi": lz2,
}

atoms = []
bonds = []
angles = []

masses = [{"type": 1, "mass": 1.0}, {"type": 2, "mass": 1}]
context["masses"] = masses

particles = []
for imol in range(npart):
    core_id = (imol*natoms_per_mol) + 1 
   
    cutoff=3*sigma
    dist=0
    while dist<cutoff:
        xcore = np.random.rand() * 2 * lx2 - lx2
        ycore= np.random.rand() * 2 * ly2 - ly2
        #zcore = np.random.rand() * 2 * lz2 - lz2
        zcore = 0.0
        i=0
        dist=cutoff + 0.1 
        while dist>cutoff and i<len(particles) and particles:  
            distx = xcore - particles[i][0]
            disty = ycore - particles[i][1]
            distz = zcore - particles[i][2]
                
            distx = np.min([distx,lx2*2-distx])
            disty = np.min([disty,ly2*2-disty])
            distz = np.min([distz,lz2*2-distz])
        
            dist = np.linalg.norm(np.array([distx,disty,distz]))
            i+=1 
        
        if i == len(particles):
            break

    particles.append(np.array([xcore,ycore,zcore]))
    
    core_i = {
            "atom_id": core_id,
            "mol_id": imol + 1, 
            "atom_type": 1,
            "charge": 1,
            "x": xcore,
            "y": ycore,
            "z": zcore, 
            }
    
    atoms.append(core_i)
   
    for ip in range(npatches):
        xpatch = xcore
        ypatch = ycore + zdist_from_center[ip]
        zpatch = zcore 
        
        patch_id = core_id + ip+1 
        patch_i = {
            "atom_id": patch_id,
            "atom_type": 2,
            "mol_id": imol + 1, 
            "charge": 1,
            "x": xpatch,
            "y": ypatch,
            "z": zpatch,
            }
      
        atoms.append(patch_i)
        
        bond = {
            "id": imol * npatches + (ip+1),
            "bond_type": 1,
            "atom_1": core_id,
            "atom_2": patch_id,
        }
        bonds.append(bond)

    angle = {
        "id": imol + 1,
        "angle_type": 1,
        "atom_1": core_id + 1,
        "atom_2": core_id,
        "atom_3": core_id + 2,
    }

    angles.append(angle)

        
context["atoms"] = atoms
context["bonds"] = bonds
context["angles"] = angles 

environment = Environment(loader=FileSystemLoader("templates/"))
template = environment.get_template("template_start_harmonic.txt")

filename = "starting_state.txt"
with open(filename, mode="w") as output:
    output.write(template.render(context))
