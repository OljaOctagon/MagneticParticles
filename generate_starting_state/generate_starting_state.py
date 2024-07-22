from jinja2 import Environment, FileSystemLoader
import numpy as np

npart = 100
npatches = 2
natoms_per_mol = npatches + 1 

natoms = npart * natoms_per_mol
# diameter of core
sigma = 1

# Box 
lx2 = 40
ly2 = 40
lz2 = 40 

context = {
    "number_of_atoms": natoms,
    "number_of_atom_types": 2,
    "xlo": -lx2,
    "xhi": lx2,
    "ylo": -lx2,
    "yhi": lx2,
    "zlo": -lz2,
    "zhi": lz2,
}

atoms = []
molecules = []

particles = []

for imol in range(npart):
    core_id = (imol*natoms_per_mol) + 1 
   
    dist=0
    while dist<sigma:
        xcore = np.random.rand() * 2 * lx2 - lx2
        ycore= np.random.rand() * 2 * ly2 - ly2
        zcore = np.random.rand() * 2 * lz2 - lz2
        i=0
        dist=sigma*2 
        while dist>sigma and i<len(particles) and particles:  
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
            "atom_type": 1,
            "charge": 1,
            "x": xcore,
            "y": ycore,
            "z": zcore, 
            }
    molecule_i = {"atom_id": core_id, "mol_id": imol}

    atoms.append(core_i)
    molecules.append(molecule_i)

    shift_percent=0.5
    # patch connection vector points upward to 0,0,1
    zdist_from_center = (shift_percent)*sigma/2*np.array([-1,1])
    for ip in range(npatches):
        xpatch = xcore
        ypatch = ycore 
        zpatch = zcore + zdist_from_center[ip]
        
        patch_id = core_id + ip+1 
        patch_i = {
            "atom_id": patch_id,
            "atom_type": 2,
            "charge": 1,
            "x": xpatch,
            "y": ypatch,
            "z": zpatch, 
            }
        molecule_i = {"atom_id": patch_id, "mol_id": imol}

        atoms.append(patch_i)
        molecules.append(molecule_i)
        
    
context["atoms"] = atoms
context["molecules"] = molecules

environment = Environment(loader=FileSystemLoader("templates/"))
template = environment.get_template("template_start.txt")

filename = "starting_state.txt"
with open(filename, mode="w") as output:
    output.write(template.render(context))
