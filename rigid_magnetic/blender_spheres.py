import bpy 
import mathutils
import numpy as np 
import cmasher as cm 
from mathutils import Vector


try:
    cube = bpy.data.objects['Cube']
    bpy.data.objects.remove(cube, do_unlink=True)
except:
    print("Object bpy.data.objects['Cube'] not found")

dir="/Users/ada/Documents/github_repos/magnetic_particles/MagneticParticles/rigid_magnetic"
#data = np.loadtxt("{}/blender_spheres.txt".format(dir)) 
#data = np.loadtxt("{}/blender_spheres_one_patch.txt".format(dir))
data = np.loadtxt("{}/blender_spheres_one_patch_linear_chains_0.2.txt".format(dir))
U = np.loadtxt("{}/dipole_per_particle.txt".format(dir))

ustart = -20
ustop = 20
urange = np.abs(ustop - ustart)
Ncolors=100
delta_u = urange/Ncolors

colorlist = cm.take_cmap_colors('cmr.ocean',Ncolors+1)


bpy.ops.outliner.orphans_purge()
sphere_location = data[::2, 2:5]

dipole_start = data[1::2,2:5]
dipole_vector = data[1::2,5:]


for i, sphere in enumerate(sphere_location):

    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=64, 
        ring_count=32, 
        radius=0.5, 
        location=(sphere[0],sphere[1],sphere[2]))  
    obj = bpy.context.object
  
    mat = bpy.data.materials.new("Blue")

    # Activate its nodes
    mat.use_nodes = True

    # Get the principled BSDF (created by default)
    principled = mat.node_tree.nodes['Principled BSDF']

    # Assign the color
    #r,g,b = np.random.rand(3)
    color_i = int(np.floor((U[i] - ustart)/delta_u))
    r,g,b = colorlist[color_i]
    principled.inputs['Base Color'].default_value = (r,g,b,1)

    # Assign the material to the object
    obj.data.materials.append(mat)
    bpy.context.object.active_material.blend_method = 'BLEND'
    bpy.context.object.active_material.shadow_method = 'NONE'
    
    new_arrow = bpy.data.objects.new(name="new empty", object_data=None)
    new_arrow.empty_display_type = "SINGLE_ARROW"
    
    new_arrow.location = Vector(dipole_start[i])
    new_vector = Vector(dipole_vector[i])
    new_arrow.rotation_euler = new_vector.to_track_quat('Z', 'Y').to_euler()
    bpy.context.collection.objects.link(new_arrow)
    
    
    #bpy.context.selected_objects[0].name = "MySphere{}".format(i)
