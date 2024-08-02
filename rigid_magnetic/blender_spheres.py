import bpy 
import mathutils
import numpy as np 

try:
    cube = bpy.data.objects['Cube']
    bpy.data.objects.remove(cube, do_unlink=True)
except:
    print("Object bpy.data.objects['Cube'] not found")

dir="/Users/ada/Documents/github_repos/magnetic_particles/MagneticParticles/rigid_magnetic"
data = np.loadtxt("{}/blender_spheres.txt".format(dir)) 

bpy.ops.outliner.orphans_purge()
sphere_location = data[::3, 2:5]

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
    r,g,b = np.random.rand(3)
    principled.inputs['Base Color'].default_value = (r,g,b,1)

    # Assign the material to the object
    obj.data.materials.append(mat)
    
    #bpy.context.selected_objects[0].name = "MySphere{}".format(i)
    
