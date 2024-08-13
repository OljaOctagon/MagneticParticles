import bpy 
import mathutils
import numpy as np 
import cmasher as cm 
from mathutils import Vector
from bpy import context as C, data as D

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
    
bpy.ops.outliner.orphans_purge()
bpy.ops.outliner.orphans_purge()
bpy.ops.outliner.orphans_purge()

def new_plane(mylocation, mysize, myrotation, myname,):
    bpy.ops.mesh.primitive_plane_add(
        size=mysize,
        calc_uvs=True,
        enter_editmode=False,
        align='WORLD',
        location=mylocation,
        rotation=myrotation,
        scale=(1, 1, 1))
    current_name = bpy.context.selected_objects[0].name
    plane = bpy.data.objects[current_name]
    plane.name = myname
    plane.data.name = myname + "_mesh"
    
    # material for cone 
    mat_plane = bpy.data.materials.new("Plane-Material")

    # Activate its nodes
    mat_plane.use_nodes = True

    # Get the principled BSDF (created by default)
    principled = mat_plane.node_tree.nodes['Principled BSDF']    
    principled.inputs['Base Color'].default_value = (162/255,139/255,85/255,1)

    # Assign the material to the object
    plane.data.materials.append(mat_plane)
      
    return


try:
    cube = bpy.data.objects['Cube']
    bpy.data.objects.remove(cube, do_unlink=True)
except:
    print("Object bpy.data.objects['Cube'] not found")


#new_plane((10,10,-1), 200, (0,0,0), "AMyFloor")
#new_plane((-10,10,70),200, (0,np.pi/2,0), "AMySide")


# create light datablock, set attributes
light_data = bpy.data.lights.new(name="light_2.80", type='SUN')
light_data.energy = 10

# create new object with our light datablock
light_object = bpy.data.objects.new(name="sun1", object_data=light_data)

# link light object
bpy.context.collection.objects.link(light_object)

# make it active 
bpy.context.view_layer.objects.active = light_object

#change location
light_object.location = (60, 10, 20)

# camera data 
cam1 = bpy.data.cameras.new("Camera 1")
cam1.lens = 18

# create the first camera object
cam_obj1 = bpy.data.objects.new("Camera 1", cam1)
cam_obj1.location = (40, 1, 10)
cam_obj1.rotation_euler = (3*np.pi/2,np.pi,-np.pi/2)
bpy.context.collection.objects.link(cam_obj1)

bpy.context.scene.camera = bpy.data.objects['Camera 1']

mus = np.linspace(30000000,30010000,10)
#for mu in mus: 
#dir="/Users/ada/Documents/github_repos/magnetic_particles/MagneticParticles/rigid_magnetic"
#data = np.loadtxt("{}/blender_spheres_one_patch_linear_chains_0.2.txt".format(dir))

dir="/Users/ada/Documents/github_repos/magnetic_particles/MagneticParticles/rigid_magnetic/2patch/3d/lambda_3/s_0.6_lambda2" 
file = "mu.dump.10300000"
data = np.loadtxt("{}/{}".format(dir,file), skiprows=9) 
U = np.loadtxt("{}/DIPOLE_{}".format(dir,file))

ustart = -5
ustop = 5
urange = np.abs(ustop - ustart)
Ncolors=10
delta_u = urange/Ncolors

colorlist = cm.take_cmap_colors('cmr.amber',Ncolors+1)

npatch = 2 
everyn= npatch + 1 
bpy.ops.outliner.orphans_purge()
sphere_location = data[::everyn, 2:5]

dipole_start = data[1::everyn,2:5]
dipole_vector = data[1::everyn,5:]


for i, sphere in enumerate(sphere_location):

    # Assign the color to particles 
    color_i = int(np.floor((U[i] - ustart)/delta_u))
    
    if color_i>Ncolors:
        color_i = Ncolors
    if color_i < 0:
        color_i = 0 
    
    r,g,b = colorlist[color_i]
    
    # make particle spheres 
    
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=64, 
        ring_count=32, 
        radius=0.5, 
        location=(sphere[0],sphere[1],sphere[2]))  
    obj = bpy.context.object
  
    # make matieral for sphere 
    mat = bpy.data.materials.new("Sphere-Material")


    # Activate its nodes
    mat.use_nodes = True

    # Get the principled BSDF (created by default)
    principled = mat.node_tree.nodes['Principled BSDF']
   
    principled.inputs['Base Color'].default_value = (1,1,1,1)
    principled.inputs['Alpha'].default_value = 0.1

    # Assign the material to the object
    obj.data.materials.append(mat)
    # make material transparent 
    bpy.context.object.active_material.blend_method = 'BLEND'
    bpy.context.object.active_material.shadow_method = 'NONE'

    
    # make empty arrows 
    
    new_arrow = bpy.data.objects.new(name="new empty", object_data=None)
    new_arrow.empty_display_type = "SINGLE_ARROW"
    
    new_arrow.location = Vector(dipole_start[i])
    
    new_vector = Vector(dipole_vector[i])
    
    new_arrow.rotation_euler = new_vector.to_track_quat('Z', 'Y').to_euler()
    bpy.context.collection.objects.link(new_arrow)

    
    # make cones for arrows in rendering 
    #bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.mesh.primitive_cone_add(location=Vector(dipole_start[i]))
    cone=bpy.context.object
    cone.rotation_euler = new_vector.to_track_quat('Z','Y').to_euler()
    cone.scale = (0.3,0.3,0.3)
    
    # material for cone 
    mat_cone = bpy.data.materials.new("Arrow-Material")

    # Activate its nodes
    mat_cone.use_nodes = True

    # Get the principled BSDF (created by default)
    principled = mat_cone.node_tree.nodes['Principled BSDF']    
    principled.inputs['Base Color'].default_value = (r,g,b,1)

    # Assign the material to the object
    cone.data.materials.append(mat_cone)
    
    #bpy.context.collection.objects.link(cone)


# Set the render engine to use (optional) 
 
# Set the output file format and location 
#bpy.context.scene.render.filepath = '{}/{}.png'.format(dir,file) 
#bpy.context.scene.render.image_settings.file_format = 'PNG' 
#bpy.context.scene.render.resolution_x =  2000
#bpy.context.scene.render.resolution_y =  2000
 
# Render the image 
#bpy.ops.render.render(write_still=True) 
  
    