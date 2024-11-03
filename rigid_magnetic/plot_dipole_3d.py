import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import cmasher as cmr 
# Approximatelty equidistant points on a sphere 

import numpy as np

def fibonacci_sphere(samples=1000):
    # Create empty arrays for coordinates
    points = np.zeros((samples, 3))
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points[i] = [x, y, z]
    
    return points

# Generate points and print first few for verification
points = fibonacci_sphere(samples=1000)

# Visualization (optional)
"""
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
except ImportError:
    print("Matplotlib not installed; skipping visualization.")

"""

def U_dipole(p1,p2,r):
    rnorm = np.linalg.norm(r)
    U = (1/np.power(rnorm,3))*np.dot(p1,p2) - (3/np.power(rnorm,5))*(np.dot(p1,r)*np.dot(p2,r))
    return U 

def get_angle(a,b):
    dotab = np.dot(a,b)
    sign_a = np.sign(dotab)
    angle = np.arccos(sign_a*min(1,np.abs(dotab)))
    return angle 


r=np.array([1,0,0])

N = 100
angle_min = 0
angle_max = np.pi
angle_span = np.pi
delta = angle_span/N

dipole_value = np.zeros([N+1,N+1,N+1])


for p1 in points:
    for p2 in points:
        u = U_dipole(p1,p2,r)
        alpha = get_angle(p1,p2)
        beta = get_angle(p1,r)
        gamma = get_angle(p2,r)
       
        i = int(np.floor((alpha-angle_min)/delta))
        j = int(np.floor((beta-angle_min) /delta))
        k = int(np.floor((gamma-angle_min)/delta))

        dipole_value[i,j,k] = u    

        #print(alpha,beta,gamma,i,j,k,u)

    
Alpha = np.linspace(0, np.pi,N+1)
Beta = np.linspace(0, np.pi,N+1)

U_min, U_max = -1,1

fig, ax = plt.subplots()

U = dipole_value[:,50,:]
print(U)

#c = ax.pcolormesh(Alpha, Beta, U , cmap='coolwarm', vmin=U_min, vmax=U_max)
c = ax.pcolormesh(Alpha, Beta, U , cmap='cmr.redshift')
ax.set_title('Dipole potential in 3D')
# set the limits of the plot to the limits of the data
ax.axis([Alpha.min(), Alpha.max(), Beta.min(), Beta.max()])
ax.set_xlabel("$\\alpha$",size=15)
ax.set_ylabel("$\\beta$",size=15)

ax.set_xticks(np.arange(0, np.pi+0.01, np.pi/4))
ax.set_yticks(np.arange(0, np.pi+0.01, np.pi/4))
labels = ['$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

fig.colorbar(c, ax=ax, label="U")
plt.show()