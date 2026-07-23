import numpy as np
import freud
import matplotlib.pyplot as plt

# System parameters
N = 1000
Lx, Ly, Lz = 20.0, 20.0, 5.0  # thin in z
r_max = 0.45 * min(Lx, Ly, Lz)

# Generate positions in slab
positions = np.random.rand(N, 3) * [Lx, Ly, Lz]

# 3D box with no PBC in z
box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz)
box.periodic = [True, True, False]

# RDF in 3D, but with open boundary in z
rdf = freud.density.RDF(bins=100, r_max=r_max)
rdf.compute(system=(box, positions))

# Plot
plt.plot(rdf.bin_centers, rdf.rdf)
plt.xlabel("r")
plt.ylabel("g(r)")
plt.title("3D RDF with open z-boundary")
plt.grid(True)
plt.show()
