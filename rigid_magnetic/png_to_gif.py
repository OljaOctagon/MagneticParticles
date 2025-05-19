import imageio
import glob
import numpy as np
images = []
frames= glob.glob("*png")
frame_values = np.sort(
[ int(frame.split("_")[-1].split(".")[0]) for frame in frames])
for i in frame_values:
    filename="frame_{}.png".format(i)
    images.append(imageio.imread(filename))

imageio.mimsave('./movie.gif', images, duration=0.2)
