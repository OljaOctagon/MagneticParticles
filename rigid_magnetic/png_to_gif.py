import imageio
import glob
import numpy as np
import argparse 

parser = argparse.ArgumentParser()

parser.add_argument("-f", type=str, default="mag2p_shift_0_lambda_6_phi2d_0.0106_rid_1")
args = parser.parse_args()

images = []
frames= glob.glob("*{}*.png".format(args.f))
frame_values = np.sort([ int(frame.split("_")[1].split(".")[0]) for frame in frames])

for frame_i in frame_values:
    filename = "frame_{}_{}.png".format(frame_i,args.f)
    images.append(imageio.imread(filename))

imageio.mimsave('./{}_movie.gif'.format(args.f), images, duration=0.2)
