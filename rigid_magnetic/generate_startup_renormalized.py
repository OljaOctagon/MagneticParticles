import numpy as np
import os


def U2_chain(s):
    sigma = 1
    sr = 0.5 * s
    theta = np.arctan(2 * sr / sigma)
    d = np.sqrt(1 + np.power(2 * sr, 2))
    U = -4 + (2 / np.power(d, 3)) * (1 - 3 * np.power(np.cos(theta), 2))
    return U


def U2_ap(s):
    sr = 0.5 * s
    U1 = -1 / np.power(1 - 2 * sr, 3)
    U2 = -1
    U3 = -1 / np.power(1 + 2 * sr, 3)
    U = U1 + 2 * U2 + U3
    return U


def renormalize_lbda(lbda, s):
    LBD0 = np.min([U2_chain(0), U2_ap(0)])
    MIN_lbda = np.min([U2_chain(s), U2_ap(s)])
    RE_lbda = lbda * (LBD0 / MIN_lbda)
    return RE_lbda


lbda_list = [
    1,
    1.5,
    2,
    2.3,
    2.5,
    3,
    3.5,
    4,
    4.5,
    5,
    6,
    7,
    8,
    9,
    10,
    15,
    20,
    25,
    30,
    35,
    40,
    50,
    100,
]

all_shifts = [
    0,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.3755,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
]


def get_box_size_3d(phi, N, sigma=1.0):
    vp = np.power(sigma, 3) * (np.pi / 6)
    V = N * vp / phi
    box_length = V ** (1 / 3)
    return box_length


N = 1000
phi = 0.03
for irun in range(1, 9):
    for shift in all_shifts:
        for lbda in lbda_list:
            renormalized_lbda = renormalize_lbda(lbda, shift)
            Lx_box = get_box_size_3d(phi, N)

            dir = "mag2p_shift_{}_lambda_{}_phi3d_{}_rid_{}".format(
                shift,
                lbda,
                irun,
                phi,
            )
            os.makedirs(dir, exist_ok=True)
            os.system("cp in.mag2patch-quasi-2d {} ".format(dir))
            os.system("cp 2patch.txt {} ".format(dir))

            mu_squared = 0.01
            temp = mu_squared / renormalized_lbda
            timestep = 0.005
            timestep_eq = 0.005
            if shift >= 0.7:
                timestep = 0.001
                timestep_eq = 0.005

            os.system("cp runlammps.sh {} ".format(dir))
            os.system("sed -i 's/Temperature/{}/' {}/runlammps.sh".format(temp, dir))
            os.system(
                "sed -i 's/timestep_eq/{}/' {}/runlammps.sh".format(timestep_eq, dir)
            )
            os.system("sed -i 's/timestep/{}/' {}/runlammps.sh".format(timestep, dir))
            os.system("sed -i 's/Lx_box/{}/' {}/runlammps.sh".format(Lx_box, dir))

            os.system("cp 2patch.txt {} ".format(dir))
            s1 = shift / 2
            s2 = -shift / 2
            os.system("sed -i 's/s1/{}/' {}/2patch.txt".format(s1, dir))
            os.system("sed -i 's/s2/{}/' {}/2patch.txt".format(s2, dir))
