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
    5,
    20,
]

all_shifts = [
    # 0,
    # 0.2,
    0.3755,
    # 0.4,
    # 0.5,
    # 0.6,
]

all_fields = [0.01]


def get_box_size_3d(phi, N, sigma=1.0):
    vp = np.power(sigma, 3) * (np.pi / 6)
    V = N * vp / phi
    box_length = V ** (1 / 3)
    return box_length


def get_box_size_quasi_2d(phi, N, sigma=1.0, height=5):
    vp = np.power(sigma, 3) * (np.pi / 6)
    V = N * vp / phi
    box_length = np.sqrt(V / height)
    return box_length


def get_box_size_2d(phi, N, sigma=1.0):
    vp = np.power(sigma, 2) * (np.pi / 4)
    V = N * vp / phi
    box_length = np.sqrt(V)
    return box_length


N = 100
phi = 0.03
nruns = 1
height = 5
for irun in range(1, nruns + 1):
    for shift in all_shifts:
        for lbda in lbda_list:
            for field in all_fields:
                renormalized_lbda = renormalize_lbda(lbda, shift)
                Lx_box = get_box_size_2d(phi, N)

                dir = "mag2p_shift_{}_lambda_{}_phi_3d{}_field_{}_rid_{}".format(
                    shift,
                    lbda,
                    phi,
                    field,
                    irun,
                )
                os.makedirs(dir, exist_ok=True)
                os.system("cp in.mag2patch-quasi-2d-field {} ".format(dir))
                os.system("cp 2patch.txt {} ".format(dir))

                mu_squared = 0.01
                temp = mu_squared / renormalized_lbda
                timestep = 0.005
                timestep_eq = 0.005
                if shift >= 0.7:
                    timestep = 0.001
                    timestep_eq = 0.005

                os.system("cp runlammps.sh {} ".format(dir))
                os.system(
                    "sed -i 's/TEMPERATURE/{}/' {}/runlammps.sh".format(temp, dir)
                )
                os.system(
                    "sed -i 's/TIMESTEP_EQ/{}/' {}/runlammps.sh".format(
                        timestep_eq, dir
                    )
                )
                os.system(
                    "sed -i 's/TIMESTEP/{}/' {}/runlammps.sh".format(timestep, dir)
                )
                os.system("sed -i 's/LX_BOX/{}/' {}/runlammps.sh".format(Lx_box, dir))

                os.system("sed -i 's/HEIGHT/{}/' {}/runlammps.sh".format(height, dir))

                os.system("sed -i 's/FIELD/{}/' {}/runlammps.sh".format(field, dir))

                os.system("sed -i 's/NPARTICLES/{}/' {}/runlammps.sh".format(N, dir))

                os.system("cp 2patch.txt {} ".format(dir))
                s1 = shift / 2
                s2 = -shift / 2
                os.system("sed -i 's/s1/{}/' {}/2patch.txt".format(s1, dir))
                os.system("sed -i 's/s2/{}/' {}/2patch.txt".format(s2, dir))
