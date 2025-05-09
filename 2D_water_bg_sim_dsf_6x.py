#!/home/toong/miniconda3/envs/cucondor/bin/python
#SBATCH --job-name='water_only'
#SBATCH --time=12-00:00:00
#SBATCH --nodes=1
#SBATCH --partition=upex
#SBATCH --constraint='EPYC'
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tong.you@icm.uu.se
#SBATCH -o slurm_output/%j.out
#SBATCH -e slurm_output/%j.out
import os, sys
from sys import stderr

sys.path.append("./")

import numpy as np

from skimage.measure import block_reduce

import time
import h5py

import scipy.constants as constants

from helper_functions import add_water_saxs

pi = constants.pi
e = constants.elementary_charge
h = constants.Planck
c = constants.speed_of_light

dsf = 6

bg_mask = "emc/make_detector/agipd_detector_mask.h5"
with h5py.File(bg_mask, "r") as det:
    det_mask = det["mask"][:]

cy, cx = det_mask.shape[0] // 2, det_mask.shape[1] // 2
det_mask = det_mask[cy - cx : cy + cx]

if dsf == 1:
    det_mask_ds = det_mask
else:
    d_mask_float = det_mask.astype(float)
    d_mask_float[det_mask == False] = np.nan
    
    det_mask_ds = block_reduce(d_mask_float, block_size=dsf, func=np.nansum)
    det_mask_ds = (det_mask_ds >= 2.) # 2. for 6x downsampling, and 1. for 4x downsampling
    
phot_eV = 9000
phot_J = phot_eV * e
phot_m = (h * c) / phot_J
pulse_energy = 200e-6

if dsf == 1:
    pixel_size = 200e-6
    dimX = det_mask.shape[1]
    dimY = det_mask.shape[0]
else:
    pixel_size = dsf * 200e-6
    dimX = det_mask.shape[1] // dsf
    dimY = det_mask.shape[0] // dsf
pixel_num_x = dimX - dimX // 2
pixel_num_y = dimY - dimY // 2

det_dist = 0.5

focus_diam = 14e-9
focus_rad = focus_diam / 2

beam_area = pi * (focus_rad**2)
beam_phots = pulse_energy / phot_J
beam_fluence = beam_phots / beam_area

theta_pixel_x = 0.5 * np.arctan((pixel_num_x * pixel_size) / det_dist)
theta_pixel_y = 0.5 * np.arctan((pixel_num_y * pixel_size) / det_dist)

resolution_x = phot_m / (2.0 * np.sin(theta_pixel_x))
resolution_y = phot_m / (2.0 * np.sin(theta_pixel_y))
pix_real_x = 0.5 * resolution_x
pix_real_y = 0.5 * resolution_y

center_to_corner = np.sqrt(
    (pixel_num_x * pixel_size) ** 2 + (pixel_num_x * pixel_size) ** 2
)
theta_max = 0.5 * np.arctan(center_to_corner / det_dist)
resolution_max = phot_m / (2.0 * np.sin(theta_max))

pat = np.ones_like(det_mask_ds)
water_bg = add_water_saxs(pat, pixel_size, det_dist, phot_m, pulse_energy)

sim_start, sim_end, sim_c = 0, 50, 1
n_sim = 20000
pat_ext = "1000k"

for s in range(sim_start, sim_end):
    print(f"\rSimulating round {sim_c}/{sim_end-sim_start}...", flush=True)
    print(f"Simulating {n_sim} water background patterns...", flush=True)
    particle_intens = np.zeros(shape=(n_sim, dimY, dimX))
    particle_real = np.zeros(shape=(n_sim, dimY, dimX))
    particle_orientations = np.zeros(shape=(n_sim, 4))

    def_rng = np.random.default_rng()

    det_mask_ds_stack = np.broadcast_to(
        det_mask_ds, (n_sim,) + det_mask_ds.shape
    ).astype(np.float64)
    water_stacked = np.broadcast_to(water_bg, (n_sim,) + water_bg.shape).astype(
        np.float64
    )

    water_stacked_mask = water_stacked.copy()
    water_stacked_mask[det_mask_ds_stack == False] = 0.0

    bg_water_poiss = def_rng.poisson(lam=water_stacked)
    bg_water_poiss_masked = def_rng.poisson(lam=water_stacked_mask)

    time_now = time.localtime(time.time())

    base_dir = f"sims_water_only/"
    folder_name = f"run_{s}_water_{pat_ext}_pats_dsf_{dsf}x/"

    if os.path.exists(base_dir + folder_name):
        print("Path exists!", flush=True)
    else:
        print("Path does not exist...", flush=True)
        print("Making new directory...", flush=True)
        os.mkdir(base_dir + folder_name)
        print("Written!", flush=True)

    print("Writing simulation parameters...", flush=True)
    with open(base_dir + folder_name + "simulation_parameters.txt", "w") as handle:
        handle.write(
            f"#########################################################################################\n"
        )
        handle.write(
            f"Date: {time_now[0]}/{time_now[1]}/{time_now[2]} {time_now[3]}:{time_now[4]}:{time_now[5]}\n"
        )
        handle.write(f"Photon energy: {phot_eV} eV\n")
        handle.write(f"Downsampling factor: {dsf}\n")
        handle.write(f"Pulse energy: {pulse_energy} J\n")
        handle.write(f"Focus diameter: {focus_diam} m\n")
        handle.write(f"Pixel size: {pixel_size} m\n")
        handle.write(f"Detector distance: {det_dist} m\n")
        handle.write(f"Photon count: {beam_phots}\n")
        handle.write(f"Beam area: {beam_area} um\u00b2\n")
        handle.write(f"Photon fluence: {beam_fluence*1e-12} ph/um\u00b2\n")
        handle.write(f"Photon fluence: {(beam_fluence*phot_J*1e-6)} uJ/um\u00b2\n")
        handle.write(f"Maximum resolution: {resolution_max*1e9} nm\n")
        handle.write(f"Resolution: {resolution_x*1e9} nm\n")
        handle.write(f"Pixel size: {pix_real_x*1e9} nm\n")
        handle.write(
            f"#########################################################################################\n"
        )
    print("Finishing writing simulation parameters...", flush=True)
    print("Saving arrays...", flush=True)
    np.save(
        base_dir + folder_name + "poisson_water_only.npy",
        arr=bg_water_poiss,
    )
    np.save(
        base_dir + folder_name + "poisson_water_only_masked.npy",
        arr=bg_water_poiss_masked,
    )
    print("Finished writing arrays...", flush=True)
    stderr.flush()

    sim_c += 1
