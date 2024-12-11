#!/home/toong/miniconda3/envs/cucondor/bin/python
#SBATCH --job-name='water_phasing'
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --partition=allgpu
#SBATCH --constraint='A100|V100'
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tong.you@icm.uu.se
#SBATCH -o slurm_output/%j.out
#SBATCH -e slurm_output/%j.out
import os, os.path, sys

sys.path.append("./")
import h5py 
import numpy as np

import time

import scipy
import scipy.constants as constants

from helper_functions import sphere_idx, cylinder_idx

import spimage

pi = scipy.constants.pi
c = constants.speed_of_light
e = constants.elementary_charge
h = constants.Planck

subBg = True
    
vnum = "2"
    
emc_file = "emc/protein_with_water_debug_0001/data_50k/prot_wat_3/output_380.h5"
with h5py.File(emc_file, "r") as f_ptr:
    I_emc = np.squeeze(f_ptr["intens"][:])
    W_emc = np.squeeze(f_ptr["inter_weight"][:])

I_emc = I_emc[:-1, :-1, :-1]
W_emc = W_emc[:-1, :-1, :-1]

if subBg:
    emc_bg_file = "emc/water_only_debug_0001/data_50k/2/output_400.h5"
    with h5py.File(emc_bg_file, "r") as f_bg:
        I_emc_bg = np.squeeze(f_bg["intens"][:])
    I_emc_bg = I_emc_bg[:-1, :-1, :-1]
    frame = I_emc - I_emc_bg
else:
    frame = I_emc

center = frame.shape[0] // 2

sphere_rad = 65
mask_emc = sphere_idx(
    shape=frame.shape, radius=sphere_rad, position=(center, center, center)
)
mask_emc = np.logical_and(mask_emc, W_emc.astype(np.bool_))
frac_emc_good = mask_emc.sum() / (mask_emc.shape[0] ** 3)

fourier_mask = "mask_emc"

file = emc_file
npats = file.split(sep="/")[2].split(sep="_")[1]
sType = file.split(sep="/")[2]
pType = "emc_" + file.split(sep="/")[3]
print(f"Phasing {pType} model ({npats} patterns)!")

e_photon_eV = 9000
lambda_photon = (h * c) / (e_photon_eV * e)
d_detector = 1.0
s_pixel = 2400e-6

dimX = frame.shape[0]
dimY = frame.shape[1]
dimZ = frame.shape[2]

pixel_num = dimX - dimX // 2
theta_pixel = 0.5 * np.arctan((pixel_num * s_pixel) / d_detector)
resolution = lambda_photon / (2.0 * np.sin(theta_pixel))
voxel_size = 0.5 * resolution

D_particle = 15e-9
R_particle = np.ceil(D_particle / voxel_size) / 2
volume_fraction_sp = ((4 / 3) * pi * (R_particle**3)) / (dimX * dimY * dimZ)

support_sp = sphere_idx(
    shape=frame.shape, radius=R_particle, position=(dimX // 2, dimY // 2, dimZ // 2)
)

H_part = 14.7e-9
D_part = 13.7e-9
R_part = D_part / 2

H_particle_vox = H_part / voxel_size
R_particle_vox = R_part / voxel_size

volume_fraction_cyl = (H_particle_vox * pi * (R_particle_vox**2)) / (
    dimX * dimY * dimZ
)

support_cyl = cylinder_idx(
    shape=frame.shape,
    height=H_particle_vox,
    radius=R_particle_vox,
    position=(dimX // 2, dimY // 2, dimZ // 2),
)

vol_frac = volume_fraction_cyl
supp_phase = support_cyl

alg = "raar"

n_recons = 450
niter_alg = 500
niter_er = 450
niter_store = 2

beta_start = 0.70
beta_end = 0.75

i_frac, f_frac = 1.1, 1.01
volume_i, volume_f = i_frac * vol_frac, f_frac * vol_frac

blur_i, blur_f = 1.5, 1.0
supp_update = 20

niter_store_errors = 100

recon_intens = frame.copy()
recon_mask = mask_emc.copy()

def_rng = np.random.default_rng()

recon_real_array = np.zeros(shape=(n_recons, niter_store, *recon_intens.shape))
recon_phase_array = np.zeros(shape=(n_recons, niter_store, *recon_intens.shape))

recon_real_array_nosupp = np.zeros(shape=(n_recons, niter_store, *recon_intens.shape))
recon_phase_array_nosupp = np.zeros(shape=(n_recons, niter_store, *recon_intens.shape))

recon_fourier_array = np.zeros(shape=(n_recons, niter_store, *recon_intens.shape))
recon_fourier_poiss_array = np.zeros(shape=(n_recons, niter_store, *recon_intens.shape))

support_array = np.zeros(shape=(n_recons, niter_store, *recon_intens.shape))

error_real_array = np.zeros(shape=(n_recons, niter_store_errors))
error_fourier_array = np.zeros(shape=(n_recons, niter_store_errors))

constraints_list = ["enforce_positivity", "enforce_real"]

time_now = time.localtime(time.time())
print(f"Phasing: {file}", flush=True)
if subBg:
    print(f"Background file: {emc_bg_file}", flush=True)
print(f"Cylinder height and diameter: {H_part*1e9} nm and {D_part*1e9} nm", flush=True)
print(f"Beta_start: {beta_start}", flush=True)
print(f"Beta_end: {beta_end}", flush=True)
print(f"Fourier mask: {fourier_mask}", flush=True)
print(f"Photon energy: {e_photon_eV}", flush=True)
print(f"Voxel size: {voxel_size*1e9} nm", flush=True)
print(f"Version: {vnum}", flush=True)
for i in range(n_recons):
    print(f"{i+1}/{n_recons}", flush=True)
    phaser = spimage.Reconstructor()

    phaser.set_number_of_iterations(niter_alg + niter_er)
    phaser.set_number_of_outputs_images(niter_store)
    phaser.set_number_of_outputs_scores(niter_store_errors)

    phaser.set_initial_support(support_mask=supp_phase)
    phaser.set_mask(recon_mask)

    phaser.set_intensities(recon_intens)

    phaser.append_support_algorithm(
        "area",
        blur_init=blur_i,
        blur_final=blur_f,
        area_init=volume_i,
        area_final=volume_f,
        update_period=supp_update,
        number_of_iterations=niter_alg + niter_er,
    )
    if alg == "diffmap":
        phaser.append_phasing_algorithm(
            alg,
            constraints=constraints_list,
            beta_init=beta_start,
            beta_final=beta_end,
            number_of_iterations=niter_alg,
            gamma1=-1 / beta_start,
            gamma2=(3 - beta_start) / (2 * beta_start),
        )
    else:
        phaser.append_phasing_algorithm(
            alg,
            constraints=constraints_list,
            beta_init=beta_start,
            beta_final=beta_end,
            number_of_iterations=niter_alg,
        )
    phaser.append_phasing_algorithm(
        "er",
        constraints=constraints_list,
        number_of_iterations=niter_er,
    )

    output = phaser.reconstruct()

    recon_real = output["real_space"]
    recon_fourier = output["fourier_space"]
    support = output["support"]
    error_real = output["real_error"]
    error_fourier = output["fourier_error"]

    object_density = np.abs(recon_real)
    object_density[support == False] = 0.0
    object_phase = np.angle(recon_real)
    object_phase[support == False] = 0.0

    object_density_nsp = np.abs(recon_real)
    object_phase_nsp = np.angle(recon_real)

    object_fourier = np.abs(recon_fourier) ** 2
    object_fourier_poiss = def_rng.poisson(lam=object_fourier)

    recon_real_array[i] = object_density
    recon_phase_array[i] = object_phase
    recon_real_array_nosupp[i] = object_density_nsp
    recon_phase_array_nosupp[i] = object_phase_nsp

    recon_fourier_array[i] = object_fourier
    recon_fourier_poiss_array[i] = object_fourier_poiss
    support_array[i] = support
    error_real_array[i] = error_real
    error_fourier_array[i] = error_fourier

save_location = f"phasing/{sType}_{pType[4:]}_v_{vnum}/"

if os.path.exists(save_location) == False:
    os.mkdir(save_location)
    
corr_model_fname = sType + pType[3:] + "_" + f"{npats}"
corr_model = recon_intens
corr_model[~recon_mask] = np.nan
with h5py.File(save_location + corr_model_fname + "_corr.h5", mode="a") as handle:
    handle["intens"] = corr_model

np.save(save_location + "recon_real.npy", recon_real_array, allow_pickle=False)
np.save(save_location + "recon_phase.npy", recon_phase_array, allow_pickle=False)
np.save(
    save_location + "recon_real_nsp.npy", recon_real_array_nosupp, allow_pickle=False
)
np.save(
    save_location + "recon_phase_nsp.npy", recon_phase_array_nosupp, allow_pickle=False
)
np.save(save_location + "recon_fourier.npy", recon_fourier_array, allow_pickle=False)
np.save(
    save_location + "recon_fourier_poiss.npy",
    recon_fourier_poiss_array,
    allow_pickle=False,
)
np.save(save_location + "support.npy", support_array, allow_pickle=False)
np.save(save_location + "error_real.npy", error_real_array, allow_pickle=False)
np.save(save_location + "error_fourier.npy", error_fourier_array, allow_pickle=False)

with open(save_location + "phasing_params.txt", "w") as handle:
    handle.write(
        f"#########################################################################################\n"
    )
    handle.write(
        f"Date: {time_now[0]}/{time_now[1]}/{time_now[2]} {time_now[3]}:{time_now[4]}:{time_now[5]}\n"
    )
    handle.write(f"File to phase: {file}\n")
    handle.write(f"EMC fraction good : {frac_emc_good}\n")
    handle.write(f"Fourier mask: {fourier_mask}\n")
    handle.write(f"Fourier mask radius: {sphere_rad}\n")
    handle.write(f"Array size: ({dimX}, {dimY}, {dimZ})\n")
    handle.write(f"Voxel size: {voxel_size*1e9} nm\n")
    handle.write(
        f"Volume percentage initial and i_frac: {volume_i*100}% and {i_frac}\n"
    )
    handle.write(f"Volume percentage final and f_frac: {volume_f*100}% and {f_frac}\n")
    handle.write(f"Number of voxels in support: {supp_phase.sum()}\n")
    handle.write(f"Number of phase reconstructions: {n_recons}\n")
    handle.write(f"Phasing constraints: {constraints_list}\n")
    handle.write(
        f"Number of iterations for {alg.upper()} and ER: {niter_alg} and {niter_er}\n"
    )
    handle.write(f"Beta parameter initial: {beta_start}\n")
    handle.write(f"Beta parameter final: {beta_end}\n")
    handle.write(f"Gaussian blur min/max: {blur_i}/{blur_f}\n")
    handle.write(f"Support update frequency: {supp_update}\n")
    handle.write(
        f"#########################################################################################\n"
    )
