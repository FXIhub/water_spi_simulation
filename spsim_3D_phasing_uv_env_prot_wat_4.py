#!/gpfs/exfel/u/scratch/SPB/202325/p006056/filipe/uv_proj/.venv/bin/python
#SBATCH --job-name='water_phasing_npy'
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --partition=allgpu
#SBATCH --constraint='A100'
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tong.you@icm.uu.se
#SBATCH -o slurm_output/%j.out
#SBATCH -e slurm_output/%j.out

import os, os.path, sys
sys.path.append("/gpfs/exfel/u/scratch/SPB/202325/p006056/tong/water_spi_paper/")
from helper_functions import sphere_idx, calc_voxel_size

import h5py 
import numpy as np
import time
import scipy
import scipy.constants as constants
import spimage

pi = scipy.constants.pi
c = constants.speed_of_light
e = constants.elementary_charge
h = constants.Planck

subBg = True
vnum = "1"
    
emc_file = "emc/protein_water_ds_4x_fin_0001/data_100k_05/prot_wat_4/output_130.h5"
with h5py.File(emc_file, "r") as f_ptr:
    I_emc = np.squeeze(f_ptr["intens"][:])
    W_emc = np.squeeze(f_ptr["inter_weight"][:])

I_emc = I_emc[:-1, :-1, :-1]
W_emc = W_emc[:-1, :-1, :-1]

if subBg:
    emc_bg_file = "emc/water_only_ds_4x_0001/data_1M_05/wat_only_1/output_130.h5"
    with h5py.File(emc_bg_file, "r") as f_bg:
        I_emc_bg = np.squeeze(f_bg["intens"][:])
    I_emc_bg = I_emc_bg[:-1, :-1, :-1]
    frame = I_emc - I_emc_bg
else:
    frame = I_emc
 
center = frame.shape[0] // 2
mask_emc = W_emc.astype(np.bool_)
frac_emc_good = mask_emc.sum() / (mask_emc.shape[0] ** 3)

fourier_mask = "mask_emc"

file = emc_file
npats = file.split(sep="/")[2].split(sep="_")[1]
dType = file.split(sep="/")[1]
sType = file.split(sep="/")[2]
pType = "emc_" + file.split(sep="/")[3]
print(f"Phasing {pType} model ({npats} patterns)!")

e_photon_eV = 9000
lambda_photon = (h * c) / (e_photon_eV * e)
d_detector = 0.5
s_pixel = 800e-6
dimX = frame.shape[0]
voxel_size, _ , _ = calc_voxel_size(lambda_photon, d_detector, s_pixel, dimX)

R_part = (15e-9) / 2
R_particle_vox = R_part / voxel_size

volume_fraction_sphere = (((4/3) * pi) * (R_particle_vox**3)) / (
    dimX * dimX * dimX
)

support_sphere = sphere_idx(
    shape=frame.shape,
    radius=R_particle_vox,
    position=(dimX // 2, dimX // 2, dimX // 2),
)

vol_frac = volume_fraction_sphere
supp_phase = support_sphere

alg = "raar"
n_recons = 130
niter_alg = 900
niter_er = 600
niter_store = 2

beta_start = 0.9
beta_end = beta_start

i_frac, f_frac = 1.3, 0.8
volume_i, volume_f = i_frac * vol_frac, f_frac * vol_frac

blur_i, blur_f = 1.5, 1.0
supp_update = 50

niter_store_errors = (niter_alg + niter_er) // supp_update

recon_real_array = np.zeros(shape=(n_recons, niter_store, *frame.shape), dtype=np.complex64)
recon_fourier_array = np.zeros(shape=(n_recons, niter_store, *frame.shape), dtype=np.complex64)
recon_support_array = np.zeros(shape=(n_recons, niter_store, *frame.shape), dtype=np.bool)

support_size_array = np.zeros(shape=(n_recons, niter_store_errors))
error_real_array = np.zeros(shape=(n_recons, niter_store_errors))
error_fourier_array = np.zeros(shape=(n_recons, niter_store_errors))

constraints_list = ["enforce_positivity","enforce_real"]

time_now = time.localtime(time.time())
print(f"Phasing: {file}", flush=True)
if subBg:
    print(f"Background file: {emc_bg_file}", flush=True)
print(f"Sphere diameter: {2*R_part*1e9} nm", flush=True)
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
    phaser.set_mask(mask_emc)
    phaser.set_intensities(frame)

    phaser.append_support_algorithm(
        "area",
        blur_init=blur_i,
        blur_final=blur_f,
        area_init=volume_i,
        area_final=volume_f,
        update_period=supp_update,
        number_of_iterations=niter_alg
    )

    phaser.append_support_algorithm(
        "area",
        blur_init=blur_i,
        blur_final=blur_f,
        area_init=volume_f,
        area_final=volume_f,
        update_period=supp_update,
        number_of_iterations=niter_er
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

    real_space = output["real_space"].astype('complex64')
    fourier_space = output["fourier_space"].astype('complex64')
    support = output["support"]
    support_size = output["support_size"]
    error_real = output["real_error"]
    error_fourier = output["fourier_error"]

    recon_real_array[i] = real_space
    recon_fourier_array[i] = fourier_space
    recon_support_array[i] = support
    support_size_array[i] = support_size
    error_real_array[i] = error_real
    error_fourier_array[i] = error_fourier

save_location = f"phasing/{dType}_{sType}_{pType[4:]}_v_{vnum}/"
if os.path.exists(save_location) == False:
    os.mkdir(save_location)
    
corr_model_fname = sType + pType[3:] + "_" + f"{npats}"
corr_model = frame.copy()
corr_model[~mask_emc] = np.nan
with h5py.File(save_location + corr_model_fname + "_corr.h5", mode="a") as handle:
    handle["intens"] = corr_model

np.save(save_location + "recon_real.npy", recon_real_array, allow_pickle=False)
np.save(save_location + "recon_fourier.npy", recon_fourier_array, allow_pickle=False)
np.save(save_location + "recon_support.npy", recon_support_array, allow_pickle=False)
np.save(save_location + "support_size.npy", support_size_array, allow_pickle=False)
np.save(save_location + "error_real.npy", error_real_array, allow_pickle=False)
np.save(save_location + "error_fourier.npy", error_fourier_array, allow_pickle=False)

with open(save_location + "phasing_params.txt", "w") as handle:
    handle.write("#########################################################################################\n")
    handle.write(f"Date: {time_now[0]}/{time_now[1]}/{time_now[2]} {time_now[3]}:{time_now[4]}:{time_now[5]}\n")
    handle.write(f"File to phase: {file}\n")
    handle.write(f"EMC fraction good : {frac_emc_good}\n")
    handle.write(f"Fourier mask: {fourier_mask}\n")
    handle.write(f"Array size: ({dimX}, {dimX}, {dimX})\n")
    handle.write(f"Voxel size: {voxel_size*1e10} Å\n")
    handle.write(f"Volume percentage initial and i_frac: {volume_i*100}% and {i_frac}\n")
    handle.write(f"Volume percentage final and f_frac: {volume_f*100}% and {f_frac}\n")
    handle.write(f"Number of voxels in support: {supp_phase.sum()}\n")
    handle.write(f"Number of phase reconstructions: {n_recons}\n")
    handle.write(f"Phasing constraints: {constraints_list}\n")
    handle.write(f"Number of iterations for {alg.upper()} and ER: {niter_alg} and {niter_er}\n")
    handle.write(f"Beta parameter initial: {beta_start}\n")
    handle.write(f"Beta parameter final: {beta_end}\n")
    handle.write(f"Gaussian blur min/max: {blur_i}/{blur_f}\n")
    handle.write(f"Support update period: {supp_update}\n")
    handle.write("#########################################################################################\n")
