import os, os.path, sys
sys.path.append("/gpfs/exfel/u/scratch/SPB/202325/p006056/tong/water_spi_paper/")

from helper_functions import sphere_idx
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

subBg = False
vnum = "8"
    
emc_file = "emc/protein_water_ds_4x_0001/data_1M_no_mask/prot_only_0/output_120.h5"
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
s_pixel = 800e-6 # 800e-6 for 4x and 1200e-6 for 6x

dimX = frame.shape[0]
dimY = frame.shape[1]
dimZ = frame.shape[2]

pixel_num = dimX - dimX // 2
theta_pixel = 0.5 * np.arctan((pixel_num * s_pixel) / d_detector)
resolution = lambda_photon / (2.0 * np.sin(theta_pixel))
voxel_size = 0.5 * resolution

R_part = (15e-9) / 2
R_particle_vox = R_part / voxel_size

volume_fraction_sphere = (((4/3) * pi) * (R_particle_vox**3)) / (dimX * dimY * dimZ)

support_sphere = sphere_idx(
    shape=frame.shape,
    radius=R_particle_vox,
    position=(dimX // 2, dimY // 2, dimZ // 2),
)

vol_frac = volume_fraction_sphere
supp_phase = support_sphere

alg = "diffmap"

n_recons = 140
niter_alg = 500
niter_er = 300
niter_store = 2

beta_start = 0.90
beta_end = 0.95

i_frac, f_frac = 1.5, 0.80
volume_i, volume_f = i_frac * vol_frac, f_frac * vol_frac

blur_i, blur_f = 1.5, 1.0
supp_update = 50

niter_store_errors = (niter_alg + niter_er) // supp_update

constraints_list = ["enforce_real"]

save_location = f"phasing/{dType}_{sType}_{pType[4:]}_v_{vnum}_h5/"
if not os.path.exists(save_location):
    os.mkdir(save_location)

results_h5_path = os.path.join(save_location, f"phasing_results_v_{vnum}.h5")
with h5py.File(results_h5_path, "a") as h5f:
    dset_recon_real = h5f.create_dataset(
        "recon_real",
        shape=(n_recons, niter_store, dimX, dimY, dimZ),
        dtype="complex64",
        chunks=True,
    )
    dset_recon_fourier = h5f.create_dataset(
        "recon_fourier",
        shape=(n_recons, niter_store, dimX, dimY, dimZ),
        dtype="complex64",
        chunks=True,
    )
    dset_recon_support = h5f.create_dataset(
        "recon_support",
        shape=(n_recons, niter_store, dimX, dimY, dimZ),
        dtype="bool",
        chunks=True,
    )
    dset_support_size = h5f.create_dataset(
        "support_size",
        shape=(n_recons, niter_store_errors),
        dtype="float64",
        chunks=True,
    )
    dset_error_real = h5f.create_dataset(
        "error_real",
        shape=(n_recons, niter_store_errors),
        dtype="float64",
        chunks=True,
    )
    dset_error_fourier = h5f.create_dataset(
        "error_fourier",
        shape=(n_recons, niter_store_errors),
        dtype="float64",
        chunks=True,
    )
    
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
            number_of_iterations=niter_alg + niter_er
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
    
        real_space = output["real_space"].astype("complex64")
        fourier_space = output["fourier_space"].astype("complex64")
        support = output["support"]
        support_size = output["support_size"]
        error_real = output["real_error"]
        error_fourier = output["fourier_error"]
    
        dset_recon_real[i, ...] = real_space
        dset_recon_fourier[i, ...] = fourier_space
        dset_recon_support[i, ...] = support
        dset_support_size[i, :] = support_size
        dset_error_real[i, :] = error_real
        dset_error_fourier[i, :] = error_fourier

        h5f.flush()
        del phaser, output

corr_model_fname = sType + pType[3:] + "_" + f"{npats}"
corr_model = frame.copy()
corr_model[~mask_emc] = np.nan
with h5py.File(os.path.join(save_location, corr_model_fname + "_corr.h5"), "a") as handle:
    handle["intens"] = corr_model

with open(os.path.join(save_location, "phasing_params.txt"), "w") as handle:
    handle.write("#########################################################################################\n")
    handle.write(f"Date: {time_now[0]}/{time_now[1]}/{time_now[2]} {time_now[3]}:{time_now[4]}:{time_now[5]}\n")
    handle.write(f"File to phase: {file}\n")
    handle.write(f"EMC fraction good : {frac_emc_good}\n")
    handle.write(f"Fourier mask: {fourier_mask}\n")
    handle.write(f"Array size: ({dimX}, {dimY}, {dimZ})\n")
    handle.write(f"Voxel size: {voxel_size*1e9} nm\n")
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