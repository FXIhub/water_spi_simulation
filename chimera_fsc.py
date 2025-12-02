from chimerax.core.commands import run
import glob

# Clearing log
run(session, "log clear")

# Fitting constants
n_searches_trans = 10
n_searches_rot = 10
n_searches = n_searches_rot + n_searches_trans
fit_metric = "correlation"

# Ground-truth DENSS electron density
denss_mrc = "../fsc_alignments/1ss8_denss.mrc"
center_x, center_y, center_z = 89.5, 89.5, 89.5

run(session, f"open {denss_mrc}")
run(session, f"volume #1 originIndex {center_x},{center_y},{center_z}")

# Merged densities
dens_files = glob.glob("../fsc_alignments/*.h5")
dens_vox_size = 2.3021449320029004 # for 0.5 m
center_dens = 186.5 # for 0.5 m
dim = 374 # for 0.5 m
N_files = len(dens_files)

# Loop over all densities and fit both the density/centrosymmetric twin
for f in range(N_files):
    file_name = dens_files[f]
    file_name_mrc = file_name.split(sep=".h5")[0] + ".mrc"
    file_name_mrc_flipped = file_name.split(sep=".h5")[0] + "_flip.mrc"

    run(session, f"log text Fitting density: {f+1}/{N_files}")

    # Only resample DENSS model first iteration
    if f == 0:
        # DENSS model on empty grid
        run(session, f"vop new size {dim} gridSpacing {dens_vox_size}") # create empty grid with correct dimensions to resample onto to preserve orientation information after fitting
        run(session, f"volume #2 originIndex {center_dens},{center_dens},{center_dens}")
        run(session, "volume resample #1 onGrid #2") #3
        run(session, f"volume #3 originIndex {center_dens},{center_dens},{center_dens}")

    # Normal density #4
    run(session, f"open {file_name}")
    run(session, f"volume #4 voxelSize {dens_vox_size}")
    run(session, f"volume #4 originIndex {center_dens},{center_dens},{center_dens}")
    # Normal density

    # Mirror density #5
    run(session, "volume flip #4 axis xyz") #5
    # Mirror density

    run(session, "fitmap #4 inMap #3")
    run(
        session,
        f"fitmap #4 inMap #3 metric {fit_metric} search {n_searches_trans} placement s",
    )
    run(
        session,
        f"fitmap #4 inMap #3 metric {fit_metric} search {n_searches_rot} placement r",
    )
    run(session, "fitmap #4 inMap #3")

    run(session, "fitmap #5 inMap #3")
    run(
        session,
        f"fitmap #5 inMap #3 metric {fit_metric} search {n_searches_trans} placement s",
    )
    run(
        session,
        f"fitmap #5 inMap #3 metric {fit_metric} search {n_searches_rot} placement r",
    )
    run(session, "fitmap #5 inMap #3")

    run(session, "volume resample #4 onGrid #3") #6
    run(session, "volume resample #5 onGrid #3") #7

    run(session, f"save {file_name_mrc} #6")
    run(session, f"save {file_name_mrc_flipped} #7")

    run(session, "close #4,5,6,7")

# Save ChimeraX density after fitting done and close all models
run(session, f"save ../fsc_alignments/1ss8_denss_rs.mrc #3")
run(session, "close all")