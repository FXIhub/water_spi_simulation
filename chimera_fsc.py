from chimerax.core.commands import run
import glob

# Clearing Log
run(session, "log clear")

# Fitting constants
n_searches_trans = 100
n_searches_rot = 100
n_searches = n_searches_rot + n_searches_trans
fit_metric = "overlap"

# Ground-truth DENSS electron density
denss_mrc = "./1ss8_denss.mrc"
center_x, center_y, center_z = 89.5, 89.5, 89.5

run(session, f"open {denss_mrc}")
run(session, f"volume #1 originIndex {center_x},{center_y},{center_z}")

# Merged density
dens_files = glob.glob("*.h5")
dens_vox_size = 4.4554243278285693
center_dens = 64.5
N_files = len(dens_files)

# Loop over all densities and fit both the density/centrosymmetric twin
for f in range(N_files):
    file_name = dens_files[f]
    file_name_mrc = file_name.split(sep=".h5")[0] + ".mrc"
    file_name_mrc_flipped = file_name.split(sep=".h5")[0] + "_flip.mrc"

    run(session, f"log text Fitting density: {f+1}/{N_files}")
    run(session, f"open {file_name}")
    run(session, f"volume #2 voxelSize {dens_vox_size}")
    run(session, f"volume #2 originIndex {center_dens},{center_dens},{center_dens}")

    run(
        session,
        f"fitmap #2 inMap #1 metric {fit_metric} search {n_searches_trans} placement s levelInside 0.01",
    )
    run(
        session,
        f"fitmap #2 inMap #1 metric {fit_metric} search {n_searches_rot} placement r levelInside 0.01",
    )

    run(session, "volume flip #2 axis xyz")
    run(
        session,
        f"fitmap #3 inMap #1 metric {fit_metric} search {n_searches_trans} placement s levelInside 0.01",
    )
    run(
        session,
        f"fitmap #3 inMap #1 metric {fit_metric} search {n_searches_rot} placement r levelInside 0.01",
    )

    run(session, "volume resample #2 onGrid #1")
    run(session, "volume resample #3 onGrid #1")

    run(session, f"save {file_name_mrc} #4")
    run(session, f"save {file_name_mrc_flipped} #5")

    run(session, "close #2,3,4,5")

run(session, "close all")
