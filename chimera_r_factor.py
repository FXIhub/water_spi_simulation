from chimerax.core.commands import run
import glob

# Clearing Log
run(session, "log clear")

# Fitting constants
n_searches_trans = 10
n_searches_rot = 10
n_searches = n_searches_rot + n_searches_trans
fit_metric = "correlation"

# Condor model
gt_file = "../r_factor_alignments/3d_groel_agipd_9_kev_denss_dsf_4x_05_trm.h5"
gt_file_mrc = gt_file.split(sep=".h5")[0] + f"_{n_searches}.mrc"
center_gt = 186.5
run(session, f"open {gt_file}")
run(session, f"volume #1 originIndex {center_gt},{center_gt},{center_gt}")

# EMC models
emc_files = glob.glob("../r_factor_alignments/*_corr.h5")
center_emc = 186.5
N_files = len(emc_files)

# Loop over all models and fit model EMC intensity into Condor intensity
for f in range(N_files):
    file_name = emc_files[f]
    file_name_mrc = file_name.split(sep=".h5")[0] + f"_{n_searches}.mrc"

    run(session, f"log text Fitting file: {f+1}/{N_files}")
    run(session, f"open {file_name}")
    run(session, f"volume #2 originIndex {center_emc},{center_emc},{center_emc}")
    run(
        session,
        f"fitmap #2 inMap #1 metric {fit_metric} search {n_searches_trans} placement s",
    )
    run(
        session,
        f"fitmap #2 inMap #1 metric {fit_metric} search {n_searches_rot} placement r",
    )
    run(session, "volume resample #2 onGrid #1")
    run(session, f"save {file_name_mrc} #3")
    run(session, "close #2,3")

run(session, f"log text Aligned {N_files} EMC files...")

# Save ground-truth model after fitting done and close all models
run(session, f"save {gt_file_mrc} #1")
run(session, "close all")