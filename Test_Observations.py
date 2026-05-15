
import os
import yaml
import json
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

from typing import Dict, List, Tuple


# tudatpy imports
from tudatpy import math
from tudatpy import constants

from tudatpy.interface import spice
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.estimation.observable_models_setup import links

import tudatpy.estimation
from tudatpy import util
#import tudatpy.estimation_setup

#from tudatpy.numerical_simulation import estimation

#from tudatpy.numerical_simulation import estimation_setup #,Time


from tudatpy import numerical_simulation

from tudatpy.astro import time_conversion, element_conversion,frame_conversion
from tudatpy.astro.time_conversion import DateTime


from tudatpy.data import save2txt

import sys
from pathlib import Path

# Add parent directory to Python path
#sys.path.append(str(Path(__file__).resolve().parent.parent))


import Analysis_All_Estimations_and_Data as UncertantyPropUtils

# Get the path to the directory containing this file
current_dir = Path(__file__).resolve().parent

# Append the HelperFunctions directory
sys.path.append(str(current_dir / "HelperFunctions"))

import ProcessingUtils
import PropFuncs
import FigUtils
import ObsFunc
import nsdc
import ObservationImplementation
#import RunMultipleEstimations
import MainPostprocessing as PostProc
import EstimationAnalysisTemplates as EstimationTemplates

matplotlib.use("PDF")  #tkagg



# Define temporal scope of the simulation - equal to the time JUICE will spend in orbit around Jupiter
simulation_start_epoch = DateTime(1963, 1,  1).epoch() #2006, 8,  27 1963, 3,  4   1989 1996
simulation_end_epoch   = DateTime(2025, 1, 1).epoch()   #2025, 1, 1    2003  2010

simulation_initial_epoch = DateTime(2006, 10, 1).epoch() #2006, 10, 1
global_frame_origin = 'SSB'
global_frame_orientation = 'ECLIPJ2000'

#--------------------------------------------------------------------------------------------
# ENVIORONMENT SETTINGS 
#--------------------------------------------------------------------------------------------
settings_env = dict()
settings_env["start_epoch"] = simulation_start_epoch
settings_env["end_epoch"] = simulation_end_epoch
settings_env["bodies"] = ['Sun','Jupiter', 'Saturn','Neptune','Triton','Uranus','Mercury','Venus','Mars','Earth'] #
settings_env["global_frame_origin"] = global_frame_origin
settings_env["global_frame_orientation"] = global_frame_orientation
settings_env["interpolator_triton_cadance"] = 60*8
settings_env["neptune_extended_gravity"] = "Jacobson2009"
settings_env['use_created_env'] = False

settings_env['Neptune_rot_model_type'] = 'IAU2015' 
    # Model Type for rotation model of Neptune:
    #  'simple_from_spice' - simple spice,
    #  'spice' - full spice,
    #  'IAU2015' - based on the IAU2015 paper
    #   'Pole_Model_Jacobson2009' - IAU rotation model estimated by Jacobson 2009
    
#--------------------------------------------------------------------------------------------
# ACCELERATION SETTINGS 
#--------------------------------------------------------------------------------------------

settings_acc = dict()
settings_acc['bodies_to_propagate'] = ['Triton']
settings_acc['central_bodies'] = ['Neptune']
settings_acc['bodies_to_simulate'] = ['Sun','Jupiter', 'Saturn','Neptune','Triton','Uranus','Mercury','Venus','Mars','Earth'] 
settings_acc['bodies'] = settings_env["bodies"]

settings_acc['neptune_extended_gravity'] =  "Jacobson2009"


accelerations_cfg = PropFuncs.build_acceleration_config(settings_acc)
settings_acc['accelerations_cfg'] = accelerations_cfg
#--------------------------------------------------------------------------------------------
# PROPAGATOR SETTINGS 
#--------------------------------------------------------------------------------------------

settings_prop = dict()
settings_prop['start_epoch'] = settings_env["start_epoch"]
settings_prop['end_epoch'] = settings_env["end_epoch"]
settings_prop['initial_epoch'] = simulation_initial_epoch
settings_prop['bodies_to_propagate'] = settings_acc['bodies_to_propagate'] 
settings_prop['central_bodies'] = settings_acc['central_bodies']
settings_prop['global_frame_orientation'] = settings_env["global_frame_orientation"]
settings_prop['fixed_step_size'] = 60*60 # 60 minutes

#--------------------------------------------------------------------------------------------
# OBSERVATION SETTINGS 
#--------------------------------------------------------------------------------------------

# --- Load names of data files you wish to include
with open("file_names.json", "r") as f:
    file_names_loaded = json.load(f)

# weights = pd.read_csv(
#         "Results/PoleEstimationRealObservations/LoopTest2/initial_state_only/0/summary.txt", #Results/BetterFigs/AllModernObservations/PostProcessing/First/weights.txt
#         sep="\t",
#         index_col="id")

settings_obs = dict()
settings_obs["mode"] = ["pos"]
settings_obs["bodies"] = [("Triton", "Neptune")]                           # bodies to observe
settings_obs["cadence"] = 60*60*3 # Every 3 hours
settings_obs["type"] = "Real" # Simulated or Real observations

#TEST FILE CHANGE
# file_names_loaded = [
#         'Triton_286_nm0090.csv',]


settings_obs["files"] = file_names_loaded             
settings_obs["observations_folder_path"] = "Observations/AllModernECLIPJ2000"  #RelativeObservations AllModernECLIPJ2000 AllModernJ2000

# weights = weights.reset_index()

settings_obs["use_weights"] = True
# settings_obs["ra_dec_independent_weights"] = False
# settings_obs["timeframe_weights"] = False
# settings_obs["weights"] = weights

settings_obs["use_loaded_obs"] = False

settings_obs["residual_filtering"] = True
settings_obs["epoch_filter_dict"] = None 


#Make sure all other weight types are off
# settings_obs['std_weights'] = False
# settings_obs["per_night_weights"] = False
# settings_obs["per_night_weights_id"] = False 
# settings_obs['per_night_weights_hybrid'] = False


# settings_obs['use_old_obs_func'] = False



#--------------------------------------------------------------------------------------------
# ESTIMATION SETTINGS 
#--------------------------------------------------------------------------------------------

settings_est = dict()
#settings_est['pseudo_observations_settings'] = pseudo_observations_settings
#settings_est['pseudo_observations'] = pseudo_observations

settings_est['est_parameters'] = ['initial_state'] #,'iau_rotation_model_pole','iau_rotation_model_pole_rate'] 
    #Possible settings: 
    # initial state - default
    #GM_Neptune - gravitational parameter Neptune
    #GM_Triton - gravitational parameter Triton
    # iau_rotation_model_pole - rotation pole position (alpha,delta) with IAU rotation model
    # iau_rotation_model_pole_rate - rotation pole rate  (alpha_dot, delta_dot) with IAU rotation model
    # iau_rotation_model_pole_librations - 1st order libration terms  
    # Spherical Harmonics Neptune (C20,C40) - extended body gravity of Neptune C20,C40 (J2,J4)
    
    #This is the proper order keep in mind !!!

    # Rotation_Pole_Position_Neptune - fixed rotation pole position (only with simple rotational model !)

settings_est['a_priori_covariance'] = False

#fill in settings 
settings = dict()
settings["env"] = settings_env
settings["acc"] = settings_acc
settings["prop"] = settings_prop
settings["obs"] = settings_obs
settings["est"] = settings_est



def make_timestamped_folder(base_path="Results"):
    folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_path = Path(base_path) / folder_name
    full_path.mkdir(parents=True, exist_ok=True)
    return full_path


##############################################################################################
# LOAD SPICE KERNELS
##############################################################################################

from pathlib import Path

# Path to the current script
current_dir = Path(__file__).resolve().parent

# Kernel folder 
kernel_folder = "Kernels" #current_dir.parent / 

#kernel_folder = "/Kernels/"
kernel_paths=[
    "pck00010.tpc",
    "gm_de440.tpc",
    "nep097.bsp",     
    #"nep105.bsp",
    "naif0012.tls"
    ]

spice.load_standard_kernels()

# Load your kernels
for k in kernel_paths:
    spice.load_kernel(os.path.join(kernel_folder, k))



settings['env']['Neptune_rot_model_type'] = 'IAU2015'

# Common settings — per-variant overrides applied in the loop below
settings['est']['a_priori_pole'] = False

# ---- Create SPICE environment ----
body_settings, system_of_bodies = PropFuncs.Create_Env(settings['env'])

from tudatpy.estimation import observations_setup as tud_obs_setup
from tudatpy.numerical_simulation import environment_setup as _num_env
from RunSinglePropagation import RunSinglePropagation as _run_propagation

arcsec_to_rad = np.pi / (180.0 * 3600.0)
j2000_epoch   = dt.datetime(2000, 1, 1, 12)

out_dir_obs = Path("Results/ObservationsAnalysis")
out_dir_obs.mkdir(parents=True, exist_ok=True)
(out_dir_obs / "per_file").mkdir(parents=True, exist_ok=True)

##############################################################################################
# STEP 1 — RUN PROPAGATION (before any observation loading)
##############################################################################################

print("\nRunning numerical propagation...")
state_history_prop, state_history_array_prop = _run_propagation(
    settings, out_dir_obs, load_kernels=False
)

# Build propagated environment: same as SPICE env but Triton uses tabulated ephemeris
body_settings_prop, _ = PropFuncs.Create_Env(settings['env'])
body_settings_prop.get("Triton").ephemeris_settings = _num_env.ephemeris.tabulated(
    state_history_prop, global_frame_origin, global_frame_orientation
)
system_of_bodies_prop = _num_env.create_system_of_bodies(body_settings_prop)

##############################################################################################
# STEP 2 — LOAD OBSERVATIONS INTO BOTH ENVIRONMENTS
# LoadObservations registers ground stations on system_of_bodies as a side effect.
# Each system_of_bodies needs its own LoadObservations call.
##############################################################################################

print("\nLoading observations (unfiltered) into SPICE environment...")
obs_unfiltered, obs_settings_all, set_ids_all, _ = ObsFunc.LoadObservations(
    settings["obs"]["observations_folder_path"],
    system_of_bodies,
    file_names_loaded,
    Residual_filtering=False, epoch_filter_dict=None
)

print("Loading observations (unfiltered) into propagated environment...")
obs_unfiltered_prop, obs_settings_prop_all, set_ids_prop, _ = ObsFunc.LoadObservations(
    settings["obs"]["observations_folder_path"],
    system_of_bodies_prop,
    file_names_loaded,
    Residual_filtering=False, epoch_filter_dict=None
)

print("Loading observations (filtered) — authoritative source for accepted/rejected epochs...")
obs_filtered, obs_settings_filt, set_ids_filt, epochs_rej_filtered = ObsFunc.LoadObservations(
    settings["obs"]["observations_folder_path"],
    system_of_bodies,
    file_names_loaded,
    Residual_filtering=True, epoch_filter_dict=None
)

##############################################################################################
# STEP 3 — CROSS-CHECKS
##############################################################################################

assert set_ids_all == set_ids_prop, \
    f"set_ids mismatch: SPICE unfiltered vs prop unfiltered\n  {set_ids_all}\n  {set_ids_prop}"
assert set_ids_all == set_ids_filt, \
    f"set_ids mismatch: SPICE unfiltered vs SPICE filtered\n  {set_ids_all}\n  {set_ids_filt}"
print(f"\nCross-check PASSED: set_ids consistent across all 3 loads ({len(set_ids_all)} sets)")

obs_times_unfiltered = obs_unfiltered.get_observation_times()
obs_times_prop       = obs_unfiltered_prop.get_observation_times()
obs_times_filtered   = obs_filtered.get_observation_times()

obs_counts  = [len(obs_times_unfiltered[j]) for j in range(len(set_ids_all))]
set_offsets = np.cumsum([0] + obs_counts)

# Derive rejected epochs from obs collections (more reliable than the dict)
# and cross-check against epochs_rej_filtered dict
rejected_epochs_per_set = {}   # authoritative, derived from obs_unfiltered vs obs_filtered
all_checks_passed = True

print("\nCross-checking per-set epoch counts...")
for j, set_id in enumerate(set_ids_all):
    unf_set  = set(float(t) for t in obs_times_unfiltered[j])
    prop_set = set(float(t) for t in obs_times_prop[j])
    filt_set = set(float(t) for t in obs_times_filtered[j])
    rej_obs  = unf_set - filt_set           # derived from obs collections
    rej_dict = set(float(t) for t in epochs_rej_filtered.get(set_id, []))

    rejected_epochs_per_set[set_id] = rej_obs

    ok = True
    if unf_set != prop_set:
        print(f"  FAIL [{set_id}]: SPICE unfiltered epochs != prop unfiltered epochs "
              f"({len(unf_set)} vs {len(prop_set)})")
        ok = False
    if filt_set - unf_set:
        print(f"  FAIL [{set_id}]: {len(filt_set - unf_set)} filtered epochs not present in unfiltered")
        ok = False
    if len(unf_set) != len(filt_set) + len(rej_obs):
        print(f"  FAIL [{set_id}]: unfiltered({len(unf_set)}) != "
              f"filtered({len(filt_set)}) + rejected({len(rej_obs)})")
        ok = False
    if rej_obs != rej_dict:
        print(f"  WARN [{set_id}]: epochs_rej_filtered dict ({len(rej_dict)} epochs) "
              f"differs from obs-derived rejection ({len(rej_obs)} epochs)")
        ok = False

    status = "OK  " if ok else "FAIL"
    print(f"  {status} [{set_id}]: total={len(unf_set)}  "
          f"accepted={len(filt_set)}  rejected={len(rej_obs)}")
    if not ok:
        all_checks_passed = False

if all_checks_passed:
    print("All per-set cross-checks PASSED.")
else:
    print("WARNING: Some cross-checks FAILED — inspect output above before trusting results.")

##############################################################################################
# STEP 4 — COMPUTE RESIDUALS (three sources)
##############################################################################################

print("\nComputing tudatpy/SPICE residuals...")
obs_sim_spice = tud_obs_setup.observations_simulation_settings.create_observation_simulators(
    obs_settings_all, system_of_bodies
)
tudatpy.estimation.observations.compute_residuals_and_dependent_variables(
    obs_unfiltered, obs_sim_spice, system_of_bodies
)
res_concat = np.array(obs_unfiltered.get_concatenated_residuals())

print("Computing SPICE nep097 residuals...")
ra_spice_flat, dec_spice_flat = ObsFunc.Get_SPICE_residual_from_observations(
    obs_unfiltered, set_ids_all, system_of_bodies,
    global_frame_orientation=global_frame_orientation
)

print("Computing propagation-based residuals...")
obs_sim_prop_sims = tud_obs_setup.observations_simulation_settings.create_observation_simulators(
    obs_settings_prop_all, system_of_bodies_prop
)
tudatpy.estimation.observations.compute_residuals_and_dependent_variables(
    obs_unfiltered_prop, obs_sim_prop_sims, system_of_bodies_prop
)
res_concat_prop = np.array(obs_unfiltered_prop.get_concatenated_residuals())

# Cross-check residual array lengths
n_total = set_offsets[-1]
assert len(res_concat)      == 2 * n_total, \
    f"res_concat length mismatch: {len(res_concat)} vs {2*n_total}"
assert len(res_concat_prop) == 2 * n_total, \
    f"res_concat_prop length mismatch: {len(res_concat_prop)} vs {2*n_total}"
assert len(ra_spice_flat)   == n_total, \
    f"ra_spice_flat length mismatch: {len(ra_spice_flat)} vs {n_total}"
print(f"Residual length cross-check PASSED: {n_total} observations total")

# Convert all to arcsec (flat arrays, same observation order as set_offsets)
ra_tud_all   = res_concat[0::2]      / arcsec_to_rad
dec_tud_all  = res_concat[1::2]      / arcsec_to_rad
ra_prop_all  = res_concat_prop[0::2] / arcsec_to_rad
dec_prop_all = res_concat_prop[1::2] / arcsec_to_rad

##############################################################################################
# STEP 4b — BIASED OBSERVATIONS (manual Dec bias per observation ID)
# apply_dec_bias_to_observations mutates the collection in-place,
# so we load a fresh unfiltered collection for the biased computation
# and leave obs_unfiltered completely unchanged.
##############################################################################################

# ── Define biases here ────────────────────────────────────────────────────────
BIAS_DICT_ARCSEC = {
    "689_nm0077": -0.2,   # Dec bias [arcsec]
}

print("\nLoading fresh observations for bias application...")
obs_for_bias, obs_settings_bias, set_ids_bias, _ = ObsFunc.LoadObservations(
    settings["obs"]["observations_folder_path"],
    system_of_bodies,
    file_names_loaded,
    Residual_filtering=False, epoch_filter_dict=None
)
assert set_ids_bias == set_ids_all, \
    f"set_ids mismatch for biased load: {set_ids_bias} vs {set_ids_all}"

obs_biased, applied_bias_rad = ObsFunc.apply_dec_bias_to_observations(
    obs_for_bias,
    obs_settings_bias,
    system_of_bodies,
    BIAS_DICT_ARCSEC
)
applied_bias_arcsec = {k: v / arcsec_to_rad for k, v in applied_bias_rad.items()}
print(f"  Applied biases [arcsec]: {applied_bias_arcsec}")

print("Computing SPICE nep097 residuals for biased observations...")
ra_spice_biased_flat, dec_spice_biased_flat = ObsFunc.Get_SPICE_residual_from_observations(
    obs_biased, set_ids_all, system_of_bodies,
    global_frame_orientation=global_frame_orientation
)
assert len(ra_spice_biased_flat) == n_total, \
    f"ra_spice_biased length mismatch: {len(ra_spice_biased_flat)} vs {n_total}"
print(f"  Biased residuals cross-check PASSED: {n_total} observations")

##############################################################################################
# STEP 5 — BUILD COMBINED TIME / MASK ARRAYS
# Rejection mask derived from rejected_epochs_per_set (obs-collection authority)
##############################################################################################

all_times_dt = []
all_mask_rej = []

for j, set_id in enumerate(set_ids_all):
    times_j    = np.array(obs_times_unfiltered[j])
    rej_set_j  = rejected_epochs_per_set[set_id]
    mask_rej_j = np.array([float(t) in rej_set_j for t in times_j])
    all_times_dt.extend([j2000_epoch + dt.timedelta(seconds=float(t)) for t in times_j])
    all_mask_rej.extend(mask_rej_j)

all_times_dt = np.array(all_times_dt)
all_mask_rej = np.array(all_mask_rej, dtype=bool)
all_mask_acc = ~all_mask_rej

##############################################################################################
# STEP 6 — FIGURES: separate PDF per observation file
##############################################################################################

print(f"\nSaving per-file residual PDFs to {out_dir_obs / 'per_file'}...")
for j, set_id in enumerate(set_ids_all):
    s, e = set_offsets[j], set_offsets[j + 1]

    times_j    = np.array(obs_times_unfiltered[j])
    times_dt_j = [j2000_epoch + dt.timedelta(seconds=float(t)) for t in times_j]

    ra_spice_j  = ra_spice_flat[s:e]
    dec_spice_j = dec_spice_flat[s:e]
    ra_tud_j    = ra_tud_all[s:e]
    dec_tud_j   = dec_tud_all[s:e]
    ra_prop_j   = ra_prop_all[s:e]
    dec_prop_j  = dec_prop_all[s:e]

    rej_set_j = rejected_epochs_per_set[set_id]
    mask_rej  = np.array([float(t) in rej_set_j for t in times_j])
    mask_acc  = ~mask_rej

    acc_dt = [times_dt_j[i] for i in np.where(mask_acc)[0]]
    rej_dt = [times_dt_j[i] for i in np.where(mask_rej)[0]]

    fig, (ax_ra, ax_dec) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f"{set_id}  |  n={e-s}  rejected={mask_rej.sum()}", fontsize=10)

    ax_ra.scatter(acc_dt, ra_tud_j[mask_acc],   s=6,  color='C0', alpha=0.8, label='tudatpy/SPICE (accepted)')
    ax_ra.scatter(acc_dt, ra_spice_j[mask_acc],  s=6,  color='C1', alpha=0.8, label='SPICE nep097 (accepted)')
    ax_ra.scatter(acc_dt, ra_prop_j[mask_acc],   s=6,  color='C2', alpha=0.8, label='Propagated (accepted)')
    if mask_rej.any():
        ax_ra.scatter(rej_dt, ra_tud_j[mask_rej],   s=30, color='C0', marker='x', zorder=5, label='tudatpy/SPICE (rej)')
        ax_ra.scatter(rej_dt, ra_spice_j[mask_rej],  s=30, color='C1', marker='x', zorder=5, label='SPICE nep097 (rej)')
        ax_ra.scatter(rej_dt, ra_prop_j[mask_rej],   s=30, color='C2', marker='x', zorder=5, label='Propagated (rej)')
    ax_ra.axhline(0, color='k', lw=0.5, ls='--')
    ax_ra.set_ylabel('RA residual [arcsec]')
    ax_ra.legend(fontsize=7, markerscale=2, ncol=3)

    ax_dec.scatter(acc_dt, dec_tud_j[mask_acc],  s=6,  color='C0', alpha=0.8, label='tudatpy/SPICE (accepted)')
    ax_dec.scatter(acc_dt, dec_spice_j[mask_acc], s=6,  color='C1', alpha=0.8, label='SPICE nep097 (accepted)')
    ax_dec.scatter(acc_dt, dec_prop_j[mask_acc],  s=6,  color='C2', alpha=0.8, label='Propagated (accepted)')
    if mask_rej.any():
        ax_dec.scatter(rej_dt, dec_tud_j[mask_rej],  s=30, color='C0', marker='x', zorder=5, label='tudatpy/SPICE (rej)')
        ax_dec.scatter(rej_dt, dec_spice_j[mask_rej], s=30, color='C1', marker='x', zorder=5, label='SPICE nep097 (rej)')
        ax_dec.scatter(rej_dt, dec_prop_j[mask_rej],  s=30, color='C2', marker='x', zorder=5, label='Propagated (rej)')
    ax_dec.axhline(0, color='k', lw=0.5, ls='--')
    ax_dec.set_ylabel('Dec residual [arcsec]')
    ax_dec.set_xlabel('Date')
    ax_dec.legend(fontsize=7, markerscale=2, ncol=3)

    ax_dec.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate()
    fig.tight_layout()

    safe_name = set_id.replace('/', '_').replace(' ', '_')
    fig.savefig(out_dir_obs / "per_file" / f"{safe_name}.pdf")
    plt.close(fig)

print(f"  Saved {len(set_ids_all)} per-file PDFs")

##############################################################################################
# COMBINED FIGURE — SPICE nep097 vs Propagated (all files)
##############################################################################################

fig_comb, (ax_ra_c, ax_dec_c) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig_comb.suptitle(
    f"All files  |  n={n_total}  rejected={all_mask_rej.sum()}  (SPICE nep097 vs Propagated)",
    fontsize=11
)

ax_ra_c.scatter(all_times_dt[all_mask_acc], ra_spice_flat[all_mask_acc],
                s=4, color='C0', alpha=0.6, label=f'SPICE nep097 accepted (n={all_mask_acc.sum()})')
ax_ra_c.scatter(all_times_dt[all_mask_acc], ra_prop_all[all_mask_acc],
                s=4, color='C2', alpha=0.6, label='Propagated accepted')
ax_ra_c.scatter(all_times_dt[all_mask_rej], ra_spice_flat[all_mask_rej],
                s=14, color='C0', alpha=0.9, marker='x', zorder=5,
                label=f'SPICE nep097 rejected (n={all_mask_rej.sum()})')
ax_ra_c.scatter(all_times_dt[all_mask_rej], ra_prop_all[all_mask_rej],
                s=14, color='C2', alpha=0.9, marker='x', zorder=5, label='Propagated rejected')
ax_ra_c.axhline(0, color='k', lw=0.5, ls='--')
ax_ra_c.set_ylabel('RA residual [arcsec]')
ax_ra_c.legend(fontsize=8, markerscale=2, ncol=2)

ax_dec_c.scatter(all_times_dt[all_mask_acc], dec_spice_flat[all_mask_acc],
                 s=4, color='C0', alpha=0.6, label='SPICE nep097 accepted')
ax_dec_c.scatter(all_times_dt[all_mask_acc], dec_prop_all[all_mask_acc],
                 s=4, color='C2', alpha=0.6, label='Propagated accepted')
ax_dec_c.scatter(all_times_dt[all_mask_rej], dec_spice_flat[all_mask_rej],
                 s=14, color='C0', alpha=0.9, marker='x', zorder=5, label='SPICE nep097 rejected')
ax_dec_c.scatter(all_times_dt[all_mask_rej], dec_prop_all[all_mask_rej],
                 s=14, color='C2', alpha=0.9, marker='x', zorder=5, label='Propagated rejected')
ax_dec_c.axhline(0, color='k', lw=0.5, ls='--')
ax_dec_c.set_ylabel('Dec residual [arcsec]')
ax_dec_c.set_xlabel('Date')
ax_dec_c.legend(fontsize=8, markerscale=2, ncol=2)

ax_dec_c.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig_comb.autofmt_xdate()
fig_comb.tight_layout()
fig_comb.savefig(out_dir_obs / "combined_residuals.pdf")
plt.close(fig_comb)
print(f"Saved: {out_dir_obs / 'combined_residuals.pdf'}")

##############################################################################################
# EXCLUDED FILES (in folder but NOT in file_names.json)
##############################################################################################

folder_path      = settings["obs"]["observations_folder_path"]
all_folder_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
excluded_files   = [f for f in all_folder_files if f not in file_names_loaded]
print(f"\nFiles in folder but not in file_names.json: {excluded_files}")

# Initialise as None — populated below only when excluded files exist
excl_data = None

if excluded_files:
    obs_excl_unfilt, obs_settings_excl, set_ids_excl, _ = ObsFunc.LoadObservations(
        folder_path, system_of_bodies, excluded_files,
        Residual_filtering=False, epoch_filter_dict=None
    )
    obs_excl_filt, _, _, epochs_rej_excl = ObsFunc.LoadObservations(
        folder_path, system_of_bodies, excluded_files,
        Residual_filtering=True, epoch_filter_dict=None
    )

    # Derive rejection from obs collections (not from the dict)
    excl_times_unfilt = obs_excl_unfilt.get_observation_times()
    excl_times_filt   = obs_excl_filt.get_observation_times()
    rejected_excl_per_set = {
        sid: set(float(t) for t in excl_times_unfilt[j]) - set(float(t) for t in excl_times_filt[j])
        for j, sid in enumerate(set_ids_excl)
    }

    obs_sim_excl = tud_obs_setup.observations_simulation_settings.create_observation_simulators(
        obs_settings_excl, system_of_bodies
    )
    tudatpy.estimation.observations.compute_residuals_and_dependent_variables(
        obs_excl_unfilt, obs_sim_excl, system_of_bodies
    )

    counts_excl  = [len(excl_times_unfilt[j]) for j in range(len(set_ids_excl))]
    offsets_excl = np.cumsum([0] + counts_excl)

    print("Computing SPICE nep097 residuals for excluded files...")
    ra_spice_excl, dec_spice_excl = ObsFunc.Get_SPICE_residual_from_observations(
        obs_excl_unfilt, set_ids_excl, system_of_bodies,
        global_frame_orientation=global_frame_orientation
    )

    all_times_excl_dt = []
    all_ra_excl       = []
    all_dec_excl      = []
    all_rej_excl      = []

    for j, sid in enumerate(set_ids_excl):
        s, e       = offsets_excl[j], offsets_excl[j + 1]
        times_j    = np.array(excl_times_unfilt[j])
        rej_set_j  = rejected_excl_per_set[sid]
        mask_rej_j = np.array([float(t) in rej_set_j for t in times_j])
        all_times_excl_dt.extend([j2000_epoch + dt.timedelta(seconds=float(t)) for t in times_j])
        all_ra_excl.extend(ra_spice_excl[s:e])
        all_dec_excl.extend(dec_spice_excl[s:e])
        all_rej_excl.extend(mask_rej_j)

    all_times_excl_dt = np.array(all_times_excl_dt)
    all_ra_excl       = np.array(all_ra_excl)
    all_dec_excl      = np.array(all_dec_excl)
    all_rej_excl      = np.array(all_rej_excl, dtype=bool)
    all_acc_excl      = ~all_rej_excl

    fig_excl, (ax_ra_excl, ax_dec_excl) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig_excl.suptitle(
        f"Files NOT in file_names.json: {', '.join(excluded_files)}\n"
        f"n={len(all_times_excl_dt)}  rejected={all_rej_excl.sum()}",
        fontsize=9
    )
    ax_ra_excl.scatter(all_times_excl_dt[all_acc_excl], all_ra_excl[all_acc_excl],
                       s=6, color='C0', alpha=0.8, label=f'accepted (n={all_acc_excl.sum()})')
    ax_ra_excl.scatter(all_times_excl_dt[all_rej_excl], all_ra_excl[all_rej_excl],
                       s=20, color='red', alpha=0.9, zorder=5, label=f'rejected (n={all_rej_excl.sum()})')
    ax_ra_excl.axhline(0, color='k', lw=0.5, ls='--')
    ax_ra_excl.set_ylabel('RA residual [arcsec]  (SPICE nep097)')
    ax_ra_excl.legend(fontsize=8, markerscale=2)

    ax_dec_excl.scatter(all_times_excl_dt[all_acc_excl], all_dec_excl[all_acc_excl],
                        s=6, color='C0', alpha=0.8, label=f'accepted (n={all_acc_excl.sum()})')
    ax_dec_excl.scatter(all_times_excl_dt[all_rej_excl], all_dec_excl[all_rej_excl],
                        s=20, color='red', alpha=0.9, zorder=5, label=f'rejected (n={all_rej_excl.sum()})')
    ax_dec_excl.axhline(0, color='k', lw=0.5, ls='--')
    ax_dec_excl.set_ylabel('Dec residual [arcsec]  (SPICE nep097)')
    ax_dec_excl.set_xlabel('Date')
    ax_dec_excl.legend(fontsize=8, markerscale=2)

    ax_dec_excl.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig_excl.autofmt_xdate()
    fig_excl.tight_layout()
    fig_excl.savefig(out_dir_obs / "excluded_files_residuals.pdf")
    plt.close(fig_excl)
    print(f"Saved: {out_dir_obs / 'excluded_files_residuals.pdf'}")

    # Store excluded files data for the .npy dict
    excl_data = {
        'files':            excluded_files,
        'set_ids':          set_ids_excl,
        'set_offsets':      offsets_excl,
        'times_dt':         all_times_excl_dt,
        'times_j2000':      np.array([(t - j2000_epoch).total_seconds()
                                      for t in all_times_excl_dt]),
        'ra_spice_arcsec':  all_ra_excl,
        'dec_spice_arcsec': all_dec_excl,
        'mask_rejected':    all_rej_excl,
        'mask_accepted':    all_acc_excl,
    }

##############################################################################################
# SAVE DATA DICT (.npy) for MatplotlibExport observations template
##############################################################################################

times_per_set_j2000 = [np.array(obs_times_unfiltered[j]) for j in range(len(set_ids_all))]

# Propagation time coverage (used in MatplotlibExport to mask valid epochs)
prop_t_min = float(state_history_array_prop[0,  0])
prop_t_max = float(state_history_array_prop[-1, 0])
print(f"\nPropagation covers J2000 [{prop_t_min:.0f}, {prop_t_max:.0f}] "
      f"≈ [{j2000_epoch + dt.timedelta(seconds=prop_t_min):%Y-%m-%d}, "
      f"{j2000_epoch + dt.timedelta(seconds=prop_t_max):%Y-%m-%d}]")

# Warn if observations fall outside the propagation arc
times_j2000_all = np.array([(t - j2000_epoch).total_seconds() for t in all_times_dt])
n_outside = int(np.sum((times_j2000_all < prop_t_min) | (times_j2000_all > prop_t_max)))
if n_outside:
    print(f"  WARNING: {n_outside}/{len(times_j2000_all)} observation epochs lie outside "
          f"the propagation arc — propagation residuals for those will be extrapolated garbage.")

obs_analysis_data = {
    # Combined (all files, unfiltered observation order matches set_offsets)
    "times_dt":             all_times_dt,
    "times_j2000":          times_j2000_all,

    # SPICE nep097 residuals [arcsec]
    "ra_spice_arcsec":      ra_spice_flat,
    "dec_spice_arcsec":     dec_spice_flat,

    # Tudatpy (SPICE environment) residuals [arcsec]
    "ra_tud_spice_arcsec":  ra_tud_all,
    "dec_tud_spice_arcsec": dec_tud_all,

    # Propagation-based residuals [arcsec]
    "ra_prop_arcsec":       ra_prop_all,
    "dec_prop_arcsec":      dec_prop_all,

    # Manual-bias corrected SPICE nep097 residuals [arcsec]
    "ra_spice_biased_arcsec":   ra_spice_biased_flat,
    "dec_spice_biased_arcsec":  dec_spice_biased_flat,
    # Bias definition used (for reference / figure annotation)
    "bias_dict_arcsec":         BIAS_DICT_ARCSEC,
    "bias_applied_arcsec":      applied_bias_arcsec,

    # Masks (derived from obs_filtered — obs-collection authority)
    "mask_rejected":        all_mask_rej,
    "mask_accepted":        all_mask_acc,

    # Per-set
    "set_ids":              set_ids_all,
    "set_offsets":          set_offsets,
    "times_per_set_j2000":  times_per_set_j2000,

    # Propagation time coverage [J2000 seconds]
    "prop_epoch_min":       prop_t_min,
    "prop_epoch_max":       prop_t_max,

    # Propagated trajectory
    "state_history_array_prop": state_history_array_prop,  # (N,7): col0=epoch [J2000 s]

    # Excluded files (None if no excluded files exist in the folder)
    "excluded":             excl_data,

    # Settings summary
    "settings_summary": {
        "start_epoch":            settings['prop']['start_epoch'],
        "end_epoch":              settings['prop']['end_epoch'],
        "initial_epoch":          settings['prop']['initial_epoch'],
        "fixed_step_size_s":      settings['prop']['fixed_step_size'],
        "neptune_rotation_model": settings['env']['Neptune_rot_model_type'],
        "observations_folder":    settings['obs']['observations_folder_path'],
        "n_files":                len(file_names_loaded),
    },
}

np.save(out_dir_obs / "obs_analysis_data.npy", obs_analysis_data)
print(f"Saved: {out_dir_obs / 'obs_analysis_data.npy'}")
