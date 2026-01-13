
import os
import yaml
import json
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from datetime import datetime
from pathlib import Path
import pandas as pd
# tudatpy imports
from tudatpy import math
from tudatpy import constants

from tudatpy.interface import spice
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
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

import MainPostprocessing as PostProc

matplotlib.use("PDF")  #tkagg



# Define temporal scope of the simulation - equal to the time JUICE will spend in orbit around Jupiter
simulation_start_epoch = DateTime(1963, 1,  1).epoch() #2006, 8,  27 1963, 3,  4  
simulation_end_epoch   = DateTime(2025, 1, 1).epoch()   #2006, 9, 2 2019, 10, 1

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


settings_env['Neptune_rot_model_type'] = 'IAU2015' 
    # Model Type for rotation model of Neptune:
    #  'simple_from_spice' - simple spice,
    #  'spice' - full spice,
    #  'IAU2015' - based on the IAU2015 paper
    
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

# --- Load names of data files
with open("file_names.json", "r") as f:
    file_names_loaded = json.load(f)

weights = pd.read_csv(
        "summary.txt", #Results/BetterFigs/AllModernObservations/PostProcessing/First/weights.txt
        sep="\t",
        index_col="id")

settings_obs = dict()
settings_obs["mode"] = ["pos"]
settings_obs["bodies"] = [("Triton", "Neptune")]                           # bodies to observe
settings_obs["cadence"] = 60*60*3 # Every 3 hours
settings_obs["type"] = "Simulated" # Simulated or Real observations

settings_obs["files"] = file_names_loaded             
settings_obs["observations_folder_path"] = "Observations/AllModernECLIPJ2000"  #RelativeObservations AllModernECLIPJ2000 AllModernJ2000

weights = weights.reset_index()

settings_obs["use_weights"] = False
settings_obs['std_weights'] = False
settings_obs["timeframe_weights"] = False
settings_obs["per_night_weights"] = False
settings_obs["per_night_weights_id"] = False 
settings_obs['per_night_weights_hybrid'] = False
settings_obs["weights"] = weights

#--------------------------------------------------------------------------------------------
# ESTIMATION SETTINGS 
#--------------------------------------------------------------------------------------------

settings_est = dict()
#settings_est['pseudo_observations_settings'] = pseudo_observations_settings
#settings_est['pseudo_observations'] = pseudo_observations


settings_est['est_parameters'] = ['initial_state','iau_rotation_model_pole','iau_rotation_model_pole_rate'] 
    #Possible settings: 
    # initial state - default
    # Rotation_Pole_Position_Neptune - fixed rotation pole position (only with simple rotational model !)
    # iau_rotation_model_pole - rotation pole position (alpha,delta) with IAU rotation model
    # iau_rotation_model_pole_rate - rotation pole rate  (alpha_dot, delta_dot) with IAU rotation model
    
    
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


# ObservationImplementation.main(settings,make_timestamped_folder("Results/EstimatedParametersSimulatedObservations/Test"))




runSimulationsBetter = True
if runSimulationsBetter == True:
    # Define all experiment variants here only once
    VARIANTS = {
        "initial_state_only": {
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/FullDuration/initial_state_only",
            'est_parameters': ['initial_state'] 
        }, 
        "rot_model_pos_only": {
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/FullDuration_BestIteration/rot_model_pos_only",
            'est_parameters': ['initial_state','iau_rotation_model_pole'] 
        },    
        "rot_model_rate_only": {
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/FullDuration_BestIteration/rot_model_rate_only",
            'est_parameters': ['initial_state','iau_rotation_model_pole_rate'] 
        },    
        "rot_model_full": {
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/FullDuration_BestIteration/rot_model_full",
            'est_parameters': ['initial_state','iau_rotation_model_pole','iau_rotation_model_pole_rate'] 
        },
        "GM_Neptune": {
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/FullDuration_BestIteration/GM_Neptune",
            'est_parameters': ['initial_state','GM_Neptune'] 
        },     
        "GM_Triton": {
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/FullDuration_BestIteration/GM_Triton",
            'est_parameters': ['initial_state','GM_Triton'] 
        },
        "GM_Neptune_Triton": {
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/FullDuration_BestIteration/GM_Neptune_Triton",
            'est_parameters': ['initial_state','GM_Neptune','GM_Triton'] 
        },
        "spherical_harmonics": {
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/FullDuration_BestIteration/spherical_harmonics",
            'est_parameters': ['initial_state','spherical_harmonics'] 
        },
        "GM_spherical_harmonics": {
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/FullDuration/GM_spherical_harmonics",
            'est_parameters': ['initial_state','GM_Neptune','GM_Triton','spherical_harmonics'] 
        },
        "spherical_harmonics_pole_full":{
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/FullDuration_BestIteration/spherical_harmonics_pole_full",
            'est_parameters': ['initial_state','iau_rotation_model_pole','iau_rotation_model_pole_rate','spherical_harmonics']  
        },
        "all": {
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/FullDuration/all",
            'est_parameters': ['initial_state','iau_rotation_model_pole','iau_rotation_model_pole_rate','GM_Neptune','GM_Triton','spherical_harmonics'] 
        },                                                  
    }


    # # Load estimations
    # estimations = {
    #     name: PostProc.load_npy_files(cfg["estimation_path"])
    #     for name, cfg in VARIANTS.items()
    # }

    out_dir = make_timestamped_folder("Results/EstimatedParametersSimulatedObservations")
    
    runSim = False
    if runSim == True:
        results = {}
        for name, content in VARIANTS.items():
            print("######################################")
            print("Running Sim ",name)
            print("######################################")
            
            out_dir_current = out_dir / name
            out_dir_current.mkdir(parents=True, exist_ok=True)
            
            settings_est['est_parameters'] = content['est_parameters'] 
            

            ObservationImplementation.main(settings,out_dir_current)


    

    # Load simulations
    simulations = {
        name: PostProc.load_npy_files(cfg["simulation_path"])
        for name, cfg in VARIANTS.items()
    }
    
    for name, cfg in VARIANTS.items():
        simulations[name]["est_parameters"] = cfg["est_parameters"]

print("#######################################################################################################")
print("CREATE SPICE STATES")
print("#######################################################################################################")

##############################################################################################
# LOAD SPICE KERNELS
##############################################################################################
kernel_folder = "Kernels/"
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


# GET SPICE Results (any sim should work)
epochs = simulations['initial_state_only']['state_history_array'][:,0]
states_SPICE = np.array([
    spice.get_body_cartesian_state_at_epoch(
        target_body_name="Triton",
        observer_body_name="Neptune",
        reference_frame_name=global_frame_orientation,
        aberration_corrections="NONE",
        ephemeris_time=epoch
    )
    for epoch in epochs
])


   
time_column = simulations['initial_state_only']['state_history_array'][:,[0]]  # keep it 2D 
states_SPICE_with_time = np.hstack((time_column, states_SPICE))

print("#######################################################################################################")
print("CREATE RSW STATES FIGS")
print("#######################################################################################################")

diff_SPICE_RSW = {}
rms_SPICE = {}
rms_Norm = {}
diff_wrt_norm_RSW = {}

states_SPICE_RSW = ProcessingUtils.rotate_inertial_3_to_rsw(time_column, states_SPICE[:,0:3], states_SPICE_with_time)

state_history_array_norm = simulations['initial_state_only']['state_history_array']
states_norm_RSW = ProcessingUtils.rotate_inertial_3_to_rsw(time_column, state_history_array_norm[:,1:4], state_history_array_norm)

RSW_RMS_PLOTS = True
if RSW_RMS_PLOTS == True:
    for key in simulations.keys():
        state_history_array = simulations[key]['state_history_array']

        
        states_sim_RSW_SPICE = ProcessingUtils.rotate_inertial_3_to_rsw(time_column, state_history_array[:,1:4], states_SPICE_with_time)
        states_sim_RSW_norm = ProcessingUtils.rotate_inertial_3_to_rsw(time_column, state_history_array[:,1:4], state_history_array_norm)
        #Diff wrt to SPICE
        # #---------------------------------------------------------------------------------------
        diff = (states_SPICE_RSW - states_sim_RSW_SPICE)/1e3
        diff_SPICE_RSW[key] = diff 
        fig_RSW = FigUtils.Residuals_RSW(diff, time_column,type="difference",title=("RSW Difference SPICE - " + key))   
        fig_RSW.savefig(out_dir / ("RSW Diff SPICE - " + key + ".pdf"))

        # unweighted scalar RMS (what Tudat prints)
        rms_SPICE[key] = np.sqrt(np.mean(diff**2)) 

        #Diff wrt to norm
        # #---------------------------------------------------------------------------------------
        diff = (states_norm_RSW- states_sim_RSW_norm)/1e3
        diff_wrt_norm_RSW[key] = diff 
        fig_RSW = FigUtils.Residuals_RSW(diff, time_column,type="difference",title=("RSW Difference Norm - " + key))   
        fig_RSW.savefig(out_dir / ("RSW Diff Norm_ " + key + ".pdf"))

        rms_Norm[key] = np.sqrt(np.mean(diff**2)) 


    # #---------------------------------------------------------------------------------------
    # rms_SPICE.pop('spherical_harmonics_pole_full') 
    # rms_Norm.pop('spherical_harmonics_pole_full')


    fig_RMS_SPICE = FigUtils.plot_rms_comparison(rms_SPICE)
    fig_RMS_SPICE.savefig(out_dir / "RMS_SPICE.pdf")

    # #---------------------------------------------------------------------------------------

print("#######################################################################################################")
print("CREATE CORRELATION PLOTS & PARAMETER PLOTS")
print("#######################################################################################################")

best_parameter_update = {}
rms_residuals = {}
for key in simulations.keys():
        correlations = simulations[key]['correlations']
        parameter_history = simulations[key]['parameter_history']
        best_iteration = simulations[key]['best_iteration']
       
        best_parameter_update[key] = parameter_history[:,0] - parameter_history[:,best_iteration]

        est_parameters = simulations[key]['est_parameters']
        
        rms_residuals[key] = np.sqrt(np.mean((simulations[key]['residuals_j2000']/1e3)**2)) 
        # print("Plotting figs for: ",key)        
        # fig = FigUtils.plot_correlation_matrix(correlations, est_parameters)
        # fig.savefig(out_dir / 'correlations.pdf')

        # fig1 = FigUtils.plot_parameter_updates(best_parameter_update[key],  est_parameters)
        # fig1.savefig(out_dir / "parameter_update.pdf")
        
        # fig2 = FigUtils.plot_parameter_history(parameter_history, est_parameters, best_iteration=best_iteration)
        # fig2.savefig(out_dir / "parameter_history.pdf")
        # print("Next")


def plot_state_magnitude_updates(best_parameter_update, 
                                 title="Position and Velocity Magnitude Updates",
                                 figsize=(14, 6)):
    """
    Plot magnitude of position and velocity updates (first 6 parameters).
    
    Parameters:
    -----------
    best_parameter_update : dict
        Dictionary with keys as scenario names and values as parameter update arrays.
        First 6 elements are assumed to be [x, y, z, vx, vy, vz] in meters and m/s.
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    fig : matplotlib figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    keys = list(best_parameter_update.keys())
    
    # Extract position and velocity updates and convert to km and km/s
    delta_r_magnitudes = []
    delta_v_magnitudes = []
    
    for key in keys:
        updates = best_parameter_update[key]
        
        # Position: x, y, z (convert from m to km)
        x, y, z = updates[0] / 1000, updates[1] / 1000, updates[2] / 1000
        delta_r = np.sqrt(x**2 + y**2 + z**2)
        delta_r_magnitudes.append(delta_r)
        
        # Velocity: vx, vy, vz (convert from m/s to km/s)
        vx, vy, vz = updates[3] / 1000, updates[4] / 1000, updates[5] / 1000
        delta_v = np.sqrt(vx**2 + vy**2 + vz**2)
        delta_v_magnitudes.append(delta_v)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    x = np.arange(len(keys))
    
    # Plot 1: Delta R magnitude
    axes[0].plot(x, delta_r_magnitudes, marker='o', linewidth=2, markersize=8, 
                color='steelblue', alpha=0.8)
    axes[0].set_ylabel('|Δr| [km]', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Estimation Scenario', fontsize=12, fontweight='bold')
    axes[0].set_title('Position Magnitude Update', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(keys, rotation=45, ha='right', fontsize=9)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, v in enumerate(delta_r_magnitudes):
        label = f'{v:.3f}'
        axes[0].text(i, v, label, ha='center', va='bottom', 
                    fontsize=8, fontweight='bold')
    
    # Plot 2: Delta V magnitude
    axes[1].plot(x, delta_v_magnitudes, marker='s', linewidth=2, markersize=8, 
                color='coral', alpha=0.8)
    axes[1].set_ylabel('|Δv| [km/s]', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Estimation Scenario', fontsize=12, fontweight='bold')
    axes[1].set_title('Velocity Magnitude Update', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(keys, rotation=45, ha='right', fontsize=9)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, v in enumerate(delta_v_magnitudes):
        label = f'{v:.6f}'
        axes[1].text(i, v, label, ha='center', va='bottom', 
                    fontsize=8, fontweight='bold')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_state_component_updates(best_parameter_update,
                                  title="Individual State Component Updates",
                                  figsize=(14, 12)):
    """
    Plot individual components of position and velocity updates (6 subplots).
    
    Parameters:
    -----------
    best_parameter_update : dict
        Dictionary with keys as scenario names and values as parameter update arrays.
        First 6 elements are assumed to be [x, y, z, vx, vy, vz] in meters and m/s.
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    fig : matplotlib figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    keys = list(best_parameter_update.keys())
    
    # Extract all components and convert to km and km/s
    x_vals = [best_parameter_update[k][0] / 1000 for k in keys]  # m -> km
    y_vals = [best_parameter_update[k][1] / 1000 for k in keys]  # m -> km
    z_vals = [best_parameter_update[k][2] / 1000 for k in keys]  # m -> km
    vx_vals = [best_parameter_update[k][3] / 1000 for k in keys]  # m/s -> km/s
    vy_vals = [best_parameter_update[k][4] / 1000 for k in keys]  # m/s -> km/s
    vz_vals = [best_parameter_update[k][5] / 1000 for k in keys]  # m/s -> km/s
    
    # Create figure with 3x2 subplots
    fig, axes = plt.subplots(3, 2, figsize=figsize, sharex=True)
    
    x = np.arange(len(keys))
    
    # Component data and labels
    components = [
        (x_vals, 'Δx [km]', 'X Position Update'),
        (y_vals, 'Δy [km]', 'Y Position Update'),
        (z_vals, 'Δz [km]', 'Z Position Update'),
        (vx_vals, 'Δvx [km/s]', 'X Velocity Update'),
        (vy_vals, 'Δvy [km/s]', 'Y Velocity Update'),
        (vz_vals, 'Δvz [km/s]', 'Z Velocity Update')
    ]
    
    colors = ['steelblue', 'steelblue', 'steelblue', 'coral', 'coral', 'coral']
    markers = ['o', 's', '^', 'o', 's', '^']
    
    # Plot each component
    for idx, (ax, (vals, ylabel, subplot_title), color, marker) in enumerate(zip(axes.flat, components, colors, markers)):
        ax.plot(x, vals, marker=marker, linewidth=2, markersize=8, 
               color=color, alpha=0.8)
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(subplot_title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        # Add value labels
        for i, v in enumerate(vals):
            # Use appropriate precision based on magnitude
            if idx < 3:  # Position components (km)
                label = f'{v:.3f}'
            else:  # Velocity components (km/s)
                label = f'{v:.6f}'
            
            va = 'bottom' if v >= 0 else 'top'
            ax.text(i, v, label, ha='center', va=va, 
                   fontsize=7, fontweight='bold')
    
    # Set x-labels only on bottom row
    for ax in axes[2, :]:
        ax.set_xlabel('Estimation Scenario', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=45, ha='right', fontsize=9)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_state_updates_combined(best_parameter_update):
    """
    Create both plots: magnitude updates and component updates.
    
    Parameters:
    -----------
    best_parameter_update : dict
        Dictionary with keys as scenario names and values as parameter update arrays
    
    Returns:
    --------
    tuple : (fig_magnitude, fig_components)
    """
    fig1 = plot_state_magnitude_updates(best_parameter_update)
    fig2 = plot_state_component_updates(best_parameter_update)
    
    return fig1, fig2


# Or use the combined function
fig_mag, fig_comp = plot_state_updates_combined(best_parameter_update)
fig_mag.savefig(out_dir / "Best_Parameter_Update_magnitude.pdf")
fig_comp.savefig(out_dir / "Best_Parameter_Update_components.pdf")

rms_residuals.pop('spherical_harmonics_pole_full') 
fig = FigUtils.plot_rms_comparison(rms_residuals)
fig.savefig(out_dir / "RMS_Residuals.pdf")



from matplotlib.backends.backend_pdf import PdfPages


# Save to a single multi-page PDF
with PdfPages(out_dir / "Combined_Analysis.pdf") as pdf:
    pdf.savefig(fig_mag)
    pdf.savefig(fig_RMS_SPICE)

print("End.")