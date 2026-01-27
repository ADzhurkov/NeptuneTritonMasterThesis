
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

matplotlib.use("PDF")  #tkagg



# Define temporal scope of the simulation - equal to the time JUICE will spend in orbit around Jupiter
simulation_start_epoch = DateTime(1989, 1,  1).epoch() #2006, 8,  27 1963, 3,  4  
simulation_end_epoch   = DateTime(2003, 1, 1).epoch()   #2025, 1, 1

simulation_initial_epoch = DateTime(2000, 10, 1).epoch() #2006, 10, 1
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

weights = pd.read_csv(
        "Results/PoleEstimationRealObservations/LoopTest2/initial_state_only/0/summary.txt", #Results/BetterFigs/AllModernObservations/PostProcessing/First/weights.txt
        sep="\t",
        index_col="id")

settings_obs = dict()
settings_obs["mode"] = ["pos"]
settings_obs["bodies"] = [("Triton", "Neptune")]                           # bodies to observe
settings_obs["cadence"] = 60*60*3 # Every 3 hours
settings_obs["type"] = "Real" # Simulated or Real observations

#TEST FILE CHANGE
file_names_loaded = ['Triton_874_nm0003.csv','Triton_874_nm0004.csv','Triton_874_nm0013.csv','Triton_689_nm0007.csv']
settings_obs["files"] = file_names_loaded             
settings_obs["observations_folder_path"] = "Observations/AllModernECLIPJ2000"  #RelativeObservations AllModernECLIPJ2000 AllModernJ2000

weights = weights.reset_index()

settings_obs["use_weights"] = False
settings_obs["ra_dec_independent_weights"] = False
settings_obs["timeframe_weights"] = False
settings_obs["weights"] = weights


settings_obs["residual_filtering"] = True
settings_obs["epoch_filter_dict"] = None 


#Make sure all other weight types are off
settings_obs['std_weights'] = False
settings_obs["per_night_weights"] = False
settings_obs["per_night_weights_id"] = False 
settings_obs['per_night_weights_hybrid'] = False


#--------------------------------------------------------------------------------------------
# ESTIMATION SETTINGS 
#--------------------------------------------------------------------------------------------

settings_est = dict()
#settings_est['pseudo_observations_settings'] = pseudo_observations_settings
#settings_est['pseudo_observations'] = pseudo_observations


settings_est['est_parameters'] = ['initial_state','iau_rotation_model_pole','iau_rotation_model_pole_rate'] 
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
        # #Pole IAU2015
        # #---------------------------------------------------------------------------------------------------------------------------------------
        # "initial_state_only": {
        #     "simulation_path": "Results/PoleEstimationRealObservations/Hybrid_Weights/initial_state_only",
        #     'est_parameters': ['initial_state'],
        #     'Neptune_rot_model_type': ['IAU'],
        # }, 
        # "initial_state_only_Jacobson2009": {
        #     "simulation_path": "Results/PoleEstimationRealObservations/Hybrid_Weights/initial_state_only",
        #     'est_parameters': ['initial_state'],
        #     'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        # }, 
        # "pole_pos": {
        #     "simulation_path": "Results/PoleEstimationRealObservations/Hybrid_Weights/pole_pos",
        #     'est_parameters': ['initial_state','iau_rotation_model_pole'],
        #     'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        # },
        # "pole_libration_amplitude":{
        #     "simulation_path": "Results/PoleEstimationRealObservations/Hybrid_Weights/pole_libration_amplitude",
        #     'est_parameters': ['initial_state','iau_rotation_model_pole_librations','pole_librations_deg2'],
        #     'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        # },
        # "pole_pos_and_libration_amplitude":{
        #     "simulation_path": "Results/PoleEstimationRealObservations/Hybrid_Weights/pole_pos_and_libration_amplitude",
        #     'est_parameters': ['initial_state','iau_rotation_model_pole','iau_rotation_model_pole_librations','pole_librations_deg2'],
        #     'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        # },    
        # "rot_model_full": {
        #     "simulation_path": "Results/PoleEstimationRealObservations/FullDuration/rot_model_full",
        #     'est_parameters': ['initial_state','iau_rotation_model_pole','iau_rotation_model_pole_rate']
        # },
        # "pole_full_and_libration_amplitude":{
        #     "simulation_path": "Results/PoleEstimationRealObservations/FullDuration/pole_full_and_libration_amplitude",
        #     'est_parameters': ['initial_state','iau_rotation_model_pole','iau_rotation_model_pole_rate','iau_rotation_model_pole_librations'],
        # },
                                                
    }


    # Define 
    VARIANTS = {
        "initial_state_only": {
            "simulation_path": "Results/PoleEstimationRealObservations/Loop/initial_state_only",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': ['Pole_Model_Jacobson2009'],
            'max_estimations': 2,
        }, 

    }


    out_dir = make_timestamped_folder("Results/PoleEstimationRealObservations")
    
  
    #-----------------#-----------------#-----------------#-----------------#-----------------#-----------------
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
            #settings_obs["weights"] = weights

            #Select different Pole Model or return to default
            if 'Neptune_rot_model_type' in content.keys():
                settings['env']['Neptune_rot_model_type'] = content['Neptune_rot_model_type']
            else:
                settings['env']['Neptune_rot_model_type'] ='IAU2015'
            
            #Run estimation
            if 'max_estimations' in content.keys():
                for i in range(content['max_estimations']):
                    
                    out_dir_current_est = out_dir_current / str(i)
                    out_dir_current_est.mkdir(parents=True,exist_ok=True)

                    if i==0:
                        settings['obs']["use_weights"] = False
                        settings['obs']["timeframe_weights"] = False
                        settings['obs']['ra_dec_independent_weights'] = False
                        settings['obs']["residual_filtering"] = True
                        #residuals_rejected_epochs_file = "Results/PoleEstimationRealObservations/LoopTest2/initial_state_only/0/residuals_rejected_epochs.json"
                        # with open(residuals_rejected_epochs_file, 'r') as f:
                        #     epochs_rejected = json.load(f)
                        # settings_obs["epoch_filter_dict"] = epochs_rejected


                        print("######################################")
                        print("Running Sim ",name,i)
                        print("######################################")
            
                    #Run estimation ~(30mins)
                    estimation_output,observations = ObservationImplementation.main(settings,out_dir_current_est)

                    #----------------------------------------------------------------------------------------------
                    #After first estimation load the rejected epochs to use for next iterations
                    if i==0:
                        residuals_rejected_epochs_file = out_dir_current_est / "residuals_rejected_epochs.json"
                        with open(residuals_rejected_epochs_file, 'r') as f:
                            epochs_rejected = json.load(f)
                        settings_obs["epoch_filter_dict"] = epochs_rejected

                        #Turn off residual filtering for next iterations
                        settings_obs["residual_filtering"] = False
                    #----------------------------------------------------------------------------------------------
                    #
                    current_estimation_sim = PostProc.load_npy_files(out_dir_current_est)
                    
                    #Create weights
                    summary = PostProc.main(file_names_loaded,out_dir_current_est)
                    
                    # Save current weights df just in case
                    summary.to_csv(out_dir_current_est / "summary.txt", sep="\t", float_format="%.8e")

                    settings['obs']["use_weights"] = True
                    settings['obs']["timeframe_weights"] = True
                    settings['obs']['ra_dec_independent_weights'] = True
                    
                    settings['obs']["weights"] = summary

                    # Set initial state of next iteration to the best solution of current estimation
                    best_iteration = current_estimation_sim['best_iteration']
                    settings['prop']['initial_state'] = current_estimation_sim['parameter_history'][:,best_iteration]
                    #----------------------------------------------------------------------------------------------
    

    # # Load simulations
    # simulations = {
    #     name: PostProc.load_npy_files(cfg["simulation_path"])
    #     for name, cfg in VARIANTS.items()
    # }


    #Dict[Dict[]] where each specific sim has sub dicts for each iteration
    #simulations = {}
    # for name,cfg in VARIANTS.items():
    #     if 'max_estimations' in cfg:
    #         for i in cfg['max_estimations']:
                
    #simulations = {}

    # from collections import defaultdict

    # simulations = defaultdict(dict)
    # simulations['initial_state_only']["0"] = PostProc.load_npy_files(
    #     "Results/PoleEstimationRealObservations/TestCaseLoop/initial_state_only/0")

    # summary = pd.read_csv(
    #     "Results/PoleEstimationRealObservations/TestCaseLoop/initial_state_only/0/summary.txt", #
    #     sep="\t",
    #     index_col="id")

    # summary = summary.reset_index()

    # # simulations['initial_state_only']["0"] = PostProc.load_npy_files(
    # #         "Results/PoleEstimationRealObservations/TestCaseLoop/initial_state_only")


    # # for name, cfg in VARIANTS.items():
    # #     simulations[name]["est_parameters"] = cfg["est_parameters"]



#Test
#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------
print("#######################################################################################################")
print("TEST BETTER WEIGHT ASSIGNMENT")
print("#######################################################################################################")

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

body_settings,system_of_bodies = PropFuncs.Create_Env(settings['env'])


observations,observations_settings,observation_set_ids, epochs_rejected = ObsFunc.LoadObservations(
        settings["obs"]["observations_folder_path"],
        system_of_bodies,
        settings['obs']["files"],
        Residual_filtering = settings["obs"]["residual_filtering"])


#estimation_output,observations = ObservationImplementation.main(settings,out_dir)

simulation = PostProc.load_npy_files(
            'Results/PoleEstimationRealObservations/LoopTest2/initial_state_only/0')

residuals = simulation['residual_history_arcseconds'][-1]



#---------------------------------------------------------------------------------------------------------------------------

def compute_and_assign_weights(residuals: np.ndarray, 
                                observations,
                                gap_threshold_hours: float = 4.0,
                                min_obs_per_frame: int = 1) -> Tuple:
    """
    Compute and assign weights to observation sets based on residuals.
    
    Parameters:
    -----------
    residuals : np.ndarray
        Array of shape [n, 3] containing [time, RA_residual, DEC_residual]
    observations : ObservationCollection
        Collection of all observations
    gap_threshold_hours : float
        Maximum time gap (hours) to keep observations in same timeframe
    min_obs_per_frame : int
        Minimum observations per timeframe before allowing a break
        
    Returns:
    --------
    observations : ObservationCollection
        Updated observation collection with assigned weights
    weights_df : pd.DataFrame
        DataFrame containing weight information for all observations
    """
    
    # Extract observation sets
    sets = observations.sorted_observation_sets
    ObservableType = list(sets.keys())[0]
    
    all_observations_sets = []
    for observable_type, inner_dict in sets.items():
        for link_end_id, observation_list in inner_dict.items():
            all_observations_sets.extend(observation_list)
    
    # Prepare data structures for results
    weights_data = []
    current_idx = 0
    
    # Process each observation set
    for set_idx, obs_set in enumerate(all_observations_sets):
        # Get reference point ID for this set
        ref_point_id = obs_set.link_definition.link_end_id(links.receiver).reference_point
        
        # Determine number of observations in this set
        n_obs = np.shape(obs_set.observation_times)[0]
        
        obs_times_obs = obs_set.observation_times
        obs_times_test = [t.to_float() for t in obs_times_obs]

        # Extract residuals for this observation set
        set_residuals = residuals[current_idx:current_idx + n_obs, :]
        times = set_residuals[:, 0]
        ra_residuals = set_residuals[:, 1]
        dec_residuals = set_residuals[:, 2]
        
      
        # Test if times are consistent:
        diff = obs_times_test - times
        assert np.allclose(obs_times_test, times), \
            f"Times don't match! Max difference: {np.max(np.abs(diff))}"


        # 1. Compute global RMSE weights (constant for entire set)
        ra_rmse_global = np.sqrt(np.mean(ra_residuals**2))
        dec_rmse_global = np.sqrt(np.mean(dec_residuals**2))
        
        weight_ra_global = 1.0 / (ra_rmse_global**2) if ra_rmse_global > 0 else 1.0
        weight_dec_global = 1.0 / (dec_rmse_global**2) if dec_rmse_global > 0 else 1.0
        
        # 2. Split into timeframes based on gaps
        timeframes = split_observations_into_timeframes(
            times, 
            gap_threshold_hours=gap_threshold_hours,
            min_obs_per_frame=min_obs_per_frame
        )
        
        n_timeframes = timeframes.max() + 1
        
        # 3. Compute local (per-timeframe) weights
        weight_ra_local = np.zeros(n_obs)
        weight_dec_local = np.zeros(n_obs)
        
        for frame_idx in range(n_timeframes):
            # Find observations in this timeframe
            mask = (timeframes == frame_idx)
            
            if np.sum(mask) > 0:
                # Compute RMSE for this timeframe
                ra_rmse_frame = np.sqrt(np.mean(ra_residuals[mask]**2))
                dec_rmse_frame = np.sqrt(np.mean(dec_residuals[mask]**2))
                
                # Assign weights (inverse variance)
                weight_ra_local[mask] = 1.0 / (ra_rmse_frame**2) if ra_rmse_frame > 0 else 1.0
                weight_dec_local[mask] = 1.0 / (dec_rmse_frame**2) if dec_rmse_frame > 0 else 1.0
        
        # 4. Compute hybrid weights (geometric mean)
        # You can adjust this combination strategy
        weight_ra_hybrid = np.sqrt(weight_ra_global * weight_ra_local)
        weight_dec_hybrid = np.sqrt(weight_dec_global * weight_dec_local)
        
        # 5. Format weights for set_tabulated_weights (interleaved RA, DEC)
        weights_array = np.zeros(2 * n_obs)
        weights_array[0::2] = weight_ra_hybrid   # RA weights at even indices
        weights_array[1::2] = weight_dec_hybrid  # DEC weights at odd indices
        
        # 6. Assign weights to observation set
        obs_set.set_tabulated_weights(weights_array)
        
        # 7. Store data for DataFrame
        for obs_idx in range(n_obs):
            weights_data.append({
                'set_index': set_idx,
                'ref_point_id': ref_point_id,
                'obs_index': obs_idx,
                'global_obs_index': current_idx + obs_idx,
                'timeframe': timeframes[obs_idx],
                'time': times[obs_idx],
                'ra_residual': ra_residuals[obs_idx],
                'dec_residual': dec_residuals[obs_idx],
                'ra_rmse_id': ra_rmse_global,
                'dec_rmse_id': dec_rmse_global,
                'weight_ra_id': weight_ra_global,
                'weight_dec_id': weight_dec_global,
                'weight_ra_local': weight_ra_local[obs_idx],
                'weight_dec_local': weight_dec_local[obs_idx],
                'weight_ra_hybrid': weight_ra_hybrid[obs_idx],
                'weight_dec_hybrid': weight_dec_hybrid[obs_idx]
            })
        
        # Update index for next set
        current_idx += n_obs
    
    # Create DataFrame
    weights_df = pd.DataFrame(weights_data)
    
    return observations, weights_df


def split_observations_into_timeframes(times: np.ndarray, 
                                       gap_threshold_hours: float = 4.0,
                                       min_obs_per_frame: int = 1) -> np.ndarray:
    """
    Split observations into timeframes based on time gaps.
    
    Parameters:
    -----------
    times : np.ndarray
        Array of observation times (assumed sorted)
    gap_threshold_hours : float
        Maximum gap between observations in same timeframe (hours)
    min_obs_per_frame : int
        Minimum observations required before allowing a frame break
        
    Returns:
    --------
    timeframes : np.ndarray
        Array of timeframe indices for each observation
    """
    n_obs = len(times)
    gap_threshold_seconds = gap_threshold_hours * 3600.0
    
    timeframes = np.zeros(n_obs, dtype=int)
    current_frame = 0
    obs_in_current_frame = 1
    
    for i in range(1, n_obs):
        time_gap = times[i] - times[i-1]
        
        # Start new frame if gap exceeds threshold AND minimum obs requirement met
        if time_gap >= gap_threshold_seconds and obs_in_current_frame >= min_obs_per_frame:
            current_frame += 1
            obs_in_current_frame = 1
        else:
            obs_in_current_frame += 1
            
        timeframes[i] = current_frame
    
    return timeframes


# Example usage:
observations_weighted, weights_info = compute_and_assign_weights(
    residuals=residuals,
    observations=observations,
    gap_threshold_hours=4.0,
    min_obs_per_frame=1
)

# Save weights to CSV
weights_info.to_csv('observation_weights.csv', index=False)

# Analyze timeframes per observation set
print(weights_info.groupby(['set_index', 'timeframe']).size())

print("Test")



def epoch_to_datetime(epoch_seconds):
    """Convert seconds since J2000 to datetime"""
    j2000 = datetime(2000, 1, 1, 12, 0, 0)  # J2000 epoch: Jan 1, 2000, 12:00:00
    return j2000 + timedelta(seconds=epoch_seconds)

def get_color_map(df, id_col='ref_point_id'):
    """Create a color map for each unique id"""
    unique_ids = df[id_col].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_ids)))
    color_map = dict(zip(unique_ids, colors))
    return color_map

def prepare_weights_summary(weights_info):
    """
    Aggregate weights_info by (ref_point_id, timeframe) to create summary statistics
    similar to the timeframe summary structure
    """
    # Group by ref_point_id and timeframe
    summary = weights_info.groupby(['ref_point_id', 'timeframe']).agg({
        'time': ['min', 'max', 'count'],
        'weight_ra_hybrid': 'mean',
        'weight_dec_hybrid': 'mean',
        'weight_ra_local': 'mean',
        'weight_dec_local': 'mean',
        'ra_residual': lambda x: np.sqrt(np.mean(x**2)),  # RMSE
        'dec_residual': lambda x: np.sqrt(np.mean(x**2))  # RMSE
    }).reset_index()
    
    # Flatten column names
    summary.columns = ['id', 'timeframe', 'start_sec', 'end_sec', 'n_obs',
                       'mean_weight_ra_hybrid', 'mean_weight_dec_hybrid',
                       'mean_weight_ra_local', 'mean_weight_dec_local',
                       'rmse_ra', 'rmse_dec']
    
    # Calculate weighted RMSE (scaled by weights)
    summary['weight_rmse_ra_tf_id_scaled'] = summary['rmse_ra'] * summary['mean_weight_ra_hybrid']
    summary['weight_rmse_dec_tf_id_scaled'] = summary['rmse_dec'] * summary['mean_weight_dec_hybrid']
    summary['mean_weight_rmse_tf_id_scaled'] = (summary['weight_rmse_ra_tf_id_scaled'] + 
                                                  summary['weight_rmse_dec_tf_id_scaled']) / 2
    
    # Calculate midpoint
    summary['midpoint_sec'] = (summary['start_sec'] + summary['end_sec']) / 2
    summary['midpoint_datetime'] = summary['midpoint_sec'].apply(epoch_to_datetime)
    
    return summary

def plot_weights_all_data(weights_info, save_dir=None):
    """Create all plots for the entire weights_info dataframe"""
    
    # Prepare summary
    df = prepare_weights_summary(weights_info)
    
    # Get color map
    color_map = get_color_map(df, id_col='id')
    
    # Plot 1: n_obs vs time
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    for id_val in df['id'].unique():
        mask = df['id'] == id_val
        ax1.scatter(df[mask]['midpoint_datetime'], df[mask]['n_obs'], 
                   color=color_map[id_val], label=id_val, alpha=0.6)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Number of Observations')
    ax1.set_title('Number of Observations vs Time')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/n_obs_vs_time_all.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: mean_weight_rmse_tf_id_scaled vs time
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    for id_val in df['id'].unique():
        mask = df['id'] == id_val
        ax2.scatter(df[mask]['midpoint_datetime'], df[mask]['mean_weight_rmse_tf_id_scaled'], 
                   color=color_map[id_val], label=id_val, alpha=0.6)
    ax2.set_yscale('log')               
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Mean Weighted RMSE (scaled)')
    ax2.set_title('Mean Weighted RMSE vs Time')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/rmse_vs_time_all.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 3: mean_weight_rmse_tf_id_scaled vs n_obs
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    for id_val in df['id'].unique():
        mask = df['id'] == id_val
        ax3.scatter(df[mask]['n_obs'], df[mask]['mean_weight_rmse_tf_id_scaled'], 
                   color=color_map[id_val], label=id_val, alpha=0.6)
    ax3.set_yscale('log')
    ax3.set_xlabel('Number of Observations')
    ax3.set_ylabel('Mean Weighted RMSE (scaled)')
    ax3.set_title('Mean Weighted RMSE vs Number of Observations')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/rmse_vs_nobs_all.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 4: RA and Dec RMSE vs time (subplots)
    fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # RA subplot
    for id_val in df['id'].unique():
        mask = df['id'] == id_val
        ax4a.scatter(df[mask]['midpoint_datetime'], df[mask]['weight_rmse_ra_tf_id_scaled'], 
                    color=color_map[id_val], label=id_val, alpha=0.6)
    ax4a.set_yscale('log')
    ax4a.set_ylabel('Weighted RMSE RA (scaled)')
    ax4a.set_title('Weighted RMSE RA vs Time')
    ax4a.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4a.grid(True, alpha=0.3)
    
    # Dec subplot
    for id_val in df['id'].unique():
        mask = df['id'] == id_val
        ax4b.scatter(df[mask]['midpoint_datetime'], df[mask]['weight_rmse_dec_tf_id_scaled'], 
                    color=color_map[id_val], label=id_val, alpha=0.6)
    ax4b.set_yscale('log')
    ax4b.set_xlabel('Time')
    ax4b.set_ylabel('Weighted RMSE Dec (scaled)')
    ax4b.set_title('Weighted RMSE Dec vs Time')
    ax4b.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4b.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/ra_dec_rmse_vs_time_all.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    
    return color_map, df


def plot_weights_per_id(weights_info, id_val, color_map=None, save_dir=None):
    """Create all plots for a specific ref_point_id"""
    
    # Prepare summary
    df = prepare_weights_summary(weights_info)
    
    # Filter data for this id
    df_id = df[df['id'] == id_val].copy()
    
    # Get color
    if color_map is None:
        color = 'blue'
    else:
        color = color_map[id_val]
    
    # Plot 1: n_obs vs time
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.scatter(df_id['midpoint_datetime'], df_id['n_obs'], color=color, alpha=0.6)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Number of Observations')
    ax1.set_title(f'Number of Observations vs Time - {id_val}')
    ax1.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/n_obs_vs_time_{id_val}.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: mean_weight_rmse_tf_id_scaled vs time
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.scatter(df_id['midpoint_datetime'], df_id['mean_weight_rmse_tf_id_scaled'], 
               color=color, alpha=0.6)
    ax2.set_yscale('log')               
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Mean Weighted RMSE (scaled)')
    ax2.set_title(f'Mean Weighted RMSE vs Time - {id_val}')
    ax2.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/rmse_vs_time_{id_val}.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 3: mean_weight_rmse_tf_id_scaled vs n_obs
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.scatter(df_id['n_obs'], df_id['mean_weight_rmse_tf_id_scaled'], 
               color=color, alpha=0.6)
    ax3.set_yscale('log')
    ax3.set_xlabel('Number of Observations')
    ax3.set_ylabel('Mean Weighted RMSE (scaled)')
    ax3.set_title(f'Mean Weighted RMSE vs Number of Observations - {id_val}')
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/rmse_vs_nobs_{id_val}.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 4: RA and Dec RMSE vs time (subplots)
    fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # RA subplot
    ax4a.scatter(df_id['midpoint_datetime'], df_id['weight_rmse_ra_tf_id_scaled'], 
                color=color, alpha=0.6)
    ax4a.set_yscale('log')                
    ax4a.set_ylabel('Weighted RMSE RA (scaled)')
    ax4a.set_title(f'Weighted RMSE RA vs Time - {id_val}')
    ax4a.grid(True, alpha=0.3)
    
    # Dec subplot
    ax4b.scatter(df_id['midpoint_datetime'], df_id['weight_rmse_dec_tf_id_scaled'], 
                color=color, alpha=0.6)
    ax4b.set_yscale('log')                
    ax4b.set_xlabel('Time')
    ax4b.set_ylabel('Weighted RMSE Dec (scaled)')
    ax4b.set_title(f'Weighted RMSE Dec vs Time - {id_val}')
    ax4b.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/ra_dec_rmse_vs_time_{id_val}.pdf", dpi=300, bbox_inches='tight')
    plt.show()


def plot_weights_all_ids(weights_info, save_dir=None):
    """Create plots for each ref_point_id in the weights_info dataframe"""
    df = prepare_weights_summary(weights_info)
    color_map = get_color_map(df, id_col='id')
    
    for id_val in df['id'].unique():
        print(f"Plotting for {id_val}...")
        plot_weights_per_id(weights_info, id_val, color_map, save_dir)


# Usage:
# Plot all data together
color_map, summary_df = plot_weights_all_data(weights_info, save_dir = out_dir / 'plots')

# Plot each ID separately
plot_weights_all_ids(weights_info, save_dir = out_dir / 'plots')

# Or plot a specific ID
plot_weights_per_id(weights_info, '689_nm0007', color_map, save_dir = out_dir / 'plots')


#How to get reference point of a observation list


print("#######################################################################################################")
print("ANALYSE DATA (AGAIN)")
print("#######################################################################################################")



def epoch_to_datetime(epoch_seconds):
    """Convert seconds since J2000 to datetime"""
    j2000 = datetime(2000, 1, 1, 12, 0, 0)  # J2000 epoch: Jan 1, 2000, 12:00:00
    return j2000 + timedelta(seconds=epoch_seconds)

def calculate_midpoint(row):
    """Calculate midpoint of timeframe"""
    return (row['start_sec'] + row['end_sec']) / 2

def get_color_map(df):
    """Create a color map for each unique id"""
    unique_ids = df['id'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_ids)))
    color_map = dict(zip(unique_ids, colors))
    return color_map

def plot_all_data(df, save_dir=None):
    """Create all plots for the entire dataframe"""
    
    # Add midpoint column
    df = df.copy()
    df['midpoint_sec'] = df.apply(calculate_midpoint, axis=1)
    df['midpoint_datetime'] = df['midpoint_sec'].apply(epoch_to_datetime)
    
    # Get color map
    color_map = get_color_map(df)
    
    # Plot 1: n_obs vs time
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    for id_val in df['id'].unique():
        mask = df['id'] == id_val
        ax1.scatter(df[mask]['midpoint_datetime'], df[mask]['n_obs'], 
                   color=color_map[id_val], label=id_val, alpha=0.6)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Number of Observations')
    ax1.set_title('Number of Observations vs Time')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/n_obs_vs_time_all.pdf", dpi=300, bbox_inches='tight')
    #plt.show()
    
    # Plot 2: mean_weight_rmse_tf_id_scaled vs time
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    for id_val in df['id'].unique():
        mask = df['id'] == id_val
        ax2.scatter(df[mask]['midpoint_datetime'], df[mask]['mean_weight_rmse_tf_id_scaled'], 
                   color=color_map[id_val], label=id_val, alpha=0.6)
    ax2.set_yscale('log')               
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Mean Weighted RMSE (scaled)')
    ax2.set_title('Mean Weighted RMSE vs Time')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/rmse_vs_time_all.pdf", dpi=300, bbox_inches='tight')
    #plt.show()
    
    # Plot 3: mean_weight_rmse_tf_id_scaled vs n_obs
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    for id_val in df['id'].unique():
        mask = df['id'] == id_val
        ax3.scatter(df[mask]['n_obs'], df[mask]['mean_weight_rmse_tf_id_scaled'], 
                   color=color_map[id_val], label=id_val, alpha=0.6)
    ax3.set_yscale('log')
    ax3.set_xlabel('Number of Observations')
    ax3.set_ylabel('Mean Weighted RMSE (scaled)')
    ax3.set_title('Mean Weighted RMSE vs Number of Observations')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/rmse_vs_nobs_all.pdf", dpi=300, bbox_inches='tight')
    #plt.show()
    
    # Plot 4: RA and Dec RMSE vs time (subplots)
    fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # RA subplot
    for id_val in df['id'].unique():
        mask = df['id'] == id_val
        ax4a.scatter(df[mask]['midpoint_datetime'], df[mask]['weight_rmse_ra_tf_id_scaled'], 
                    color=color_map[id_val], label=id_val, alpha=0.6)
    ax4a.set_yscale('log')
    ax4a.set_ylabel('Weighted RMSE RA (scaled)')
    ax4a.set_title('Weighted RMSE RA vs Time')
    ax4a.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4a.grid(True, alpha=0.3)
    
    # Dec subplot
    for id_val in df['id'].unique():
        mask = df['id'] == id_val
        ax4b.scatter(df[mask]['midpoint_datetime'], df[mask]['weight_rmse_dec_tf_id_scaled'], 
                    color=color_map[id_val], label=id_val, alpha=0.6)
    ax4b.set_yscale('log')
    ax4b.set_xlabel('Time')
    ax4b.set_ylabel('Weighted RMSE Dec (scaled)')
    ax4b.set_title('Weighted RMSE Dec vs Time')
    ax4b.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4b.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/ra_dec_rmse_vs_time_all.pdf", dpi=300, bbox_inches='tight')
    #plt.show()
    
    return color_map


def plot_per_id(df, id_val, color_map=None, save_dir=None):
    """Create all plots for a specific id"""
    
    # Filter data for this id
    df_id = df[df['id'] == id_val].copy()
    df_id['midpoint_sec'] = df_id.apply(calculate_midpoint, axis=1)
    df_id['midpoint_datetime'] = df_id['midpoint_sec'].apply(epoch_to_datetime)
    
    # Get color
    if color_map is None:
        color = 'blue'
    else:
        color = color_map[id_val]
    
    # Plot 1: n_obs vs time
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.scatter(df_id['midpoint_datetime'], df_id['n_obs'], color=color, alpha=0.6)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Number of Observations')
    ax1.set_title(f'Number of Observations vs Time - {id_val}')
    ax1.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/n_obs_vs_time_{id_val}.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: mean_weight_rmse_tf_id_scaled vs time
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.scatter(df_id['midpoint_datetime'], df_id['mean_weight_rmse_tf_id_scaled'], 
               color=color, alpha=0.6)
    ax2.set_yscale('log')               
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Mean Weighted RMSE (scaled)')
    ax2.set_title(f'Mean Weighted RMSE vs Time - {id_val}')
    ax2.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/rmse_vs_time_{id_val}.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 3: mean_weight_rmse_tf_id_scaled vs n_obs
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.scatter(df_id['n_obs'], df_id['mean_weight_rmse_tf_id_scaled'], 
               color=color, alpha=0.6)
    ax3.set_yscale('log')
    ax3.set_xlabel('Number of Observations')
    ax3.set_ylabel('Mean Weighted RMSE (scaled)')
    ax3.set_title(f'Mean Weighted RMSE vs Number of Observations - {id_val}')
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/rmse_vs_nobs_{id_val}.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 4: RA and Dec RMSE vs time (subplots)
    fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # RA subplot
    ax4a.scatter(df_id['midpoint_datetime'], df_id['weight_rmse_ra_tf_id_scaled'], 
                color=color, alpha=0.6)
    ax4a.set_yscale('log')                
    ax4a.set_ylabel('Weighted RMSE RA (scaled)')
    ax4a.set_title(f'Weighted RMSE RA vs Time - {id_val}')
    ax4a.grid(True, alpha=0.3)
    
    # Dec subplot
    ax4b.scatter(df_id['midpoint_datetime'], df_id['weight_rmse_dec_tf_id_scaled'], 
                color=color, alpha=0.6)
    ax4b.set_yscale('log')                
    ax4b.set_xlabel('Time')
    ax4b.set_ylabel('Weighted RMSE Dec (scaled)')
    ax4b.set_title(f'Weighted RMSE Dec vs Time - {id_val}')
    ax4b.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/ra_dec_rmse_vs_time_{id_val}.pdf", dpi=300, bbox_inches='tight')
    plt.show()

def plot_all_ids(df, save_dir=None):
    """Create plots for each id in the dataframe"""
    color_map = get_color_map(df)
    
    for id_val in df['id'].unique():
        print(f"Plotting for {id_val}...")
        plot_per_id(df, id_val, color_map, save_dir)


plot_all_data(summary, save_dir=out_dir)
plot_all_ids(summary, save_dir=out_dir)

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

RSW_RMS_PLOTS = False
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


    # Remove simulations with low RMS (initial state and rate only)
    # rms_SPICE.pop('initial_state_only')
    # rms_SPICE.pop('rot_model_rate_only')
    # rms_SPICE.pop('initial_state_only_Pole_Jacbson2009')
    # rms_SPICE.pop('rot_model_rate_only_Pole_Jacobson2009')

    # fig_RMS_SPICE_low = FigUtils.plot_rms_comparison(rms_SPICE)
    # fig_RMS_SPICE_low.savefig(out_dir / "RMS_SPICE_low_RMS_only.pdf")


    # # Compute differences for most promising simulations
    diff_RSW = diff_SPICE_RSW['initial_state_only'] - diff_SPICE_RSW['pole_libration_amplitude']
    fig_RSW = FigUtils.Residuals_RSW(diff_RSW, time_column,type="difference",title=("RSW Difference Initial State only vs pole libration amplitude"))   
    fig_RSW.savefig(out_dir / ("RSW Difference Initial State only vs pole libration amplitude.pdf"))

    # diff_IAU_Jacobson2009_Pole_models_full_rot = diff_SPICE_RSW['pole_full_and_libration_amplitude_Pole_Jacobson2009'] - diff_SPICE_RSW['pole_full_and_libration_amplitude']
    # fig_RSW = FigUtils.Residuals_RSW(diff_IAU_Jacobson2009_Pole_models_full_rot, time_column,type="difference",title=("RSW Difference IAU - Jacobson 2009 full rot model"))   
    # fig_RSW.savefig(out_dir / ("RSW Difference IAU - Jacobson 2009 full rot model.pdf"))
    
    # #---------------------------------------------------------------------------------------


#Check initial state problems:
# diff =simulations['rot_model_rate_only']['state_history_array_full'][0,:,:] - simulations['GM_Triton']['state_history_array_full'][0,:,:] 
# fig_RSW = FigUtils.Residuals_RSW(diff, time_column,type="difference",title=("RSW Difference SPICE - " + key))   
# fig_RSW.savefig(out_dir / ("XYZ Diff Initial rot rate only - GM Triton.pdf"))


print("#######################################################################################################")
print("CREATE CORRELATION PLOTS & PARAMETER PLOTS")
print("#######################################################################################################")

best_parameter_update = {}
rms_residuals = {}
est_parameters_indicies = {}

#Create a dict of indicies for est parameters per simulation, as it is easier to work with this for later plots.
for key in simulations.keys():
        correlations = simulations[key]['correlations']
        parameter_history = simulations[key]['parameter_history']
        best_iteration = simulations[key]['best_iteration']
       
        best_parameter_update[key] = parameter_history[:,0] - parameter_history[:,best_iteration]

        est_parameters = simulations[key]['est_parameters']
        
        #rms_residuals[key] = np.sqrt(np.mean((simulations[key]['residuals_j2000']/1e3)**2)) 


        # Generate parameter labels and units
        labels = []
        units = []
        groups = []  # For coloring by parameter type

        for param in est_parameters:
            if param == 'initial_state':
                labels.extend(['x', 'y', 'z', 'vx', 'vy', 'vz'])
                units.extend(['m', 'm', 'm', 'm/s', 'm/s', 'm/s'])
                groups.extend(['position'] * 3 + ['velocity'] * 3)
            
            elif param == 'iau_rotation_model_pole':
                labels.extend(['α₀', 'δ₀'])
                units.extend(['rad', 'rad'])
                groups.extend(['pole_position'] * 2)
            
            elif param == 'iau_rotation_model_pole_rate':
                labels.extend(['α̇₀', 'δ̇₀'])
                units.extend(['rad/s', 'rad/s'])
                groups.extend(['pole_rate'] * 2)
            
            elif param == 'iau_rotation_model_pole_librations':
                labels.extend([[r'\alpha_{i}', r'\delta_{i}']])
                units.extend(['rad', 'rad'])
                groups.extend(['pole_lib'] * 2)
                if 'pole_librations_deg2' in est_parameters:
                    groups.extend(['pole_lib_deg2']*2)

            elif param == 'GM_Neptune':
                labels.append('GM_Nep')
                units.append('km³/s²')
                groups.append('gravity')
            
            elif param == 'GM_Triton':
                labels.append('GM_Tri')
                units.append('km³/s²')
                groups.append('gravity')
            
            elif param == 'spherical_harmonics':
                # Hardcoded for C20 and C40 only
                labels.extend(['C20', 'C40'])
                units.extend(['[-]', '[-]'])
                groups.extend(['spherical_harmonics', 'spherical_harmonics'])
                
        est_parameters_indicies[key] = groups
        # print("Plotting figs for: ",key)        
        # fig = FigUtils.plot_correlation_matrix(correlations, est_parameters)
        # fig.savefig(out_dir / 'correlations.pdf')

        # fig1 = FigUtils.plot_parameter_updates(best_parameter_update[key],  est_parameters)
        # fig1.savefig(out_dir / "parameter_update.pdf")
        
        # fig2 = FigUtils.plot_parameter_history(parameter_history, est_parameters, best_iteration=best_iteration)
        # fig2.savefig(out_dir / "parameter_history.pdf")
        # print("Next")

fig_mag, fig_comp = FigUtils.plot_state_updates_combined(best_parameter_update)
fig_mag.savefig(out_dir / "Best_Parameter_Update_magnitude.pdf")
fig_comp.savefig(out_dir / "Best_Parameter_Update_components.pdf")



print("#######################################################################################################")
print("CREATE LATEX TABLES")
print("#######################################################################################################")


# Define parameter names and lengths
param_names = [
    'initial_state',   # 6 elements now
    'GM_Neptune',      # 1
    'GM_Triton',       # 1
    'iau_rotation_model_pole',  # 2
    'iau_rotation_model_pole_rate', # 2
    'iau_rotation_model_pole_librations', #2
    'spherical_harmonics'  # 2
]

param_lengths = [6, 1, 1, 2, 2, 2]  # updated for initial_state

# Flatten multi-element parameters into individual entries for the table
param_labels = [
    'initial_state_x', 'initial_state_y', 'initial_state_z',
    'initial_state_vx', 'initial_state_vy', 'initial_state_vz',
    'GM_Neptune', 'GM_Triton',
    'iau_rotation_model_pole_alpha', 'iau_rotation_model_pole_delta',
    'iau_rotation_model_pole_rate_alpha_dot', 'iau_rotation_model_pole_rate_delta_dot',
    'iau_rotation_model_pole_librations_alpha_1','iau_rotation_model_pole_librations_delta_1'
    'spherical_harmonics_C20', 'spherical_harmonics_C40'
]



# Loop through all simulations
tables = {}

#initial_values = simulations['all']['parameter_history'][:,0]  # always use all initial values
    
for sim_name in best_parameter_update.keys():
    estimated_values = best_parameter_update[sim_name]
    initial_values = simulations[sim_name]['parameter_history'][:,0]
    
    # Determine which parameters are estimated in this simulation
    est_parameters = simulations[sim_name]['est_parameters']
    labels = []
    
    for param in est_parameters:
        if param == 'initial_state':
            labels.extend(['$x$ [m]', '$y$ [m]', '$z$ [m]', 
                          '$v_x$ [m/s]', '$v_y$ [m/s]', '$v_z$ [m/s]'])
        elif param == 'iau_rotation_model_pole':
            labels.extend([r'$\alpha_0$ [rad]', r'$\delta_0$ [rad]'])
        elif param == 'iau_rotation_model_pole_rate':
            labels.extend([r'$\dot{\alpha}_0$ [rad/s]', r'$\dot{\delta}_0$ [rad/s]'])
        elif param == 'iau_rotation_model_pole_librations':
            labels.extend([r'$\alpha_1$ [rad/]', r'$\delta_1$ [rad]'])
            if 'pole_librations_deg2' in est_parameters:
                labels.extend([r'$\alpha_2$ [rad/]', r'$\delta_2$ [rad]'])
        elif param == 'GM_Neptune':
            labels.append(r'$GM_{\text{Nep}}$ [m$^3$/s$^2$]')
        elif param == 'GM_Triton':
            labels.append(r'$GM_{\text{Tri}}$ [m$^3$/s$^2$]')
        elif param == 'spherical_harmonics':
            labels.extend(['$C_{20}$', '$C_{40}$'])
    
    est_parameters_labels = labels
    
    # Compute % change (handle division by zero)
    percentile_change = np.where(initial_values != 0, 
                                  estimated_values/initial_values*100, 
                                  np.inf)
    
    # Create table
    df = pd.DataFrame({
        'Parameter': est_parameters_labels,
        'Initial Value': initial_values,
        'Estimated Value Update': estimated_values,
        '\% Change': percentile_change
    })
    
    tables[sim_name] = df
    
    # Save files
    csv_file = out_dir / f"estimated_parameters_{sim_name}.csv"
    latex_file = out_dir / f"estimated_parameters_{sim_name}.tex"
    
    # Save CSV (without LaTeX formatting for readability)
    df_csv = df.copy()
    df_csv['Parameter'] = df_csv['Parameter'].str.replace(r'\$', '', regex=True)
    df_csv['Parameter'] = df_csv['Parameter'].str.replace(r'\\text\{|\}|\\dot\{|\}|\{|\}|_|\\alpha|\\delta', '', regex=True)
    df_csv.to_csv(csv_file, index=False)
    
    # Save LaTeX with proper formatting
    latex_content = df.to_latex(
        index=False,
        float_format="%.2e",  # Changed to %.2e for more compact notation
        caption=f"Estimated parameters for simulation {sim_name}",
        label=f"tab:estimated_parameters_{sim_name}",
        escape=False,  # IMPORTANT: Don't escape LaTeX commands
        column_format='lrrr'
    )
    
    # Post-process LaTeX to use scientific notation properly
    import re
    
    # Replace e notation with \times 10^ notation
    def format_sci_notation(match):
        mantissa = float(match.group(1))
        exponent = int(match.group(2))
        if exponent == 0:
            return f"${mantissa:.2f}$"
        return f"${mantissa:.2f} \\times 10^{{{exponent}}}$"
    
    latex_content = re.sub(r'(-?\d+\.\d+)e([+-]\d+)', format_sci_notation, latex_content)
    
    # Replace inf with ---
    latex_content = latex_content.replace('inf', '---')
    
    with open(latex_file, 'w') as f:
        f.write(latex_content)



print("#######################################################################################################")
print("SIMULATE POLE MOVEMENT")
print("#######################################################################################################")

alpha_array_dict = {}
delta_array_dict = {}
for sim_name in best_parameter_update.keys():
    print('Creating pole movement fig for ' + sim_name)
    estimated_values = best_parameter_update[sim_name]
    est_parameters = simulations[sim_name]['est_parameters']
    parameters_indicies = est_parameters_indicies[sim_name]

    models_Jacobson = ['initial_state_only_Pole_Jacbson2009', 
                        'rot_model_pos_only_Pole_Jacobson2009',
                        'rot_model_rate_only_Pole_Jacobson2009',
                        'rot_model_full_Pole_Jacobson2009',
                        'pole_libration_amplitude_deg1_Pole_Jacobson2009',
                        'pole_libration_amplitude_deg2_Pole_Jacobson2009', 
                        'pole_pos_and_libration_amplitude_Pole_Jacobson2009',
                        'pole_full_and_libration_amplitude_Pole_Jacobson2009']
    
    #if sim_name in models_Jacobson:
    model_type = 'Jacobson2009'
    #else:
    #    model_type = 'IAU'

    parameter_update = [0,0,0,0,0,0,0,0]
    if 'pole_position' in parameters_indicies:
        index = parameters_indicies.index('pole_position')
        parameter_update[0:2] = estimated_values[index:index+2]
    if 'pole_rate' in parameters_indicies:
        index = parameters_indicies.index('pole_rate')
        parameter_update[2:4] = estimated_values[index:index+2]
    if 'pole_lib' in parameters_indicies:
        index = parameters_indicies.index('pole_lib')
        parameter_update[4:6] = estimated_values[index:index+2]
    if 'pole_lib_deg2' in parameters_indicies:

        index = parameters_indicies.index('pole_lib_deg2')
        parameter_update[6:8] = estimated_values[index:index+2]
    # alpha_0, delta_0
    # alpha_dot_0, delta_dot_0
    # alpha_dot_1, delta_dot_1
    # alpha_dot_2, delta_dot_2
    alpha_array_dict[sim_name],delta_array_dict[sim_name] = PropFuncs.PoleModel(time_column,parameter_update,model_type)

    fig = FigUtils.plot_pole_movement(time_column,alpha_array_dict[sim_name],delta_array_dict[sim_name],title=('Pole Movement vs Time ' + sim_name))

    fig.savefig(out_dir / ("pole_movement_" + sim_name + ".pdf"))
    #fig_RSW.savefig(out_dir / ("RSW Diff SPICE - " + key + ".pdf"))


alpha_array_diff = alpha_array_dict['initial_state_only'] - alpha_array_dict['pole_pos']
delta_array_diff = delta_array_dict['initial_state_only'] - delta_array_dict['pole_pos']

fig = FigUtils.plot_pole_movement(time_column,alpha_array_diff,delta_array_diff,title=('Diff IAU 2015 initial vs fitted'))
fig.savefig(out_dir / ("Diff IAU initial vs fitted pole pos only .pdf"))


alpha_array_diff = alpha_array_dict['initial_state_only'] - alpha_array_dict['pole_pos_and_libration_amplitude']
delta_array_diff = delta_array_dict['initial_state_only'] - delta_array_dict['pole_pos_and_libration_amplitude']

fig = FigUtils.plot_pole_movement(time_column,alpha_array_diff,delta_array_diff,title=('Diff IAU 2015 initial vs fitted'))
fig.savefig(out_dir / ("Diff IAU initial vs fitted pole pos and lib .pdf"))




print("End.")