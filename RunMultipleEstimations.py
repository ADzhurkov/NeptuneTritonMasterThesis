
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
simulation_end_epoch   = DateTime(2025, 1, 1).epoch()   #2025, 1, 1

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
        "initial_state_only": {
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/FullDuration/initial_state_only",
            'est_parameters': ['initial_state'] 
        }, 
        "rot_model_pos_only_Pole": {
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/FullDuration_BestIteration/rot_model_pos_only",
            'est_parameters': ['initial_state','iau_rotation_model_pole'],
        },    
        "rot_model_rate_only": {
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/FullDuration_BestIteration/rot_model_rate_only",
            'est_parameters': ['initial_state','iau_rotation_model_pole_rate']
        },    
        "rot_model_full": {
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/FullDuration_BestIteration/rot_model_full",
            'est_parameters': ['initial_state','iau_rotation_model_pole','iau_rotation_model_pole_rate']
        },

        "pole_libration_amplitude":{
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/PoleLibrations/pole_libration_amplitude",
            'est_parameters': ['initial_state','iau_rotation_model_pole_librations'],
        },
        "pole_pos_and_libration_amplitude":{
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/PoleLibrations/pole_pos_and_libration_amplitude",
            'est_parameters': ['initial_state','iau_rotation_model_pole','iau_rotation_model_pole_librations'],
        },
        "pole_full_and_libration_amplitude":{
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/PoleLibrations/pole_full_and_libration_amplitude",
            'est_parameters': ['initial_state','iau_rotation_model_pole','iau_rotation_model_pole_rate','iau_rotation_model_pole_librations'],
        },

        # #Pole Jacobson 2009
        # #---------------------------------------------------------------------------------------------------------------------------------------
        "initial_state_only_Pole_Jacbson2009": {
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/Jacobson2009Pole/initial_state_only_Pole_Jacbson2009",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        },

        "rot_model_pos_only_Pole_Jacobson2009": {
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/Jacobson2009Pole/rot_model_pos_only_Pole_Jacobson2009",
            'est_parameters': ['initial_state','iau_rotation_model_pole'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        },
            
        "rot_model_rate_only_Pole_Jacobson2009": {
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/Jacobson2009Pole/rot_model_rate_only_Pole_Jacobson2009",
            'est_parameters': ['initial_state','iau_rotation_model_pole_rate'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        },
            
        "rot_model_full_Pole_Jacobson2009": {
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/Jacobson2009Pole/rot_model_full_Pole_Jacobson2009",
            'est_parameters': ['initial_state','iau_rotation_model_pole','iau_rotation_model_pole_rate'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        },

        "pole_libration_amplitude_deg1_Pole_Jacobson2009":{
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/Jacobson2009Pole/pole_libration_amplitude_deg1_Pole_Jacobson2009",
            'est_parameters': ['initial_state','iau_rotation_model_pole_librations','pole_librations_deg1'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009' 
        },
        "pole_libration_amplitude_deg2_Pole_Jacobson2009":{
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/Jacobson2009Pole/pole_libration_amplitude_deg2_Pole_Jacobson2009",
            'est_parameters': ['initial_state','iau_rotation_model_pole_librations','pole_librations_deg2'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009' 
        },

        "pole_pos_and_libration_amplitude_Pole_Jacobson2009":{
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/Jacobson2009Pole/pole_pos_and_libration_amplitude_Pole_Jacobson2009",
            'est_parameters': ['initial_state','iau_rotation_model_pole','iau_rotation_model_pole_librations','pole_librations_deg2'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        },
        "pole_full_and_libration_amplitude_Pole_Jacobson2009":{
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/Jacobson2009Pole/pole_full_and_libration_amplitude_Pole_Jacobson2009",
            'est_parameters': ['initial_state','iau_rotation_model_pole','iau_rotation_model_pole_rate','iau_rotation_model_pole_librations','pole_librations_deg2'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        },
        #---------------------------------------------------------------------------------------------------------------------------------------

        # "GM_Neptune": {
        #     "simulation_path": "Results/EstimatedParametersSimulatedObservations/FullDuration_BestIteration/GM_Neptune",
        #     'est_parameters': ['initial_state','GM_Neptune'] 
        # },     
        # "GM_Triton": {
        #     "simulation_path": "Results/EstimatedParametersSimulatedObservations/FullDuration_BestIteration/GM_Triton",
        #     'est_parameters': ['initial_state','GM_Triton'] 
        # },
        # "GM_Neptune_Triton": {
        #     "simulation_path": "Results/EstimatedParametersSimulatedObservations/FullDuration_BestIteration/GM_Neptune_Triton",
        #     'est_parameters': ['initial_state','GM_Neptune','GM_Triton'] 
        # },
        # "spherical_harmonics": {
        #     "simulation_path": "Results/EstimatedParametersSimulatedObservations/FullDuration_BestIteration/spherical_harmonics",
        #     'est_parameters': ['initial_state','spherical_harmonics'] 
        # },
        # "GM_spherical_harmonics": {
        #     "simulation_path": "Results/EstimatedParametersSimulatedObservations/FullDuration/GM_spherical_harmonics",
        #     'est_parameters': ['initial_state','GM_Neptune','GM_Triton','spherical_harmonics'] 
        # },
        # "spherical_harmonics_pole_full":{
        #     "simulation_path": "Results/EstimatedParametersSimulatedObservations/FullDuration_BestIteration/spherical_harmonics_pole_full",
        #     'est_parameters': ['initial_state','iau_rotation_model_pole','iau_rotation_model_pole_rate','spherical_harmonics']  
        # },

        # "all": {
        #     "simulation_path": "Results/EstimatedParametersSimulatedObservations/PoleLibrations/all",
        #     'est_parameters': ['initial_state','GM_Neptune','GM_Triton','iau_rotation_model_pole','iau_rotation_model_pole_rate','iau_rotation_model_pole_librations','spherical_harmonics'] 
        # },                                                  
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
            
            #Select different Pole Model or return to default
            if 'Neptune_rot_model_type' in content.keys():
                settings['env']['Neptune_rot_model_type'] = content['Neptune_rot_model_type']
            else:
                settings['env']['Neptune_rot_model_type'] ='IAU2015'
            
            #Run estimation
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
        # diff = (states_norm_RSW- states_sim_RSW_norm)/1e3
        # diff_wrt_norm_RSW[key] = diff 
        # fig_RSW = FigUtils.Residuals_RSW(diff, time_column,type="difference",title=("RSW Difference Norm - " + key))   
        # fig_RSW.savefig(out_dir / ("RSW Diff Norm_ " + key + ".pdf"))

        # rms_Norm[key] = np.sqrt(np.mean(diff**2)) 


    # #---------------------------------------------------------------------------------------
    # rms_SPICE.pop('spherical_harmonics_pole_full') 
    # rms_Norm.pop('spherical_harmonics_pole_full')


    fig_RMS_SPICE = FigUtils.plot_rms_comparison(rms_SPICE)
    fig_RMS_SPICE.savefig(out_dir / "RMS_SPICE.pdf")


    # Remove simulations with low RMS (initial state and rate only)
    rms_SPICE.pop('initial_state_only')
    rms_SPICE.pop('rot_model_rate_only')
    rms_SPICE.pop('initial_state_only_Pole_Jacbson2009')
    rms_SPICE.pop('rot_model_rate_only_Pole_Jacobson2009')

    fig_RMS_SPICE_low = FigUtils.plot_rms_comparison(rms_SPICE)
    fig_RMS_SPICE_low.savefig(out_dir / "RMS_SPICE_low_RMS_only.pdf")


    # Compute differences for most promising simulations
    diff_IAU_Jacobson2009_Pole_models_pos_lib_only = diff_SPICE_RSW['pole_pos_and_libration_amplitude_Pole_Jacobson2009'] - diff_SPICE_RSW['pole_pos_and_libration_amplitude']
    fig_RSW = FigUtils.Residuals_RSW(diff_IAU_Jacobson2009_Pole_models_pos_lib_only, time_column,type="difference",title=("RSW Difference IAU - Jacobson 2009 pole and lib only"))   
    fig_RSW.savefig(out_dir / ("RSW Difference IAU - Jacobson 2009 pole and lib only.pdf"))

    diff_IAU_Jacobson2009_Pole_models_full_rot = diff_SPICE_RSW['pole_full_and_libration_amplitude_Pole_Jacobson2009'] - diff_SPICE_RSW['pole_full_and_libration_amplitude']
    fig_RSW = FigUtils.Residuals_RSW(diff_IAU_Jacobson2009_Pole_models_full_rot, time_column,type="difference",title=("RSW Difference IAU - Jacobson 2009 full rot model"))   
    fig_RSW.savefig(out_dir / ("RSW Difference IAU - Jacobson 2009 full rot model.pdf"))
    
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
        
        rms_residuals[key] = np.sqrt(np.mean((simulations[key]['residuals_j2000']/1e3)**2)) 


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


fig_mag, fig_comp = FigUtils.plot_state_updates_combined(best_parameter_update)
fig_mag.savefig(out_dir / "Best_Parameter_Update_magnitude.pdf")
fig_comp.savefig(out_dir / "Best_Parameter_Update_components.pdf")

# #rms_residuals.pop('spherical_harmonics_pole_full') 
# fig = FigUtils.plot_rms_comparison(rms_residuals)
# fig.savefig(out_dir / "RMS_Residuals.pdf")


# from matplotlib.backends.backend_pdf import PdfPages

# # Save to a single multi-page PDF
# if RSW_RMS_PLOTS == True:
#     with PdfPages(out_dir / "Combined_Analysis.pdf") as pdf:
#         pdf.savefig(fig_mag)
#         pdf.savefig(fig_RMS_SPICE)


#-----------------------------#-----------------------------#-----------------------------#-----------------------------#-----------------------------

# Plot IAU rotation model pole
fig_pole = FigUtils.plot_parameter_update(best_parameter_update,simulations, 'iau_rotation_model_pole')

# Plot IAU rotation model pole rate
fig_pole_rate = FigUtils.plot_parameter_update(best_parameter_update,simulations, 'iau_rotation_model_pole_rate')

# Plot IAU rotation model pole rate
fig_pole_lib_deg1 = FigUtils.plot_parameter_update(best_parameter_update,simulations, 'iau_rotation_model_pole_librations',libration_deg='deg1')

# Plot IAU rotation model pole rate
fig_pole_lib_deg2 = FigUtils.plot_parameter_update(best_parameter_update,simulations, 'iau_rotation_model_pole_librations',libration_deg='deg2')




fig_pole.savefig(out_dir / "Parameter_Update_pole_pos.pdf")
fig_pole_rate.savefig(out_dir / "Parameter_Update_pole_rate.pdf")
fig_pole_lib_deg1.savefig(out_dir / "Parameter_Update_pole_lib_deg1.pdf")
fig_pole_lib_deg2.savefig(out_dir / "Parameter_Update_pole_lib_deg2.pdf")

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
    
    if sim_name in models_Jacobson:
        model_type = 'Jacobson2009'
    else:
        model_type = 'IAU'

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

# Difference between pole + lib IAU vs Jacobson2009
alpha_array_diff = alpha_array_dict['pole_pos_and_libration_amplitude'] - alpha_array_dict['pole_pos_and_libration_amplitude_Pole_Jacobson2009']
delta_array_diff = delta_array_dict['pole_pos_and_libration_amplitude'] - delta_array_dict['pole_pos_and_libration_amplitude_Pole_Jacobson2009']

fig = FigUtils.plot_pole_movement(time_column,alpha_array_diff,delta_array_diff,title=('Pole Diff IAU 2015 vs Jacobson 2009 for est pole position and lib'))
fig.savefig(out_dir / ("Diff IAU 2015 vs Jacobson 2009 for est pole position and lib.pdf"))

alpha_array_diff = alpha_array_dict['initial_state_only'] - alpha_array_dict['initial_state_only_Pole_Jacbson2009']
delta_array_diff = delta_array_dict['initial_state_only'] - delta_array_dict['initial_state_only_Pole_Jacbson2009']

fig = FigUtils.plot_pole_movement(time_column,alpha_array_diff,delta_array_diff,title=('Diff IAU 2015 vs Jacobson 2009 initial'))
fig.savefig(out_dir / ("Diff IAU 2015 vs Jacobson 2009 initial.pdf"))

alpha_array_diff = alpha_array_dict['initial_state_only'] - alpha_array_dict['pole_pos_and_libration_amplitude']
delta_array_diff = delta_array_dict['initial_state_only'] - delta_array_dict['pole_pos_and_libration_amplitude']

fig = FigUtils.plot_pole_movement(time_column,alpha_array_diff,delta_array_diff,title=('Diff IAU 2015 initial vs fitted'))
fig.savefig(out_dir / ("Diff IAU initial vs fitted.pdf"))




print("End.")