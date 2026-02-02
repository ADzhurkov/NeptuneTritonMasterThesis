
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
# file_names_loaded = [
#         'Triton_874_nm0013.csv',
#         'Triton_689_nm0007.csv',
#         'Triton_337_nm0085.csv',
#         'Triton_337_nm0015.csv',
#         'Triton_337_nm0019.csv']


settings_obs["files"] = file_names_loaded             
settings_obs["observations_folder_path"] = "Observations/AllModernECLIPJ2000"  #RelativeObservations AllModernECLIPJ2000 AllModernJ2000

weights = weights.reset_index()

settings_obs["use_weights"] = True
settings_obs["ra_dec_independent_weights"] = False
settings_obs["timeframe_weights"] = False
settings_obs["weights"] = weights

settings_obs["use_loaded_obs"] = False

settings_obs["residual_filtering"] = True
settings_obs["epoch_filter_dict"] = None 


#Make sure all other weight types are off
settings_obs['std_weights'] = False
settings_obs["per_night_weights"] = False
settings_obs["per_night_weights_id"] = False 
settings_obs['per_night_weights_hybrid'] = False


settings_obs['use_old_obs_func'] = False



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


# ObservationImplementation.main(settings,make_timestamped_folder("Results/EstimatedParametersSimulatedObservations/Test"))




runSimulationsBetter = True
if runSimulationsBetter == True:
    # Define all variants for CASE 1
    VARIANTS = {
        # #Pole IAU2015
        # #---------------------------------------------------------------------------------------------------------------------------------------
        "initial_state_only": {
            "simulation_path": "Results/PoleEstimationRealObservations/FullDuration/initial_state_only",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': ['IAU'],
        }, 
        "initial_state_only_outliers": {
            "simulation_path": "Results/PoleEstimationRealObservations/NewWeightsFull/initial_state_only",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        }, 
        "hybrid_new": {
            "simulation_path": "Results/PoleEstimationRealObservations/NewWeightsFull/hybrid_new",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        }, 
        "Timeframe_new": {
            "simulation_path": "Results/PoleEstimationRealObservations/NewWeightsFull/Timeframe_weights",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        },
        "ID_Weights": {
            "simulation_path": "Results/PoleEstimationRealObservations/NewWeightsFull/ID_Weights",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        }, 
        "Hybrid_old_weights": {
            "simulation_path": "Results/PoleEstimationRealObservations/NewWeightsFull/Hybrid_old_weights",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        }, 
        "Old_Obs_Func": {
            "simulation_path": "Results/PoleEstimationRealObservations/NewWeightsFull/Old_Obs_Func",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        },
        "Hybrid_old_no_loop": {
            "simulation_path": "Results/PoleEstimationRealObservations/NewWeightsFull/Hybrid_old_weights_no_loop",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        }, 

                                                
    }

    # Weight Test CASE 2 (small set)
    VARIANTS = {
        "Initial_State_No_Weights": {
            "simulation_path": "Results/PoleEstimationRealObservations/WeightTest_CASE2/Initial_State_No_Weights",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        },
        "initial_state_hybrid_old_weights": {
            "simulation_path": "Results/PoleEstimationRealObservations/WeightTest_CASE2/initial_state_hybrid_old_weights",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        },
        "initial_state_hybrid_weights": {
            "simulation_path": "Results/PoleEstimationRealObservations/WeightTest_CASE2/initial_state_hybrid_weights",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        },
        "initial_state_id_weights": {
            "simulation_path": "Results/PoleEstimationRealObservations/WeightTest_CASE2/initial_state_id_weights",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        },
        "initial_state_tf_weights": {
            "simulation_path": "Results/PoleEstimationRealObservations/WeightTest_CASE2/initial_state_tf_weights",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        },
    }

    # Weight Test CASE 1 (full set)
    VARIANTS = {
        "Initial_State_No_Weights": {
            "simulation_path": "Results/PoleEstimationRealObservations/WeightTest_CASE1/Initial_State_No_Weights",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        },
        "initial_state_hybrid_old_weights": {
            "simulation_path": "Results/PoleEstimationRealObservations/WeightTest_CASE1/initial_state_hybrid_old_weights",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        },
        "initial_state_hybrid_weights": {
            "simulation_path": "Results/PoleEstimationRealObservations/WeightTest_CASE1/initial_state_hybrid_weights",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        },
        "initial_state_id_weights": {
            "simulation_path": "Results/PoleEstimationRealObservations/WeightTest_CASE1/initial_state_id_weights",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        },
        "initial_state_tf_weights": {
            "simulation_path": "Results/PoleEstimationRealObservations/WeightTest_CASE1/initial_state_tf_weights",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        },
    }



    # # Define all variants for 
    # VARIANTS = {
    #     # #Pole IAU2015
    #     # #---------------------------------------------------------------------------------------------------------------------------------------
    #     "initial_state_only": {
    #         "simulation_path": "Results/PoleEstimationRealObservations/CASE2/Initial_State_No_Weights",
    #         'est_parameters': ['initial_state'],
    #         'Neptune_rot_model_type': 'Pole_Model_Jacobson2009',
    #     }, 
    #     "initial_state_hybrid_weights": {
    #         "simulation_path": "Results/PoleEstimationRealObservations/CASE2/initial_state_hybrid_weights",
    #         'est_parameters': ['initial_state'],
    #         'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
    #     }, 
    #     "pole_lib_deg_1_hybrid_weights": {
    #         "simulation_path": "Results/PoleEstimationRealObservations/CASE2/pole_lib_deg_1_hybrid_weights",
    #         'est_parameters': ['initial_state'],
    #         'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
    #     }, 
    #     "pole_lib_deg_2_hybrid_weights": {
    #         "simulation_path": "Results/PoleEstimationRealObservations/CASE2/pole_lib_deg_2_hybrid_weights",
    #         'est_parameters': ['initial_state'],
    #         'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
    #     },
    #     "pole_pos_hybrid_weights": {
    #         "simulation_path": "Results/PoleEstimationRealObservations/CASE2/pole_pos_hybrid_weights",
    #         'est_parameters': ['initial_state'],
    #         'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
    #     }, 
    #     "pole_pos_lib_deg_1_hybrid_weights": {
    #         "simulation_path": "Results/PoleEstimationRealObservations/CASE2/pole_pos_lib_deg_1_hybrid_weights",
    #         'est_parameters': ['initial_state'],
    #         'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
    #     }
                                                
    # }



    # # Define 
    # VARIANTS = {
    #     "initial_state_only": {
    #         "simulation_path": "Results/PoleEstimationRealObservations/Loop/initial_state_only",
    #         'est_parameters': ['initial_state'],
    #         'Neptune_rot_model_type': ['Pole_Model_Jacobson2009'],
    #         'max_estimations': 2,
    #     }, 

    # }


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
    

    # Load simulations
    simulations = {
        name: PostProc.load_npy_files(cfg["simulation_path"])
        for name, cfg in VARIANTS.items()
    }


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


    for name, cfg in VARIANTS.items():
        simulations[name]["est_parameters"] = cfg["est_parameters"]



#Test
#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------
print("#######################################################################################################")
print("LOAD KERNELS")
print("#######################################################################################################")
#Load kernels
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

# EXAMPLE CODE 
##################################################################################################################

# #Create Environment 
# body_settings,system_of_bodies = PropFuncs.Create_Env(settings['env'])

# #Load observations
# observations,observations_settings,observation_set_ids, epochs_rejected = ObsFunc.LoadObservations(
#         settings["obs"]["observations_folder_path"],
#         system_of_bodies,
#         settings['obs']["files"],
#         Residual_filtering = settings["obs"]["residual_filtering"])

# #Load residuals from previous simulation to use as weights
# simulation = PostProc.load_npy_files('Results/PoleEstimationRealObservations/TestNewWeights_Full_NoImprovement/First')

# best_iteration = simulation['best_iteration']
# residuals = simulation['residual_history_arcseconds'][best_iteration]

# # Convert RA and DEC columns from arcseconds to radians
# residuals[:, 1] = residuals[:, 1] / (3600 * 180 / np.pi)  # RA
# residuals[:, 2] = residuals[:, 2] / (3600 * 180 / np.pi)  # DEC


# settings['prop']['initial_state'] = simulation['parameter_history'][:,best_iteration]
# settings['obs']['use_loaded_obs'] = True

##################################################################################################################




# # # RUN ESTIMATION WITHOUT WEIGHTS FIRST
# print("######################################")
# print("Running NO WEIGHTS INITIAL STATE SIM")
# print("######################################")

# #fitted_pole_pos_lib_sim = PostProc.load_npy_files("Results/EstimatedParametersSimulatedObservations/PoleLibrations/pole_pos_and_libration_amplitude")

# settings['est']['a_priori_covariance'] = False

# settings['obs']['use_loaded_obs'] = False
# settings['prop']['initial_state'] = None #simulation['parameter_history'][:,best_iteration]


# settings['est']['est_parameters'] = ['initial_state']

# # pole_params = fitted_pole_pos_lib_sim['parameter_history'][6:,-1]
# # # Create the numpy array for [alpha_i, delta_i] as a 2x1 column vector

# # settings['env']['initial_Pole_Pos'] = pole_params[0:2]

# # settings['env']['initial_Pole_lib_deg1'] = pole_params[2:4]


# out_dir_current = out_dir / "Initial_State_No_Weights"
# out_dir_current.mkdir(parents=True, exist_ok=True)

# estimation_output,observations,observations_settings,body_settings,system_of_bodies = ObservationImplementation.main(
#         settings,
#         out_dir_current)

# #Create Environment 
# #body_settings,system_of_bodies = PropFuncs.Create_Env(settings['env'])

# # #Load observations
# # observations,observations_settings,observation_set_ids, epochs_rejected = ObsFunc.LoadObservations(
# #         settings["obs"]["observations_folder_path"],
# #         system_of_bodies,
# #         settings['obs']["files"],
# #         Residual_filtering = settings["obs"]["residual_filtering"])


# simulation = PostProc.load_npy_files(out_dir_current)

# residuals = simulation['residual_history_arcseconds'][-1]

# # Convert RA and DEC columns from arcseconds to radians
# residuals[:, 1] = residuals[:, 1] / (3600 * 180 / np.pi)  # RA
# residuals[:, 2] = residuals[:, 2] / (3600 * 180 / np.pi)  # DEC


# # EXTRACT RESIDUALS FROM INITIAL SIM 
# # AND COMPUTE/ASSIGN WEIGHTS FROM THEM
# # observations_weighted, weights_info = ObsFunc.compute_and_assign_weights(
# #     residuals=residuals,
# #     observations=observations,
# #     gap_threshold_hours=4.0,
# #     min_obs_per_frame=1,
# #     weight_type = 'hybrid'
# # )



# # RUN ESTIMATION HYBRID WEIGHTS
# print("######################################") 
# print("Running INITIAL STATE HYBRID WEIGHTS SIM")
# print("######################################")

# observations_weighted, weights_info = ObsFunc.compute_and_assign_weights(
#     residuals=residuals,
#     observations=observations,
#     gap_threshold_hours=4.0,
#     min_obs_per_frame=1,
#     weight_type = 'hybrid'
# )

# settings['obs']['use_loaded_obs'] = True

# out_dir_current = out_dir / "initial_state_hybrid_weights"
# out_dir_current.mkdir(parents=True, exist_ok=True)


# # # Save weights to CSV
# weights_info.to_csv(out_dir_current / 'observation_weights.csv', index=False)

# print("Observation weights: ",observations_weighted.get_concatenated_weights())
# estimation_output,observations,observations_settings,body_settings,system_of_bodies = ObservationImplementation.main(
#         settings,
#         out_dir_current,
#         observations=observations_weighted,
#         observations_settings=observations_settings)


# # RUN ESTIMATION HYBRID WEIGHTS
# print("######################################") 
# print("Running HYBRID OLD WEIGHTS SIM")
# print("######################################")


# observations_weighted, weights_info = ObsFunc.compute_and_assign_weights(
#     residuals=residuals,
#     observations=observations,
#     gap_threshold_hours=4.0,
#     min_obs_per_frame=1,
#     weight_type = 'hybrid_old'
# )

# settings['obs']['use_loaded_obs'] = True

# out_dir_current = out_dir / "initial_state_hybrid_old_weights"
# out_dir_current.mkdir(parents=True, exist_ok=True)



# weights_info.to_csv(out_dir_current / 'observation_weights.csv', index=False)

# print("Observation weights: ",observations_weighted.get_concatenated_weights())

# estimation_output,observations,observations_settings,body_settings,system_of_bodies = ObservationImplementation.main(
#         settings,
#         out_dir_current,
#         observations=observations_weighted,
#         observations_settings=observations_settings)

# # RUN ESTIMATION TF WEIGHTS
# print("######################################") 
# print("Running TF WEIGHTS SIM")
# print("######################################")


# observations_weighted, weights_info = ObsFunc.compute_and_assign_weights(
#     residuals=residuals,
#     observations=observations,
#     gap_threshold_hours=4.0,
#     min_obs_per_frame=1,
#     weight_type = 'timeframe'
# )

# settings['obs']['use_loaded_obs'] = True

# out_dir_current = out_dir / "initial_state_tf_weights"
# out_dir_current.mkdir(parents=True, exist_ok=True)


# # # Save weights to CSV
# weights_info.to_csv(out_dir_current / 'observation_weights.csv', index=False)

# print("Observation weights: ",observations_weighted.get_concatenated_weights())

# estimation_output,observations,observations_settings,body_settings,system_of_bodies = ObservationImplementation.main(
#         settings,
#         out_dir_current,
#         observations=observations_weighted,
#         observations_settings=observations_settings)


# # RUN ESTIMATION TF WEIGHTS
# print("######################################") 
# print("Running ID WEIGHTS SIM")
# print("######################################")


# observations_weighted, weights_info = ObsFunc.compute_and_assign_weights(
#     residuals=residuals,
#     observations=observations,
#     gap_threshold_hours=4.0,
#     min_obs_per_frame=1,
#     weight_type = 'id'
# )

# settings['obs']['use_loaded_obs'] = True

# out_dir_current = out_dir / "initial_state_id_weights"
# out_dir_current.mkdir(parents=True, exist_ok=True)


# # # Save weights to CSV
# weights_info.to_csv(out_dir_current / 'observation_weights.csv', index=False)

# print("Observation weights: ",observations_weighted.get_concatenated_weights())

# estimation_output,observations,observations_settings,body_settings,system_of_bodies = ObservationImplementation.main(
#         settings,
#         out_dir_current,
#         observations=observations_weighted,
#         observations_settings=observations_settings)



# print("Test")




def epoch_to_datetime(epoch_seconds):
    """Convert seconds since J2000 to datetime"""
    j2000 = datetime(2000, 1, 1, 12, 0, 0)  # J2000 epoch: Jan 1, 2000, 12:00:00
    return j2000 + timedelta(seconds=epoch_seconds)

def rad_to_mas(rad):
    """Convert radians to milliarcseconds"""
    return rad * (180/np.pi) * 3600 * 1000

def weight_to_sigma_mas(weight):
    """Convert weight (1/sigma^2 in rad^2) to sigma in mas"""
    sigma_rad = 1.0 / np.sqrt(weight)
    return rad_to_mas(sigma_rad)

def plot_number_of_obs_vs_time(data, out_dir='/mnt/user-data/outputs'):
    """
    Plot 1: Number of Observations vs Time (matching your screenshot)
    Groups observations by timeframe (night)
    
    Parameters:
    -----------
    data : pd.DataFrame
        Must contain columns: 'time', 'ref_point_id', 'timeframe'
    out_dir : str
        Output directory for PDF
    """
    os.makedirs(out_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Get unique reference points
    ref_points = data['ref_point_id'].unique()
    
    # Plot each reference point with different color
    for ref in ref_points:
        ref_data = data[data['ref_point_id'] == ref]
        
        # Group by timeframe and get mean time and count for each frame
        timeframe_stats = ref_data.groupby('timeframe').agg({
            'time': 'mean',  # Use mean time of observations in the frame
            'ref_point_id': 'size'  # Count observations
        }).reset_index()
        timeframe_stats.columns = ['timeframe', 'time', 'count']
        
        ax.scatter(timeframe_stats['time'], timeframe_stats['count'], 
                  label=ref, alpha=0.7, s=30)
    
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Number of Observations', fontsize=14)
    ax.set_title('Number of Observations vs Time', fontsize=16, fontweight='bold', 
                color='#1E88E5', loc='left')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(out_dir, 'number_of_obs_vs_time.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_ra_dec_rmse_vs_time(residuals_data, out_dir='/mnt/user-data/outputs'):
    """
    Plot 2: RA and DEC RMSE vs Time (matching your screenshot)
    Groups observations by timeframe (night)
    
    Parameters:
    -----------
    residuals_data : pd.DataFrame
        Must contain: 'time', 'residual_ra', 'residual_dec', 'ref_point_id', 'timeframe'
        Residuals should be in radians
    out_dir : str
        Output directory for PDF
    """
    os.makedirs(out_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Convert residuals to mas
    residuals_data = residuals_data.copy()
    residuals_data['residual_ra_mas'] = rad_to_mas(residuals_data['residual_ra'])
    residuals_data['residual_dec_mas'] = rad_to_mas(residuals_data['residual_dec'])
    
    # Get unique reference points
    ref_points = residuals_data['ref_point_id'].unique()
    
    # Calculate RMSE per timeframe
    for ref in ref_points:
        ref_data = residuals_data[residuals_data['ref_point_id'] == ref]
        
        # Group by timeframe and calculate RMSE for each
        timeframe_stats = ref_data.groupby('timeframe').agg({
            'time': 'mean',  # Mean time of the timeframe
            'residual_ra_mas': 'mean',
            'residual_dec_mas': 'mean',
        }).reset_index()
        
        # Plot RA RMSE
        axes[0].scatter(timeframe_stats['time'], timeframe_stats['residual_ra_mas'],
                       label=ref, alpha=0.7, s=30)
        
        # Plot DEC RMSE
        axes[1].scatter(timeframe_stats['time'], timeframe_stats['residual_dec_mas'],
                       label=ref, alpha=0.7, s=30)
    
    axes[0].set_ylabel('Residual RA (mas)', fontsize=12)
    axes[0].set_title('RA and DEC Residual vs Time', fontsize=16, fontweight='bold',
                     color='#1E88E5', loc='left')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    #axes[0].set_yscale('log')
    
    axes[1].set_ylabel('Residual Dec (mas)', fontsize=12)
    axes[1].set_xlabel('Time', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    #axes[1].set_yscale('log')
    
    plt.tight_layout()
    output_path = os.path.join(out_dir, 'ra_dec_residual_vs_time.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Saved: {output_path}")
    plt.close(fig)

def plot_rmse_vs_time(residuals_data, out_dir='/mnt/user-data/outputs'):
    """
    Plot 3: Weighted RMSE RA/DEC vs Time (matching your screenshot)
    Groups observations by timeframe (night)
    
    Parameters:
    -----------
    residuals_data : pd.DataFrame
        Must contain: 'time', 'residual_ra', 'residual_dec', 'ref_point_id', 
                     'weight_ra', 'weight_dec', 'timeframe'
        Residuals should be in radians, weights in rad^-2
    out_dir : str
        Output directory for PDF
    """
    os.makedirs(out_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    residuals_data = residuals_data.copy()
    
    # Calculate weighted residuals (normalized by uncertainty)
    residuals_data['weighted_res_ra'] = (
        residuals_data['residual_ra'] * np.sqrt(residuals_data['weight_ra'])
    )
    residuals_data['weighted_res_dec'] = (
        residuals_data['residual_dec'] * np.sqrt(residuals_data['weight_dec'])
    )
    
    # Get unique reference points
    ref_points = residuals_data['ref_point_id'].unique()
    
    for ref in ref_points:
        ref_data = residuals_data[residuals_data['ref_point_id'] == ref]
        
        # Group by timeframe
        timeframe_stats = ref_data.groupby('timeframe').agg({
            'time': 'mean',  # Mean time of the timeframe
            'weighted_res_ra': lambda x: np.sqrt(np.mean(x**2)),  # RMSE for RA
            'weighted_res_dec': lambda x: np.sqrt(np.mean(x**2))  # RMSE for DEC
        }).reset_index()
        
        # Plot RA
        axes[0].scatter(timeframe_stats['time'], timeframe_stats['weighted_res_ra'],
                       label=ref, alpha=0.7, s=30)
        
        # Plot DEC
        axes[1].scatter(timeframe_stats['time'], timeframe_stats['weighted_res_dec'],
                       label=ref, alpha=0.7, s=30)
    
    axes[0].set_ylabel('Weighted RMSE RA (scaled)', fontsize=12)
    axes[0].set_title('Weighted RMSE vs Time', fontsize=16, fontweight='bold',
                     color='#1E88E5', loc='left')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    #axes[0].set_yscale('log')
    
    axes[1].set_ylabel('Weighted RMSE DEC (scaled)', fontsize=12)
    axes[1].set_xlabel('Time', fontsize=14)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    #axes[1].set_yscale('log')
    
    plt.tight_layout()
    output_path = os.path.join(out_dir, 'weighted_rmse_vs_time.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_weight_vs_n_obs(residuals_data, out_dir='/mnt/user-data/outputs'):
    """
    Plot 4: Weights vs Number of Observations
    Shows how weights relate to number of observations per timeframe
    One point per timeframe
    
    Parameters:
    -----------
    residuals_data : pd.DataFrame
        Must contain: 'weight_ra', 'weight_dec', 'ref_point_id', 'timeframe'
        Weights in rad^-2
    out_dir : str
        Output directory for PDF
    """
    os.makedirs(out_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    residuals_data = residuals_data.copy()
    
    # Get unique reference points
    ref_points = residuals_data['ref_point_id'].unique()
    
    for ref in ref_points:
        ref_data = residuals_data[residuals_data['ref_point_id'] == ref]
        
        # Get mean weight and count per timeframe
        timeframe_stats = ref_data.groupby('timeframe').agg({
            'weight_ra': 'mean',  # Mean RA weight for the timeframe
            'weight_dec': 'mean'  # Mean DEC weight for the timeframe
        }).reset_index()
        
        # Calculate number of observations per timeframe
        obs_per_frame = ref_data.groupby('timeframe').size().reset_index(name='n_obs')
        timeframe_stats = timeframe_stats.merge(obs_per_frame, on='timeframe')
        
        # Plot RA weights vs n_obs
        axes[0].scatter(timeframe_stats['n_obs'], timeframe_stats['weight_ra'], 
                       label=ref, alpha=0.7, s=30)
        
        # Plot DEC weights vs n_obs
        axes[1].scatter(timeframe_stats['n_obs'], timeframe_stats['weight_dec'], 
                       label=ref, alpha=0.7, s=30)
    
    axes[0].set_ylabel('Weight RA [rad⁻²]', fontsize=12)
    axes[0].set_title('Weights vs Number of Observations', fontsize=16, fontweight='bold',
                     color='#1E88E5', loc='left')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    axes[1].set_ylabel('Weight DEC [rad⁻²]', fontsize=12)
    axes[1].set_xlabel('Number of Observations per Timeframe', fontsize=14)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    output_path = os.path.join(out_dir, 'weights_vs_num_obs.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Saved: {output_path}")
    plt.close(fig)

def plot_weights_analysis(weights_info, out_dir='/mnt/user-data/outputs'):
    """
    Plot 5: Comprehensive Weights Analysis (9 subplots)
    
    Parameters:
    -----------
    weights_info : pd.DataFrame
        Must contain: 'weight_ra_hybrid', 'weight_dec_hybrid', 'ref_point_id',
                     'global_obs_index'
    out_dir : str
        Output directory for PDF
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Convert weights to uncertainties in mas
    sigma_ra = weight_to_sigma_mas(weights_info['weight_ra_hybrid'])
    sigma_dec = weight_to_sigma_mas(weights_info['weight_dec_hybrid'])
    
    residuals_ra = rad_to_mas(weights_info['ra_residual'])
    residuals_dec = rad_to_mas(weights_info['dec_residual'])

    fig = plt.figure(figsize=(16, 12))
    
    # =========================================================================
    # Plot 1: Residuals Histogram
    # =========================================================================
    ax1 = plt.subplot(3, 3, 1)
    
    ax1.hist(residuals_ra, bins=50, alpha=0.6, label='RA (σ)', color='blue', edgecolor='black')
    ax1.hist(residuals_dec, bins=50, alpha=0.6, label='DEC (σ)', color='red', edgecolor='black')
    ax1.set_xlabel('Residuals (mas)', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Distribution of Observation Residuals', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    #ax1.set_yscale('log')
    
    # =========================================================================
    # Plot 2: Weights on Log Scale
    # =========================================================================
    ax2 = plt.subplot(3, 3, 2)
    
    ax2.hist(np.log10(weights_info['weight_ra_hybrid']), bins=50, alpha=0.6, 
             label='RA', color='blue', edgecolor='black')
    ax2.hist(np.log10(weights_info['weight_dec_hybrid']), bins=50, alpha=0.6,
             label='DEC', color='red', edgecolor='black')
    ax2.set_xlabel('log₁₀(Weight) [rad⁻²]', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Distribution of Weights (Log Scale)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # =========================================================================
    # Plot 3: RA vs DEC Residual Scatter
    # =========================================================================
    ax3 = plt.subplot(3, 3, 3)

    # Get unique reference points and assign colors
    ref_points = weights_info['ref_point_id'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(ref_points)))
    ref_to_color = dict(zip(ref_points, colors))

    # Plot each reference point with its unique color
    for ref in ref_points:
        mask = weights_info['ref_point_id'] == ref
        ax3.scatter(residuals_ra[mask], residuals_dec[mask], 
                    c=[ref_to_color[ref]], 
                    label=ref, alpha=0.6, s=20)

    max_sigma = max(residuals_ra.max(), residuals_dec.max())
    ax3.plot([0, max_sigma], [0, max_sigma], 
            'k--', alpha=0.5, label='σ_RA = σ_DEC', linewidth=1.5)
    ax3.set_xlabel('RA Residual (mas)', fontsize=11)
    ax3.set_ylabel('DEC Residual (mas)', fontsize=11)
    ax3.set_title('RA vs DEC Residual', fontsize=12, fontweight='bold')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, 
            framealpha=0.9, ncol=1)
    ax3.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 4: Residual vs Time
    # =========================================================================
    ax4 = plt.subplot(3, 3, 4)
    
    ax4.scatter(weights_info['time'], residuals_ra, 
               alpha=0.5, s=10, label='RA', color='blue')
    ax4.scatter(weights_info['time'], residuals_dec,
               alpha=0.5, s=10, label='DEC', color='red')
    ax4.set_xlabel('Time', fontsize=11)
    ax4.set_ylabel('Residual (mas)', fontsize=11)
    ax4.set_title('Residuals vs Time', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    #ax4.set_yscale('log')
    
    # =========================================================================
    # Plot 5: Residual by Set ID
    # =========================================================================
    ax5 = plt.subplot(3, 3, 5)
    
    ref_points = weights_info['ref_point_id'].unique()
    
    ra_by_ref = [sigma_ra[weights_info['ref_point_id'] == ref].values
                 for ref in ref_points]
    dec_by_ref = [sigma_dec[weights_info['ref_point_id'] == ref].values
                  for ref in ref_points]
    
    positions_ra = np.arange(len(ref_points)) * 2 - 0.3
    positions_dec = np.arange(len(ref_points)) * 2 + 0.3
    
    bp1 = ax5.boxplot(ra_by_ref, positions=positions_ra, widths=0.5,
                     patch_artist=True, showfliers=True)
    bp2 = ax5.boxplot(dec_by_ref, positions=positions_dec, widths=0.5,
                     patch_artist=True, showfliers=True)
    
    for patch in bp1['boxes']:
        patch.set_facecolor('blue')
        patch.set_alpha(0.6)
    for patch in bp2['boxes']:
        patch.set_facecolor('red')
        patch.set_alpha(0.6)
    
    ax5.set_xlabel('Set ID', fontsize=11)
    ax5.set_ylabel('RMSE (mas)', fontsize=11)
    ax5.set_title('RMSE by Set ID', fontsize=12, fontweight='bold')
    ax5.set_xticks(np.arange(len(ref_points)) * 2)
    ax5.set_xticklabels(ref_points, rotation=45, ha='right')
    ax5.grid(True, alpha=0.3, axis='y')
    #ax5.set_yscale('log')
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', alpha=0.6, label='RA'),
                      Patch(facecolor='red', alpha=0.6, label='DEC')]
    ax5.legend(handles=legend_elements, loc='upper right')
    
    # =========================================================================
    # Plot 6: Mean residuals by Set ID
    # =========================================================================
    ax6 = plt.subplot(3, 3, 6)
    
    stats_data = []
    for ref in ref_points:
        mask = weights_info['ref_point_id'] == ref
        stats_data.append({
            'ref': ref,
            'count': mask.sum(),
            'mean_ra': sigma_ra[mask].mean(),
            'mean_dec': sigma_dec[mask].mean()
        })
    stats_df = pd.DataFrame(stats_data)
    
    x = np.arange(len(ref_points))
    width = 0.35
    
    bars1 = ax6.bar(x - width/2, stats_df['mean_ra'], width, 
                   label='RA Mean', alpha=0.7, color='blue')
    bars2 = ax6.bar(x + width/2, stats_df['mean_dec'], width,
                   label='DEC Mean', alpha=0.7, color='red')
    
    ax6.set_xlabel('Set ID', fontsize=11)
    ax6.set_ylabel('Mean Residual (mas)', fontsize=11)
    ax6.set_title('Mean Residual by Set ID', fontsize=12, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(ref_points, rotation=45, ha='right')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    # =========================================================================
    # Plot 7: Cumulative Distribution
    # =========================================================================
    ax7 = plt.subplot(3, 3, 7)
    
    sorted_ra = np.sort(residuals_ra)
    sorted_dec = np.sort(residuals_dec)
    cumulative = np.arange(1, len(sorted_ra) + 1) / len(sorted_ra) * 100
    
    ax7.plot(sorted_ra, cumulative, label='RA', color='blue', linewidth=2)
    ax7.plot(sorted_dec, cumulative, label='DEC', color='red', linewidth=2)
    ax7.axhline(50, color='gray', linestyle='--', alpha=0.5, label='Median')
    ax7.set_xlabel('Uncertainty (mas)', fontsize=11)
    ax7.set_ylabel('Cumulative Percentage (%)', fontsize=11)
    ax7.set_title('Cumulative Distribution of Residuals', fontsize=12, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    #ax7.set_xscale('log')
    
    # =========================================================================
    # Plot 8: Uncertainty Ratio (RA/DEC)
    # =========================================================================
    ax8 = plt.subplot(3, 3, 8)
    
    uncertainty_ratio = sigma_ra / sigma_dec
    
    ax8.hist(uncertainty_ratio, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax8.axvline(1.0, color='black', linestyle='--', linewidth=2, label='Equal uncertainties')
    ax8.axvline(uncertainty_ratio.median(), color='red', linestyle='--', 
               linewidth=2, label=f'Median = {uncertainty_ratio.median():.2f}')
    ax8.set_xlabel('σ_RA / σ_DEC Ratio', fontsize=11)
    ax8.set_ylabel('Count', fontsize=11)
    ax8.set_title('RA/DEC Residual Ratio', fontsize=12, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # =========================================================================
    # Plot 9: Summary Statistics
    # =========================================================================
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    stats_text = f"""
    WEIGHT STATISTICS SUMMARY
    {'='*45}
    
    Total Observations: {len(weights_info):,}
    Reference Points: {len(ref_points)}
    
    RA Residuals (mas):
      Min:    {residuals_ra.min():.4f}
      Max:    {residuals_ra.max():.4f}
      Mean:   {residuals_ra.mean():.4f}
      Median: {residuals_ra.median():.4f}
      Std:    {residuals_ra.std():.4f}
      RMS:    {np.sqrt(np.mean(residuals_ra**2)):.4f}
    
    DEC Residuals (mas):
      Min:    {residuals_dec.min():.4f}
      Max:    {residuals_dec.max():.4f}
      Mean:   {residuals_dec.mean():.4f}
      Median: {residuals_dec.median():.4f}
      Std:    {residuals_dec.std():.4f}
      RMS:    {np.sqrt(np.mean(residuals_dec**2)):.4f}

    Ratio σ_RA/σ_DEC:
      Mean:   {uncertainty_ratio.mean():.4f}
      Median: {uncertainty_ratio.median():.4f}
    """
    
    ax9.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax9.transAxes)
    
    plt.tight_layout()
    output_path = os.path.join(out_dir, 'weights_analysis.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Saved: {output_path}")
    plt.close(fig)


# ============================================================================
# MAIN FUNCTION TO CREATE ALL WEIGHT PLOTS
# ============================================================================

def create_all_plots(weights_info, out_dir='/mnt/user-data/outputs'):
    """
    Create all analysis plots from weights_info DataFrame
    
    Parameters:
    -----------
    weights_info : pd.DataFrame
        Complete weights and residuals data with columns:
        - time: observation times
        - ref_point_id: reference point identifier  
        - global_obs_index: global observation index
        - ra_residual, dec_residual: residuals in radians
        - weight_ra_hybrid, weight_dec_hybrid: hybrid weights in rad^-2
        - weight_ra_id, weight_dec_id: ID-level weights
        - weight_ra_local, weight_dec_local: local/timeframe weights
    
    out_dir : str
        Output directory for all PDFs
    """
    print("="*70)
    print("CREATING ALL ANALYSIS PLOTS")
    print("="*70)
    
    # Verify required columns exist
    required_cols = ['time', 'ref_point_id', 'global_obs_index', 
                     'ra_residual', 'dec_residual',
                     'weight_ra_hybrid', 'weight_dec_hybrid']
    missing_cols = [col for col in required_cols if col not in weights_info.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in weights_info: {missing_cols}")
    
    # Convert time to datetime if needed
    weights_info = weights_info.copy()
    if not pd.api.types.is_datetime64_any_dtype(weights_info['time']):
        # Time is in seconds since J2000
        print("Converting time from J2000 epoch seconds to datetime...")
        weights_info['time'] = weights_info['time'].apply(epoch_to_datetime)
        weights_info['time'] = pd.to_datetime(weights_info['time'])
    
    # Prepare residuals data with correct column names for residual functions
    residuals_data = weights_info.copy()
    if 'weight_ra' not in residuals_data.columns:
        residuals_data['weight_ra'] = residuals_data['weight_ra_hybrid']
        residuals_data['weight_dec'] = residuals_data['weight_dec_hybrid']
    if 'residual_ra' not in residuals_data.columns:
        residuals_data['residual_ra'] = residuals_data['ra_residual']
        residuals_data['residual_dec'] = residuals_data['dec_residual']
    
    # Create all plots
    print("\n[1/5] Creating weights analysis...")
    plot_weights_analysis(weights_info, out_dir)
    
    print("\n[2/5] Creating number of observations vs time...")
    plot_number_of_obs_vs_time(weights_info, out_dir)
    
    print("\n[3/5] Creating RA and DEC RMSE vs time...")
    plot_ra_dec_rmse_vs_time(residuals_data, out_dir)
    
    print("\n[4/5] Creating RMSE vs time...")
    plot_rmse_vs_time(residuals_data, out_dir)
    
    print("\n[5/5] Creating Weight vs number of observations...")
    plot_weight_vs_n_obs(residuals_data, out_dir)
    
    print("\n" + "="*70)
    print("ALL PLOTS COMPLETED!")
    print("="*70)




#create_all_plots(weights_info, out_dir)


def plot_hybrid_weights_vs_time(weights_info, save_path=None):
    """
    Plot hybrid weights vs time, grouped by ref_point_id and timeframe
    
    Parameters:
    -----------
    weights_info : pd.DataFrame
        DataFrame with columns including 'time', 'ref_point_id', 'timeframe',
        'weight_ra_hybrid', 'weight_dec_hybrid'
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    
    # Group by ref_point_id and timeframe, calculate mean weights and time
    grouped = weights_info.groupby(['ref_point_id', 'timeframe']).agg({
        'time': 'mean',  # Use mean time for the timeframe
        'weight_ra_hybrid': 'mean',
        'weight_dec_hybrid': 'mean'
    }).reset_index()
    
    # Convert times to datetime
    grouped['datetime'] = grouped['time'].apply(epoch_to_datetime)
    
    # Get unique ref_point_ids and create color map
    unique_ids = grouped['ref_point_id'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_ids)))
    color_map = dict(zip(unique_ids, colors))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot RA weights
    for ref_id in unique_ids:
        mask = grouped['ref_point_id'] == ref_id
        ax1.scatter(grouped[mask]['datetime'], 
                   grouped[mask]['weight_ra_hybrid'],
                   color=color_map[ref_id], 
                   label=ref_id, 
                   alpha=0.6,
                   s=50)
    
    ax1.set_yscale('log')
    ax1.set_ylabel('Weight RA Hybrid', fontsize=12)
    ax1.set_title('Hybrid Weights vs Time (grouped by ID and Timeframe)', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot DEC weights
    for ref_id in unique_ids:
        mask = grouped['ref_point_id'] == ref_id
        ax2.scatter(grouped[mask]['datetime'], 
                   grouped[mask]['weight_dec_hybrid'],
                   color=color_map[ref_id], 
                   label=ref_id, 
                   alpha=0.6,
                   s=50)
    
    ax2.set_yscale('log')
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Weight DEC Hybrid', fontsize=12)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

# Usage:
# fig = plot_hybrid_weights_vs_time(weights_info, save_path='hybrid_weights_vs_time.pdf')
    
#How to get reference point of a observation list


print("#######################################################################################################")
print("ANALYSE DATA (AGAIN)")
print("#######################################################################################################")

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


# plot_all_data(summary, save_dir=out_dir)
# plot_all_ids(summary, save_dir=out_dir)

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


# # GET SPICE Results (any sim should work)
# epochs = simulations['Initial_State_No_Weights']['state_history_array'][:,0]


# GET SPICE Results (any sim should work)
epochs_full = simulations['Initial_State_No_Weights']['state_history_array'][:,0]

# Downsample: select every N-th point (adjust N for desired spacing)
# For ~4-5 hours with 1-hour spacing, use N=4 or N=5
downsample_factor = 5  # This gives you 1 point every 5 hours
epochs = epochs_full[::downsample_factor]


print(f"Original number of epochs: {len(epochs_full)}")
print(f"Downsampled number of epochs: {len(epochs)} (every {downsample_factor} hours)")


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


   
time_column = epochs.reshape(-1, 1)
states_SPICE_with_time = np.hstack((time_column, states_SPICE))

print("#######################################################################################################")
print("CREATE RSW STATES FIGS")
print("#######################################################################################################")

diff_SPICE_RSW = {}
rms_SPICE = {}
rms_Norm = {}
diff_wrt_norm_RSW = {}

states_SPICE_RSW = ProcessingUtils.rotate_inertial_3_to_rsw(time_column, states_SPICE[:,0:3], states_SPICE_with_time)

# state_history_array_norm = simulations['initial_state_only']['state_history_array']
# states_norm_RSW = ProcessingUtils.rotate_inertial_3_to_rsw(time_column, state_history_array_norm[:,1:4], state_history_array_norm)

RSW_RMS_PLOTS = True
if RSW_RMS_PLOTS == True:
    for key in simulations.keys():
        state_history_array_full = simulations[key]['state_history_array_full'][-1]
        
        # Downsample the simulation state history using the same factor
        state_history_array = state_history_array_full[::downsample_factor]
        

        states_sim_RSW_SPICE = ProcessingUtils.rotate_inertial_3_to_rsw(time_column, state_history_array[:,1:4], states_SPICE_with_time)
        #states_sim_RSW_norm = ProcessingUtils.rotate_inertial_3_to_rsw(time_column, state_history_array[:,1:4], state_history_array_norm)
        #Diff wrt to SPICE
        # #---------------------------------------------------------------------------------------
        diff = (states_SPICE_RSW - states_sim_RSW_SPICE)/1e3
        diff_SPICE_RSW[key] = diff 
        fig_RSW = FigUtils.Residuals_RSW(diff, time_column,type="difference",title=("RSW Difference SPICE - " + key))   
        fig_RSW.savefig(out_dir / ("RSW Diff SPICE - " + key + ".pdf"))

        # unweighted scalar RMS wrt SPICE
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

    stats = {}
    for key, arr in diff_SPICE_RSW.items():
        stats[key] = {
            "mean": arr.mean(axis=0),  # [mean_R, mean_S, mean_W] as array
            "std": arr.std(axis=0),    # [std_R, std_S, std_W] as array
            "rms_R": np.sqrt(np.mean(arr[:, 0]**2)),
            "rms_S": np.sqrt(np.mean(arr[:, 1]**2)),
            "rms_W": np.sqrt(np.mean(arr[:, 2]**2)),
            "max_R": np.abs(arr[:, 0]).max(),
            "max_S": np.abs(arr[:, 1]).max(),
            "max_W": np.abs(arr[:, 2]).max(),
        }


    def plot_rsw_statistics(stats):
        """
        Create a comprehensive RSW statistics comparison plot with line plots
        
        Parameters:
        -----------
        stats : dict
            Dictionary with simulation names as keys, each containing:
            - 'mean': array [mean_R, mean_S, mean_W]
            - 'std': array [std_R, std_S, std_W]
            - 'rms_R', 'rms_S', 'rms_W': scalar RMS values
            - 'max_R', 'max_S', 'max_W': scalar max values
        
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure with 3x4 subplots showing RSW statistics
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Prepare data
        sim_names = list(stats.keys())
        n_sims = len(sim_names)
        
        # Extract data for each metric
        mean_R = [stats[key]['mean'][0] for key in sim_names]
        mean_S = [stats[key]['mean'][1] for key in sim_names]
        mean_W = [stats[key]['mean'][2] for key in sim_names]
        
        std_R = [stats[key]['std'][0] for key in sim_names]
        std_S = [stats[key]['std'][1] for key in sim_names]
        std_W = [stats[key]['std'][2] for key in sim_names]
        
        rms_R = [stats[key]['rms_R'] for key in sim_names]
        rms_S = [stats[key]['rms_S'] for key in sim_names]
        rms_W = [stats[key]['rms_W'] for key in sim_names]
        
        max_R = [stats[key]['max_R'] for key in sim_names]
        max_S = [stats[key]['max_S'] for key in sim_names]
        max_W = [stats[key]['max_W'] for key in sim_names]
        
        # Create figure with 3 rows, 4 columns
        fig, axes = plt.subplots(3, 4, figsize=(20, 12))
        
        x_pos = np.arange(n_sims)
        
        # Row 0: R component (blue)
        axes[0, 0].plot(x_pos, mean_R, 'o-', color='tab:blue', linewidth=2, markersize=8)
        axes[0, 0].set_ylabel('R [km]', fontsize=11)
        axes[0, 0].set_title('Mean of Difference wrt SPICE RSW', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        
        axes[0, 1].plot(x_pos, std_R, 'o-', color='tab:blue', linewidth=2, markersize=8)
        axes[0, 1].set_ylabel('R [km]', fontsize=11)
        axes[0, 1].set_title('STD of Difference wrt SPICE RSW', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].plot(x_pos, rms_R, 'o-', color='tab:blue', linewidth=2, markersize=8)
        axes[0, 2].set_ylabel('R [km]', fontsize=11)
        axes[0, 2].set_title('RMS of Difference wrt SPICE RSW', fontsize=12, fontweight='bold')
        axes[0, 2].grid(True, alpha=0.3)
        
        axes[0, 3].plot(x_pos, max_R, 'o-', color='tab:blue', linewidth=2, markersize=8)
        axes[0, 3].set_ylabel('R [km]', fontsize=11)
        axes[0, 3].set_title('Maximum of Difference wrt SPICE RSW', fontsize=12, fontweight='bold')
        axes[0, 3].grid(True, alpha=0.3)
        
        # Row 1: S component (orange)
        axes[1, 0].plot(x_pos, mean_S, 'o-', color='tab:orange', linewidth=2, markersize=8)
        axes[1, 0].set_ylabel('S [km]', fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        
        axes[1, 1].plot(x_pos, std_S, 'o-', color='tab:orange', linewidth=2, markersize=8)
        axes[1, 1].set_ylabel('S [km]', fontsize=11)
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].plot(x_pos, rms_S, 'o-', color='tab:orange', linewidth=2, markersize=8)
        axes[1, 2].set_ylabel('S [km]', fontsize=11)
        axes[1, 2].grid(True, alpha=0.3)
        
        axes[1, 3].plot(x_pos, max_S, 'o-', color='tab:orange', linewidth=2, markersize=8)
        axes[1, 3].set_ylabel('S [km]', fontsize=11)
        axes[1, 3].grid(True, alpha=0.3)
        
        # Row 2: W component (green)
        axes[2, 0].plot(x_pos, mean_W, 'o-', color='tab:green', linewidth=2, markersize=8)
        axes[2, 0].set_ylabel('W [km]', fontsize=11)
        axes[2, 0].set_xlabel('Simulation', fontsize=11)
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        
        axes[2, 1].plot(x_pos, std_W, 'o-', color='tab:green', linewidth=2, markersize=8)
        axes[2, 1].set_ylabel('W [km]', fontsize=11)
        axes[2, 1].set_xlabel('Simulation', fontsize=11)
        axes[2, 1].grid(True, alpha=0.3)
        
        axes[2, 2].plot(x_pos, rms_W, 'o-', color='tab:green', linewidth=2, markersize=8)
        axes[2, 2].set_ylabel('W [km]', fontsize=11)
        axes[2, 2].set_xlabel('Simulation', fontsize=11)
        axes[2, 2].grid(True, alpha=0.3)
        
        axes[2, 3].plot(x_pos, max_W, 'o-', color='tab:green', linewidth=2, markersize=8)
        axes[2, 3].set_ylabel('W [km]', fontsize=11)
        axes[2, 3].set_xlabel('Simulation', fontsize=11)
        axes[2, 3].grid(True, alpha=0.3)
        
        # Set x-axis labels for all plots
        for row in range(3):
            for col in range(4):
                axes[row, col].set_xticks(x_pos)
                axes[row, col].set_xticklabels(sim_names, rotation=45, ha='right', fontsize=9)
        
        plt.tight_layout()
        
        return fig 
        
    fig = plot_rsw_statistics(stats)

    # Save it
    fig.savefig(out_dir / 'RSW_statistics_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    # Remove simulations with low RMS (initial state and rate only)
    # rms_SPICE.pop('initial_state_only')
    # rms_SPICE.pop('rot_model_rate_only')
    # rms_SPICE.pop('initial_state_only_Pole_Jacbson2009')
    # rms_SPICE.pop('rot_model_rate_only_Pole_Jacobson2009')

    # fig_RMS_SPICE_low = FigUtils.plot_rms_comparison(rms_SPICE)
    # fig_RMS_SPICE_low.savefig(out_dir / "RMS_SPICE_low_RMS_only.pdf")


    # # Compute differences for most promising simulations
    # diff_RSW = diff_SPICE_RSW['initial_state_only'] - diff_SPICE_RSW['pole_libration_amplitude']
    # fig_RSW = FigUtils.Residuals_RSW(diff_RSW, time_column,type="difference",title=("RSW Difference Initial State only vs pole libration amplitude"))   
    # fig_RSW.savefig(out_dir / ("RSW Difference Initial State only vs pole libration amplitude.pdf"))

    # diff_IAU_Jacobson2009_Pole_models_full_rot = diff_SPICE_RSW['pole_full_and_libration_amplitude_Pole_Jacobson2009'] - diff_SPICE_RSW['pole_full_and_libration_amplitude']
    # fig_RSW = FigUtils.Residuals_RSW(diff_IAU_Jacobson2009_Pole_models_full_rot, time_column,type="difference",title=("RSW Difference IAU - Jacobson 2009 full rot model"))   
    # fig_RSW.savefig(out_dir / ("RSW Difference IAU - Jacobson 2009 full rot model.pdf"))
    
    # #---------------------------------------------------------------------------------------


#Check initial state problems:
# diff =simulations['rot_model_rate_only']['state_history_array_full'][0,:,:] - simulations['GM_Triton']['state_history_array_full'][0,:,:] 
# fig_RSW = FigUtils.Residuals_RSW(diff, time_column,type="difference",title=("RSW Difference SPICE - " + key))   
# fig_RSW.savefig(out_dir / ("XYZ Diff Initial rot rate only - GM Triton.pdf"))


print("#######################################################################################################")
print("CREATE RMS PLOTS BASED ON REAL OBSERVATIONS")
print("#######################################################################################################")

rms_ra = {}
rms_dec = {}
for sim_name in simulations.keys():
    best_iteration = simulations[sim_name]['best_iteration']
    residuals = simulations[sim_name]['residual_history_arcseconds'][best_iteration]
    residuals_ra = residuals[:,1]
    residuals_dec = residuals[:,2]
    rms_ra[sim_name] = np.sqrt(np.mean(residuals_ra**2))
    rms_dec[sim_name] = np.sqrt(np.mean(residuals_dec**2))


def plot_rms_comparison(rms_ra, rms_dec):
    """Plot RMS residuals for RA and Dec as a single combined metric."""
    import matplotlib.pyplot as plt
    
    sim_names = list(rms_ra.keys())
    ra_values = np.array([rms_ra[name] for name in sim_names])
    dec_values = np.array([rms_dec[name] for name in sim_names])
    
    # Combine RA and Dec into single RMS metric
    combined_rms = np.sqrt(ra_values**2 + dec_values**2)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(sim_names))
    
    # Plot with lines and markers
    ax.plot(x, combined_rms, 'o-', markersize=8, linewidth=2, label='RMS SPICE')
    
    # Add value labels on points with more decimal places
    for i, val in enumerate(combined_rms):
        ax.text(i, val, f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('RMS [arcseconds]', fontsize=12)
    ax.set_xlabel('Estimation Scenario', fontsize=12)
    ax.set_title('RMS Comparison Across Estimation Scenarios', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(sim_names, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig

fig = plot_rms_comparison(rms_ra, rms_dec)
fig.savefig(out_dir / 'rms_comparison.pdf', dpi=300, bbox_inches='tight')

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

# print("#######################################################################################################")
# print("CREATE INTERACTIVE PLOTS FOR WEIGHTS/RESIDAULS PER FILE")
# print("#######################################################################################################")

# weight_info = pd.read_csv('Results/PoleEstimationRealObservations/WeightTest_CASE1/initial_state_tf_weights/observation_weights.csv')


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