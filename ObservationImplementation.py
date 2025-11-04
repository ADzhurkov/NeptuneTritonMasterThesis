# General imports
#import math

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

matplotlib.use("PDF")  #tkagg

def main(settings: dict,out_dir):
    print("Running Main File...")

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

    ##############################################################################################
    # CREATE ENVIRONMENT  
    ##############################################################################################

    body_settings,system_of_bodies = PropFuncs.Create_Env(settings_env)

    ##############################################################################################
    # CREATE ACCELERATION MODELS  
    ##############################################################################################

    acceleration_models,accelerations_cfg = PropFuncs.Create_Acceleration_Models(settings_acc,system_of_bodies)

    ##############################################################################################
    # CREATE PROPAGATOR
    ##############################################################################################

    propagator_settings = PropFuncs.Create_Propagator_Settings(settings_prop,acceleration_models)

    ##############################################################################################
    # CREATE PSEUDO OR LOAD REAL OBSERVATIONS
    ##############################################################################################
    if settings["obs"]["type"] == "Simulated":
        observations,observations_settings = PropFuncs.make_relative_position_pseudo_observations(
            simulation_start_epoch,simulation_end_epoch, system_of_bodies, settings)
    elif settings["obs"]["type"] == "Real":
        
        files = settings["obs"]["files"]
        weights = settings["obs"]["weights"]
        observations,observations_settings,observation_set_ids = ObsFunc.LoadObservations("Observations/ProcessedOutliers/",system_of_bodies,file_names_loaded,weights = weights)

        #observations,observations_settings,observation_set_ids = ObsFunc.LoadObservations("Observations/ProcessedOutliers/",system_of_bodies,files)
        Observatories = []
        for id in observation_set_ids:
            Observatory =  id.split("_")[0]
            Observatories.append(Observatory)
        print("observation_set_ids: ",observation_set_ids)
    else:
        print("No Observation type selected")
    ##############################################################################################
    # CREATE & RUN ESTIMATOR  
    ##############################################################################################


    estimation_output, original_parameter_vector= PropFuncs.Create_Estimation_Output(settings_est,
    system_of_bodies,propagator_settings,observations_settings,observations)

    print("END OF ESTIMATION")

    ##############################################################################################
    # RETRIEVE INFO
    ##############################################################################################

    state_history = estimation_output.simulation_results_per_iteration[-1].dynamics_results.state_history_float
    state_history_array = util.result2array(state_history)

    state_history_initial = estimation_output.simulation_results_per_iteration[0].dynamics_results.state_history_float
    state_history_initial_array = util.result2array(state_history_initial)

    dep_vars_history = estimation_output.simulation_results_per_iteration[-1].dynamics_results.dependent_variable_history
    dep_vars_array = util.result2array(dep_vars_history)

    ##############################################################################################
    # GET SPICE OBSERVATIONS
    ##############################################################################################

    fixed_step_size = settings_prop["fixed_step_size"]
    
    # Get Triton's state relative to Neptune SPICE
    epochs = state_history_array[:,0]  #np.arange(simulation_start_epoch, simulation_end_epoch+60*5, fixed_step_size ) #test_settings_obs["cadence"]
    states_SPICE = np.array([
        spice.get_body_cartesian_state_at_epoch(
            target_body_name="Triton",
            observer_body_name="Neptune",
            reference_frame_name="ECLIPJ2000",
            aberration_corrections="NONE",
            ephemeris_time=epoch
        )
        for epoch in epochs
    ])
  

    ##############################################################################################
    # EXTRACTING RESIDUALS, PLOTTING AND SAVING DATA AND FIGS
    ##############################################################################################

    if settings['obs']['type'] == 'Real':
        print("Saving real observations residuals...")
        
        residual_history = ProcessingUtils.format_residual_history_abs_astrometric(
            estimation_output.residual_history,
            observations.get_concatenated_observation_times()
        )

        # residual_history_arcseconds = residual_history
        # for i in range(np.shape(residual_history)[0]):
        #     #residual_history_arcseconds.append(residual_history[i])
        #     residual_history_arcseconds[i][:,1:] = residual_history_arcseconds[i][:,1:]*3600 * 180/np.pi
        
        residual_history_arcseconds = [
        np.column_stack([arr[:,0], arr[:,1:] * 3600 * 180/np.pi]) 
        for arr in residual_history
        ]

        observation_times = observations.get_concatenated_observation_times()
        

        residuals_ra_initial_arcseconds = residual_history_arcseconds[0][:,1]
        residuals_dec_initial_arcseconds = residual_history_arcseconds[0][:,2]
        
        residuals_ra_arcseconds = residual_history_arcseconds[-1][:,1]
        residuals_dec_arcseconds = residual_history_arcseconds[-1][:,2]
        
        ##############################################################################################
        # RESIDUALS RA / DEC
        ##############################################################################################

        observation_times_DateFormat = FigUtils.ConvertToDateTime(observation_times)
                
        residuals = ObsFunc.Get_SPICE_residual_from_observations(observations,Observatories,system_of_bodies)
        residuals_RA_SPICE = residuals[0]
        residuals_DEC_SPICE = residuals[1]

        fig_estimation_residuals = FigUtils.plot_RA_DEC_residuals(
            observation_times_DateFormat,
            residuals_ra_arcseconds,
            residuals_ra_initial_arcseconds,
            residuals_dec_arcseconds,
            residuals_dec_initial_arcseconds)

        labels = ['final','SPICE']
        fig_SPICE_residuals = FigUtils.plot_RA_DEC_residuals(
            observation_times_DateFormat,
            residuals_ra_arcseconds,
            residuals_RA_SPICE,
            residuals_dec_arcseconds,
            residuals_DEC_SPICE,
            labels)


        #--------------------------------------------------------------------------------
        # Extract the time column (first column)
        time_column = state_history_array[:, [0]]   # keep it 2D (shape = (289, 1))
        states_SPICE_with_time = np.hstack((time_column, states_SPICE))

        fig_Cartesian = FigUtils.PlotCartesianDifference(state_history_array, states_SPICE_with_time)
        

        fig_rms = FigUtils.Residuals_RMS(residual_history_arcseconds)


        states_SPICE_RSW = ProcessingUtils.rotate_inertial_3_to_rsw(time_column, states_SPICE[:,0:3], state_history_array)
        states_sim_RSW = []
        states_sim_RSW.append(ProcessingUtils.rotate_inertial_3_to_rsw(time_column, state_history_initial_array[:,1:4], state_history_array))
        states_sim_RSW.append(ProcessingUtils.rotate_inertial_3_to_rsw(time_column, state_history_array[:,1:4], state_history_array))

        diff_sim_RSW = (states_sim_RSW[0] - states_sim_RSW[-1])/1e3
        diff_sim_SPICE_RSW = (states_sim_RSW[0] - states_SPICE_RSW)/1e3
        diff_final_sim_SPICE_RSW = (states_sim_RSW[-1] - states_SPICE_RSW)/1e3

        fig_sim_RSW = FigUtils.Residuals_RSW(diff_sim_RSW, time_column,type="difference",title="RSW Difference Initial Final")
        fig_sim_SPICE_rsw = FigUtils.Residuals_RSW(diff_sim_SPICE_RSW, time_column,type="difference",title="RSW Difference Initial SPICE")
        fig_final_sim_SPICE_rsw = FigUtils.Residuals_RSW(diff_final_sim_SPICE_RSW, time_column,type="difference",title="RSW Difference Final SPICE")
            
        fig_sim_RSW_abs = FigUtils.Residuals_RSW(states_sim_RSW[-1], time_column,type="normal",title="RSW Final Simulation Absolute")
        fig_SPICE_RSW_abs = FigUtils.Residuals_RSW(states_SPICE_RSW, time_column,type="normal",title="RSW SPICE Absolute")
        #--------------------------------------------------------------------------------
        #save figs
        fig_rms.savefig(out_dir / "Residual_RMS.pdf")
        fig_sim_RSW.savefig(out_dir / "RSW_diff_Sim.pdf")
        fig_sim_SPICE_rsw.savefig(out_dir / "RSW_diff_initial_Sim_SPICE.pdf")
        fig_final_sim_SPICE_rsw.savefig(out_dir / "RSW_diff_final_Sim_SPICE.pdf")
        
        #fig_sim_RSW_abs.savefig(out_dir / "RSW_abs_final_Sim.pdf")
        #fig_SPICE_RSW_abs.savefig(out_dir / "RSW_abs_SPICE.pdf")
        
        
        
        
        fig_Cartesian.savefig(out_dir / "Cartesian_Difference_SPICE.pdf")
        fig_estimation_residuals.savefig(out_dir / "Estimation_residual.pdf")
        fig_SPICE_residuals.savefig(out_dir / "SPICE_residual.pdf")
        #fig_RA.savefig(out_dir / "RA_res.pdf")
        #fig_DEC.savefig(out_dir / "DEC_res.pdf")

        arr = np.stack(residual_history_arcseconds, axis=0)   # shape ??
        np.save(out_dir / "residual_history_arcseconds.npy", arr)

      
        arr1 = np.stack(state_history_array, axis=0)
        np.save(out_dir / "state_history_array.npy", arr1)

        arr2 = np.stack(state_history_initial_array, axis=0)
        np.save(out_dir / "state_history_initial_array.npy", arr2)

        arr3 = np.stack(states_SPICE_with_time, axis=0)
        np.save(out_dir / "states_SPICE_with_time.npy", arr3)

        arr4 = np.stack(observation_set_ids, axis=0)
        np.save(out_dir / "observation_set_ids.npy", arr4)

        
        observations_sorted = observations.get_observations()
        np.save(out_dir / "observations_sorted.npy",np.array(observations_sorted,dtype=object))

        np.save(out_dir / "observation_times.npy",np.array(observation_times))

        residuals_sorted = observations.get_residuals()
        np.save(out_dir / "residuals_sorted.npy",np.array(residuals_sorted,dtype=object))
    
    elif settings['obs']['type'] == 'Simulated':
        print("Saving simulated observations residuals...")
        residuals_j2000, residuals_rsw = ProcessingUtils.format_residual_history(estimation_output.residual_history,
                                                                observations.get_concatenated_observation_times(),
                                                                state_history)

        ##############################################################################################
        #PLOTTING
        ##############################################################################################
        
        residuals_j2000_final = residuals_j2000[-1][:,1:4]/1e3
        residauls_rsw_final_time = residuals_rsw[-1][:,0]
        residuals_rsw_final = residuals_rsw[-1][:,1:4]/1e3
        
        residuals_rsw_fig = FigUtils.Residuals_RSW(residuals_rsw_final, residauls_rsw_final_time)

        #-------------------------------------------------------------------------------
        rms_fig = FigUtils.Residuals_RMS(residuals_j2000)

        #--------------------------------------------------------------------------
        #Get Different Flavors of FFTs
        #--------------------------------------------------------------------------
        #Get Triton Mean Motion
        kep = dep_vars_array[:, 10:16]  # [a, e, i, ω, Ω, ν]
        a = kep[:, 0]                   # meters
        # GM values (Tudat):
        mu_N = spice.get_body_gravitational_parameter("Neptune")
        mu_T = spice.get_body_gravitational_parameter("Triton")
        mu = mu_N + mu_T               
        # Mean motion time series (rad/s) 
        n_series = np.sqrt(mu / a**3)
        n_med = np.nanmedian(n_series)  # one way to take the mean of the mean motion
        f_rot_hz = 1/(n_med / (2*np.pi)) #  T (seconds)

        fft_fig_Jonas = FigUtils.create_fft_residual_figure(residuals_rsw[-1],f_rot_hz)

        fft_fig_Spectrum = FigUtils.periodogram_rsw(residuals_rsw[-1],1/f_rot_hz,mode='spectrum')


        ##############################################################################################
        #SAVE FIGS AND WRITE TO FILE
        ##############################################################################################
        residuals_rsw_fig.savefig(out_dir / "Residuals_RSW.pdf")
        rms_fig.savefig(out_dir / "rms.pdf")
        #Orbit_3D_fig.savefig(out_dir / "Orbit_3D.pdf")
        fft_fig_Jonas.savefig(out_dir / "fft_Jonas.pdf")

        fft_fig_Spectrum.savefig(out_dir / "fft_spectrum.pdf")
        #----------------------------------------------------------------------------------------------
        # Save residuals as numpy files
        arr = np.stack(residuals_rsw, axis=0)   # shape (5, 254, 4)
        np.save(out_dir / "residuals_rsw.npy", arr)

        arr2 = np.stack(residuals_j2000,axis=0)
        np.save(out_dir / "residuals_j2000.npy", arr2)
        



    #Save yaml settings file
    with open(out_dir / "settings.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(settings, f, sort_keys=False, allow_unicode=True)



def make_timestamped_folder(base_path="Results"):
    folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_path = Path(base_path) / folder_name
    full_path.mkdir(parents=True, exist_ok=True)
    return full_path

if __name__ == "__main__":
        
    # Define temporal scope of the simulation - equal to the time JUICE will spend in orbit around Jupiter
    simulation_start_epoch = DateTime(1986, 1,  1).epoch() #2006, 8,  27 1963, 3,  4  
    simulation_end_epoch   = DateTime(2020, 1, 1).epoch()   #2006, 9, 2 2019, 10, 1
    
    simulation_initial_epoch = DateTime(2007, 1, 1).epoch()
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
    settings_prop['fixed_step_size'] = 60*60 # 30 minutes
  
    #--------------------------------------------------------------------------------------------
    # OBSERVATION SETTINGS 
    #--------------------------------------------------------------------------------------------

    settings_obs = dict()
    settings_obs["mode"] = ["pos"]
    settings_obs["bodies"] = [("Triton", "Neptune")]                           # bodies to observe
    settings_obs["cadence"] = 60*60*3 # Every 3 hours
    settings_obs["type"] = "Real" # Simulated or Real observations

    # --- Load ---
    with open("file_names.json", "r") as f:
        file_names_loaded = json.load(f)

    settings_obs["files"] = file_names_loaded             #["Triton_286_nm0084.csv","Triton_337_nm0088.csv","Triton_673_nm0079"] #"Triton_337_nm0088.csv","Triton_689_nm0078.csv","Triton_689_nm0077.csv","Triton_689_nm0007.csv"]
   
    #weights = pd.read_csv("Results/BetterFigs/Outliers/PostProcessing/First/FirstWeights/weights.txt", sep="\t")
    # LOAD — set that column back as the index
    weights = pd.read_csv(
        "Results/BetterFigs/Outliers/PostProcessing/First/FirstWeights/weights.txt",
        sep="\t",
        index_col="id"       # <-- important
    )

    settings_obs["weights"] = weights
   
    #--------------------------------------------------------------------------------------------
    # OBSERVATION SETTINGS 
    #--------------------------------------------------------------------------------------------

    settings_est = dict()
    #settings_est['pseudo_observations_settings'] = pseudo_observations_settings
    #settings_est['pseudo_observations'] = pseudo_observations
    settings_est['est_parameters'] = ['initial_state'] #,'Rotation_Pole_Position_Neptune'] #, 'Rotation_Pole_Position_Neptune']


    
    #fill in settings 
    settings = dict()
    settings["env"] = settings_env
    settings["acc"] = settings_acc
    settings["prop"] = settings_prop
    settings["obs"] = settings_obs
    settings["est"] = settings_est
   

    main(settings,make_timestamped_folder("Results/BetterFigs/Weights"))


    #path_list = ["PoleOrientation/SimpleRotationModel/residuals_rsw.npy","PoleOrientation/EstimationSimpleRotationModel/residuals_rsw.npy"]
    #label_list = ["No Estimation","Estimated Simple Rotational Model"]
    #fig,fig_diff = FigUtils.Compare_RSW_Different_Solutions(path_list,label_list)

    #fig.savefig("R_RSW_Estimation_Comparison.pdf")
    #fig_diff.savefig("Diff_R_RSW_Estimation_Comparison.pdf")


    #----------------------------------------------------------------------------
