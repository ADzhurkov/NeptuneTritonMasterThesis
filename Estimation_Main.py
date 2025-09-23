# General imports
#import math

import os
import yaml
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from datetime import datetime
from pathlib import Path

# tudatpy imports
from tudatpy import util,math
from tudatpy import constants

from tudatpy.interface import spice
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.numerical_simulation import estimation, estimation_setup,Time


from tudatpy import numerical_simulation

from tudatpy.astro import time_conversion, element_conversion,frame_conversion
from tudatpy.astro.time_conversion import DateTime


from tudatpy.data import save2txt


import ProcessingUtils
import PropFuncs
import FigUtils

matplotlib.use("PDF")  #tkagg

def main(settings: dict,out_dir):
    print("Running Main File...")

    ##############################################################################################
    # LOAD SPICE KERNELS
    ##############################################################################################
    kernel_folder = "Kernels/"
    kernel_paths=[
        "pck00010.tpc",
        "gm_de440.tpc",
        "nep097.bsp",     
        "nep105.bsp",
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
    # CREATE PSEUDO OBSERVATIONS 
    ##############################################################################################

    pseudo_observations, pseudo_observations_settings = PropFuncs.make_relative_position_pseudo_observations(
        simulation_start_epoch.to_float(),simulation_end_epoch.to_float(), system_of_bodies, settings)

    ##############################################################################################
    # CREATE & RUN ESTIMATOR  
    ##############################################################################################


    estimation_output, original_parameter_vector= PropFuncs.Create_Estimation_Output(settings_est,
    system_of_bodies,propagator_settings,pseudo_observations_settings,pseudo_observations)

    print("END OF ESTIMATION")

    ##############################################################################################
    # RETRIEVE INFO
    ##############################################################################################

    state_history = estimation_output.simulation_results_per_iteration[-1].dynamics_results.state_history_float
    state_history_array = util.result2array(state_history)

    dep_vars_history = estimation_output.simulation_results_per_iteration[-1].dynamics_results.dependent_variable_history_float
    dep_vars_array = util.result2array(dep_vars_history)

    ##############################################################################################
    # GET SPICE OBSERVATIONS
    ##############################################################################################

    fixed_step_size = settings_prop["fixed_step_size"]

    # Get Triton's state relative to Neptune SPICE
    epochs = np.arange(simulation_start_epoch.to_float(), simulation_end_epoch.to_float()+60*5, fixed_step_size ) #test_settings_obs["cadence"]
    states_SPICE = np.array([
        spice.get_body_cartesian_state_at_epoch(
            target_body_name="Triton",
            observer_body_name="Neptune",
            reference_frame_name="J2000",
            aberration_corrections="NONE",
            ephemeris_time=epoch
        )
        for epoch in epochs
    ])
  

    ##############################################################################################
    # RESIDUALS JONAS
    ##############################################################################################

    residuals_j2000, residuals_rsw = ProcessingUtils.format_residual_history(estimation_output.residual_history,
                                                                pseudo_observations.get_concatenated_float_observation_times(),
                                                                state_history)




    ##############################################################################################
    # PLOTTING
    ##############################################################################################
    
    residuals_j2000_final = residuals_j2000[-1][:,1:4]/1e3
    residauls_rsw_final_time = residuals_rsw[-1][:,0]
    residuals_rsw_final = residuals_rsw[-1][:,1:4]/1e3
    
    residuals_rsw_fig = FigUtils.Residuals_RSW(residuals_rsw_final, residauls_rsw_final_time)

    #-------------------------------------------------------------------------------
    rms_fig = FigUtils.Residuals_RMS(residuals_j2000)


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
    
    #--------------------------------------------------------------------------
    #Get Different Flavors of FFTs
    #--------------------------------------------------------------------------
    fft_fig_Jonas = FigUtils.create_fft_residual_figure(residuals_rsw[-1],f_rot_hz)
    
    fft_fig_Welch = FigUtils.psd_rsw(residuals_rsw[-1],1/f_rot_hz) 
    #fft_fig3 = FigUtils.psd_rsw(residuals_rsw[-1],1/f_rot_hz,type='ASD')
    
    fft_fig_PSD_no_detrend = FigUtils.periodogram_rsw(residuals_rsw[-1],1/f_rot_hz,detrend=False)                       
    fft_fig_PSD_linear_detrend = FigUtils.periodogram_rsw(residuals_rsw[-1],1/f_rot_hz,detrend='linear') 
    fft_fig_Spectrum = FigUtils.periodogram_rsw(residuals_rsw[-1],1/f_rot_hz,mode='spectrum')


    ##############################################################################################
    # SAVE FIGS AND WRITE TO FILE
    ##############################################################################################
    
    residuals_rsw_fig.savefig(out_dir / "Residuals_RSW.pdf")
    rms_fig.savefig(out_dir / "rms.pdf")
    #Orbit_3D_fig.savefig(out_dir / "Orbit_3D.pdf")
    fft_fig_Jonas.savefig(out_dir / "fft_Jonas.pdf")
    fft_fig_Welch.savefig(out_dir / "fft_Welch.pdf")
    fft_fig_PSD_no_detrend.savefig(out_dir / "fft_no_detrend.pdf")
    fft_fig_PSD_linear_detrend.savefig(out_dir / "fft_linear_detrend.pdf")
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
    simulation_start_epoch = DateTime(2025, 7,  10).epoch()
    simulation_end_epoch   = DateTime(2025, 8, 11).epoch()
    global_frame_origin = 'SSB'
    global_frame_orientation = 'J2000'

    #--------------------------------------------------------------------------------------------
    # ENVIORONMENT SETTINGS 
    #--------------------------------------------------------------------------------------------
    settings_env = dict()
    settings_env["start_epoch"] = simulation_start_epoch.to_float()
    settings_env["end_epoch"] = simulation_end_epoch.to_float()
    settings_env["bodies"] = ['Sun','Jupiter', 'Saturn','Neptune','Triton','Uranus','Mercury','Venus','Mars','Earth']
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
    settings_acc['bodies_to_simulate'] = settings_env["bodies"]
    settings_acc['bodies'] = settings_env["bodies"]

    settings_acc['neptune_extended_gravity'] = "Jacobson2009"


    accelerations_cfg = PropFuncs.build_acceleration_config(settings_acc)
    settings_acc['accelerations_cfg'] = accelerations_cfg
    #--------------------------------------------------------------------------------------------
    # PROPAGATOR SETTINGS 
    #--------------------------------------------------------------------------------------------

    settings_prop = dict()
    settings_prop['start_epoch'] = settings_env["start_epoch"]
    settings_prop['end_epoch'] = settings_env["end_epoch"]

    settings_prop['bodies_to_propagate'] = settings_acc['bodies_to_propagate'] 
    settings_prop['central_bodies'] = settings_acc['central_bodies']
    settings_prop['global_frame_orientation'] = settings_env["global_frame_orientation"]
    settings_prop['fixed_step_size'] = 60*30 # 30 minutes
  
    #--------------------------------------------------------------------------------------------
    # OBSERVATION SETTINGS 
    #--------------------------------------------------------------------------------------------

    settings_obs = dict()
    settings_obs["mode"] = ["pos"]
    settings_obs["bodies"] = [("Triton", "Neptune")]                           # bodies to observe
    settings_obs["cadence"] = 60*60*3 # Every 3 hours

    #--------------------------------------------------------------------------------------------
    # OBSERVATION SETTINGS 
    #--------------------------------------------------------------------------------------------

    settings_est = dict()
    #settings_est['pseudo_observations_settings'] = pseudo_observations_settings
    #settings_est['pseudo_observations'] = pseudo_observations
    settings_est['est_parameters'] = ['initial_state','Rotation_Pole_Position_Neptune']


    
    #fill in settings 
    settings = dict()
    settings["env"] = settings_env
    settings["acc"] = settings_acc
    settings["prop"] = settings_prop
    settings["obs"] = settings_obs
    settings["est"] = settings_est



    #main(settings,make_timestamped_folder("Results/PoleOrientation"))


    path_list = ["PoleOrientation/SimpleRotationModel/residuals_rsw.npy","PoleOrientation/EstimationSimpleRotationModel/residuals_rsw.npy"]
    label_list = ["No Estimation","Estimated Simple Rotational Model"]
    fig,fig_diff = FigUtils.Compare_RSW_Different_Solutions(path_list,label_list)

    fig.savefig("R_RSW_Estimation_Comparison.pdf")
    fig_diff.savefig("Diff_R_RSW_Estimation_Comparison.pdf")


    #----------------------------------------------------------------------------
