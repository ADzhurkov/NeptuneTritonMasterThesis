import numpy as np
from pathlib import Path
from typing import Dict
from datetime import datetime, timedelta
import sys
import json
import pandas as pd
import matplotlib
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import matplotlib.cm as cm

import pickle
import os 
import yaml

from datetime import datetime, timedelta
import matplotlib.dates as mdates


from tudatpy.interface import spice

from tudatpy.astro import time_conversion, element_conversion,frame_conversion
from tudatpy.astro.time_conversion import DateTime
from tudatpy import numerical_simulation
import tudatpy
from tudatpy import util
from tudatpy.dynamics import parameters_setup
from tudatpy.dynamics import simulator

from tudatpy.estimation import estimation_analysis

# Get the path to the directory containing this file
current_dir = Path(__file__).resolve().parent

# Append the HelperFunctions directory
sys.path.append(str(current_dir / "HelperFunctions"))

import MainPostprocessing as PostProc


import ProcessingUtils
import PropFuncs
import FigUtils
import ObsFunc
import nsdc


J2000_EPOCH = datetime(2000, 1, 1, 12, 0, 0)


def make_timestamped_folder(base_path="Results"):
    folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_path = Path(base_path) / folder_name
    full_path.mkdir(parents=True, exist_ok=True)
    return full_path

def RunSinglePropagation(settings: dict, out_dir,folder_name=None):

    ##############################################################################################
    # CREATE ENVIRONMENT  
    ##############################################################################################

    body_settings,system_of_bodies = PropFuncs.Create_Env(settings["env"])

    ##############################################################################################
    # CREATE ACCELERATION MODELS  
    ##############################################################################################

    acceleration_models,accelerations_cfg = PropFuncs.Create_Acceleration_Models(settings["acc"],system_of_bodies)

    ##############################################################################################
    # CREATE PROPAGATOR
    ##############################################################################################

    propagator_settings = PropFuncs.Create_Propagator_Settings(settings["prop"],acceleration_models)

    ##############################################################################################
    # CREATE & RUN ESTIMATOR  
    ##############################################################################################
    
    # Create initial state variation equation
    parameter_settings = parameters_setup.initial_states(propagator_settings, system_of_bodies)


    # Create the parameters that will be estimated
    parameters_to_estimate = parameters_setup.create_parameter_set(parameter_settings, system_of_bodies)

    print("################################################################################")
    print("START OF SIMULATION")
    print("################################################################################")
    
    # dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    # system_of_bodies, propagator_settings)
   
    # Create the variational equation solver and propagate the dynamics
    variational_equations_solver = simulator.create_variational_equations_solver(
        system_of_bodies, propagator_settings, parameters_to_estimate, simulate_dynamics_on_creation=True
    )

    print("################################################################################")
    print("END OF SIMULATION")
    print("################################################################################")

    ##############################################################################################
    # SAVE PROPAGATION RESULTS  
    ##############################################################################################
    
    # state_history = dynamics_simulator.propagation_results.state_history
    # state_history_array = util.result2array(state_history)
    
    # dep_vars_history = dynamics_simulator.propagation_results.dependent_variable_history
    # dep_vars_history_array = util.result2array(dep_vars_history)


    # Extract the resulting state history, state transition matrix history, and sensitivity matrix history
    state_history = variational_equations_solver.state_history
    state_transition_interface = variational_equations_solver.state_transition_interface
    
    state_history_array = util.result2array(state_history)

    output_times = state_history_array[:,0]

    epochs = output_times


    if folder_name is not None:
        out_dir = out_dir / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)


    # Save residuals as numpy files

    np.save(out_dir / "state_history_array.npy", state_history_array)


    # Get Neptune rotation model
    nep_rot_model = system_of_bodies.get("Neptune").rotation_model

    # Generate all rotation matrices at once
    rotation_matrices = np.array([
        nep_rot_model.body_fixed_to_inertial_rotation(epoch) 
        for epoch in epochs
    ])

    # Save rotation matricies in the approritate folder
    np.save(out_dir / "rotation_matricies.npy", rotation_matrices)


    #np.save(out_dir / "delta_initial_state_array.npy", delta_initial_state_array)
    #np.save(out_dir / "state_transition_matrices.npy", state_transition_matrices)
    #np.save(out_dir / "covariances_array.npy",covariances)
    #np.save(out_dir / "formal_errors_array.npy",formal_errors1)

    #Convert initial state to list to be able to save to yaml
    #settings["prop"]["initial_state"] = settings["prop"]["initial_state"].tolist()
    #settings['prop']['initial_covariance'] = settings['prop']['initial_covariance'].tolist()
    #settings["prop"]["initial_state_uncertanity"] = settings["prop"]["initial_state_uncertanity"].tolist()
    
    #Save yaml settings file
    with open(out_dir / "settings.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(settings, f, sort_keys=False, allow_unicode=True)
    
    return state_history_array


if __name__ == "__main__":
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



    # Define temporal scope of the simulation - equal to the time JUICE will spend in orbit around Jupiter
    simulation_start_epoch = DateTime(2000, 1,  1).epoch()  #1963, 1,  1  
    simulation_end_epoch   = DateTime(2025, 1, 1).epoch()   #2006, 9, 2 2019, 10, 1
    
    simulation_initial_epoch = DateTime(2006, 10, 1).epoch()
    global_frame_origin = 'SSB'
    global_frame_orientation = 'J2000' # ECLIPJ2000 default frame switch when done with experimenting !!

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
    settings_prop['fixed_step_size'] = 60*60 # 60 minutes
    

    settings = dict()
    settings["env"] = settings_env
    settings["acc"] = settings_acc
    settings["prop"] = settings_prop
    
    ################################################################################################
    # Load initial states and other stuff 
    ################################################################################################
    
    out_dir = make_timestamped_folder("Results/PoleSimObservations/SinglePropAnalysis")
    

    runSimulationsBetter = True
    if runSimulationsBetter == True:
        # Define all experiment variants here only once
        VARIANTS = {
            "simple_rot_model": {
                "simulation_path": "Results/PoleSimObservations/SinglePropAnalysis/J2000/simple_rot_model",
                'rotation_model': None
            }, 
            "full_spice_rot_model": {
                "simulation_path": "Results/PoleSimObservations/SinglePropAnalysis/J2000/full_spice_rot_model",
                'rotation_model': "spice"
            },
            "IAU_rot_model": {
                "simulation_path": "Results/PoleSimObservations/SinglePropAnalysis/J2000/IAU_rot_model",
                'rotation_model': "IAU2015"
            },                              
        }


        # # Load estimations
        # estimations = {
        #     name: PostProc.load_npy_files(cfg["estimation_path"])
        #     for name, cfg in VARIANTS.items()
        # }

        runSim = False
        if runSim == True:
            results = {}
            for name, content in VARIANTS.items():
                settings['env']['Neptune_rot_model_type'] = content['rotation_model'] 
                sim_array = RunSinglePropagation(settings, out_dir, name)
                results[name] = sim_array


        # Load simulations
        simulations = {
            name: PostProc.load_npy_files(cfg["simulation_path"])
            for name, cfg in VARIANTS.items()
        }


    print("#######################################################################################################")
    print("CREATE SPICE STATES")
    print("#######################################################################################################")

    # GET SPICE Results (any sim should work)
    epochs = simulations['simple_rot_model']['state_history_array'][:,0]
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
    
    time_column = simulations['simple_rot_model']['state_history_array'][:,[0]]  # keep it 2D 
    states_SPICE_with_time = np.hstack((time_column, states_SPICE))

    print("#######################################################################################################")
    print("CREATE RSW STATES FIGS")
    print("#######################################################################################################")

    diff_SPICE_RSW = {}
    max_formal_errors_RSW = {}
    max_formal_errors = {}
    initial_formal_errors = {}
    number_of_matches = {}

    for key in simulations.keys():
        state_history_array = simulations[key]['state_history_array']

        states_SPICE_RSW = ProcessingUtils.rotate_inertial_3_to_rsw(time_column, states_SPICE[:,0:3], state_history_array)
        states_sim_RSW = ProcessingUtils.rotate_inertial_3_to_rsw(time_column, state_history_array[:,1:4], state_history_array)

        diff = (states_SPICE_RSW - states_sim_RSW)/1e3
        diff_SPICE_RSW[key] = diff 
        fig_RSW = FigUtils.Residuals_RSW(diff, time_column,type="difference",title=("RSW Difference SPICE - " + key))   
        fig_RSW.savefig(out_dir / ("RSW Diff SPICE - " + key + ".pdf"))

        # #---------------------------------------------------------------------------------------

    diff_wrt_norm_RSW = {}
    number_of_matches_norm = {}
    state_history_array_norm = simulations['simple_rot_model']['state_history_array']
    states_norm_RSW = ProcessingUtils.rotate_inertial_3_to_rsw(time_column, state_history_array_norm[:,1:4], state_history_array_norm)
    #parameters_norm  = estimations['id_rmse_weights_descaled']['final_paramaters'] 
    for key in simulations.keys():
        if key is not 'id_rmse_weights_descaled':
            state_history_array = simulations[key]['state_history_array']
            states_sim_RSW = ProcessingUtils.rotate_inertial_3_to_rsw(time_column, state_history_array[:,1:4], state_history_array_norm)

            # parameters_sim  = estimations[key]['final_paramaters']      # note the key name as provided
            # sigma   = estimations[key]['formal_errors']

            # pos_diff, vel_diff = (parameters_norm[:3] - parameters_sim[:3])/1e3, parameters_norm[3:] - parameters_sim[3:]
            # pos_err, vel_err = sigma[:3]/1e3, sigma[3:]
            # number_of_matches_norm[key] = np.sum(abs(pos_err)>abs(pos_diff))+ np.sum(abs(vel_err)>abs(vel_diff))


            diff = (states_norm_RSW- states_sim_RSW)/1e3
            diff_wrt_norm_RSW[key] = diff 
            fig_RSW = FigUtils.Residuals_RSW(diff, time_column,type="difference",title=("RSW Difference Norm - " + key))   
            fig_RSW.savefig(out_dir / ("RSW Diff Norm_ " + key + ".pdf"))


            




    print("#######################################################################################################")
    print("COMPUTE EULER ANGLES")
    print("#######################################################################################################")
    def plot_euler_angles(W_deg, alpha_deg, delta_deg, time, title=None):
        """
        Plot Euler angles W, alpha, and delta over time.
        
        Parameters:
        -----------
        W_deg : array
            Prime meridian angle in degrees
        alpha_deg : array
            Right ascension of rotation pole in degrees
        delta_deg : array
            Declination of rotation pole in degrees
        time : array
            Time array (assumed to be in seconds since J2000)
        title : str, optional
            Custom title for the figure
        
        Returns:
        --------
        fig : matplotlib figure
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime, timedelta
        
        # Convert time to datetime
        time_dt = FigUtils.ConvertToDateTime(time)
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        
        # Labels
        labels = [r'$W$ (Prime Meridian)', r'$\alpha$ (RA of Pole)', r'$\delta$ (Dec of Pole)']
        data = [W_deg, alpha_deg, delta_deg]
        
        # Get colors
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        # Plot each angle
        for i, ax in enumerate(axes):
            ax.plot(time_dt, data[i], color=colors[i], linewidth=1.5)
            ax.set_ylabel(f'{labels[i]} [deg]', fontsize=12)
            ax.grid(True, alpha=0.3)
        
        # X-axis formatting
        axes[-1].set_xlabel('Time', fontsize=12)
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        axes[-1].xaxis.set_major_locator(locator)
        axes[-1].xaxis.set_major_formatter(formatter)
        
        # Title
        if title is not None:
            fig.suptitle(title, fontsize=14)
        else:
            fig.suptitle('Euler Angles from IAU Rotation Model', fontsize=14)
        
        fig.tight_layout()
        
        return fig



    from scipy.spatial.transform import Rotation as R

    W_deg = {}
    alpha_deg = {}
    delta_deg = {}
    for key in simulations.keys():
        rotation_matrices = simulations[key]['rotation_matricies']

        # Convert all matrices at once
        rot = R.from_matrix(rotation_matrices)

        # Extract Euler angles for all matrices (returns radians by default)
        euler_zxz = rot.as_euler('zxz', degrees=False)  # shape: (n_epochs, 3)

        # Recover W, α, δ for all epochs at once (vectorized)
        W = euler_zxz[:, 2]                    # Prime meridian angle
        alpha = euler_zxz[:, 0] - np.pi/2      # Right ascension of pole
        delta = np.pi/2 - euler_zxz[:, 1]      # Declination of pole

        # Convert W, α, δ in degrees
        W_deg[key] = np.rad2deg(W)
        alpha_deg[key] = np.rad2deg(alpha)
        delta_deg[key] = np.rad2deg(delta)

        fig = plot_euler_angles(W_deg[key], alpha_deg[key], delta_deg[key], time_column, title=("Euler angles - " + key))
        fig.savefig(out_dir / ("Euler_Angles_ " + key + ".pdf"))


    def plot_rotation_matrix_difference(rot_matrices_1, rot_matrices_2, time, 
                                        label1="Model 1", label2="Model 2", title=None):
        """
        Plot the difference between two sets of rotation matrices.
        
        Parameters:
        -----------
        rot_matrices_1, rot_matrices_2 : array
            Arrays of rotation matrices with shape (n, 3, 3)
        time : array
            Time array (seconds since J2000)
        label1, label2 : str
            Labels for the two models
        title : str, optional
            Custom title for the figure
        
        Returns:
        --------
        fig : matplotlib figure
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime, timedelta
        import numpy as np
        
        # Convert time to datetime
        time_dt = FigUtils.ConvertToDateTime(time)
        
        # Calculate element-wise difference
        diff = rot_matrices_1 - rot_matrices_2
        
        # Create 3x3 subplot grid
        fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
        
        # Element labels
        row_labels = ['X', 'Y', 'Z']
        col_labels = ['X', 'Y', 'Z']
        
        # Plot each difference
        for i in range(3):
            for j in range(3):
                ax = axes[i, j]
                
                # Extract element (i,j) difference from all matrices
                diff_values = diff[:, i, j]
                
                ax.plot(time_dt, diff_values, linewidth=1.5, color='red')
                ax.set_ylabel(f'ΔR[{row_labels[i]},{col_labels[j]}]', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
                
                # Auto-scale y-axis based on data
                max_abs = np.max(np.abs(diff_values))
                if max_abs > 0:
                    ax.set_ylim(-1.1*max_abs, 1.1*max_abs)
        
        # X-axis formatting (only bottom row)
        for ax in axes[2, :]:
            ax.set_xlabel('Time', fontsize=10)
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
        
        # Title
        if title is not None:
            fig.suptitle(title, fontsize=14)
        else:
            fig.suptitle(f'Rotation Matrix Difference: {label1} - {label2}', fontsize=14)
        
        fig.tight_layout()
        
        return fig



    rotation_matrices_norm = simulations['simple_rot_model']['rotation_matricies']
    
    for key in simulations.keys():
        if key != 'simple_rot_model':    
            W_deg_diff = W_deg[key] - W_deg['simple_rot_model']
            alpha_deg_diff = alpha_deg[key] - alpha_deg['simple_rot_model']
            delta_deg_diff = delta_deg[key] - delta_deg['simple_rot_model']
            
            fig = plot_euler_angles(W_deg_diff, alpha_deg_diff, delta_deg_diff, time_column, title=("Diff Euler angles: " + key + "- simple_rot_model"))
            fig.savefig(out_dir / ("Diff_Euler_Angles_ " + key + ".pdf"))
            
            rotation_matrices_current = simulations[key]['rotation_matricies']

            fig = plot_rotation_matrix_difference(rotation_matrices_current, rotation_matrices_norm, time_column, 
                                        label1=key, label2="simple_rot_model")
            fig.savefig(out_dir / ("Diff_Rot_Matrices_ " + key + ".pdf"))

 
    #uncertanity_states = simulations["timeframe_weights_10mas_min"]['delta_initial_state_array']/1e3


