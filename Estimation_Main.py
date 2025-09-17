# General imports
#import math

import os
import yaml
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import datetime as dt

# tudatpy imports
from tudatpy import util,math
from tudatpy import constants
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.astro import time_conversion, element_conversion
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.numerical_simulation import estimation, estimation_setup,Time
from tudatpy.astro.time_conversion import DateTime
from tudatpy.astro import frame_conversion

import ProcessingUtils
import PropFuncs

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

    settings_est = dict()
    settings_est['pseudo_observations_settings'] = pseudo_observations_settings
    settings_est['pseudo_observations'] = pseudo_observations

    estimation_output, original_parameter_vector= PropFuncs.Create_Estimation_Output(settings_est,
    system_of_bodies,propagator_settings)

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


    residuals_j2000_final = residuals_j2000[-1][:,1:4]/1e3
    residauls_rsw_final_time = residuals_rsw[-1][:,0]
    residuals_rsw_final = residuals_rsw[-1][:,1:4]/1e3
    time_format = []
    for t in residauls_rsw_final_time:
        t = DateTime.from_epoch(Time(t)).to_python_datetime()
        time_format.append(t)



    ##############################################################################################
    # PLOTTING
    ##############################################################################################

    plt.figure(figsize=(9,5))


    plt.plot(time_format,residuals_j2000_final[:,0], label='Δx')
    plt.plot(time_format,residuals_j2000_final[:,1], label='Δy')
    plt.plot(time_format,residuals_j2000_final[:,2], label='Δz')

    plt.plot(time_format,residuals_rsw_final[:,0], label='ΔR',linestyle='dashed')
    plt.plot(time_format,residuals_rsw_final[:,1], label='ΔS',linestyle='dashed')
    plt.plot(time_format,residuals_rsw_final[:,2], label='ΔW',linestyle='dashed')



    plt.xlabel('Time [s]')
    plt.ylabel('Position difference [km]')
    plt.title('Relative position in RSW')
    plt.grid(True); plt.legend(); plt.tight_layout()


    #-------------------------------------------------------------------------------


    ##############################################################################################
    # SAVE FIGS AND WRITE TO FILE
    ##############################################################################################
    
    plt.savefig(out_dir / "Residuals_RSW.pdf")

    with open(out_dir / "settings.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(settings, f, sort_keys=False, allow_unicode=True)




if __name__ == "__main__":
        
    # Define temporal scope of the simulation - equal to the time JUICE will spend in orbit around Jupiter
    simulation_start_epoch = DateTime(2025, 7,  10).epoch()
    simulation_end_epoch   = DateTime(2025, 8, 11).epoch()
    global_frame_origin = 'Neptune'
    global_frame_orientation = 'J2000'

    #--------------------------------------------------------------------------------------------
    # ENVIORONMENT SETTINGS 
    #--------------------------------------------------------------------------------------------
    settings_env = dict()
    settings_env["start_epoch"] = simulation_start_epoch.to_float()
    settings_env["end_epoch"] = simulation_end_epoch.to_float()
    settings_env["bodies"] = ['Sun','Jupiter', 'Saturn','Neptune','Triton']
    settings_env["global_frame_origin"] = global_frame_origin
    settings_env["global_frame_orientation"] = global_frame_orientation
    settings_env["interpolator_triton_cadance"] = 60*8


    #--------------------------------------------------------------------------------------------
    # ACCELERATION SETTINGS 
    #--------------------------------------------------------------------------------------------

    settings_acc = dict()
    settings_acc['bodies_to_propagate'] = ['Triton']
    settings_acc['central_bodies'] = ['Neptune']
    settings_acc['bodies_to_simulate'] = settings_env["bodies"]
    settings_acc['bodies'] = settings_env["bodies"]

    settings_acc['use_neptune_extended_gravity'] = False


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

    
    #fill in settings 
    settings = dict()
    settings["env"] = settings_env
    settings["acc"] = settings_acc
    settings["prop"] = settings_prop
    settings["obs"] = settings_obs

    from datetime import datetime
    from pathlib import Path

    def make_timestamped_folder(base_path="Results"):
        folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        full_path = Path(base_path) / folder_name
        full_path.mkdir(parents=True, exist_ok=True)
        return full_path

    
    main(settings,make_timestamped_folder())


# ##############################################################################################
# # 3D PLOT
# ##############################################################################################
# states_observations = pseudo_observations.get_observations()[0]
# states_observations_size = np.size(states_observations)/3
# states_observations = states_observations.reshape(int(states_observations_size),3)/1e3
# x1 = r_km[:,0]
# y1 = r_km[:,1]
# z1 = r_km[:,2]

# fig = plt.figure(figsize=(8, 7))
# ax = fig.add_subplot(111, projection='3d')

# # Orbit path
# ax.plot(x1, y1, z1, lw=1.5,label='Triton Orbit Fitted 2nd iteration ')
# #ax.plot(x2, y2, z2, lw=1.5, label='Triton Orbit More Planets')
# ax.plot(states_SPICE[:,0]/1e3,states_SPICE[:,1]/1e3,states_SPICE[:,2]/1e3,lw=1.5,label='SPICE Triton Oribt',linestyle='dashed')

# ax.scatter(states_observations[:,0],states_observations[:,1],states_observations[:,2],lw=1.5,label='Observations')
# # Start and end points
# ax.scatter([x1[0]], [y1[0]], [z1[0]], s=40, label='Start', marker='o')
# ax.scatter([x1[-1]], [y1[-1]], [z1[-1]], s=40, label='End', marker='^')

# # Neptune at origin
# ax.scatter([0], [0], [0], s=80, label='Neptune', marker='*')

# ax.set_xlabel('x [km]')
# ax.set_ylabel('y [km]')
# ax.set_zlabel('z [km]')
# ax.set_title('Triton orbit (Neptune-centered)')

# # Equal aspect ratio
# max_range = np.array([x1.max()-x1.min(), y1.max()-y1.min(), z1.max()-z1.min()]).max()
# mid_x = 0.5*(x1.max()+x1.min()); mid_y = 0.5*(y1.max()+y1.min()); mid_z = 0.5*(z1.max()+z1.min())
# ax.set_xlim(mid_x - 0.5*max_range, mid_x + 0.5*max_range)
# ax.set_ylim(mid_y - 0.5*max_range, mid_y + 0.5*max_range)
# ax.set_zlim(mid_z - 0.5*max_range, mid_z + 0.5*max_range)
# # try:
# #     ax.set_box_aspect([1,1,1])  # Matplotlib >=3.3
# # except Exception:
# #     pass

# ax.legend(loc='upper right')
# plt.tight_layout()
# plt.show()
