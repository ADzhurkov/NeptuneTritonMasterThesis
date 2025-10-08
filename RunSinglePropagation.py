
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
from tudatpy import math
from tudatpy import constants

from tudatpy.interface import spice
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
import tudatpy.estimation
from tudatpy import util

from tudatpy import numerical_simulation

from tudatpy.astro import time_conversion, element_conversion,frame_conversion
from tudatpy.astro.time_conversion import DateTime


from tudatpy.data import save2txt


import ProcessingUtils
import PropFuncs
import FigUtils

matplotlib.use("PDF")  #tkagg
def RunSinglePropagation(settings: dict, out_dir):
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
    
    print("################################################################################")
    print("START OF SIMULATION")
    print("################################################################################")
    
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    system_of_bodies, propagator_settings)
    
    print("################################################################################")
    print("END OF SIMULATION")
    print("################################################################################")

    ##############################################################################################
    # SAVE PROPAGATION RESULTS  
    ##############################################################################################
    
    state_history = dynamics_simulator.propagation_results.state_history
    state_history_array = util.result2array(state_history)
    
    dep_vars_history = dynamics_simulator.propagation_results.dependent_variable_history
    dep_vars_history_array = util.result2array(dep_vars_history)

    out_dir = out_dir / str(settings["prop"]["fixed_step_size"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save residuals as numpy files
    # arr = np.stack(residuals_rsw, axis=0)   # shape (5, 254, 4)
    np.save(out_dir / "state_history_array.npy", state_history_array)
    np.save(out_dir / "state_history.npy", state_history)

    # arr2 = np.stack(residuals_j2000,axis=0)
    np.save(out_dir / "dep_vars_history_array.npy", dep_vars_history_array)
    np.save(out_dir / "dep_vars_history.npy", dep_vars_history_array)
    

    #Save yaml settings file
    with open(out_dir / "settings.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(settings, f, sort_keys=False, allow_unicode=True)
    
    return state_history,state_history_array


def main(settings: dict,out_dir):
    print("Running Main File...")



    ##############################################################################################
    # RETRIEVE INFO
    ##############################################################################################

    state_history = dynamics_simulator.propagation_results.state_history
    state_history_array = util.result2array(state_history)
    
    dep_vars_history = dynamics_simulator.propagation_results.dependent_variable_history
    dep_vars_history_array = util.result2array(dep_vars_history)



    ##############################################################################################
    # GET SPICE OBSERVATIONS
    ##############################################################################################

    fixed_step_size = settings_prop["fixed_step_size"]

    # Get Triton's state relative to Neptune SPICE
    epochs = np.arange(simulation_start_epoch, simulation_end_epoch+60*5, fixed_step_size ) #test_settings_obs["cadence"]
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
    # GET RSW POSITIONS
    ##############################################################################################
    rsw_state = ProcessingUtils.rotate_inertial_3_to_rsw(state_history_array[:,0],
                                            state_history_array[:,1:4], state_history)



    ##############################################################################################
    # PLOTTING
    ##############################################################################################
    rsw_fig = FigUtils.Residuals_RSW(rsw_state, state_history_array[:,0])
    cartesian_fig = FigUtils.Residuals_RSW(state_history_array[:,1:4],state_history_array[:,0])

    ##############################################################################################
    # SAVE FIGS AND WRITE TO FILE
    ##############################################################################################
    
    rsw_fig.savefig(out_dir / "RSW.pdf")
    cartesian_fig.savefig(out_dir / "Cartesian.pdf")
    #rms_fig.savefig(out_dir / "rms.pdf")
    
    #Orbit_3D_fig.savefig(out_dir / "Orbit_3D.pdf")
    # fft_fig_Jonas.savefig(out_dir / "fft_Jonas.pdf")
    # fft_fig_Welch.savefig(out_dir / "fft_Welch.pdf")
    # fft_fig_PSD_no_detrend.savefig(out_dir / "fft_no_detrend.pdf")
    # fft_fig_PSD_linear_detrend.savefig(out_dir / "fft_linear_detrend.pdf")
    # fft_fig_Spectrum.savefig(out_dir / "fft_spectrum.pdf")
    # #----------------------------------------------------------------------------------------------
    # Save residuals as numpy files
    # arr = np.stack(residuals_rsw, axis=0)   # shape (5, 254, 4)
    # np.save(out_dir / "residuals_rsw.npy", arr)

    # arr2 = np.stack(residuals_j2000,axis=0)
    # np.save(out_dir / "residuals_j2000.npy", arr2)
    



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
    settings_env["start_epoch"] = simulation_start_epoch
    settings_env["end_epoch"] = simulation_end_epoch
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
  
    
    #fill in settings 
    settings = dict()
    settings["env"] = settings_env
    settings["acc"] = settings_acc
    settings["prop"] = settings_prop



    #--------------------------------------------------------------------------------------------
    # RUN SIM AND SAVE IN FOLDER
    #--------------------------------------------------------------------------------------------

    out_dir = make_timestamped_folder("Results_Step_Size/")
    
    state_history = dict()
    state_history_array = dict()

    step_sizes = [1800*16,1800*8,1800*4,1800*2,1800,1800/2]
    label_list = [f"{int(step_sizes[i])}-{int(step_sizes[i+1])}" 
              for i in range(len(step_sizes)-1)]

    # for step_size in step_sizes:

    #     settings["prop"]["fixed_step_size"] = step_size
    #     print("Current Step Size: ",settings["prop"]["fixed_step_size"])
        
    #     state_history[str(step_size)], state_history_array[str(step_size)] = RunSinglePropagation(settings, out_dir)

    # np.save(out_dir / "state_history.npy", state_history)
    # np.save(out_dir / "state_history_array.npy", state_history_array)
    
    state_history = np.load("/home/atn/Documents/Year 5/Thesis/Github/NeptuneTritonMasterThesis/Results_Step_Size/10Years/state_history.npy",allow_pickle=True).item()
    state_history_array = np.load("/home/atn/Documents/Year 5/Thesis/Github/NeptuneTritonMasterThesis/Results_Step_Size/10Years/state_history_array.npy",allow_pickle=True).item()

    

    
    #--------------------------------------------------------------------------------------------   
    # PLOTTING
    #--------------------------------------------------------------------------------------------

    fig_compare, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    
    diff_mag = dict()
    fits = {}

    for i in range(len(step_sizes)-1):
        # Grab the state for this array + next
        if step_sizes[i] != 3600:
            continue

        arr1 = state_history_array[str(step_sizes[i])]
        
        arr2 = state_history_array[str(step_sizes[i+1])]
        arr2_dict = state_history[str(step_sizes[i+1])]
        
        arr2_interpolator = math.interpolators.create_one_dimensional_vector_interpolator(
        arr2_dict, math.interpolators.lagrange_interpolation(8))

        arr1_times = arr1[:,0]
        arr2_interpolated = []
        for epoch in arr1_times:
            arr2_interpolated.append(arr2_interpolator.interpolate(epoch))

        diff = arr2_interpolated - arr1[:,1:]
        diff = diff[:,0:3]
        fig_compare,ax = FigUtils.Residuals_Magnitude_Compare(diff, arr1_times,
                        fig=fig_compare, ax=ax,
                        label_prefix=label_list[i],
                        overlay_idx=i)

        
        diff_mag[str(step_sizes[i])] = np.linalg.norm(diff, axis=1)

        
        label = str(step_sizes[i])
        time = state_history_array[str(step_sizes[i])][:,0]
        y = diff_mag[str(step_sizes[i])]

        # Linear regression: y ≈ m*x + b
        m, b = np.polyfit(time, y, 1)
        fits[label] = (m, b)
        print(f"{label}: slope = {m:.3e}, intercept = {b:.3e}")
        m, b = fits[label]
        fig_compare,ax = FigUtils.Plot_Polynomial_Fit(m,b,time,fig_compare,ax,label)

        #fig_pos = FigUtils.Plot_Components(diff[:,0:3],arr1_times)
        #fig_vel = FigUtils.Plot_Components(diff[:,3:],arr1_times)
        
        #file_name = str(step_sizes[i])
        #fig_pos.savefig(out_dir / (file_name + "Positions.pdf"))
        #fig_vel.savefig(out_dir / (file_name + "Velocities.pdf"))
      
    fig_compare.savefig(out_dir / "Compare_Positions.pdf")


    state_history = np.load("/home/atn/Documents/Year 5/Thesis/Github/NeptuneTritonMasterThesis/Results_Step_Size/20Years/state_history.npy",allow_pickle=True).item()
    state_history_array = np.load("/home/atn/Documents/Year 5/Thesis/Github/NeptuneTritonMasterThesis/Results_Step_Size/20Years/state_history_array.npy",allow_pickle=True).item()

    

    
    #--------------------------------------------------------------------------------------------   
    # PLOTTING
    #--------------------------------------------------------------------------------------------

    fig_compare, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    
    diff_mag = dict()
    fits_next = {}
    fig_test, ax_test = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for i in range(len(step_sizes)-1):
        if step_sizes[i] != 3600:
            continue
        # Grab the state for this array + next
        arr1 = state_history_array[str(step_sizes[i])]
        
        arr2 = state_history_array[str(step_sizes[i+1])]
        arr2_dict = state_history[str(step_sizes[i+1])]
        
        arr2_interpolator = math.interpolators.create_one_dimensional_vector_interpolator(
        arr2_dict, math.interpolators.lagrange_interpolation(8))

        arr1_times = arr1[:,0]
        arr2_interpolated = []
        for epoch in arr1_times:
            arr2_interpolated.append(arr2_interpolator.interpolate(epoch))

        diff = arr2_interpolated - arr1[:,1:]
        diff = diff[:,0:3]
        fig_compare,ax = FigUtils.Residuals_Magnitude_Compare(diff, arr1_times,
                        fig=fig_compare, ax=ax,
                        label_prefix=label_list[i],
                        overlay_idx=i)

        
        diff_mag[str(step_sizes[i])] = np.linalg.norm(diff, axis=1)

        
        label = str(step_sizes[i])
        time = state_history_array[str(step_sizes[i])][:,0]
        y = diff_mag[str(step_sizes[i])]

        # Linear regression: y ≈ m*x + b
        m, b = np.polyfit(time, y, 1)
        fits_next[label] = (m, b)
        print(f"{label}: slope = {m:.3e}, intercept = {b:.3e}")
        
        fig_compare,ax = FigUtils.Plot_Polynomial_Fit(m,b,time,fig_compare,ax,label)
        
        m, b_false = fits[label]
        fig_compare,ax = FigUtils.Plot_Polynomial_Fit(m,b,time,fig_compare,ax,(label+'10 years'))
       
            
        ax.legend(loc='lower right')

        #fig_pos = FigUtils.Plot_Components(diff[:,0:3],arr1_times)
        #fig_vel = FigUtils.Plot_Components(diff[:,3:],arr1_times)
        
        #file_name = str(step_sizes[i])
        #fig_pos.savefig(out_dir / (file_name + "Positions.pdf"))
        #fig_vel.savefig(out_dir / (file_name + "Velocities.pdf"))
      
    fig_compare.savefig(out_dir / "Compare_Positions_20_Years_polyfit.pdf")





    # #--------------------------------------------------------------------------------------------
    # # Error with respect to lowest timestep
    # #--------------------------------------------------------------------------------------------
    
    # arr2 = state_history_array[str(step_sizes[-1])]
    # arr2_dict = state_history[str(step_sizes[-1])]
    
    # arr2_interpolator = math.interpolators.create_one_dimensional_vector_interpolator(
    # arr2_dict, math.interpolators.lagrange_interpolation(8))
    # accumulated_difference = np.zeros(len(step_sizes)-1)
    # for i in range(len(step_sizes)-1):
    #     # Grab the state for this array + next
    #     arr1 = state_history_array[str(step_sizes[i])]
        

    #     arr1_last_time = arr1[-1,0]
    #     arr2_interpolated = arr2_interpolator.interpolate(arr1_last_time)
    #     diff = arr2_interpolated - arr1[-1,1:]
    #     accumulated_difference[i] = np.linalg.norm(diff[0:3])


    # fig_accumulated, ax_acc = plt.subplots(figsize=(10, 5), constrained_layout=True)

    # # Drop the last step size so arrays align
    # ax_acc.plot(step_sizes[:-1], accumulated_difference, marker="o", linestyle="-")

    # # Log scale for y-axis
    # ax_acc.set_yscale("log")

    # # Labels
    # ax_acc.set_xlabel("Step size [sec]")
    # ax_acc.set_ylabel("Accumulated Difference at Simulation End [m]")

    # # Optional: grid for readability
    # ax_acc.grid(True, which="both", alpha=0.3)

    # fig_accumulated.savefig(out_dir / "Accumulated_Differences.pdf")