# General imports
#import math

import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

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

import Utilities

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

# for k in kernel_paths:
#     spice.load_kernel(k)

# Load your kernels
for k in kernel_paths:
    spice.load_kernel(os.path.join(kernel_folder, k))


# Define temporal scope of the simulation - equal to the time JUICE will spend in orbit around Jupiter
simulation_start_epoch = DateTime(2025, 8,  10).epoch()
simulation_end_epoch   = DateTime(2025, 8, 11).epoch()
global_frame_origin = 'Neptune'
global_frame_orientation = 'J2000'


##############################################################################################
# CREATE ENVIRONMENT  
##############################################################################################
# settings = settings_dict["env"]

settings_env = dict()
settings_env["bodies"] = ['Sun','Jupiter', 'Saturn','Neptune','Triton']
settings_env["global_frame_origin"] = global_frame_origin
settings_env["global_frame_orientation"] = global_frame_orientation
settings_env["interpolator_triton_cadance"] = 60*8

body_settings,system_of_bodies = Utilities.Create_Env(settings_env,simulation_start_epoch,simulation_end_epoch)


##############################################################################################
# CREATE ACCELERATION SETTINGS 
##############################################################################################

settings_acc = dict()
settings_acc['bodies_to_propagate'] = ['Triton']
settings_acc['central_bodies'] = ['Neptune']
settings_acc['bodies_to_simulate'] = settings_env["bodies"]
settings_acc['bodies'] = settings_env["bodies"]
settings_acc['system_of_bodies'] = system_of_bodies
settings_acc['use_neptune_extended_gravity'] = False

acceleration_models,acceleration_settings = Utilities.Create_Acceleration_Models(settings_acc)


##############################################################################################
# PROPAGATOR
##############################################################################################
#includes: 
# simulation start,end, initial_state,dependend variables, integrator,termination conditions

settings_prop = dict()
settings_prop['start_epoch'] = simulation_start_epoch
settings_prop['end_epoch'] = simulation_end_epoch
settings_prop['bodies_to_propagate'] = 'Triton'
settings_prop['central_bodies'] = 'Neptune'
settings_prop['global_frame_orientation'] = global_frame_orientation
settings_prop['fixed_step_size'] = 60*30 # 30 minutes
settings_prop['acceleration_models'] = acceleration_models

propagator_settings = Utilities.Create_Propagator_Settings(settings_prop)


##############################################################################################
# Create Settings Dictionary
##############################################################################################


settings_obs = dict()
settings_obs["mode"] = ["pos"]
settings_obs["bodies"] = [("Triton", "Neptune")]                           # bodies to observe
settings_obs["times"] = "180yrs"
settings_obs["cadence"] = 60*60*3 # Every 3 hours

settings = dict()
settings["obs"] = settings_obs
##############################################################################################
# PSEUDO OBSERVATIONS 
##############################################################################################

pseudo_observations, pseudo_observations_settings = Utilities.make_relative_position_pseudo_observations(
    simulation_start_epoch,simulation_end_epoch, system_of_bodies, settings)


########################################################################################################
#### ESTIMATOR    ######################################################################################

settings_est = dict()
settings_est['pseudo_observations_settings'] = pseudo_observations_settings
settings_est['pseudo_observations'] = pseudo_observations


estimation_output, original_parameter_vector= Utilities.Create_Estimation_Output(settings_est,system_of_bodies,propagator_settings)



print("END OF ESTIMATION")
########################################################################################################
#### RETRIEVE INFO    ##################################################################################
def rotate_inertial_3_to_rsw(epochs, inertial_3, state_history):

    R_func = make_rsw_rotation_from_state_history(state_history, size=3)
    rsw_coll = []

    for epoch, inertial in zip(epochs, inertial_3):

        R = R_func(epoch)
        rsw = R.dot(inertial)
        rsw_coll.append(rsw)

    assert np.array(rsw_coll).shape == inertial_3.shape

    return np.array(rsw_coll)

def format_residual_history(residual_history, obs_times, state_history):
    residuals_per_iteration = []
    rsw_residuals_per_iteration = []

    for i in range(residual_history.shape[1]):
        res_i = residual_history[:, i]
        reshaped_residuals = res_i.reshape(-1, 3)
        residuals_per_iteration.append(np.hstack([np.array(obs_times).reshape(-1, 1), reshaped_residuals]))

        # rsw_residuals = rotate_inertial_3_to_rsw(np.array(obs_times).reshape(-1, 1),
        #                                          reshaped_residuals, state_history)

        # rsw_residuals_per_iteration.append(np.hstack([np.array(obs_times).reshape(-1, 1), rsw_residuals]))

    return residuals_per_iteration, rsw_residuals_per_iteration





state_history = estimation_output.simulation_results_per_iteration[0].dynamics_results.state_history_float
state_history_array = util.result2array(state_history)

residuals_j2000, residuals_rsw = format_residual_history(estimation_output.residual_history,
                                                            pseudo_observations.get_concatenated_float_observation_times(),
                                                            state_history)

dep_vars_history = estimation_output.simulation_results_per_iteration[0].dynamics_results.dependent_variable_history_float
dep_vars_array = util.result2array(dep_vars_history)

#print('residuals: ',residuals_j2000)




########################################################################################################
#### RETRIEVE RSW RESIDUALS   ##########################################################################
def extract_rsw_rot_matrices(R_flat):
    # Take 9 columns and reshape to (N,3,3)
    #R_flat = dep_vars_array[:, start:start+9]     # (N, 9)
    R_rsw_to_I = R_flat.reshape((-1, 3, 3))       # (N, 3, 3)
    return R_rsw_to_I


# EXTRACT RSW MATRIX
R_rsw_to_I = extract_rsw_rot_matrices(dep_vars_array[:,1:10])
R_I_to_rsw = np.transpose(R_rsw_to_I, axes=(0,2,1))   # inertial→RSW

# Get Triton's state relative to Neptune
epochs = np.arange(simulation_start_epoch.to_float(), simulation_end_epoch.to_float()+60*5, fixed_step_size.to_float() ) #test_settings_obs["cadence"]

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

state_history_time =state_history_array[:,0]
time_format = []
for t in state_history_time:
    t = DateTime.from_epoch(Time(t)).to_python_datetime()
    time_format.append(t)

r_RSW = np.einsum('nij,nj->ni', R_I_to_rsw, state_history_array[:,1:4])
v_RSW = np.einsum('nij,nj->ni', R_I_to_rsw, state_history_array[:,4:])

r_SPICE_RSW = np.einsum('nij,nj->ni', R_I_to_rsw, states_SPICE[:,1:4])
#dv_SPICE_RSW = np.einsum('nij,nj->ni', R_I_to_rsw, dv_SPICE)




r_km = state_history_array[:,1:4]/1e3

# plt.figure(figsize=(9,5))

# plt.plot(time_format,r_RSW[:,0], label='ΔR')
# plt.plot(time_format,r_RSW[:,1], label='ΔS')
# plt.plot(time_format,r_RSW[:,2], label='ΔW')

# plt.plot(time_format,r_SPICE_RSW[:,0], label='ΔR SPICE')
# plt.plot(time_format,r_SPICE_RSW[:,1], label='ΔS SPICE')
# plt.plot(time_format,r_SPICE_RSW[:,2], label='ΔW SPICE')


# plt.xlabel('Time [s]')
# plt.ylabel('Position difference [km]')
# plt.title('Relative position in RSW (deputy − chief)')
# plt.grid(True); plt.legend(); plt.tight_layout()


##############################################################################################
# RESIDUAL PLOT
##############################################################################################

def format_residual_history(residual_history, obs_times, state_history):
    residuals_per_iteration = []
    rsw_residuals_per_iteration = []

    for i in range(residual_history.shape[1]):
        res_i = residual_history[:, i]
        reshaped_residuals = res_i.reshape(-1, 3)
        residuals_per_iteration.append(np.hstack([np.array(obs_times).reshape(-1, 1), reshaped_residuals]))

        rsw_residuals = rotate_inertial_3_to_rsw(np.array(obs_times).reshape(-1, 1),
                                                 reshaped_residuals, state_history)

        rsw_residuals_per_iteration.append(np.hstack([np.array(obs_times).reshape(-1, 1), rsw_residuals]))

    return residuals_per_iteration, rsw_residuals_per_iteration



def make_rsw_rotation_from_state_history(state_history, size=3):

    if type(state_history) == dict:
        state_history_dict = state_history
    else:
        state_history_dict = array_to_dict(state_history)

    state_history_interpolator = math.interpolators.create_one_dimensional_vector_interpolator_from_float(
        state_history_dict, math.interpolators.lagrange_interpolation(8))

    def rsw_rotation_at_epoch(sample_epoch):

        R = frame_conversion.inertial_to_rsw_rotation_matrix(state_history_interpolator.interpolate(sample_epoch))

        if size == 3:
            pass

        elif size == 6:
            R = linalg.block_diag(R, R)

        else:
            raise NotImplementedError("Stacking of rotation matrices over size 6 not implemented.")

        return R

    return rsw_rotation_at_epoch



def rotate_inertial_3_to_rsw(epochs, inertial_3, state_history):

    R_func = make_rsw_rotation_from_state_history(state_history, size=3)
    rsw_coll = []

    for epoch, inertial in zip(epochs, inertial_3):

        R = R_func(epoch)
        rsw = R.dot(inertial)
        rsw_coll.append(rsw)

    assert np.array(rsw_coll).shape == inertial_3.shape

    return np.array(rsw_coll)


def rotate_inertial_6_to_rsw(epochs, inertial_6, state_history):

    R_func = make_rsw_rotation_from_state_history(state_history, size=6)
    rsw_coll = []

    for epoch, inertial in zip(epochs, inertial_6):

        R = R_func(epoch)
        rsw = R.dot(inertial)
        rsw_coll.append(rsw)

    assert np.array(rsw_coll).shape == inertial_6.shape

    return np.array(rsw_coll)



def array_to_dict(state_history_array):

    state_history_dict = dict()

    for row in state_history_array:

        state_history_dict[row[0]] = row[1:7]

    return state_history_dict

residuals_j2000, residuals_rsw = format_residual_history(estimation_output.residual_history,
                                                            pseudo_observations.get_concatenated_float_observation_times(),
                                                            state_history)


residuals_rsw_final = residuals_j2000[-1][:,1:4]
residuals_rsw_final_time = residuals_j2000[-1][:,0]

residuals_rsw_inital = residuals_j2000[0][:,1:4]
residuals_rsw_inital_time = residuals_j2000[0][:,0]

#Is residual history already in kms? 
final_residuals = estimation_output.residual_history[:,-1]
final_residuals_array = final_residuals.reshape(78,3)


initial_residuals = estimation_output.residual_history[:,0]
initial_residuals_array = final_residuals.reshape(78,3)


plt.figure(figsize=(9,5))
plt.plot(residuals_rsw_final_time,residuals_rsw_final[:,0], label='Δx')
plt.plot(residuals_rsw_final_time,residuals_rsw_final[:,1], label='Δy')
plt.plot(residuals_rsw_final_time,residuals_rsw_final[:,2], label='Δz')


plt.plot(residuals_rsw_inital_time,residuals_rsw_inital[:,0], label='Δx',linestyle="dashed")
plt.plot(residuals_rsw_inital_time,residuals_rsw_inital[:,1], label='Δy',linestyle="dashed")
plt.plot(residuals_rsw_inital_time,residuals_rsw_inital[:,2], label='Δz',linestyle="dashed")


# plt.plot(residuals_rsw_final_time,final_residuals_array[:,0], label='Δx',linestyle="dashed")
# plt.plot(residuals_rsw_final_time,final_residuals_array[:,1], label='Δy',linestyle="dashed")
# plt.plot(residuals_rsw_final_time,final_residuals_array[:,2], label='Δz',linestyle="dashed")

# plt.plot(residuals_rsw_final_time,initial_residuals_array[:,0], label='Δx inital',linestyle="dashed")
# plt.plot(residuals_rsw_final_time,initial_residuals_array[:,1], label='Δy initial',linestyle="dashed")

# plt.plot(residuals_rsw_final_time,initial_residuals_array[:,2], label='Δz initial',linestyle="dashed")



# plt.plot(time_format,r_SPICE_RSW[:,0], label='ΔR SPICE')
# plt.plot(time_format,r_SPICE_RSW[:,1], label='ΔS SPICE')
# plt.plot(time_format,r_SPICE_RSW[:,2], label='ΔW SPICE')


plt.xlabel('Time [s]')
plt.ylabel('Position difference [km]')
plt.title('Relative position residual [km]')
plt.grid(True); plt.legend(); plt.tight_layout()





##############################################################################################
# 3D PLOT
##############################################################################################
states_observations = pseudo_observations.get_observations()[0]
states_observations = states_observations.reshape(78,3)/1e3
x1 = r_km[:,0]
y1 = r_km[:,1]
z1 = r_km[:,2]

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection='3d')

# Orbit path
ax.plot(x1, y1, z1, lw=1.5,label='Triton Orbit Fitted 2nd iteration ')
#ax.plot(x2, y2, z2, lw=1.5, label='Triton Orbit More Planets')
ax.plot(states_SPICE[:,0]/1e3,states_SPICE[:,1]/1e3,states_SPICE[:,2]/1e3,lw=1.5,label='SPICE Triton Oribt',linestyle='dashed')

ax.scatter(states_observations[:,0],states_observations[:,1],states_observations[:,2],lw=1.5,label='Observations')
# Start and end points
ax.scatter([x1[0]], [y1[0]], [z1[0]], s=40, label='Start', marker='o')
ax.scatter([x1[-1]], [y1[-1]], [z1[-1]], s=40, label='End', marker='^')

# Neptune at origin
ax.scatter([0], [0], [0], s=80, label='Neptune', marker='*')

ax.set_xlabel('x [km]')
ax.set_ylabel('y [km]')
ax.set_zlabel('z [km]')
ax.set_title('Triton orbit (Neptune-centered)')

# Equal aspect ratio
max_range = np.array([x1.max()-x1.min(), y1.max()-y1.min(), z1.max()-z1.min()]).max()
mid_x = 0.5*(x1.max()+x1.min()); mid_y = 0.5*(y1.max()+y1.min()); mid_z = 0.5*(z1.max()+z1.min())
ax.set_xlim(mid_x - 0.5*max_range, mid_x + 0.5*max_range)
ax.set_ylim(mid_y - 0.5*max_range, mid_y + 0.5*max_range)
ax.set_zlim(mid_z - 0.5*max_range, mid_z + 0.5*max_range)
# try:
#     ax.set_box_aspect([1,1,1])  # Matplotlib >=3.3
# except Exception:
#     pass

ax.legend(loc='upper right')
plt.tight_layout()
plt.show()
