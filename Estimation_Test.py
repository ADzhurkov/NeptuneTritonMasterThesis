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
simulation_start_epoch = DateTime(2025, 7,  2).epoch()
simulation_end_epoch   = DateTime(2025, 8, 20).epoch()
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
# CREATE PROPAGATOR SETTINGS 
##############################################################################################

settings_prop = dict()
settings_prop['bodies_to_propagate'] = ['Triton']
settings_prop['central_bodies'] = ['Neptune']
settings_prop['bodies_to_simulate'] = settings_env["bodies"]
settings_prop['bodies'] = settings_env["bodies"]
settings_prop['system_of_bodies'] = system_of_bodies
settings_prop['use_neptune_extended_gravity'] = False



acceleration_models = Utilities.Create_Propagation_Settings(settings_prop)





##############################################################################################
# SET SIMULATION START AND END, initial_state,dependend_vars, integrator,termination_codns
##############################################################################################

simulation_start_epoch = DateTime(2025,8, 1).epoch()
simulation_end_epoch   = DateTime(2025, 8, 11).epoch()


# 2) Get Triton’s state w.r.t. Neptune in J2000 at your epoch (seconds TDB from J2000)
initial_state = spice.get_body_cartesian_state_at_epoch(
    target_body_name       = "Triton",
    observer_body_name     = "Neptune",
    reference_frame_name   = "J2000",
    aberration_corrections = "none",
    ephemeris_time         = simulation_start_epoch.to_float(),
)

dependent_variables_to_save = [
    propagation_setup.dependent_variable.rsw_to_inertial_rotation_matrix("Triton", "Neptune"),
    #propagation_setup.dependent_variable.total_acceleration("Triton"),
    propagation_setup.dependent_variable.keplerian_state("Triton", "Neptune"),
    propagation_setup.dependent_variable.latitude("Triton", "Neptune"),
    propagation_setup.dependent_variable.longitude("Triton", "Neptune"),
    
]


# Create termination settings
termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch.to_float())

# Create numerical integrator settings
fixed_step_size = Time(60*30) # 30 minutes
integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
    fixed_step_size, coefficient_set=propagation_setup.integrator.CoefficientSets.rk_4
)

# Create propagation settings
propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    initial_state,
    simulation_start_epoch,
    integrator_settings,
    termination_condition,
    output_variables=dependent_variables_to_save
)


##############################################################################################
# Create Settings Dictionary
##############################################################################################

# test_settings_env = dict()
# test_settings_env["bodies"] = []                                        # bodies to create, default
# test_settings_env["Triton"] = ['Durante_2019', 'noe', 'synchronous_forced']       # gravity_model, ephemeris, rotation_model
# test_settings_env["Neptune"] = ['Lainey_2018', 'IMCCE']                  # gravity_model, rotation_model

# test_settings_prop = dict()
# test_settings_prop["bodies_to_propagate"] = ["Triton"]
# test_settings_prop["central_bodies"] = ["Neptune"]
# test_settings_prop["timestep"] = 60*30                              # fixed integration timestep
# test_settings_prop["times"] = "180yrs"                               # where do simulation times come from, default
# test_settings_prop["dep_vars"] = "standard" #"dissipation"                       # "dissipation"
# test_settings_prop["dissipation"] = True


test_settings_obs = dict()
test_settings_obs["mode"] = ["pos"]
test_settings_obs["bodies"] = [("Triton", "Neptune")]                           # bodies to observe
test_settings_obs["times"] = "180yrs"
test_settings_obs["cadence"] = 60*60*3 # Every 3 hours

# test_settings_est = dict()
# test_settings_est['gravity'] = []
# test_settings_est['dissipation'] = True

# test_settings_plot = dict()
# test_settings_plot["dep"] = []                                      # [lat1, lon1] --> Titan on Saturn []
#                                                                     # [lat2, lon2] --> Saturn on Titan
# test_settings_plot["res"] = ["rsw", "fft"]
# test_settings_plot["other"] = ["correlations"]
# test_settings_plot["savefigs"] = True


# test_settings_write = dict()
# test_settings_write["res"] = ["rsw", "fft"]
# test_settings_write["dep"] = ["a"] #, "dissipation"]  # , "dissipation"]                              # "dissipation",
# test_settings_write["params"] = True


test_settings = dict()
# test_settings["env"] = test_settings_env
#test_settings["prop"] = test_settings_prop
test_settings["obs"] = test_settings_obs
# test_settings["est"] = test_settings_est
# test_settings["write"] = test_settings_write
# test_settings["plot"] = test_settings_plot




##############################################################################################
# PSEUDO OBSERVATIONS 
##############################################################################################


def make_relative_position_pseudo_observations(start_epoch,end_epoch, bodies, settings_dict: dict):
    settings = settings_dict['obs']
    cadence = settings['cadence']

    observation_start_epoch = start_epoch.to_float()
    observation_end_epoch = end_epoch.to_float()
    
    observation_times = np.arange(observation_start_epoch + cadence, observation_end_epoch - cadence, cadence)

    observation_times = np.array([Time(t) for t in observation_times])
    observation_times_test = observation_times[0].to_float()
    bodies_to_observe = settings["bodies"]
    relative_position_observation_settings = []

    for obs_bodies in bodies_to_observe:
        link_ends = {
            estimation_setup.observation.observed_body: estimation_setup.observation.body_origin_link_end_id(obs_bodies[0]),
            estimation_setup.observation.observer: estimation_setup.observation.body_origin_link_end_id(obs_bodies[1])}
        link_definition = estimation_setup.observation.LinkDefinition(link_ends)


        relative_position_observation_settings.append(estimation_setup.observation.relative_cartesian_position(link_definition))

        observation_simulation_settings = [estimation_setup.observation.tabulated_simulation_settings(
                estimation_setup.observation.relative_position_observable_type,
                link_definition,
                observation_times,
                reference_link_end_type=estimation_setup.observation.observed_body)]


    # Create observation simulators
    ephemeris_observation_simulators = estimation_setup.create_observation_simulators(
        relative_position_observation_settings, bodies)

    # Get ephemeris states as ObservationCollection
    print('Checking spice for position pseudo observations...')
    simulated_pseudo_observations = estimation.simulate_observations(
        observation_simulation_settings,
        ephemeris_observation_simulators,
        bodies)
    #print("Test")
    return simulated_pseudo_observations, relative_position_observation_settings



pseudo_observations, pseudo_observations_settings = make_relative_position_pseudo_observations(simulation_start_epoch,simulation_end_epoch, bodies, test_settings)




########################################################################################################
#### ESTIMATOR    ######################################################################################

parameters_to_estimate_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)

# if "Saturn_mass" in settings_dict['est']['gravity'] or "Saturn_GM" in settings_dict['est']['gravity']:
#     parameters_to_estimate_settings.append(estimation_setup.parameter.gravitational_parameter(
#         "Saturn"
#     ))

#if settings_dict['est']['dissipation']:
#    parameters_to_estimate_settings.append(estimation_setup.parameter.inverse_tidal_quality_factor(
#        "Saturn", "Titan"
#    ))

# if settings_dict['est']['dissipation']:
#     parameters_to_estimate_settings.append(estimation_setup.parameter.direct_tidal_dissipation_time_lag(
#         "Saturn", "Titan"
#     ))

parameters_to_estimate = estimation_setup.create_parameter_set(parameters_to_estimate_settings,
                                                                bodies,
                                                                propagator_settings)

original_parameter_vector = parameters_to_estimate.parameter_vector

print('Running propagation...')
estimator = numerical_simulation.Estimator(bodies, parameters_to_estimate,
                                            pseudo_observations_settings, propagator_settings)

convergence_settings = estimation.estimation_convergence_checker(maximum_iterations=5)

# Create input object for the estimation
estimation_input = estimation.EstimationInput(observations_and_times=pseudo_observations,
                                                convergence_checker=convergence_settings)
# Set methodological options
estimation_input.define_estimation_settings(save_state_history_per_iteration=True)

# Perform the estimation
print('Performing the estimation...')
print(f'Original initial states: {original_parameter_vector}')


estimation_output = estimator.perform_estimation(estimation_input)


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
