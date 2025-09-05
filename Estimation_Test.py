# General imports
import math
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

# tudatpy imports
from tudatpy import util
from tudatpy import constants
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.astro import time_conversion, element_conversion
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.numerical_simulation import estimation, estimation_setup,Time
from tudatpy.astro.time_conversion import DateTime


# Load spice kernels
kernel_paths=[
    "pck00010.tpc",
    "gm_de440.tpc",     
    "nep097.bsp"]

spice.load_standard_kernels()

for k in kernel_paths:
    spice.load_kernel(k)


# Define temporal scope of the simulation - equal to the time JUICE will spend in orbit around Jupiter
simulation_start_epoch = DateTime(2031, 7,  2).epoch()
simulation_end_epoch   = DateTime(2035, 4, 20).epoch()
test_DateTime = DateTime.from_epoch(simulation_start_epoch)

##############################################################################################
# CREATE ENVIRONMENT  
##############################################################################################


# Create default body settings for selected celestial bodies
moons_to_create = ['Triton']
planets_to_create = ['Jupiter', 'Saturn','Neptune']
stars_to_create = ['Sun']
bodies_to_create = np.concatenate((moons_to_create, planets_to_create, stars_to_create))

# Create default body settings for bodies_to_create, with 'Jupiter'/'J2000'
# as global frame origin and orientation.
global_frame_origin = 'Neptune'
global_frame_orientation = 'ECLIPJ2000'
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation)

### Ephemeris Settings Moons ###
for moon in moons_to_create:
    # Apply tabulated ephemeris settings
    body_settings.get(moon).ephemeris_settings = environment_setup.ephemeris.tabulated_from_existing(
    body_settings.get(moon).ephemeris_settings,
    simulation_start_epoch.to_float(),
    simulation_end_epoch.to_float(),
    time_step=60.0 * 30.0)

print('body settings: ',body_settings)
### Rotational Models ### (No Need for Triton)

# # Create system of selected bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

##############################################################################################
# CREATE PROPAGATOR SETTINGS 
##############################################################################################

# Define bodies that are propagated, and their central bodies of propagation
bodies_to_propagate = ['Triton']
central_bodies = ['Neptune']
accelerations_settings_Triton = dict()
accelerations_settings_Triton = {
    'Sun': [propagation_setup.acceleration.point_mass_gravity()],
    'Neptune': [
        propagation_setup.acceleration.point_mass_gravity(),
        #propagation_setup.acceleration.spherical_harmonic_gravity(4, 0)
    ],
    'Saturn': [propagation_setup.acceleration.point_mass_gravity()]
}

# Create global accelerations settings dictionary.
acceleration_settings = {"Triton": accelerations_settings_Triton}


# Create acceleration models.
acceleration_models = propagation_setup.create_acceleration_models(
    bodies,
    acceleration_settings,
    bodies_to_propagate,
    central_bodies)



##############################################################################################
# SET SIMULATION START AND END
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
    propagation_setup.dependent_variable.total_acceleration("Triton"),
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
    print('Test')
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

        rsw_residuals = rotate_inertial_3_to_rsw(np.array(obs_times).reshape(-1, 1),
                                                 reshaped_residuals, state_history)

        rsw_residuals_per_iteration.append(np.hstack([np.array(obs_times).reshape(-1, 1), rsw_residuals]))

    return residuals_per_iteration, rsw_residuals_per_iteration





state_history = estimation_output.simulation_results_per_iteration[-1].dynamics_results.state_history_float
state_history_array = util.result2array(state_history)

residuals_j2000, residuals_rsw = format_residual_history(estimation_output.residual_history,
                                                            pseudo_observations.get_concatenated_float_observation_times(),
                                                            state_history)

dep_vars_history = estimation_output.simulation_results_per_iteration[-1].dynamics_results.dependent_variable_history_float
dep_vars_array = util.result2array(dep_vars_history)

print('residuals: ',residuals_rsw)

# flyby_data_dict = {

#     "ca_epoch": mid.to_float(),
#     "residuals_j2000": residuals_j2000,
#     "residuals_rsw": residuals_rsw,
#     "dep_vars_array": dep_vars_array
# }


# state_parameters_info, other_parameters_info = print_total_parameter_update_in_tnw(estimation_output.parameter_history)

# my_figures_dict = dict()

# if 'rsw' in settings_dict['plot']['res']:

#     my_figures_dict['residuals_rsw'] = create_pseudo_residual_figure(flyby_data_dict=flyby_data_dict, frame='rsw')


# if 'j2000' in settings_dict['plot']['res']:

#     my_figures_dict['residuals_j2000'] = create_pseudo_residual_figure(flyby_data_dict=flyby_data_dict, frame='j2000')


# if 'fft' in settings_dict['plot']['res']:

#     my_figures_dict['residuals_fft'], fft_vals = create_fft_residual_figure(residuals_data=residuals_rsw[-1])


# if 'correlations' in settings_dict['plot']['other']:
#     my_figures_dict['correlations'] = create_correlations_figure(estimation_output)




# ##############################################################################################
# # ORBIT ESTIMATION 
# ##############################################################################################

# # Create Link Ends for the Moons
# link_ends_Triton = dict()
# link_ends_Triton[estimation_setup.observation.observed_body] = estimation_setup.observation.body_origin_link_end_id('Triton')
# link_definition_Triton = estimation_setup.observation.LinkDefinition(link_ends_Triton)

# link_definition_dict = {
#     'Triton': link_ends_Triton
# }

# # Observation Model Settings
# position_observation_settings = [estimation_setup.observation.cartesian_position(link_definition_Triton)]


# # Define epochs at which the ephemerides shall be checked
# observation_times = np.arange(simulation_start_epoch, simulation_end_epoch, 3.0 * 3600)

# # Create the observation simulation settings per moon
# observation_simulation_settings = list()
# for moon in link_definition_dict.keys():
#     observation_simulation_settings.append(estimation_setup.observation.tabulated_simulation_settings(
#         estimation_setup.observation.position_observable_type,
#         link_definition_dict[moon],
#         observation_times,
#         reference_link_end_type=estimation_setup.observation.observed_body))

# #---------------------------------------------------------------------------------------------
# # Simulated Ephemeries States of Triton
# #---------------------------------------------------------------------------------------------

# # Create observation simulators
# ephemeris_observation_simulators = estimation_setup.create_observation_simulators(
#     position_observation_settings, bodies)

# # Get ephemeris states as ObservationCollection
# print('Checking ephemerides...')
# ephemeris_satellite_states = estimation.simulate_observations(
#     observation_simulation_settings,
#     ephemeris_observation_simulators,
#     bodies)

# # Define Estimatable Parameters
# parameters_to_estimate_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)
# parameters_to_estimate = estimation_setup.create_parameter_set(parameters_to_estimate_settings, bodies)
# original_parameter_vector = parameters_to_estimate.parameter_vector

# ##############################################################################################
# # PERFORM ESTIMATION
# ##############################################################################################


# print('Running propagation...')
# with util.redirect_std():
#     estimator = numerical_simulation.Estimator(bodies, parameters_to_estimate,
#                                                position_observation_settings, propagator_settings)

# # Create input object for the estimation
# estimation_input = estimation.EstimationInput(ephemeris_satellite_states)
# # Set methodological options
# estimation_input.define_estimation_settings(save_state_history_per_iteration=True)
# # Perform the estimation
# print('Performing the estimation...')
# print(f'Original initial states: {original_parameter_vector}')

# with util.redirect_std(redirect_out=False):
#     estimation_output = estimator.perform_estimation(estimation_input)
# initial_states_updated = parameters_to_estimate.parameter_vector
# print('Done with the estimation...')
# print(f'Updated initial states: {initial_states_updated}')
