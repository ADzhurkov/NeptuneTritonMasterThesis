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
from tudatpy.numerical_simulation import estimation, estimation_setup
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
    simulation_start_epoch,
    simulation_end_epoch,
    time_step=5.0 * 60.0)

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
    ephemeris_time         = simulation_start_epoch,
)

dependent_variables_to_save = [
    propagation_setup.dependent_variable.total_acceleration("Triton"),
    propagation_setup.dependent_variable.keplerian_state("Triton", "Neptune"),
    propagation_setup.dependent_variable.latitude("Triton", "Neptune"),
    propagation_setup.dependent_variable.longitude("Triton", "Neptune"),

]


# Create termination settings
termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)

# Create numerical integrator settings
fixed_step_size = 60*30 # 30 minutes
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
# ORBIT ESTIMATION 
##############################################################################################

# Create Link Ends for the Moons
link_ends_Triton = dict()
link_ends_Triton[estimation_setup.observation.observed_body] = estimation_setup.observation.body_origin_link_end_id('Triton')
link_definition_Triton = estimation_setup.observation.LinkDefinition(link_ends_Triton)

link_definition_dict = {
    'Triton': link_ends_Triton
}

# Observation Model Settings
position_observation_settings = [estimation_setup.observation.cartesian_position(link_definition_Triton)]


# Define epochs at which the ephemerides shall be checked
observation_times = np.arange(simulation_start_epoch, simulation_end_epoch, 3.0 * 3600)

# Create the observation simulation settings per moon
observation_simulation_settings = list()
for moon in link_definition_dict.keys():
    observation_simulation_settings.append(estimation_setup.observation.tabulated_simulation_settings(
        estimation_setup.observation.position_observable_type,
        link_definition_dict[moon],
        observation_times,
        reference_link_end_type=estimation_setup.observation.observed_body))

#---------------------------------------------------------------------------------------------
# Simulated Ephemeries States of Triton
#---------------------------------------------------------------------------------------------

# Create observation simulators
ephemeris_observation_simulators = estimation_setup.create_observation_simulators(
    position_observation_settings, bodies)

# Get ephemeris states as ObservationCollection
print('Checking ephemerides...')
ephemeris_satellite_states = estimation.simulate_observations(
    observation_simulation_settings,
    ephemeris_observation_simulators,
    bodies)

# Define Estimatable Parameters
parameters_to_estimate_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)
parameters_to_estimate = estimation_setup.create_parameter_set(parameters_to_estimate_settings, bodies)
original_parameter_vector = parameters_to_estimate.parameter_vector

##############################################################################################
# PERFORM ESTIMATION
##############################################################################################


print('Running propagation...')
with util.redirect_std():
    estimator = numerical_simulation.Estimator(bodies, parameters_to_estimate,
                                               position_observation_settings, propagator_settings)

# Create input object for the estimation
estimation_input = estimation.EstimationInput(ephemeris_satellite_states)
# Set methodological options
estimation_input.define_estimation_settings(save_state_history_per_iteration=True)
# Perform the estimation
print('Performing the estimation...')
print(f'Original initial states: {original_parameter_vector}')

with util.redirect_std(redirect_out=False):
    estimation_output = estimator.perform_estimation(estimation_input)
initial_states_updated = parameters_to_estimate.parameter_vector
print('Done with the estimation...')
print(f'Updated initial states: {initial_states_updated}')
