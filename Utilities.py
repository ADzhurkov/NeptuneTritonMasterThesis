import numpy as np

from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.numerical_simulation import estimation, estimation_setup,Time

from tudatpy.interface import spice

def Create_Env(settings_dict,start_epoch,end_epoch):
    # Create default body settings and bodies at Neptune J2000


    global_frame_origin = settings_dict["global_frame_origin"] 
    global_frame_orientation = settings_dict["global_frame_orientation"]
    bodies_to_create = settings_dict["bodies"]
    interpolator_triton_cadance = settings_dict["interpolator_triton_cadance"]


    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create, global_frame_origin, global_frame_orientation)


    ## Triton ########################################################################################

    body_settings.get("Triton").ephemeris_settings = environment_setup.ephemeris.interpolated_spice(
        start_epoch.to_float()-100*30, end_epoch.to_float()+100*30, interpolator_triton_cadance, 
        global_frame_origin, global_frame_orientation)


    # # Create system of selected bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)
    return body_settings,bodies


def Create_Acceleration_Models(settings_dict):

    # Define bodies that are propagated, and their central bodies of propagation
    bodies_to_propagate = settings_dict['bodies_to_propagate']
    central_bodies = settings_dict['central_bodies']
    bodies_to_simulate = settings_dict['bodies_to_simulate']
    bodies = settings_dict['bodies']
    system_of_bodies = settings_dict['system_of_bodies']

    use_neptune_extended_gravity = settings_dict['use_neptune_extended_gravity']
    


    acts: Dict[str, List] = {}
    # acts = {
    #     'Sun': [propagation_setup.acceleration.point_mass_gravity()],
    #     'Neptune': [
    #         propagation_setup.acceleration.point_mass_gravity(),
    #         #propagation_setup.acceleration.spherical_harmonic_gravity(4, 0)
    #     ],
    #     'Saturn': [propagation_setup.acceleration.point_mass_gravity()],
    #     'Jupiter': [propagation_setup.acceleration.point_mass_gravity()]
    # }

    for body_name in bodies_to_simulate:
        if body_name == bodies_to_propagate:
            continue  # no self-acceleration
        if body_name not in bodies:
            print("Body ",body_name,"not created in env")
            continue  # not created

        # Special case: Neptune
        if body_name == "Neptune":
            acts.setdefault("Neptune", []).append(
                propagation_setup.acceleration.point_mass_gravity())

            if use_neptune_extended_gravity is True:
                acts.setdefault("Neptune", []).append(
                    propagation_setup.acceleration.spherical_harmonic_gravity(4, 0))
            continue

        # Special case: Sun
        if body_name == "Sun":
            acts.setdefault("Sun", []).append(
                propagation_setup.acceleration.point_mass_gravity()
            )
            continue

        # Default for "most" other bodies: point-mass gravity
        acts.setdefault(body_name, []).append(
            propagation_setup.acceleration.point_mass_gravity()
        )

    # Create global accelerations settings dictionary.
    acceleration_settings = {bodies_to_propagate[0]: acts}


    # Create acceleration models.
    acceleration_models = propagation_setup.create_acceleration_models(
        system_of_bodies,
        acceleration_settings,
        bodies_to_propagate,
        central_bodies)

    return acceleration_models

def Create_Propagator_Settings(settings_dict):

    simulation_start_epoch = settings_dict['start_epoch'] 
    simulation_end_epoch   = settings_dict['end_epoch']
    bodies_to_propagate = settings_dict['bodies_to_propagate']
    central_bodies = settings_dict['central_bodies']
    global_frame_orientation = settings_dict['global_frame_orientation']
    fixed_step_size = settings_dict['fixed_step_size']
    acceleration_models = settings_dict['acceleration_models']


    # 2) Get Triton’s state w.r.t. Neptune in J2000 at your epoch (seconds TDB from J2000)
    initial_state = spice.get_body_cartesian_state_at_epoch(
        target_body_name       = bodies_to_propagate,
        observer_body_name     = central_bodies,
        reference_frame_name   = global_frame_orientation,
        aberration_corrections = "none",
        ephemeris_time         = simulation_start_epoch.to_float(),
    )

    #TODO Arrange this better and make expandable
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
    #fixed_step_size = Time(60*30) # 30 minutes
    integrator_settings = create_rkf78_integrator_settings(fixed_step_size)
    #propagation_setup.integrator.runge_kutta_fixed_step(fixed_step_size, coefficient_set=propagation_setup.integrator.CoefficientSets.rk_4)

    # Create propagation settings
    propagator_settings = propagation_setup.propagator.translational(
        [central_bodies],
        acceleration_models,
        [bodies_to_propagate],
        initial_state,
        simulation_start_epoch,
        integrator_settings,
        termination_condition,
        output_variables=dependent_variables_to_save
    )
    return propagator_settings

def create_rkf78_integrator_settings(timestep=60):

    # Create numerical integrator settings
    fixed_step_size = Time(timestep)
    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
        fixed_step_size, coefficient_set=propagation_setup.integrator.CoefficientSets.rkf_78
    )

    return integrator_settings



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


def Create_Estimation_Input(settings_dict,system_of_bodies,propagator_settings):

    pseudo_observations_settings = settings_dict['pseudo_observations_settings']
    pseudo_observations = settings_dict['pseudo_observations']

    parameters_to_estimate_settings = estimation_setup.parameter.initial_states(propagator_settings, system_of_bodies)

    parameters_to_estimate = estimation_setup.create_parameter_set(parameters_to_estimate_settings,
                                                                    system_of_bodies,
                                                                    propagator_settings)

    original_parameter_vector = parameters_to_estimate.parameter_vector

    print('Running propagation...')
    estimator = numerical_simulation.Estimator(system_of_bodies, parameters_to_estimate,
                                                pseudo_observations_settings, propagator_settings)

    convergence_settings = estimation.estimation_convergence_checker(maximum_iterations=5)

    # Create input object for the estimation
    estimation_input = estimation.EstimationInput(observations_and_times=pseudo_observations,
                                                    convergence_checker=convergence_settings)
    # Set methodological options
    estimation_input.define_estimation_settings(save_state_history_per_iteration=True)

    return estimation_input, original_parameter_vector