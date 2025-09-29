import numpy as np
import yaml

from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
import tudatpy.estimation
from tudatpy.estimation.observable_models_setup import links, model_settings
from tudatpy.estimation import observations_setup
from tudatpy.estimation import estimation_analysis
from tudatpy.dynamics import parameters_setup
#from tudatpy.numerical_simulation import estimation, estimation_setup,Time

from tudatpy.interface import spice


def Create_Env(settings_dict):
    # Create default body settings and bodies at Neptune J2000
    start_epoch = settings_dict["start_epoch"]
    end_epoch = settings_dict["end_epoch"]
    global_frame_origin = settings_dict["global_frame_origin"] 
    global_frame_orientation = settings_dict["global_frame_orientation"]
    bodies_to_create = settings_dict["bodies"]
    interpolator_triton_cadance = settings_dict["interpolator_triton_cadance"]

    neptune_extended_gravity = settings_dict["neptune_extended_gravity"]

    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create, global_frame_origin, global_frame_orientation)


    ## Triton ########################################################################################

    body_settings.get("Triton").ephemeris_settings = environment_setup.ephemeris.interpolated_spice(
        start_epoch-100*30, end_epoch+100*30, interpolator_triton_cadance, 
        'Neptune', global_frame_orientation)


    ## Neptune 
    if neptune_extended_gravity == "Jacobson2009":
        # Define spherical harmonics from Jacobson 2009
        J2 = 3408.428530717952e-6
        J4 = -33.398917590066e-6
        C20 = -J2 / np.sqrt(5.0)   # C̄20 = -J2 / sqrt(2*2+1)
        C40 = -J4 / 3.0            # C̄40 = -J4 / sqrt(2*4+1) = -J4/3

        # Build coefficient matrices (normalized)
        l_max, m_max = 4, 0
        Cbar = np.zeros((l_max+1, l_max+1))
        Sbar = np.zeros_like(Cbar)
        Cbar[2, 0] = C20
        Cbar[4, 0] = C40

        #Get GM and radius from SPICE
        mu_N = spice.get_body_gravitational_parameter("Neptune")
        radii_km = spice.get_body_properties("Neptune","RADII",3)   # returns [Rx, Ry, Rz] in km in tudatpy >=0.8
        R_eq = radii_km[0] * 1e3   # meters (use equatorial as reference radius)


        body_settings.get("Neptune").gravity_field_settings = environment_setup.gravity_field.spherical_harmonic(
            gravitational_parameter=mu_N,
            reference_radius=R_eq,
            normalized_cosine_coefficients=Cbar,
            normalized_sine_coefficients=Sbar,
            associated_reference_frame="IAU_Neptune",
        )


    # set parameters for defining the rotation between frames
    original_frame = "J2000"
    target_frame = "IAU_Neptune"
    target_frame_spice = "IAU_Neptune"
   
    # create rotation model settings and assign to body settings of "Neptune"
    body_settings.get( "Neptune" ).rotation_model_settings = environment_setup.rotation_model.simple_from_spice(
    original_frame, target_frame, target_frame_spice, start_epoch)

    body_settings.get("Neptune").ephemeris_settings = environment_setup.ephemeris.direct_spice(
        global_frame_origin, global_frame_orientation)


    # # Create system of selected bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)
    
    return body_settings,bodies


def Create_Acceleration_Models(settings_dict,system_of_bodies):

    # Define bodies that are propagated, and their central bodies of propagation
    bodies_to_propagate = settings_dict['bodies_to_propagate']
    central_bodies = settings_dict['central_bodies']
    bodies_to_simulate = settings_dict['bodies_to_simulate']
    bodies = settings_dict['bodies']
    #system_of_bodies = settings_dict['system_of_bodies']

    neptune_extended_gravity = settings_dict['neptune_extended_gravity']
    


    acts: Dict[str, List] = {}


    for body_name in bodies_to_simulate:
        if body_name == bodies_to_propagate[0]:
            continue  # no self-acceleration
        if body_name not in bodies:
            print("Body ",body_name,"not created in env")
            continue  # not created

        # Special case: Neptune
        if body_name == "Neptune":
            #if neptune_extended_gravity is "None":
            acts.setdefault("Neptune", []).append(
            propagation_setup.acceleration.point_mass_gravity())

            if neptune_extended_gravity != "None":
                acts = SetExtendedGravityNeptune(neptune_extended_gravity,acts)
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

    accelerations_cfg = build_acceleration_config(settings_dict)
    return acceleration_models, accelerations_cfg

def SetExtendedGravityNeptune(neptune_extended_gravity,acts):
    
    if neptune_extended_gravity == "Jacobson2009":
        acts.setdefault("Neptune", []).append(
        propagation_setup.acceleration.spherical_harmonic_gravity(4, 0))
    return acts


def build_acceleration_config(settings_dict):
    bodies_to_propagate = settings_dict['bodies_to_propagate']
    central_bodies = settings_dict['central_bodies']
    bodies_to_simulate = settings_dict['bodies_to_simulate']
    bodies = settings_dict['bodies']
    target = bodies_to_propagate[0]
    neptune_extended_gravity = settings_dict['neptune_extended_gravity']
    
    cfg_acts = {}  # acting_body -> list of model descriptors
    for body_name in bodies_to_simulate:
        if body_name == target:
            continue  # no self-acceleration
        if body_name not in bodies:
            # body not present in the environment; skip
            continue

        models = []

        if body_name == "Neptune":
            models.append({"type": "point_mass_gravity"})
            if neptune_extended_gravity == "Jacobson2009":
                models.append({"type": "spherical_harmonic_gravity_Jacobson_2009", "degree": 4, "order": 0})
        else:
            # Sun, Jupiter, Saturn, etc. → default point-mass gravity
            models.append({"type": "point_mass_gravity"})

        cfg_acts[body_name] = models

    # Final YAML-friendly dict (keeps Tudat's nesting: {target: {...}})
    accel_cfg = {
        "target": target,
        "accelerations": cfg_acts
    }
    return accel_cfg



def Create_Propagator_Settings(settings_dict,acceleration_models):

    simulation_start_epoch = settings_dict['start_epoch'] 
    simulation_end_epoch   = settings_dict['end_epoch']
    bodies_to_propagate = settings_dict['bodies_to_propagate'][0]
    central_bodies = settings_dict['central_bodies'][0]
    global_frame_orientation = settings_dict['global_frame_orientation']
    fixed_step_size = settings_dict['fixed_step_size']
    #acceleration_models = settings_dict['acceleration_models']


    # 2) Get Triton’s state w.r.t. Neptune in J2000 at your epoch (seconds TDB from J2000)
    initial_state = spice.get_body_cartesian_state_at_epoch(
        target_body_name       = bodies_to_propagate,
        observer_body_name     = central_bodies,
        reference_frame_name   = global_frame_orientation,
        aberration_corrections = "none",
        ephemeris_time         = simulation_start_epoch,
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
    termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)

    # Create numerical integrator settings
    #fixed_step_size = Time(60*30) # 30 minutes
    integrator_settings = create_rkf78_integrator_settings(fixed_step_size)
    #integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(Time(fixed_step_size), coefficient_set=propagation_setup.integrator.CoefficientSets.rk_4)

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
    fixed_step_size = timestep
    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
        fixed_step_size, coefficient_set=propagation_setup.integrator.CoefficientSets.rkf_78
    )

    return integrator_settings



def make_relative_position_pseudo_observations(start_epoch,end_epoch, system_of_bodies, settings_dict: dict):
    settings = settings_dict['obs']
    cadence = settings['cadence']

    observation_start_epoch = start_epoch
    observation_end_epoch = end_epoch
    
    observation_times = np.arange(observation_start_epoch + cadence, observation_end_epoch - cadence, cadence)

    observation_times = np.array([t for t in observation_times])
    observation_times_test = observation_times[0]
    bodies_to_observe = settings["bodies"]
    relative_position_observation_settings = []

    for obs_bodies in bodies_to_observe:
        
        # link_ends = {
        #     estimation_setup.observation.observed_body: estimation_setup.observation.body_origin_link_end_id(obs_bodies[0]),
        #     estimation_setup.observation.observer: estimation_setup.observation.body_origin_link_end_id(obs_bodies[1])}
        
        link_ends = dict()
        
        link_ends[links.observed_body] = links.body_origin_link_end_id(obs_bodies[0])
        link_ends[links.observer] = links.body_origin_link_end_id(obs_bodies[1])
        
        
        #link_definition = estimation_setup.observation.LinkDefinition(link_ends)
        link_definition = links.LinkDefinition(link_ends)

        relative_position_observation_settings.append(model_settings.relative_cartesian_position(link_definition))

        observation_simulation_settings = [observations_setup.observations_simulation_settings.tabulated_simulation_settings(
                model_settings.relative_position_observable_type,
                link_definition,
                observation_times,
                reference_link_end_type = links.LinkEndType.observed_body)] #estimation_setup.observation.observed_body


    # Create observation simulators
    ephemeris_observation_simulators = observations_setup.observations_simulation_settings.create_observation_simulators(
        relative_position_observation_settings, system_of_bodies)

    # Get ephemeris states as ObservationCollection
    print('Checking spice for position pseudo observations...')
    simulated_pseudo_observations = observations_setup.observations_wrapper.simulate_observations(
        observation_simulation_settings,
        ephemeris_observation_simulators,
        system_of_bodies)

    return simulated_pseudo_observations, relative_position_observation_settings


def Create_Estimation_Output(settings_dict,system_of_bodies,propagator_settings,pseudo_observations_settings,pseudo_observations):

    #pseudo_observations_settings = settings_dict['pseudo_observations_settings']
    #pseudo_observations = settings_dict['pseudo_observations']

    parameters_to_estimate_settings = parameters_setup.initial_states(propagator_settings, system_of_bodies)
    
    if "Rotation_Pole_Position_Neptune" in settings_dict["est_parameters"]:
        parameters_to_estimate_settings.append(parameters_setup.rotation_pole_position('Neptune'))


    parameters_to_estimate = parameters_setup.create_parameter_set(parameters_to_estimate_settings,
                                                                    system_of_bodies,
                                                                    propagator_settings)

    original_parameter_vector = parameters_to_estimate.parameter_vector

    print('Running propagation...')

    estimator = estimation_analysis.Estimator(system_of_bodies, parameters_to_estimate,
                                                pseudo_observations_settings, propagator_settings)

    convergence_settings = estimation_analysis.estimation_convergence_checker(maximum_iterations=5)

    # Create input object for the estimation
    estimation_input = estimation_analysis.EstimationInput(observations_and_times=pseudo_observations,
                                                    convergence_checker=convergence_settings)
    # Set methodological options
    estimation_input.define_estimation_settings(save_state_history_per_iteration=True)

    # Perform the estimation
    print('Performing the estimation...')
    print(f'Original initial states: {original_parameter_vector}')

    estimation_output = estimator.perform_estimation(estimation_input)



    return estimation_output, original_parameter_vector