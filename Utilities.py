from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.numerical_simulation import estimation, estimation_setup,Time

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


def Create_Propagation_Settings(settings_dict):

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

