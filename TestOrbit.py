# Load standard modules
import numpy as np

import matplotlib
from matplotlib import pyplot as plt

# Load tudatpy modules
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment
from tudatpy.numerical_simulation import environment_setup, propagation_setup
from tudatpy.astro import element_conversion
from tudatpy import constants
from tudatpy.util import result2array
from tudatpy.astro.time_conversion import DateTime

# Load spice kernels
#spice.load_standard_kernels()


print("Before loading, kernels:", spice.get_total_count_of_kernels() if hasattr(spice, "get_total_count_of_kernels") else "unknown")
spice.load_standard_kernels()
print("After load_standard_kernels, kernels:", spice.get_total_count_of_kernels() if hasattr(spice, "get_total_count_of_kernels") else "unknown")

# If available in your tudatpy, list the first few kernels:
if hasattr(spice, "get_loaded_kernels"):
    print("Loaded kernels sample:", spice.get_loaded_kernels()[:5])
# Define string names for bodies to be created from default.
bodies_to_create = ["Sun", "Earth", "Moon", "Mars", "Venus"]

# Use "Earth"/"J2000" as global frame origin and orientation.
global_frame_origin = "Earth"
global_frame_orientation = "J2000"

# Create default body settings
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation)


# Create empty body settings for the satellite
body_settings.add_empty_settings("Delfi-C3")

# Create aerodynamic coefficient interface settings
reference_area_drag = (4*0.3*0.1+2*0.1*0.1)/4  # Average projection area of a 3U CubeSat
drag_coefficient = 1.2
aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
    reference_area_drag, [drag_coefficient, 0.0, 0.0]
)

# Add the aerodynamic interface to the body settings
body_settings.get("Delfi-C3").aerodynamic_coefficient_settings = aero_coefficient_settings


# Create radiation pressure settings
reference_area_radiation = (4*0.3*0.1+2*0.1*0.1)/4  # Average projection area of a 3U CubeSat
radiation_pressure_coefficient = 1.2
occulting_bodies_dict = dict()
occulting_bodies_dict["Sun"] = ["Earth"]
vehicle_target_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
    reference_area_radiation, radiation_pressure_coefficient, occulting_bodies_dict )

# Add the radiation pressure interface to the body settings
body_settings.get("Delfi-C3").radiation_pressure_target_settings = vehicle_target_settings


bodies = environment_setup.create_system_of_bodies(body_settings)
bodies.get("Delfi-C3").mass = 2.2 #kg


# Define bodies that are propagated
bodies_to_propagate = ["Delfi-C3"]

# Define central bodies of propagation
central_bodies = ["Earth"]


# Define accelerations acting on Delfi-C3 by Sun and Earth.
accelerations_settings_delfi_c3 = dict(
    Sun=[
        propagation_setup.acceleration.radiation_pressure(),
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Earth=[
        propagation_setup.acceleration.spherical_harmonic_gravity(5, 5),
        propagation_setup.acceleration.aerodynamic()
    ],
    Moon=[
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Mars=[
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Venus=[
        propagation_setup.acceleration.point_mass_gravity()
    ]
)

# Create global accelerations settings dictionary.
acceleration_settings = {"Delfi-C3": accelerations_settings_delfi_c3}

# Create acceleration models.
acceleration_models = propagation_setup.create_acceleration_models(
    bodies,
    acceleration_settings,
    bodies_to_propagate,
    central_bodies)


# Set simulation start and end epochs
simulation_start_epoch = DateTime(2008, 4, 28).epoch()
simulation_end_epoch   = DateTime(2008, 4, 29).epoch()


# Retrieve the initial state of Delfi-C3 using Two-Line-Elements (TLEs)
delfi_tle = environment.Tle(
    "1 32789U 07021G   08119.60740078 -.00000054  00000-0  00000+0 0  9999",
    "2 32789 098.0082 179.6267 0015321 307.2977 051.0656 14.81417433    68",
)
delfi_ephemeris = environment.TleEphemeris("Earth", "J2000", delfi_tle, False)
initial_state = delfi_ephemeris.cartesian_state( simulation_start_epoch )


# Define list of dependent variables to save
dependent_variables_to_save = [
    propagation_setup.dependent_variable.total_acceleration("Delfi-C3"),
    propagation_setup.dependent_variable.keplerian_state("Delfi-C3", "Earth"),
    propagation_setup.dependent_variable.latitude("Delfi-C3", "Earth"),
    propagation_setup.dependent_variable.longitude("Delfi-C3", "Earth"),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "Delfi-C3", "Sun"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "Delfi-C3", "Moon"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "Delfi-C3", "Mars"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "Delfi-C3", "Venus"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.spherical_harmonic_gravity_type, "Delfi-C3", "Earth"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.aerodynamic_type, "Delfi-C3", "Earth"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.radiation_pressure_type, "Delfi-C3", "Sun"
    )
]



# Create termination settings
termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)

# Create numerical integrator settings
fixed_step_size = 10.0
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



# Create simulation object and propagate the dynamics
dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    bodies, propagator_settings
)

# Extract the resulting state and dependent variable history and convert it to an ndarray
states = dynamics_simulator.propagation_results.state_history
states_array = result2array(states)
dep_vars = dynamics_simulator.propagation_results.dependent_variable_history
dep_vars_array = result2array(dep_vars)



# Plot total acceleration as function of time
time_hours = (dep_vars_array[:,0] - dep_vars_array[0,0])/3600
total_acceleration_norm = np.linalg.norm(dep_vars_array[:,1:4], axis=1)
plt.figure(figsize=(9, 5))
plt.title("Total acceleration norm on Delfi-C3 over the course of propagation.")
plt.plot(time_hours, total_acceleration_norm)
plt.xlabel('Time [hr]')
plt.ylabel('Total Acceleration [m/s$^2$]')
plt.xlim([min(time_hours), max(time_hours)])
plt.grid()
plt.tight_layout()


plt.figure(figsize=(9, 5))

# Point Mass Gravity Acceleration Sun
acceleration_norm_pm_sun = dep_vars_array[:,12]
plt.plot(time_hours, acceleration_norm_pm_sun, label='PM Sun')

# Point Mass Gravity Acceleration Moon
acceleration_norm_pm_moon = dep_vars_array[:,13]
plt.plot(time_hours, acceleration_norm_pm_moon, label='PM Moon')

# Point Mass Gravity Acceleration Mars
acceleration_norm_pm_mars = dep_vars_array[:,14]
plt.plot(time_hours, acceleration_norm_pm_mars, label='PM Mars')

# Point Mass Gravity Acceleration Venus
acceleration_norm_pm_venus = dep_vars_array[:,15]
plt.plot(time_hours, acceleration_norm_pm_venus, label='PM Venus')

# Spherical Harmonic Gravity Acceleration Earth
acceleration_norm_sh_earth = dep_vars_array[:,16]
plt.plot(time_hours, acceleration_norm_sh_earth, label='SH Earth')

# Aerodynamic Acceleration Earth
acceleration_norm_aero_earth = dep_vars_array[:,17]
plt.plot(time_hours, acceleration_norm_aero_earth, label='Aerodynamic Earth')

# Cannonball Radiation Pressure Acceleration Sun
acceleration_norm_rp_sun = dep_vars_array[:,18]
plt.plot(time_hours, acceleration_norm_rp_sun, label='Radiation Pressure Sun')

plt.xlim([min(time_hours), max(time_hours)])
plt.xlabel('Time [hr]')
plt.ylabel('Acceleration Norm [m/s$^2$]')

plt.legend(bbox_to_anchor=(1.005, 1))
plt.suptitle("Accelerations norms on Delfi-C3, distinguished by type and origin, over the course of propagation.")
plt.yscale('log')
plt.grid()
plt.tight_layout()


plt.show()