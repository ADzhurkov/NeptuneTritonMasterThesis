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
spice.load_standard_kernels()
spice.load_kernel("gm_Horizons.pck")
spice.load_kernel("nep097.bsp")
# print("Loaded kernals :",spice.get_total_count_of_kernels_loaded()) 

# print("Neptune: ",spice.get_body_gravitational_parameter("Neptune"))
# print("Triton: ",spice.get_body_gravitational_parameter("Triton"))




radii_km = spice.get_body_properties("Neptune","RADII",3)   # returns [Rx, Ry, Rz] in km in tudatpy >=0.8
print("Neptune radii (km):", radii_km)
R_eq = radii_km[0] * 1e3   # meters (use equatorial as reference radius)


# Define string names for bodies to be created from default.
bodies_to_create = ["Sun", "Neptune", "Triton"]

# Use "Earth"/"J2000" as global frame origin and orientation.
global_frame_origin = "Neptune"
global_frame_orientation = "J2000"

# Create default body settings
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation)

# Define gravitational parameters from Jacobson 2009
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

# --- 2) GM from SPICE (SI) ---
mu_N = spice.get_body_gravitational_parameter("Neptune")


body_settings.get("Neptune").gravity_field_settings = environment_setup.gravity_field.spherical_harmonic(
    gravitational_parameter=mu_N,
    reference_radius=R_eq,
    normalized_cosine_coefficients=Cbar,
    normalized_sine_coefficients=Sbar,
    associated_reference_frame="IAU_Neptune",
)



bodies = environment_setup.create_system_of_bodies(body_settings)

# Define bodies that are propagated
bodies_to_propagate = ["Triton"]

# Define central bodies of propagation
central_bodies = ["Neptune"]

# Define accelerations acting on Triton
accelerations_settings_Triton = dict(
    Sun=[
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Neptune=[
        propagation_setup.acceleration.point_mass_gravity(),
        propagation_setup.acceleration.spherical_harmonic_gravity(4, 0)
    ],

)

# Create global accelerations settings dictionary.
acceleration_settings = {"Triton": accelerations_settings_Triton}


# Create acceleration models.
acceleration_models = propagation_setup.create_acceleration_models(
    bodies,
    acceleration_settings,
    bodies_to_propagate,
    central_bodies)


# Set simulation start and end epochs
simulation_start_epoch = DateTime(2000,7, 10).epoch()
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



# Extract and convert to km
t   = states_array[:, 0]
x   = states_array[:, 1] / 1e3
y   = states_array[:, 2] / 1e3
z   = states_array[:, 3] / 1e3

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection='3d')

# Orbit path
ax.plot(x, y, z, lw=1.5)

# Start and end points
ax.scatter([x[0]], [y[0]], [z[0]], s=40, label='Start', marker='o')
ax.scatter([x[-1]], [y[-1]], [z[-1]], s=40, label='End', marker='^')

# Neptune at origin
ax.scatter([0], [0], [0], s=80, label='Neptune', marker='*')

ax.set_xlabel('x [km]')
ax.set_ylabel('y [km]')
ax.set_zlabel('z [km]')
ax.set_title('Triton orbit (Neptune-centered)')

# Equal aspect ratio
max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
mid_x = 0.5*(x.max()+x.min()); mid_y = 0.5*(y.max()+y.min()); mid_z = 0.5*(z.max()+z.min())
ax.set_xlim(mid_x - 0.5*max_range, mid_x + 0.5*max_range)
ax.set_ylim(mid_y - 0.5*max_range, mid_y + 0.5*max_range)
ax.set_zlim(mid_z - 0.5*max_range, mid_z + 0.5*max_range)
try:
    ax.set_box_aspect([1,1,1])  # Matplotlib >=3.3
except Exception:
    pass

ax.legend(loc='upper right')
plt.tight_layout()
plt.show()


