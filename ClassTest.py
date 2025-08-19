# 1) Neptune radius & Jacobson J2/J4 → normalized (if you want SH)
import numpy as np
from matplotlib import pyplot as plt

from tudatpy.interface import spice
import SimulationClass as sim
# Make sure a PCK with radii is loaded by your kernel list (e.g., pck00011.tpc).
# If not sure, you can load then read once here or just hardcode the radius you trust.
# radii_km = spice.get_body_radii("Neptune")  # returns [Rx,Ry,Rz] in km (if your tudatpy supports it)
# R_eq_m = radii_km[0] * 1e3
R_eq_m = 24764e3  # fallback: Neptune equatorial radius ~24,764 km

J2 = 3408.428530717952e-6
J4 = -33.398917590066e-6
Cbar20 = -J2 / np.sqrt(5.0)
Cbar40 = -J4 / 3.0

# 2) Build the runner


runner = sim.TudatOrbitRunner(
    kernel_paths=[
        "pck00010.tpc",
        "gm_de440.tpc",     
        "nep097.bsp"
    ],
    use_neptune_point_mass=True,
    use_sun_point_mass=False,
    use_neptune_sh=True,
    neptune_sh=sim.NeptuneSH(Cbar20, Cbar40, R_eq_m)
)

# 3) Scenario (TDB seconds from J2000—use your DateTime(...).epoch())
from tudatpy.astro.time_conversion import DateTime
sc = sim.Scenario(
    start_epoch=DateTime(2025, 8, 10).epoch(),
    end_epoch=DateTime(2025, 8, 11).epoch(),
    step=10.0,
    target="Triton",
    center="Neptune"
)

# 4) Run
states_array, dep_vars_array = runner.run(sc)

# 5) Toggle experiments:
#    - Sun on:
runner.use_sun_point_mass = True
states_sun, _ = runner.run(sc)

#    - Neptune as point-mass only:
runner.use_neptune_sh = False
states_array, dep_vars_array = runner.run(sc)


##############################################################################################
# PLOT
##############################################################################################
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
