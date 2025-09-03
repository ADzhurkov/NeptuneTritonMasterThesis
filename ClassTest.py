# 1) Neptune radius & Jacobson J2/J4 → normalized (if you want SH)
import numpy as np
from matplotlib import pyplot as plt
from tudatpy.interface import spice
from tudatpy.astro.time_conversion import DateTime

import SimulationClass as sim

##############################################################################################
# DEFINE PROPAGATION TIME AND INTEGRATION STEP
##############################################################################################
start_epoch = DateTime(2000, 8, 1).epoch()
end_epoch = DateTime(2025, 8, 1).epoch()
step_size = 60*30 # 30 minutes


# Define spherical harmonics from Jacobson 2009
J2 = 3408.428530717952e-6
J4 = -33.398917590066e-6
C20 = -J2 / np.sqrt(5.0)   # C̄20 = -J2 / sqrt(2*2+1)
C40 = -J4 / 3.0            # C̄40 = -J4 / sqrt(2*4+1) = -J4/3


##############################################################################################
# BUILD SIMULATION
##############################################################################################

#Define dynamical parameters and kernels
runner = sim.TudatOrbitRunner(
    kernel_paths=[
        "pck00010.tpc",
        "gm_de440.tpc",     
        "nep097.bsp"
    ],
    use_neptune_point_mass=True,
    use_sun_point_mass=True,
    use_neptune_sh=False,
    neptune_sh=sim.NeptuneSH(C20, C40)
)

# Set Scenario values
sc = sim.Scenario(
    start_epoch=start_epoch,
    end_epoch=end_epoch,
    step=step_size,                              
    target="Triton",
    center="Neptune"
)

# Run Simulation without Neptune J2 and J4
states_array_1, dep_vars_array_1 = runner.run(sc)


# Run Another Simulation with Neptune J2 and J4
runner.use_neptune_sh = True  # Use SH 
#runner.bodies_to_create = ["Sun", "Neptune", "Triton","Jupiter", "Saturn", "Uranus"]  # Add other planets
states_array_3, dep_vars_array_3 = runner.run(sc)


# EXTRACT RSW MATRIX
R_rsw_to_I = runner.extract_rsw_rot_matrices(dep_vars_array_3)
E_I_to_rsw = np.transpose(R_rsw_to_I, axes=(0,2,1))   # inertial→RSW
#print('shape of rsw: ',np.shape(R_rsw_to_I))

##############################################################################################
# EXTRACT SPICE CARTESIAN COORDINATES
##############################################################################################

# Time settings (e.g. one year starting Jan 1 2025)

epochs = np.arange(start_epoch, end_epoch+60*5, step_size)


spice.load_standard_kernels()
spice.load_kernel("nep097.bsp")

# Get Triton's state relative to Neptune
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

##############################################################################################
# PLOT
##############################################################################################

#----------------------------------------------------------------
#Extract 'inertial' (Neptune Centered) states
#----------------------------------------------------------------
t = states_array_1[:, 0]
t1, r1, v1 = states_array_3[:,0], states_array_3[:,1:4], states_array_3[:,4:7]
t2, r2, v2 = states_array_1[:,0], states_array_1[:,1:4], states_array_1[:,4:7]

# Differences is nominal case '3' and test case '1'
dr_I = r2 - r1
dv_I = v2 - v1

# Difference SPICE and nominal case
dr_SPICE = states_SPICE[:,0:3] - r1
dv_SPICE = states_SPICE[:,3:6] - v1
print('shape dr_SPICE: ',np.shape(dr_SPICE))
print('shape dv_SPICE: ',np.shape(dv_SPICE))



dr_RSW = np.einsum('nij,nj->ni', E_I_to_rsw, dr_I)
dv_RSW = np.einsum('nij,nj->ni', E_I_to_rsw, dv_I)

dr_SPICE_RSW = np.einsum('nij,nj->ni', E_I_to_rsw, dr_SPICE)
dv_SPICE_RSW = np.einsum('nij,nj->ni', E_I_to_rsw, dv_SPICE)


# components: radial, along-track, cross-track
dr_RSW_km = dr_RSW/1e3
dv_RSW_km = dv_RSW/1e3

dr_SPICE_RSW_km = dr_SPICE_RSW/1e3
dv_SPICE_RSW_km = dv_SPICE_RSW/1e3

#----------------------------------------------------------------
# Position differences in RSW
#----------------------------------------------------------------
plt.figure(figsize=(9,5))
plt.plot(t, dr_RSW_km[:,0], label='ΔR')
plt.plot(t, dr_RSW_km[:,1], label='ΔS')
plt.plot(t, dr_RSW_km[:,2], label='ΔW')
plt.xlabel('Time [s]')
plt.ylabel('Position difference [km]')
plt.title('Relative position in RSW (deputy − chief)')
plt.grid(True); plt.legend(); plt.tight_layout()

# Velocity differences in RSW
plt.figure(figsize=(9,5))
plt.plot(t, dv_RSW_km[:,0], label='ΔṘ')
plt.plot(t, dv_RSW_km[:,1], label='ΔṠ')
plt.plot(t, dv_RSW_km[:,2], label='ΔẆ')
plt.xlabel('Time [s]')
plt.ylabel('Velocity difference [km/s]')
plt.title('Relative velocity in RSW (deputy − chief)')
plt.grid(True); plt.legend(); plt.tight_layout()


#----------------------------------------------------------------
# Position differences in RSW SPICE
#----------------------------------------------------------------

plt.figure(figsize=(9,5))
plt.plot(t, dr_SPICE_RSW_km[:,0], label='ΔR')
plt.plot(t, dr_SPICE_RSW_km[:,1], label='ΔS')
plt.plot(t, dr_SPICE_RSW_km[:,2], label='ΔW')
plt.xlabel('Time [s]')
plt.ylabel('Position difference [km]')
plt.title('Relative position in RSW SPICE (deputy − chief)')
plt.grid(True); plt.legend(); plt.tight_layout()

# Velocity differences in RSW
plt.figure(figsize=(9,5))
plt.plot(t, dv_SPICE_RSW_km[:,0], label='ΔṘ')
plt.plot(t, dv_SPICE_RSW_km[:,1], label='ΔṠ')
plt.plot(t, dv_SPICE_RSW_km[:,2], label='ΔẆ')
plt.xlabel('Time [s]')
plt.ylabel('Velocity difference [km/s]')
plt.title('Relative velocity in RSW SPICE (deputy − chief)')
plt.grid(True); plt.legend(); plt.tight_layout()


#----------------------------------------------------------------
# Plot 3D orbit of Triton around Neptune
#----------------------------------------------------------------
x1 = r1[:,0]/1e3
y1 = r1[:,1]/1e3
z1 = r1[:,2]/1e3

x2 = r2[:,0]/1e3
y2 = r2[:,1]/1e3
z2 = r2[:,2]/1e3


fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection='3d')

# Orbit path
ax.plot(x1, y1, z1, lw=1.5,label='Triton Orbit Default')
ax.plot(x2, y2, z2, lw=1.5, label='Triton Orbit More Planets')
ax.plot(states_SPICE[:,0]/1e3,states_SPICE[:,1]/1e3,states_SPICE[:,2]/1e3,lw=1.5,label='SPICE Triton Oribt')
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
try:
    ax.set_box_aspect([1,1,1])  # Matplotlib >=3.3
except Exception:
    pass

ax.legend(loc='upper right')
plt.tight_layout()
plt.show()
