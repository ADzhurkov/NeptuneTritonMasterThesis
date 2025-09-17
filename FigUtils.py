import matplotlib
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

from tudatpy.astro.time_conversion import DateTime
from tudatpy.numerical_simulation import Time
def ConvertToDateTime(float_time_J2000):
    time_DateTime = []

    for t in float_time_J2000:
        t = DateTime.from_epoch(Time(t)).to_python_datetime()
        time_DateTime.append(t)

    return time_DateTime

def Residuals_RSW(residuals_J2000, residuals_RSW, time):
    
    time_format = ConvertToDateTime(time)

    fig, ax = plt.subplots(figsize=(9, 5))

    # ax.plot(time_format, residuals_J2000[:,0], label='Δx')
    # ax.plot(time_format, residuals_J2000[:,1], label='Δy')
    # ax.plot(time_format, residuals_J2000[:,2], label='Δz')

    ax.plot(time_format, residuals_RSW[:,0], label='ΔR') #, linestyle='dashed')
    ax.plot(time_format, residuals_RSW[:,1], label='ΔS') #, linestyle='dashed')
    ax.plot(time_format, residuals_RSW[:,2], label='ΔW') #, linestyle='dashed')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position difference [km]')
    ax.set_title('Relative position in RSW')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    return fig




# ##############################################################################################
# # 3D PLOT
# ##############################################################################################
# states_observations = pseudo_observations.get_observations()[0]
# states_observations_size = np.size(states_observations)/3
# states_observations = states_observations.reshape(int(states_observations_size),3)/1e3
# x1 = r_km[:,0]
# y1 = r_km[:,1]
# z1 = r_km[:,2]

# fig = plt.figure(figsize=(8, 7))
# ax = fig.add_subplot(111, projection='3d')

# # Orbit path
# ax.plot(x1, y1, z1, lw=1.5,label='Triton Orbit Fitted 2nd iteration ')
# #ax.plot(x2, y2, z2, lw=1.5, label='Triton Orbit More Planets')
# ax.plot(states_SPICE[:,0]/1e3,states_SPICE[:,1]/1e3,states_SPICE[:,2]/1e3,lw=1.5,label='SPICE Triton Oribt',linestyle='dashed')

# ax.scatter(states_observations[:,0],states_observations[:,1],states_observations[:,2],lw=1.5,label='Observations')
# # Start and end points
# ax.scatter([x1[0]], [y1[0]], [z1[0]], s=40, label='Start', marker='o')
# ax.scatter([x1[-1]], [y1[-1]], [z1[-1]], s=40, label='End', marker='^')

# # Neptune at origin
# ax.scatter([0], [0], [0], s=80, label='Neptune', marker='*')

# ax.set_xlabel('x [km]')
# ax.set_ylabel('y [km]')
# ax.set_zlabel('z [km]')
# ax.set_title('Triton orbit (Neptune-centered)')

# # Equal aspect ratio
# max_range = np.array([x1.max()-x1.min(), y1.max()-y1.min(), z1.max()-z1.min()]).max()
# mid_x = 0.5*(x1.max()+x1.min()); mid_y = 0.5*(y1.max()+y1.min()); mid_z = 0.5*(z1.max()+z1.min())
# ax.set_xlim(mid_x - 0.5*max_range, mid_x + 0.5*max_range)
# ax.set_ylim(mid_y - 0.5*max_range, mid_y + 0.5*max_range)
# ax.set_zlim(mid_z - 0.5*max_range, mid_z + 0.5*max_range)
# # try:
# #     ax.set_box_aspect([1,1,1])  # Matplotlib >=3.3
# # except Exception:
# #     pass

# ax.legend(loc='upper right')
# plt.tight_layout()
# plt.show()
