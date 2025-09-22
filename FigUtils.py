import numpy as np

import matplotlib
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator, FixedFormatter, LogFormatterSciNotation
from matplotlib import pyplot as plt


import scipy.linalg as linalg
from scipy.fft import fft, fftfreq

from scipy.signal import welch, get_window
from scipy.signal import periodogram, get_window

from tudatpy.astro.time_conversion import DateTime
from tudatpy.numerical_simulation import Time

def ConvertToDateTime(float_time_J2000):
    time_DateTime = []

    for t in float_time_J2000:
        t = DateTime.from_epoch(Time(t)).to_python_datetime()
        time_DateTime.append(t)

    return time_DateTime


# ##############################################################################################
# # RESIDUALS RSW
# ##############################################################################################
def Residuals_RSW(residuals_J2000, residuals_RSW, time):

    time_dt = ConvertToDateTime(time)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)

    labels = [r'$\Delta R$', r'$\Delta S$', r'$\Delta W$']
    Y = [residuals_RSW[:,0], residuals_RSW[:,1], residuals_RSW[:,2]]

    # Pick consistent colors for R,S,W from the current color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = (colors + colors + colors)[:3]  # ensure at least 3

    for k, ax in enumerate(axes):
        # Plot all three series; make the current one opaque, others faint
        for j in range(3):
            alpha = 1.0 if j == k else 0.50   # 50% opacity for the non-focus series
            lw    = 1.6 if j == k else 1.0
            z     = 3 if j == k else 1
            ax.plot(time_dt, Y[j], color=colors[j], alpha=alpha, lw=lw, zorder=z)
            if j==k: 
                m = np.nanmax(np.abs(Y[j]))
                ax.set_ylim(-1.1*m, 1.1*m)

                #ax.set_ylim(-1.1*min(Y[j]), 1.1*max(Y[j]))

        ax.set_ylabel(f'{labels[k]} [km]')
        ax.grid(True, alpha=0.3)
        

    axes[-1].set_xlabel('Time [Date format]')

    # --- mdates: nice date ticks/labels on the shared x-axis ---
    locator   = mdates.AutoDateLocator()               # chooses sensible tick spacing
    formatter = mdates.ConciseDateFormatter(locator)   # compact, smart formatting
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(formatter)

    # (optional) minor ticks if you want extra granularity:
    # axes[-1].xaxis.set_minor_locator(mdates.AutoDateLocator(minticks=5, maxticks=12))

    fig.suptitle('Relative position in RSW') #, y=0.995)
    

    return fig



# ##############################################################################################
# # RESIDUALS PER ITERATION
# ##############################################################################################
def Residuals_RMS(residuals_j2000):
    
    iterations = np.shape(residuals_j2000)[0]
    rms = np.zeros(iterations)
    for i in range(iterations):
        A = residuals_j2000[i][:, 1:4]          # (N,3) residuals (same RMS as inertial; RSW is a rotation)

        # unweighted scalar RMS (what Tudat prints)
        rms[i] = np.sqrt(np.mean(A**2))     

    fig_rms, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(1, len(rms)+1)  # if you haven't set x yet
    ax.scatter(x, rms)

    for i, v in enumerate(rms, start=1):
        ax.annotate(f"{v:.0f}", (i, v), textcoords="offset points", xytext=(0, 6), ha="center")

    ax.set_xticks(x)                               # only integer ticks (1..N)
    ax.set_xlim(0.5, len(rms) + 0.5)               # a bit of padding
    ax.set_xlabel('Iterations [-]')
    ax.set_ylabel('RMS Residual [m]')
    ax.grid(True) 
    ax.set_title('RMS of Residuals per iteration')

    fig_rms.tight_layout()
    return fig_rms


# ##############################################################################################
# # FFT RESIDUALS FIG JONAS
# ##############################################################################################

def make_fft(residuals_data: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    signals = [residuals_data[:, 1], residuals_data[:, 2], residuals_data[:, 3]]
    sample_spacing = residuals_data[1:, 0] - residuals_data[:-1, 0]
    T = sample_spacing[0] if np.all(sample_spacing == sample_spacing[0]) else 0

    y_list = []
    for signal in signals:
        fft_signal = fft(signal)
        power_density = np.abs(fft_signal) ** 2
        y_list.append(power_density)

    N = len(residuals_data[:, 0])
    x = fftfreq(N, T)

    return y_list, len(y_list) * [x]


def create_fft_residual_figure(residuals_data,f_rot_hz):


    y_list, x_list = make_fft(residuals_data)
    assert len(y_list) == len(x_list)
    N = len(x_list[0])

    labels = ['r', 's', 'w']
    #colors, lead_color = get_band_color_scheme("ka_band")
    #colors = colors[0:2] + [colors[3]]


    fig, ax = plt.subplots(figsize=(10, 6))
    #pretty_grid(ax)

    for xf, yf, label in zip(x_list, y_list, labels):

        ax.plot(1/xf[1:N//2], yf[1:N//2], label=label, alpha=0.7)

    #ax.vlines((16*24*60*60), ymin=0, ymax=1, linestyles=':', color='grey', transform=ax.get_xaxis_transform())
    ax.vlines(f_rot_hz,ymin=0,ymax=1,linestyles=':',color='grey',transform=ax.get_xaxis_transform(), label = 'Triton Period')
    plt.legend(prop={'size': 12})

    ax.set_xlabel("freq [Hz]")
    ax.set_ylabel("signal strength [-]")

    ax.set_xscale('log')
    ax.set_yscale('log')

    year_in_s = 365*24*60*60
    month_in_s = 30*24*60*60
    day_in_s = 24*60*60
    hour_in_s = 60*60

    #ax.set_xlim([np.log10(day_in_s), np.log10(1000*year_in_s)])

    xtick_locs = ax.xaxis.get_majorticklocs()

    my_xtick_labels = []

    for x_loc in xtick_locs:

        x_val = x_loc

        if x_val > year_in_s:
            x_years = (x_val/year_in_s)
            my_xtick_labels.append(f"{x_years:.1f} years")

        elif x_val > month_in_s:
            x_month = (x_val / month_in_s)
            my_xtick_labels.append(f"{x_month:.1f} months")

        elif x_val > day_in_s:
            x_day = (x_val / day_in_s)
            my_xtick_labels.append(f"{x_day:.1f} days")

        else:
            x_hour = (x_val / hour_in_s)
            my_xtick_labels.append(f"{x_hour:.1f} hours")

    ax.xaxis.set_ticks(xtick_locs)
    ax.xaxis.set_ticklabels(my_xtick_labels)
    ax.tick_params(axis='x', labelrotation=25)

    #plt.show()

    return fig #,ax), #(1/xf[1:N//2], yf[1:N//2])



# ##############################################################################################
# # FFT RESIDUALS FIG Welch Segmenting ?
# ##############################################################################################

def psd_rsw(residuals_data, f_rot_hz, units='m',type='PSD'):
    """
    residuals_data: array with columns [t, R, S, W], t in seconds and (nearly) uniform.
    f_rot_hz: Triton spin/orbital frequency in Hz.
    """
    t = residuals_data[:, 0].astype(float)
    R = residuals_data[:, 1].astype(float)
    S = residuals_data[:, 2].astype(float)
    W = residuals_data[:, 3].astype(float)
    Y = [R, S, W]
    labels = [r'$\Delta R$', r'$\Delta S$', r'$\Delta W$']

    # Uniform sampling check
    dt = np.diff(t)
    if not np.allclose(dt, np.median(dt), rtol=1e-4, atol=0):
        raise ValueError("Time is not uniformly sampled; use a Lomb–Scargle periodogram instead.")
    fs = 1.0 / float(np.median(dt))  # Hz
    N = t.size

    # --- Choose robust segmenting params ---
    if N < 8:  # too short for Welch
        raise ValueError(f"Signal too short for Welch (N={N}).")

    nperseg = min(256, max(8, N // 2))      # strictly < N if N>=16
    if nperseg >= N:
        nperseg = max(8, N - 1)             # ensure at least 2 segments
    noverlap = nperseg // 2                  # 50% overlap
    win = get_window('hann', nperseg)

    fig, ax = plt.subplots(figsize=(10, 6))
    for y, lab in zip(Y, labels):
        y = y - np.mean(y)
        f, Pxx = welch(
            y, fs=fs, window=win, nperseg=nperseg, noverlap=noverlap,
            detrend='constant', return_onesided=True, scaling='density'
        )  # PSD → (units)^2 / Hz

        # drop DC bin for log plots
        mask = f > 0

                
        if type == 'PSD':
            ax.plot(f[1:], Pxx[1:], label=lab, lw=1.2, alpha=0.9)  # drop DC bin
            ax.set_ylabel(f'PSD [{units}²/Hz]')
    
        elif type == 'ASD':
            ASD = np.sqrt(Pxx)
            ax.plot(f[1:], ASD[1:], label=lab, lw=1.2, alpha=0.9)
            ax.set_ylabel(f'ASD [{units}/√Hz]')

        #ax.plot(f[mask], Pxx[mask], label=lab, lw=1.2, alpha=0.9)

    # Triton mean-motion line (Hz)
    ax.vlines(f_rot_hz, ymin=0, ymax=1, linestyles=':', color='grey',
              transform=ax.get_xaxis_transform(), label='Triton mean motion')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('frequency [Hz]')
    #ax.set_ylabel(f'PSD [{units}²/Hz]')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig



# ##############################################################################################
# # FFT RESIDUALS FIG RAW
# ##############################################################################################

def periodogram_rsw(
    residuals_data,                 # array with columns [t, R, S, W]
    f_rot_hz,                       # Triton rotation/orbital frequency (Hz)
    units='m',                     # label for residual units (e.g., 'km' or 'm')
    mode='PSD',                     # 'PSD', 'ASD', or 'spectrum'
    window=None,                    # None -> boxcar; or e.g. 'hann', 'blackman'
    detrend='constant',             # 'constant', 'linear', or False
    zero_pad=1                      # 1=no padding; >1 to zero-pad N*zero_pad
):
    t = residuals_data[:, 0].astype(float)
    Y = [residuals_data[:, 1].astype(float),
         residuals_data[:, 2].astype(float),
         residuals_data[:, 3].astype(float)]
    labels = [r'$\Delta R$', r'$\Delta S$', r'$\Delta W$']

    # ---- sampling check
    dt = np.diff(t)
    if not np.allclose(dt, np.median(dt), rtol=1e-4, atol=0):
        raise ValueError("Non-uniform sampling: use a Lomb–Scargle periodogram.")
    fs = 1.0 / float(np.median(dt))   # Hz

    # window
    win = 'boxcar' if window is None else get_window(window, int(len(t)))

    # choose scaling
    if mode.upper() == 'PSD':
        scaling = 'density'           # -> units^2 / Hz
        y_label = f'PSD [{units}²/Hz]'
        post = lambda P: P
    elif mode.upper() == 'ASD':
        scaling = 'density'           # take sqrt of PSD
        y_label = f'ASD [{units}/√Hz]'
        post = np.sqrt
    elif mode.lower() == 'spectrum':
        scaling = 'spectrum'          # -> units^2
        y_label = f'Power spectrum [{units}²]'
        post = lambda P: P
    else:
        raise ValueError("mode must be 'PSD', 'ASD', or 'spectrum'")

    nfft = int(len(t) * max(1, int(zero_pad)))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for y, lab, c in zip(Y, labels, colors[:3]):
        # remove mean if detrending; keeps the DC bin clean
        f, P = periodogram(
            y, fs=fs, window=win, detrend=detrend,
            return_onesided=True, scaling=scaling, nfft=nfft
        )
        # drop DC for log plots
        mask = f > 0
        ax.plot(f[mask], post(P[mask]), label=lab, lw=1.2, alpha=0.9, color=c)
    
    df = 1.0 / (t[-1] - t[0])
    ax.axvspan(f_rot_hz - 0.5*df, f_rot_hz + 0.5*df,
           color='k', alpha=0.1, label='±½ bin at f_rot')


    # Triton mean-motion line (Hz)
    ax.vlines(f_rot_hz, ymin=0, ymax=1, linestyles=':', color='grey',
              transform=ax.get_xaxis_transform(), label='Triton mean motion')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel(y_label)
    ax.grid(True, which='both', alpha=0.25)
    ax.legend()
    fig.tight_layout()

    # --- usage (after your vline) ---
    add_special_tick_at(ax, f_rot_hz)





    return fig

def add_special_tick_at(ax, x0, decimals=2, emphasize=True, color="gray", space_before_e=True):
    """
    Add a real major tick at x0 on a log-x axis and label it with scientific notation
    rounded to `decimals` (default 2). Keeps existing log ticks.

    Call this AFTER ax.set_xscale('log') and after x-limits are final.
    """
    x0 = float(x0)
    if not np.isfinite(x0) or x0 <= 0:
        raise ValueError("x0 must be a positive finite number for a log-x axis.")

    # Ensure x0 is within the visible range (expand a bit if needed)
    xmin, xmax = ax.get_xlim()
    lo, hi = sorted([xmin, xmax])
    if not (lo < x0 < hi):
        lo = min(lo, x0 * 0.9)
        hi = max(hi, x0 * 1.1)
        ax.set_xlim(lo, hi)

    # Current major ticks (now in log scale); append x0 and lock them
    base_ticks = ax.get_xticks()
    ticks = np.sort(np.unique(np.append(base_ticks, x0)))
    ax.xaxis.set_major_locator(FixedLocator(ticks))

    # Build labels: normal log formatting except at x0 -> numeric with 2 decimals
    logfmt = LogFormatterSciNotation()
    special = f"{x0:.{decimals}e}"
    if space_before_e:
        special = special.replace("e", " e")

    labels = [special if np.isclose(t, x0, rtol=0, atol=abs(x0)*1e-12) else logfmt(t) for t in ticks]
    ax.xaxis.set_major_formatter(FixedFormatter(labels))

    # Emphasize the special tick label (optional)
    if emphasize:
        ax.figure.canvas.draw()  # ensure tick labels exist
        for txt in ax.get_xticklabels():
            if txt.get_text() == special:
                txt.set_fontweight("bold")
                txt.set_color(color)
                break




# ##############################################################################################
# # 3D PLOT
# ##############################################################################################
def Plot3D(pseudo_observations,state_history_array,states_SPICE):
    states_observations = pseudo_observations.get_observations()[0]
    states_observations_size = np.size(states_observations)/3
    states_observations = states_observations.reshape(int(states_observations_size),3)/1e3

    r_km = state_history_array[:,1:4]/1e3
    x1 = r_km[:,0]
    y1 = r_km[:,1]
    z1 = r_km[:,2]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Orbit path
    ax.plot(x1, y1, z1, lw=1.5,label='Triton Orbit Fitted')
    #ax.plot(x2, y2, z2, lw=1.5, label='Triton Orbit More Planets')
    ax.plot(states_SPICE[:,0]/1e3,states_SPICE[:,1]/1e3,states_SPICE[:,2]/1e3,lw=1.5,label='SPICE Triton Oribt',linestyle='dashed')

    ax.scatter(states_observations[:,0],states_observations[:,1],states_observations[:,2],lw=1.5,label='Observations')
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
    # try:
    #     ax.set_box_aspect([1,1,1])  # Matplotlib >=3.3
    # except Exception:
    #     pass

    ax.legend(loc='upper right')
    fig.tight_layout()
    return fig
