# General imports
#import math

import os
import yaml
import json
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import matplotlib.dates as mdates
import datetime as dt
from datetime import datetime, timedelta

from pathlib import Path

# tudatpy imports
from tudatpy import math
from tudatpy import constants

from tudatpy.interface import spice
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
import tudatpy.estimation
from tudatpy import util
#import tudatpy.estimation_setup

#from tudatpy.numerical_simulation import estimation

#from tudatpy.numerical_simulation import estimation_setup #,Time


from tudatpy import numerical_simulation

from tudatpy.astro import time_conversion, element_conversion,frame_conversion
from tudatpy.astro.time_conversion import DateTime


from tudatpy.data import save2txt

import sys

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Get the path to the directory containing this file
current_dir = Path(__file__).resolve().parent

# Append the HelperFunctions directory
sys.path.append(str(current_dir / "HelperFunctions"))

import ProcessingUtils
import PropFuncs
import FigUtils
import ObsFunc
import nsdc

matplotlib.use("PDF")  #tkagg


# J2000 epoch: January 1, 2000, 12:00:00 TT (approximately)
J2000_EPOCH = datetime(2000, 1, 1, 12, 0, 0)

#--------------------------------------------------------------------------------------------
# FUNCTIONS
#--------------------------------------------------------------------------------------------

#Make a folder and extract folder path
def make_timestamped_folder(base_path="Results"):
    folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_path = Path(base_path) / folder_name
    full_path.mkdir(parents=True, exist_ok=True)
    return full_path

def Get_SPICE_residual_from_observations(observations,Observatories,system_of_bodies):
    observation_times_all = observations.get_observation_times()

    #Does not work
    #observatories = observations.get_reference_points_in_link_ends()

    observations_list = observations.get_observations()
    
    diflist = []
    for j in range(len(observation_times_all)):
        
        observation_times = observation_times_all[j]
        reshaped_observations_list = observations_list[j].reshape(-1,2)


        for i in range(len(reshaped_observations_list)):
            observatory_ephemerides = environment_setup.create_ground_station_ephemeris(
            system_of_bodies.get_body("Earth"),
            Observatories[j],
            system_of_bodies
            )


            RA_spice, DEC_spice = nsdc.get_angle_rel_body(DateTime.from_epoch(observation_times[i]),'ECLIPJ2000',observatory_ephemerides, "Triton",'ECLIPJ2000',global_frame_origin=global_frame_origin)
            RA, DEC = reshaped_observations_list[i]
            diflist.append([RA-RA_spice,DEC-DEC_spice])

    diflist = np.array(diflist)


    uncertainty_ra_SPICE = diflist[:,0] * 180/np.pi * 3600 
    uncertainty_dec_SPICE = diflist[:,1] * 180/np.pi * 3600 
    uncertainty_SPICE = [uncertainty_ra_SPICE,uncertainty_dec_SPICE]
    
    mean_ra = np.average(uncertainty_ra_SPICE)
    std_ra = np.std(uncertainty_ra_SPICE)
    
    mean_dec = np.average(uncertainty_dec_SPICE)
    std_dec = np.std(uncertainty_dec_SPICE)

    mean = [mean_ra,mean_dec]
    std = [std_ra,std_dec]
    return uncertainty_SPICE,mean,std


def PlotCountHistogram(observation_times,path,bin_type="Month"):
    '''
    Bin Type defines how the bin count is calculated:
    Month - each month is 1 bin, suitable for large timespans and data count
    Count - number of observations = number of bins    
    '''
    flat_times = np.concatenate(observation_times).tolist()

    flat_times_sorted = np.sort(flat_times)
    flat_times_Datetime = FigUtils.ConvertToDateTime(flat_times_sorted)
    
    df_flat_times = pd.DataFrame({"datetime": flat_times_Datetime})
    
    if bin_type == "Month":
        first_entry = df_flat_times.values[0]
        last_entry = df_flat_times.values[-1]
        
        year_first = first_entry.astype('datetime64[Y]').astype(int) + 1970
        month_first = first_entry.astype('datetime64[M]').astype(int) % 12 + 1
        
        year_last = last_entry.astype('datetime64[Y]').astype(int) + 1970
        month_last = last_entry.astype('datetime64[M]').astype(int) % 12 + 1
        
        
        bins_num = 12*(year_last-year_first-1)+abs(month_first-11)+month_last
        bins_num = bins_num[0]
    elif bin_type == "Count":
        bins_num = len(flat_times_sorted)
    else:
        print("Unrecognized bin_type entry.")

    
    # Convert your datetimes to Matplotlib date numbers
    
    # Compute histogram manually (this gives you counts and bins)
    hist, bins = np.histogram(flat_times_sorted, bins=bins_num)
    
    # Find the bin with the maximum count
    max_index = np.argmax(hist)
    max_bin_start = bins[max_index] # DateTime(2006, 9, 17, 4, 43, 11.925976)
    max_bin_end = bins[max_index + 1]
    
    
    fig, ax = plt.subplots(figsize=(9, 4))
    df_flat_times.hist(bins=bins_num, ax=ax, edgecolor='black')
    
    
    
    ax.set_xlabel('Time')
    locator   = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    
    #ax.set_yscale("log") 
    ax.set_ylabel('Count')
    ax.set_title('Observations per Month')
    plt.tight_layout()
    
    fig.savefig(path / "Observation_Histogram_PerMonth.pdf", bbox_inches="tight")


def PlotResidualHistorgram(residuals,output_folder_path):
    
    residuals_RA = residuals[0]
    residuals_DEC = residuals[1]

    df_residuals_RA = pd.DataFrame({"residuals_RA": residuals_RA})
    df_residuals_DEC = pd.DataFrame({"residuals_DEC": residuals_DEC})

    average_max = np.average([max(residuals[0]),max(residuals[1])])
    average_min = np.average([min(residuals[0]),min(residuals[1])])
    resolution = 100 # 100 miliarcseconds
    bin_size =  int((average_max - average_min)*1000/resolution)

    if bin_size == 0:
        bin_size = 1
        print("bin size is 0, setting to 1.")
        print("number of residuals: ",len(residuals[0]))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    # Plot RA residuals
    df_residuals_RA.hist(ax=ax[0], bins=bin_size, color='steelblue', edgecolor='black')
    ax[0].set_title("RA Residuals")
    ax[0].set_xlabel("Residual [rad]")
    ax[0].set_ylabel("Count")

    # Plot DEC residuals
    df_residuals_DEC.hist(ax=ax[1], bins=bin_size, color='darkorange', edgecolor='black')
    ax[1].set_title("DEC Residuals")
    ax[1].set_xlabel("Residual [rad]")

    plt.tight_layout()
    fig.savefig(output_folder_path / "Histogram_Residuals_RA_DEC.pdf", bbox_inches="tight")


        



def PlotResidualsTime(observations,Observatories,system_of_bodies,output_folder):
    '''
    Saves a pdf figure of all the observation from a Tudat object.

    '''
    residuals_SPICE = Get_SPICE_residual_from_observations(observations,Observatories,system_of_bodies)

    observation_times_DateFormat = FigUtils.ConvertToDateTime(observations.get_concatenated_observation_times())

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.scatter(observation_times_DateFormat,residuals_SPICE[0])
    ax.set_xlabel('Observation epoch [years since J2000]')
    ax.set_ylabel('spice-observed RA [arcseconds]')
    ax.grid(True, alpha=0.3)

    locator   = mdates.AutoDateLocator()               # chooses sensible tick spacing
    formatter = mdates.ConciseDateFormatter(locator)   # compact, smart formatting
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    fig.savefig(output_folder_path / "RA_residuals_SPICE.pdf")

    #--------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.scatter(observation_times_DateFormat,residuals_SPICE[1])
    ax.set_xlabel('Observation epoch [years since J2000]')
    ax.set_ylabel('spice-observed DEC [arcseconds]')
    ax.grid(True, alpha=0.3)

    locator   = mdates.AutoDateLocator()               # chooses sensible tick spacing
    formatter = mdates.ConciseDateFormatter(locator)   # compact, smart formatting
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


    fig.savefig(output_folder_path / "DEC_residuals_SPICE.pdf")


def create_color_mapping(df):
    """
    Create a consistent color mapping for all files in the dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with 'nr' column
    
    Returns:
    --------
    dict : Dictionary mapping file number to color
    """
    file_numbers = sorted(df['nr'].unique())
    n_files = len(file_numbers)
    
    # Generate colors
    if n_files <= 20:
        colors_array = plt.cm.tab20(np.linspace(0, 1, 20))
    else:
        colors_array = plt.cm.tab20b(np.linspace(0, 1, n_files))
    
    # Create a dictionary mapping file number to color
    file_colors = {file_nr: colors_array[i] for i, file_nr in enumerate(file_numbers)}
    
    return file_colors


def filter_dataframe(df, max_mean_ra=0.1, max_mean_dec=0.1):
    """
    Filter dataframe based on mean RA and DEC thresholds.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with 'mean' column containing [RA, DEC] values
    max_mean_ra : float
        Maximum absolute mean for RA
    max_mean_dec : float
        Maximum absolute mean for DEC
    
    Returns:
    --------
    pd.DataFrame : Filtered dataframe
    """
    # Extract mean RA and DEC
    df['mean_RA'] = df['mean'].apply(lambda x: abs(x[0]))
    df['mean_DEC'] = df['mean'].apply(lambda x: abs(x[1]))
    
    # Filter based on thresholds
    filtered_df = df[(df['mean_RA'] < max_mean_ra) & (df['mean_DEC'] < max_mean_dec)].copy()
    
    print(f"Original data: {len(df)} files")
    print(f"Filtered data: {len(filtered_df)} files")
    print(f"Removed: {len(df) - len(filtered_df)} files")
    
    return filtered_df


def plot_observation_analysis(df, file_colors=None, title_suffix=""):
    """
    Create stacked histogram and bar chart of observations over time.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with columns: 'nr', 'observatory', 'times', 'mean', 'std'
    file_colors : dict, optional
        Dictionary mapping file number to color. If None, creates new mapping.
    title_suffix : str
        Additional text to add to plot titles
    
    Returns:
    --------
    dict : Color mapping used in the plot
    fig  : Figure to save to specific folder later
    """
    file_numbers = sorted(df['nr'].unique())
    n_files = len(file_numbers)
    
    if n_files == 0:
        print("No data to plot!")
        return file_colors
    
    # Use provided color mapping or create new one
    if file_colors is None:
        file_colors = create_color_mapping(df)
    
    # Explode the dataframe so each observation time gets its own row
    rows = []
    for _, row in df.iterrows():
        for time in row['times']:
            # Convert Tudat time (seconds since J2000) to datetime
            time_datetime = J2000_EPOCH + timedelta(seconds=float(time))
            rows.append({
                'nr': row['nr'],
                'observatory': row['observatory'],
                'time': time_datetime
            })
    
    df_exploded = pd.DataFrame(rows)
    
    # Create a period column for monthly bins
    df_exploded['month'] = df_exploded['time'].dt.to_period('M')
    
    # Count observations per file per month
    monthly_counts = df_exploded.groupby(['month', 'nr']).size().unstack(fill_value=0)
    
    # Calculate n_observations for the bar plot
    df['n_observations'] = df['times'].apply(len)
    
    # Create figure with two subplots (stacked vertically)
    fig = plt.figure(figsize=(18, 12))
    
    # Adjust subplot positioning to leave room for legend
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3,
                          left=0.08, right=0.82, top=0.95, bottom=0.08)
    
    # Top subplot: Stacked histogram
    ax1 = fig.add_subplot(gs[0])
    
    # Get colors in the correct order for the stacked plot (only for files in current data)
    plot_colors = [file_colors[nr] for nr in monthly_counts.columns]
    
    monthly_counts.plot(
        kind='bar',
        stacked=True,
        ax=ax1,
        color=plot_colors,
        width=1.0,
        legend=False
    )
    
    ax1.set_xlabel('Time (Month)', fontsize=12)
    ax1.set_ylabel('Number of Observations', fontsize=12)
    title = f'Observation Count Over Time by File{title_suffix}'
    ax1.set_title(title, fontsize=14)
    
    # Format x-axis to show fewer labels
    n_labels = 20
    tick_positions = np.linspace(0, len(monthly_counts) - 1, n_labels, dtype=int)
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels([str(monthly_counts.index[i]) for i in tick_positions], 
                         rotation=45, ha='right')
    
    # Create legend outside the plot area (only for files in current data)
    handles = [Patch(facecolor=file_colors[nr], label=f'File {nr}') 
               for nr in file_numbers]
    ax1.legend(handles=handles, 
               bbox_to_anchor=(1.02, 1), 
               loc='upper left',
               ncol=1,
               fontsize=8,
               frameon=True)
    
    # Bottom subplot: Bar chart of observations per file
    ax2 = fig.add_subplot(gs[1])
    
    # Use the same colors for each file
    bar_colors = [file_colors[nr] for nr in df['nr']]
    ax2.bar(df['nr'], df['n_observations'], color=bar_colors)
    ax2.set_xlabel('File Number', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Number of Observations per File', fontsize=14)
    ax2.tick_params(axis='x', rotation=45)
    

    return fig, file_colors

def plot_mean_std_analysis(df, file_colors=None, title_suffix=""):
    """
    Create bar charts showing mean and std for RA and DEC per file.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with columns: 'nr', 'mean', 'std'
    file_colors : dict, optional
        Dictionary mapping file number to color. If None, creates new mapping.
    title_suffix : str
        Additional text to add to plot titles
    save_path : Path or str, optional
        Path to save the figure. If None, doesn't save.
    
    Returns:
    --------
    dict : Color mapping used in the plot
    fig  : Figure to save later
    """
    # Use provided color mapping or create new one
    if file_colors is None:
        file_colors = create_color_mapping(df)
    
    # Extract mean and std for RA and DEC
    df['mean_RA'] = df['mean'].apply(lambda x: x[0])
    df['mean_DEC'] = df['mean'].apply(lambda x: x[1])
    df['std_RA'] = df['std'].apply(lambda x: x[0])
    df['std_DEC'] = df['std'].apply(lambda x: x[1])
    
    # Get colors for each file
    bar_colors = [file_colors[nr] for nr in df['nr']]
    
    # Create figure with 2 rows and 2 columns (RA and DEC for mean and std)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Mean RA per file
    axes[0, 0].bar(df['nr'], df['mean_RA'], color=bar_colors)
    axes[0, 0].set_xlabel('File Number')
    axes[0, 0].set_ylabel('Mean RA [arcseconds]')
    axes[0, 0].set_title(f'Mean RA Residuals per File{title_suffix}')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Std RA per file
    axes[0, 1].bar(df['nr'], df['std_RA'], color=bar_colors)
    axes[0, 1].set_xlabel('File Number')
    axes[0, 1].set_ylabel('Std RA [arcseconds]')
    axes[0, 1].set_title(f'Standard Deviation RA per File{title_suffix}')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Mean DEC per file
    axes[1, 0].bar(df['nr'], df['mean_DEC'], color=bar_colors)
    axes[1, 0].set_xlabel('File Number')
    axes[1, 0].set_ylabel('Mean DEC [arcseconds]')
    axes[1, 0].set_title(f'Mean DEC Residuals per File{title_suffix}')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Std DEC per file
    axes[1, 1].bar(df['nr'], df['std_DEC'], color=bar_colors)
    axes[1, 1].set_xlabel('File Number')
    axes[1, 1].set_ylabel('Std DEC [arcseconds]')
    axes[1, 1].set_title(f'Standard Deviation DEC per File{title_suffix}')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    return fig,file_colors


##############################################################################################
# LOAD SPICE KERNELS
##############################################################################################



# Path to the current script
current_dir = Path(__file__).resolve().parent

# Kernel folder 
kernel_folder = "Kernels" #current_dir.parent / 

#kernel_folder = "/Kernels/"
kernel_paths=[
    "pck00010.tpc",
    "gm_de440.tpc",
    "nep097.bsp",     
    "nep105.bsp",
    "naif0012.tls"
    ]

spice.load_standard_kernels()

# Load your kernels
for k in kernel_paths:
    spice.load_kernel(os.path.join(kernel_folder, k))



##############################################################################################
# CREATE ENVIRONMENT  
##############################################################################################

# Define temporal scope of the simulation - equal to the time JUICE will spend in orbit around Jupiter
simulation_start_epoch = DateTime(1960, 1,  1).epoch()
simulation_end_epoch   = DateTime(2025, 1, 1).epoch()
global_frame_origin = 'SSB'
global_frame_orientation = 'ECLIPJ2000'

#--------------------------------------------------------------------------------------------
# ENVIORONMENT SETTINGS 
#--------------------------------------------------------------------------------------------
settings_env = dict()
settings_env["start_epoch"] = simulation_start_epoch
settings_env["end_epoch"] = simulation_end_epoch
settings_env["bodies"] = ['Sun','Jupiter', 'Saturn','Neptune','Triton','Uranus','Mercury','Venus','Mars','Earth']
settings_env["global_frame_origin"] = global_frame_origin
settings_env["global_frame_orientation"] = global_frame_orientation
settings_env["interpolator_triton_cadance"] = 60*8
settings_env["neptune_extended_gravity"] = "Jacobson2009"


body_settings,system_of_bodies = PropFuncs.Create_Env(settings_env)


#--------------------------------------------------------------------------------------------
# EXTRACT OBSERVATIONS
#--------------------------------------------------------------------------------------------
folder_path = "Observations/ProcessedOutliers/" 
files = os.listdir(folder_path) #['Triton_119_nm0017.csv'] 

#files = ["Triton_337_nm0088.csv","Triton_755_nm0081.csv","Triton_689_nm0078.csv","Triton_689_nm0077.csv","Triton_689_nm0007.csv"]

data = []
for f in files: 
    print('processing file: ',f)

    observations,observations_settings,observation_set_ids = ObsFunc.LoadObservations(folder_path,system_of_bodies,[f])
    Observatories = []
    for id in observation_set_ids:
        Observatory =  id.split("_")[0]
        Observatories.append(Observatory)
    observation_times = observations.get_observation_times()
    residuals,mean,std = Get_SPICE_residual_from_observations(observations,Observatories,system_of_bodies)

    string_to_split = f.split(".")[0]
    file_nr = string_to_split.split("_")[1:]
    id = file_nr[0] + "_" + file_nr[1]

    data.append({
        "nr": id,
        "observatory": Observatories,
        "times": observation_times[0],
        "residuals": residuals,
        "mean": mean,
        "std": std
    })

df = pd.DataFrame(data)
df = df.sort_values('nr')

print(df)
df['n_observations'] = df['times'].apply(len)

# Extract mean and std for RA and DEC
df['mean_RA'] = df['mean'].apply(lambda x: x[0])
df['mean_DEC'] = df['mean'].apply(lambda x: x[1])
df['std_RA'] = df['std'].apply(lambda x: x[0])
df['std_DEC'] = df['std'].apply(lambda x: x[1])



#--------------------------------------------------------------------------------------------
# PLOTS
#--------------------------------------------------------------------------------------------
output_folder = "Observations/Figures/Analysis/Outliers" #+ f.split(".")[0]

output_folder_path = make_timestamped_folder(output_folder)





# Step 1: Create consistent color mapping from original data
file_colors = create_color_mapping(df)

# Step 2: Plot unfiltered data
analysis_fig,_ = plot_observation_analysis(df, file_colors=file_colors, 
                         title_suffix=" (All Data)")
mean_std_fig,_ = plot_mean_std_analysis(df, file_colors=file_colors, 
                      title_suffix=" (All Data)",)

# Step 3: Filter the data
filtered_df = filter_dataframe(df, max_mean_ra=0.1, max_mean_dec=0.1)

filtered_files = list(zip(filtered_df['nr']))
#print(filtered_files)

file_names = []
for file in filtered_files:
    file_string = 'Triton_' + file[0] + ".csv"
    file_names.append(file_string)




# --- Save ---
with open("observation_set_ids.json", "w") as f:
    json.dump(observation_set_ids, f, indent=2)

# --- Save ---
with open("file_names.json", "w") as f:
    json.dump(file_names, f, indent=2)


# --- Load ---
with open("file_names.json", "r") as f:
    file_names_loaded = json.load(f)



# Step 4: Plot filtered data with same colors
filtered_analysis_fig,_ = plot_observation_analysis(filtered_df, file_colors=file_colors, 
                         title_suffix=" (Filtered: |mean| < 0.1)")
filtered_mean_std_fig,_ = plot_mean_std_analysis(filtered_df, file_colors=file_colors, 
                      title_suffix=" (Filtered: |mean| < 0.1)")
                      

analysis_fig.savefig(output_folder_path / "Count_raw.pdf")

mean_std_fig.savefig(output_folder_path / "Mean_std_raw.pdf")


filtered_analysis_fig.savefig(output_folder_path / "Count_filtered.pdf")

filtered_mean_std_fig.savefig(output_folder_path / "Mean_std_filtered.pdf")



# Leave top right empty or use for combined count
#axes[0, 1].axis('off')

# # Plot 1: Mean RA per file
# axes[0, 0].bar(df['nr'], df['mean_RA'],color=bar_colors)
# axes[0, 0].set_xlabel('File Number')
# axes[0, 0].set_ylabel('Mean RA [arcseconds]')
# axes[0, 0].set_title('Mean RA Residuals per File')
# axes[0, 0].tick_params(axis='x', rotation=45)

# # Plot 2: Mean DEC per file
# axes[1, 0].bar(df['nr'], df['mean_DEC'],color=bar_colors)
# axes[1, 0].set_xlabel('File Number')
# axes[1, 0].set_ylabel('Mean DEC [arcseconds]')
# axes[1, 0].set_title('Mean DEC Residuals per File')
# axes[1, 0].tick_params(axis='x', rotation=45)

# # Plot 3: Std RA per file
# axes[0, 1].bar(df['nr'], df['std_RA'],color=bar_colors)
# axes[0, 1].set_xlabel('File Number')
# axes[0, 1].set_ylabel('Std RA [arcseconds]')
# axes[0, 1].set_title('Standard Deviation RA per File')
# axes[0, 1].tick_params(axis='x', rotation=45)

# # Plot 4: Std DEC per file
# axes[1, 1].bar(df['nr'], df['std_DEC'],color=bar_colors)
# axes[1, 1].set_xlabel('File Number')
# axes[1, 1].set_ylabel('Std DEC [arcseconds]')
# axes[1, 1].set_title('Standard Deviation DEC per File')
# axes[1, 1].tick_params(axis='x', rotation=45)

# plt.tight_layout()



#--------------------------------------------------------------------------------------------
# PLOTS
#--------------------------------------------------------------------------------------------
# PlotCountHistogram(observation_times,output_folder_path,bin_type="Count")

# PlotResidualsTime(observations,Observatories,system_of_bodies,output_folder_path)

# residuals = Get_SPICE_residual_from_observations(observations,Observatories,system_of_bodies)

# #210127288.1842448
# observatory_ephemerides = environment_setup.create_ground_station_ephemeris(
# system_of_bodies.get_body("Earth"),
# str(327),
# system_of_bodies
# )
# orientation = settings_env["global_frame_orientation"]
# RA_relative, DEC_relative = nsdc.get_angle_rel_body(t,orientation,observatory_ephemerides,"Triton", orientation,global_frame_origin)
