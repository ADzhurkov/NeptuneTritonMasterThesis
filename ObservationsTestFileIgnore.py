# General imports
#import math

import os
import yaml
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from datetime import datetime
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
from pathlib import Path


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
    return uncertainty_SPICE


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




##############################################################################################
# LOAD SPICE KERNELS
##############################################################################################

from pathlib import Path

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
simulation_start_epoch = DateTime(1987, 8,  29).epoch()
simulation_end_epoch   = DateTime(2006, 9, 1).epoch()
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
folder_path = "Observations/ObservationsProcessedTest"
files = os.listdir(folder_path) #['Triton_119_nm0017.csv'] 

files = ["Triton_337_nm0088.csv","Triton_755_nm0081.csv","Triton_689_nm0078.csv","Triton_689_nm0077.csv","Triton_689_nm0007.csv"]

#for f in files: 
observations,observations_settings,Observatories = ObsFunc.LoadObservations("Observations/ObservationsProcessedTest",system_of_bodies,files)

observation_times =  observations.get_observation_times()


output_folder = "Observations/Figures/Selected" #+ f.split(".")[0]

output_folder_path = make_timestamped_folder(output_folder)



#--------------------------------------------------------------------------------------------
# PLOTS
#--------------------------------------------------------------------------------------------
PlotCountHistogram(observation_times,output_folder_path,bin_type="Count")

PlotResidualsTime(observations,Observatories,system_of_bodies,output_folder_path)

residuals = Get_SPICE_residual_from_observations(observations,Observatories,system_of_bodies)

# #210127288.1842448
# observatory_ephemerides = environment_setup.create_ground_station_ephemeris(
# system_of_bodies.get_body("Earth"),
# str(327),
# system_of_bodies
# )
# orientation = settings_env["global_frame_orientation"]
# RA_relative, DEC_relative = nsdc.get_angle_rel_body(t,orientation,observatory_ephemerides,"Triton", orientation,global_frame_origin)
