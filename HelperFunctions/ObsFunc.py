import os
import yaml
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from datetime import datetime
from pathlib import Path
import csv

# tudatpy imports
from tudatpy import math
from tudatpy import constants
from tudatpy.interface import spice
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
import tudatpy.estimation as estimation
from tudatpy import util
from tudatpy.estimation.observable_models_setup import links, model_settings

from tudatpy import numerical_simulation

from tudatpy.astro import time_conversion, element_conversion,frame_conversion
from tudatpy.astro.time_conversion import DateTime


from tudatpy.data import save2txt

import sys
from pathlib import Path

# Get the path to the directory containing this file
current_dir = Path(__file__).resolve().parent

sys.path.append(str(current_dir / "HelperFunctions"))

import ProcessingUtils
import PropFuncs
import FigUtils
import ObsFunc
import nsdc

def observatory_info (Observatory): #Positive to north and east
    if len(Observatory) == 2:                   #Making sure 098 and 98 are the same
        Observatory = '0' + Observatory
    elif len(Observatory) == 1:                   #Making sure 098 and 98 are the same
        Observatory = '00' + Observatory
    with open('Observations/Observatories.txt', 'r') as file:    #https://www.projectpluto.com/obsc.htm, https://www.projectpluto.com/mpc_stat.txt
        lines = file.readlines()
        for line in lines[1:]:  # Ignore the first line
            columns = line.split()
            if columns[1] == Observatory:
                longitude = float(columns[2])
                latitude = float(columns[3])
                altitude = float(columns[4])
                return np.deg2rad(longitude),  np.deg2rad(latitude), altitude
        print('No matching Observatory found')

def LoadObservations(folder_path,system_of_bodies,files='None',weights = None):
    
    #folder_path = 'ObservationsProcessed/CurrentProcess'
    if files == 'None':
        raw_observation_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    else:
        #print("Taking from list: ",files)
        # Instead of iterating over os.listdir(), iterate over files
        raw_observation_files = [os.path.join(folder_path, f) for f in files if f in os.listdir(folder_path)]
        print('observation files selected: ',files)
    

    obstimes = []
    observation_set_list = []
    observation_settings_list = []
    #Observatories = []
    observation_set_ids = []
    #Start loop over every csv in folder
    observation_collection_full = estimation.observations.ObservationCollection([])
    for file in raw_observation_files:
        # if file != 'Observations/ObservationsProcessedTest/Triton_327_nm0082.csv':
        #     continue
        arc_start_times_local = []
        bias_values_local = []
        #Reading information from file name
        string_to_split = file.split("/")[-1]
        split_string = string_to_split.split("_")

        Moon = split_string[0]
        Observatory = split_string[1]
        file_id = split_string[2].split(".")[0]
        set_id = Observatory + "_" + file_id
        observation_set_ids.append(set_id)


        # Define the position of the observatory on Earth
        observatory_longitude, observatory_latitude, observatory_altitude = observatory_info(Observatory)

        # Add the ground station to the environment
        environment_setup.add_ground_station(
        system_of_bodies.get_body("Earth"),
        Observatory,
        [observatory_altitude, observatory_latitude, observatory_longitude],
        element_conversion.geodetic_position_type)



        #Reading observational data from file
        ObservationList = []
        Timelist = []
        uncertainty_ra = []
        uncertainty_dec = []
        with open(file, 'r') as f:
            csv_reader = csv.reader(f)
            next(csv_reader) 

            arc_times = []
            arc_uncertainties_ra = []
            arc_uncertainties_dec = []

            for row in csv_reader:
                time = float(row[0])
                Timelist.append(time)
                obstimes.append(time)
                ObservationList.append(np.asarray([float(row[1]), float(row[2])]))
                uncertainty_ra.append(float(row[3]))
                uncertainty_dec.append(float(row[4]))
        
        angles = ObservationList
        times = Timelist

        
        # Define link ends
        link_ends = dict()                  #To change
        link_ends[links.transmitter] = links.body_origin_link_end_id("Triton")
        link_ends[links.receiver] = links.body_reference_point_link_end_id("Earth", str(Observatory))
        link_definition = links.LinkDefinition(link_ends)


        # Create observation set 
        observation_set_list.append(estimation.observations.single_observation_set(
            model_settings.angular_position_type, 
            link_definition,
            angles,
            times, 
            links.LinkEndType.receiver #observation.receiver 
        ))

        observation_settings_list.append(model_settings.angular_position(link_definition))

        #Create Observation Collection for the current file and assign weights
        if weights is not None:
            observation_single_set_current = estimation.observations.single_observation_set(
            model_settings.angular_position_type, 
            link_definition,
            angles,
            times, 
            links.LinkEndType.receiver)
            
            observation_collection_current = estimation.observations.ObservationCollection([observation_single_set_current]) 

            w = weights.loc[set_id, ['weight_ra', 'weight_dec']].to_numpy()
            w_avg = np.mean(w)
            observation_collection_current.set_constant_weight(
                    w_avg,
                    estimation.observations.observations_processing.observation_parser(model_settings.angular_position_type)
                    )

                
            # if observation_collection_full is None:
            #     print("Observation collection full is none creating it...")
            #     observation_collection_full = estimation.observations.ObservationCollection(observation_set_list) 
            
            #     w = weights.loc[set_id, ['weight_ra', 'weight_dec']].to_numpy()
                
            #     observation_collection_full.set_constant_weight(
            #             w,
            #             estimation.observations.observations_processing.observation_parser(model_settings.angular_position_type)
            #             )

            #else:
            print("Appending observation collection with current with size: ",len(observation_collection_current.get_concatenated_observation_times()))
            observation_collection_full.append(observation_collection_current)  #
            print('size of full observations: ',len(observation_collection_full.get_concatenated_observation_times()))
                
        
        #print("set id: ",set_id)
        #print("len of times: ",len(times))
    
    
    observations = estimation.observations.ObservationCollection(observation_set_list) 
    print('size of full observations without weights: ',len(observations.get_concatenated_observation_times()))
                
    if weights is not None:
        observations = observation_collection_full

    return observations,observation_settings_list,observation_set_ids





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


            RA_spice, DEC_spice = nsdc.get_angle_rel_body(DateTime.from_epoch(observation_times[i]),'ECLIPJ2000',observatory_ephemerides, "Triton",'ECLIPJ2000',global_frame_origin="SSB")
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



