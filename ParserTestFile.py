import numpy as np
from pathlib import Path
from typing import Dict
from datetime import datetime, timedelta
import sys
import json
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import pickle
import os 
import csv


from datetime import datetime, timedelta
import matplotlib.dates as mdates


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

# Append the HelperFunctions directory
sys.path.append(str(current_dir / "HelperFunctions"))


import PropFuncs

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



def LoadObservations(folder_path,system_of_bodies,files='None',weights = None,timeframe_weights=False):
    
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
    observation_collections_list = []
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

            if timeframe_weights == True:
                # Filter all rows belonging to this id  
                weights_id = weights[weights["id"] == set_id]

                # Loop through each timeframe for this id
                for _, row in weights_id.iterrows():
                    min_time = float(row["start_sec"]-10)  # seconds since J2000
                    max_time = float(row["end_sec"]+10)

                    # Create parser for this time interval
                    parser = estimation.observations.observations_processing.observation_parser(
                        (min_time, max_time)
                    )

                    # Load weights from this timeframe
                    w_ra = row["weight_ra"]
                    w_dec = row["weight_dec"]
                    w_avg = np.nanmean([w_ra, w_dec])  # average of RA/DEC weights

                    # Assign weights for this timeframe interval
                    observation_collection_current.set_constant_weight(
                        w_avg,
                        parser
                    )
            else:
                w = weights.loc[set_id, ['weight_ra', 'weight_dec']].to_numpy()
                w_avg = np.mean(w)

                #assign weights
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
            #print("Appending observation collection with current with size: ",len(observation_collection_current.get_concatenated_observation_times()))
            observation_collections_list.append(observation_collection_current) #
            #print('size of full observations: ',len(observation_collection_full.get_concatenated_observation_times()))
                
        
        #print("set id: ",set_id)
        #print("len of times: ",len(times))
    
    observation_collection_full = estimation.observations.merge_observation_collections(observation_collections_list)
    observations = estimation.observations.ObservationCollection(observation_set_list) 
    
    print('size of full observations without weights: ', len(observations.get_concatenated_observation_times()))
   
    if weights is not None:
        observations = observation_collection_full
        print('size of full observations with weights: ', len(observation_collection_full.get_concatenated_observation_times()))
    return observations,observation_settings_list,observation_set_ids



#############################################################################################
# LOAD SPICE KERNELS
##############################################################################################

print("Loading Kernels...")

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

print("Creating Env...")

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



weights = pd.read_csv(
    "summary.txt",
    sep="\t",
    float_precision="high",  # improves float accuracy when reading
    index_col="id"           # if your file has an index column named 'id'
)
weights = weights.reset_index()

    # weights = pd.read_csv(
    #     "Results/BetterFigs/Outliers/PostProcessing/First/FirstWeights/weights.txt",
    #     sep="\t",
    #     index_col="id"       # <-- important
    # )


with open("file_names.json", "r") as f:
    file_names_loaded = json.load(f)


#observations,observations_settings,observation_set_ids = LoadObservations("Observations/ProcessedOutliers/",system_of_bodies,file_names_loaded)
observations,observations_settings,observation_set_ids = LoadObservations(
        "Observations/ProcessedOutliers/",
        system_of_bodies,file_names_loaded,
        weights = weights,
        timeframe_weights=True)

print("end")