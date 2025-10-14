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


def observatory_info (Observatory): #Positive to north and east
    if len(Observatory) == 2:                   #Making sure 098 and 98 are the same
        Observatory = '0' + Observatory
    elif len(Observatory) == 1:                   #Making sure 098 and 98 are the same
        Observatory = '00' + Observatory
    with open('Observatories.txt', 'r') as file:    #https://www.projectpluto.com/obsc.htm, https://www.projectpluto.com/mpc_stat.txt
        lines = file.readlines()
        for line in lines[1:]:  # Ignore the first line
            columns = line.split()
            if columns[1] == Observatory:
                longitude = float(columns[2])
                latitude = float(columns[3])
                altitude = float(columns[4])
                return np.deg2rad(longitude),  np.deg2rad(latitude), altitude
        print('No matching Observatory found')

def LoadObservations(folder_path,system_of_bodies):
    #folder_path = 'ObservationsProcessed/CurrentProcess'
    raw_observation_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

    obstimes = []

    observation_set_list = []
    observation_settings_list = []


    #Start loop over every csv in folder
    for file in raw_observation_files:
        arc_start_times_local = []
        bias_values_local = []
        #Reading information from file name
        string_to_split = file.split("/")[-1]
        split_string = string_to_split.split("_")

        Moon = split_string[0]
        Observatory = split_string[1]
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
        observation_set_list.append( estimation.observations.single_observation_set(
            model_settings.angular_position_type, 
            link_definition,
            angles,
            times, 
            links.LinkEndType.receiver #observation.receiver 
        ))

        observation_settings_list.append(model_settings.angular_position(link_definition)) 
    
    
    observations = estimation.observations.ObservationCollection(observation_set_list) 
    
    return observations,observation_settings_list






