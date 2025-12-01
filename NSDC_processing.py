import os
import csv
import numpy as np

import matplotlib
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
matplotlib.use("tkagg")  #tkagg

from tudatpy.interface import spice


import sys
from pathlib import Path

# # Add parent directory to Python path
# sys.path.append(str(Path(__file__).resolve().parent.parent))

# Get the path to the directory containing this file
current_dir = Path(__file__).resolve().parent

# Append the HelperFunctions directory
sys.path.append(str(current_dir / "HelperFunctions"))

import ProcessingUtils
import PropFuncs
import FigUtils
import ObsFunc
import nsdc


#/home/atn/Documents/Year 5/Thesis/Github/NeptuneTritonMasterThesis/Kernels

kernel_folder = "/home/atn/Documents/Year 5/Thesis/Github/NeptuneTritonMasterThesis/Kernels"
kernel_paths=[
    "pck00010.tpc",
    "gm_de440.tpc",
    "nep097.bsp",     
    #"nep105.bsp",
    "naif0012.tls"
    ]

spice.load_standard_kernels()

# Load your kernels
for k in kernel_paths:
    spice.load_kernel(os.path.join(kernel_folder, k))


# spice.load_standard_kernels()
#spice.load_kernel('jup344.bsp')                     #https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/


#Select Folders
#--------------------------------------------------------------------------------------------------------------------------
folder_path = "Observations/NeptuneObservations/" 
raw_observation_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]

folder_path_rel = "Observations/RawRelativeObservations/" 
raw_observation_files_rel = [os.path.join(folder_path_rel, f) for f in os.listdir(folder_path_rel) if f.endswith('.txt')]


folder_path_csv = "Observations/RelativeObservations/" 
observation_files_csv = [os.path.join(folder_path_csv, f) for f in os.listdir(folder_path_csv) if f.endswith('.csv')]

#----------------------------------------------------------------------------------------------------------------------------
#Get already processed files
suffixes_processed = []
for i in range(len(observation_files_csv)):
    file = observation_files_csv[i]
    filename = Path(file).stem
    suffix = filename.split('_')[-1]
    suffixes_processed.append(suffix)

#suffixes.append("nm0091") #error with time entries in ndsc.py
#suffixes.append("nm0090") #error with time entires in ndsc.py
suffixes_processed.append("nm0001") #DEC values in header are more than in file
suffixes_processed.append("nm0005") #DEC values in header are more than in file

#----------------------------------------------------------------------------------------------------------------------------
#Process ABS files
suffixes2 = []
for i in range(len(raw_observation_files)):
    file = raw_observation_files[i]
    filename = Path(file).stem
    suffix = filename.split('_')[-1]
    suffixes2.append(suffix)
    if suffix in suffixes_processed:
        print("Skipping already processed file...",suffix)
    else:
        print("Currently processing: ",file)
        nsdc.process_nsdc_file(file,True, 'ECLIPJ2000',folder_path_csv)


#----------------------------------------------------------------------------------------------------------------------------
# Process relative files
suffixes_rel = []
for i in range(len(raw_observation_files_rel)):
    file = raw_observation_files_rel[i]
    filename = Path(file).stem
    suffix = filename.split('_')[-1]
    suffixes_rel.append(suffix)
    if suffix in suffixes_processed:
        print("Skipping already processed file...",suffix)
    else:
        print("Currently processing: ",file)
        nsdc.process_nsdc_file(file,True, 'ECLIPJ2000',folder_path_csv)


print(suffixes_rel)
print(suffixes2)

#----------------------------------------------------------------------------------------------------------------------------
#Create SPICE figures
#--------------------------------------------------------------------------------------------------------------------------

# for file in observation_files_csv:
#     #Reading observational data from file
    
#     ObservationList = []
#     Timelist = []
#     uncertainty_ra = []
#     uncertainty_dec = []
#     filename = Path(file).stem
#     print("Plotting for: ",filename)
    
#     with open(file, 'r') as f:
#         csv_reader = csv.reader(f)
#         next(csv_reader) 


#         for row in csv_reader:
#             time = float(row[0])
#             Timelist.append(time)
#             #obstimes.append(time)
#             ObservationList.append(np.asarray([float(row[1]), float(row[2])]))
#             uncertainty_ra_SPICE = float(row[3]) * 180/np.pi * 3600 
#             uncertainty_dec_SPICE = float(row[4]) * 180/np.pi * 3600 
#             uncertainty_ra.append(uncertainty_ra_SPICE)
#             uncertainty_dec.append(uncertainty_dec_SPICE)



#     Time_DateFormat = FigUtils.ConvertToDateTime(Timelist)

#     #--------------------------------------------------------------------------------
#     fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
#     ax.scatter(Time_DateFormat,np.asarray(uncertainty_ra))
#     ax.set_xlabel('Observation epoch [years since J2000]')
#     ax.set_ylabel('spice-observed RA [arcseconds]')
#     ax.grid(True, alpha=0.3)

#     locator   = mdates.AutoDateLocator()               # chooses sensible tick spacing
#     formatter = mdates.ConciseDateFormatter(locator)   # compact, smart formatting
#     ax.xaxis.set_major_locator(locator)
#     ax.xaxis.set_major_formatter(formatter)
    
#     output_path = f"SPICE_Residuals/RA_residuals_{filename}.pdf"
#     fig.savefig(output_path)
#     plt.close(fig)
#     #--------------------------------------------------------------------------------
#     fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
#     ax.scatter(Time_DateFormat,np.asarray(uncertainty_dec))
#     ax.set_xlabel('Observation epoch [years since J2000]')
#     ax.set_ylabel('spice-observed DEC [arcseconds]')
#     ax.grid(True, alpha=0.3)

#     locator   = mdates.AutoDateLocator()               # chooses sensible tick spacing
#     formatter = mdates.ConciseDateFormatter(locator)   # compact, smart formatting
#     ax.xaxis.set_major_locator(locator)
#     ax.xaxis.set_major_formatter(formatter)

#     output_path = f"SPICE_Residuals/DEC_residuals_{filename}.pdf"
#     fig.savefig(output_path)
#     plt.close(fig)