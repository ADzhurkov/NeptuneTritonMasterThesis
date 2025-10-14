import os
import csv
import numpy as np

import matplotlib
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
matplotlib.use("tkagg")  #tkagg

from tudatpy.interface import spice

import nsdc

import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import ProcessingUtils
import PropFuncs
import FigUtils
import ObsFunc


#/home/atn/Documents/Year 5/Thesis/Github/NeptuneTritonMasterThesis/Kernels

kernel_folder = "/home/atn/Documents/Year 5/Thesis/Github/NeptuneTritonMasterThesis/Kernels"
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


# spice.load_standard_kernels()
#spice.load_kernel('jup344.bsp')                     #https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/

filename = "NeptuneObservations/nm0082.txt"
nsdc.process_nsdc_file(filename,True, 'ECLIPJ2000')

#Reading observational data from file
ObservationList = []
Timelist = []
uncertainty_ra = []
uncertainty_dec = []

file = "ObservationsProcessedTest/Triton_327_nm0082.csv"
with open(file, 'r') as f:
    csv_reader = csv.reader(f)
    next(csv_reader) 


    for row in csv_reader:
        time = float(row[0])
        Timelist.append(time)
        #obstimes.append(time)
        ObservationList.append(np.asarray([float(row[1]), float(row[2])]))
        uncertainty_ra.append(float(row[3]))
        uncertainty_dec.append(float(row[4]))



Time_DateFormat = FigUtils.ConvertToDateTime(Timelist)

#--------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
ax.scatter(Time_DateFormat,np.asarray(uncertainty_ra))
ax.set_xlabel('Observation epoch [years since J2000]')
ax.set_ylabel('spice-observed RA [rad]')
ax.grid(True, alpha=0.3)

locator   = mdates.AutoDateLocator()               # chooses sensible tick spacing
formatter = mdates.ConciseDateFormatter(locator)   # compact, smart formatting
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

fig.savefig("SPICE_Residuals/RA_residuals_mine.pdf")

#--------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
ax.scatter(Time_DateFormat,np.asarray(uncertainty_dec))
ax.set_xlabel('Observation epoch [years since J2000]')
ax.set_ylabel('spice-observed DEC [rad]')
ax.grid(True, alpha=0.3)

locator   = mdates.AutoDateLocator()               # chooses sensible tick spacing
formatter = mdates.ConciseDateFormatter(locator)   # compact, smart formatting
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)


fig.savefig("SPICE_Residuals/DEC_residuals_mine.pdf")
