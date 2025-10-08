import nsdc
import os
from tudatpy.interface import spice
import matplotlib
matplotlib.use("tkagg")  #tkagg

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
