"""Export configuration: ObservationalDataset
Observational dataset summary — per-ID residuals vs NEP097 (SPICE) and vs
the initial propagation, histograms, and a LaTeX table.

Edit the settings below, then run:
    python MatplotlibExport.py
after setting ACTIVE_CONFIG = 'ObservationalDataset' in MatplotlibExport.py.
"""

# ── Dataset selection ─────────────────────────────────────────────────────────
# Set to the key in DashInteractivePlotFull DATA_FILES that contains a simulation
# with real-observation residuals (residual_history_arcseconds + weight_info).
DATASET_LABEL = 'PoleEst_MB'   # TODO: set to the matching pkl label

# The simulation used for the "initial propagation" residual columns.
# Any sim from a real-observations run with residual_history_arcseconds works.
_SIM_INITIAL = 'IAUPole_initial_state'        # TODO: set to a sim name in that dataset

SELECTED_SIMS = [_SIM_INITIAL]
SIM_LABELS    = ['initial propagation']

OUTPUT_DIR = 'ThesisFigures'

# ── Path to obs_analysis_data.npy (produced by Test_Observations.py) ─────────
OBS_ANALYSIS_DATA = 'Results/ObservationsAnalysis/obs_analysis_data.npy'

# ── Paths to observation files ────────────────────────────────────────────────
# Folder containing the processed CSVs (Triton_<code>_<nmXXXX>.csv).
# Each file must have 5 columns: time, RA, Dec, O-C RA, O-C Dec.
# O-C RA is stored as 2π + residual_rad; O-C Dec is the residual_rad directly.
OBS_FOLDER = 'Observations/AllModernJ2000'

# Folders containing the raw NSDC text files, used to read the observation
# type (ABS / REL) from the first word of each file header.
#   RawRelativeObservations/ — holds REL datasets (nm0002, nm0003, nm0004, …)
#   NeptuneObservations/     — holds ABS datasets (nm0007, nm0013, nm0015, …)
RAW_OBS_FOLDERS = [
    'Observations/RawRelativeObservations',
    'Observations/NeptuneObservations',
]

# ── Observation type override ─────────────────────────────────────────────────
# Only needed for IDs whose raw NSDC file is not available in RAW_OBS_FOLDERS.
# The code reads ABS/REL directly from the raw files, so this dict can stay
# empty unless you need to override an auto-detected value.
OBS_TYPES_OVERRIDE = {}

# ── LaTeX table settings ──────────────────────────────────────────────────────
# Passed to generate_obs_dataset_table() via main().
# Remove or set to None to skip table generation.
OBS_DATASET_TABLE = {
    'obs_folder':      OBS_FOLDER,
    'raw_obs_folder':  RAW_OBS_FOLDERS,   # list — searched in order
    'obs_types':       OBS_TYPES_OVERRIDE,
    'file_names_json': 'file_names.json', # restrict table to these datasets only
    # Caption and label for the LaTeX table.
    'caption': (
        r'Summary of astrometric observation datasets used in this work. '
        r'NSDC Listing gives the dataset identifier in the Natural Satellites '
        r'Data Centre archive. '
        r'MPC Code is the three-digit Minor Planet Center observatory code. '
        r'$N_{\mathrm{obs}}$ is the total number of astrometric observations. '
        r'Obs.\ Type indicates whether the original measurements are relative '
        r'(Rel.) or absolute (Abs.) astrometry. '
        r'RMS O$-$C RA and RMS O$-$C Dec are the root-mean-square observed '
        r'minus computed residuals in right ascension and declination, '
        r'respectively, evaluated against the NEP097 ephemeris [$^{\prime\prime}$].'
    ),
    'label': 'tab:obs-dataset-summary',
}

# ── Figures to export ─────────────────────────────────────────────────────────
FIGURES_TO_EXPORT = [
    # 1. All files in folder: excluded (not in file_names.json) = red, included = blue
    ('obs_analysis_all_in_folder', {
        'data_path': OBS_ANALYSIS_DATA,
        'title':     r'O$-$C Residuals vs NEP097 — All Files in Folder',
    }),
    # 2. Included files coloured by obs-file ID; legend as right-panel (2 col)
    ('obs_analysis_spice_by_id', {
        'data_path': OBS_ANALYSIS_DATA,
        'title':     r'O$-$C Residuals vs NEP097 — Included Files by ID',
    }),
    # 2b. Same but accepted observations only (no rejected markers)
    ('obs_analysis_spice_by_id_accepted', {
        'data_path': OBS_ANALYSIS_DATA,
        'title':     r'O$-$C Residuals vs NEP097 — Accepted Observations by ID',
    }),
    # 2c. Propagation residuals coloured by ID
    ('obs_analysis_prop_by_id', {
        'data_path': OBS_ANALYSIS_DATA,
        'title':     r'O$-$C Residuals vs Numerical Propagation — Included Files by ID',
    }),
    # 2d. Bias-corrected SPICE residuals coloured by ID
    ('obs_analysis_spice_biased_by_id', {
        'data_path': OBS_ANALYSIS_DATA,
        'title':     r'O$-$C Residuals vs NEP097 — Bias-Corrected by ID',
    }),
    # 2e. Overlay: unbiased (faded) vs bias-corrected (solid), same per-ID colours
    ('obs_analysis_spice_bias_overlay', {
        'data_path': OBS_ANALYSIS_DATA,
        'title':     r'O$-$C Residuals vs NEP097 — Bias Correction Overlay',
    }),
    # 3. Included files: accepted = blue, rejected = red (no per-ID colouring)
    ('obs_analysis_spice_filtered_highlight', {
        'data_path': OBS_ANALYSIS_DATA,
        'title':     r'O$-$C Residuals vs NEP097 — Accepted / Rejected',
    }),
    # 4. Combined count figure: stacked-by-file bar per year + bar per file
    ('obs_analysis_combined_count', {
        'data_path': OBS_ANALYSIS_DATA,
        'bin_years': 1,
        'title':     'Observation Count',
    }),
    # 5. Per-file RA/Dec residual figures — each saved as a separate PDF in a subfolder
    ('obs_analysis_spice_per_file', {
        'data_path':    OBS_ANALYSIS_DATA,
        'title_prefix': r'O$-$C Residuals vs NEP097',
        'subfolder':    'per_file',
    }),
]

SINGLE_SIM_TABLES = []

# No per-sim colors needed (single-sim config).
SIM_COLORS    = {_SIM_INITIAL: '#1f77b4'}
SIM_MARKERS   = {_SIM_INITIAL: 'o'}
SIM_LINESTYLE = {_SIM_INITIAL: '-'}
