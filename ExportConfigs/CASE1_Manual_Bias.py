"""Export configuration: CASE1_Manual_Bias
Pole estimation with manual bias correction — IAUPole and FitPole variants.

FitPole (displayed as "Fit." in figures) uses a Neptune pole model fitted to
the NEP097 ephemeris.  The internal simulation keys in the pickle still use
the prefix 'SimPole' — these are the actual data keys and must not be changed
here without regenerating the pickle.  'FitPole' is the display name only.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NAMING CONVENTION  (figure labels)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Internal key → figure label

  IAUPole_initial_state              →  state
  IAUPole_pole_pos_cov               →  state+pos.
  IAUPole_pole_lib_cov               →  state+lib.
  IAUPole_pole_pos_cov_pole_lib_cov  →  state+pos.+lib.
  SimPole_initial_state              →  state (Fit.)
  SimPole_pole_pos_cov               →  state+pos. (Fit.)
  SimPole_pole_lib_cov               →  state+lib. (Fit.)
  SimPole_pole_pos_cov_pole_lib_cov  →  state+pos.+lib. (Fit.)

Marker shape encodes the rotation model (consistent across all CASE1 figures):
  'o'  →  IAUPole (IAU 2015 rotation model)        (solid line)
  'D'  →  SimPole / FitPole (NEP097-fitted model)  (dashed line)

Colors — Paul Tol "muted" palette (distinct per sim) for timeseries;
  SIM_GROUP_COLORS (blue/vermillion) for aggregate use_group_colors=True figures.

Abbreviations used in figure labels:
  pos.   pole position (α₀, δ₀)
  lib.   pole librations (α₁, δ₁)
  Fit.   FitPole — NEP097-fitted pole model (internal key prefix: SimPole)
"""

DATASET_LABEL = 'PoleEst_MB'  # must match key in DashInteractivePlotFull DATA_FILES

# ─── Complete ordered list of all simulations ─────────────────────────────────
# Keys must match the sim names in the pickle file (SimPole_* prefix).
SELECTED_SIMS = [
    'IAUPole_initial_state',
    'IAUPole_pole_pos_cov',
    'IAUPole_pole_lib_cov',
    'IAUPole_pole_pos_cov_pole_lib_cov',
    'SimPole_initial_state',
    'SimPole_pole_pos_cov',
    'SimPole_pole_lib_cov',
    'SimPole_pole_pos_cov_pole_lib_cov',
]

# Short labels for figures — rotation model encoded by marker shape,
# so labels only describe what is estimated.
SIM_LABELS = [
    'state',                    # IAUPole_initial_state
    'state+pos.',               # IAUPole_pole_pos_cov
    'state+lib.',               # IAUPole_pole_lib_cov
    'state+pos.+lib.',          # IAUPole_pole_pos_cov_pole_lib_cov
    'state (Fit.)',             # SimPole_initial_state
    'state+pos. (Fit.)',        # SimPole_pole_pos_cov
    'state+lib. (Fit.)',        # SimPole_pole_lib_cov
    'state+pos.+lib. (Fit.)',   # SimPole_pole_pos_cov_pole_lib_cov
]

OUTPUT_DIR = 'ThesisFigures'
SUPPRESS_TITLES = True

# ─── Named subsets for reuse across figure entries ────────────────────────────
_PROMINENT = [
    'IAUPole_pole_lib_cov',
    'IAUPole_pole_pos_cov_pole_lib_cov',
    'SimPole_pole_lib_cov',
    'SimPole_pole_pos_cov_pole_lib_cov',
]

# Key result sims: state (IAU), state (Fit.) side-by-side, then state+lib. pair
_KEY_SIMS = [
    'IAUPole_initial_state',
    'SimPole_initial_state',
    'IAUPole_pole_lib_cov',
    'SimPole_pole_lib_cov',
]

FIGURES_TO_EXPORT = [
    # ════════════════════════════════════════════════════════════════════
    # 00  Legend
    # ════════════════════════════════════════════════════════════════════
    ('legend', {
        'subfolder': '00_legend',
        'title': 'CASE1 Manual Bias — Simulation Legend',
        'ncols': 2,
        'shape_labels': {
            'o': 'IAU 2015 rotation model  (solid line)',
            'D': 'NEP097-fitted pole model  (dashed line)',
        },
        'notes': [
            'Abbreviations (period = abbreviated word):',
            '  state       Triton initial state (6 elements, not abbreviated)',
            '  pos.        pole position (alpha_0, delta_0)',
            '  lib.        pole librations (alpha_1, delta_1)',
            '  Fit.        NEP097-fitted pole model',
        ],
    }),

    # ════════════════════════════════════════════════════════════════════
    # 01  Goodness-of-fit and RMS comparison
    #     Colors: one per group (IAUPole / FitPole), not per sim.
    # ════════════════════════════════════════════════════════════════════
    ('gof_combined', {
        'subfolder': '01_gof_rms',
        'show_initial': True,
        'thick_lines': True,        # thicker lines so initial values are visible
        'use_group_colors': True,   # show group legend (IAUPole / FitPole)
        'title': 'Goodness of Fit — WRMS, RMS, Cost Function',
    }),
    ('rms_compare', {
        'subfolder': '01_gof_rms',
        'use_group_colors': True,
        'title': 'Total RMS vs NEP097',
    }),
    ('rms_formal', {
        'subfolder': '01_gof_rms',
        'use_group_colors': True,
        'title': 'RMS of Formal Errors (RSW)',
    }),
    ('rms_ratio', {
        'subfolder': '01_gof_rms',
        'use_group_colors': True,
        'title': r'RMS vs NEP097 / RMS Formal Errors',
    }),

    # ════════════════════════════════════════════════════════════════════
    # 02  RSW statistics — split: NEP097 difference vs. formal errors
    # ════════════════════════════════════════════════════════════════════
    ('rsw_stats', {
        'subfolder': '02_rsw_stats',
        'fig_label': 'all_diff',
        'show_formal': False,
        'fontsize_scale': 1.3,
        'title': 'RSW Statistics vs NEP097 — All Simulations',
    }),
    ('rsw_stats', {
        'subfolder': '02_rsw_stats',
        'fig_label': 'all_formal',
        'show_diff': False,
        'show_formal': True,
        'fontsize_scale': 1.3,
        'title': 'RSW Formal Errors — All Simulations',
    }),
    ('rsw_stats', {
        'subfolder': '02_rsw_stats',
        'fig_label': 'prominent_diff',
        'sim_subset': _PROMINENT,
        'show_formal': False,
        'fontsize_scale': 1.3,
        'title': 'RSW Statistics vs NEP097 — Prominent Solutions',
    }),
    ('rsw_stats', {
        'subfolder': '02_rsw_stats',
        'fig_label': 'prominent_formal',
        'sim_subset': _PROMINENT,
        'show_diff': False,
        'show_formal': True,
        'fontsize_scale': 1.3,
        'title': 'RSW Formal Errors — Prominent Solutions',
    }),
    # Combined 3×3 grid: rows = RMS diff / Formal σ RMS / Ratio, cols = R/S/W
    ('rsw_rms_ratio_grid', {
        'subfolder':      '02_rsw_stats',
        'fig_label':      'rms_formal_ratio_grid',
        'use_group_colors': True,
        'fontsize_scale': 1.2,
        'title':          'RSW RMS Diff, Formal Error RMS and Ratio — All Simulations',
    }),

    # ════════════════════════════════════════════════════════════════════
    # 03  RSW difference timeseries — one figure per estimation vs state
    #     Subfolder: RSWDiffTimeseries/IAUPole
    # ════════════════════════════════════════════════════════════════════
    ('rsw_with_zoom', {
        'subfolder': 'RSWDiffTimeseries/IAUPole',
        'fig_label': 'pole_pos',
        'sim_subset': ['IAUPole_initial_state', 'IAUPole_pole_pos_cov'],
        'title': 'RSW Difference vs NEP097 — state vs state+pos. (IAU)',
    }),
    ('rsw_with_zoom', {
        'subfolder': 'RSWDiffTimeseries/IAUPole',
        'fig_label': 'pole_lib',
        'sim_subset': ['IAUPole_initial_state', 'IAUPole_pole_lib_cov'],
        'title': 'RSW Difference vs NEP097 — state vs state+lib. (IAU)',
    }),
    ('rsw_with_zoom', {
        'subfolder': 'RSWDiffTimeseries/IAUPole',
        'fig_label': 'pole_pos_lib',
        'sim_subset': ['IAUPole_initial_state', 'IAUPole_pole_pos_cov_pole_lib_cov'],
        'title': r'RSW Difference vs NEP097 — state vs state+pos.+lib. (IAU)',
    }),

    # ════════════════════════════════════════════════════════════════════
    # 04  RSW difference timeseries — FitPole (SimPole data keys)
    #     Subfolder: RSWDiffTimeseries/FitPole
    # ════════════════════════════════════════════════════════════════════
    ('rsw_with_zoom', {
        'subfolder': 'RSWDiffTimeseries/FitPole',
        'fig_label': 'pole_pos',
        'sim_subset': ['SimPole_initial_state', 'SimPole_pole_pos_cov'],
        'title': 'RSW Difference vs NEP097 — state vs state+pos. (Fit.)',
    }),
    ('rsw_with_zoom', {
        'subfolder': 'RSWDiffTimeseries/FitPole',
        'fig_label': 'pole_lib',
        'sim_subset': ['SimPole_initial_state', 'SimPole_pole_lib_cov'],
        'title': 'RSW Difference vs NEP097 — state vs state+lib. (Fit.)',
    }),
    ('rsw_with_zoom', {
        'subfolder': 'RSWDiffTimeseries/FitPole',
        'fig_label': 'pole_pos_lib',
        'sim_subset': ['SimPole_initial_state', 'SimPole_pole_pos_cov_pole_lib_cov'],
        'title': r'RSW Difference vs NEP097 — state vs state+pos.+lib. (Fit.)',
    }),

    # ════════════════════════════════════════════════════════════════════
    # 05  Formal errors timeseries — IAUPole
    #     Subfolder: FormalTimeseries/IAUPole
    # ════════════════════════════════════════════════════════════════════
    ('formal_with_zoom', {
        'subfolder': 'FormalTimeseries/IAUPole',
        'fig_label': 'pole_pos',
        'sim_subset': ['IAUPole_initial_state', 'IAUPole_pole_pos_cov'],
        'title': 'Formal Errors RSW — state vs state+pos. (IAU)',
    }),
    ('formal_with_zoom', {
        'subfolder': 'FormalTimeseries/IAUPole',
        'fig_label': 'pole_lib',
        'sim_subset': ['IAUPole_initial_state', 'IAUPole_pole_lib_cov'],
        'title': 'Formal Errors RSW — state vs state+lib. (IAU)',
    }),
    ('formal_with_zoom', {
        'subfolder': 'FormalTimeseries/IAUPole',
        'fig_label': 'pole_pos_lib',
        'sim_subset': ['IAUPole_initial_state', 'IAUPole_pole_pos_cov_pole_lib_cov'],
        'title': r'Formal Errors RSW — state vs state+pos.+lib. (IAU)',
    }),

    # ════════════════════════════════════════════════════════════════════
    # 06  Formal errors timeseries — FitPole (SimPole data keys)
    #     Subfolder: FormalTimeseries/FitPole
    # ════════════════════════════════════════════════════════════════════
    ('formal_with_zoom', {
        'subfolder': 'FormalTimeseries/FitPole',
        'fig_label': 'pole_pos',
        'sim_subset': ['SimPole_initial_state', 'SimPole_pole_pos_cov'],
        'title': 'Formal Errors RSW — state vs state+pos. (Fit.)',
    }),
    ('formal_with_zoom', {
        'subfolder': 'FormalTimeseries/FitPole',
        'fig_label': 'pole_lib',
        'sim_subset': ['SimPole_initial_state', 'SimPole_pole_lib_cov'],
        'title': 'Formal Errors RSW — state vs state+lib. (Fit.)',
    }),
    ('formal_with_zoom', {
        'subfolder': 'FormalTimeseries/FitPole',
        'fig_label': 'pole_pos_lib',
        'sim_subset': ['SimPole_initial_state', 'SimPole_pole_pos_cov_pole_lib_cov'],
        'title': r'Formal Errors RSW — state vs state+pos.+lib. (Fit.)',
    }),

    # ════════════════════════════════════════════════════════════════════
    # 07  Correlation heatmaps: state, state+lib., state+pos.+lib.
    # ════════════════════════════════════════════════════════════════════
    ('corr_heatmap', {
        'subfolder': '07_correlations',
        'fig_label': 'IAUPole_initial_state',
        'sim_name':  'IAUPole_initial_state',
        'title':     '|Correlations| — State Only (IAU 2015)',
        'cmap':      'YlOrRd',
    }),
    ('corr_heatmap', {
        'subfolder': '07_correlations',
        'fig_label': 'IAUPole_pole_lib_cov',
        'sim_name':  'IAUPole_pole_lib_cov',
        'title':     '|Correlations| — State + lib. (IAU 2015)',
        'cmap':      'YlOrRd',
    }),
    ('corr_heatmap', {
        'subfolder': '07_correlations',
        'fig_label': 'IAUPole_pole_pos_cov_pole_lib_cov',
        'sim_name':  'IAUPole_pole_pos_cov_pole_lib_cov',
        'title':     '|Correlations| — State + pos. + lib. (IAU 2015)',
        'cmap':      'YlOrRd',
    }),
    # FitPole (SimPole) correlation heatmaps
    ('corr_heatmap', {
        'subfolder': '07_correlations',
        'fig_label': 'SimPole_initial_state',
        'sim_name':  'SimPole_initial_state',
        'title':     '|Correlations| — State Only (Fit.)',
        'cmap':      'YlOrRd',
    }),
    ('corr_heatmap', {
        'subfolder': '07_correlations',
        'fig_label': 'SimPole_pole_lib_cov',
        'sim_name':  'SimPole_pole_lib_cov',
        'title':     '|Correlations| — State + lib. (Fit.)',
        'cmap':      'YlOrRd',
    }),
    ('corr_heatmap', {
        'subfolder': '07_correlations',
        'fig_label': 'SimPole_pole_pos_cov_pole_lib_cov',
        'sim_name':  'SimPole_pole_pos_cov_pole_lib_cov',
        'title':     '|Correlations| — State + pos. + lib. (Fit.)',
        'cmap':      'YlOrRd',
    }),
    # Condition numbers of all correlation matrices — written to a text file.
    ('condition_numbers_txt', {
        'subfolder': '07_correlations',
    }),

    # ════════════════════════════════════════════════════════════════════
    # 08  Parameter updates
    #     variant='fitpole' routes to non-IAUPole sims (SimPole_* keys)
    # ════════════════════════════════════════════════════════════════════
    ('param_state', {'subfolder': '08_params', 'fig_label': 'iau',      'variant': 'iau'}),
    ('param_state', {'subfolder': '08_params', 'fig_label': 'fitpole',  'variant': 'fitpole'}),
    ('param_state', {'subfolder': '08_params', 'fig_label': 'combined', 'variant': 'combined'}),
    ('param_rsw',   {'subfolder': '08_params', 'fig_label': 'iau',      'variant': 'iau'}),
    ('param_rsw',   {'subfolder': '08_params', 'fig_label': 'fitpole',  'variant': 'fitpole'}),
    ('param_rsw',   {'subfolder': '08_params', 'fig_label': 'combined', 'variant': 'combined'}),
    ('param_pole_pos', {'subfolder': '08_params', 'fig_label': 'iau',      'variant': 'iau'}),
    ('param_pole_pos', {'subfolder': '08_params', 'fig_label': 'fitpole',  'variant': 'fitpole'}),
    ('param_pole_pos', {'subfolder': '08_params', 'fig_label': 'combined', 'variant': 'combined'}),
    ('param_pole_lib', {'subfolder': '08_params', 'fig_label': 'iau',      'variant': 'iau'}),
    ('param_pole_lib', {'subfolder': '08_params', 'fig_label': 'fitpole',  'variant': 'fitpole'}),
    ('param_pole_lib', {'subfolder': '08_params', 'fig_label': 'combined', 'variant': 'combined'}),

    # ════════════════════════════════════════════════════════════════════
    # 09  Initial RSW difference (pre-estimation)
    # ════════════════════════════════════════════════════════════════════
    ('rsw_initial', {
        'subfolder': '09_initial_rsw',
        'sim_subset': ['IAUPole_initial_state', 'SimPole_initial_state'],
        'title': 'Initial RSW Difference vs NEP097 (pre-estimation)',
    }),
    ('rsw_initial', {
        'subfolder': '09_initial_rsw',
        'fig_label': 'fitpole_m',
        'sim_subset': ['SimPole_initial_state'],
        'unit': 'm',
        'title': 'FitPole Initial RSW Difference vs NEP097 (pre-estimation)',
    }),

    # ════════════════════════════════════════════════════════════════════
    # FinalEstimation  — key result figures for state+lib. (Fit.)
    # ════════════════════════════════════════════════════════════════════
    ('rsw_with_zoom', {
        'subfolder':  'FinalEstimation',
        'fig_label':  'rsw_diff',
        'sim_subset': ['SimPole_pole_lib_cov'],
        'title':      'RSW Difference vs NEP097 \u2014 state+lib. (Fit.)',
    }),
    ('formal_with_zoom', {
        'subfolder':  'FinalEstimation',
        'fig_label':  'formal',
        'sim_subset': ['SimPole_pole_lib_cov'],
        'title':      'Formal Errors RSW \u2014 state+lib. (Fit.)',
    }),
    # RSW diff with ±1σ formal error shading (cloud)
    ('rsw_with_formal_cloud', {
        'subfolder':  'FinalEstimation',
        'fig_label':  'rsw_formal_cloud',
        'sim_subset': ['SimPole_pole_lib_cov'],
        'title':      'RSW Difference vs NEP097 with \u00b11\u03c3 Formal Error \u2014 state+lib. (Fit.)',
    }),
    # RSW diff and formal error as separate colored lines
    ('rsw_and_formal_lines', {
        'subfolder':  'FinalEstimation',
        'fig_label':  'rsw_formal_lines',
        'sim_subset': ['SimPole_pole_lib_cov'],
        'title':      'RSW Difference vs NEP097 and Formal Errors \u2014 state+lib. (Fit.)',
    }),
    ('residual_timeseries', {
        'subfolder':  'FinalEstimation',
        'fig_label':  'residuals',
        'sim_subset': ['SimPole_pole_lib_cov'],
        'title':      'Observation Residuals RA/DEC \u2014 state+lib. (Fit.)',
    }),

    # ════════════════════════════════════════════════════════════════════
    # RSW diff comparisons: IAU vs Fit. side-by-side
    # ════════════════════════════════════════════════════════════════════
    ('rsw_with_zoom', {
        'subfolder': 'RSWDiffTimeseries/Comparison',
        'fig_label': 'state_iau_vs_fit',
        'sim_subset': ['IAUPole_initial_state', 'SimPole_initial_state'],
        'title': 'RSW Difference vs NEP097 \u2014 state (IAU) vs state (Fit.)',
    }),
    ('rsw_with_zoom', {
        'subfolder': 'RSWDiffTimeseries/Comparison',
        'fig_label': 'lib_iau_vs_fit',
        'sim_subset': ['IAUPole_pole_lib_cov', 'SimPole_pole_lib_cov'],
        'title': 'RSW Difference vs NEP097 \u2014 state+lib. (IAU) vs state+lib. (Fit.)',
    }),

    # Initial (nominal) vs final (estimated) for state sims
    ('rsw_initial_vs_final', {
        'subfolder': 'RSWDiffTimeseries/Comparison',
        'fig_label': 'state_iau_initial_vs_final',
        'sim_subset': ['IAUPole_initial_state'],
        'title': 'RSW Difference vs NEP097 \u2014 IAU nominal vs state',
    }),
    ('rsw_initial_vs_final', {
        'subfolder': 'RSWDiffTimeseries/Comparison',
        'fig_label': 'state_fit_initial_vs_final',
        'sim_subset': ['SimPole_initial_state'],
        'title': 'RSW Difference vs NEP097 \u2014 FitPole nominal vs state (Fit.)',
    }),
    # Initial (nominal) vs final (estimated) for state+lib. sims
    ('rsw_initial_vs_final', {
        'subfolder': 'RSWDiffTimeseries/Comparison',
        'fig_label': 'lib_iau_initial_vs_final',
        'sim_subset': ['IAUPole_pole_lib_cov'],
        'title': 'RSW Difference vs NEP097 \u2014 IAU nominal vs state+lib.',
    }),
    ('rsw_initial_vs_final', {
        'subfolder': 'RSWDiffTimeseries/Comparison',
        'fig_label': 'lib_fit_initial_vs_final',
        'sim_subset': ['SimPole_pole_lib_cov'],
        'title': 'RSW Difference vs NEP097 \u2014 FitPole nominal vs state+lib. (Fit.)',
    }),

    # ════════════════════════════════════════════════════════════════════
    # Formal errors comparisons: IAU vs Fit. side-by-side
    # ════════════════════════════════════════════════════════════════════
    ('formal_with_zoom', {
        'subfolder': 'FormalTimeseries/Comparison',
        'fig_label': 'state_iau_vs_fit',
        'sim_subset': ['IAUPole_initial_state', 'SimPole_initial_state'],
        'title': 'Formal Errors RSW \u2014 state (IAU) vs state (Fit.)',
    }),
    ('formal_with_zoom', {
        'subfolder': 'FormalTimeseries/Comparison',
        'fig_label': 'lib_iau_vs_fit',
        'sim_subset': ['IAUPole_pole_lib_cov', 'SimPole_pole_lib_cov'],
        'title': 'Formal Errors RSW \u2014 state+lib. (IAU) vs state+lib. (Fit.)',
    }),

    # ════════════════════════════════════════════════════════════════════
    # PoleMovement — Neptune pole trajectory for all / key estimations
    # ════════════════════════════════════════════════════════════════════
    # All simulations: full trajectory (RA, Dec vs time)
    ('pole_model', {
        'subfolder': 'PoleMovement',
        'fig_label': 'all_sims',
        'title':     'Neptune Pole Trajectory \u2014 All Estimations',
    }),
    # All simulations: deviation from nominal (Δα, Δδ vs time)
    ('pole_model_diff', {
        'subfolder': 'PoleMovement',
        'fig_label': 'all_sims_diff',
        'title':     'Neptune Pole Deviation from Nominal \u2014 All Estimations',
    }),
    # Key sims: baseline IAUPole + FitPole (from state sims) and state+lib. pair
    ('pole_model', {
        'subfolder': 'PoleMovement',
        'fig_label': 'key_sims',
        'sim_subset': _KEY_SIMS,
        'title':     'Neptune Pole Trajectory \u2014 Baseline and state+lib. Estimations',
    }),
    # Key sims deviation from nominal
    ('pole_model_diff', {
        'subfolder': 'PoleMovement',
        'fig_label': 'key_sims_diff',
        'sim_subset': _KEY_SIMS,
        'title':     'Neptune Pole Deviation from Nominal \u2014 Baseline and state+lib. Estimations',
    }),

    # ════════════════════════════════════════════════════════════════════
    # Residuals  — one figure per estimation (arcsec, legend on right)
    # ════════════════════════════════════════════════════════════════════
    ('residual_timeseries_by_id', {
        'subfolder': 'residuals',
        'title':     'Observation Residuals RA/Dec',
    }),

    # ════════════════════════════════════════════════════════════════════
    # Residual histograms  — one figure per estimation
    # ════════════════════════════════════════════════════════════════════
    ('residual_histogram_per_sim', {
        'subfolder': 'residuals',
        'fig_label': 'histogram',
        'title':     'Residual Histogram',
        'fit_gauss': True,
    }),

    # ════════════════════════════════════════════════════════════════════
    # 08_params/KeyResults  — state, state+lib. (IAU), state+lib. (Fit.)
    # ════════════════════════════════════════════════════════════════════
    ('param_state', {
        'subfolder':  '08_params/KeyResults',
        'fig_label':  'total_pos_vel',
        'variant':    'combined',
        'sim_subset': _KEY_SIMS,
        'show_annot': False,
        'title':      'Initial State Update — Total Magnitude',
    }),
    ('param_rsw', {
        'subfolder':  '08_params/KeyResults',
        'fig_label':  'rsw_pos_vel',
        'variant':    'combined',
        'sim_subset': _KEY_SIMS,
        'show_annot': False,
        'title':      'Initial State Update — RSW',
    }),
    ('param_pole_lib', {
        'subfolder':  '08_params/KeyResults',
        'fig_label':  'pole_lib',
        'variant':    'combined',
        'sim_subset': _KEY_SIMS,
        'show_annot': False,
        'title':      r'Pole Libration Update  (\u03b1\u2081, \u03b4\u2081)',
    }),
]

# ─── Parameter formal errors table ───────────────────────────────────────────
# Generates: FinalEstimation/table_param_formal.tex
# Lists all estimated parameters for SimPole_pole_lib_cov with formal errors
# (1σ from the post-fit covariance diagonal), alongside IAU₀ and FP₀ references.
PARAM_FORMAL_TABLE = {
    'sim_name':  'SimPole_pole_lib_cov',
    'subfolder': 'FinalEstimation',
}

# ─── Observation residuals table ─────────────────────────────────────────────
# Generates: FinalEstimation/table_obs_residuals.tex
# Per-dataset RMS O-C residuals (RA and Dec in arcsec) from the final estimation.
OBS_RESIDUALS_TABLE = {
    'sim_name':       'SimPole_pole_lib_cov',
    'subfolder':      'FinalEstimation',
    'raw_obs_folder': 'Observations/RawRelativeObservations',
    'label':          'tab:obs-residuals-final',
}

# ─── Final estimation tables config ─────────────────────────────────────────
# Triggers generate_final_estimation_tables() for IAU-baseline and Fit-baseline
# comparison of state+lib. simulations.
FINAL_ESTIMATION_TABLES = {
    'iau_sim':     'IAUPole_pole_lib_cov',
    'fitpole_sim': 'SimPole_pole_lib_cov',
}

SINGLE_SIM_TABLES = [
    'SimPole_pole_lib_cov',
]

# Pole comparison table: compares pole parameter estimates across IAUPole variants.
POLE_TABLE_SIMS = [
    'IAUPole_pole_pos_cov',
    'IAUPole_pole_lib_cov',
    'IAUPole_pole_pos_cov_pole_lib_cov',
]

# Key results table: parameter differences vs IAU and FitPole references for 3 sims.
# Output: 08_params/KeyResults/key_results_params.tex
KEY_RESULTS_CONFIG = {
    'subfolder': '08_params/KeyResults',
    'key_sims':  _KEY_SIMS,
    'caption':   (r'Key parameter estimates for selected simulations. '
                  r'IAU$_0$ and FP$_0$ are the initial values for the IAU 2015 '
                  r'and NEP097-fitted pole models, respectively. '
                  r'$\Delta_\mathrm{IAU} = \mathrm{Est.} - \mathrm{IAU}_0$; '
                  r'$\Delta_\mathrm{FP}  = \mathrm{Est.} - \mathrm{FP}_0$.'),
    'label':     'tab:key-results',
}

# ─── Human-readable descriptions for the naming-convention LaTeX table ────────
SIM_DESCRIPTIONS = {
    'IAUPole_initial_state':             r'Triton initial state only ($x, y, z, \dot{x}, \dot{y}, \dot{z}$), IAU 2015 pole model',
    'IAUPole_pole_pos_cov':              r'Initial state + pole pos.\ ($\alpha_0, \delta_0$), IAU 2015 pole model',
    'IAUPole_pole_lib_cov':              r'Initial state + pole lib.\ ($\alpha_1, \delta_1$), IAU 2015 pole model',
    'IAUPole_pole_pos_cov_pole_lib_cov': r'Initial state + pole pos.\ + lib.\ ($\alpha_0, \delta_0, \alpha_1, \delta_1$), IAU 2015 pole model',
    'SimPole_initial_state':             r'Triton initial state only ($x, y, z, \dot{x}, \dot{y}, \dot{z}$), NEP097-fitted pole model',
    'SimPole_pole_pos_cov':              r'Initial state + pole pos.\ ($\alpha_0, \delta_0$), NEP097-fitted pole model',
    'SimPole_pole_lib_cov':              r'Initial state + pole lib.\ ($\alpha_1, \delta_1$), NEP097-fitted pole model',
    'SimPole_pole_pos_cov_pole_lib_cov': r'Initial state + pole pos.\ + lib.\ ($\alpha_0, \delta_0, \alpha_1, \delta_1$), NEP097-fitted pole model',
}

# ─── LaTeX naming convention table ────────────────────────────────────────────
# Generates: table_naming_convention.tex in the timestamped output directory.
# Columns: Label | Description | Neptune Pole Model
NAMING_TABLE_TEX = r"""
\begin{table}[htbp]
  \centering
  \caption{Naming conventions used in figures for \texttt{CASE1\_Manual\_Bias}.}
  \label{tab:case1_naming}
  \begin{tabular}{lll}
    \toprule
    \textbf{Label} & \textbf{Description} & \textbf{Neptune Pole Model} \\
    \midrule
    \texttt{state}              & Triton initial state only ($x, y, z, \dot{x}, \dot{y}, \dot{z}$) & IAU 2015 \\
    \texttt{state+pos.}         & Initial state + pole position ($\alpha_0, \delta_0$)               & IAU 2015 \\
    \texttt{state+lib.}         & Initial state + pole libration ($\alpha_1, \delta_1$)              & IAU 2015 \\
    \texttt{state+pos.+lib.}    & Initial state + pole position + libration                          & IAU 2015 \\
    \addlinespace
    \texttt{state (Fit.)}           & Triton initial state only ($x, y, z, \dot{x}, \dot{y}, \dot{z}$) & NEP097-fitted \\
    \texttt{state+pos. (Fit.)}      & Initial state + pole position ($\alpha_0, \delta_0$)               & NEP097-fitted \\
    \texttt{state+lib. (Fit.)}      & Initial state + pole libration ($\alpha_1, \delta_1$)              & NEP097-fitted \\
    \texttt{state+pos.+lib. (Fit.)} & Initial state + pole position + libration                          & NEP097-fitted \\
    \bottomrule
  \end{tabular}
\end{table}
"""

# ─── Per-simulation colors (distinct, for timeseries figures) ────────────────
# Paul Tol "muted" palette — colorblind-safe, 9 maximally distinct colors.
# Each sim gets a unique color; group identity is conveyed by linestyle/marker.
SIM_COLORS = {
    # ── IAUPole: cool spectrum ──────────────────────────────────────────────
    'IAUPole_initial_state':             '#332288',  # indigo
    'IAUPole_pole_pos_cov':              '#88CCEE',  # cyan
    'IAUPole_pole_lib_cov':              '#44AA99',  # teal
    'IAUPole_pole_pos_cov_pole_lib_cov': '#117733',  # green
    # ── SimPole / FitPole: warm spectrum ───────────────────────────────────
    'SimPole_initial_state':             '#CC6677',  # rose
    'SimPole_pole_pos_cov':              '#DDCC77',  # sand
    'SimPole_pole_lib_cov':              '#882255',  # wine
    'SimPole_pole_pos_cov_pole_lib_cov': '#AA4499',  # purple
}

# Group colors used by aggregate figures with use_group_colors=True.
# _group_color() prefers this prefix-lookup over SIM_COLORS.
SIM_GROUP_COLORS = {
    'IAUPole': '#0072B2',  # blue       — all IAUPole variants
    'SimPole': '#D55E00',  # vermillion — all SimPole / FitPole variants
}

# ─── Per-simulation markers ───────────────────────────────────────────────────
# 'o'  IAUPole (IAU 2015 rotation model)
# 'D'  SimPole / FitPole (NEP097-fitted pole model)
SIM_MARKERS = {
    'IAUPole_initial_state':             'o',
    'IAUPole_pole_pos_cov':              'o',
    'IAUPole_pole_lib_cov':              'o',
    'IAUPole_pole_pos_cov_pole_lib_cov': 'o',
    'SimPole_initial_state':             'D',
    'SimPole_pole_pos_cov':              'D',
    'SimPole_pole_lib_cov':              'D',
    'SimPole_pole_pos_cov_pole_lib_cov': 'D',
}

# ─── Per-simulation linestyles ────────────────────────────────────────────────
# IAUPole: solid.  SimPole / FitPole: dashed.
SIM_LINESTYLE = {
    'IAUPole_initial_state':             '-',
    'IAUPole_pole_pos_cov':              '-',
    'IAUPole_pole_lib_cov':              '-',
    'IAUPole_pole_pos_cov_pole_lib_cov': '-',
    'SimPole_initial_state':             '--',
    'SimPole_pole_pos_cov':              '--',
    'SimPole_pole_lib_cov':              '--',
    'SimPole_pole_pos_cov_pole_lib_cov': '--',
}
