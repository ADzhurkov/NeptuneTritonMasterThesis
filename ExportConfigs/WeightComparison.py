"""Export configuration: WeightComparison
Cross-dataset comparison of two weight-scheme analyses:
  • WeightAnalysis_Old  — initial_state only, old dataset (WA-IAU)
  • WeightAnalysis_Pole — initial_state + pole (pole_lib_cov), current dataset (WA-FIT-PL)

Multi-panel figures compare the two datasets side-by-side.
Single-dataset figures use the WA-FIT-PL (current) dataset as the primary reference.
"""

from ExportConfigs import WeightAnalysis as _wa
import ExportConfigs.WeightAnalysis_Pole as _wa_pole

# Primary dataset for single-dataset figures (WA-FIT-PL sims).
DATASET_LABEL = 'WeightAnalysis_Pole'

# Use same sims / labels as the WeightAnalysis_Old config.
SELECTED_SIMS = _wa_pole.SELECTED_SIMS
SIM_LABELS    = _wa_pole.SIM_LABELS

OUTPUT_DIR       = 'ThesisFigures'
SUPPRESS_TITLES  = True  # strip suptitles from all exported figures

# ── Per-sim opacity (0.0–1.0) ─────────────────────────────────────────────────
# Lines/markers drawn on top of others can be made more transparent so all
# curves remain distinguishable.  Sims not listed default to 1.0 (fully opaque).
SIM_ALPHAS = {
    'id_weights':               1.0,
    'id_new_2_weights':         1.0,
    'tf_weights':               1.0,
    'tf_weights_no_limit':      1.0,
    'hybrid_new_id_weights':    1.0,
    'hybrid_old_new_id_weights': 1.0,
    'id_new_2_weights_no_cov':  1.0,
}

# ── Figure group export control ───────────────────────────────────────────────
# Set EXPORT_GROUPS to a list of group tags to export only those groups, or
# leave as None to export all figures.
#
# Available groups:
#   'rsw_diff'    — RSW difference time series and summary grids (rsw_with_zoom,
#                   rsw_rms_ratio_grid) — these are the slow timeseries plots
#   'formal_rsw'  — Formal error time series (formal_with_zoom) — also slow
#   'weights'     — Per-file / per-timeframe weight & uncertainty figures
#   'other'       — Comparison, GoF, legend, RMS summary and all remaining plots
#
# Examples:
#   EXPORT_GROUPS = ['rsw_diff', 'formal_rsw']  # only timeseries figures
#   EXPORT_GROUPS = ['weights']                  # only weight figures
#   EXPORT_GROUPS = ['other']                    # only fast summary figures
EXPORT_GROUPS = ['rsw_diff','formal_rsw','other'] #None  # export all groups

# Two datasets and their display titles.
_DATASETS       = ['WeightAnalysis_Old', 'WeightAnalysis_Pole']
_DATASET_TITLES = ['WA-IAU', 'WA-FIT-PL']

# Per-file weight subgroups.
_GROUP1_SIMS = ['id_weights', 'id_new_2_weights']                        # per file, scaled per file
_GROUP2_SIMS = ['id_new_2_weights', 'tf_weights']                        # scaled per file, per timeframe
_GROUP3_SIMS = ['tf_weights', 'tf_weights_no_limit']                     # per timeframe, per timeframe free
_GROUP4_SIMS = ['hybrid_new_id_weights', 'hybrid_old_new_id_weights']    # scaled hybrid geom., scaled hybrid arith.

# Files for per-timeframe uncertainty plots (verify ref_point_id values in your data).
_TF_FILE_IDS = ['874_nm0004', '337_nm0085']

FIGURES_TO_EXPORT = [
    # ── Multi-panel comparison (WA-IAU vs WA-FIT-PL) ─────────────────────────
    ('rms_compare_multi', {
        'group':          'other',
        'subfolder':      'comparison',
        'datasets':       _DATASETS,
        'dataset_titles': _DATASET_TITLES,
        'title':          'RMS vs NEP097 — Weight Scheme Comparison',
        'show_annot':     False,
    }),
    ('gof_metric_multi', {
        'group':          'other',
        'fig_label':      'wrms',
        'subfolder':      'comparison',
        'datasets':       _DATASETS,
        'dataset_titles': _DATASET_TITLES,
        'metric':         'wrms',
        'show_initial':   True,
        'title':          'WRMS — Weight Scheme Comparison',
    }),
    ('gof_metric_multi', {
        'group':          'other',
        'fig_label':      'rms',
        'subfolder':      'comparison',
        'datasets':       _DATASETS,
        'dataset_titles': _DATASET_TITLES,
        'metric':         'rms',
        'show_initial':   True,
        'title':          'RMS — Weight Scheme Comparison',
    }),
    ('gof_metric_multi', {
        'group':          'other',
        'fig_label':      'cost',
        'subfolder':      'comparison',
        'datasets':       _DATASETS,
        'dataset_titles': _DATASET_TITLES,
        'metric':         'cost',
        'show_initial':   True,
        'title':          'Cost Function — Weight Scheme Comparison',
    }),
    ('formal_rms_multi', {
        'group':          'other',
        'subfolder':      'comparison',
        'datasets':       _DATASETS,
        'dataset_titles': _DATASET_TITLES,
        'title':          'Formal Error RMS — Weight Scheme Comparison',
    }),
    ('rms_ratio_multi', {
        'group':          'other',
        'subfolder':      'comparison',
        'datasets':       _DATASETS,
        'dataset_titles': _DATASET_TITLES,
        'title':          'RMS / Formal Error RMS — Weight Scheme Comparison',
    }),

    # ── Single-dataset figures for WA-FIT-PL ──────────────────────────────────
    ('legend', {
        'group':     'other',
        'subfolder': 'legend',
        'title': 'Weight Scheme Legend',
        'ncols': 2,
        'shape_labels': {
            'o': 'All variants estimate initial_state only  (SimPole, manual bias)',
        },
        'notes': [
            'Abbreviations:',
            '  ID        per-observation-file RMSE weights',
            '  ID v2     ID scaled by \u03c3_global/\u03c3_file',
            '  TF        per-timeframe RMSE weights',
            '  G / A     two hybrid combination variants (G: ID v2, A: old ID)',
            '  no cap    no upper limit applied to computed weights',
        ],
    }),
    ('gof_combined', {
        'group':        'other',
        'subfolder':    'gof',
        'show_initial': True,
        'title':        'Goodness of Fit — WA-FIT-PL Weight Scheme Comparison',
    }),
    ('rms_compare', {
        'group':     'other',
        'subfolder': 'rms',
        'title':     'Total RMS vs NEP097 — WA-FIT-PL Weight Scheme Comparison',
    }),
    ('rms_ratio', {
        'group':     'other',
        'subfolder': 'rms',
        'title':     'RMS / Formal Error RMS — WA-FIT-PL Weight Scheme Comparison',
    }),
    ('rsw_ratio', {
        'group':     'other',
        'subfolder': 'rsw',
        'title':     'RSW RMS / Formal \u03c3 RMS — WA-FIT-PL Weight Scheme Comparison',
    }),
    ('rsw_stats', {
        'group':     'other',
        'fig_label': 'all',
        'subfolder': 'rsw',
        'title':     'RSW Statistics vs NEP097 — WA-FIT-PL Weight Scheme Comparison',
    }),
    ('rsw_rms_ratio_grid', {
        'group':     'rsw_diff',
        'fig_label': 'all',
        'subfolder': 'rsw',
        'title':     'RSW RMS Diff, Formal Error RMS and Ratio — WA-FIT-PL Weight Scheme Comparison',
    }),

    # ── Per-file observation uncertainty [mas] = 1/sqrt(weight) ──────────────
    ('weight_uncertainty_per_file', {
        'group':     'weights',
        'subfolder': 'per_file',
        'title':     'Per-File Observation Uncertainty — WA-FIT-PL',
    }),
    ('weight_uncertainty_overlay', {
        'group':     'weights',
        'subfolder': 'per_file',
        'title':     'Per-File Uncertainty — All Weight Schemes (WA-FIT-PL)',
    }),

    # ── Per-file RMS update (initial − final) [mas] — all schemes overlaid ───
    ('rms_delta_per_file', {
        'group':     'weights',
        'subfolder': 'per_file',
        'title':     'Per-File RMS Update (Initial − Final) — WA-FIT-PL',
    }),

    # ── Initial vs final RMS per file — all schemes ───────────────────────────
    ('rms_initial_final_per_file', {
        'group':     'weights',
        'subfolder': 'per_file',
        'title':     'Per-File Initial vs Final RMS — WA-FIT-PL',
    }),

    # ── Number of observations per timeframe per file ─────────────────────────
    ('n_obs_per_timeframe', {
        'group':     'weights',
        'subfolder': 'per_file',
        'title':     'N_obs / N_timeframes per File — WA-FIT-PL',
    }),

    # ── Per-file subset: Group 1 — per file, scaled per file ─────────────────────
    ('weight_uncertainty_overlay', {
        'group':      'weights',
        'fig_label':  'g1',
        'subfolder':  'per_file',
        'sim_subset': _GROUP1_SIMS,
        'title':      'Per-File Uncertainty — Group 1 (per file, scaled per file) (WA-FIT-PL)',
    }),
    ('weight_uncertainty_per_file', {
        'group':      'weights',
        'fig_label':  'g1',
        'subfolder':  'per_file',
        'sim_subset': _GROUP1_SIMS,
        'title':      'Per-File Observation Uncertainty — Group 1 (per file, scaled per file) (WA-FIT-PL)',
    }),
    ('rms_delta_per_file', {
        'group':      'weights',
        'fig_label':  'g1',
        'subfolder':  'per_file',
        'sim_subset': _GROUP1_SIMS,
        'title':      'Per-File RMS Update (Initial − Final) — Group 1 (per file, scaled per file) (WA-FIT-PL)',
    }),
    ('rms_delta_per_file', {
        'group':        'weights',
        'fig_label':    'g1_no83nm0083',
        'subfolder':    'per_file',
        'sim_subset':   _GROUP1_SIMS,
        'file_exclude': ['83_nm0083'],
        'title':        'Per-File RMS Update (Initial − Final) — Group 1 excl. 83 nm0083 (WA-FIT-PL)',
    }),
    ('rms_initial_final_per_file', {
        'group':      'weights',
        'fig_label':  'g1',
        'subfolder':  'per_file',
        'sim_subset': _GROUP1_SIMS,
        'title':      'Per-File Initial vs Final RMS — Group 1 (per file, scaled per file) (WA-FIT-PL)',
    }),
    ('weight_uncertainty_per_timeframe', {
        'group':         'weights',
        'fig_label':     'g1',
        'subfolder':     'per_timeframe',
        'sim_subset':    _GROUP1_SIMS,
        'file_ids':      _TF_FILE_IDS,
        'title':         'Per-Timeframe Uncertainty — Group 1 (per file, scaled per file) (WA-FIT-PL)',
        'show_suptitle': False,
        'tick_fontsize': 14,
    }),
    ('rsw_compare', {
        'group':      'rsw_diff',
        'fig_label':  'g1',
        'subfolder':  'rsw',
        'sim_subset': _GROUP1_SIMS,
        'title':      'RSW Difference vs NEP097 — Group 1 (per file, scaled per file) (WA-FIT-PL)',
    }),
    ('formal_compare', {
        'group':      'formal_rsw',
        'fig_label':  'g1',
        'subfolder':  'rsw',
        'sim_subset': _GROUP1_SIMS,
        'title':      'Formal Errors RSW — Group 1 (per file, scaled per file) (WA-FIT-PL)',
    }),
    ('rsw_rms_ratio_grid', {
        'group':      'rsw_diff',
        'fig_label':  'g1',
        'subfolder':  'rsw',
        'sim_subset': _GROUP1_SIMS,
        'title':      'RSW RMS Diff, Formal Error RMS and Ratio — Group 1 (per file, scaled per file) (WA-FIT-PL)',
    }),

    # ── Per-file subset: Group 2 — scaled per file, per timeframe ───────────────
    ('weight_uncertainty_overlay', {
        'group':      'weights',
        'fig_label':  'g2',
        'subfolder':  'per_file',
        'sim_subset': _GROUP2_SIMS,
        'title':      'Per-File Uncertainty — Group 2 (scaled per file, per timeframe) (WA-FIT-PL)',
    }),
    ('weight_uncertainty_per_file', {
        'group':      'weights',
        'fig_label':  'g2',
        'subfolder':  'per_file',
        'sim_subset': _GROUP2_SIMS,
        'title':      'Per-File Observation Uncertainty — Group 2 (scaled per file, per timeframe) (WA-FIT-PL)',
    }),
    ('rms_delta_per_file', {
        'group':      'weights',
        'fig_label':  'g2',
        'subfolder':  'per_file',
        'sim_subset': _GROUP2_SIMS,
        'title':      'Per-File RMS Update (Initial − Final) — Group 2 (scaled per file, per timeframe) (WA-FIT-PL)',
    }),
    ('rms_initial_final_per_file', {
        'group':      'weights',
        'fig_label':  'g2',
        'subfolder':  'per_file',
        'sim_subset': _GROUP2_SIMS,
        'title':      'Per-File Initial vs Final RMS — Group 2 (scaled per file, per timeframe) (WA-FIT-PL)',
    }),
    ('rsw_compare', {
        'group':      'rsw_diff',
        'fig_label':  'g2',
        'subfolder':  'rsw',
        'sim_subset': _GROUP2_SIMS,
        'title':      'RSW Difference vs NEP097 — Group 2 (scaled per file, per timeframe) (WA-FIT-PL)',
    }),
    ('formal_compare', {
        'group':      'formal_rsw',
        'fig_label':  'g2',
        'subfolder':  'rsw',
        'sim_subset': _GROUP2_SIMS,
        'title':      'Formal Errors RSW — Group 2 (scaled per file, per timeframe) (WA-FIT-PL)',
    }),
    ('rsw_compare', {
        'group':      'rsw_diff',
        'fig_label':  'g3',
        'subfolder':  'rsw',
        'sim_subset': _GROUP3_SIMS,
        'title':      'RSW Difference vs NEP097 — Group 3 (per timeframe, per timeframe free) (WA-FIT-PL)',
    }),
    ('formal_compare', {
        'group':      'formal_rsw',
        'fig_label':  'g3',
        'subfolder':  'rsw',
        'sim_subset': _GROUP3_SIMS,
        'title':      'Formal Errors RSW — Group 3 (per timeframe, per timeframe free) (WA-FIT-PL)',
    }),
    ('rsw_rms_ratio_grid', {
        'group':      'rsw_diff',
        'fig_label':  'g2',
        'subfolder':  'rsw',
        'sim_subset': _GROUP2_SIMS,
        'title':      'RSW RMS Diff, Formal Error RMS and Ratio — Group 2 (scaled per file, per timeframe) (WA-FIT-PL)',
    }),

    # ── Per-file subset: Group 4 — scaled hybrid geom., scaled hybrid arith. ────
    ('weight_uncertainty_overlay', {
        'group':      'weights',
        'fig_label':  'g4',
        'subfolder':  'per_file',
        'sim_subset': _GROUP4_SIMS,
        'title':      'Per-File Uncertainty — Group 4 (scaled hybrid geom., scaled hybrid arith.) (WA-FIT-PL)',
    }),
    ('weight_uncertainty_per_file', {
        'group':      'weights',
        'fig_label':  'g4',
        'subfolder':  'per_file',
        'sim_subset': _GROUP4_SIMS,
        'title':      'Per-File Observation Uncertainty — Group 4 (scaled hybrid geom., scaled hybrid arith.) (WA-FIT-PL)',
    }),
    ('rms_delta_per_file', {
        'group':      'weights',
        'fig_label':  'g4',
        'subfolder':  'per_file',
        'sim_subset': _GROUP4_SIMS,
        'title':      'Per-File RMS Update (Initial − Final) — Group 4 (scaled hybrid geom., scaled hybrid arith.) (WA-FIT-PL)',
    }),
    ('rms_initial_final_per_file', {
        'group':      'weights',
        'fig_label':  'g4',
        'subfolder':  'per_file',
        'sim_subset': _GROUP4_SIMS,
        'title':      'Per-File Initial vs Final RMS — Group 4 (scaled hybrid geom., scaled hybrid arith.) (WA-FIT-PL)',
    }),
    # ── RSW diff and formal errors (Group 4) ─────────────────────────────────
    ('rsw_compare', {
        'group':      'rsw_diff',
        'fig_label':  'g4',
        'subfolder':  'rsw',
        'sim_subset': _GROUP4_SIMS,
        'title':      'RSW Difference vs NEP097 — Group 4 (scaled hybrid geom., scaled hybrid arith.) (WA-FIT-PL)',
    }),
    ('formal_compare', {
        'group':      'formal_rsw',
        'fig_label':  'g4',
        'subfolder':  'rsw',
        'sim_subset': _GROUP4_SIMS,
        'title':      'Formal Errors RSW — Group 4 (scaled hybrid geom., scaled hybrid arith.) (WA-FIT-PL)',
    }),
    ('rsw_rms_ratio_grid', {
        'group':      'rsw_diff',
        'fig_label':  'g4',
        'subfolder':  'rsw',
        'sim_subset': _GROUP4_SIMS,
        'title':      'RSW RMS Diff, Formal Error RMS and Ratio — Group 4 (scaled hybrid geom., scaled hybrid arith.) (WA-FIT-PL)',
    }),

    # ── Weights and uncertainty vs datetime — WA-FIT-PL ──────────────────────
    ('weight_vs_datetime', {
        'group':     'weights',
        'subfolder': 'weights_vs_time',
        'title':     'Observation Weights vs Time — WA-FIT-PL',
    }),
    ('uncertainty_vs_datetime', {
        'group':     'weights',
        'subfolder': 'weights_vs_time',
        'title':     'Observation Uncertainty \u03c3 = 1/\u221aweight vs Time [mas] — WA-FIT-PL',
    }),

    # ── WA-IAU analysis (WeightAnalysis_Old dataset) ──────────────────────────
    ('rsw_rms_ratio_grid', {
        'group':       'rsw_diff',
        'fig_label':   'all',
        'subfolder':   'WA_IAU_analysis/rsw',
        'from_config': 'WeightAnalysis_Old',
        'title':       'RSW RMS Diff, Formal Error RMS and Ratio — WA-IAU',
    }),
    ('rsw_compare', {
        'group':       'rsw_diff',
        'fig_label':   'g1',
        'subfolder':   'WA_IAU_analysis/rsw',
        'from_config': 'WeightAnalysis_Old',
        'sim_subset':  _GROUP1_SIMS,
        'title':       'RSW Difference vs NEP097 — Group 1 (per file, scaled per file) (WA-IAU)',
    }),
    ('formal_compare', {
        'group':       'formal_rsw',
        'fig_label':   'g1',
        'subfolder':   'WA_IAU_analysis/rsw',
        'from_config': 'WeightAnalysis_Old',
        'sim_subset':  _GROUP1_SIMS,
        'title':       'Formal Errors RSW — Group 1 (per file, scaled per file) (WA-IAU)',
    }),
    ('rsw_compare', {
        'group':       'rsw_diff',
        'fig_label':   'g2',
        'subfolder':   'WA_IAU_analysis/rsw',
        'from_config': 'WeightAnalysis_Old',
        'sim_subset':  _GROUP2_SIMS,
        'title':       'RSW Difference vs NEP097 — Group 2 (scaled per file, per timeframe) (WA-IAU)',
    }),
    ('formal_compare', {
        'group':       'formal_rsw',
        'fig_label':   'g2',
        'subfolder':   'WA_IAU_analysis/rsw',
        'from_config': 'WeightAnalysis_Old',
        'sim_subset':  _GROUP2_SIMS,
        'title':       'Formal Errors RSW — Group 2 (scaled per file, per timeframe) (WA-IAU)',
    }),
    ('rsw_compare', {
        'group':       'rsw_diff',
        'fig_label':   'g3',
        'subfolder':   'WA_IAU_analysis/rsw',
        'from_config': 'WeightAnalysis_Old',
        'sim_subset':  _GROUP3_SIMS,
        'title':       'RSW Difference vs NEP097 — Group 3 (per timeframe, per timeframe free) (WA-IAU)',
    }),
    ('formal_compare', {
        'group':       'formal_rsw',
        'fig_label':   'g3',
        'subfolder':   'WA_IAU_analysis/rsw',
        'from_config': 'WeightAnalysis_Old',
        'sim_subset':  _GROUP3_SIMS,
        'title':       'Formal Errors RSW — Group 3 (per timeframe, per timeframe free) (WA-IAU)',
    }),
    ('rsw_compare', {
        'group':       'rsw_diff',
        'fig_label':   'g4',
        'subfolder':   'WA_IAU_analysis/rsw',
        'from_config': 'WeightAnalysis_Old',
        'sim_subset':  _GROUP4_SIMS,
        'title':       'RSW Difference vs NEP097 — Group 4 (scaled hybrid geom., scaled hybrid arith.) (WA-IAU)',
    }),
    ('formal_compare', {
        'group':       'formal_rsw',
        'fig_label':   'g4',
        'subfolder':   'WA_IAU_analysis/rsw',
        'from_config': 'WeightAnalysis_Old',
        'sim_subset':  _GROUP4_SIMS,
        'title':       'Formal Errors RSW — Group 4 (scaled hybrid geom., scaled hybrid arith.) (WA-IAU)',
    }),
    ('weight_uncertainty_overlay', {
        'group':       'weights',
        'fig_label':   'g1',
        'subfolder':   'WA_IAU_analysis/per_file',
        'from_config': 'WeightAnalysis_Old',
        'sim_subset':  _GROUP1_SIMS,
        'title':       'Per-File Uncertainty — Group 1 (per file, scaled per file) (WA-IAU)',
    }),
    ('n_obs_per_timeframe', {
        'group':       'weights',
        'subfolder':   'WA_IAU_analysis/per_file',
        'from_config': 'WeightAnalysis_Old',
        'title':       'N_obs / N_timeframes per File — WA-IAU',
    }),
    ('weight_uncertainty_per_timeframe', {
        'group':         'weights',
        'fig_label':     'g1',
        'subfolder':     'WA_IAU_analysis/per_timeframe',
        'from_config':   'WeightAnalysis_Old',
        'sim_subset':    _GROUP1_SIMS,
        'file_ids':      _TF_FILE_IDS,
        'title':         'Per-Timeframe Uncertainty — Group 1 (per file, scaled per file) (WA-IAU)',
        'show_suptitle': False,
        'tick_fontsize': 14,
    }),
    ('rms_delta_per_file', {
        'group':       'weights',
        'fig_label':   'g1',
        'subfolder':   'WA_IAU_analysis/per_file',
        'from_config': 'WeightAnalysis_Old',
        'sim_subset':  _GROUP1_SIMS,
        'title':       'Per-File RMS Update (Initial − Final) — Group 1 (per file, scaled per file) (WA-IAU)',
    }),
    ('rms_delta_per_file', {
        'group':       'weights',
        'fig_label':   'g2',
        'subfolder':   'WA_IAU_analysis/per_file',
        'from_config': 'WeightAnalysis_Old',
        'sim_subset':  _GROUP2_SIMS,
        'title':       'Per-File RMS Update (Initial − Final) — Group 2 (scaled per file, scaled hybrid G, scaled hybrid A) (WA-IAU)',
    }),
    ('weight_vs_datetime', {
        'group':       'weights',
        'subfolder':   'WA_IAU_analysis/weights_vs_time',
        'from_config': 'WeightAnalysis_Old',
        'title':       'Observation Weights vs Time — WA-IAU',
    }),
    ('uncertainty_vs_datetime', {
        'group':       'weights',
        'subfolder':   'WA_IAU_analysis/weights_vs_time',
        'from_config': 'WeightAnalysis_Old',
        'title':       'Observation Uncertainty \u03c3 = 1/\u221aweight vs Time [mas] — WA-IAU',
    }),
]

SINGLE_SIM_TABLES = []

# Per-sim colors / markers / linestyles — imported from WeightAnalysis.
SIM_COLORS    = _wa_pole.SIM_COLORS
SIM_MARKERS   = _wa_pole.SIM_MARKERS
# tf_weights gets a dashed line so it stays visible against the light cyan color.
SIM_LINESTYLE = {**_wa_pole.SIM_LINESTYLE, 'tf_weights_no_limit': '--'}

# ── Dataset overview table (Table 7.1-style): two analyses → descriptions ─────
DATASET_OVERVIEW_TABLE = {
    'rows': [
        ('WA-IAU',    'IAU pole model, SPICE Neptune position, initial state estimation only'),
        ('WA-FIT-PL', 'NEP097-fitted pole and initial state, initial state and pole libration estimation'),
    ],
    'caption':   r'Naming convention for the two weighting strategy analyses.',
    'label':     'tab:wa-analysis-naming',
    'subfolder': 'tables',
}

# ── Extra naming-convention tables (Table 7.2-style per dataset) ──────────────
EXTRA_NAMING_TABLES = [
    {
        'selected_sims':    _wa_pole.SELECTED_SIMS,
        'sim_labels':       _wa_pole.SIM_LABELS,
        'sim_descriptions': _wa_pole.SIM_DESCRIPTIONS,
        'dataset_label':    'WA-FIT-PL',
        'subfolder':        'tables',
        'out_filename':     'table_naming_conventions_wa_fit_pl.tex',
    },
]

# ── Residual statistics table: mean and std of final residuals per scheme ─────
RESIDUAL_STATS_TABLE = {
    'subfolder':   'tables',
    'data_source': 'final',
    'caption':     r'Mean and standard deviation of final residuals per weighting scheme (WA-FIT-PL) [mas].',
    'label':       'tab:residual-stats-wa-fit-pl',
}
