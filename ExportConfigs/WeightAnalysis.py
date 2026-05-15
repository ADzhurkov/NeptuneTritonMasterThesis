"""Export configuration: WeightAnalysis
Comparison of weight schemes — all variants estimate initial_state only,
using SimPole starting conditions with manual bias applied.
"""

DATASET_LABEL = 'WeightAnalysis'  # TODO: set to matching key in DashInteractivePlotFull DATA_FILES

SELECTED_SIMS = [
    'id_weights',
    'id_new_2_weights',
    'tf_weights',
    'tf_weights_no_limit',
    #'hybrid_weights',
    #'hybrid_old_weights',
    'hybrid_new_id_weights',
    'hybrid_old_new_id_weights',
]

SIM_LABELS = [
    'per file',                     # id_weights
    'scaled per file',              # id_new_2_weights
    'per timeframe',                # tf_weights
    'per timeframe free',           # tf_weights_no_limit
    #'hybrid geom.',                 # hybrid_weights
    #'hybrid arith.',                # hybrid_old_weights
    'scaled hybrid geom.',          # hybrid_new_id_weights
    'scaled hybrid arith.',         # hybrid_old_new_id_weights
]

OUTPUT_DIR = 'ThesisFigures'

FIGURES_TO_EXPORT = [
    # ── Standalone legend ─────────────────────────────────────────────────────
    ('legend', {
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
    # ── Goodness of fit (3-panel: WRMS / RMS / Cost) ─────────────────────────
    ('gof_combined', {'show_initial': True, 'title': 'Goodness of Fit — Weight Scheme Comparison'}),
    # ── RMS vs NEP097 [km] ────────────────────────────────────────────────────
    ('rms_compare', {'title': 'Total RMS vs NEP097 — Weight Scheme Comparison'}),
    # ── RMS / formal error RMS ratio (total and per-direction) ───────────────
    ('rms_ratio', {'title': 'RMS / Formal Error RMS — Weight Scheme Comparison'}),
    ('rsw_ratio', {'title': 'RSW RMS / Formal σ RMS — Weight Scheme Comparison'}),
    # ── RSW Statistics vs NEP097 — all weight schemes ─────────────────────────
    ('rsw_stats', {
        'fig_label': 'all',
        'title': 'RSW Statistics vs NEP097 — Weight Scheme Comparison',
    }),
    # ── RSW difference with zoom ──────────────────────────────────────────────
    ('rsw_with_zoom', {
        'fig_label': 'subset',
        'sim_subset': ['id_weights', 'id_new_2_weights', 'tf_weights', 'hybrid_old_new_id_weights'],
        'title': 'RSW Difference vs NEP097 — Weight Scheme Comparison',
    }),
    # ── Formal errors with zoom ───────────────────────────────────────────────
    ('formal_with_zoom', {
        'fig_label': 'subset',
        'sim_subset': ['id_weights', 'id_new_2_weights', 'tf_weights', 'hybrid_old_new_id_weights'],
        'title': 'Formal Errors RSW — Weight Scheme Comparison',
    }),
]

SINGLE_SIM_TABLES = []

# Human-readable descriptions for the naming-convention LaTeX table.
SIM_DESCRIPTIONS = {
    'id_new_2_weights':          r'Per-file RMSE weights, scaled by $\sigma_{\mathrm{global}} / \sigma_{\mathrm{file}}$ (variant 2)',
    'id_weights':                r'Per-observation-file RMSE weights ($1/\sigma_{\mathrm{file}}^2$)',
    'tf_weights':                r'Per-timeframe RMSE weights ($1/\sigma_{\mathrm{tf}}^2$)',
    'tf_weights_no_limit':       r'Per-timeframe RMSE weights, no upper limit on weight magnitude',
    'hybrid_weights':            r'Hybrid G: geometric mean of per-file and per-timeframe weights',
    'hybrid_old_weights':        r'Hybrid A: arithmetic mean of per-file and per-timeframe weights',
    'hybrid_new_id_weights':     r'Hybrid G with ID\,v2 scaling applied to per-file component',
    'hybrid_old_new_id_weights': r'Hybrid A with ID\,v2 scaling applied to per-file component',
}

# Per-simulation colors — Paul Tol "muted" palette (colorblind-safe, 9 colors).
# Distinct from the Wong palette used in CASE1_Manual_Bias.
# All weight-scheme sims use circle markers (no IAU/SimPole distinction here).
SIM_MARKERS = {sn: 'o' for sn in SELECTED_SIMS}

SIM_COLORS = {
    'id_weights':              '#332288',  # indigo
    'id_new_2_weights':        '#117733',  # green
    'tf_weights':              '#88CCEE',  # cyan
    'tf_weights_no_limit':     '#DDCC77',  # sand
    'hybrid_weights':          '#CC6677',  # rose
    'hybrid_old_weights':      '#882255',  # wine
    'hybrid_new_id_weights':   '#AA4499',  # purple
    'hybrid_old_new_id_weights': '#999933', # olive
}
SIM_LINESTYLE = {sn: '-' for sn in SIM_COLORS}
