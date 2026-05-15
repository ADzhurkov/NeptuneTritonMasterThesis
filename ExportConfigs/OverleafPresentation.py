"""Export configuration: OverleafPresentation
Figures sized for a 16:9 Beamer presentation slide deck.

Slide constraints (16:9 = 160×90 mm, usable ~154×73 mm after title/footer):
  • portrait figures (\\includegraphics[width=0.55\\textwidth]) → ~3.33 in wide
  • landscape figures (width=0.85\\textwidth)                  → ~5.15 in wide
  • landscape figures (width=0.75\\textwidth)                  → ~4.55 in wide
  • max usable height ~2.87 in

Strategy: this template only configures *what* to export and *with what figsize*;
suptitles are stripped via SUPPRESS_TITLES (slide title carries the message).
Source data is pulled from existing ExportConfig datasets via `from_config`.

  ─ Source dataset map ───────────────────────────────────────────────────────
   #1, #2a, #2b   ObservationalDataset       (PoleEst_MB)
   #3, #4, #5     SimObs_ParameterAnalysis   (SimObs_ParameterAnalysis)
   #6, #7         WeightAnalysis_Pole        (WeightAnalysis_Pole)

Font sizes (axes 14pt / ticks 12pt / legend 11pt) are passed through as
explicit kwargs to the underlying plot functions
(`axes_fontsize` / `tick_fontsize` / `legend_fontsize`); the `mpl_*`
functions in MatplotlibExport.py accept these on a per-figure basis.
"""

# ── Primary dataset (used when from_config is not specified) ──────────────────
# SimObs is primary because most figures (#3, #4, #5) come from it.
import ExportConfigs.SimObs_ParameterAnalysis as _simobs

DATASET_LABEL = _simobs.DATASET_LABEL  # 'SimObs_ParameterAnalysis'
SELECTED_SIMS = _simobs.SELECTED_SIMS
SIM_LABELS    = _simobs.SIM_LABELS

OUTPUT_DIR      = 'ThesisFigures/OverleafPresentation'
SUPPRESS_TITLES = True   # strip suptitles; slide title carries the message

# ── Slide-tuned figsizes (inches) ─────────────────────────────────────────────
# Landscape 3-panel RSW (3×1, R/S/W rows) — width 0.85\textwidth
_FIGSIZE_RSW_LANDSCAPE   = (10.0, 5.5)
# Landscape 2-panel RA/Dec residual time series — width 0.85\textwidth
_FIGSIZE_RES_LANDSCAPE   = (10.0, 5.0)
# Landscape single panel (per-year obs count, RMS compare) — width 0.75–0.85
_FIGSIZE_SINGLE_BIG      = (10.0, 4.5)
_FIGSIZE_SINGLE_MEDIUM   = ( 8.0, 4.5)

# Slide-tuned font sizes (axes 14pt / ticks 12pt / legend 11pt).
_FONT_AXES   = 14
_FONT_TICK   = 12
_FONT_LEGEND = 11
_FONT_TITLE  = 14

# ── Sim subsets reused in figure entries ──────────────────────────────────────
# Figure #3: pure propagation (no estimation), IAU vs Jacobson — from SimObs.
_NO_EST_PAIR = ['no_est_IAU', 'no_est_Jacobson']

# Figure #4: state-only estimation, IAU vs Jacobson — from SimObs.
_STATE_PAIR  = ['initial_state_IAU', 'initial_state_Jacobson']

# Figure #5: 4 IAU pole-estimation variants (state, +pos., +lib., +pos.+lib.).
_POLE_IAU_5 = [
    'initial_state_IAU',
    'pole_pos_IAU',
    'pole_lib_IAU',
    'pole_pos_lib_IAU',
]

# Figure #6, #7: per-file vs scaled-per-file weighting — from WeightAnalysis_Pole.
_GROUP1_SIMS = ['id_weights', 'id_new_2_weights']

# ── Path to obs_analysis_data.npy (figures #1, #2a, #2b) ──────────────────────
# Mirrors the path used in ObservationalDataset config.
_OBS_ANALYSIS_DATA = 'Results/ObservationsAnalysis/obs_analysis_data.npy'

# ── Path to cached Sim. Fit. Pole base params (figures #11) ───────────────────
# Built once by `python build_pole_base_cache.py` from the SimObs pickle.
# When present, lets pole_model_compare resolve Sim. Fit. Pole / Real. Fit. Pole
# WITHOUT loading the 6 GB SimObs dataset — critical on memory-constrained runs.
_SIM_FIT_POLE_CACHE = 'Results/cache/sim_fit_pole_base.npy'

FIGURES_TO_EXPORT = [
    # ════════════════════════════════════════════════════════════════════════
    # 01  Per-ID accepted residuals vs NEP097 (RA + DEC, landscape 2-panel)
    # ════════════════════════════════════════════════════════════════════════
    ('obs_analysis_spice_by_id_accepted', {
        'subfolder':                '01_residuals_by_id',
        'fig_label':                'pres',
        'from_config':              'ObservationalDataset',
        'data_path':                _OBS_ANALYSIS_DATA,
        # No legend → narrower aspect ratio.  Beamer "usable" area is ~6.06 × 2.87 in
        # for width=\\textwidth; 0.85\\textwidth ≈ 5.15 × 2.87 in.  Use 10×4 so two
        # stacked RA/Dec panels each clear ~1.7 in tall when scaled.
        'figsize':                  (10.0, 4.0),
        'axes_fontsize':            _FONT_AXES,
        'tick_fontsize':            _FONT_TICK,
        'legend_fontsize':          _FONT_LEGEND,
        'legend_observatory_names': True,
        'show_legend':              False,
        # Matched RA/Dec y-limits + ticks so the two panels read at the same
        # vertical scale; unit label spelled out.
        'match_y_axes':             True,
        'y_unit_label':             "''",
        'panel_hspace':             0.30,
        # Both panels clamped to the same range, with 0.5-arcsec steps (7 ticks
        # at -1.5/-1.0/-0.5/0/0.5/1.0/1.5).  0.25 was tried but crowded the
        # short-aspect panels and caused ylabel/tick collisions.
        'y_lim':                    (-1.5, 1.5),
        'y_tick_step':              0.5,
        'title':                    '',
    }),

    # ════════════════════════════════════════════════════════════════════════
    # 02a  Observation count per year (landscape, single panel)
    # ════════════════════════════════════════════════════════════════════════
    ('obs_analysis_combined_count', {
        'subfolder':                '02_obs_count',
        'fig_label':                'per_year_pres',
        'from_config':              'ObservationalDataset',
        'data_path':                _OBS_ANALYSIS_DATA,
        'bin_years':                1,
        'which':                    'year',
        'figsize':                  _FIGSIZE_SINGLE_BIG,
        'axes_fontsize':            _FONT_AXES,
        'tick_fontsize':            _FONT_TICK,
        'legend_fontsize':          _FONT_LEGEND,
        'legend_observatory_names': True,
        'suppress_panel_titles':    True,
        'title':                    '',
    }),

    # ════════════════════════════════════════════════════════════════════════
    # 02b  Observation count per file (backup slide, landscape, single panel)
    # ════════════════════════════════════════════════════════════════════════
    ('obs_analysis_combined_count', {
        'subfolder':                '02_obs_count',
        'fig_label':                'per_file_pres',
        'from_config':              'ObservationalDataset',
        'data_path':                _OBS_ANALYSIS_DATA,
        'bin_years':                1,
        'which':                    'file',
        'figsize':                  _FIGSIZE_SINGLE_BIG,
        'axes_fontsize':            _FONT_AXES,
        'tick_fontsize':            _FONT_TICK,
        'legend_fontsize':          _FONT_LEGEND,
        'legend_observatory_names': True,
        'suppress_panel_titles':    True,
        'title':                    '',
    }),

    # ════════════════════════════════════════════════════════════════════════
    # 03a  RSW diff vs NEP097 — pure propagation, IAU only
    # ════════════════════════════════════════════════════════════════════════
    ('rsw_compare', {
        'subfolder':         '03_rsw_no_est',
        'fig_label':         'iau_only_pres',
        'sim_subset':        ['no_est_IAU'],
        'figsize':           _FIGSIZE_RSW_LANDSCAPE,
        'axes_fontsize':     _FONT_AXES,
        'tick_fontsize':     _FONT_TICK,
        'legend_fontsize':   _FONT_LEGEND,
        'title_fontsize':    _FONT_TITLE,
        'show_rms_in_legend': False,
        'title':             '',
    }),

    # ════════════════════════════════════════════════════════════════════════
    # 03b  RSW diff vs NEP097 — pure propagation, IAU vs Jacobson (no zoom)
    # ════════════════════════════════════════════════════════════════════════
    # IAU plotted first (underneath), Jacobson plotted on top.
    ('rsw_compare', {
        'subfolder':         '03_rsw_no_est',
        'fig_label':         'iau_vs_jac_pres',
        'sim_subset':        _NO_EST_PAIR,
        'figsize':           _FIGSIZE_RSW_LANDSCAPE,
        'axes_fontsize':     _FONT_AXES,
        'tick_fontsize':     _FONT_TICK,
        'legend_fontsize':   _FONT_LEGEND,
        'title_fontsize':    _FONT_TITLE,
        'show_rms_in_legend': False,
        'per_sim_alpha': {
            'no_est_IAU':      1.0,
            'no_est_Jacobson': 1.0,
        },
        'per_sim_linestyle': {
            'no_est_Jacobson': '-',
        },
        'title':             '',
    }),

    # ════════════════════════════════════════════════════════════════════════
    # 04  RSW diff vs NEP097 — state-only estimation, IAU vs Jacobson
    # ════════════════════════════════════════════════════════════════════════
    ('rsw_compare', {
        'subfolder':       '04_rsw_state',
        'fig_label':       'iau_vs_jac_pres',
        'sim_subset':      _STATE_PAIR,
        'figsize':         _FIGSIZE_RSW_LANDSCAPE,
        'axes_fontsize':   _FONT_AXES,
        'tick_fontsize':   _FONT_TICK,
        'legend_fontsize': _FONT_LEGEND,
        'title_fontsize':  _FONT_TITLE,
        'title':           '',
    }),

    # ════════════════════════════════════════════════════════════════════════
    # 05  Total RMS vs NEP097 — 5 IAU pole-estimation variants
    # ════════════════════════════════════════════════════════════════════════
    ('rms_compare', {
        'subfolder':       '05_rms_compare',
        'fig_label':       'pole_iau_all_pres',
        'sim_subset':      _POLE_IAU_5,
        'figsize':         _FIGSIZE_SINGLE_MEDIUM,
        'axes_fontsize':   _FONT_AXES,
        'tick_fontsize':   _FONT_TICK,
        'annotate':        True,
        'annot_fontsize':  _FONT_TICK,
        'annot_format':    '{:.3f}',
        'show_suptitle':   False,
        'uniform_color':   '#0072B2',
        'ymax':            425,
        'title':           '',
    }),

    # ════════════════════════════════════════════════════════════════════════
    # 06  RSW diff (G1: per file vs scaled per file) — landscape 3×1
    # ════════════════════════════════════════════════════════════════════════
    ('rsw_compare', {
        'subfolder':       '06_rsw_g1',
        'fig_label':       'g1_pres',
        'from_config':     'WeightAnalysis_Pole',
        'sim_subset':      _GROUP1_SIMS,
        'figsize':         _FIGSIZE_RSW_LANDSCAPE,
        'axes_fontsize':   _FONT_AXES,
        'tick_fontsize':   _FONT_TICK,
        'legend_fontsize': _FONT_LEGEND,
        'title_fontsize':  _FONT_TITLE,
        'title':           '',
    }),

    # ════════════════════════════════════════════════════════════════════════
    # 07  Formal errors RSW (G1: per file vs scaled per file) — landscape 3×1
    # ════════════════════════════════════════════════════════════════════════
    ('formal_compare', {
        'subfolder':       '07_formal_g1',
        'fig_label':       'g1_pres',
        'from_config':     'WeightAnalysis_Pole',
        'sim_subset':      _GROUP1_SIMS,
        'figsize':         _FIGSIZE_RSW_LANDSCAPE,
        'axes_fontsize':   _FONT_AXES,
        'tick_fontsize':   _FONT_TICK,
        'legend_fontsize': _FONT_LEGEND,
        'title_fontsize':  _FONT_TITLE,
        'title':           '',
    }),

    # ════════════════════════════════════════════════════════════════════════
    # 08  Pole-estimation real-obs RSW diff: state vs state+lib. (Fit.) overlay
    #     — diff only (no formal errors), opacity to distinguish.
    # ════════════════════════════════════════════════════════════════════════
    # state+lib. (Fit.) drawn first (fully opaque, behind), state (IAU)
    # drawn second on top with reduced opacity so the orange shows through.
    ('rsw_and_formal_lines', {
        'subfolder':      '08_pole_real_overlay',
        'fig_label':      'state_vs_lib_diff_only',
        'from_config':    'CASE1_Manual_Bias',
        'sim_subset':     ['SimPole_pole_lib_cov', 'IAUPole_initial_state'],
        'show_formal':    False,
        'show_zoom':      False,
        'layout':         'vertical',
        'per_sim_alpha': {
            'SimPole_pole_lib_cov':  1.0,
            'IAUPole_initial_state': 0.45,
        },
        # Same full-textwidth ratio as fig #09 (h/w = 0.5 → ~6.06 × 3.03 in
        # printed at width=\textwidth).
        'figsize':         (12.0, 6.0),
        'axes_fontsize':   _FONT_AXES,
        'tick_fontsize':   _FONT_TICK,
        'legend_fontsize': _FONT_LEGEND,
        'title_fontsize':  _FONT_TITLE,
        'legend_outside':  True,
        'title':           '',
    }),

    # ════════════════════════════════════════════════════════════════════════
    # 09  rsw_and_formal_lines for state+lib. (Fit.) — no zoom, side-by-side
    # ════════════════════════════════════════════════════════════════════════
    ('rsw_and_formal_lines', {
        'subfolder':       '09_pole_real_rsw_formal',
        'fig_label':       'rsw_formal_lines_no_zoom',
        'from_config':     'CASE1_Manual_Bias',
        'sim_subset':      ['SimPole_pole_lib_cov'],
        'show_formal':     True,
        'show_zoom':       False,
        'layout':          'vertical',
        # Full-textwidth slide placement.  Bumping h/w to ~0.5 so the panels
        # are taller — at width=\textwidth (~6.06 in) this prints ~3.03 in
        # tall, near the slide's vertical limit.
        'figsize':         (12.0, 6.0),
        'axes_fontsize':   _FONT_AXES,
        'tick_fontsize':   _FONT_TICK,
        'legend_fontsize': _FONT_LEGEND,
        'title_fontsize':  _FONT_TITLE,
        # Render formal 1σ as a transparent band around 0 — no outline (the
        # earlier 1.2-pt edge made the panels look bordered).
        'formal_as_band':     True,
        'formal_band_alpha':  0.6,
        'formal_band_edge':   False,
        # Diff line at matched opacity so the line/band overlap darkens
        # naturally via alpha compositing.
        'per_sim_alpha': {
            'SimPole_pole_lib_cov': 0.6,
        },
        'show_rms_in_legend': False,
        'diff_label_suffix':  ' diff with NEP097',
        'formal_line_colors': {
            'SimPole_pole_lib_cov': '#882255',  # wine — matches diff color
        },
        'legend_outside':  True,
        # More numbered ticks — locator forces round multiples (e.g. 25 km
        # steps) so the panel edges land on whole, labelled values like
        # 200 / 550 / 1000.
        'y_max_nticks':       10,
        'y_integer_ticks':    True,
        'title':           '',
    }),

    # ════════════════════════════════════════════════════════════════════════
    # 09b  Same as #09 but with the diff drawn as an envelope and the
    #      diff/formal-band intersection painted in a distinct (darker) color
    #      so the overlap region reads unambiguously.
    # ════════════════════════════════════════════════════════════════════════
    ('rsw_and_formal_lines', {
        'subfolder':       '09_pole_real_rsw_formal',
        'fig_label':       'rsw_formal_lines_no_zoom_intersection',
        'from_config':     'CASE1_Manual_Bias',
        'sim_subset':      ['SimPole_pole_lib_cov'],
        'show_formal':     True,
        'show_zoom':       False,
        'layout':          'vertical',
        'figsize':         (12.0, 6.0),
        'axes_fontsize':   _FONT_AXES,
        'tick_fontsize':   _FONT_TICK,
        'legend_fontsize': _FONT_LEGEND,
        'title_fontsize':  _FONT_TITLE,
        'formal_as_band':     True,
        'formal_band_alpha':  0.6,
        'formal_band_edge':   False,
        # Diff as envelope (rolling min/max) so the intersection of the two
        # bands is well-defined geometrically rather than per-pixel-luck.
        'diff_as_envelope':     True,
        'envelope_window':      365,   # ~1-year rolling window
        'diff_envelope_alpha':  0.55,
        'show_rms_in_legend':   False,
        'diff_label_suffix':    ' diff with NEP097',
        'formal_line_colors': {
            'SimPole_pole_lib_cov': '#882255',  # wine
        },
        # Intersection — deep wine, almost opaque, drawn on top of both bands.
        'intersection_color':   '#3d0f1e',
        'intersection_alpha':   0.85,
        'legend_outside':  True,
        'y_max_nticks':       10,
        'y_integer_ticks':    True,
        'title':           '',
    }),

    # ════════════════════════════════════════════════════════════════════════
    # 10  Weight analysis — 2-row R/S/W grid: RMS diff vs NEP097 (top)
    #     and diff/formal-σ ratio (bottom) for the 3 main schemes.
    # ════════════════════════════════════════════════════════════════════════
    ('rsw_rms_ratio_grid', {
        'subfolder':       '10_weight_rsw_diff_ratio',
        'fig_label':       'three_schemes_pres',
        'from_config':     'WeightAnalysis_Pole',
        # Dropped tf_weights — confusing in this comparison.
        'sim_subset':      ['id_weights', 'id_new_2_weights'],
        'rows':            ('diff', 'ratio'),
        # Taller canvas; bar chart needs more vertical room.
        'figsize':         (13.0, 8.0),
        'fontsize_scale':  0.85,
        'ymargin_top':     0.45,
        'as_bars':         True,
        'ymin_zero':       True,
        'row_label_overrides': {
            'ratio': 'True / Formal error ratio [—]',
        },
        # Top-row R / S clip the bar-value annotations — bump just those.
        'ylim_overrides': {
            (0, 0): 230,
            (0, 1): 520,
        },
        # Make the ratio = 1 reference more visible.
        'ratio_ref_linewidth': 1.6,
        'title':           '',
    }),

    # ════════════════════════════════════════════════════════════════════════
    # 10alt  Same as #10 but with each row's ylim shared across R/S/W
    #        (visually aligns bar heights row-by-row).
    # ════════════════════════════════════════════════════════════════════════
    ('rsw_rms_ratio_grid', {
        'subfolder':       '10_weight_rsw_diff_ratio',
        'fig_label':       'three_schemes_pres_shared_row',
        'from_config':     'WeightAnalysis_Pole',
        'sim_subset':      ['id_weights', 'id_new_2_weights'],
        'rows':            ('diff', 'ratio'),
        'figsize':         (13.0, 8.0),
        'fontsize_scale':  0.85,
        'ymargin_top':     0.45,
        'as_bars':         True,
        'ymin_zero':       True,
        'row_label_overrides': {
            'ratio': 'True / Formal error ratio [—]',
        },
        'ylim_overrides': {
            (0, 0): 230,
            (0, 1): 520,
        },
        'share_row_ylims':     True,
        'ratio_ref_linewidth': 1.6,
        'title':               '',
    }),

    # ════════════════════════════════════════════════════════════════════════
    # 10b  Real-data analogue — 2-row diff/ratio grid for state (IAU) vs
    #      state+lib. (Fit.).
    # ════════════════════════════════════════════════════════════════════════
    ('rsw_rms_ratio_grid', {
        'subfolder':       '10_real_data_rsw_diff_ratio',
        'fig_label':       'state_vs_lib_pres',
        'from_config':     'CASE1_Manual_Bias',
        'sim_subset':      ['IAUPole_initial_state', 'SimPole_pole_lib_cov'],
        'rows':            ('diff', 'ratio'),
        'figsize':         (10.0, 6.0),
        'fontsize_scale':  0.9,
        'ymargin_top':     0.40,
        'as_bars':         True,
        'ymin_zero':       True,
        'row_label_overrides': {
            'ratio': 'True / Formal error ratio [—]',
        },
        # Annotation clipping — bump the panels that need it.
        # rows=(diff, ratio); cols=(R, S, W).
        'ylim_overrides': {
            (0, 0): 200,   # diff R
            (0, 1): 550,   # diff S
            (1, 0): 6,     # ratio R
        },
        'ratio_ref_linewidth': 1.6,
        'title':           '',
    }),

    # ════════════════════════════════════════════════════════════════════════
    # 10b alt  Same as #10b but with each row sharing a common ylim across R/S/W.
    # ════════════════════════════════════════════════════════════════════════
    ('rsw_rms_ratio_grid', {
        'subfolder':       '10_real_data_rsw_diff_ratio',
        'fig_label':       'state_vs_lib_pres_shared_row',
        'from_config':     'CASE1_Manual_Bias',
        'sim_subset':      ['IAUPole_initial_state', 'SimPole_pole_lib_cov'],
        'rows':            ('diff', 'ratio'),
        'figsize':         (10.0, 6.0),
        'fontsize_scale':  0.9,
        'ymargin_top':     0.40,
        'as_bars':         True,
        'ymin_zero':       True,
        'row_label_overrides': {
            'ratio': 'True / Formal error ratio [—]',
        },
        'ylim_overrides': {
            (0, 0): 200,
            (0, 1): 550,
            (1, 0): 6,
        },
        'share_row_ylims':     True,
        'ratio_ref_linewidth': 1.6,
        'title':               '',
    }),

    # ════════════════════════════════════════════════════════════════════════
    # 11  Pole-movement comparison
    # ════════════════════════════════════════════════════════════════════════
    # Four pole models drawn on a shared time axis (taken from a SimObs sim,
    # which spans 1963–2025).
    #
    #   IAU 2015      — analytic, hard-coded constants from PropFuncs.py
    #   Jacobson 2009 — analytic, hard-coded constants from PropFuncs.py
    #   Sim. Fit. Pole  — IAU 2015 base + parameter deltas from
    #                     state+pos.+lib. estimation against simulated
    #                     NEP097 observations  (pole_pos_lib_IAU, SimObs)
    #   Real. Fit. Pole — Sim. Fit. Pole base + parameter deltas from
    #                     state+lib. (Fit.) estimation against the real
    #                     observations  (SimPole_pole_lib_cov, CASE1)
    # Build entries up via successive overlays so the same plot is shown four
    # times with one extra curve added each step.  All four share the same
    # ref_entries (the full 4-curve set) so the y-limits are locked to the
    # final figure — gives the visual feel of curves being "added in" without
    # the axes rescaling between slides.
]

_POLE_FULL_ENTRIES = [
    {'kind': 'iau', 'label': 'IAU 2015',
     'color': '#0072B2', 'linestyle': '--',
     'linewidth': 1.6, 'alpha': 0.95},
    {'kind': 'jacobson_2009', 'label': 'Jacobson 2009',
     'color': '#009E73', 'linestyle': '-',
     'linewidth': 1.6, 'alpha': 0.95},
    {'kind': 'iau', 'label': 'Sim. Fit. Pole',
     'color': '#D55E00', 'linestyle': '--',
     'linewidth': 1.6, 'alpha': 0.95,
     'absolute_from': [
         # cache_path short-circuits the SimObs dataset load when the cache
         # exists (run build_pole_base_cache.py once).
         {'sim': 'pole_pos_lib_IAU',
          'from_config': 'SimObs_ParameterAnalysis',
          'cache_path': _SIM_FIT_POLE_CACHE},
     ]},
    {'kind': 'iau', 'label': 'Real. Fit. Pole',
     'color': '#882255', 'linestyle': '-',
     'linewidth': 2.0, 'alpha': 1.0,
     'absolute_from': [
         {'sim': 'pole_pos_lib_IAU',
          'from_config': 'SimObs_ParameterAnalysis',
          'cache_path': _SIM_FIT_POLE_CACHE},
         {'sim': 'SimPole_pole_lib_cov',
          'from_config': 'CASE1_Manual_Bias'},
     ]},
]

# Variant entry list with a Gaussian (linear) ±1σ band on Real. Fit. Pole.
# Kept separate from _POLE_FULL_ENTRIES so the progressive 01→02→03→04 sequence
# stays band-free; only figure 05 draws the band.
_REAL_FIT_POLE_GAUSS = dict(_POLE_FULL_ENTRIES[-1])
_REAL_FIT_POLE_GAUSS['uncertainty'] = {
    'method':       'gaussian',
    'sigma_source': {'sim': 'SimPole_pole_lib_cov',
                     'from_config': 'CASE1_Manual_Bias'},
    'n_sigma':      1.0,
    'band_alpha':   0.30,
    'band_color':   '#882255',
    'label':        r'Real. Fit. Pole  $\pm 1\sigma$ (Gauss.)',
}
_POLE_FULL_ENTRIES_GAUSS = _POLE_FULL_ENTRIES[:-1] + [_REAL_FIT_POLE_GAUSS]

# Shared ylims across all figure-11 sub-figures so the progressive 01→06
# sequence has consistent axes (auto-scaling only considers central curves
# and would clip the ±1σ bands of the Gaussian variants).
_POLE_YLIM_ALPHA = (296.8, 301.7)
_POLE_YLIM_DELTA = (42.7, 44.5)

FIGURES_TO_EXPORT += [
    ('pole_model_compare', {
        'subfolder':       '11_pole_model_compare',
        'fig_label':       _label,
        'figsize':         (10.0, 5.5),
        'axes_fontsize':   _FONT_AXES,
        'tick_fontsize':   _FONT_TICK,
        'legend_fontsize': _FONT_LEGEND,
        'title_fontsize':  _FONT_TITLE,
        'show_suptitle':   False,
        'title':           '',
        'time_source': {
            'sim':         'SimPole_pole_lib_cov',
            'from_config': 'CASE1_Manual_Bias',
        },
        'entries':         _POLE_FULL_ENTRIES[:_n],
        'ref_entries':     _POLE_FULL_ENTRIES,
        'ylim_alpha':      _POLE_YLIM_ALPHA,
        'ylim_delta':      _POLE_YLIM_DELTA,
    })
    for _label, _n in [
        ('01_iau_only',        1),
        ('02_iau_jac',         2),
        ('03_iau_jac_simfit',  3),
        ('04_four_models',     4),  # plain four lines, no band
    ]
]

# ── 11 (cont.) 05  Four models + Gaussian-propagation 1σ band on Real. Fit. Pole
# Same 4-curve figure as 04_four_models but with a translucent ±1σ band drawn
# around the Real. Fit. Pole curve.  Sigma is propagated linearly through the
# IAU model from σ_α₁ / σ_δ₁ (last two entries of formal_errors on
# SimPole_pole_lib_cov in PoleEst_MB).
FIGURES_TO_EXPORT.append((
    'pole_model_compare', {
        'subfolder':       '11_pole_model_compare',
        'fig_label':       '05_four_models_band_gauss',
        'figsize':         (10.0, 5.5),
        'axes_fontsize':   _FONT_AXES,
        'tick_fontsize':   _FONT_TICK,
        'legend_fontsize': _FONT_LEGEND,
        'title_fontsize':  _FONT_TITLE,
        'show_suptitle':   False,
        'title':           '',
        'time_source': {
            'sim':         'SimPole_pole_lib_cov',
            'from_config': 'CASE1_Manual_Bias',
        },
        'entries':         _POLE_FULL_ENTRIES_GAUSS,
        'ref_entries':     _POLE_FULL_ENTRIES_GAUSS,
        'ylim_alpha':      _POLE_YLIM_ALPHA,
        'ylim_delta':      _POLE_YLIM_DELTA,
    },
))

# ── 11 (cont.) 06  Real. Fit. Pole built directly on the IAU 2015 baseline
# (no Sim. Fit. intermediate step) with its own Gaussian ±1σ band, shown
# alongside the original Real. Fit. Pole (Sim. Fit. base) for comparison.
# Sigma source is the same SimPole_pole_lib_cov estimation in both cases —
# only the constant baseline differs.
_REAL_FIT_POLE_IAU_BASE = {
    'kind': 'iau', 'label': 'Real. Fit. Pole (IAU base)',
    'color': '#117733', 'linestyle': '-',
    'linewidth': 2.0, 'alpha': 1.0,
    'absolute_from': [
        {'sim': 'SimPole_pole_lib_cov',
         'from_config': 'CASE1_Manual_Bias'},
    ],
}
_REAL_FIT_POLE_IAU_BASE_GAUSS = dict(_REAL_FIT_POLE_IAU_BASE)
_REAL_FIT_POLE_IAU_BASE_GAUSS['uncertainty'] = {
    'method':       'gaussian',
    'sigma_source': {'sim': 'SimPole_pole_lib_cov',
                     'from_config': 'CASE1_Manual_Bias'},
    'n_sigma':      1.0,
    'band_alpha':   0.30,
    'band_color':   '#117733',
    'label':        r'Real. Fit. Pole (IAU base)  $\pm 1\sigma$ (Gauss.)',
}

_POLE_IAU_BASE_COMPARE = [
    _POLE_FULL_ENTRIES[0],          # IAU 2015 reference
    _POLE_FULL_ENTRIES[2],          # Sim. Fit. Pole
    _POLE_FULL_ENTRIES_GAUSS[-1],   # Real. Fit. Pole (Sim. Fit. base) + band
    _REAL_FIT_POLE_IAU_BASE_GAUSS,  # Real. Fit. Pole (IAU base) + band
]

FIGURES_TO_EXPORT.append((
    'pole_model_compare', {
        'subfolder':       '11_pole_model_compare',
        'fig_label':       '06_state_lib_iau_base_band',
        'figsize':         (10.0, 5.5),
        'axes_fontsize':   _FONT_AXES,
        'tick_fontsize':   _FONT_TICK,
        'legend_fontsize': _FONT_LEGEND,
        'title_fontsize':  _FONT_TITLE,
        'show_suptitle':   False,
        'title':           '',
        'time_source': {
            'sim':         'SimPole_pole_lib_cov',
            'from_config': 'CASE1_Manual_Bias',
        },
        'entries':         _POLE_IAU_BASE_COMPARE,
        'ref_entries':     _POLE_IAU_BASE_COMPARE,
        'ylim_alpha':      _POLE_YLIM_ALPHA,
        'ylim_delta':      _POLE_YLIM_DELTA,
    },
))

# ── 12  Gaussian-vs-Monte-Carlo validation figure ──────────────────────────────
# Side-by-side comparison: same Real. Fit. Pole curve, both 1σ bands, and
# explicit σ(t) profiles so the agreement (≈identical for a linear model with
# independent Gaussian inputs) is directly visible.
FIGURES_TO_EXPORT.append((
    'pole_uncertainty_validation', {
        'subfolder':       '12_pole_uncertainty_validation',
        'fig_label':       'gauss_vs_mc',
        'figsize':         (12.0, 7.5),
        'axes_fontsize':   _FONT_AXES,
        'tick_fontsize':   _FONT_TICK,
        'legend_fontsize': _FONT_LEGEND,
        'title_fontsize':  _FONT_TITLE,
        'show_suptitle':   False,
        'n_samples':       10000,
        'rng_seed':        42,
        'n_sigma':         1.0,
        'time_source':     {'sim': 'SimPole_pole_lib_cov',
                            'from_config': 'CASE1_Manual_Bias'},
        'sigma_source':    {'sim': 'SimPole_pole_lib_cov',
                            'from_config': 'CASE1_Manual_Bias'},
        # Same chained 'absolute_from' as the Real. Fit. Pole entry so the
        # central curve in the validation matches the one in figure #11.
        'entry': {
            'kind':  'iau',
            'label': 'Real. Fit. Pole',
            'color': '#882255',
            'absolute_from': [
                {'sim': 'pole_pos_lib_IAU',
                 'from_config': 'SimObs_ParameterAnalysis',
                 'cache_path': _SIM_FIT_POLE_CACHE},
                {'sim': 'SimPole_pole_lib_cov',
                 'from_config': 'CASE1_Manual_Bias'},
            ],
        },
        'title':           '',
    },
))

SINGLE_SIM_TABLES = []

# Reuse marker / color / linestyle dicts from the primary (SimObs) config.
# When `from_config` is used, the foreign config's own SIM_COLORS etc. are
# loaded via _load_dataset_for_config — so the per-sim styling for figures
# 1, 2, 6, 7 follows their source-config palette automatically.
SIM_COLORS    = _simobs.SIM_COLORS
SIM_MARKERS   = _simobs.SIM_MARKERS
SIM_LINESTYLE = _simobs.SIM_LINESTYLE
