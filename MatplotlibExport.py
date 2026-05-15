"""MatplotlibExport.py — Publication-quality PDF figure export for thesis.

Reads the same .pkl files as DashInteractivePlotFull.py (by importing its
data-loading logic) and saves selected plots as a multi-page PDF.

QUICK-START
-----------
1. Set DATASET_LABEL to the dataset key you want (or leave None for first).
2. Set SELECTED_SIMS and SIM_LABELS (or leave None to use all).
3. Edit FIGURES_TO_EXPORT to pick which figures to include.
4. Run:  python MatplotlibExport.py
   → saves OUTPUT_PDF in the working directory.

AVAILABLE FIGURE TYPES
-----------------------
  'rms_compare'        — bar chart of final RMS vs SPICE per simulation
  'rms_compare_rsw'    — 3-panel dot plot of per-component RMS (R / S / W)
  'pole_model'         — IAU pole RA/Dec over time (fitted vs nominal)
  'pole_model_diff'    — IAU pole deviation from nominal (Δα, Δδ in mdeg)
  'rsw_compare'        — 3-row RSW difference vs SPICE time series
  'formal_compare'     — 3-row formal errors (σ_R/S/W) time series
  'rsw_stats'          — 3×3 grid: rows=R/S/W, cols=Mean/RMS/Max
  'gof'                — WRMS/RMS/cost comparison, optional initial overlay
  'corr_heatmap'       — |correlation| matrix heatmap (one sim at a time)
  'residual_histogram' — RA/Dec residual histograms with optional Gaussian fit
"""

import sys
import os

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  EXPORT MEMORY-SAVING TOGGLES                                             ║
# ║                                                                           ║
# ║  Uncomment one of the presets below to limit what gets loaded into RAM.  ║
# ║  Each preset sets env vars BEFORE the Dash module loads pickles, so the  ║
# ║  unwanted datasets / sims are never opened in the first place.           ║
# ║                                                                           ║
# ║  Leave EVERYTHING commented to load whatever DATA_FILES has uncommented  ║
# ║  in DashInteractivePlotFull.py (the legacy default).                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── Preset A: pole-movement figure only ──────────────────────────────────────
# Loads only PoleEst_MB → keeps only SimPole_pole_lib_cov → renders only #11.
# Requires one-time `python build_pole_base_cache.py` (replaces SimObs lookup).
os.environ['EXPORT_DATASETS']        = 'PoleEst_MB'
os.environ['EXPORT_SIMS_FILTER']     = 'SimPole_pole_lib_cov'
os.environ['EXPORT_FIGURES_FILTER']  = '11_pole_model_compare'

# ── Preset B: PoleEst_MB-block (figs #8, #9, #10b, #11, #12) ─────────────────
# os.environ['EXPORT_DATASETS']        = 'PoleEst_MB'
# os.environ['EXPORT_FIGURES_FILTER']  = '08_pole_real_overlay,09_pole_real_rsw_formal,10_real_data_rsw_diff_ratio,11_pole_model_compare,12_pole_uncertainty_validation'

# ── Preset C: WeightAnalysis-block (figs #6, #7, #10) ────────────────────────
# os.environ['EXPORT_DATASETS']        = 'WeightAnalysis'
# os.environ['EXPORT_FIGURES_FILTER']  = '06_rsw_g1,07_formal_g1,10_weight_rsw_diff_ratio'

# ── Preset D: SimObs-block (figs #1–#5) ──────────────────────────────────────
os.environ['EXPORT_DATASETS']        = 'SimObs'
os.environ['EXPORT_FIGURES_FILTER']  = '01_residuals_by_id,02_obs_count,03_rsw_no_est,04_rsw_state,05_rms_compare'

# ── Preset E: full export (no filtering) — needs ~30+ GB RAM ─────────────────
# del os.environ['EXPORT_DATASETS']
# del os.environ['EXPORT_SIMS_FILTER']
# del os.environ['EXPORT_FIGURES_FILTER']

# Echo whatever is in effect, for the run log.
for _k in ('EXPORT_DATASETS', 'EXPORT_SIMS_FILTER', 'EXPORT_FIGURES_FILTER'):
    print(f"  {_k}={os.environ.get(_k, '<unset>')}")
# ────────────────────────────────────────────────────────────────────────────

from datetime import datetime, timedelta
import matplotlib
matplotlib.use('PDF')           # must come before any pyplot import
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy import stats as sp_stats
import pandas as pd
from pathlib import Path

# ── Import data-processing functions from the Dash script ───────────────────
# This also triggers module-level data loading (DATA_FILES → all_datasets).
# app.run() is guarded by __name__ == '__main__', so no server is started.
from DashInteractivePlotFull import (
    all_datasets,
    get_active_data,
    convert_time_array_to_datetime,
    get_rsw_times,
    compute_rsw_statistics,
    compute_formal_error_statistics,
    compute_wrms_and_cost,
    get_parameter_labels,
    get_parameter_info,
)

# ============================================================================
# GLOBAL STYLE  (edit once — inherited by all figures)
# ============================================================================

plt.rcParams.update({
    'font.size':         14,
    'axes.titlesize':    12,
    'axes.labelsize':    14,
    'xtick.labelsize':   14,
    'ytick.labelsize':   14,
    'legend.fontsize':   14,
    'figure.dpi':        150,
    'savefig.dpi':       300,
    # Uncomment the two lines below if LaTeX is installed:
    # 'text.usetex':     True,
    # 'font.family':     'serif',
    'font.family':       'sans-serif',
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'lines.linewidth':   1.2,
    'lines.markersize':  4,
})

# Figure widths (inches).  3.5" = single column, 6.5" = double column.
FIG_W_SINGLE  = 3.5
FIG_W_DOUBLE  = 6.5
FIG_H_DEFAULT = 4.5

# Named RSW colors for consistency across plots.
RSW_COLORS = {'R': '#1f77b4', 'S': '#d62728', 'W': '#2ca02c'}

# Fall-back color cycle (matplotlib default).
_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Per-simulation colors and linestyles — loaded from the active config.
# Falls back to empty dicts so unknown sim names use the default color cycle.


# ============================================================================
# CONFIGURATION  — set ACTIVE_5CONFIG to the desired ExportConfigs/*.py name
# ============================================================================

#ACTIVE_CONFIG = 'CASE1_Manual_Bias'          # Pole estimation — real obs., manual bias correction
#ACTIVE_CONFIG = 'WeightAnalysis'            # Weight scheme comparison (initial_state only)
#ACTIVE_CONFIG = 'WeightAnalysis_Old'        # Weight scheme comparison — old dataset
#ACTIVE_CONFIG = 'WeightAnalysis_Pole'       # Weight scheme comparison (initial_state + pole)
#ACTIVE_CONFIG = 'WeightComparison'          # Cross-dataset comparison of all three weight analyses
#ACTIVE_CONFIG = 'SimObs_ParameterAnalysis'  # Simulated-obs. parameter analysis (IAU vs Jacobson)
#ACTIVE_CONFIG = 'ObservationalDataset'
ACTIVE_CONFIG = 'OverleafPresentation'      # Slide-tuned figures for 16:9 Beamer presentation

# Set True to skip rsw_with_zoom and formal_with_zoom figures (they are slow).
SKIP_TIMESERIES = False
_SLOW_FIGURE_TYPES = frozenset({
    'rsw_with_zoom', 'formal_with_zoom',
    'rsw_with_formal', 'rsw_with_formal_cloud', 'rsw_and_formal_lines',
    'rsw_initial_vs_final',
})

import importlib
_cfg = importlib.import_module(f'ExportConfigs.{ACTIVE_CONFIG}')

DATASET_LABEL      = _cfg.DATASET_LABEL
SELECTED_SIMS      = _cfg.SELECTED_SIMS
SIM_LABELS         = _cfg.SIM_LABELS
OUTPUT_DIR         = _cfg.OUTPUT_DIR
FIGURES_TO_EXPORT  = _cfg.FIGURES_TO_EXPORT
SINGLE_SIM_TABLES  = _cfg.SINGLE_SIM_TABLES
_SIM_COLORS        = getattr(_cfg, 'SIM_COLORS',       {})
_SIM_LINESTYLE     = getattr(_cfg, 'SIM_LINESTYLE',    {})
_SIM_MARKERS       = getattr(_cfg, 'SIM_MARKERS',      {})
# Optional per-sim opacity (0.0–1.0).  Useful when lines overlap: set lower alpha
# for sims drawn on top of others.  Falls back to 1.0 for unlisted sims.
# Example in config:  SIM_ALPHAS = {'tf_weights': 0.55, 'id_weights': 0.75}
_SIM_ALPHAS        = getattr(_cfg, 'SIM_ALPHAS',       {})
# Optional: maps sim-name prefix → group color for use_group_colors=True figures.
# e.g. {'IAUPole': '#0072B2', 'SimPole': '#D55E00'}
_SIM_GROUP_COLORS  = getattr(_cfg, 'SIM_GROUP_COLORS', {})
# Set SUPPRESS_TITLES = True in the config to strip figure suptitles on export.
_SUPPRESS_TITLES   = getattr(_cfg, 'SUPPRESS_TITLES',  False)
# EXPORT_GROUPS: list of group tags to export, or None to export all.
# Groups are assigned via the 'group' key in each FIGURES_TO_EXPORT entry.
# Defined groups in WeightComparison: 'rsw_diff', 'formal_rsw', 'weights', 'other'
# Example:  EXPORT_GROUPS = ['rsw_diff', 'weights']
_EXPORT_GROUPS     = getattr(_cfg, 'EXPORT_GROUPS',    None)


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _get_data():
    """Return (simulations_dict, sim_names_list) from the configured dataset."""
    label = DATASET_LABEL
    if label is None:
        keys = list(all_datasets.keys())
        if not keys:
            sys.exit("ERROR: No datasets loaded.  Check DATA_FILES in DashInteractivePlotFull.py.")
        label = keys[0]
    return get_active_data(label)


def _get_sims_and_labels(sims, all_names):
    """Return (selected_sim_names, display_labels) respecting configuration.

    Sims missing from the loaded data are dropped.  Labels stay aligned with
    names via index lookup into SIM_LABELS (using the original SELECTED_SIMS
    position), so missing sims do not shift label assignments.
    """
    selected = SELECTED_SIMS if SELECTED_SIMS is not None else list(all_names)
    # Build (name, label) pairs, skipping sims absent from the data.
    if SIM_LABELS is not None and len(SIM_LABELS) == len(selected):
        pairs = [(n, SIM_LABELS[i])
                 for i, n in enumerate(selected) if n in sims]
    else:
        pairs = [(n, n) for n in selected if n in sims]
    if not pairs:
        return [], []
    names, labels = zip(*pairs)
    return list(names), list(labels)


def _color(i: int) -> str:
    return _COLORS[i % len(_COLORS)]


def _marker(sn: str, fallback: str = 'o') -> str:
    return _SIM_MARKERS.get(sn, fallback)


def _group_color(sn: str, fallback_index: int = 0) -> str:
    """Return the group color for sim `sn` using _SIM_GROUP_COLORS lookup.

    Keys starting with '_' are treated as suffixes (sn.endswith(key));
    all other keys are treated as prefixes (sn.startswith(key)).
    Falls back to _SIM_COLORS then the default color cycle if no match is found.
    Used by figures with use_group_colors=True so that all sims in the same
    rotation-model group share one color while individual timeseries keep their
    own per-sim colors via _SIM_COLORS.
    """
    for key, col in _SIM_GROUP_COLORS.items():
        if key.startswith('_'):
            if sn.endswith(key):
                return col
        elif sn.startswith(key):
            return col
    return _SIM_COLORS.get(sn, _color(fallback_index))


# ============================================================================
# OBSERVATIONAL DATASET HELPERS
# ============================================================================

_TWO_PI        = 2.0 * np.pi
_RAD_TO_ARCSEC = 3600.0 * 180.0 / np.pi


def _get_observatory_name(obs_code: str) -> str:
    """Return the observatory name for a 3-digit MPC code from Observatories.txt.

    Pads single- or double-digit codes to 3 characters.  Returns the code
    itself if the file is not found or the code is not listed.
    """
    obs_code = obs_code.zfill(3)
    try:
        with open('Observations/Observatories.txt', 'r') as fh:
            for line in fh.readlines()[1:]:
                cols = line.split()
                if len(cols) >= 5 and cols[1] == obs_code:
                    # Format: Pl Code Lon Lat Alt rho_cos rho_sin region [name ...]
                    # Name starts at index 8 (index 7 = region string).
                    return ' '.join(cols[8:]) if len(cols) >= 9 else ' '.join(cols[7:])
    except FileNotFoundError:
        pass
    return obs_code


def _get_obs_type(nsdc_id: str,
                  raw_obs_folder=None,
                  obs_types_override: dict = None) -> str:
    """Return the observation type string ('Rel.' or 'Abs.') for an nsdc ID.

    Search order:
    1. *obs_types_override* dict (keyed by nsdc_id).
    2. Raw NSDC text files in each folder listed in *raw_obs_folder*.
       Accepts either a single path string or a list of path strings.
       Relative observations live in RawRelativeObservations/;
       absolute observations live in NeptuneObservations/.
    3. Falls back to '---' if no match is found.
    """
    if obs_types_override and nsdc_id in obs_types_override:
        return obs_types_override[nsdc_id]
    folders = []
    if raw_obs_folder:
        folders = [raw_obs_folder] if isinstance(raw_obs_folder, str) else list(raw_obs_folder)
    for folder in folders:
        raw_path = Path(folder) / f'{nsdc_id}.txt'
        if raw_path.exists():
            try:
                first_word = raw_path.read_text().split()[0].upper()
                if first_word == 'ABS':
                    return 'Abs.'
                if first_word in ('REL', 'SEP', 'DIF'):
                    return 'Rel.'
            except Exception:
                pass
    return '---'


def _load_spice_residuals_df(obs_folder: str) -> pd.DataFrame:
    """Load O-C residuals (vs NEP097) from all Triton_*.csv files in obs_folder.

    Returns a DataFrame with columns:
        ref_point_id     : str   — e.g. '689_nm0077'
        time_j2000       : float — seconds since J2000
        ra_resid_arcsec  : float — RA residual [arcsec]
        dec_resid_arcsec : float — Dec residual [arcsec]

    Convention in the CSV files:
        column 3 (O-C RA)  is stored as 2*pi + residual_rad  → subtract 2*pi
        column 4 (O-C Dec) is stored directly as residual_rad
    """
    rows = []
    for csv_path in sorted(Path(obs_folder).glob('Triton_*.csv')):
        parts = csv_path.stem.split('_')   # ['Triton', '<code>', '<nmXXXX>']
        if len(parts) < 3:
            continue
        ref_id = f'{parts[1]}_{parts[2]}'
        try:
            df_raw = pd.read_csv(csv_path)
        except Exception as exc:
            print(f'  WARNING: could not read {csv_path.name}: {exc}')
            continue
        times   = df_raw.iloc[:, 0].values
        ra_res  = (df_raw.iloc[:, 3].values - _TWO_PI) * _RAD_TO_ARCSEC
        dec_res =  df_raw.iloc[:, 4].values            * _RAD_TO_ARCSEC
        rows.append(pd.DataFrame({
            'ref_point_id':    ref_id,
            'time_j2000':      times,
            'ra_resid_arcsec': ra_res,
            'dec_resid_arcsec': dec_res,
        }))
    if not rows:
        return pd.DataFrame(columns=['ref_point_id', 'time_j2000',
                                     'ra_resid_arcsec', 'dec_resid_arcsec'])
    return pd.concat(rows, ignore_index=True)


def _apply_date_formatter(ax):
    """Tidy date formatter: major ticks every 5 years, rotated labels."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')


# ============================================================================
# PLOT FUNCTIONS
# ============================================================================

def mpl_rms_compare(sims, names, labels,
                    sim_subset=None,
                    use_group_colors=False,
                    title='RMS vs SPICE Comparison',
                    show_suptitle=True,
                    tick_fontsize=None,
                    axes_fontsize=None,
                    annot_fontsize=None,
                    annot_format='{:.0f}',
                    figsize=None,
                    annotate=False,
                    uniform_color=None,
                    ymax=None,
                    ymin=None):
    """Dot plot of final total RMS per simulation.

    use_group_colors is accepted for API consistency; colors are taken from
    _SIM_COLORS which already encodes group membership when set in the config.
    """
    if sim_subset is not None:
        pairs  = [(n, labels[names.index(n)]) for n in sim_subset if n in sims and n in names]
        names  = [p[0] for p in pairs]
        labels = [p[1] for p in pairs]
    vals = [sims[n].get('rms_SPICE', np.nan) for n in names]
    x = np.arange(len(names))

    if figsize is None:
        figsize = (max(FIG_W_DOUBLE, 0.8 * len(names)), FIG_H_DEFAULT)
    fig, ax = plt.subplots(figsize=figsize)

    # Connecting line through all non-NaN points
    valid_x = [xi for xi, v in enumerate(vals) if not np.isnan(v)]
    valid_v = [v  for v  in vals               if not np.isnan(v)]
    if len(valid_x) > 1:
        ax.plot(valid_x, valid_v, '-', color='gray', linewidth=1.0,
                alpha=0.5, zorder=1)

    for i, (xi, v) in enumerate(zip(x, vals)):
        if not np.isnan(v):
            if uniform_color is not None:
                col = uniform_color
            else:
                col = (_group_color(names[i], i) if use_group_colors
                       else _SIM_COLORS.get(names[i], _color(i)))
            mk  = _marker(names[i])
            ax.plot(xi, v, marker=mk, color=col, markersize=8,
                    markeredgecolor='black', markeredgewidth=0.5,
                    linestyle='none', zorder=2)
            if annotate:
                ax.annotate(annot_format.format(v),
                            xy=(xi, v), xytext=(0, 8),
                            textcoords='offset points',
                            ha='center', va='bottom',
                            fontsize=(annot_fontsize if annot_fontsize is not None else 9))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right',
                       **(({'fontsize': tick_fontsize}) if tick_fontsize is not None else {}))
    if tick_fontsize is not None:
        ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.set_ylabel('RMS [km]',
                  **(({'fontsize': axes_fontsize}) if axes_fontsize is not None else {}))
    if show_suptitle:
        ax.set_title(title)
    ax.set_xlim(-0.6, len(names) - 0.4)
    if ymax is not None or ymin is not None:
        cur_lo, cur_hi = ax.get_ylim()
        ax.set_ylim(ymin if ymin is not None else cur_lo,
                    ymax if ymax is not None else cur_hi)
    fig.tight_layout()
    return fig


def mpl_rms_compare_rsw(sims, names, labels,
                        sim_subset=None,
                        use_group_colors=False,
                        title='RMS vs NEP097 — R / S / W Components',
                        show_suptitle=True,
                        tick_fontsize=None):
    """Three-panel dot plot of per-component RMS (R, S, W) across simulations.

    Same dot-plot style as mpl_rms_compare but split into three stacked subplots,
    one per RSW direction.  Per-component RMS is computed as
    sqrt(mean(diff_SPICE_RSW[:, i]**2)) for i in {0, 1, 2}.
    """
    if sim_subset is not None:
        pairs  = [(n, labels[names.index(n)]) for n in sim_subset if n in sims and n in names]
        names  = [p[0] for p in pairs]
        labels = [p[1] for p in pairs]

    comps    = ['R', 'S', 'W']
    ylabels  = [r'RMS $\Delta R$ [km]', r'RMS $\Delta S$ [km]', r'RMS $\Delta W$ [km]']
    x        = np.arange(len(names))
    fig_w    = max(FIG_W_DOUBLE, 0.8 * len(names))

    fig, axes = plt.subplots(3, 1, figsize=(fig_w, FIG_H_DEFAULT * 2.2), sharex=True)

    for ci, (ax, comp, ylab) in enumerate(zip(axes, comps, ylabels)):
        col_line = RSW_COLORS[comp]

        vals = []
        for sn in names:
            sd = sims.get(sn, {})
            if 'diff_SPICE_RSW' in sd:
                vals.append(float(np.sqrt(np.mean(sd['diff_SPICE_RSW'][:, ci] ** 2))))
            else:
                vals.append(np.nan)

        valid_x = [xi for xi, v in enumerate(vals) if not np.isnan(v)]
        valid_v = [v  for v  in vals               if not np.isnan(v)]
        if len(valid_x) > 1:
            ax.plot(valid_x, valid_v, '-', color='gray', linewidth=1.0,
                    alpha=0.5, zorder=1)

        for i, (xi, v) in enumerate(zip(x, vals)):
            if not np.isnan(v):
                col = (_group_color(names[i], i) if use_group_colors
                       else _SIM_COLORS.get(names[i], _color(i)))
                mk  = _marker(names[i])
                ax.plot(xi, v, marker=mk, color=col, markersize=8,
                        markeredgecolor='black', markeredgewidth=0.5,
                        linestyle='none', zorder=2)


        ax.set_ylabel(ylab)
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.set_title(comp, loc='right', fontsize=10, color=col_line)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=30, ha='right',
                             **(({'fontsize': tick_fontsize}) if tick_fontsize is not None else {}))
    if show_suptitle:
        fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


def mpl_formal_rms_rsw(sims, names, labels,
                       sim_subset=None,
                       use_group_colors=False,
                       title='Formal Error RMS — R / S / W Components'):
    """Three-panel dot plot of per-component formal error RMS across simulations.

    Mirror of mpl_rms_compare_rsw but for formal_errors_RSW_km instead of
    diff_SPICE_RSW.  One stacked subplot per RSW direction.
    """
    if sim_subset is not None:
        pairs  = [(n, labels[names.index(n)]) for n in sim_subset if n in sims and n in names]
        names  = [p[0] for p in pairs]
        labels = [p[1] for p in pairs]

    comps   = ['R', 'S', 'W']
    ylabels = [r'RMS $\sigma_R$ [km]', r'RMS $\sigma_S$ [km]', r'RMS $\sigma_W$ [km]']
    x       = np.arange(len(names))
    fig_w   = max(FIG_W_DOUBLE, 0.8 * len(names))

    fig, axes = plt.subplots(3, 1, figsize=(fig_w, FIG_H_DEFAULT * 2.2), sharex=True)

    for ci, (ax, comp, ylab) in enumerate(zip(axes, comps, ylabels)):
        col_line = RSW_COLORS[comp]

        vals = []
        for sn in names:
            sd = sims.get(sn, {})
            if 'formal_errors_RSW_km' in sd:
                vals.append(float(np.sqrt(np.mean(sd['formal_errors_RSW_km'][:, ci] ** 2))))
            else:
                vals.append(np.nan)

        valid_x = [xi for xi, v in enumerate(vals) if not np.isnan(v)]
        valid_v = [v  for v  in vals               if not np.isnan(v)]
        if len(valid_x) > 1:
            ax.plot(valid_x, valid_v, '-', color='gray', linewidth=1.0,
                    alpha=0.5, zorder=1)

        for i, (xi, v) in enumerate(zip(x, vals)):
            if not np.isnan(v):
                col = (_group_color(names[i], i) if use_group_colors
                       else _SIM_COLORS.get(names[i], _color(i)))
                mk  = _marker(names[i])
                ax.plot(xi, v, marker=mk, color=col, markersize=8,
                        markeredgecolor='black', markeredgewidth=0.5,
                        linestyle='none', zorder=2)


        ax.set_ylabel(ylab)
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.set_title(comp, loc='right', fontsize=10, color=col_line)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=30, ha='right')
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


def mpl_rsw_rms_ratio_grid(sims, names, labels,
                            sim_subset=None,
                            use_group_colors=False,
                            fontsize_scale=1.0,
                            title='RSW RMS Summary',
                            figsize=None,
                            rows=('diff', 'formal', 'ratio'),
                            per_sim_alpha=None,
                            row_label_overrides=None,
                            ymargin_top=0.18,
                            as_bars=False,
                            ymin_zero=False,
                            ylim_overrides=None,
                            share_row_ylims=False,
                            ratio_ref_linewidth=0.8):
    """3 × 3 grid: rows = RMS diff / Formal σ RMS / Ratio, cols = R / S / W.

    Row 0 — RMS of diff_SPICE_RSW per component [km]
    Row 1 — RMS of formal_errors_RSW_km per component [km]
    Row 2 — ratio: RMS diff / Formal σ RMS  [dimensionless]

    Columns share the same x-axis (simulations).  A dashed line at 1.0 marks
    the ratio = 1 reference in row 2.
    """
    if sim_subset is not None:
        pairs  = [(n, labels[names.index(n)]) for n in sim_subset if n in sims and n in names]
        names  = [p[0] for p in pairs]
        labels = [p[1] for p in pairs]

    comps         = ['R', 'S', 'W']
    _all_rows     = [
        ('RMS diff [km]',     'diff'),
        ('Formal σ RMS [km]', 'formal'),
        ('Ratio [—]',         'ratio'),
    ]
    rows_set      = set(rows)
    row_defs      = [r for r in _all_rows if r[1] in rows_set]
    if row_label_overrides:
        row_defs = [(row_label_overrides.get(src, lbl), src)
                    for (lbl, src) in row_defs]
    if not row_defs:
        return None
    per_sim_alpha = per_sim_alpha or {}
    n_rows        = len(row_defs)
    last_row_idx  = n_rows - 1

    x = np.arange(len(names))

    fs_tick   = max(16, int(18 * fontsize_scale))
    fs_annot  = max(11, int(12 * fontsize_scale))
    fs_ylabel = max(14, int(15 * fontsize_scale))
    fs_title  = max(15, int(16 * fontsize_scale))

    if figsize is None:
        fig_w = max(FIG_W_DOUBLE * 1.8, 0.8 * len(names) * 3)
        figsize = (fig_w, FIG_H_DEFAULT * (n_rows / 3.0) * 2.5)

    fig, axes = plt.subplots(n_rows, 3, figsize=figsize, squeeze=False)

    for ri, (row_label, src) in enumerate(row_defs):
        for ci, comp in enumerate(comps):
            ax  = axes[ri, ci]
            col = RSW_COLORS[comp]

            vals = []
            for sn in names:
                sd = sims.get(sn, {})
                diff_rms   = np.nan
                formal_rms = np.nan
                if 'diff_SPICE_RSW' in sd:
                    diff_rms = float(np.sqrt(np.mean(sd['diff_SPICE_RSW'][:, ci] ** 2)))
                if 'formal_errors_RSW_km' in sd:
                    formal_rms = float(np.sqrt(np.mean(sd['formal_errors_RSW_km'][:, ci] ** 2)))
                if src == 'diff':
                    vals.append(diff_rms)
                elif src == 'formal':
                    vals.append(formal_rms)
                else:  # ratio
                    if not (np.isnan(diff_rms) or np.isnan(formal_rms) or formal_rms == 0):
                        vals.append(diff_rms / formal_rms)
                    else:
                        vals.append(np.nan)

            # Connecting line (dots/line mode only — not for bars)
            valid_x = [xi for xi, v in enumerate(vals) if not np.isnan(v)]
            valid_v = [v  for v  in vals               if not np.isnan(v)]
            if not as_bars and len(valid_x) > 1:
                ax.plot(valid_x, valid_v, '-', color=col, linewidth=1.2,
                        alpha=0.5, zorder=1)

            for xi, v in enumerate(vals):
                if not np.isnan(v):
                    dot_col = (_group_color(names[xi], xi) if use_group_colors
                               else _SIM_COLORS.get(names[xi], _color(xi)))
                    if as_bars:
                        ax.bar(xi, v, width=0.6, color=dot_col,
                               edgecolor='black', linewidth=0.5,
                               alpha=0.85, zorder=2)
                    else:
                        ax.plot(xi, v, marker=_marker(names[xi]), color=dot_col,
                                markersize=6, markeredgecolor='black',
                                markeredgewidth=0.4, linestyle='none', zorder=2)


            # Value annotations above each point
            for xi, v in enumerate(vals):
                if not np.isnan(v):
                    fmt = f'{v:.2f}' if src == 'ratio' else f'{v:.0f}'
                    ax.annotate(fmt, xy=(xi, v),
                                xytext=(0, 6), textcoords='offset points',
                                ha='center', va='bottom', fontsize=fs_annot,
                                color='black')

            if src == 'ratio':
                ax.axhline(1.0, color='gray', linewidth=ratio_ref_linewidth,
                           linestyle='--', alpha=0.6)

            ax.set_xticks(x)
            ax.set_xlim(-0.5, len(names) - 0.5)
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=8, steps=[1, 2, 5, 10]))
            ax.tick_params(axis='y', labelsize=fs_tick)
            # Headroom for the value annotations above the highest point.
            valid_for_pad = [v for v in vals if not np.isnan(v)]
            if valid_for_pad and ymargin_top > 0:
                vmin = min(valid_for_pad)
                vmax = max(valid_for_pad)
                span = max(vmax - vmin, abs(vmax) * 0.05, 1e-9)
                lower = 0.0 if (as_bars or ymin_zero) else vmin - 0.05 * span
                ax.set_ylim(lower, vmax + ymargin_top * span)

            # Per-axis ylim override.  Accepts:
            #   • a number       → upper limit, lower kept from auto
            #   • (lo, hi) tuple → explicit (None entries keep auto)
            if ylim_overrides:
                ovr = ylim_overrides.get((ri, ci))
                if ovr is not None:
                    cur_lo, cur_hi = ax.get_ylim()
                    if isinstance(ovr, (int, float)):
                        ax.set_ylim(cur_lo, float(ovr))
                    else:
                        lo, hi = ovr
                        ax.set_ylim(cur_lo if lo is None else lo,
                                    cur_hi if hi is None else hi)
            if ri == last_row_idx:
                ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=fs_tick)
            else:
                ax.set_xticklabels([])
            if ci == 0:
                ax.set_ylabel(row_label, fontsize=fs_ylabel)
            if ri == 0:
                ax.set_title(comp, fontsize=fs_title, color=col)

    # Optionally force every column in a given row to share a common ylim
    # (taken as the union of each axis's current ylim).  Done after individual
    # ylim_overrides so explicit per-axis settings still propagate.
    if share_row_ylims:
        for ri in range(n_rows):
            los = [axes[ri, ci].get_ylim()[0] for ci in range(3)]
            his = [axes[ri, ci].get_ylim()[1] for ci in range(3)]
            row_lo, row_hi = min(los), max(his)
            for ci in range(3):
                axes[ri, ci].set_ylim(row_lo, row_hi)

    #fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


# ── IAU pole model helpers (inlined from PropFuncs.PoleModel — no tudatpy needed) ──

_POLE_PARAM_TYPES = {
    'iau_rotation_model_pole',
    'iau_rotation_model_pole_rate',
    'iau_rotation_model_pole_librations',
}

def _pole_linestyle(sd):
    """Linestyle based on number of pole parameter groups estimated.

    0 groups (state only) → solid '-'
    1 group              → dashed '--'
    2 groups             → dash-dot '-.'
    3 groups             → dotted ':'

    Ensures that combined estimations (pos+lib, pos+rot, full) are visually
    distinct from single-group ones even when colours are similar.
    """
    n = sum(1 for p in sd.get('est_parameters', []) if p in _POLE_PARAM_TYPES)
    return ('-', '--', '-.', ':')[min(n, 3)]

def _iau_pole_model(time_j2000, update):
    """Evaluate IAU 2015 Neptune pole RA/Dec over time (delta-based, kept for back-compat).

    Parameters
    ----------
    time_j2000 : array_like  — seconds from J2000 epoch
    update     : array_like  — [Δα₀, Δδ₀, Δα̇₀, Δδ̇₀, Δα₁, Δδ₁] in rad / (rad/s)
    """
    t = np.asarray(time_j2000, dtype=float)
    alpha_0   = np.deg2rad(299.36)  + update[0]
    delta_0   = np.deg2rad(43.46)   + update[1]
    alpha_dot = 0.0                 + update[2]
    delta_dot = 0.0                 + update[3]
    alpha_1   = np.deg2rad(0.7)     + update[4]
    delta_1   = np.deg2rad(-0.51)   + update[5]
    omega     = np.deg2rad(52.316 / 36525.0 / 86400.0)
    phi       = np.deg2rad(357.85)
    phase     = omega * t + phi
    alpha = alpha_0 + alpha_dot * t + alpha_1 * np.sin(phase)
    delta = delta_0 + delta_dot * t + delta_1 * np.cos(phase)
    return alpha, delta


def _extract_iau_pole_update(sd):
    """Extract 6-element IAU pole parameter update vector (delta from initial)."""
    _LBL_TO_SLOT = {'α₀': 0, 'δ₀': 1, 'α̇₀': 2, 'δ̇₀': 3, 'α₁': 4, 'δ₁': 5}
    update = np.zeros(6)
    if 'parameter_history' not in sd or 'est_parameters' not in sd:
        return update
    ph   = sd['parameter_history']
    lbls, _, _ = get_parameter_info(sd['est_parameters'])
    for i, lbl in enumerate(lbls):
        if lbl in _LBL_TO_SLOT:
            update[_LBL_TO_SLOT[lbl]] = ph[i, -1] - ph[i, 0]
    return update


# IAU 2015 pole model default constants (radians).
_IAU2015_DEFAULTS = np.array([
    np.deg2rad(299.36),   # α₀
    np.deg2rad(43.46),    # δ₀
    0.0,                   # α̇₀
    0.0,                   # δ̇₀
    np.deg2rad(0.7),      # α₁
    np.deg2rad(-0.51),    # δ₁
])

_POLE_LBL_TO_SLOT = {'α₀': 0, 'δ₀': 1, 'α̇₀': 2, 'δ̇₀': 3, 'α₁': 4, 'δ₁': 5}


def _pole_model_from_params(time_j2000, params):
    """Evaluate Neptune pole RA/Dec from absolute parameter values.

    params : array-like (6,) — [α₀, δ₀, α̇₀, δ̇₀, α₁, δ₁] in radians.
    Returns alpha, delta in radians.
    """
    t     = np.asarray(time_j2000, dtype=float)
    omega = np.deg2rad(52.316 / 36525.0 / 86400.0)
    phi   = np.deg2rad(357.85)
    phase = omega * t + phi
    alpha = params[0] + params[2] * t + params[4] * np.sin(phase)
    delta = params[1] + params[3] * t + params[5] * np.cos(phase)
    return alpha, delta


def _extract_absolute_pole_params(sd, iteration, defaults):
    """Extract absolute pole params [α₀,δ₀,α̇₀,δ̇₀,α₁,δ₁] at a given iteration.

    Non-estimated parameters fall back to `defaults` (a 6-element array in radians).
    """
    params = np.array(defaults, dtype=float)
    if 'parameter_history' not in sd or 'est_parameters' not in sd:
        return params
    ph   = sd['parameter_history']
    lbls, _, _ = get_parameter_info(sd['est_parameters'])
    for i, lbl in enumerate(lbls):
        if lbl in _POLE_LBL_TO_SLOT:
            params[_POLE_LBL_TO_SLOT[lbl]] = ph[i, iteration]
    return params


def _has_pole_params(sd):
    """Return True if sim has any estimated pole parameters."""
    return any(p in sd.get('est_parameters', []) for p in _POLE_PARAM_TYPES)


def _get_pole_nominal_params(sims, names, prefix, fallback_defaults):
    """Find initial (iteration=0) pole params from the most complete matching sim.

    Prefers the sim containing 'pole_pos_cov_pole_lib_cov'; falls back to any
    sim with the given prefix that has pole parameters.
    Returns a 6-element numpy array of absolute pole params.
    """
    # Priority: fully-estimated (pos + lib) sim
    for sn in names:
        if sn.startswith(prefix) and 'pole_pos_cov_pole_lib_cov' in sn and sn in sims:
            sd = sims[sn]
            if _has_pole_params(sd) and 'parameter_history' in sd:
                return _extract_absolute_pole_params(sd, 0, fallback_defaults)
    # Fallback: any sim with the prefix that has pole params
    for sn in names:
        if sn.startswith(prefix) and sn in sims:
            sd = sims[sn]
            if _has_pole_params(sd) and 'parameter_history' in sd:
                return _extract_absolute_pole_params(sd, 0, fallback_defaults)
    return np.array(fallback_defaults, dtype=float)


def mpl_pole_model(sims, names, labels,
                   sim_subset=None,
                   title='Neptune Pole Trajectory (α, δ)'):
    """Two-panel time series of Neptune pole RA and Dec.

    Upper panel: right ascension α in degrees.
    Lower panel: declination δ in degrees.

    Reference lines:
      • IAU nominal   — initial params of IAUPole_pole_pos_cov_pole_lib_cov (black solid)
      • FitPole nominal — initial params of SimPole_pole_pos_cov_pole_lib_cov (gray solid)

    Each pole-estimating simulation is plotted using its final estimated params.
    Sims without pole estimation (state, state Fit.) are skipped — they are
    identical to their respective nominal.
    """
    import matplotlib.lines as mlines

    # Resolve the full name/label lists used for color/marker lookup
    all_names  = list(names)
    all_labels = list(labels)

    if sim_subset is not None:
        pairs  = [(n, all_labels[all_names.index(n)])
                  for n in sim_subset if n in sims and n in all_names]
        plot_names  = [p[0] for p in pairs]
        plot_labels = [p[1] for p in pairs]
    else:
        plot_names  = all_names
        plot_labels = all_labels

    # ── nominal params from pos+lib sims (iteration 0 = initial values) ──────
    iau_defaults     = _get_pole_nominal_params(sims, all_names, 'IAUPole', _IAU2015_DEFAULTS)
    fitpole_defaults = _get_pole_nominal_params(sims, all_names, 'SimPole', iau_defaults)

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W_DOUBLE * 1.4, FIG_H_DEFAULT * 1.8),
                             sharex=True)

    t_ref_j2000 = None
    ref_times   = None

    for j, (sn, lbl) in enumerate(zip(plot_names, plot_labels)):
        sd = sims.get(sn, {})
        if 'state_history_array' not in sd:
            continue
        if not _has_pole_params(sd):
            continue   # skip state / state (Fit.) — identical to respective nominal

        t_j2000 = sd['state_history_array'][:, 0]
        times   = convert_time_array_to_datetime(t_j2000)
        if t_ref_j2000 is None:
            t_ref_j2000 = t_j2000
            ref_times   = times

        # Use the group-appropriate defaults for non-estimated params
        defaults = fitpole_defaults if sn.startswith('SimPole') else iau_defaults
        params   = _extract_absolute_pole_params(sd, -1, defaults)
        alpha, delta = _pole_model_from_params(t_j2000, params)

        col = _SIM_COLORS.get(sn, _color(j))
        ls  = _pole_linestyle(sd)
        axes[0].plot(times, np.rad2deg(alpha), color=col, linewidth=1.4,
                     linestyle=ls, label=lbl, zorder=2)
        axes[1].plot(times, np.rad2deg(delta), color=col, linewidth=1.4,
                     linestyle=ls, label='_nolegend_', zorder=2)

    # ── nominal reference lines ───────────────────────────────────────────────
    if t_ref_j2000 is not None:
        a_iau, d_iau = _pole_model_from_params(t_ref_j2000, iau_defaults)
        axes[0].plot(ref_times, np.rad2deg(a_iau), color='black',
                     linewidth=2.0, linestyle='-', label='IAU nominal', zorder=4)
        axes[1].plot(ref_times, np.rad2deg(d_iau), color='black',
                     linewidth=2.0, linestyle='-', label='_nolegend_', zorder=4)

        a_fp, d_fp = _pole_model_from_params(t_ref_j2000, fitpole_defaults)
        axes[0].plot(ref_times, np.rad2deg(a_fp), color='gray',
                     linewidth=2.0, linestyle='-', label='FitPole nominal', zorder=4)
        axes[1].plot(ref_times, np.rad2deg(d_fp), color='gray',
                     linewidth=2.0, linestyle='-', label='_nolegend_', zorder=4)

    axes[0].set_ylabel(r'$\alpha$ [deg]')
    axes[1].set_ylabel(r'$\delta$ [deg]')
    axes[1].set_xlabel('Date')
    for ax in axes:
        _apply_date_formatter(ax)
    axes[0].legend(loc='best', fontsize=8, ncol=2)
    axes[0].set_title(title)
    fig.tight_layout()
    return fig


def mpl_pole_model_diff(sims, names, labels,
                        sim_subset=None,
                        title='Neptune Pole Deviation from Nominal (Δα, Δδ)'):
    """Two-panel time series of deviation from the respective group nominal.

    Upper panel: Δα = α_estimated − α_nominal  (millidegrees)
    Lower panel: Δδ = δ_estimated − δ_nominal  (millidegrees)

    IAUPole sims are compared against the IAU nominal (initial params of
    IAUPole_pole_pos_cov_pole_lib_cov).  SimPole sims are compared against the
    FitPole nominal (initial params of SimPole_pole_pos_cov_pole_lib_cov).
    Sims without estimated pole parameters are skipped.
    """
    import matplotlib.lines as mlines

    all_names  = list(names)
    all_labels = list(labels)

    if sim_subset is not None:
        pairs  = [(n, all_labels[all_names.index(n)])
                  for n in sim_subset if n in sims and n in all_names]
        plot_names  = [p[0] for p in pairs]
        plot_labels = [p[1] for p in pairs]
    else:
        plot_names  = all_names
        plot_labels = all_labels

    # ── nominal params ────────────────────────────────────────────────────────
    iau_defaults     = _get_pole_nominal_params(sims, all_names, 'IAUPole', _IAU2015_DEFAULTS)
    fitpole_defaults = _get_pole_nominal_params(sims, all_names, 'SimPole', iau_defaults)

    _MDEG = 1e3   # degrees → millidegrees

    fig, axes = plt.subplots(2, 1, figsize=(FIG_W_DOUBLE * 1.4, FIG_H_DEFAULT * 1.8),
                             sharex=True)
    axes[0].axhline(0, color='black', linewidth=0.8, linestyle='--', zorder=1)
    axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--', zorder=1)

    plotted = False
    for j, (sn, lbl) in enumerate(zip(plot_names, plot_labels)):
        sd = sims.get(sn, {})
        if 'state_history_array' not in sd:
            continue
        if not _has_pole_params(sd):
            continue   # skip state / state (Fit.)

        t_j2000 = sd['state_history_array'][:, 0]
        times   = convert_time_array_to_datetime(t_j2000)

        # Use the group-appropriate defaults for non-estimated params
        defaults   = fitpole_defaults if sn.startswith('SimPole') else iau_defaults
        nominal    = defaults          # nominal = initial params of this group
        params_est = _extract_absolute_pole_params(sd, -1, defaults)

        alpha_est, delta_est = _pole_model_from_params(t_j2000, params_est)
        alpha_nom, delta_nom = _pole_model_from_params(t_j2000, nominal)

        d_alpha = np.rad2deg(alpha_est - alpha_nom) * _MDEG
        d_delta = np.rad2deg(delta_est - delta_nom) * _MDEG

        col = _SIM_COLORS.get(sn, _color(j))
        ls  = _pole_linestyle(sd)
        axes[0].plot(times, d_alpha, color=col, linewidth=1.4,
                     linestyle=ls, label=lbl, zorder=2)
        axes[1].plot(times, d_delta, color=col, linewidth=1.4,
                     linestyle=ls, label='_nolegend_', zorder=2)
        plotted = True

    if not plotted:
        plt.close(fig)
        print('  SKIP pole_model_diff: no sims with estimated pole parameters.')
        return None

    axes[0].set_ylabel(r'$\Delta\alpha$ [mdeg]')
    axes[1].set_ylabel(r'$\Delta\delta$ [mdeg]')
    axes[1].set_xlabel('Date')
    for ax in axes:
        _apply_date_formatter(ax)
    axes[0].legend(loc='best', fontsize=8, ncol=2)
    axes[0].set_title(title)
    fig.tight_layout()
    return fig


# ── Hard-coded pole models (from HelperFunctions/PropFuncs.py:79–168) ────────
# IAU 2015 (Archinal et al.) — single-frequency periodic term
# alpha(t) = α₀ + α̇₀·t + α₁·sin(ω·t + φ)
# delta(t) = δ₀ + δ̇₀·t + δ₁·cos(ω·t + φ)
_IAU2015 = {
    'alpha_0':  np.deg2rad(299.36),
    'delta_0':  np.deg2rad( 43.46),
    'alpha_dot': 0.0,
    'delta_dot': 0.0,
    'alpha_1':  np.deg2rad( 0.7),
    'delta_1':  np.deg2rad(-0.51),
    'omega':    np.deg2rad(52.316 / 36525.0 / 86400.0),  # rad/sec
    'phi':      np.deg2rad(357.85),
}


def _eval_iau2015_pole(time_j2000, alpha_0, delta_0, alpha_dot, delta_dot,
                        alpha_1, delta_1, omega=None, phi=None):
    """IAU 2015 pole model with optional override of base constants."""
    if omega is None:
        omega = _IAU2015['omega']
    if phi is None:
        phi = _IAU2015['phi']
    t     = np.asarray(time_j2000, dtype=float)
    phase = omega * t + phi
    alpha = alpha_0 + alpha_dot * t + alpha_1 * np.sin(phase)
    delta = delta_0 + delta_dot * t + delta_1 * np.cos(phase)
    return alpha, delta


def _eval_jacobson2009_pole(time_j2000):
    """Jacobson 2009 Neptune pole model (two periodic terms at ω̇ and 2ω̇).

    Implements the same parametrisation as
    ``HelperFunctions/PropFuncs.py:124-168``.
    """
    alpha_r   = np.deg2rad(299.4608612607558)
    delta_r   = np.deg2rad( 43.4048107907141)
    epsilon   = np.deg2rad(  0.4616274249865)
    omega_dot = np.deg2rad(52.3836218446110 / 36525.0 / 86400.0)  # rad/sec
    omega_0   = np.deg2rad(352.1753923868973)  # 1989-08-25 epoch

    # Adjust omega_0 to J2000 epoch.
    t0 = np.datetime64('1989-08-25T00:00:00')
    t1 = np.datetime64('2000-01-01T12:00:00')
    seconds_to_J2000 = (t1 - t0) / np.timedelta64(1, 's')
    omega_0 = omega_0 + omega_dot * seconds_to_J2000

    alpha_0 = alpha_r
    alpha_1 = epsilon * (1.0 / np.cos(delta_r))
    alpha_2 = -0.5 * epsilon ** 2 * np.tan(delta_r) / np.cos(delta_r)

    delta_0 = delta_r - 0.25 * epsilon ** 2 * np.tan(delta_r)
    delta_1 = -epsilon
    delta_2 =  0.25 * epsilon ** 2 * np.tan(delta_r)

    t       = np.asarray(time_j2000, dtype=float)
    phase_1 = omega_dot     * t + omega_0
    phase_2 = 2.0 * omega_dot * t + 2.0 * omega_0

    alpha = (alpha_0
             + alpha_1 * np.sin(phase_1)
             + alpha_2 * np.sin(phase_2))
    delta = (delta_0
             + delta_1 * np.cos(phase_1)
             + delta_2 * np.cos(phase_2))
    return alpha, delta


def _extract_iau_pole_deltas(sd):
    """Return 6-element [Δα₀, Δδ₀, Δα̇₀, Δδ̇₀, Δα₁, Δδ₁] from a sim's
    parameter_history (final − initial).  Missing slots = 0.

    Kept for callers that explicitly want the increment from a sim's own
    starting point.  For chained-override semantics, prefer reading the
    absolute final values via ``_extract_absolute_pole_params(sd, -1, base)``.
    """
    deltas = np.zeros(6)
    if 'parameter_history' not in sd or 'est_parameters' not in sd:
        return deltas
    ph         = sd['parameter_history']
    lbls, _, _ = get_parameter_info(sd['est_parameters'])
    for i, lbl in enumerate(lbls):
        if lbl in _POLE_LBL_TO_SLOT:
            slot = _POLE_LBL_TO_SLOT[lbl]
            deltas[slot] = ph[i, -1] - ph[i, 0]
    return deltas


def _pole_uncertainty_sigma(t_j2000, base_params,
                            sigma_alpha1, sigma_delta1,
                            method='gaussian',
                            n_samples=10000,
                            rng_seed=42,
                            omega=None, phi=None,
                            return_samples=False):
    """1σ envelope of the IAU 2015 pole curve under uncertainty in (α₁, δ₁).

    The IAU 2015 model is α(t) = α₀ + α̇₀·t + α₁·sin(ωt+φ) and
    δ(t) = δ₀ + δ̇₀·t + δ₁·cos(ωt+φ).  Only the α₁ / δ₁ slots carry the
    estimation-supplied 1σ — the other base parameters are treated as fixed
    (matching how Real. Fit. Pole is built in the figure: state+lib estimation
    only updates α₁ and δ₁).

    method = 'gaussian'  — analytic linear propagation:
        σ_α(t) = |sin(phase)|·σ_α₁,  σ_δ(t) = |cos(phase)|·σ_δ₁
    method = 'mc'        — Monte-Carlo: draw n_samples from
        α₁ ~ N(α₁_base, σ_α₁²), δ₁ ~ N(δ₁_base, σ_δ₁²) (independent),
        propagate through the model, take per-time std across samples.

    Returns
    -------
    alpha_c, delta_c : central curves [rad]
    sigma_alpha, sigma_delta : per-time 1σ [rad]
    samples : dict with 'alpha' and 'delta' (n_samples × n_t) arrays —
              only when return_samples=True (else None).
    """
    if omega is None:
        omega = _IAU2015['omega']
    if phi is None:
        phi = _IAU2015['phi']
    t       = np.asarray(t_j2000, dtype=float)
    phase   = omega * t + phi
    a0, d0, adot, ddot, a1m, d1m = (float(x) for x in base_params)
    alpha_c = a0 + adot * t + a1m * np.sin(phase)
    delta_c = d0 + ddot * t + d1m * np.cos(phase)

    if method == 'gaussian':
        sigma_alpha = float(sigma_alpha1) * np.abs(np.sin(phase))
        sigma_delta = float(sigma_delta1) * np.abs(np.cos(phase))
        return alpha_c, delta_c, sigma_alpha, sigma_delta, None

    if method == 'mc':
        rng    = np.random.default_rng(rng_seed)
        a1_s   = rng.normal(a1m, float(sigma_alpha1), size=int(n_samples))
        d1_s   = rng.normal(d1m, float(sigma_delta1), size=int(n_samples))
        sin_p  = np.sin(phase)[None, :]
        cos_p  = np.cos(phase)[None, :]
        alpha_samp = (a0 + adot * t)[None, :] + a1_s[:, None] * sin_p
        delta_samp = (d0 + ddot * t)[None, :] + d1_s[:, None] * cos_p
        sigma_alpha = alpha_samp.std(axis=0, ddof=1)
        sigma_delta = delta_samp.std(axis=0, ddof=1)
        samples = {'alpha': alpha_samp, 'delta': delta_samp} if return_samples else None
        return alpha_c, delta_c, sigma_alpha, sigma_delta, samples

    raise ValueError(f'unknown uncertainty method {method!r}')


_POLE_CACHE_WARNED = set()

def _try_apply_pole_cache(d, base):
    """If an absolute_from / deltas_from entry has ``cache_path`` and the
    file exists, fold its absolute pole params into ``base`` and return
    True (skipping any sim/dataset load).  Cache schema must match what
    build_pole_base_cache.py writes:
        {'absolute_params': ndarray(6,) [rad],
         'estimated_slots': list[int]   — slots to overwrite from cache}
    Prints a one-time warning per (cache_path, from_config) pair if the
    cache_path was specified but the file is missing — so the user sees
    why a heavy dataset load is about to happen.
    """
    cache_path = (d or {}).get('cache_path')
    if not cache_path:
        return False
    import os as _os
    if not _os.path.isfile(cache_path):
        key = (cache_path, d.get('from_config'))
        if key not in _POLE_CACHE_WARNED:
            _POLE_CACHE_WARNED.add(key)
            print(f"  WARNING: pole cache missing → {cache_path}\n"
                  f"           will fall back to loading "
                  f"{d.get('from_config')!r} (memory-heavy).\n"
                  f"           Run `python build_pole_base_cache.py` once to "
                  f"build the cache and avoid this load.")
        return False
    try:
        cache = np.load(cache_path, allow_pickle=True).item()
    except Exception as exc:
        print(f"  pole cache load failed ({cache_path}): {exc}")
        return False
    cached = np.asarray(cache['absolute_params'], dtype=float)
    slots  = list(cache.get('estimated_slots', range(6)))
    for s in slots:
        base[s] = cached[s]
    return True


def _resolve_iau_base_params(entry, _get_sims_fn):
    """Re-derive the 6-element IAU base param vector for an 'iau' entry,
    mirroring the chained logic in mpl_pole_model_compare._eval_entry.
    Returned in radians.  Honours per-step ``cache_path`` so an
    absolute_from step backed by a cache .npy never triggers a dataset
    load (used to avoid loading the 6 GB SimObs pickle just to chain
    Sim. Fit. Pole into Real. Fit. Pole)."""
    base = list(entry.get('base_params') or [
        _IAU2015['alpha_0'], _IAU2015['delta_0'],
        _IAU2015['alpha_dot'], _IAU2015['delta_dot'],
        _IAU2015['alpha_1'], _IAU2015['delta_1'],
    ])
    base = np.array(base, dtype=float)
    for d in entry.get('absolute_from', []) or []:
        if _try_apply_pole_cache(d, base):
            continue
        src_sims = _get_sims_fn(d.get('from_config'))
        sd       = src_sims.get(d['sim'], {})
        base     = _extract_absolute_pole_params(sd, -1, base)
    for d in entry.get('deltas_from', []) or []:
        src_sims = _get_sims_fn(d.get('from_config'))
        sd       = src_sims.get(d['sim'], {})
        base    += _extract_iau_pole_deltas(sd)
    return base


def _pull_pole_sigmas(sims_root, names_root, sigma_source, _get_sims_fn):
    """Pull (σ_α₁, σ_δ₁) in radians from a sim's ``formal_errors`` array.
    Convention (per CASE1_Manual_Bias / state+lib estimation): the last two
    entries of formal_errors correspond to α₁ and δ₁ respectively.
    """
    src_sims = (_get_sims_fn(sigma_source.get('from_config'))
                if sigma_source.get('from_config') else sims_root)
    sd       = src_sims.get(sigma_source['sim'], {})
    fe       = sd.get('formal_errors')
    if fe is None or len(fe) < 2:
        return None, None
    return float(fe[-2]), float(fe[-1])


def mpl_pole_model_compare(sims, names, labels,
                            entries=None,
                            title='Neptune Pole Trajectory — Model Comparison',
                            figsize=None,
                            axes_fontsize=11,
                            tick_fontsize=10,
                            legend_fontsize=10,
                            title_fontsize=12,
                            show_suptitle=True,
                            time_source=None,
                            ref_entries=None,
                            ylim_alpha=None,
                            ylim_delta=None,
                            time_decimation_target=8000):
    """Two-panel α / δ time series for arbitrary pole-model definitions.

    Each entry in ``entries`` defines one curve.  Three "kinds" are supported:

      'iau' — IAU 2015 functional form.  Optional ``base_params`` (6-element
              list/tuple [α₀,δ₀,α̇₀,δ̇₀,α₁,δ₁] in radians) overrides the
              IAU 2015 defaults.  Two ways to fold a sim's estimated values
              onto the base (either, neither, or both can be supplied):
                ``absolute_from`` — list of ``{'sim','from_config'}`` dicts
                  applied left-to-right; each step replaces the slots that
                  the named sim *estimated* with that sim's absolute final
                  parameter values (``parameter_history[:,-1]``), keeping
                  prior slots untouched.  This is the clean way to chain
                  fitted poles (Sim. Fit. Pole → Real. Fit. Pole).
                ``deltas_from`` — list of ``{'sim','from_config'}`` dicts
                  whose ``(final − initial)`` increments are summed onto the
                  current vector.  Equivalent to ``absolute_from`` when the
                  sim's initial parameter values match the current base.

      'jacobson_2009' — hard-coded Jacobson 2009 model (no overrides).

      'precomputed' — read ``pole_alpha`` / ``pole_delta`` arrays from the
              pickle for the named sim (legacy path).

    Common entry keys: 'label', 'color', 'linestyle' (default '-'),
    'linewidth' (default 1.6), 'alpha' (default 1.0).

    The shared time axis is taken from ``time_source`` (a dict with
    'sim' / 'from_config' keys) when provided; otherwise the first entry that
    has a sim reference (deltas_from / sim) is used.
    """
    if not entries:
        print('  SKIP pole_model_compare: no entries provided.')
        return None

    if figsize is None:
        figsize = (FIG_W_DOUBLE * 1.6, FIG_H_DEFAULT * 1.6)
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Dataset cache for from_config lookups.
    _ds_cache = {None: (sims, names, labels)}

    def _get_sims(from_cfg):
        if from_cfg not in _ds_cache:
            try:
                _src_sims, _src_names, _src_labels, _, _, _ = (
                    _load_dataset_for_config(from_cfg))
            except Exception as exc:
                print(f"  pole_model_compare: failed to load "
                      f"from_config={from_cfg!r} ({exc})")
                _ds_cache[from_cfg] = ({}, [], [])
                return {}
            _ds_cache[from_cfg] = (_src_sims, _src_names, _src_labels)
        return _ds_cache[from_cfg][0]

    def _try_time_from(sim_ref):
        """Return t_j2000 array for a {'sim','from_config'} dict, or None."""
        if not sim_ref or 'sim' not in sim_ref:
            return None
        src_sims = _get_sims(sim_ref.get('from_config'))
        sd       = src_sims.get(sim_ref['sim'], {})
        if 'state_history_array' in sd:
            arr = sd['state_history_array']
            if arr.ndim >= 2 and arr.shape[1] >= 1:
                return arr[:, 0]
        return None

    # ── Resolve shared time axis (J2000 seconds) ─────────────────────────────
    t_j2000 = None
    if time_source is not None:
        t_j2000 = _try_time_from(time_source)
        if t_j2000 is None:
            print(f"  pole_model_compare: time_source {time_source!r} "
                  f"missing state_history_array; falling back.")
    if t_j2000 is None:
        for entry in entries:
            if entry.get('kind') == 'precomputed' and 'sim' in entry:
                t_j2000 = _try_time_from(
                    {'sim': entry['sim'],
                     'from_config': entry.get('from_config')})
                if t_j2000 is not None:
                    break
            for key in ('absolute_from', 'deltas_from'):
                for d in entry.get(key, []) or []:
                    t_j2000 = _try_time_from(d)
                    if t_j2000 is not None:
                        break
                if t_j2000 is not None:
                    break
            if t_j2000 is not None:
                break
    if t_j2000 is None:
        # Last-resort: any sim from the primary dataset that has a state history.
        for sn in names:
            sd = sims.get(sn, {})
            if 'state_history_array' in sd:
                t_j2000 = sd['state_history_array'][:, 0]
                print(f"  pole_model_compare: using time axis from primary "
                      f"sim {sn!r}.")
                break
    if t_j2000 is None:
        plt.close(fig)
        print('  SKIP pole_model_compare: no time axis could be resolved.')
        return None
    # Uniform decimation — propagation can leave 10s of millions of points;
    # plotting that many ticks balloons matplotlib memory and adds nothing
    # visible.  Result is smooth at any reasonable display resolution.
    t_j2000 = np.asarray(t_j2000, dtype=float)
    if (time_decimation_target and time_decimation_target > 0
            and len(t_j2000) > time_decimation_target):
        stride  = int(np.ceil(len(t_j2000) / float(time_decimation_target)))
        n_full  = len(t_j2000)
        t_j2000 = t_j2000[::stride]
        print(f'  pole_model_compare: decimated time axis '
              f'{n_full} → {len(t_j2000)} points (stride={stride}).')
    times = convert_time_array_to_datetime(t_j2000)

    def _eval_entry(entry):
        kind = entry.get('kind', 'iau')
        if kind == 'jacobson_2009':
            return _eval_jacobson2009_pole(t_j2000)
        if kind == 'iau':
            base = list(entry.get('base_params') or [
                _IAU2015['alpha_0'], _IAU2015['delta_0'],
                _IAU2015['alpha_dot'], _IAU2015['delta_dot'],
                _IAU2015['alpha_1'], _IAU2015['delta_1'],
            ])
            base = np.array(base, dtype=float)
            for d in entry.get('absolute_from', []) or []:
                if _try_apply_pole_cache(d, base):
                    continue   # cache hit — no dataset load needed
                src_sims = _get_sims(d.get('from_config'))
                sd       = src_sims.get(d['sim'], {})
                base     = _extract_absolute_pole_params(sd, -1, base)
            for d in entry.get('deltas_from', []) or []:
                src_sims = _get_sims(d.get('from_config'))
                sd       = src_sims.get(d['sim'], {})
                base    += _extract_iau_pole_deltas(sd)
            return _eval_iau2015_pole(
                t_j2000, base[0], base[1], base[2], base[3], base[4], base[5])
        if kind == 'precomputed':
            sn       = entry['sim']
            src_sims = _get_sims(entry.get('from_config'))
            sd       = src_sims.get(sn, {})
            if 'pole_alpha' not in sd or 'pole_delta' not in sd:
                return None, None
            return np.asarray(sd['pole_alpha']), np.asarray(sd['pole_delta'])
        return None, None

    plotted = False
    for j, entry in enumerate(entries):
        lbl  = entry['label']
        col  = entry.get('color') or _color(j)
        ls   = entry.get('linestyle', '-')
        lw   = entry.get('linewidth', 1.6)
        alp  = entry.get('alpha', 1.0)
        alpha, delta = _eval_entry(entry)
        if alpha is None or delta is None:
            print(f"  SKIP pole_model_compare entry {entry!r}: could not evaluate.")
            continue

        n_min = min(len(times), len(alpha), len(delta))
        axes[0].plot(times[:n_min], np.rad2deg(alpha[:n_min]),
                     color=col, linestyle=ls, linewidth=lw, alpha=alp,
                     label=lbl, zorder=2 + j)
        axes[1].plot(times[:n_min], np.rad2deg(delta[:n_min]),
                     color=col, linestyle=ls, linewidth=lw, alpha=alp,
                     label='_nolegend_', zorder=2 + j)
        plotted = True

        # ── Optional uncertainty band (Gaussian or MC) around this entry ─────
        unc = entry.get('uncertainty')
        if unc is not None and entry.get('kind', 'iau') == 'iau':
            sa1, sd1 = _pull_pole_sigmas(sims, names,
                                         unc['sigma_source'], _get_sims)
            if sa1 is None:
                print(f"  pole_model_compare: missing formal_errors for "
                      f"{unc['sigma_source']!r}, skipping band.")
            else:
                base_p = _resolve_iau_base_params(entry, _get_sims)
                a_c, d_c, s_a, s_d, _ = _pole_uncertainty_sigma(
                    t_j2000, base_p,
                    sigma_alpha1=unc.get('n_sigma', 1.0) * sa1,
                    sigma_delta1=unc.get('n_sigma', 1.0) * sd1,
                    method=unc.get('method', 'gaussian'),
                    n_samples=int(unc.get('n_samples', 10000)),
                    rng_seed=int(unc.get('rng_seed', 42)),
                )
                band_color = unc.get('band_color') or col
                band_alpha = float(unc.get('band_alpha', 0.25))
                band_lbl   = unc.get('label')
                axes[0].fill_between(
                    times[:n_min],
                    np.rad2deg((a_c - s_a)[:n_min]),
                    np.rad2deg((a_c + s_a)[:n_min]),
                    color=band_color, alpha=band_alpha,
                    edgecolor='none', linewidth=0,
                    label=(band_lbl or '_nolegend_'),
                    zorder=1.5 + j)
                axes[1].fill_between(
                    times[:n_min],
                    np.rad2deg((d_c - s_d)[:n_min]),
                    np.rad2deg((d_c + s_d)[:n_min]),
                    color=band_color, alpha=band_alpha,
                    edgecolor='none', linewidth=0,
                    label='_nolegend_',
                    zorder=1.5 + j)

    if not plotted:
        plt.close(fig)
        return None

    # Optionally lock the y-axis to a reference set of entries (so a sequence
    # of plots with progressively-added curves share identical axis limits).
    if (ylim_alpha is None or ylim_delta is None) and ref_entries:
        a_lo, a_hi, d_lo, d_hi = np.inf, -np.inf, np.inf, -np.inf
        for entry in ref_entries:
            alpha, delta = _eval_entry(entry)
            if alpha is None or delta is None:
                continue
            n_min = min(len(times), len(alpha), len(delta))
            a = np.rad2deg(alpha[:n_min])
            d = np.rad2deg(delta[:n_min])
            a_lo = min(a_lo, float(np.nanmin(a)))
            a_hi = max(a_hi, float(np.nanmax(a)))
            d_lo = min(d_lo, float(np.nanmin(d)))
            d_hi = max(d_hi, float(np.nanmax(d)))
        if np.isfinite(a_lo) and np.isfinite(a_hi):
            pad_a = max((a_hi - a_lo) * 0.05, 1e-6)
            pad_d = max((d_hi - d_lo) * 0.05, 1e-6)
            if ylim_alpha is None:
                ylim_alpha = (a_lo - pad_a, a_hi + pad_a)
            if ylim_delta is None:
                ylim_delta = (d_lo - pad_d, d_hi + pad_d)
    if ylim_alpha is not None:
        axes[0].set_ylim(*ylim_alpha)
    if ylim_delta is not None:
        axes[1].set_ylim(*ylim_delta)

    axes[0].set_ylabel(r'$\alpha$ [deg]', fontsize=axes_fontsize)
    axes[1].set_ylabel(r'$\delta$ [deg]', fontsize=axes_fontsize)
    axes[1].set_xlabel('Date', fontsize=axes_fontsize)
    for ax in axes:
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        _apply_date_formatter(ax)
    axes[0].legend(loc='best', fontsize=legend_fontsize)
    if show_suptitle and title:
        axes[0].set_title(title, fontsize=title_fontsize)
    fig.tight_layout()
    return fig


def mpl_pole_uncertainty_validation(
        sims, names, labels,
        entry=None,
        time_source=None,
        sigma_source=None,
        n_samples=10000,
        rng_seed=42,
        n_sigma=1.0,
        title='Pole-band Uncertainty: Gaussian vs Monte-Carlo',
        figsize=None,
        axes_fontsize=11,
        tick_fontsize=10,
        legend_fontsize=10,
        title_fontsize=12,
        show_suptitle=True,
        gaussian_color='#0072B2',
        mc_color='#D55E00'):
    """Side-by-side validation that linear (Gaussian) error propagation and
    Monte-Carlo sampling produce the same 1σ envelope around the IAU 2015
    pole curve.

    Layout: 2×2.
        (0,0)  α(t) central + Gaussian band + MC band overlaid
        (0,1)  σ_α(t) lines: Gaussian (analytic) vs MC (sample std)
        (1,0)  δ(t) central + Gaussian band + MC band overlaid
        (1,1)  σ_δ(t) lines: Gaussian (analytic) vs MC (sample std)

    Parameters
    ----------
    entry : dict
        Same schema as one entry of mpl_pole_model_compare (kind='iau' with
        absolute_from / deltas_from chain).  Defines the central curve.
    time_source : dict {'sim','from_config'}
        Pulls a state_history_array to define the shared time axis.
    sigma_source : dict {'sim','from_config'}
        Pulls formal_errors[-2:] = (σ_α₁, σ_δ₁) in radians.
    """
    if entry is None or time_source is None or sigma_source is None:
        print('  SKIP pole_uncertainty_validation: entry / time_source / '
              'sigma_source all required.')
        return None

    _ds_cache = {None: (sims, names, labels)}

    def _get_sims(from_cfg):
        if from_cfg not in _ds_cache:
            try:
                _src_sims, _src_names, _src_labels, _, _, _ = (
                    _load_dataset_for_config(from_cfg))
            except Exception as exc:
                print(f"  pole_uncertainty_validation: failed to load "
                      f"from_config={from_cfg!r} ({exc})")
                _ds_cache[from_cfg] = ({}, [], [])
                return {}
            _ds_cache[from_cfg] = (_src_sims, _src_names, _src_labels)
        return _ds_cache[from_cfg][0]

    src_sims = _get_sims(time_source.get('from_config'))
    sd_t     = src_sims.get(time_source['sim'], {})
    if 'state_history_array' not in sd_t:
        print('  SKIP pole_uncertainty_validation: time_source has no '
              'state_history_array.')
        return None
    t_j2000  = sd_t['state_history_array'][:, 0]
    times    = convert_time_array_to_datetime(t_j2000)

    sa1, sd1 = _pull_pole_sigmas(sims, names, sigma_source, _get_sims)
    if sa1 is None:
        print('  SKIP pole_uncertainty_validation: sigma_source missing '
              'formal_errors.')
        return None
    base_p = _resolve_iau_base_params(entry, _get_sims)

    a_c, d_c, sg_a, sg_d, _ = _pole_uncertainty_sigma(
        t_j2000, base_p, n_sigma * sa1, n_sigma * sd1, method='gaussian')
    a_c2, d_c2, smc_a, smc_d, _ = _pole_uncertainty_sigma(
        t_j2000, base_p, n_sigma * sa1, n_sigma * sd1, method='mc',
        n_samples=n_samples, rng_seed=rng_seed)

    # Convert to deg + mas/-equivalent for σ profiles.
    rad2deg = np.rad2deg
    rad2mas = (180.0 * 3600.0 * 1000.0) / np.pi   # for the σ profile readout

    if figsize is None:
        figsize = (FIG_W_DOUBLE * 1.7, FIG_H_DEFAULT * 1.7)
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)

    n_min = len(times)
    # ── (0,0) α band overlay ─────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(times, rad2deg(a_c), color='black', linewidth=1.4,
            label='central α(t)', zorder=4)
    ax.fill_between(times, rad2deg(a_c - sg_a), rad2deg(a_c + sg_a),
                    color=gaussian_color, alpha=0.30, edgecolor='none',
                    label=f'Gaussian ±{n_sigma:g}σ', zorder=2)
    ax.fill_between(times, rad2deg(a_c2 - smc_a), rad2deg(a_c2 + smc_a),
                    color=mc_color, alpha=0.30, edgecolor='none',
                    label=f'Monte-Carlo ±{n_sigma:g}σ  (N={n_samples})',
                    zorder=3)
    ax.set_ylabel(r'$\alpha$ [deg]', fontsize=axes_fontsize)
    ax.legend(loc='best', fontsize=legend_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)

    # ── (1,0) δ band overlay ─────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(times, rad2deg(d_c), color='black', linewidth=1.4, zorder=4)
    ax.fill_between(times, rad2deg(d_c - sg_d), rad2deg(d_c + sg_d),
                    color=gaussian_color, alpha=0.30, edgecolor='none',
                    zorder=2)
    ax.fill_between(times, rad2deg(d_c2 - smc_d), rad2deg(d_c2 + smc_d),
                    color=mc_color, alpha=0.30, edgecolor='none', zorder=3)
    ax.set_ylabel(r'$\delta$ [deg]', fontsize=axes_fontsize)
    ax.set_xlabel('Date', fontsize=axes_fontsize)
    _apply_date_formatter(ax)
    ax.tick_params(axis='both', labelsize=tick_fontsize)

    # ── (0,1) σ_α profiles ───────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(times, sg_a * rad2mas,  color=gaussian_color, linewidth=1.6,
            label='Gaussian σ_α', zorder=3)
    ax.plot(times, smc_a * rad2mas, color=mc_color, linewidth=1.2,
            linestyle='--', label='MC σ_α', zorder=4)
    # Residual line on twin axis: |MC − Gaussian| as % of Gaussian.
    ax_t = ax.twinx()
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.where(sg_a > 0, 100.0 * (smc_a - sg_a) / sg_a, np.nan)
    ax_t.plot(times, rel, color='gray', linewidth=0.8, alpha=0.7,
              label='(MC−Gauss)/Gauss [%]')
    ax_t.set_ylabel('residual [%]', fontsize=axes_fontsize - 1, color='gray')
    ax_t.tick_params(axis='y', labelsize=tick_fontsize, colors='gray')
    ax.set_ylabel(r'$\sigma_\alpha$ [mas]', fontsize=axes_fontsize)
    ax.legend(loc='upper right', fontsize=legend_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)

    # ── (1,1) σ_δ profiles ───────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(times, sg_d * rad2mas,  color=gaussian_color, linewidth=1.6,
            label='Gaussian σ_δ', zorder=3)
    ax.plot(times, smc_d * rad2mas, color=mc_color, linewidth=1.2,
            linestyle='--', label='MC σ_δ', zorder=4)
    ax_t = ax.twinx()
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.where(sg_d > 0, 100.0 * (smc_d - sg_d) / sg_d, np.nan)
    ax_t.plot(times, rel, color='gray', linewidth=0.8, alpha=0.7)
    ax_t.set_ylabel('residual [%]', fontsize=axes_fontsize - 1, color='gray')
    ax_t.tick_params(axis='y', labelsize=tick_fontsize, colors='gray')
    ax.set_ylabel(r'$\sigma_\delta$ [mas]', fontsize=axes_fontsize)
    ax.set_xlabel('Date', fontsize=axes_fontsize)
    ax.legend(loc='upper right', fontsize=legend_fontsize)
    _apply_date_formatter(ax)
    ax.tick_params(axis='both', labelsize=tick_fontsize)

    # Print a one-line numeric summary for sanity-check.
    max_rel_a = float(np.nanmax(np.abs(np.where(sg_a > 0,
                       (smc_a - sg_a) / sg_a, np.nan)))) * 100
    max_rel_d = float(np.nanmax(np.abs(np.where(sg_d > 0,
                       (smc_d - sg_d) / sg_d, np.nan)))) * 100
    print(f'  pole_uncertainty_validation: max |σ_MC − σ_Gauss| / σ_Gauss '
          f'(α: {max_rel_a:.2f}%, δ: {max_rel_d:.2f}%)  '
          f'with N={n_samples} MC samples.')

    if show_suptitle and title:
        fig.suptitle(title, fontsize=title_fontsize)
    fig.tight_layout()
    return fig


def mpl_rms_formal(sims, names, labels,
                   sim_subset=None,
                   use_group_colors=False,
                   title='RMS of Formal Errors (RSW)'):
    """Dot plot of total formal-error RMS per simulation.

    Total formal RMS = sqrt(mean(formal_errors_RSW_km ** 2)) over all R/S/W
    components — the scalar counterpart to rms_SPICE from mpl_rms_compare.
    use_group_colors is accepted for API consistency but has no effect beyond
    what is already encoded in _SIM_COLORS.
    """
    if sim_subset is not None:
        pairs  = [(n, labels[names.index(n)]) for n in sim_subset if n in sims and n in names]
        names  = [p[0] for p in pairs]
        labels = [p[1] for p in pairs]

    vals = []
    for sn in names:
        sd = sims.get(sn, {})
        if 'formal_errors_RSW_km' in sd:
            vals.append(float(np.sqrt(np.mean(sd['formal_errors_RSW_km'] ** 2))))
        else:
            vals.append(np.nan)

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(FIG_W_DOUBLE, 0.8 * len(names)), FIG_H_DEFAULT))

    valid_x = [xi for xi, v in enumerate(vals) if not np.isnan(v)]
    valid_v = [v  for v  in vals               if not np.isnan(v)]
    if len(valid_x) > 1:
        ax.plot(valid_x, valid_v, '-', color='gray', linewidth=1.0, alpha=0.5, zorder=1)

    for i, (xi, v) in enumerate(zip(x, vals)):
        if not np.isnan(v):
            col = (_group_color(names[i], i) if use_group_colors
                   else _SIM_COLORS.get(names[i], _color(i)))
            mk  = _marker(names[i])
            ax.plot(xi, v, marker=mk, color=col, markersize=8,
                    markeredgecolor='black', markeredgewidth=0.5,
                    linestyle='none', zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel(r'Formal Error RMS [km]')
    ax.set_title(title)
    ax.set_xlim(-0.6, len(names) - 0.4)
    fig.tight_layout()
    return fig


def mpl_rsw_compare(sims, names, labels,
                    sim_subset=None,
                    show_initial=False,
                    title='RSW Difference vs SPICE',
                    figsize=None,
                    axes_fontsize=9,
                    tick_fontsize=8,
                    legend_fontsize=7,
                    title_fontsize=9,
                    per_sim_alpha=None,
                    per_sim_linestyle=None,
                    decimate=1,
                    rasterized=False,
                    show_rms_in_legend=True):
    """3-row time-series of RSW difference vs SPICE for multiple simulations.

    Sized for half-textwidth placement in LaTeX (pair with mpl_formal_compare).
    figsize default = (FIG_W_DOUBLE / 2, 7.0) ≈ (3.25, 7.0 in).
    """
    if sim_subset is not None:
        pairs  = [(n, labels[names.index(n)]) for n in sim_subset if n in sims and n in names]
        names  = [p[0] for p in pairs]
        labels = [p[1] for p in pairs]

    if figsize is None:
        figsize = (FIG_W_DOUBLE / 2, 5.8)

    per_sim_alpha     = per_sim_alpha or {}
    per_sim_linestyle = per_sim_linestyle or {}

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    ylabels = [r'$\Delta R$ [km]', r'$\Delta S$ [km]', r'$\Delta W$ [km]']

    for i, (ax, ylab) in enumerate(zip(axes, ylabels)):
        for j, (sn, lbl) in enumerate(zip(names, labels)):
            if 'diff_SPICE_RSW' not in sims.get(sn, {}):
                continue
            dr    = sims[sn]['diff_SPICE_RSW']
            times = get_rsw_times(sims[sn], n_points=len(dr))
            rms   = sims[sn].get('rms_SPICE', None)
            leg   = f"{lbl} (RMS: {rms:.0f} km)" if (rms and show_rms_in_legend) else lbl
            col   = _SIM_COLORS.get(sn, _color(j))
            ls    = per_sim_linestyle.get(sn, _SIM_LINESTYLE.get(sn, '-'))
            alp   = per_sim_alpha.get(sn, max(0.55, 0.92 - j * 0.15))
            step  = max(int(decimate), 1)
            t_d   = times[::step] if step > 1 else times
            d_d   = dr[::step, i]  if step > 1 else dr[:, i]
            ax.plot(t_d, d_d,
                    color=col, linestyle=ls, linewidth=1.2, alpha=alp,
                    rasterized=rasterized,
                    label=leg if i == 0 else '_nolegend_')

            if show_initial and 'diff_SPICE_RSW_initial' in sims.get(sn, {}):
                dr_init = sims[sn]['diff_SPICE_RSW_initial']
                t_init = convert_time_array_to_datetime(
                    sims[sn]['time_column_initial'].reshape(-1, 1))
                ax.plot(t_init, dr_init[:, i],
                        color=col, linestyle='--',
                        linewidth=0.7, alpha=0.5,
                        label='_nolegend_')

        ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
        ax.set_ylabel(ylab, fontsize=axes_fontsize, labelpad=2)
        ax.yaxis.set_label_coords(-0.13, 0.5)
        nbins = 8 if i == 2 else 5
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=nbins, prune=None))
        ax.tick_params(axis='y', labelsize=tick_fontsize, length=3)
        _apply_date_formatter(ax)

    for ax in axes[:-1]:
        _hide_xticklabels(ax)
    axes[0].set_title(title, fontsize=title_fontsize)
    axes[0].legend(loc='upper right', fontsize=legend_fontsize, handlelength=1.5)
    axes[-1].set_xlabel('Date', fontsize=axes_fontsize)
    axes[-1].tick_params(axis='x', labelsize=tick_fontsize, length=3)
    # Apply after all plotting so date formatters don't override margins.
    fig.subplots_adjust(left=0.22, right=0.97, top=0.93, bottom=0.12, hspace=0.08)
    return fig


def mpl_formal_compare(sims, names, labels,
                       sim_subset=None,
                       title='Formal Errors RSW',
                       figsize=None,
                       axes_fontsize=9,
                       tick_fontsize=8,
                       legend_fontsize=7,
                       title_fontsize=9):
    """3-row time-series of formal errors σ_R, σ_S, σ_W.

    Sized for half-textwidth placement in LaTeX (pair with mpl_rsw_compare).
    figsize default = (FIG_W_DOUBLE / 2, 7.0) ≈ (3.25, 7.0 in).
    """
    if sim_subset is not None:
        pairs  = [(n, labels[names.index(n)]) for n in sim_subset if n in sims and n in names]
        names  = [p[0] for p in pairs]
        labels = [p[1] for p in pairs]

    if figsize is None:
        figsize = (FIG_W_DOUBLE / 2, 5.8)

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    ylabels = [r'$\sigma_R$ [km]', r'$\sigma_S$ [km]', r'$\sigma_W$ [km]']
    comps = ['R', 'S', 'W']

    for i, (ax, ylab, comp) in enumerate(zip(axes, ylabels, comps)):
        for j, (sn, lbl) in enumerate(zip(names, labels)):
            if 'formal_errors_RSW_km' not in sims.get(sn, {}):
                continue
            fe    = sims[sn]['formal_errors_RSW_km']
            sha   = sims[sn].get('state_history_array')
            times = (convert_time_array_to_datetime(sha[:, 0])
                     if sha is not None else list(range(len(fe))))
            st    = compute_formal_error_statistics(fe)
            mx    = st[comp]['max']
            leg   = f"{lbl} (max: {mx:.1f} km)"
            col   = _SIM_COLORS.get(sn, _color(j))
            ls    = _SIM_LINESTYLE.get(sn, '-')
            alp   = _SIM_ALPHAS.get(sn, 1.0)
            ax.plot(times, fe[:, i],
                    color=col, linestyle=ls, linewidth=1.2, alpha=alp,
                    label=leg if i == 0 else '_nolegend_')

        ax.set_ylabel(ylab, fontsize=axes_fontsize, labelpad=2)
        ax.yaxis.set_label_coords(-0.13, 0.5)
        nbins = 8 if i == 2 else 5
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=nbins, prune=None))
        ax.tick_params(axis='y', labelsize=tick_fontsize, length=3)
        _apply_date_formatter(ax)

    for ax in axes[:-1]:
        _hide_xticklabels(ax)
    axes[0].set_title(title, fontsize=title_fontsize)
    axes[0].legend(loc='upper right', fontsize=legend_fontsize, handlelength=1.5)
    axes[-1].set_xlabel('Date', fontsize=axes_fontsize)
    axes[-1].tick_params(axis='x', labelsize=tick_fontsize, length=3)
    # Apply after all plotting so date formatters don't override margins.
    fig.subplots_adjust(left=0.22, right=0.97, top=0.93, bottom=0.12, hspace=0.08)
    return fig


def mpl_rsw_stats(sims, names, labels,
                  sim_subset=None,
                  show_formal=True,
                  show_diff=True,
                  fontsize_scale=1.0,
                  title='RSW Statistics',
                  figsize=None):
    """Grid of RSW statistics: cols = R / S / W.

    Row selection is controlled by show_diff and show_formal:
      show_diff=True,  show_formal=True  → 5×3 grid (diff Mean/RMS/Max + formal Max/RMS)
      show_diff=True,  show_formal=False → 3×3 grid (diff rows only)
      show_diff=False, show_formal=True  → 2×3 grid (formal rows only)

    fontsize_scale multiplies the default tick/label/annotation font sizes,
    making the figure easier to read when many simulations are compared.

    A line connects simulation points to show the trend across simulations."""
    if sim_subset is not None:
        pairs  = [(n, labels[names.index(n)]) for n in sim_subset if n in sims and n in names]
        names  = [p[0] for p in pairs]
        labels = [p[1] for p in pairs]

    # (row_label, data_source, stat_key, y_unit)
    diff_row_defs = [
        ('RSW diff — Mean',  'diff',   'mean', 'km'),
        ('RSW diff — RMS',   'diff',   'rms',  'km'),
        ('RSW diff — Max',   'diff',   'max',  'km'),
    ]
    formal_row_defs = [
        ('Formal σ — Max',   'formal', 'max',  'km'),
        ('Formal σ — RMS',   'formal', 'rms',  'km'),
    ]
    row_defs = []
    if show_diff:
        row_defs += diff_row_defs
    if show_formal:
        row_defs += formal_row_defs

    if not row_defs:
        print("WARNING: mpl_rsw_stats called with show_diff=False and show_formal=False — nothing to plot.")
        return None

    comps = ['R', 'S', 'W']
    x = np.arange(len(names))

    fs_tick   = max(6, int(8  * fontsize_scale))
    fs_annot  = max(5, int(7  * fontsize_scale))
    fs_ylabel = max(7, int(9  * fontsize_scale))
    fs_title  = max(9, int(11 * fontsize_scale))

    if figsize is None:
        figsize = (FIG_W_DOUBLE * 1.8, FIG_H_DEFAULT * len(row_defs) / 1.5)

    fig, axes = plt.subplots(len(row_defs), 3, figsize=figsize)
    if len(row_defs) == 1:
        axes = axes.reshape(1, -1)

    for ri, (row_label, src, stat_key, unit) in enumerate(row_defs):
        for ci, comp in enumerate(comps):
            ax = axes[ri, ci]
            col = RSW_COLORS[comp]

            vals = []
            for sn in names:
                sd = sims.get(sn, {})
                if src == 'diff' and 'diff_SPICE_RSW' in sd:
                    ds = compute_rsw_statistics(sd['diff_SPICE_RSW'])
                    vals.append(ds[comp][stat_key])
                elif src == 'formal' and 'formal_errors_RSW_km' in sd:
                    st = compute_formal_error_statistics(sd['formal_errors_RSW_km'])
                    vals.append(st[comp][stat_key])
                else:
                    vals.append(np.nan)

            # Connecting line through all non-NaN points
            valid_x = [xi for xi, v in enumerate(vals) if not np.isnan(v)]
            valid_v = [v  for v  in vals               if not np.isnan(v)]
            if len(valid_x) > 1:
                ax.plot(valid_x, valid_v, '-', color=col, linewidth=1.2,
                        alpha=0.5, zorder=1)

            # Dots + value annotations
            for xi, v in enumerate(vals):
                if not np.isnan(v):
                    ax.plot(xi, v, marker=_marker(names[xi]), color=col, markersize=6,
                            markeredgecolor='black', markeredgewidth=0.4,
                            linestyle='none', zorder=2)


            ax.set_xticks(x)
            ax.set_xlim(-0.5, len(names) - 0.5)
            if ri == len(row_defs) - 1:
                ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=fs_tick)
            else:
                ax.set_xticklabels([])
            if ci == 0:
                ax.set_ylabel(f'{row_label} [{unit}]', fontsize=fs_ylabel)
            if ri == 0:
                ax.set_title(comp, fontsize=fs_title)

    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


def mpl_rms_ratio(sims, names, labels,
                  use_group_colors=False,
                  title='RMS / Formal Error RMS'):
    """Dot plot of total RMS vs SPICE divided by total formal error RMS per simulation.

    use_group_colors is accepted for API consistency; colors are taken from
    _SIM_COLORS which already encodes group membership when set in the config.

    Total formal error RMS is computed as sqrt(mean(formal_errors_RSW_km ** 2))
    across all R/S/W components, matching the scalar nature of rms_SPICE.
    """
    ratios = []
    for sn in names:
        sd = sims.get(sn, {})
        rms_spice = sd.get('rms_SPICE', np.nan)
        if 'formal_errors_RSW_km' in sd and not np.isnan(rms_spice):
            formal_rms = np.sqrt(np.mean(sd['formal_errors_RSW_km'] ** 2))
            ratios.append(rms_spice / formal_rms if formal_rms > 0 else np.nan)
        else:
            ratios.append(np.nan)

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(FIG_W_DOUBLE, 0.8 * len(names)), FIG_H_DEFAULT))

    valid_x = [xi for xi, v in enumerate(ratios) if not np.isnan(v)]
    valid_v = [v  for v  in ratios               if not np.isnan(v)]
    if len(valid_x) > 1:
        ax.plot(valid_x, valid_v, '-', color='gray', linewidth=1.0, alpha=0.5, zorder=1)

    for i, (xi, v) in enumerate(zip(x, ratios)):
        if not np.isnan(v):
            col = (_group_color(names[i], i) if use_group_colors
                   else _SIM_COLORS.get(names[i], _color(i)))
            mk  = _marker(names[i])
            ax.plot(xi, v, marker=mk, color=col, markersize=8,
                    markeredgecolor='black', markeredgewidth=0.5,
                    linestyle='none', zorder=2)

    ax.axhline(1.0, color='gray', linewidth=0.8, linestyle='--', alpha=0.6,
               label='ratio = 1')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel('RMS$_{\\mathrm{SPICE}}$ / RMS$_{\\sigma}$  [—]')
    ax.set_title(title)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def mpl_rsw_ratio(sims, names, labels,
                  sim_subset=None,
                  title='RSW RMS / Formal σ RMS',
                  figsize=None,
                  axes_fontsize=None,
                  tick_fontsize=None,
                  panel_title_fontsize=None,
                  show_suptitle=True):
    """3-subplot dot plot: per-direction ratio of RSW diff RMS to formal error RMS.

    For each component c ∈ {R, S, W}:
        ratio_c = RMS(diff_SPICE_RSW[:, c]) / RMS(formal_errors_RSW_km[:, c])
    """
    if sim_subset is not None:
        pairs  = [(n, labels[names.index(n)]) for n in sim_subset if n in sims and n in names]
        names  = [p[0] for p in pairs]
        labels = [p[1] for p in pairs]

    comps = ['R', 'S', 'W']
    x = np.arange(len(names))

    if figsize is None:
        figsize = (FIG_W_DOUBLE * 1.6, FIG_H_DEFAULT)
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ci, (ax, comp) in enumerate(zip(axes, comps)):
        col = RSW_COLORS[comp]
        ratios = []
        for sn in names:
            sd = sims.get(sn, {})
            if 'diff_SPICE_RSW' in sd and 'formal_errors_RSW_km' in sd:
                diff_rms   = compute_rsw_statistics(sd['diff_SPICE_RSW'])[comp]['rms']
                formal_rms = compute_formal_error_statistics(sd['formal_errors_RSW_km'])[comp]['rms']
                ratios.append(diff_rms / formal_rms if formal_rms > 0 else np.nan)
            else:
                ratios.append(np.nan)

        valid_x = [xi for xi, v in enumerate(ratios) if not np.isnan(v)]
        valid_v = [v  for v  in ratios               if not np.isnan(v)]
        if len(valid_x) > 1:
            ax.plot(valid_x, valid_v, '-', color=col, linewidth=1.2, alpha=0.5, zorder=1)

        for xi, v in enumerate(ratios):
            if not np.isnan(v):
                ax.plot(xi, v, marker=_marker(names[xi]), color=col, markersize=6,
                        markeredgecolor='black', markeredgewidth=0.4,
                        linestyle='none', zorder=2)

        ax.axhline(1.0, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha='right',
                           fontsize=(tick_fontsize if tick_fontsize is not None else 8))
        ax.set_title(comp,
                     fontsize=(panel_title_fontsize
                               if panel_title_fontsize is not None else 11))
        ax.set_xlim(-0.5, len(names) - 0.5)
        if tick_fontsize is not None:
            ax.tick_params(axis='y', labelsize=tick_fontsize)
        if ci == 0:
            ax.set_ylabel('RMS$_{\\mathrm{SPICE}}$ / RMS$_{\\sigma}$  [—]',
                          **(({'fontsize': axes_fontsize}) if axes_fontsize is not None else {}))

    if show_suptitle and title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def mpl_gof(sims, names, labels,
            metric='rms',
            show_initial=False,
            log_y=False,
            title=None):
    """Goodness-of-fit: WRMS / RMS / cost-function comparison across sims."""
    wm = compute_wrms_and_cost(sims, names)
    if not wm:
        print("WARNING: No WRMS/RMS data, skipping GoF figure.")
        return None

    metric_map = {
        'wrms': ('final_wrms_combined_method1_mas',
                 'initial_wrms_combined_method1_mas',
                 'WRMS [mas]'),
        'rms':  ('final_rms_combined_mas',
                 'initial_rms_combined_mas',
                 'RMS [mas]'),
        'cost': ('final_cost_function',
                 'initial_cost_function',
                 'Cost Function'),
    }
    fk, ik, ylabel = metric_map.get(metric, metric_map['rms'])

    avs      = [n for n in names if n in wm]
    avlabels = [labels[names.index(n)] for n in avs]
    fv       = [wm[n].get(fk) for n in avs]
    iv       = [wm[n].get(ik) for n in avs]
    x        = np.arange(len(avs))

    if title is None:
        title = f'{ylabel} Comparison'

    fig, ax = plt.subplots(
        figsize=(max(FIG_W_DOUBLE, 0.8 * len(avs)), FIG_H_DEFAULT))

    # Connecting lines (neutral color, behind markers)
    valid_f = [(xi, v) for xi, v in enumerate(fv) if v is not None]
    if len(valid_f) > 1:
        ax.plot([p[0] for p in valid_f], [p[1] for p in valid_f],
                '-', color='steelblue', linewidth=1.0, alpha=0.4, zorder=1)
    if show_initial:
        valid_i = [(xi, v) for xi, v in enumerate(iv) if v is not None]
        if len(valid_i) > 1:
            ax.plot([p[0] for p in valid_i], [p[1] for p in valid_i],
                    '--', color='lightcoral', linewidth=1.0, alpha=0.4, zorder=1)

    # Per-sim markers
    for xi, sn in enumerate(avs):
        col = _SIM_COLORS.get(sn, _color(xi))
        mk  = _marker(sn)
        if fv[xi] is not None:
            ax.plot(xi, fv[xi], marker=mk, color=col, markersize=7,
                    markeredgecolor='black', markeredgewidth=0.5,
                    linestyle='none', zorder=2, label=avlabels[xi])
        if show_initial and iv[xi] is not None:
            ax.plot(xi, iv[xi], marker=mk, color=col, markersize=5,
                    markeredgecolor='black', markeredgewidth=0.5,
                    linestyle='none', zorder=2, alpha=0.45,
                    markerfacecolor='none')

    # Manual legend entries for initial/final if show_initial
    if show_initial:
        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Line2D([0], [0], linestyle='-',  color='steelblue',   linewidth=1.2, label='Final'),
            Line2D([0], [0], linestyle='--', color='lightcoral',  linewidth=1.2, label='Initial'),
        ])

    ax.set_xticks(x)
    ax.set_xticklabels(avlabels, rotation=30, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_y:
        ax.set_yscale('log')

    fig.tight_layout()
    return fig


def mpl_gof_combined(sims, names, labels,
                     show_initial=False,
                     log_y=False,
                     thick_lines=False,
                     use_group_colors=False,
                     title='Goodness of Fit Comparison',
                     figsize=None):
    """1×3 subplots: WRMS [mas], RMS [mas], Cost Function — all simulations.

    Parameters
    ----------
    thick_lines : bool
        If True, connecting lines are drawn thicker (lw=2.0, alpha=0.65) so the
        initial-value line remains visible when it overlaps with the final line.
    use_group_colors : bool
        If True, the show_initial legend is replaced with a two-entry group
        legend (IAUPole / FitPole) instead of one entry per simulation.
        Colors are still taken from _SIM_COLORS (which should already encode
        group membership in the config's SIM_COLORS dict).
    """
    wm = compute_wrms_and_cost(sims, names)
    if not wm:
        print("WARNING: No WRMS/RMS data, skipping GoF combined figure.")
        return None

    avs      = [n for n in names if n in wm]
    avlabels = [labels[names.index(n)] for n in avs]
    x        = np.arange(len(avs))

    metrics = [
        ('final_wrms_combined_method1_mas', 'initial_wrms_combined_method1_mas', 'WRMS [mas]'),
        ('final_rms_combined_mas',          'initial_rms_combined_mas',          'RMS [mas]'),
        ('final_cost_function',             'initial_cost_function',             'Cost Function'),
    ]

    line_lw    = 2.0 if thick_lines else 1.0
    line_alpha = 0.65 if thick_lines else 0.4

    if figsize is None:
        figsize = (FIG_W_DOUBLE * 1.4, FIG_H_DEFAULT * len(metrics) / 1.5)

    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize, sharex=True)

    for i, (ax, (fk, ik, ylabel)) in enumerate(zip(axes, metrics)):
        fv = [wm[n].get(fk) for n in avs]
        iv = [wm[n].get(ik) for n in avs]

        # Connecting lines (neutral, behind markers)
        valid_f = [(xi, v) for xi, v in enumerate(fv) if v is not None]
        if len(valid_f) > 1:
            ax.plot([p[0] for p in valid_f], [p[1] for p in valid_f],
                    '-', color='steelblue', linewidth=line_lw, alpha=line_alpha, zorder=1)
        if show_initial:
            valid_i = [(xi, v) for xi, v in enumerate(iv) if v is not None]
            if len(valid_i) > 1:
                ax.plot([p[0] for p in valid_i], [p[1] for p in valid_i],
                        '--', color='lightcoral', linewidth=line_lw, alpha=line_alpha, zorder=1)

        # Per-sim markers
        for xi, sn in enumerate(avs):
            col = (_group_color(sn, xi) if use_group_colors
                   else _SIM_COLORS.get(sn, _color(xi)))
            mk  = _marker(sn)
            if fv[xi] is not None:
                ax.plot(xi, fv[xi], marker=mk, color=col, markersize=7,
                        markeredgecolor='black', markeredgewidth=0.5,
                        linestyle='none', zorder=2)
            if show_initial and iv[xi] is not None:
                ax.plot(xi, iv[xi], marker=mk, color=col, markersize=5,
                        markeredgecolor='black', markeredgewidth=0.5,
                        linestyle='none', zorder=2, alpha=0.45,
                        markerfacecolor='none')

        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontsize=10)
        if log_y:
            ax.set_yscale('log')

    # x-tick labels only on bottom subplot
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(avlabels, rotation=30, ha='right')

    if show_initial:
        from matplotlib.lines import Line2D
        if use_group_colors:
            # Build a group legend: one entry per unique group color + marker
            seen_groups = {}
            for sn in avs:
                col = _group_color(sn)
                mk  = _marker(sn)
                key = (col, mk)
                if key not in seen_groups:
                    grp_lbl = ('IAUPole' if sn.startswith('IAUPole')
                               else 'FitPole' if sn.startswith(('FitPole', 'SimPole'))
                               else sn.split('_')[0])
                    seen_groups[key] = grp_lbl
            group_handles = [
                Line2D([0], [0], color=col, marker=mk, linestyle='-',
                       linewidth=1.5, markersize=7,
                       markeredgecolor='black', markeredgewidth=0.4,
                       label=lbl)
                for (col, mk), lbl in seen_groups.items()
            ]
            group_handles += [
                Line2D([0], [0], linestyle='-',  color='steelblue',  linewidth=line_lw, label='Final'),
                Line2D([0], [0], linestyle='--', color='lightcoral', linewidth=line_lw, label='Initial'),
            ]
            axes[0].legend(handles=group_handles, fontsize=9)
        else:
            axes[0].legend(handles=[
                Line2D([0], [0], linestyle='-',  color='steelblue',  linewidth=line_lw, label='Final'),
                Line2D([0], [0], linestyle='--', color='lightcoral', linewidth=line_lw, label='Initial (hollow)'),
            ], fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def mpl_corr_heatmap(sims, sim_name,
                     title=None, figsize=None, cmap='YlOrRd'):
    """Absolute correlation matrix heatmap for a single simulation.

    Parameters
    ----------
    cmap : str
        Matplotlib colormap name.  Default 'YlOrRd' (yellow→red, colorblind-safe).
        Use 'Blues', 'viridis', or any valid colormap string.
    """
    if 'correlations' not in sims.get(sim_name, {}):
        print(f"WARNING: No correlation data for '{sim_name}', skipping.")
        return None

    cm     = np.abs(sims[sim_name]['correlations'])
    labels = get_parameter_labels(sims[sim_name].get('est_parameters', []))
    n      = len(labels)

    if figsize is None:
        side    = max(3.5, 0.55 * n)
        figsize = (side, side)
    if title is None:
        title = f'|Correlations|: {sim_name}'

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    fig.colorbar(im, ax=ax, label='|Correlation|', shrink=0.8)

    fontsize = max(5, 9 - n // 3)
    for i in range(n):
        for j in range(n):
            v = cm[i, j]
            ax.text(j, i, f'{v:.2f}',
                    ha='center', va='center',
                    fontsize=fontsize,
                    color='white' if v > 0.6 else 'black')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def write_condition_numbers(sims, names, out_path):
    """Compute condition numbers of correlation matrices and write to a text file.

    Parameters
    ----------
    sims : dict
        Simulation data dict.
    names : list of str
        Simulation names to include.
    out_path : str
        Full path for the output text file.
    """
    import datetime
    lines = [
        'Condition Numbers of Correlation Matrices',
        f'Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '',
        f'{"Simulation":<50}  {"Condition Number":>18}',
        '-' * 70,
    ]
    for name in names:
        sim = sims.get(name, {})
        corr = sim.get('correlations')
        if corr is None:
            lines.append(f'{name:<50}  {"N/A (no data)":>18}')
        else:
            cond = np.linalg.cond(corr)
            lines.append(f'{name:<50}  {cond:>18.6e}')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  Saved: {out_path}')


def mpl_residual_histogram(sims, names, labels,
                           iteration='final',
                           fit_gauss=True,
                           bins=50,
                           figsize=None,
                           title='Residual Histogram'):
    """RA / Dec residual histograms with optional Gaussian fit."""
    if figsize is None:
        figsize = (FIG_W_DOUBLE, FIG_H_DEFAULT)

    col_ra  = ('ra_residual_final_mas'   if iteration == 'final'
               else 'ra_residual_initial_mas')
    col_dec = ('dec_residual_final_mas'  if iteration == 'final'
               else 'dec_residual_initial_mas')

    fig, (ax_ra, ax_dec) = plt.subplots(1, 2, figsize=figsize)

    for j, (sn, lbl) in enumerate(zip(names, labels)):
        if 'residual_df' not in sims.get(sn, {}):
            continue
        df  = sims[sn]['residual_df']
        ra  = df[col_ra].dropna().values
        dec = df[col_dec].dropna().values

        ax_ra.hist(ra,  bins=bins, alpha=0.5,
                   color=_color(j), label=lbl, density=True)
        ax_dec.hist(dec, bins=bins, alpha=0.5,
                    color=_color(j), density=True)

        if fit_gauss:
            for ax, data in [(ax_ra, ra), (ax_dec, dec)]:
                mu, sigma = sp_stats.norm.fit(data)
                xr = np.linspace(data.min(), data.max(), 200)
                ax.plot(xr, sp_stats.norm.pdf(xr, mu, sigma),
                        color=_color(j), linewidth=1.5, linestyle='--')

    ax_ra.set_xlabel('RA Residual [mas]')
    ax_dec.set_xlabel('Dec Residual [mas]')
    ax_ra.set_ylabel('Density')
    ax_ra.set_title('RA Residuals')
    ax_dec.set_title('Dec Residuals')
    ax_ra.legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def mpl_residual_timeseries(sims, names, labels,
                            sim_subset=None,
                            title='Observation Residuals RA/DEC',
                            figsize=None):
    """Final-iteration RA / Dec residuals [mas] as a scatter, coloured by ref_point_id.

    One figure per simulation in sim_subset.  Layout: 2-row plot + right legend
    panel (GridSpec width_ratios=[3, 1]) — legend as a single column outside the
    plot area.
    """
    from matplotlib.lines import Line2D

    if figsize is None:
        figsize = (FIG_W_DOUBLE * 2.0, 6.0)

    targets = sim_subset if sim_subset is not None else names
    figs = []

    for sn in targets:
        if sn not in sims:
            print(f"  SKIP residual_timeseries for '{sn}': not in sims.")
            continue
        sd = sims[sn]
        if 'residual_df' not in sd:
            print(f"  SKIP residual_timeseries for '{sn}': no residual_df.")
            continue
        df  = sd['residual_df']
        lbl = labels[names.index(sn)] if sn in names else sn

        fig = plt.figure(figsize=figsize)
        gs  = fig.add_gridspec(2, 2, width_ratios=[3, 1],
                               hspace=0.08, wspace=0.04)
        ax_ra  = fig.add_subplot(gs[0, 0])
        ax_dec = fig.add_subplot(gs[1, 0], sharex=ax_ra)
        ax_leg = fig.add_subplot(gs[:, 1])
        ax_leg.axis('off')

        ref_ids = sorted(df['ref_point_id'].unique())
        legend_handles = []
        for i, rid in enumerate(ref_ids):
            mask = df['ref_point_id'] == rid
            t    = df.loc[mask, 'datetime']
            ra   = df.loc[mask, 'ra_residual_final_mas']
            dec  = df.loc[mask, 'dec_residual_final_mas']
            col  = _color(i)
            ax_ra.scatter(t,  ra,  s=4, color=col, alpha=0.5, zorder=2)
            ax_dec.scatter(t, dec, s=4, color=col, alpha=0.5, zorder=2)
            legend_handles.append(
                Line2D([0], [0], linestyle='none', marker='o', color=col,
                       markersize=5, label=str(rid))
            )

        for ax in (ax_ra, ax_dec):
            ax.axhline(0, color='k', linewidth=0.8, linestyle='--')

        ax_ra.set_ylabel('RA Residual [mas]')
        ax_dec.set_ylabel('Dec Residual [mas]')
        ax_dec.set_xlabel('Date')
        plt.setp(ax_ra.get_xticklabels(), visible=False)
        _apply_date_formatter(ax_dec)

        ax_leg.legend(legend_handles, [h.get_label() for h in legend_handles],
                      title='Obs. file ID', loc='center left', fontsize=7,
                      ncol=1, markerscale=1.5, frameon=True, borderaxespad=0.2)

        fig.suptitle(f'{title} — {lbl}')
        figs.append(fig)

    if not figs:
        return None
    return figs[0] if len(figs) == 1 else figs


def mpl_residual_timeseries_by_id(sims, names, labels,
                                   sim_subset=None,
                                   title='Observation Residuals RA/Dec',
                                   figsize=None):
    """Final-iteration RA / Dec residuals [arcsec] scatter, coloured by ref_point_id.

    Legend placed in a right panel (2 × 2 GridSpec).  One figure per simulation.
    Returns a list of (fig, sim_label) tuples so each is saved as a separate PDF.
    """
    from matplotlib.lines import Line2D

    if figsize is None:
        figsize = (FIG_W_DOUBLE * 2.0, 6.5)

    targets = sim_subset if sim_subset is not None else names
    figs = []

    for sn in targets:
        if sn not in sims:
            print(f"  SKIP residual_timeseries_by_id for '{sn}': not in sims.")
            continue
        sd = sims[sn]
        if 'residual_df' not in sd:
            print(f"  SKIP residual_timeseries_by_id for '{sn}': no residual_df.")
            continue
        df    = sd['residual_df']
        lbl   = labels[names.index(sn)] if sn in names else sn

        fig = plt.figure(figsize=figsize)
        gs  = fig.add_gridspec(2, 2, width_ratios=[3, 1],
                               hspace=0.08, wspace=0.04)
        ax_ra  = fig.add_subplot(gs[0, 0])
        ax_dec = fig.add_subplot(gs[1, 0], sharex=ax_ra)
        ax_leg = fig.add_subplot(gs[:, 1])
        ax_leg.axis('off')

        ref_ids = sorted(df['ref_point_id'].unique())
        legend_handles = []
        for i, rid in enumerate(ref_ids):
            mask    = df['ref_point_id'] == rid
            t       = df.loc[mask, 'datetime']
            ra_as   = df.loc[mask, 'ra_residual_final_mas'] / 1000.0   # mas → arcsec
            dec_as  = df.loc[mask, 'dec_residual_final_mas'] / 1000.0
            col     = _obs_color(i)
            lbl_rid = rid.replace('_', ' ')
            ax_ra.scatter(t,  ra_as,  s=5, color=col, alpha=0.65, zorder=2)
            ax_dec.scatter(t, dec_as, s=5, color=col, alpha=0.65, zorder=2)
            legend_handles.append(
                Line2D([0], [0], linestyle='none', marker='o', color=col,
                       markersize=5, label=lbl_rid)
            )

        for ax in (ax_ra, ax_dec):
            ax.axhline(0, color='k', linewidth=0.8, linestyle='--')

        ax_ra.set_ylabel("RA residual [$''$]")
        ax_dec.set_ylabel("Dec residual [$''$]")
        ax_dec.set_xlabel('Year')
        plt.setp(ax_ra.get_xticklabels(), visible=False)
        _apply_date_formatter(ax_dec)

        ax_leg.legend(legend_handles, [h.get_label() for h in legend_handles],
                      loc='center left', fontsize=8, ncol=1,
                      markerscale=1.5, frameon=True, borderaxespad=0.2)

        # Sanitise lbl for use as file label (used in dispatcher)
        safe = lbl.replace(' ', '_').replace('/', '-').replace('+', 'p').replace('.', '')
        fig.suptitle(f'{title} — {lbl}')
        figs.append((fig, safe))

    if not figs:
        return None
    return figs


def mpl_residual_histogram_per_sim(sims, names, labels,
                                    sim_subset=None,
                                    bins=50,
                                    fit_gauss=True,
                                    title='Residual Histogram',
                                    figsize=None):
    """RA / Dec residual histograms [arcsec] — one figure per simulation.

    Returns a list of (fig, sim_label) tuples so each is saved as a separate PDF.
    """
    if figsize is None:
        figsize = (FIG_W_DOUBLE, FIG_H_DEFAULT)

    targets = sim_subset if sim_subset is not None else names
    figs = []

    for sn in targets:
        if sn not in sims:
            print(f"  SKIP residual_histogram_per_sim for '{sn}': not in sims.")
            continue
        sd = sims[sn]
        if 'residual_df' not in sd:
            print(f"  SKIP residual_histogram_per_sim for '{sn}': no residual_df.")
            continue
        df  = sd['residual_df']
        lbl = labels[names.index(sn)] if sn in names else sn

        ra_as  = (df['ra_residual_final_mas'].dropna() / 1000.0).values
        dec_as = (df['dec_residual_final_mas'].dropna() / 1000.0).values

        fig, (ax_ra, ax_dec) = plt.subplots(1, 2, figsize=figsize)

        for ax, data, xlabel in [
            (ax_ra,  ra_as,  "RA residual [$''$]"),
            (ax_dec, dec_as, "Dec residual [$''$]"),
        ]:
            data = data[np.isfinite(data)]
            if len(data) == 0:
                continue
            ax.hist(data, bins=bins, alpha=0.6, color='#1f77b4', density=True)
            if fit_gauss and len(data) >= 5:
                mu, sigma = sp_stats.norm.fit(data)
                xr = np.linspace(data.min(), data.max(), 300)
                ax.plot(xr, sp_stats.norm.pdf(xr, mu, sigma),
                        color='#d62728', linewidth=1.6, linestyle='--',
                        label=f'μ={mu:.4f}''"'', σ={sigma:.4f}''"')
                ax.legend(fontsize=8)
            ax.axvline(0, color='k', linewidth=0.8, linestyle=':')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Density')

        ax_ra.set_title('RA')
        ax_dec.set_title('Dec')
        safe = lbl.replace(' ', '_').replace('/', '-').replace('+', 'p').replace('.', '')
        fig.suptitle(f'{title} — {lbl}')
        fig.tight_layout()
        figs.append((fig, safe))

    if not figs:
        return None
    return figs


# ============================================================================
# PARAMETER UPDATE HELPERS
# ============================================================================

def _get_iau_reference(sims, all_names):
    """Build reference dicts from IAUPole_pole_pos_cov_pole_lib_cov initial values.

    This sim always estimates [initial_state, iau_rotation_model_pole,
    iau_rotation_model_pole_librations] in that fixed order, giving a complete
    reference for all parameter types (state + pole pos + pole lib).

    For FitPole sims that do not estimate all parameters, _param_update falls
    back to the sim's own initial value for any label not present here.

    Returns (iau_ref_inertial, iau_ref_rsw) — both are dicts mapping
    parameter label → initial value.  Returns empty dicts if the reference sim
    is not found or has no parameter data.
    """
    REF_SIM = 'IAUPole_pole_pos_cov_pole_lib_cov'
    iau_ref_inertial = {}
    iau_ref_rsw      = {}
    rsw_map = {'X': 'R', 'Y': 'S', 'Z': 'W', 'VX': 'VR', 'VY': 'VS', 'VZ': 'VW'}

    sd = sims.get(REF_SIM, {})
    if 'parameter_history' not in sd or 'est_parameters' not in sd:
        print(f"WARNING: Reference sim '{REF_SIM}' not found or has no parameter data. "
              "Parameter update plots will use each sim's own initial values as reference.")
        return iau_ref_inertial, iau_ref_rsw

    ph_in    = sd['parameter_history']
    ph_rsw   = sd.get('parameter_history_RSW', ph_in)
    lbls, _, _ = get_parameter_info(sd['est_parameters'])
    rsw_lbls = [rsw_map.get(l, l) for l in lbls]

    for i, lbl in enumerate(lbls):
        iau_ref_inertial[lbl] = ph_in[i, 0]
    for i, lbl in enumerate(rsw_lbls):
        if lbl not in iau_ref_rsw:
            iau_ref_rsw[lbl] = ph_rsw[i, 0]

    return iau_ref_inertial, iau_ref_rsw


def _param_update(sim_data, group_filter, scale=1.0, iau_ref=None):
    """Extract the final–initial parameter update for a given group.

    Returns (values, labels, units) filtered to `group_filter` (list of group
    names).  `scale` is applied to every value (e.g. rad→deg).
    When `iau_ref` is provided (dict label→value) the reference is taken from
    that dict instead of the sim's own initial, falling back to own initial for
    any label not present in iau_ref.
    Returns (None, None, None) if the sim has no parameter_history.
    """
    if 'parameter_history' not in sim_data or 'est_parameters' not in sim_data:
        return None, None, None
    ph = sim_data['parameter_history']
    labels, groups, units = get_parameter_info(sim_data['est_parameters'])
    sel_v, sel_l, sel_u = [], [], []
    for i, (lbl, grp, unt) in enumerate(zip(labels, groups, units)):
        if grp in group_filter:
            ref = iau_ref.get(lbl, ph[i, 0]) if iau_ref is not None else ph[i, 0]
            sel_v.append((ph[i, -1] - ref) * scale)
            sel_l.append(lbl)
            sel_u.append(unt)
    return np.array(sel_v) if sel_v else None, sel_l, sel_u


def _param_update_rsw(sim_data, group_filter, scale=1.0, iau_ref=None):
    """Same as _param_update but prefers parameter_history_RSW if available.

    `iau_ref` should use RSW-remapped labels (R/S/W/VR/VS/VW) when provided.
    """
    if 'parameter_history' not in sim_data or 'est_parameters' not in sim_data:
        return None, None, None
    if 'parameter_history_RSW' in sim_data:
        ph      = sim_data['parameter_history_RSW']
        rsw_map = {'X': 'R', 'Y': 'S', 'Z': 'W', 'VX': 'VR', 'VY': 'VS', 'VZ': 'VW'}
    else:
        ph      = sim_data['parameter_history']
        rsw_map = {}
    labels, groups, units = get_parameter_info(sim_data['est_parameters'])
    labels = [rsw_map.get(l, l) for l in labels]
    sel_v, sel_l, sel_u = [], [], []
    for i, (lbl, grp, unt) in enumerate(zip(labels, groups, units)):
        if grp in group_filter:
            ref = iau_ref.get(lbl, ph[i, 0]) if iau_ref is not None else ph[i, 0]
            sel_v.append((ph[i, -1] - ref) * scale)
            sel_l.append(lbl)
            sel_u.append(unt)
    return np.array(sel_v) if sel_v else None, sel_l, sel_u


# ── Shared bar-chart helper ───────────────────────────────────────────────────

def _bar_ax(ax, vals, bar_color, x_labels, unit_str, param_title,
            show_xticks=True, fontsize_annot=7, show_annot=True, fontsize_tick=8):
    """Draw a bar chart on `ax` with value annotations above/below each bar."""
    x = np.arange(len(vals))
    bars = ax.bar(x, vals,
                  color=bar_color, edgecolor='black', linewidth=0.5, alpha=0.85)
    if show_annot:
        for bar, v in zip(bars, vals):
            if np.isnan(v):
                continue
            if v >= 0:
                ypos = bar.get_y() + bar.get_height()   # tip of positive bar
                va_  = 'bottom'                          # text goes above tip
            else:
                ypos = bar.get_y()                       # tip of negative bar
                va_  = 'top'                             # text hangs below tip
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_title(f'Δ{param_title}  [{unit_str}]', fontsize=10)
    ax.set_xticks(x)
    if show_xticks:
        ax.set_xticklabels(x_labels, rotation=35, ha='right', fontsize=fontsize_tick)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=0)


# ── PDF 1: Total position and velocity magnitude update ───────────────────────

_RAD_TO_DEG = 180.0 / np.pi

def mpl_param_state(sims, names, labels,
                    title='Initial State Update — Total Magnitude',
                    figsize=None,
                    variant='iau',
                    sim_subset=None,
                    show_annot=True,
                    show_suptitle=True,
                    tick_fontsize=8):
    """1×2 bar chart: |Δpos| [km] and |Δvel| [km/s] across simulations."""
    iau_prefix = 'IAUPole'
    fit_prefix = 'FitPole'
    iau_ref_in, _ = _get_iau_reference(sims, names)

    if variant == 'iau':
        f_names      = [n for n in names if n.startswith(iau_prefix)]
        ref_map      = {n: None for n in f_names}
        title_suffix = ' — IAU'
    elif variant in ('fitpole', 'simpole'):
        f_names      = [n for n in names if not n.startswith(iau_prefix)]
        ref_map      = {n: None for n in f_names}
        title_suffix = f' — {fit_prefix}'
    else:  # combined
        f_names      = list(names)
        ref_map      = {n: (iau_ref_in if not n.startswith(iau_prefix) else None)
                        for n in f_names}
        title_suffix = f' — Combined ({fit_prefix} vs IAU ref)'

    if sim_subset is not None:
        f_names = [n for n in f_names if n in sim_subset]
        ref_map = {n: ref_map[n] for n in f_names}

    f_labels = [labels[names.index(n)] for n in f_names]
    _display_title = show_suptitle and bool(title.strip())
    title    = title + title_suffix

    if figsize is None:
        figsize = (FIG_W_DOUBLE * 1.4, FIG_H_DEFAULT)

    pos_mags, vel_mags = [], []
    for sn in f_names:
        vals, plabels, _ = _param_update(
            sims.get(sn, {}), ['Position', 'Velocity'], iau_ref=ref_map[sn])
        if vals is None:
            pos_mags.append(np.nan)
            vel_mags.append(np.nan)
            continue
        pos_idx = [i for i, l in enumerate(plabels) if l in ('X', 'Y', 'Z')]
        vel_idx = [i for i, l in enumerate(plabels) if l in ('VX', 'VY', 'VZ')]
        pos_mags.append(np.linalg.norm(vals[pos_idx]) / 1000.0 if pos_idx else np.nan)
        vel_mags.append(np.linalg.norm(vals[vel_idx]) / 1000.0 if vel_idx else np.nan)

    fig, (ax_p, ax_v) = plt.subplots(1, 2, figsize=figsize)
    _bar_ax(ax_p, pos_mags, _color(0), f_labels, 'km',   '|pos|', show_annot=show_annot, fontsize_tick=tick_fontsize)
    _bar_ax(ax_v, vel_mags, _color(1), f_labels, 'km/s', '|vel|', show_annot=show_annot, fontsize_tick=tick_fontsize)
    if _display_title:
        fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


# ── PDF 2: RSW position and velocity update ───────────────────────────────────

def mpl_param_rsw(sims, names, labels,
                  title='Initial State Update — RSW',
                  figsize=None,
                  variant='iau',
                  sim_subset=None,
                  show_annot=True):
    """2×3 bar-chart grid: ΔR/ΔS/ΔW (top) and ΔVR/ΔVS/ΔVW (bottom) [km, km/s]."""
    iau_prefix = 'IAUPole'
    fit_prefix = 'FitPole'
    _, iau_ref_rsw = _get_iau_reference(sims, names)

    if variant == 'iau':
        f_names      = [n for n in names if n.startswith(iau_prefix)]
        ref_map      = {n: None for n in f_names}
        title_suffix = ' — IAU'
    elif variant in ('fitpole', 'simpole'):
        f_names      = [n for n in names if not n.startswith(iau_prefix)]
        ref_map      = {n: None for n in f_names}
        title_suffix = f' — {fit_prefix}'
    else:  # combined
        f_names      = list(names)
        ref_map      = {n: (iau_ref_rsw if not n.startswith(iau_prefix) else None)
                        for n in f_names}
        title_suffix = f' — Combined ({fit_prefix} vs IAU ref)'

    if sim_subset is not None:
        f_names = [n for n in f_names if n in sim_subset]
        ref_map = {n: ref_map[n] for n in f_names}

    f_labels   = [labels[names.index(n)] for n in f_names]
    _display_title = bool(title.strip())
    title      = title + title_suffix
    pos_labels = ['R',  'S',  'W' ]
    vel_labels = ['VR', 'VS', 'VW']

    def _vals_for(comp_label):
        base = comp_label.lstrip('V')
        col  = RSW_COLORS.get(base, _color(0))
        row  = []
        for sn in f_names:
            v_all, pl, _ = _param_update_rsw(
                sims.get(sn, {}), ['Position', 'Velocity'], iau_ref=ref_map[sn])
            if v_all is None or comp_label not in pl:
                row.append(np.nan)
            else:
                row.append(v_all[pl.index(comp_label)] / 1000.0)
        return row, col

    if figsize is None:
        figsize = (FIG_W_DOUBLE * 1.5, FIG_H_DEFAULT * 1.8)
    fig, axes = plt.subplots(2, 3, figsize=figsize, squeeze=False)

    for ci, (pl, vl) in enumerate(zip(pos_labels, vel_labels)):
        top_vals, top_col = _vals_for(pl)
        bot_vals, bot_col = _vals_for(vl)
        _bar_ax(axes[0, ci], top_vals, top_col, f_labels, 'km',   pl,  show_xticks=False, show_annot=show_annot)
        _bar_ax(axes[1, ci], bot_vals, bot_col, f_labels, 'km/s', vl,  show_xticks=True,  show_annot=show_annot)

    if _display_title:
        fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


# ── PDF 3: Pole position update ───────────────────────────────────────────────

def mpl_param_pole_pos(sims, names, labels,
                       title='Pole Position Update  (α₀, δ₀)',
                       figsize=None,
                       variant='iau',
                       sim_subset=None,
                       show_annot=True):
    """1×2 bar chart: Δα₀ and Δδ₀ across simulations [deg]."""
    iau_prefix = 'IAUPole'
    fit_prefix = 'FitPole'
    iau_ref_in, _ = _get_iau_reference(sims, names)

    if variant == 'iau':
        base_names   = [n for n in names if n.startswith(iau_prefix)]
        ref_map      = {n: None for n in base_names}
        title_suffix = ' — IAU'
    elif variant in ('fitpole', 'simpole'):
        base_names   = [n for n in names if not n.startswith(iau_prefix)]
        ref_map      = {n: None for n in base_names}
        title_suffix = f' — {fit_prefix}'
    else:  # combined
        base_names   = list(names)
        ref_map      = {n: (iau_ref_in if not n.startswith(iau_prefix) else None)
                        for n in base_names}
        title_suffix = f' — Combined ({fit_prefix} vs IAU ref)'

    if sim_subset is not None:
        base_names = [n for n in base_names if n in sim_subset]
        ref_map    = {n: ref_map[n] for n in base_names if n in ref_map}
        title_suffix = ''

    _display_title = bool(title.strip())
    title = title + title_suffix

    # Keep only sims (within variant filter) that estimate pole position.
    filt = [(sn, labels[names.index(sn)]) for sn in base_names
            if _param_update(sims.get(sn, {}), ['Pole Position'])[0] is not None]
    f_names, f_labels = (zip(*filt) if filt else ([], []))

    if figsize is None:
        figsize = (FIG_W_DOUBLE * 1.2, FIG_H_DEFAULT)

    fig, (ax_a, ax_d) = plt.subplots(1, 2, figsize=figsize)
    pole_colors = {'α₀': '#9467bd', 'δ₀': '#e377c2'}

    for ax, plabel in zip((ax_a, ax_d), ('α₀', 'δ₀')):
        vals = []
        for sn in f_names:
            v_all, pl, _ = _param_update(
                sims.get(sn, {}), ['Pole Position'],
                scale=_RAD_TO_DEG, iau_ref=ref_map[sn])
            vals.append(v_all[pl.index(plabel)] if plabel in pl else np.nan)
        _bar_ax(ax, vals, pole_colors[plabel], f_labels, 'deg', plabel, fontsize_annot=8, show_annot=show_annot)

    if _display_title:
        fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


# ── PDF 4: Pole libration update ──────────────────────────────────────────────

def mpl_param_pole_lib(sims, names, labels,
                       title='Pole Libration Update  (α₁, δ₁)',
                       figsize=None,
                       variant='iau',
                       sim_subset=None,
                       show_annot=True):
    """1×2 bar chart: Δα₁ and Δδ₁ across simulations [deg]."""
    iau_prefix = 'IAUPole'
    fit_prefix = 'FitPole'
    iau_ref_in, _ = _get_iau_reference(sims, names)

    if variant == 'iau':
        base_names   = [n for n in names if n.startswith(iau_prefix)]
        ref_map      = {n: None for n in base_names}
        title_suffix = ' — IAU'
    elif variant in ('fitpole', 'simpole'):
        base_names   = [n for n in names if not n.startswith(iau_prefix)]
        ref_map      = {n: None for n in base_names}
        title_suffix = f' — {fit_prefix}'
    else:  # combined
        base_names   = list(names)
        ref_map      = {n: (iau_ref_in if not n.startswith(iau_prefix) else None)
                        for n in base_names}
        title_suffix = f' — Combined ({fit_prefix} vs IAU ref)'

    if sim_subset is not None:
        base_names = [n for n in base_names if n in sim_subset]
        ref_map    = {n: ref_map[n] for n in base_names if n in ref_map}
        title_suffix = ''

    _display_title = bool(title.strip())
    title = title + title_suffix

    # Keep only sims (within variant filter) that estimate pole librations.
    filt = [(sn, labels[names.index(sn)]) for sn in base_names
            if _param_update(sims.get(sn, {}), ['Pole Librations'])[0] is not None]
    f_names, f_labels = (zip(*filt) if filt else ([], []))

    if figsize is None:
        figsize = (FIG_W_DOUBLE * 1.2, FIG_H_DEFAULT)

    fig, (ax_a, ax_d) = plt.subplots(1, 2, figsize=figsize)
    lib_colors = {'α₁': '#17becf', 'δ₁': '#bcbd22'}

    for ax, plabel in zip((ax_a, ax_d), ('α₁', 'δ₁')):
        vals = []
        for sn in f_names:
            v_all, pl, _ = _param_update(
                sims.get(sn, {}), ['Pole Librations'],
                scale=_RAD_TO_DEG, iau_ref=ref_map[sn])
            vals.append(v_all[pl.index(plabel)] if plabel in pl else np.nan)
        _bar_ax(ax, vals, lib_colors[plabel], f_labels, 'deg', plabel, fontsize_annot=8, show_annot=show_annot)

    if _display_title:
        fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


# ── RSW difference: full + zoomed subfigure ───────────────────────────────────

def _setup_zoom_xaxis(ax):
    """Apply auto date locator + '%d %b' formatter to a zoomed axis."""
    loc = mdates.AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=10)
    ax.tick_params(axis='x', labelsize=10, length=4)


def _hide_xticklabels(ax):
    """Hide tick labels (but keep tick marks) on a shared-x axis."""
    ax.tick_params(labelbottom=False, bottom=True)


def mpl_rsw_with_zoom(sims, names, labels,
                      sim_subset=None,
                      zoom_days=90,
                      title='RSW Difference vs NEP097',
                      figsize=None):
    """3×2 figure: left = full time range, right = first zoom_days of data.

    Uses _SIM_COLORS / _SIM_LINESTYLE (IAU = cool, FitPole = warm).
    X-tick labels only on bottom row; zoom column uses '%d %b' date format.
    """
    # ── resolve sim subset ───────────────────────────────────────────────────
    if sim_subset is not None:
        pairs = [
            (n, labels[names.index(n)] if n in names else n.replace('_cov', ''))
            for n in sim_subset if n in sims
        ]
    else:
        pairs = list(zip(names, labels))
    pairs = [(n, lb) for n, lb in pairs if 'diff_SPICE_RSW' in sims.get(n, {})]
    if not pairs:
        return None
    plot_names, plot_labels = zip(*pairs)

    # ── zoom window: earliest data point + zoom_days ─────────────────────────
    zoom_start = zoom_end = None
    for sn in plot_names:
        dr    = sims[sn]['diff_SPICE_RSW']
        times = get_rsw_times(sims[sn], n_points=len(dr))
        if times is not None and len(times) > 0:
            zoom_start = times[0]
            zoom_end   = zoom_start + timedelta(days=zoom_days)
            break

    # ── figure layout: 3 rows × 2 cols ───────────────────────────────────────
    if figsize is None:
        figsize = (FIG_W_DOUBLE * 2, 8.5)
    fig = plt.figure(figsize=figsize)
    gs  = GridSpec(3, 2, figure=fig,
                   hspace=0.08, wspace=0.30,
                   left=0.07, right=0.98, top=0.91, bottom=0.10)

    axes_full = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    axes_zoom = [fig.add_subplot(gs[i, 1]) for i in range(3)]
    for i in range(1, 3):
        axes_full[i].sharex(axes_full[0])
        axes_zoom[i].sharex(axes_zoom[0])

    ylabels = [r'$\Delta R$ [km]', r'$\Delta S$ [km]', r'$\Delta W$ [km]']

    # ── plot data ─────────────────────────────────────────────────────────────
    for axes, apply_zoom in [(axes_full, False), (axes_zoom, True)]:
        for row, (ax, ylab) in enumerate(zip(axes, ylabels)):
            for j, (sn, lbl) in enumerate(zip(plot_names, plot_labels)):
                dr    = sims[sn]['diff_SPICE_RSW']
                times = get_rsw_times(sims[sn], n_points=len(dr))
                rms   = sims[sn].get('rms_SPICE')
                leg   = f'{lbl} (RMS: {rms:.0f} km)' if rms else lbl
                col   = _SIM_COLORS.get(sn, _color(j))
                ls    = _SIM_LINESTYLE.get(sn, '-')
                alp   = _SIM_ALPHAS.get(sn, 0.92)
                ax.plot(times, dr[:, row],
                        color=col, linestyle=ls, linewidth=1.8, alpha=alp,
                        label=leg if row == 0 else '_nolegend_')
            ax.axhline(0, color='gray', linewidth=0.6, linestyle=':')
            ax.set_ylabel(ylab, fontsize=11)
            ax.tick_params(axis='y', labelsize=10, length=4)
            if apply_zoom and zoom_start and zoom_end:
                ax.set_xlim(zoom_start, zoom_end)

    # ── x-axis formatting: labels only on bottom row ─────────────────────────
    for ax in axes_full[:2]:
        _hide_xticklabels(ax)
    for ax in axes_zoom[:2]:
        _hide_xticklabels(ax)

    # Full column bottom row: yearly ticks
    _apply_date_formatter(axes_full[-1])
    axes_full[-1].tick_params(axis='x', labelsize=10, length=4)
    axes_full[-1].set_xlabel('Date', fontsize=11)

    # Zoom column bottom row: auto-picked ticks, '%d %b' labels
    _setup_zoom_xaxis(axes_zoom[-1])
    axes_zoom[-1].set_xlabel('Date', fontsize=11)

    # ── titles & legend ───────────────────────────────────────────────────────
    axes_full[0].set_title('(a) Full time range', fontsize=11, loc='left')
    zoom_year = f' ({zoom_start.year})' if zoom_start else ''
    axes_zoom[0].set_title(f'(b) Zoomed{zoom_year}', fontsize=11, loc='left')
    axes_full[0].legend(loc='upper right', fontsize=9)

    fig.suptitle(title, fontsize=13)
    return fig


# ── Initial propagation RSW diff (pre-estimation) ────────────────────────────

def mpl_rsw_initial(sims, names, labels,
                    sim_subset=None,
                    unit='km',
                    title='Initial RSW Difference vs NEP097 (pre-estimation)',
                    figsize=None):
    """3-row time-series of RSW difference vs SPICE for the initial (pre-estimation)
    propagation.  Uses diff_SPICE_RSW_initial and time_column_initial.

    Typical use: pass one IAUPole sim and one FitPole sim to compare starting points.
    If sim_subset is None, auto-selects the first IAUPole and first FitPole sim found.
    unit: 'km' (default) or 'm' — data is stored in km, 'm' multiplies by 1000.
    """
    # ── resolve which sims to plot ────────────────────────────────────────────
    if sim_subset is not None:
        pairs = [
            (n, labels[names.index(n)] if n in names else n.replace('_cov', ''))
            for n in sim_subset if n in sims
        ]
    else:
        # Auto-pick first IAUPole and first FitPole (or SimPole) that have initial data
        auto = []
        iau_found = fit_found = False
        for n in names:
            sd = sims.get(n, {})
            if not iau_found and n.startswith('IAUPole') and 'diff_SPICE_RSW_initial' in sd:
                auto.append((n, labels[names.index(n)]))
                iau_found = True
            elif not fit_found and not n.startswith('IAUPole') and 'diff_SPICE_RSW_initial' in sd:
                auto.append((n, labels[names.index(n)]))
                fit_found = True
            if iau_found and fit_found:
                break
        pairs = auto

    pairs = [(n, lb) for n, lb in pairs if 'diff_SPICE_RSW_initial' in sims.get(n, {})]
    if not pairs:
        print("WARNING: No initial RSW data found, skipping mpl_rsw_initial.")
        return None
    plot_names, plot_labels = zip(*pairs)

    # ── unit scaling ─────────────────────────────────────────────────────────
    scale    = 1000.0 if unit == 'm' else 1.0
    unit_str = unit

    # ── figure ────────────────────────────────────────────────────────────────
    if figsize is None:
        figsize = (FIG_W_DOUBLE, 7.0)

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    ylabels = [rf'$\Delta R$ [{unit_str}]',
               rf'$\Delta S$ [{unit_str}]',
               rf'$\Delta W$ [{unit_str}]']

    for row, (ax, ylab) in enumerate(zip(axes, ylabels)):
        for j, (sn, lbl) in enumerate(zip(plot_names, plot_labels)):
            dr    = sims[sn]['diff_SPICE_RSW_initial'] * scale
            t_col = sims[sn].get('time_column_initial')
            if t_col is not None:
                times = convert_time_array_to_datetime(t_col.reshape(-1, 1))
            else:
                times = list(range(len(dr)))

            col = _SIM_COLORS.get(sn, _color(j))
            ls  = _SIM_LINESTYLE.get(sn, '-')
            rms = float(np.sqrt(np.mean(dr ** 2)))
            leg = f'{lbl} (RMS: {rms:.0f} {unit_str})'
            ax.plot(times, dr[:, row],
                    color=col, linestyle=ls, linewidth=1.8, alpha=0.92,
                    label=leg if row == 0 else '_nolegend_')

        ax.axhline(0, color='gray', linewidth=0.6, linestyle=':')
        ax.set_ylabel(ylab, fontsize=11)
        _apply_date_formatter(ax)

    axes[0].set_title(title)
    axes[0].legend(loc='upper right', fontsize=9)
    axes[-1].set_xlabel('Date', fontsize=11)
    fig.tight_layout()
    return fig


# ── Formal errors: full + zoomed subfigure ────────────────────────────────────

def mpl_formal_with_zoom(sims, names, labels,
                         sim_subset=None,
                         zoom_days=90,
                         title='Formal Errors RSW',
                         figsize=None):
    """3×2 figure: left = full time range, right = first zoom_days of data.

    Uses _SIM_COLORS / _SIM_LINESTYLE.
    X-tick labels only on bottom row; zoom column uses '%d %b' date format.
    """
    # ── resolve sim subset ───────────────────────────────────────────────────
    if sim_subset is not None:
        pairs = [
            (n, labels[names.index(n)] if n in names else n.replace('_cov', ''))
            for n in sim_subset if n in sims
        ]
    else:
        pairs = list(zip(names, labels))
    pairs = [(n, lb) for n, lb in pairs if 'formal_errors_RSW_km' in sims.get(n, {})]
    if not pairs:
        return None
    plot_names, plot_labels = zip(*pairs)

    # ── zoom window ───────────────────────────────────────────────────────────
    zoom_start = zoom_end = None
    for sn in plot_names:
        sha = sims[sn].get('state_history_array')
        if sha is not None and len(sha) > 0:
            times = convert_time_array_to_datetime(sha[:, 0])
            if times is not None and len(times) > 0:
                zoom_start = times[0]
                zoom_end   = zoom_start + timedelta(days=zoom_days)
                break

    # ── figure layout ─────────────────────────────────────────────────────────
    if figsize is None:
        figsize = (FIG_W_DOUBLE * 2, 8.5)
    fig = plt.figure(figsize=figsize)
    gs  = GridSpec(3, 2, figure=fig,
                   hspace=0.08, wspace=0.30,
                   left=0.07, right=0.98, top=0.91, bottom=0.10)

    axes_full = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    axes_zoom = [fig.add_subplot(gs[i, 1]) for i in range(3)]
    for i in range(1, 3):
        axes_full[i].sharex(axes_full[0])
        axes_zoom[i].sharex(axes_zoom[0])

    ylabels = [r'$\sigma_R$ [km]', r'$\sigma_S$ [km]', r'$\sigma_W$ [km]']

    # ── plot data ─────────────────────────────────────────────────────────────
    for axes, apply_zoom in [(axes_full, False), (axes_zoom, True)]:
        for row, (ax, ylab) in enumerate(zip(axes, ylabels)):
            for j, (sn, lbl) in enumerate(zip(plot_names, plot_labels)):
                fe  = sims[sn]['formal_errors_RSW_km']
                sha = sims[sn].get('state_history_array')
                times = (convert_time_array_to_datetime(sha[:, 0])
                         if sha is not None else list(range(len(fe))))
                st  = compute_formal_error_statistics(fe)
                mx  = st[['R', 'S', 'W'][row]]['max']
                leg = f'{lbl} (max: {mx:.1f} km)'
                col = _SIM_COLORS.get(sn, _color(j))
                ls  = _SIM_LINESTYLE.get(sn, '-')
                alp = _SIM_ALPHAS.get(sn, 1.0)
                ax.plot(times, fe[:, row],
                        color=col, linestyle=ls, linewidth=1.8, alpha=alp,
                        label=leg if row == 0 else '_nolegend_')
            ax.set_ylabel(ylab, fontsize=11)
            ax.tick_params(axis='y', labelsize=10, length=4)
            if apply_zoom and zoom_start and zoom_end:
                ax.set_xlim(zoom_start, zoom_end)

    # ── x-axis formatting: labels only on bottom row ─────────────────────────
    for ax in axes_full[:2]:
        _hide_xticklabels(ax)
    for ax in axes_zoom[:2]:
        _hide_xticklabels(ax)

    _apply_date_formatter(axes_full[-1])
    axes_full[-1].tick_params(axis='x', labelsize=10, length=4)
    axes_full[-1].set_xlabel('Date', fontsize=11)

    _setup_zoom_xaxis(axes_zoom[-1])
    axes_zoom[-1].set_xlabel('Date', fontsize=11)

    # ── titles & legend ───────────────────────────────────────────────────────
    axes_full[0].set_title('(a) Full time range', fontsize=11, loc='left')
    zoom_year = f' ({zoom_start.year})' if zoom_start else ''
    axes_zoom[0].set_title(f'(b) Zoomed{zoom_year}', fontsize=11, loc='left')
    axes_full[0].legend(loc='upper right', fontsize=9)

    fig.suptitle(title, fontsize=13)
    return fig


# ── RSW diff with ±1σ formal error cloud (shading only) ─────────────────────

def _rsw_formal_resolve_pairs(sims, names, labels, sim_subset):
    """Resolve (plot_names, plot_labels) for rsw+formal figures."""
    if sim_subset is not None:
        pairs = [
            (n, labels[names.index(n)] if n in names else n.replace('_cov', ''))
            for n in sim_subset if n in sims
        ]
    else:
        pairs = list(zip(names, labels))
    return [(n, lb) for n, lb in pairs
            if 'diff_SPICE_RSW' in sims.get(n, {})
            and 'formal_errors_RSW_km' in sims.get(n, {})]


def _rsw_formal_zoom_window(sims, plot_names, zoom_days):
    """Return (zoom_start, zoom_end) datetimes for first available sim."""
    for sn in plot_names:
        dr    = sims[sn]['diff_SPICE_RSW']
        times = get_rsw_times(sims[sn], n_points=len(dr))
        if times is not None and len(times) > 0:
            return times[0], times[0] + timedelta(days=zoom_days)
    return None, None


def _rsw_formal_make_grid(figsize):
    """Return (fig, axes_full[3], axes_zoom[3]) for a 3×2 layout."""
    fig = plt.figure(figsize=figsize)
    gs  = GridSpec(3, 2, figure=fig,
                   hspace=0.08, wspace=0.30,
                   left=0.07, right=0.98, top=0.91, bottom=0.10)
    axes_full = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    axes_zoom = [fig.add_subplot(gs[i, 1]) for i in range(3)]
    for i in range(1, 3):
        axes_full[i].sharex(axes_full[0])
        axes_zoom[i].sharex(axes_zoom[0])
    return fig, axes_full, axes_zoom


def _rsw_formal_finish(fig, axes_full, axes_zoom, zoom_start, title):
    """Apply axis labels, date formatters, titles."""
    ylabels = [r'$\Delta R$ [km]', r'$\Delta S$ [km]', r'$\Delta W$ [km]']
    for ax, ylab in zip(axes_full, ylabels):
        ax.set_ylabel(ylab, fontsize=11)
    for ax, ylab in zip(axes_zoom, ylabels):
        ax.set_ylabel(ylab, fontsize=11)
    for ax in axes_full[:2]:
        _hide_xticklabels(ax)
    for ax in axes_zoom[:2]:
        _hide_xticklabels(ax)
    _apply_date_formatter(axes_full[-1])
    axes_full[-1].tick_params(axis='x', labelsize=10, length=4)
    axes_full[-1].set_xlabel('Date', fontsize=11)
    _setup_zoom_xaxis(axes_zoom[-1])
    axes_zoom[-1].set_xlabel('Date', fontsize=11)
    axes_full[0].set_title('(a) Full time range', fontsize=11, loc='left')
    zoom_year = f' ({zoom_start.year})' if zoom_start else ''
    axes_zoom[0].set_title(f'(b) Zoomed{zoom_year}', fontsize=11, loc='left')
    axes_full[0].legend(loc='upper right', fontsize=9)
    fig.suptitle(title, fontsize=13)


def mpl_rsw_with_formal_cloud(sims, names, labels,
                               sim_subset=None,
                               zoom_days=90,
                               title='RSW Difference vs NEP097 with ±1σ Formal Error',
                               figsize=None):
    """3×2 figure: RSW diff line with ±1σ formal error shading (cloud).

    Left column: full time range.  Right column: first zoom_days of data.
    The shaded band shows diff ± formal_error (1σ confidence region).
    """
    pairs = _rsw_formal_resolve_pairs(sims, names, labels, sim_subset)
    if not pairs:
        return None
    plot_names, plot_labels = zip(*pairs)
    zoom_start, zoom_end = _rsw_formal_zoom_window(sims, plot_names, zoom_days)

    if figsize is None:
        figsize = (FIG_W_DOUBLE * 2, 8.5)
    fig, axes_full, axes_zoom = _rsw_formal_make_grid(figsize)

    for axes, apply_zoom in [(axes_full, False), (axes_zoom, True)]:
        for row, ax in enumerate(axes):
            for j, (sn, lbl) in enumerate(zip(plot_names, plot_labels)):
                dr    = sims[sn]['diff_SPICE_RSW']
                fe    = sims[sn]['formal_errors_RSW_km']
                times = get_rsw_times(sims[sn], n_points=len(dr))
                rms   = sims[sn].get('rms_SPICE')
                col   = _SIM_COLORS.get(sn, _color(j))
                ls    = _SIM_LINESTYLE.get(sn, '-')

                fe_interp = np.interp(
                    np.linspace(0, 1, len(dr)),
                    np.linspace(0, 1, len(fe)),
                    fe[:, row],
                )
                leg = (f'{lbl} (RMS: {rms:.0f} km)' if rms else lbl) if row == 0 else '_nolegend_'
                ax.fill_between(times,
                                dr[:, row] - fe_interp,
                                dr[:, row] + fe_interp,
                                color=col, alpha=0.30, linewidth=0,
                                label='_nolegend_', zorder=2)
                ax.plot(times, dr[:, row],
                        color=col, linestyle=ls, linewidth=1.8, alpha=0.92,
                        label=leg, zorder=3)
            ax.axhline(0, color='gray', linewidth=0.6, linestyle=':')
            ax.tick_params(axis='y', labelsize=10, length=4)
            if apply_zoom and zoom_start and zoom_end:
                ax.set_xlim(zoom_start, zoom_end)

    _rsw_formal_finish(fig, axes_full, axes_zoom, zoom_start, title)
    return fig


# ── RSW diff + formal error as separate colored lines ────────────────────────

# Secondary colors for formal error lines — contrasts with _SIM_COLORS.
# Paul Tol "muted" palette picks that differ from both IAUPole (cool) and
# SimPole (warm) assigned colors, so both groups get a readable contrast.
_FORMAL_LINE_COLORS = {
    'IAUPole_initial_state':             '#DDCC77',  # sand
    'IAUPole_pole_pos_cov':              '#44AA99',  # teal
    'IAUPole_pole_lib_cov':              '#88CCEE',  # cyan
    'IAUPole_pole_pos_cov_pole_lib_cov': '#DDCC77',  # sand
    'SimPole_initial_state':             '#44AA99',  # teal
    'SimPole_pole_pos_cov':              '#88CCEE',  # cyan
    'SimPole_pole_lib_cov':              '#44AA99',  # teal
    'SimPole_pole_pos_cov_pole_lib_cov': '#88CCEE',  # cyan
}


def mpl_rsw_and_formal_lines(sims, names, labels,
                              sim_subset=None,
                              zoom_days=90,
                              title='RSW Difference vs NEP097 and Formal Errors',
                              figsize=None,
                              show_formal=True,
                              show_zoom=True,
                              layout='vertical',
                              per_sim_alpha=None,
                              axes_fontsize=11,
                              tick_fontsize=10,
                              legend_fontsize=9,
                              title_fontsize=13,
                              formal_line_colors=None,
                              legend_loc=None,
                              legend_outside=False,
                              formal_as_band=False,
                              formal_band_alpha=0.25,
                              formal_band_edge=True,
                              show_rms_in_legend=True,
                              diff_label_suffix=' diff',
                              diff_as_envelope=False,
                              envelope_window=365,
                              diff_envelope_alpha=0.55,
                              intersection_color=None,
                              intersection_alpha=0.6,
                              y_max_nticks=None,
                              y_integer_ticks=False):
    """RSW diff (and optionally formal error) as separate lines.

    Default layout: 3×2 grid (R/S/W rows × full + zoom cols), formal-error
    lines included (legacy behaviour).

    Parameters
    ----------
    show_formal : bool
        If False, omit the formal-error dashed lines (only RSW diff is drawn).
    show_zoom : bool
        If False, drop the zoom column.
    layout : {'vertical', 'horizontal'}
        Only effective when show_zoom is False.
        'vertical'   → 3×1 (R/S/W stacked rows).
        'horizontal' → 1×3 (R/S/W side-by-side).
    per_sim_alpha : dict[str, float] | None
        Per-sim opacity override for the diff lines.  Sims not listed use 0.92.
    """
    pairs = _rsw_formal_resolve_pairs(sims, names, labels, sim_subset)
    if not pairs:
        return None
    plot_names, plot_labels = zip(*pairs)
    zoom_start, zoom_end = _rsw_formal_zoom_window(sims, plot_names, zoom_days)
    per_sim_alpha = per_sim_alpha or {}

    # ── Layout selection ─────────────────────────────────────────────────────
    if show_zoom:
        if figsize is None:
            figsize = (FIG_W_DOUBLE * 2, 8.5)
        fig, axes_full, axes_zoom = _rsw_formal_make_grid(figsize)
        panels = [(axes_full, False), (axes_zoom, True)]
    elif layout == 'horizontal':
        if figsize is None:
            figsize = (FIG_W_DOUBLE * 2.2, 4.5)
        fig, axes_full = plt.subplots(1, 3, figsize=figsize, sharex=True)
        axes_full = list(axes_full)
        axes_zoom = []
        panels    = [(axes_full, False)]
    else:  # vertical, no zoom
        if figsize is None:
            figsize = (FIG_W_DOUBLE * 1.4, 7.5)
        fig, axes_full = plt.subplots(3, 1, figsize=figsize, sharex=True)
        axes_full = list(axes_full)
        axes_zoom = []
        panels    = [(axes_full, False)]

    ylabels = [r'$\Delta R$ [km]', r'$\Delta S$ [km]', r'$\Delta W$ [km]']

    for axes, apply_zoom in panels:
        for row, ax in enumerate(axes):
            for j, (sn, lbl) in enumerate(zip(plot_names, plot_labels)):
                dr       = sims[sn]['diff_SPICE_RSW']
                times    = get_rsw_times(sims[sn], n_points=len(dr))
                rms      = sims[sn].get('rms_SPICE')
                col_diff = _SIM_COLORS.get(sn, _color(j))
                ls       = _SIM_LINESTYLE.get(sn, '-')
                a_diff   = per_sim_alpha.get(sn, 0.92)

                if row == 0:
                    base_lbl = f'{lbl} {diff_label_suffix.lstrip()}'
                    diff_leg = (f'{base_lbl} (RMS: {rms:.0f} km)'
                                if (rms and show_rms_in_legend) else base_lbl)
                else:
                    diff_leg = '_nolegend_'

                if diff_as_envelope:
                    import pandas as _pd
                    s = _pd.Series(dr[:, row])
                    win = max(int(envelope_window), 1)
                    hi = s.rolling(win, center=True, min_periods=1).max().to_numpy()
                    lo = s.rolling(win, center=True, min_periods=1).min().to_numpy()
                    ax.fill_between(times, lo, hi,
                                    color=col_diff, alpha=diff_envelope_alpha,
                                    linewidth=0, label=diff_leg, zorder=3)
                else:
                    ax.plot(times, dr[:, row],
                            color=col_diff, linestyle=ls, linewidth=1.8, alpha=a_diff,
                            label=diff_leg, zorder=3)

                if show_formal and 'formal_errors_RSW_km' in sims.get(sn, {}):
                    fe       = sims[sn]['formal_errors_RSW_km']
                    sha      = sims[sn].get('state_history_array')
                    times_fe = (convert_time_array_to_datetime(sha[:, 0])
                                if sha is not None else times)
                    if formal_line_colors and sn in formal_line_colors:
                        col_fe = formal_line_colors[sn]
                    else:
                        col_fe = _FORMAL_LINE_COLORS.get(sn, _color(j + 4))
                    fe_leg   = (f'{lbl}  formal 1$\\sigma$' if formal_as_band
                                else f'{lbl}  formal error') if row == 0 else '_nolegend_'
                    if formal_as_band:
                        _edge_kw = ({'edgecolor': col_fe, 'linewidth': 1.2}
                                    if formal_band_edge
                                    else {'edgecolor': 'none', 'linewidth': 0})
                        ax.fill_between(times_fe, -fe[:, row], fe[:, row],
                                        color=col_fe, alpha=formal_band_alpha,
                                        label=fe_leg, zorder=5, **_edge_kw)

                        # Intersection of diff envelope and formal band — drawn
                        # on top in a distinct colour to make the overlap region
                        # read clearly even when both base layers are translucent.
                        if intersection_color is not None and diff_as_envelope:
                            # Reuse the rolling-min/max envelope computed for
                            # the diff layer; interpolate onto the formal time
                            # axis so the two bands share a common abscissa.
                            import pandas as _pd
                            from matplotlib.dates import date2num as _d2n
                            t_diff_num = _d2n(np.asarray(times))
                            t_fe_num   = _d2n(np.asarray(times_fe))
                            lo_i = np.interp(t_fe_num, t_diff_num, lo)
                            hi_i = np.interp(t_fe_num, t_diff_num, hi)
                            inter_lo = np.maximum(lo_i, -fe[:, row])
                            inter_hi = np.minimum(hi_i,  fe[:, row])
                            inter_lo_v = np.where(inter_hi > inter_lo, inter_lo, np.nan)
                            inter_hi_v = np.where(inter_hi > inter_lo, inter_hi, np.nan)
                            inter_leg = ('overlap (|Δ| ≤ 1σ)'
                                         if row == 0 else '_nolegend_')
                            ax.fill_between(times_fe, inter_lo_v, inter_hi_v,
                                            color=intersection_color,
                                            alpha=intersection_alpha,
                                            edgecolor='none', linewidth=0,
                                            label=inter_leg, zorder=6)
                    else:
                        ax.plot(times_fe, fe[:, row],
                                color=col_fe, linestyle='--', linewidth=1.4,
                                alpha=0.88, label=fe_leg, zorder=3)

            ax.axhline(0, color='gray', linewidth=0.6, linestyle=':')
            ax.tick_params(axis='y', labelsize=tick_fontsize, length=4)
            # Denser, whole-number y-ticks when requested.  steps=[1,2,2.5,5,10]
            # ensures locations land on round values (… 25, 50, 75 …) so the
            # top/bottom edges typically carry a labelled tick.
            if y_max_nticks is not None:
                ax.yaxis.set_major_locator(plt.MaxNLocator(
                    nbins=y_max_nticks,
                    integer=y_integer_ticks,
                    steps=[1, 2, 2.5, 5, 10],
                ))
            if apply_zoom and zoom_start and zoom_end:
                ax.set_xlim(zoom_start, zoom_end)
            if not show_zoom and layout == 'horizontal':
                ax.set_ylabel(ylabels[row], fontsize=axes_fontsize)
                ax.set_xlabel('Date', fontsize=axes_fontsize)
                _apply_date_formatter(ax)
                ax.tick_params(axis='x', labelsize=tick_fontsize)

    if show_zoom:
        _rsw_formal_finish(fig, axes_full, axes_zoom, zoom_start, title)
    elif layout == 'horizontal':
        loc = legend_loc or 'upper right'
        axes_full[0].legend(loc=loc, fontsize=legend_fontsize)
        if title:
            fig.suptitle(title, fontsize=title_fontsize)
        fig.tight_layout()
    else:  # vertical, no zoom
        for ax, ylab in zip(axes_full, ylabels):
            ax.set_ylabel(ylab, fontsize=axes_fontsize)
        for ax in axes_full[:-1]:
            _hide_xticklabels(ax)
        _apply_date_formatter(axes_full[-1])
        axes_full[-1].set_xlabel('Date', fontsize=axes_fontsize)
        axes_full[-1].tick_params(axis='x', labelsize=tick_fontsize)
        if legend_outside:
            handles, lbls_ = axes_full[0].get_legend_handles_labels()
            fig.legend(handles, lbls_, loc='upper center',
                       bbox_to_anchor=(0.5, 0.99),
                       ncol=min(len(handles), 4),
                       fontsize=legend_fontsize, frameon=True)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
        else:
            loc = legend_loc or 'upper right'
            axes_full[0].legend(loc=loc, fontsize=legend_fontsize)
            if title:
                fig.suptitle(title, fontsize=title_fontsize)
            fig.tight_layout()
    return fig


# kept for back-compatibility — wraps rsw_with_formal_cloud
def mpl_rsw_with_formal(sims, names, labels, **kwargs):
    return mpl_rsw_with_formal_cloud(sims, names, labels, **kwargs)


# ── Standalone legend figure ──────────────────────────────────────────────────

def mpl_legend(sims, names, labels,
               title='Simulation Legend',
               ncols=2,
               shape_labels=None,
               notes=None,
               figsize=None):
    """Standalone legend figure for use across thesis figures.

    Produces a two-section legend:
      • Shape key  — marker shape encodes the rotation model (IAU vs FitPole).
      • Color key  — one entry per simulation with its color, marker, linestyle.
    Optionally adds an abbreviation glossary below the legend box.

    Parameters
    ----------
    shape_labels : dict, optional
        Maps marker string → human-readable label for the shape-key section.
        Default: {'o': 'IAU 2015 rotation model',
                  'D': 'FitPole rotation model'}.
    notes : list of str, optional
        Lines of explanatory text rendered as a monospaced glossary below the
        legend box (e.g. abbreviation expansions).  Each string is one line.
    ncols : int
        Number of columns in the legend (default 2).
    """
    from matplotlib.lines import Line2D

    if shape_labels is None:
        shape_labels = {
            'o': 'IAU 2015 rotation model',
            'D': 'FitPole rotation model',
        }

    # ── Collect unique marker shapes that actually appear in this dataset ──
    used_markers = dict.fromkeys(                       # ordered, deduped
        _marker(sn) for sn in names
    )
    shape_handles = []
    for mk in used_markers:
        lbl = shape_labels.get(mk, f'marker: {mk}')
        ls  = '-' if mk == 'o' else '--'                # mirror linestyle convention
        shape_handles.append(
            Line2D([0], [0], color='#555555', marker=mk, linestyle=ls,
                   linewidth=1.5, markersize=9,
                   markeredgecolor='black', markeredgewidth=0.5,
                   label=lbl)
        )

    # ── Per-simulation entries ─────────────────────────────────────────────
    sim_handles = []
    for i, (sn, lbl) in enumerate(zip(names, labels)):
        col = _SIM_COLORS.get(sn, _color(i))
        mk  = _marker(sn)
        ls  = _SIM_LINESTYLE.get(sn, '-')
        sim_handles.append(
            Line2D([0], [0], color=col, marker=mk, linestyle=ls,
                   linewidth=1.5, markersize=8,
                   markeredgecolor='black', markeredgewidth=0.4,
                   label=lbl)
        )

    # ── Combine: shape key first, blank separator, then per-sim entries ───
    sep = Line2D([], [], linestyle='none', label='')
    all_handles = shape_handles + [sep] + sim_handles
    all_labels  = [h.get_label() for h in all_handles]

    # ── Figure size: auto-scale to legend rows + optional notes block ─────
    notes_lines = len(notes) if notes else 0
    if figsize is None:
        n_rows = -(-len(all_handles) // ncols)          # ceiling division
        figsize = (FIG_W_DOUBLE * max(1, ncols / 2),
                   0.42 * n_rows + 0.9 + 0.22 * notes_lines)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    # ── Notes block — rendered below the legend in a fixed-width font ─────
    notes_height = 0.22 * notes_lines / figsize[1] if notes_lines else 0.0
    legend_bottom = notes_height + 0.02            # legend sits above the notes

    if notes:
        notes_text = '\n'.join(notes)
        fig.text(0.04, notes_height / 2,           # vertically centred in notes strip
                 notes_text,
                 ha='left', va='center',
                 fontsize=9,
                 fontfamily='monospace',
                 transform=fig.transFigure)

    ax.legend(handles=all_handles, labels=all_labels,
              loc='center',
              bbox_to_anchor=(0.5, (legend_bottom + 1.0) / 2),
              ncol=ncols,
              fontsize=10,
              frameon=True,
              framealpha=0.8,
              title=title,
              title_fontsize=11,
              handlelength=2.5,
              handleheight=1.0,
              borderpad=0.9,
              labelspacing=0.55)
    fig.tight_layout(rect=[0, notes_height, 1, 1])
    return fig


# ── Initial propagation difference: (sim_a − sim_b) ──────────────────────────

def mpl_rsw_initial_diff(sims, names, labels,
                         sim_subset=None,
                         title='RSW Initial Propagation Difference vs NEP097',
                         figsize=None):
    """3-row time series of (sim_a − sim_b) initial RSW difference.

    sim_subset must be a list of exactly two simulation names:
        [sim_a, sim_b]  →  diff = diff_SPICE_RSW_initial[sim_a] − diff_SPICE_RSW_initial[sim_b]

    Both sims must have diff_SPICE_RSW_initial and time_column_initial.
    Arrays are trimmed to the shorter length if they differ.
    """
    if sim_subset is None or len(sim_subset) < 2:
        print("WARNING: rsw_initial_diff requires sim_subset=[sim_a, sim_b], skipping.")
        return None

    sn_a, sn_b = sim_subset[0], sim_subset[1]

    def _get_rsw(sn):
        """Return RSW array, preferring initial diff then falling back to final."""
        sd = sims.get(sn, {})
        for key in ('diff_SPICE_RSW_initial', 'diff_SPICE_RSW'):
            if key in sd:
                return sd[key]
        return None

    dr_a = _get_rsw(sn_a)
    dr_b = _get_rsw(sn_b)
    for sn, dr in ((sn_a, dr_a), (sn_b, dr_b)):
        if dr is None:
            print(f"WARNING: '{sn}' has no RSW diff data, skipping rsw_initial_diff.")
            return None

    lbl_a = labels[names.index(sn_a)] if sn_a in names else sn_a
    lbl_b = labels[names.index(sn_b)] if sn_b in names else sn_b

    t_col = sims[sn_a].get('time_column_initial') or sims[sn_a].get('time_column')
    if t_col is not None:
        times = convert_time_array_to_datetime(t_col.reshape(-1, 1))
    else:
        times = list(range(len(dr_a)))

    # Trim to common length
    n_pts   = min(len(dr_a), len(dr_b), len(times))
    dr_diff = dr_a[:n_pts] - dr_b[:n_pts]
    times   = times[:n_pts]

    if figsize is None:
        figsize = (FIG_W_DOUBLE, 7.0)

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    ylabels = [
        r'$\Delta R_{\,\mathrm{IAU}-\mathrm{Jac.}}$ [km]',
        r'$\Delta S_{\,\mathrm{IAU}-\mathrm{Jac.}}$ [km]',
        r'$\Delta W_{\,\mathrm{IAU}-\mathrm{Jac.}}$ [km]',
    ]

    diff_color = '#7B2D8B'  # purple — neutral, not associated with either model

    for row, (ax, ylab) in enumerate(zip(axes, ylabels)):
        ax.plot(times, dr_diff[:, row],
                color=diff_color, linewidth=1.8, alpha=0.92)
        ax.axhline(0, color='gray', linewidth=0.6, linestyle=':')
        ax.set_ylabel(ylab, fontsize=11)
        _apply_date_formatter(ax)

    axes[0].set_title(title)
    axes[0].text(0.01, 0.97, f'({lbl_a}) − ({lbl_b})',
                 transform=axes[0].transAxes,
                 ha='left', va='top', fontsize=9, style='italic', color='gray')
    axes[-1].set_xlabel('Date', fontsize=11)
    fig.tight_layout()
    return fig


# ── RSW diff: initial (nominal) vs final (estimated) for one sim ─────────────

def mpl_rsw_initial_vs_final(sims, names, labels,
                              sim_subset=None,
                              zoom_days=90,
                              title=None,
                              figsize=None):
    """3×2 figure: initial-iteration RSW diff (nominal) vs final-iteration diff.

    For each sim in sim_subset the initial propagation (diff_SPICE_RSW_initial,
    time_column_initial) is plotted as a solid black/gray reference line labeled
    'IAU nominal' (for IAUPole_* sims) or 'FitPole nominal' (for SimPole_*).
    The final estimated diff (diff_SPICE_RSW) is plotted in the sim's color.

    Typically called with a single sim in sim_subset to produce one figure per
    estimation.
    """
    targets = sim_subset if sim_subset is not None else names

    # Build pairs that have both initial and final RSW data
    pairs = []
    for sn in targets:
        sd = sims.get(sn, {})
        if 'diff_SPICE_RSW' in sd and 'diff_SPICE_RSW_initial' in sd:
            lbl = labels[names.index(sn)] if sn in names else sn
            pairs.append((sn, lbl))
    if not pairs:
        return None
    plot_names, plot_labels = zip(*pairs)

    # ── zoom window from first sim's final diff ───────────────────────────────
    zoom_start = zoom_end = None
    for sn in plot_names:
        dr    = sims[sn]['diff_SPICE_RSW']
        times = get_rsw_times(sims[sn], n_points=len(dr))
        if times is not None and len(times) > 0:
            zoom_start = times[0]
            zoom_end   = zoom_start + timedelta(days=zoom_days)
            break

    # ── figure layout ─────────────────────────────────────────────────────────
    if figsize is None:
        figsize = (FIG_W_DOUBLE * 2, 8.5)
    fig = plt.figure(figsize=figsize)
    gs  = GridSpec(3, 2, figure=fig,
                   hspace=0.08, wspace=0.30,
                   left=0.07, right=0.98, top=0.91, bottom=0.10)
    axes_full = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    axes_zoom = [fig.add_subplot(gs[i, 1]) for i in range(3)]
    for i in range(1, 3):
        axes_full[i].sharex(axes_full[0])
        axes_zoom[i].sharex(axes_zoom[0])

    ylabels = [r'$\Delta R$ [km]', r'$\Delta S$ [km]', r'$\Delta W$ [km]']

    for axes, apply_zoom in [(axes_full, False), (axes_zoom, True)]:
        for row, (ax, ylab) in enumerate(zip(axes, ylabels)):
            for j, (sn, lbl) in enumerate(zip(plot_names, plot_labels)):
                sd  = sims[sn]
                col = _SIM_COLORS.get(sn, _color(j))
                ls  = _SIM_LINESTYLE.get(sn, '-')

                # Nominal label depends on group
                nom_lbl = ('FitPole nominal' if sn.startswith('SimPole')
                           else 'IAU nominal')
                nom_col = 'gray' if sn.startswith('SimPole') else 'black'

                # ── initial (nominal) ─────────────────────────────────────────
                dr_init = sd['diff_SPICE_RSW_initial']
                t_init  = sd.get('time_column_initial')
                if t_init is not None:
                    times_init = convert_time_array_to_datetime(
                        t_init.reshape(-1, 1))
                else:
                    times_init = get_rsw_times(sd, n_points=len(dr_init))

                rms_init = float(np.sqrt(np.mean(dr_init ** 2)))
                nom_leg  = (f'{nom_lbl} (RMS: {rms_init:.0f} km)'
                            if row == 0 else '_nolegend_')
                ax.plot(times_init, dr_init[:, row],
                        color=nom_col, linestyle='-', linewidth=1.8, alpha=0.75,
                        label=nom_leg, zorder=2)

                # ── final (estimated) ─────────────────────────────────────────
                dr_final = sd['diff_SPICE_RSW']
                times_final = get_rsw_times(sd, n_points=len(dr_final))
                rms_final   = sd.get('rms_SPICE')
                est_leg = (f'{lbl} (RMS: {rms_final:.0f} km)' if rms_final
                           else lbl) if row == 0 else '_nolegend_'
                ax.plot(times_final, dr_final[:, row],
                        color=col, linestyle=ls, linewidth=1.8, alpha=0.92,
                        label=est_leg, zorder=3)

                ax.axhline(0, color='gray', linewidth=0.6, linestyle=':')
                ax.set_ylabel(ylab, fontsize=11)
                ax.tick_params(axis='y', labelsize=10, length=4)
                if apply_zoom and zoom_start and zoom_end:
                    ax.set_xlim(zoom_start, zoom_end)

    # ── x-axis formatting ─────────────────────────────────────────────────────
    for ax in axes_full[:2]:
        _hide_xticklabels(ax)
    for ax in axes_zoom[:2]:
        _hide_xticklabels(ax)
    _apply_date_formatter(axes_full[-1])
    axes_full[-1].tick_params(axis='x', labelsize=10, length=4)
    axes_full[-1].set_xlabel('Date', fontsize=11)
    _setup_zoom_xaxis(axes_zoom[-1])
    axes_zoom[-1].set_xlabel('Date', fontsize=11)

    axes_full[0].set_title('(a) Full time range', fontsize=11, loc='left')
    zoom_year = f' ({zoom_start.year})' if zoom_start else ''
    axes_zoom[0].set_title(f'(b) Zoomed{zoom_year}', fontsize=11, loc='left')
    axes_full[0].legend(loc='upper right', fontsize=9)

    _title = title if title is not None else (
        f'RSW Difference vs NEP097 — nominal vs estimated ({plot_labels[0]})'
        if len(plot_labels) == 1 else 'RSW Difference vs NEP097 — Nominal vs Estimated')
    fig.suptitle(_title, fontsize=13)
    return fig


# ── Gravitational parameter update bar chart ──────────────────────────────────

def mpl_param_gm(sims, names, labels,
                 sim_subset=None,
                 title='Gravitational Parameter Update',
                 figsize=None,
                 show_suptitle=True,
                 tick_fontsize=8):
    """1×2 bar chart: ΔGM_Nep [m³/s²] and ΔGM_Tri [m³/s²] across simulations.

    Keeps only simulations that estimate at least one GM parameter (group='Gravity').
    Each sim uses its own initial value as reference — appropriate for SimObs analysis
    where no external IAU/SimPole reference is defined.
    """
    if sim_subset is not None:
        pairs  = [(n, labels[names.index(n)]) for n in sim_subset if n in sims and n in names]
        names  = [p[0] for p in pairs]
        labels = [p[1] for p in pairs]

    # Keep only sims that estimate at least one GM parameter
    filt = [(sn, labels[names.index(sn)]) for sn in names
            if _param_update(sims.get(sn, {}), ['Gravity'])[0] is not None]
    if not filt:
        print("WARNING: No simulations with GM parameters found, skipping mpl_param_gm.")
        return None
    f_names, f_labels = zip(*filt)

    if figsize is None:
        figsize = (FIG_W_DOUBLE * 1.2, FIG_H_DEFAULT)

    gm_colors = {'GM_Nep': '#0072B2', 'GM_Tri': '#D55E00'}
    fig, (ax_n, ax_t) = plt.subplots(1, 2, figsize=figsize)

    for ax, plabel in zip((ax_n, ax_t), ('GM_Nep', 'GM_Tri')):
        vals = []
        for sn in f_names:
            v_all, pl, _ = _param_update(sims.get(sn, {}), ['Gravity'])
            vals.append(v_all[pl.index(plabel)] if (pl is not None and plabel in pl) else np.nan)
        _bar_ax(ax, vals, gm_colors[plabel], f_labels,
                r'm$^3$/s$^2$', plabel, fontsize_annot=7, fontsize_tick=tick_fontsize)

    if show_suptitle:
        fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


# ── Pole rotation rate update bar chart ───────────────────────────────────────

def mpl_param_pole_rate(sims, names, labels,
                        sim_subset=None,
                        title='Pole Rotation Rate Update  (α̇₀, δ̇₀)',
                        figsize=None,
                        variant='combined'):
    """1×2 bar chart: Δα̇₀ and Δδ̇₀ across simulations [deg].

    Keeps only simulations that estimate at least one pole-rate parameter.
    Scale: _RAD_TO_DEG (values are stored in rad/s; display shows the raw
    radian→degree scaled update for comparison across sims).
    """
    iau_prefix = 'IAUPole'
    fit_prefix = 'FitPole'
    iau_ref_in, _ = _get_iau_reference(sims, names)

    if variant == 'iau':
        base_names   = [n for n in names if n.startswith(iau_prefix)]
        ref_map      = {n: None for n in base_names}
        title_suffix = ' — IAU'
    elif variant in ('fitpole', 'simpole'):
        base_names   = [n for n in names if not n.startswith(iau_prefix)]
        ref_map      = {n: None for n in base_names}
        title_suffix = f' — {fit_prefix}'
    else:  # combined
        base_names   = list(names)
        ref_map      = {n: (iau_ref_in if not n.startswith(iau_prefix) else None)
                        for n in base_names}
        title_suffix = f' — Combined ({fit_prefix} vs IAU ref)'

    if sim_subset is not None:
        base_names = [n for n in base_names if n in sim_subset]
        ref_map    = {n: ref_map[n] for n in base_names if n in ref_map}
        title_suffix = ''

    title = title + title_suffix

    filt = [(sn, labels[names.index(sn)]) for sn in base_names
            if _param_update(sims.get(sn, {}), ['Pole Rate'])[0] is not None]
    if not filt:
        print("WARNING: No simulations with pole-rate parameters found, skipping mpl_param_pole_rate.")
        return None
    f_names, f_labels = zip(*filt)

    if figsize is None:
        figsize = (FIG_W_DOUBLE * 1.2, FIG_H_DEFAULT)

    rate_colors = {'α̇₀': '#4A148C', 'δ̇₀': '#7B1FA2'}
    fig, (ax_a, ax_d) = plt.subplots(1, 2, figsize=figsize)

    for ax, plabel in zip((ax_a, ax_d), ('α̇₀', 'δ̇₀')):
        vals = []
        for sn in f_names:
            v_all, pl, _ = _param_update(
                sims.get(sn, {}), ['Pole Rate'],
                scale=_RAD_TO_DEG, iau_ref=ref_map[sn])
            vals.append(v_all[pl.index(plabel)] if (pl is not None and plabel in pl) else np.nan)
        _bar_ax(ax, vals, rate_colors[plabel], f_labels, 'deg', plabel, fontsize_annot=8)

    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


# ── Spherical harmonics update bar chart ──────────────────────────────────────

def mpl_param_sh(sims, names, labels,
                 sim_subset=None,
                 title='Spherical Harmonics Update  (C₂₀, C₄₀)',
                 figsize=None):
    """1×2 bar chart: ΔC₂₀ and ΔC₄₀ across simulations [dimensionless].

    Keeps only simulations that estimate at least one spherical-harmonics
    parameter (group='Spherical Harmonics').
    """
    if sim_subset is not None:
        pairs  = [(n, labels[names.index(n)]) for n in sim_subset if n in sims and n in names]
        names  = [p[0] for p in pairs]
        labels = [p[1] for p in pairs]

    filt = [(sn, labels[names.index(sn)]) for sn in names
            if _param_update(sims.get(sn, {}), ['Spherical Harmonics'])[0] is not None]
    if not filt:
        print("WARNING: No simulations with SH parameters found, skipping mpl_param_sh.")
        return None
    f_names, f_labels = zip(*filt)

    if figsize is None:
        figsize = (FIG_W_DOUBLE * 1.2, FIG_H_DEFAULT)

    sh_colors = {'C₂₀': '#E65100', 'C₄₀': '#EF6C00'}
    fig, (ax_c20, ax_c40) = plt.subplots(1, 2, figsize=figsize)

    for ax, plabel in zip((ax_c20, ax_c40), ('C₂₀', 'C₄₀')):
        vals = []
        for sn in f_names:
            v_all, pl, _ = _param_update(sims.get(sn, {}), ['Spherical Harmonics'])
            vals.append(v_all[pl.index(plabel)] if (pl is not None and plabel in pl) else np.nan)
        _bar_ax(ax, vals, sh_colors[plabel], f_labels, '[-]', plabel, fontsize_annot=8)

    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


# ============================================================================
# MULTI-DATASET HELPERS AND FIGURES
# ============================================================================

def _load_dataset_for_config(config_name):
    """Load (sims, names, labels, sim_colors, sim_linestyle, sim_markers) for a
    named ExportConfig module.  Uses the module's SELECTED_SIMS / SIM_LABELS to
    filter and order the simulations, then falls back to all available names.

    Honours EXPORT_DATASETS env var: configs requesting an unloaded dataset
    return empty results instead of crashing or silently re-loading."""
    cfg  = importlib.import_module(f'ExportConfigs.{config_name}')
    _wanted = os.environ.get('EXPORT_DATASETS')
    if _wanted:
        wanted_set = {x.strip() for x in _wanted.split(',') if x.strip()}
        if cfg.DATASET_LABEL not in wanted_set:
            print(f"  [EXPORT_DATASETS] config {config_name!r} wants dataset "
                  f"{cfg.DATASET_LABEL!r} which is not loaded — returning empty.")
            return {}, [], [], {}, {}, {}
    sims, all_names = get_active_data(cfg.DATASET_LABEL)

    sel = cfg.SELECTED_SIMS if cfg.SELECTED_SIMS is not None else list(all_names)
    names = [n for n in sel if n in sims]

    if cfg.SIM_LABELS is not None and len(cfg.SIM_LABELS) == len(sel):
        idx_map = {n: i for i, n in enumerate(sel)}
        labels  = [cfg.SIM_LABELS[idx_map[n]] for n in names]
    else:
        labels = list(names)

    colors    = getattr(cfg, 'SIM_COLORS',    {})
    linestyle = getattr(cfg, 'SIM_LINESTYLE', {})
    markers   = getattr(cfg, 'SIM_MARKERS',   {})
    return sims, names, labels, colors, linestyle, markers


_MULTI_PANEL_H   = 3.5    # figure height for each individual split figure
_MULTI_PANEL_W   = FIG_W_SINGLE * 1.4   # ~4.9" — fits in a half-page minipage at print size
_MULTI_FS_TICKS  = 13     # x-tick label fontsize
_MULTI_FS_ANNOT  = 12     # value annotation fontsize
_MULTI_FS_YLABEL = 14     # y-axis label fontsize
_MULTI_FS_TITLE  = 15     # panel title fontsize


def _multi_dot_panel(ax, sims, names, labels, vals,
                     colors, markers, col, panel_title, show_annot=True):
    """Draw a single dot-plot panel (used by all *_multi figures)."""
    x = np.arange(len(names))
    valid_x = [xi for xi, v in enumerate(vals) if not np.isnan(v)]
    valid_v = [v  for v  in vals               if not np.isnan(v)]
    if len(valid_x) > 1:
        ax.plot(valid_x, valid_v, '-', color='gray', linewidth=1.0, alpha=0.5, zorder=1)
    for xi, v in enumerate(vals):
        if not np.isnan(v):
            sn  = names[xi]
            c   = colors.get(sn, col)
            mk  = markers.get(sn, 'o')
            ax.plot(xi, v, marker=mk, color=c, markersize=7,
                    markeredgecolor='black', markeredgewidth=0.4,
                    linestyle='none', zorder=2)
            if show_annot:
                ax.text(xi, v, f'  {v:.2f}', ha='left', va='center',
                        fontsize=_MULTI_FS_ANNOT)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=_MULTI_FS_TICKS)
    ax.set_xlim(-0.5, len(names) - 0.5)
    ax.set_title(panel_title, fontsize=_MULTI_FS_TITLE)


def mpl_rms_compare_multi(sims, names, labels,
                           datasets=None,
                           dataset_titles=None,
                           title='RMS vs NEP097 — Dataset Comparison',
                           show_annot=True):
    """N×1 figure: one rms_compare panel per dataset, stacked vertically."""
    if not datasets:
        return mpl_rms_compare(sims, names, labels, title=title)

    n_ds = len(datasets)
    if dataset_titles is None:
        dataset_titles = datasets

    fig, axes = plt.subplots(n_ds, 1, figsize=(_MULTI_PANEL_W, _MULTI_PANEL_H * n_ds))
    if n_ds == 1:
        axes = [axes]

    for ax, cfg_name, ds_title in zip(axes, datasets, dataset_titles):
        ds, ns, ls, cols, lss, mks = _load_dataset_for_config(cfg_name)
        vals = [ds[n].get('rms_SPICE', np.nan) if n in ds else np.nan for n in ns]
        _multi_dot_panel(ax, ds, ns, ls, vals, cols, mks, '#1f77b4', ds_title,
                         show_annot=show_annot)
        ax.set_ylabel('RMS [km]', fontsize=_MULTI_FS_YLABEL)

    fig.tight_layout()
    return fig


def mpl_gof_combined_multi(sims, names, labels,
                            datasets=None,
                            dataset_titles=None,
                            show_initial=False,
                            title='Goodness of Fit — Dataset Comparison'):
    """N×3 figure: rows = one dataset each, columns = WRMS / RMS / Cost.
    Each subplot has its own y-axis (no sharey) to avoid cross-dataset scale collapse."""
    if not datasets:
        return mpl_gof_combined(sims, names, labels,
                                show_initial=show_initial, title=title)

    n_ds = len(datasets)
    if dataset_titles is None:
        dataset_titles = datasets

    metrics = [
        ('final_wrms_combined_method1_mas', 'initial_wrms_combined_method1_mas', 'WRMS [mas]'),
        ('final_rms_combined_mas',          'initial_rms_combined_mas',          'RMS [mas]'),
        ('final_cost_function',             'initial_cost_function',             'Cost Function'),
    ]

    fig, axes = plt.subplots(n_ds, len(metrics),
                             figsize=(FIG_W_DOUBLE * 1.4, FIG_H_DEFAULT * 0.8 * n_ds))
    if n_ds == 1:
        axes = axes.reshape(1, -1)

    for ri, (cfg_name, ds_title) in enumerate(zip(datasets, dataset_titles)):
        ds, ns, ls, cols, lss, mks = _load_dataset_for_config(cfg_name)
        wm = compute_wrms_and_cost(ds, ns)
        avs = [n for n in ns if n in wm]
        avl = [ls[ns.index(n)] for n in avs]
        x   = np.arange(len(avs))

        for ci, (fk, ik, ylabel) in enumerate(metrics):
            ax = axes[ri, ci]
            fv = [wm[n].get(fk) for n in avs]
            iv = [wm[n].get(ik) for n in avs]

            valid_f = [(xi, v) for xi, v in enumerate(fv) if v is not None]
            if len(valid_f) > 1:
                ax.plot([p[0] for p in valid_f], [p[1] for p in valid_f],
                        '-', color='steelblue', linewidth=1.0, alpha=0.4, zorder=1)
            if show_initial:
                valid_i = [(xi, v) for xi, v in enumerate(iv) if v is not None]
                if len(valid_i) > 1:
                    ax.plot([p[0] for p in valid_i], [p[1] for p in valid_i],
                            '--', color='lightcoral', linewidth=1.0, alpha=0.4, zorder=1)

            for xi, sn in enumerate(avs):
                c  = cols.get(sn, _color(xi))
                mk = mks.get(sn, 'o')
                if fv[xi] is not None:
                    ax.plot(xi, fv[xi], marker=mk, color=c, markersize=7,
                            markeredgecolor='black', markeredgewidth=0.5,
                            linestyle='none', zorder=2)
                if show_initial and iv[xi] is not None:
                    ax.plot(xi, iv[xi], marker=mk, color=c, markersize=5,
                            markeredgecolor='black', markeredgewidth=0.5,
                            linestyle='none', zorder=2, alpha=0.45,
                            markerfacecolor='none')

            ax.set_xticks(x)
            ax.set_xticklabels(avl, rotation=35, ha='right', fontsize=8)
            ax.set_xlim(-0.5, len(avs) - 0.5)
            if ci == 0:
                ax.set_ylabel(ds_title, fontsize=9)
            if ri == 0:
                ax.set_title(ylabel, fontsize=10)

    if show_initial:
        from matplotlib.lines import Line2D
        axes[0, 0].legend(handles=[
            Line2D([0], [0], linestyle='-',  color='steelblue',  linewidth=1.2, label='Final'),
            Line2D([0], [0], linestyle='--', color='lightcoral', linewidth=1.2, label='Initial'),
        ], fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def mpl_gof_metric_multi(sims, names, labels,
                          datasets=None,
                          dataset_titles=None,
                          metric='wrms',
                          show_initial=False,
                          title=None):
    """N×1 figure: one panel per dataset, showing a single GoF metric.

    Parameters
    ----------
    metric : 'wrms' | 'rms' | 'cost'
        Which goodness-of-fit metric to plot.
    """
    _metric_map = {
        'wrms': ('final_wrms_combined_method1_mas', 'initial_wrms_combined_method1_mas', 'WRMS [mas]'),
        'rms':  ('final_rms_combined_mas',          'initial_rms_combined_mas',          'RMS [mas]'),
        'cost': ('final_cost_function',             'initial_cost_function',             'Cost Function'),
    }
    fk, ik, ylabel = _metric_map.get(metric, _metric_map['wrms'])

    if title is None:
        title = f'{ylabel} — Dataset Comparison'

    if not datasets:
        return mpl_gof(sims, names, labels, metric=metric,
                       show_initial=show_initial, title=title)

    if dataset_titles is None:
        dataset_titles = datasets

    n_ds = len(datasets)
    fig, axes = plt.subplots(n_ds, 1, figsize=(_MULTI_PANEL_W, _MULTI_PANEL_H * n_ds))
    if n_ds == 1:
        axes = [axes]

    for ax, cfg_name, ds_title in zip(axes, datasets, dataset_titles):
        ds, ns, ls, cols, lss, mks = _load_dataset_for_config(cfg_name)
        wm = compute_wrms_and_cost(ds, ns)
        avs = [n for n in ns if n in wm]
        avl = [ls[ns.index(n)] for n in avs]
        x   = np.arange(len(avs))

        fv = [wm[n].get(fk) for n in avs]
        iv = [wm[n].get(ik) for n in avs]

        valid_f = [(xi, v) for xi, v in enumerate(fv) if v is not None]
        if len(valid_f) > 1:
            ax.plot([p[0] for p in valid_f], [p[1] for p in valid_f],
                    '-', color='steelblue', linewidth=1.0, alpha=0.4, zorder=1)
        if show_initial:
            valid_i = [(xi, v) for xi, v in enumerate(iv) if v is not None]
            if len(valid_i) > 1:
                ax.plot([p[0] for p in valid_i], [p[1] for p in valid_i],
                        '--', color='lightcoral', linewidth=1.0, alpha=0.4, zorder=1)

        for xi, sn in enumerate(avs):
            c  = cols.get(sn, _color(xi))
            mk = mks.get(sn, 'o')
            if fv[xi] is not None:
                ax.plot(xi, fv[xi], marker=mk, color=c, markersize=7,
                        markeredgecolor='black', markeredgewidth=0.5,
                        linestyle='none', zorder=2)
            if show_initial and iv[xi] is not None:
                ax.plot(xi, iv[xi], marker=mk, color=c, markersize=5,
                        markeredgecolor='black', markeredgewidth=0.5,
                        linestyle='none', zorder=2, alpha=0.45,
                        markerfacecolor='none')

        ax.set_xticks(x)
        ax.set_xticklabels(avl, rotation=35, ha='right', fontsize=_MULTI_FS_TICKS)
        ax.set_xlim(-0.5, len(avs) - 0.5)
        ax.set_ylabel(ylabel, fontsize=_MULTI_FS_YLABEL)
        ax.set_title(ds_title, fontsize=_MULTI_FS_TITLE)

    if show_initial:
        from matplotlib.lines import Line2D
        axes[0].legend(handles=[
            Line2D([0], [0], linestyle='-',  color='steelblue',  linewidth=1.2, label='Final'),
            Line2D([0], [0], linestyle='--', color='lightcoral', linewidth=1.2, label='Initial'),
        ], fontsize=_MULTI_FS_TICKS)

    fig.tight_layout()
    return fig


def mpl_formal_rms_multi(sims, names, labels,
                          datasets=None,
                          dataset_titles=None,
                          sharey=False,
                          title='Formal Error RMS — Dataset Comparison'):
    """1×N figure: total formal error RMS per simulation per dataset.

    Total formal error RMS = sqrt(mean(formal_errors_RSW_km ** 2)) over all
    R/S/W components — the scalar counterpart to rms_SPICE.
    """
    if not datasets:
        # Fallback: single-panel using current dataset
        vals = []
        for sn in names:
            sd = sims.get(sn, {})
            if 'formal_errors_RSW_km' in sd:
                vals.append(np.sqrt(np.mean(sd['formal_errors_RSW_km'] ** 2)))
            else:
                vals.append(np.nan)
        fig, ax = plt.subplots(figsize=(max(FIG_W_DOUBLE, 0.8 * len(names)), FIG_H_DEFAULT))
        _multi_dot_panel(ax, sims, names, labels, vals, _SIM_COLORS, _SIM_MARKERS, '#2ca02c', title)
        ax.set_ylabel('Formal Error RMS [km]')
        fig.tight_layout()
        return fig

    if dataset_titles is None:
        dataset_titles = datasets

    n_ds = len(datasets)
    fig, axes = plt.subplots(n_ds, 1, figsize=(_MULTI_PANEL_W, _MULTI_PANEL_H * n_ds))
    if n_ds == 1:
        axes = [axes]

    for ax, cfg_name, ds_title in zip(axes, datasets, dataset_titles):
        ds, ns, ls, cols, lss, mks = _load_dataset_for_config(cfg_name)
        vals = []
        for sn in ns:
            sd = ds.get(sn, {})
            if 'formal_errors_RSW_km' in sd:
                vals.append(np.sqrt(np.mean(sd['formal_errors_RSW_km'] ** 2)))
            else:
                vals.append(np.nan)
        _multi_dot_panel(ax, ds, ns, ls, vals, cols, mks, '#2ca02c', ds_title)
        ax.set_ylabel('Formal Error RMS [km]', fontsize=_MULTI_FS_YLABEL)

    fig.tight_layout()
    return fig


def mpl_rms_ratio_multi(sims, names, labels,
                         datasets=None,
                         dataset_titles=None,
                         title='RMS / Formal Error RMS — Dataset Comparison'):
    """N×1 figure: rms_SPICE / total formal error RMS per simulation per dataset."""
    if not datasets:
        return mpl_rms_ratio(sims, names, labels, title=title)

    if dataset_titles is None:
        dataset_titles = datasets

    n_ds = len(datasets)
    fig, axes = plt.subplots(n_ds, 1, figsize=(_MULTI_PANEL_W, _MULTI_PANEL_H * n_ds))
    if n_ds == 1:
        axes = [axes]

    for ax, cfg_name, ds_title in zip(axes, datasets, dataset_titles):
        ds, ns, ls, cols, lss, mks = _load_dataset_for_config(cfg_name)
        vals = []
        for sn in ns:
            sd = ds.get(sn, {})
            rms_spice = sd.get('rms_SPICE', np.nan)
            if 'formal_errors_RSW_km' in sd and not np.isnan(rms_spice):
                formal_rms = np.sqrt(np.mean(sd['formal_errors_RSW_km'] ** 2))
                vals.append(rms_spice / formal_rms if formal_rms > 0 else np.nan)
            else:
                vals.append(np.nan)
        _multi_dot_panel(ax, ds, ns, ls, vals, cols, mks, '#9467bd', ds_title, show_annot=False)
        ax.axhline(1.0, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
        ax.set_ylabel('RMS$_{\\mathrm{SPICE}}$ / RMS$_{\\sigma}$  [—]',
                      fontsize=_MULTI_FS_YLABEL)

    fig.tight_layout()
    return fig


# ============================================================================
# OBSERVATIONAL DATASET FIGURE FUNCTIONS
# ============================================================================

# Categorical color palette (12 qualitative colors, colorblind-aware).
_OBS_PALETTE = [
    '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78',
]


def _obs_color(i: int) -> str:
    return _OBS_PALETTE[i % len(_OBS_PALETTE)]


def mpl_obs_spice_timeseries(sims, names, labels,
                              obs_folder: str = 'Observations/AllModernJ2000',
                              title: str = r'O$-$C Residuals vs NEP097',
                              figsize=None):
    """RA / Dec residual time series vs NEP097 from all observation CSV files.

    One figure with two stacked panels (RA top, Dec bottom), each observation
    ID shown in a distinct colour.
    """
    df = _load_spice_residuals_df(obs_folder)
    if df.empty:
        print('  SKIP obs_spice_timeseries: no CSV data found.')
        return None

    if figsize is None:
        figsize = (FIG_W_DOUBLE, 6.0)
    fig, (ax_ra, ax_dec) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    ref_ids = sorted(df['ref_point_id'].unique())
    for i, rid in enumerate(ref_ids):
        mask = df['ref_point_id'] == rid
        sub  = df.loc[mask].copy()
        times_dt = convert_time_array_to_datetime(
            sub['time_j2000'].values.reshape(-1, 1))
        col = _obs_color(i)
        ax_ra.scatter(times_dt,  sub['ra_resid_arcsec'].values,
                      s=8, color=col, label=rid.replace('_', ' '), alpha=0.7)
        ax_dec.scatter(times_dt, sub['dec_resid_arcsec'].values,
                       s=8, color=col, alpha=0.7)

    ax_ra.axhline(0, color='k', linewidth=0.6, linestyle='--')
    ax_dec.axhline(0, color='k', linewidth=0.6, linestyle='--')
    ax_ra.set_ylabel('RA residual [$\'\'$]')
    ax_dec.set_ylabel('Dec residual [$\'\'$]')
    ax_dec.set_xlabel('Year')
    _apply_date_formatter(ax_dec)
    ax_ra.legend(fontsize=7, ncol=3, loc='upper right', markerscale=2)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def mpl_obs_initial_timeseries(sims, names, labels,
                                sim_name: str = None,
                                title: str = 'Residuals vs Initial Propagation',
                                figsize=None):
    """RA / Dec residual time series for the initial (pre-estimation) iteration.

    Uses *sim_name*'s ``residual_df``.  Each observation ID is coloured
    distinctly.  Layout identical to mpl_obs_spice_timeseries.
    """
    if sim_name is None and names:
        sim_name = names[0]
    if sim_name not in sims or 'residual_df' not in sims.get(sim_name, {}):
        print(f'  SKIP obs_initial_timeseries: no residual_df for {sim_name!r}.')
        return None

    df = sims[sim_name]['residual_df']
    if figsize is None:
        figsize = (FIG_W_DOUBLE, 6.0)
    fig, (ax_ra, ax_dec) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    ref_ids = sorted(df['ref_point_id'].unique())
    for i, rid in enumerate(ref_ids):
        mask = df['ref_point_id'] == rid
        t    = df.loc[mask, 'datetime']
        ra   = df.loc[mask, 'ra_residual_initial_mas'] / 1000.0   # mas → arcsec
        dec  = df.loc[mask, 'dec_residual_initial_mas'] / 1000.0
        col  = _obs_color(i)
        ax_ra.scatter(t,  ra.values,  s=8, color=col,
                      label=rid.replace('_', ' '), alpha=0.7)
        ax_dec.scatter(t, dec.values, s=8, color=col, alpha=0.7)

    ax_ra.axhline(0, color='k', linewidth=0.6, linestyle='--')
    ax_dec.axhline(0, color='k', linewidth=0.6, linestyle='--')
    ax_ra.set_ylabel('RA residual [$\'\'$]')
    ax_dec.set_ylabel('Dec residual [$\'\'$]')
    ax_dec.set_xlabel('Year')
    _apply_date_formatter(ax_dec)
    ax_ra.legend(fontsize=7, ncol=3, loc='upper right', markerscale=2)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def mpl_obs_histogram_per_id(sims, names, labels,
                              sim_name: str = None,
                              obs_folder: str = 'Observations/AllModernJ2000',
                              source: str = 'both',
                              bins: int = 30,
                              fit_gauss: bool = True,
                              title: str = 'Residual Histograms',
                              figsize=None):
    """Per-ID RA / Dec residual histograms with optional Gaussian fit.

    Parameters
    ----------
    source : {'spice', 'initial', 'both'}
        Which residuals to show.  'spice' reads O-C from CSV files (vs
        NEP097); 'initial' reads the first-iteration residuals from
        *sim_name*'s residual_df; 'both' overlays both on the same axes.

    Returns a list of (fig, ref_point_id) tuples — one figure per ID.
    """
    if sim_name is None and names:
        sim_name = names[0]

    # Load SPICE residuals if needed.
    spice_df = None
    if source in ('spice', 'both'):
        spice_df = _load_spice_residuals_df(obs_folder)
        if spice_df.empty:
            print('  WARNING obs_histogram_per_id: no SPICE residual CSV data.')
            spice_df = None

    # Load initial propagation residuals if needed.
    init_df = None
    if source in ('initial', 'both'):
        if sim_name in sims and 'residual_df' in sims.get(sim_name, {}):
            init_df = sims[sim_name]['residual_df']
        else:
            print(f'  WARNING obs_histogram_per_id: no residual_df for {sim_name!r}.')

    if spice_df is None and init_df is None:
        return None

    # Union of all IDs present in either source.
    ids_spice = set(spice_df['ref_point_id'].unique()) if spice_df is not None else set()
    ids_init  = set(init_df['ref_point_id'].unique())  if init_df  is not None else set()
    all_ids   = sorted(ids_spice | ids_init)

    if figsize is None:
        figsize = (FIG_W_DOUBLE, 4.0)

    figs = []
    for rid in all_ids:
        fig, (ax_ra, ax_dec) = plt.subplots(1, 2, figsize=figsize)

        def _plot_hist(ax, data, label, color):
            data = data[np.isfinite(data)]
            if len(data) == 0:
                return
            ax.hist(data, bins=bins, alpha=0.5, color=color,
                    label=label, density=True)
            if fit_gauss and len(data) >= 5:
                mu, sigma = sp_stats.norm.fit(data)
                xr = np.linspace(data.min(), data.max(), 200)
                ax.plot(xr, sp_stats.norm.pdf(xr, mu, sigma),
                        color=color, linewidth=1.5, linestyle='--')

        if spice_df is not None and rid in ids_spice:
            sub = spice_df.loc[spice_df['ref_point_id'] == rid]
            _plot_hist(ax_ra,  sub['ra_resid_arcsec'].values,  'NEP097', '#1f77b4')
            _plot_hist(ax_dec, sub['dec_resid_arcsec'].values, 'NEP097', '#1f77b4')

        if init_df is not None and rid in ids_init:
            sub = init_df.loc[init_df['ref_point_id'] == rid]
            ra_as  = sub['ra_residual_initial_mas'].values  / 1000.0
            dec_as = sub['dec_residual_initial_mas'].values / 1000.0
            _plot_hist(ax_ra,  ra_as,  'Init. prop.', '#d62728')
            _plot_hist(ax_dec, dec_as, 'Init. prop.', '#d62728')

        rid_display = rid.replace('_', ' ')
        ax_ra.set_xlabel('RA residual [$\'\'$]')
        ax_dec.set_xlabel('Dec residual [$\'\'$]')
        ax_ra.set_ylabel('Density')
        ax_ra.set_title('RA')
        ax_dec.set_title('Dec')
        ax_ra.legend(fontsize=8)
        fig.suptitle(f'{title} — {rid_display}')
        fig.tight_layout()
        figs.append((fig, rid))

    return figs if figs else None


# ============================================================================
# OBS-ANALYSIS FIGURE FUNCTIONS  (read from obs_analysis_data.npy)
# ============================================================================

_OBS_ANALYSIS_CACHE: dict = {}   # path → loaded dict, so we don't re-read per figure


def _load_obs_analysis(data_path: str) -> dict | None:
    """Load and cache the obs_analysis_data.npy produced by Test_Observations.py."""
    if data_path in _OBS_ANALYSIS_CACHE:
        return _OBS_ANALYSIS_CACHE[data_path]
    p = Path(data_path)
    if not p.exists():
        print(f'  SKIP: obs_analysis data not found at {data_path!r}')
        return None
    data = np.load(data_path, allow_pickle=True).item()
    for key in ('ra_spice_arcsec', 'dec_spice_arcsec',
                 'ra_tud_spice_arcsec', 'dec_tud_spice_arcsec',
                 'ra_prop_arcsec', 'dec_prop_arcsec',
                 'times_j2000'):
        if key in data:
            data[key] = np.asarray(data[key], dtype=float)
    data['mask_accepted'] = np.asarray(data['mask_accepted'], dtype=bool)
    data['mask_rejected'] = np.asarray(data['mask_rejected'], dtype=bool)
    data['set_offsets']   = np.asarray(data['set_offsets'], dtype=int)
    # Normalise excluded sub-dict if present
    excl = data.get('excluded')
    if excl is not None:
        for key in ('ra_spice_arcsec', 'dec_spice_arcsec', 'times_j2000'):
            if key in excl:
                excl[key] = np.asarray(excl[key], dtype=float)
        excl['mask_accepted'] = np.asarray(excl.get('mask_accepted', []), dtype=bool)
        excl['mask_rejected'] = np.asarray(excl.get('mask_rejected', []), dtype=bool)
        excl['set_offsets']   = np.asarray(excl.get('set_offsets', [0]), dtype=int)
    _OBS_ANALYSIS_CACHE[data_path] = data
    return data


def _iqr_clip_limits(arr: np.ndarray, k: float = 5.0):
    """Return (lo, hi) based on k × IQR around median, for sensible axis limits."""
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return -1.0, 1.0
    q25, q75 = np.percentile(arr, [25, 75])
    iqr = q75 - q25
    if iqr == 0:
        spread = max(np.std(arr) * k, 0.01)
        return q25 - spread, q75 + spread
    return q25 - k * iqr, q75 + k * iqr


def _obs_timeseries_plot(times_dt, ra, dec, mask_acc, mask_rej,
                         set_ids, set_offsets, title, figsize,
                         color_offset=0, extra_handles=None):
    """Shared implementation for accepted-by-ID / rejected-red timeseries.

    Returns fig.  Legend is placed in a dedicated bottom sub-axes panel.
    """
    from matplotlib.lines import Line2D

    if figsize is None:
        figsize = (FIG_W_DOUBLE, 8.5)
    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(3, 1, height_ratios=[4, 4, 1.3], hspace=0.12)
    ax_ra  = fig.add_subplot(gs[0])
    ax_dec = fig.add_subplot(gs[1], sharex=ax_ra)
    ax_leg = fig.add_subplot(gs[2])
    ax_leg.axis('off')

    for i, set_id in enumerate(set_ids):
        s, e = int(set_offsets[i]), int(set_offsets[i + 1])
        col  = _obs_color(i + color_offset)
        lbl  = set_id.replace('_', ' ')
        t_s  = times_dt[s:e]
        ra_s = ra[s:e];   dec_s = dec[s:e]
        m_a  = mask_acc[s:e];  m_r = mask_rej[s:e]

        if m_a.any():
            ax_ra.scatter(t_s[m_a],  ra_s[m_a],  s=6, color=col, alpha=0.7,
                          label=lbl, zorder=2)
            ax_dec.scatter(t_s[m_a], dec_s[m_a], s=6, color=col, alpha=0.7,
                           zorder=2)
        if m_r.any():
            ax_ra.scatter(t_s[m_r],  ra_s[m_r],  s=28, color='red', alpha=0.9,
                          marker='x', zorder=5, linewidths=0.8)
            ax_dec.scatter(t_s[m_r], dec_s[m_r], s=28, color='red', alpha=0.9,
                           marker='x', zorder=5, linewidths=0.8)

    n_rej = int(mask_rej.sum());  n_acc = int(mask_acc.sum())
    rej_handle = Line2D([0], [0], linestyle='none', marker='x', color='red',
                        markersize=6, markeredgewidth=0.8,
                        label=f'rejected  (n={n_rej})')

    ax_ra.axhline(0, color='k', linewidth=0.6, linestyle='--')
    ax_dec.axhline(0, color='k', linewidth=0.6, linestyle='--')
    ax_ra.set_ylabel("RA residual ['']")
    ax_dec.set_ylabel("Dec residual ['']")
    ax_dec.set_xlabel('Year')
    plt.setp(ax_ra.get_xticklabels(), visible=False)
    _apply_date_formatter(ax_dec)

    # Collect legend handles from the RA axes + the rejected marker
    h, l = ax_ra.get_legend_handles_labels()
    if extra_handles:
        for eh in extra_handles:
            h.append(eh);  l.append(eh.get_label())
    h.append(rej_handle);  l.append(rej_handle.get_label())
    ncol = min(6, max(3, (len(h) + 1) // 2))
    ax_leg.legend(h, l, loc='center', fontsize=8, ncol=ncol,
                  markerscale=1.5, frameon=True, borderaxespad=0)
    fig.tight_layout()
    return fig


def mpl_obs_analysis_spice_timeseries(
        sims, names, labels,
        data_path: str = 'Results/ObservationsAnalysis/obs_analysis_data.npy',
        title: str = r'O$-$C Residuals vs NEP097 (SPICE)',
        figsize=None):
    """RA / Dec residual time series vs SPICE (NEP097).

    Accepted observations coloured by obs-file ID; rejected marked red × on top.
    Legend is placed in a dedicated sub-panel below the plots.
    """
    data = _load_obs_analysis(data_path)
    if data is None:
        return None
    return _obs_timeseries_plot(
        data['times_dt'], data['ra_spice_arcsec'], data['dec_spice_arcsec'],
        data['mask_accepted'], data['mask_rejected'],
        list(data['set_ids']), data['set_offsets'],
        title, figsize,
    )


def mpl_obs_analysis_prop_timeseries(
        sims, names, labels,
        data_path: str = 'Results/ObservationsAnalysis/obs_analysis_data.npy',
        title: str = r'O$-$C Residuals vs Numerical Propagation',
        figsize=None):
    """RA / Dec residual time series vs the numerical propagation.

    Only shows observations that fall within the propagation time coverage.
    Accepted observations coloured by obs-file ID; rejected marked red × on top.
    Legend is placed in a dedicated sub-panel below the plots.
    """
    data = _load_obs_analysis(data_path)
    if data is None:
        return None

    t_min = float(data.get('prop_epoch_min', -np.inf))
    t_max = float(data.get('prop_epoch_max',  np.inf))
    t_j2000 = np.asarray(data['times_j2000'], dtype=float)
    in_arc  = (t_j2000 >= t_min) & (t_j2000 <= t_max)
    n_outside = int((~in_arc).sum())

    # Build masked arrays (set out-of-arc to NaN so they don't plot)
    ra_plot  = np.where(in_arc, data['ra_prop_arcsec'],  np.nan)
    dec_plot = np.where(in_arc, data['dec_prop_arcsec'], np.nan)

    # Suppress rejected markers for out-of-arc points too
    mask_rej_plot = data['mask_rejected'] & in_arc
    mask_acc_plot = data['mask_accepted'] & in_arc

    extra_note = (f'  [{n_outside} obs outside propagation arc hidden]'
                  if n_outside else '')

    return _obs_timeseries_plot(
        data['times_dt'], ra_plot, dec_plot,
        mask_acc_plot, mask_rej_plot,
        list(data['set_ids']), data['set_offsets'],
        title + extra_note, figsize,
    )


def mpl_obs_analysis_histogram(
        sims, names, labels,
        data_path: str = 'Results/ObservationsAnalysis/obs_analysis_data.npy',
        bins: int = 50,
        fit_gauss: bool = True,
        accepted_only: bool = True,
        iqr_k: float = 5.0,
        title: str = 'Residual Histograms: SPICE vs Propagation',
        figsize=None):
    """RA / Dec residual histograms — SPICE (NEP097) and propagation overlaid.

    Propagation residuals are IQR-clipped to the same x-range as SPICE
    (pre-propagation-arc observations extrapolate wildly and would otherwise
    dominate the axis scale).  The number of clipped points is annotated.

    Parameters
    ----------
    accepted_only : bool
        If True (default), exclude rejected observations from both histograms.
    iqr_k : float
        k-factor for IQR clipping applied to the propagation residuals.
    """
    data = _load_obs_analysis(data_path)
    if data is None:
        return None

    # Propagation coverage mask (exclude out-of-arc extrapolated values)
    t_min   = float(data.get('prop_epoch_min', -np.inf))
    t_max   = float(data.get('prop_epoch_max',  np.inf))
    t_j2000 = np.asarray(data['times_j2000'], dtype=float)
    in_arc  = (t_j2000 >= t_min) & (t_j2000 <= t_max)

    base_mask = data['mask_accepted'] if accepted_only else np.ones(len(in_arc), dtype=bool)
    n_tag = 'accepted' if accepted_only else 'all'

    ra_spice  = data['ra_spice_arcsec'][base_mask]
    dec_spice = data['dec_spice_arcsec'][base_mask]
    # Propagation: only within arc
    prop_mask = base_mask & in_arc
    ra_prop   = data['ra_prop_arcsec'][prop_mask]
    dec_prop  = data['dec_prop_arcsec'][prop_mask]
    n_spice   = int(base_mask.sum())
    n_prop    = int(prop_mask.sum())
    n_outside = n_spice - n_prop

    _C_SPICE = '#1f77b4'
    _C_PROP  = '#d62728'

    if figsize is None:
        figsize = (FIG_W_DOUBLE, FIG_H_DEFAULT)
    fig, (ax_ra, ax_dec) = plt.subplots(1, 2, figsize=figsize)

    def _plot_hist(ax, spice, prop, xlabel, n_outside_note):
        spice = spice[np.isfinite(spice)]
        prop  = prop[np.isfinite(prop)]
        if len(spice) == 0:
            return

        # x-range from SPICE (reliable for all epochs)
        lo, hi = _iqr_clip_limits(spice, k=iqr_k)
        prop_in_range = prop[(prop >= lo) & (prop <= hi)]
        n_prop_clipped = len(prop) - len(prop_in_range)

        ax.hist(spice, bins=bins, range=(lo, hi), alpha=0.5, color=_C_SPICE,
                density=True, label='SPICE (NEP097)')
        if len(prop_in_range) >= 5:
            ax.hist(prop_in_range, bins=bins, range=(lo, hi), alpha=0.5,
                    color=_C_PROP, density=True, label='Propagated')

        if fit_gauss:
            for dat, col in [(spice, _C_SPICE), (prop_in_range, _C_PROP)]:
                if len(dat) >= 10:
                    mu, sigma = sp_stats.norm.fit(dat)
                    xr = np.linspace(lo, hi, 400)
                    ax.plot(xr, sp_stats.norm.pdf(xr, mu, sigma),
                            color=col, linewidth=1.8, linestyle='--',
                            label=f'  μ={mu:.3f}''"'', σ={sigma:.3f}''"')

        ax.axvline(0, color='k', linewidth=0.8, linestyle=':')
        ax.set_xlim(lo, hi)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Density')
        note_parts = []
        if n_outside_note:
            note_parts.append(f'{n_outside_note} obs outside prop. arc')
        if n_prop_clipped:
            note_parts.append(f'{n_prop_clipped} prop. pts beyond ±{iqr_k}×IQR')
        if note_parts:
            ax.annotate('\n'.join(note_parts), xy=(0.02, 0.97), xycoords='axes fraction',
                        fontsize=7, va='top', color='grey')
        ax.legend(fontsize=8)

    _plot_hist(ax_ra,  ra_spice, ra_prop,  "RA residual ['']",  n_outside)
    _plot_hist(ax_dec, dec_spice, dec_prop, "Dec residual ['']", n_outside)
    ax_ra.set_title('RA')
    ax_dec.set_title('Dec')
    fig.suptitle(f'{title}  ({n_tag}, SPICE n={n_spice}, prop n={n_prop})')
    fig.tight_layout()
    return fig


def mpl_obs_analysis_excluded_spice_timeseries(
        sims, names, labels,
        data_path: str = 'Results/ObservationsAnalysis/obs_analysis_data.npy',
        title: str = r'O$-$C Residuals vs NEP097 — Excluded Files',
        figsize=None):
    """RA / Dec SPICE residuals for files NOT in file_names.json.

    Useful for understanding why those files were excluded: large systematics,
    poor data quality, or outlier epochs stand out clearly here.
    Each excluded file gets its own colour.  Rejected (internally by the filter)
    are marked with red ×.
    """
    data = _load_obs_analysis(data_path)
    if data is None:
        return None
    excl = data.get('excluded')
    if excl is None or len(excl.get('set_ids', [])) == 0:
        print('  SKIP: no excluded files recorded in obs_analysis_data')
        return None
    return _obs_timeseries_plot(
        excl['times_dt'], excl['ra_spice_arcsec'], excl['dec_spice_arcsec'],
        excl['mask_accepted'], excl['mask_rejected'],
        list(excl['set_ids']), excl['set_offsets'],
        title, figsize,
    )


def mpl_obs_analysis_all_obs_combined(
        sims, names, labels,
        data_path: str = 'Results/ObservationsAnalysis/obs_analysis_data.npy',
        title: str = r'O$-$C Residuals vs NEP097 — All Observations',
        figsize=None):
    """RA / Dec SPICE residual time series for ALL files in the folder.

    Included files (in file_names.json) are coloured by obs-file ID.
    Excluded files are shown in a distinctly different colour palette (warm
    tones starting after the included file colours) so they stand out.
    A separate legend entry marks the boundary between the two groups.
    Rejected observations (from either group) are shown as red ×.
    """
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches

    data = _load_obs_analysis(data_path)
    if data is None:
        return None
    excl = data.get('excluded')

    # ── merge included + excluded arrays ────────────────────────────────────
    inc_ids  = list(data['set_ids'])
    inc_off  = data['set_offsets']
    inc_times = data['times_dt']
    inc_ra    = data['ra_spice_arcsec']
    inc_dec   = data['dec_spice_arcsec']
    inc_acc   = data['mask_accepted']
    inc_rej   = data['mask_rejected']

    has_excl = excl is not None and len(excl.get('set_ids', [])) > 0
    if has_excl:
        excl_ids  = list(excl['set_ids'])
        excl_off  = excl['set_offsets']
        excl_times = excl['times_dt']
        excl_ra    = excl['ra_spice_arcsec']
        excl_dec   = excl['dec_spice_arcsec']
        excl_acc   = excl['mask_accepted']
        excl_rej   = excl['mask_rejected']
    else:
        excl_ids = []

    n_inc = len(inc_ids)

    if figsize is None:
        figsize = (FIG_W_DOUBLE, 8.5)
    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(3, 1, height_ratios=[4, 4, 1.5], hspace=0.12)
    ax_ra  = fig.add_subplot(gs[0])
    ax_dec = fig.add_subplot(gs[1], sharex=ax_ra)
    ax_leg = fig.add_subplot(gs[2])
    ax_leg.axis('off')

    def _scatter(ax, times, ra_or_dec, m_a, m_r, col, lbl):
        if m_a.any():
            ax.scatter(times[m_a], ra_or_dec[m_a], s=6, color=col, alpha=0.7,
                       label=lbl, zorder=2)
        if m_r.any():
            ax.scatter(times[m_r], ra_or_dec[m_r], s=28, color='red', alpha=0.9,
                       marker='x', zorder=5, linewidths=0.8)

    # Included files
    for i, set_id in enumerate(inc_ids):
        s, e = int(inc_off[i]), int(inc_off[i + 1])
        col  = _obs_color(i)
        lbl  = set_id.replace('_', ' ')
        _scatter(ax_ra,  inc_times[s:e], inc_ra[s:e],  inc_acc[s:e], inc_rej[s:e], col, lbl)
        _scatter(ax_dec, inc_times[s:e], inc_dec[s:e], inc_acc[s:e], inc_rej[s:e], col, None)

    # Excluded files — colour offset so they use a distinct palette region
    if has_excl:
        for j, set_id in enumerate(excl_ids):
            s, e = int(excl_off[j]), int(excl_off[j + 1])
            col  = _obs_color(n_inc + j + 5)   # offset to avoid colour clash
            lbl  = f'{set_id.replace("_", " ")} [excl.]'
            _scatter(ax_ra,  excl_times[s:e], excl_ra[s:e],  excl_acc[s:e], excl_rej[s:e], col, lbl)
            _scatter(ax_dec, excl_times[s:e], excl_dec[s:e], excl_acc[s:e], excl_rej[s:e], col, None)

    n_acc_tot = int(inc_acc.sum()) + (int(excl_acc.sum()) if has_excl else 0)
    n_rej_tot = int(inc_rej.sum()) + (int(excl_rej.sum()) if has_excl else 0)

    rej_handle = Line2D([0], [0], linestyle='none', marker='x', color='red',
                        markersize=6, markeredgewidth=0.8,
                        label=f'rejected  (n={n_rej_tot})')

    ax_ra.axhline(0, color='k', linewidth=0.6, linestyle='--')
    ax_dec.axhline(0, color='k', linewidth=0.6, linestyle='--')
    ax_ra.set_ylabel("RA residual ['']")
    ax_dec.set_ylabel("Dec residual ['']")
    ax_dec.set_xlabel('Year')
    plt.setp(ax_ra.get_xticklabels(), visible=False)
    _apply_date_formatter(ax_dec)

    h, l = ax_ra.get_legend_handles_labels()
    h.append(rej_handle);  l.append(rej_handle.get_label())
    ncol = min(6, max(3, (len(h) + 1) // 2))
    ax_leg.legend(h, l, loc='center', fontsize=8, ncol=ncol,
                  markerscale=1.5, frameon=True, borderaxespad=0)
    fig.tight_layout()
    return fig


def mpl_obs_analysis_n_obs_datetime(
        sims, names, labels,
        data_path: str = 'Results/ObservationsAnalysis/obs_analysis_data.npy',
        bin_years: int = 1,
        title: str = 'Observations per Year',
        figsize=None):
    """Stacked bar chart of accepted and rejected observations per year.

    Parameters
    ----------
    bin_years : int
        Width of each time bin in years.  Default 1 (annual).
        Use 5 for a coarser view over the full 1963–2025 arc.
    """
    data = _load_obs_analysis(data_path)
    if data is None:
        return None

    times_dt = data['times_dt']
    mask_acc = data['mask_accepted']
    mask_rej = data['mask_rejected']

    years = np.array([t.year for t in times_dt])
    yr_min = int(years.min());  yr_max = int(years.max())

    # Build bins aligned to multiples of bin_years
    if bin_years == 1:
        bins = np.arange(yr_min, yr_max + 2)
    else:
        start = (yr_min // bin_years) * bin_years
        bins  = np.arange(start, yr_max + bin_years + 1, bin_years)

    centers    = (bins[:-1] + bins[1:]) / 2.0
    widths     = np.diff(bins) * 0.85

    counts_acc = np.array([np.sum((years >= bins[k]) & (years < bins[k+1]) & mask_acc)
                           for k in range(len(bins) - 1)])
    counts_rej = np.array([np.sum((years >= bins[k]) & (years < bins[k+1]) & mask_rej)
                           for k in range(len(bins) - 1)])

    if figsize is None:
        figsize = (FIG_W_DOUBLE, FIG_H_DEFAULT)
    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(centers, counts_acc, width=widths, color='#1f77b4', alpha=0.8,
           label=f'accepted  (n={int(mask_acc.sum())})')
    ax.bar(centers, counts_rej, bottom=counts_acc, width=widths, color='red',
           alpha=0.8, label=f'rejected  (n={int(mask_rej.sum())})')

    ax.set_xlabel('Year')
    ax.set_ylabel('Number of observations')
    ax.set_xlim(bins[0] - 0.5, bins[-1] + 0.5)
    bin_label = f'{bin_years}-year bins' if bin_years > 1 else 'annual'
    ax.set_title(f'{title}  ({bin_label})')
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def mpl_obs_analysis_n_obs_id(
        sims, names, labels,
        data_path: str = 'Results/ObservationsAnalysis/obs_analysis_data.npy',
        title: str = 'Observations per Dataset ID',
        figsize=None):
    """Horizontal stacked bar chart: accepted + rejected count per observation file.

    Bars are sorted by total count (largest at top).
    """
    data = _load_obs_analysis(data_path)
    if data is None:
        return None

    set_ids     = list(data['set_ids'])
    set_offsets = data['set_offsets']
    mask_acc    = data['mask_accepted']
    mask_rej    = data['mask_rejected']

    counts_acc = []
    counts_rej = []
    for i in range(len(set_ids)):
        s, e = int(set_offsets[i]), int(set_offsets[i + 1])
        counts_acc.append(int(mask_acc[s:e].sum()))
        counts_rej.append(int(mask_rej[s:e].sum()))

    counts_acc = np.array(counts_acc)
    counts_rej = np.array(counts_rej)
    totals     = counts_acc + counts_rej

    # Sort by total count (descending → top of chart = most obs)
    order      = np.argsort(totals)
    set_ids_s  = [set_ids[k]  for k in order]
    counts_acc = counts_acc[order]
    counts_rej = counts_rej[order]
    totals     = totals[order]

    short_ids = [sid.replace('_', ' ') for sid in set_ids_s]
    y         = np.arange(len(set_ids_s))

    if figsize is None:
        height = max(4.0, 0.32 * len(set_ids_s))
        figsize = (FIG_W_DOUBLE, height)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(y, counts_acc, color='#1f77b4', alpha=0.85,
            label=f'accepted  (total={int(counts_acc.sum())})')
    ax.barh(y, counts_rej, left=counts_acc, color='red', alpha=0.85,
            label=f'rejected  (total={int(counts_rej.sum())})')

    ax.set_yticks(y)
    ax.set_yticklabels(short_ids, fontsize=8)
    ax.set_xlabel('Number of observations')
    ax.set_title(title)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim(0, totals.max() * 1.12)
    fig.tight_layout()
    return fig


def mpl_obs_analysis_all_in_folder(
        sims, names, labels,
        data_path: str = 'Results/ObservationsAnalysis/obs_analysis_data.npy',
        title: str = r'O$-$C Residuals vs NEP097 — All Files in Folder',
        figsize=None):
    """RA / Dec SPICE residuals for every file in the observations folder.

    Two-colour view: files in file_names.json (included) = blue;
    files NOT in file_names.json (excluded) = red.
    Filtered (rejected) observations are shown with × markers in dark red.
    """
    from matplotlib.lines import Line2D

    data = _load_obs_analysis(data_path)
    if data is None:
        return None
    excl = data.get('excluded')

    inc_times = data['times_dt']
    inc_ra    = data['ra_spice_arcsec']
    inc_dec   = data['dec_spice_arcsec']
    inc_acc   = data['mask_accepted']
    inc_rej   = data['mask_rejected']

    has_excl  = excl is not None and len(excl.get('set_ids', [])) > 0

    _C_INC  = '#1f77b4'
    _C_EXCL = '#d62728'
    _C_REJ  = '#8b0000'

    if figsize is None:
        figsize = (FIG_W_DOUBLE, 7.0)
    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(3, 1, height_ratios=[4, 4, 0.8], hspace=0.12)
    ax_ra  = fig.add_subplot(gs[0])
    ax_dec = fig.add_subplot(gs[1], sharex=ax_ra)
    ax_leg = fig.add_subplot(gs[2])
    ax_leg.axis('off')

    n_acc = int(inc_acc.sum());  n_rej = int(inc_rej.sum())
    n_inc = n_acc + n_rej

    if inc_acc.any():
        ax_ra.scatter(inc_times[inc_acc], inc_ra[inc_acc],   s=5, color=_C_INC, alpha=0.6,
                      label=f'included  (n={n_inc})', zorder=2)
        ax_dec.scatter(inc_times[inc_acc], inc_dec[inc_acc], s=5, color=_C_INC, alpha=0.6,
                       zorder=2)
    if inc_rej.any():
        ax_ra.scatter(inc_times[inc_rej], inc_ra[inc_rej],   s=22, color=_C_REJ, alpha=0.85,
                      marker='x', zorder=5, linewidths=0.8,
                      label=f'rejected  (n={n_rej})')
        ax_dec.scatter(inc_times[inc_rej], inc_dec[inc_rej], s=22, color=_C_REJ, alpha=0.85,
                       marker='x', zorder=5, linewidths=0.8)

    n_excl = 0
    if has_excl:
        excl_times = excl['times_dt']
        excl_ra    = excl['ra_spice_arcsec']
        excl_dec   = excl['dec_spice_arcsec']
        n_excl     = len(excl_ra)
        ax_ra.scatter(excl_times, excl_ra,   s=5, color=_C_EXCL, alpha=0.6,
                      label=f'excluded  (n={n_excl})', zorder=3)
        ax_dec.scatter(excl_times, excl_dec, s=5, color=_C_EXCL, alpha=0.6, zorder=3)

    ax_ra.axhline(0,  color='k', linewidth=0.6, linestyle='--')
    ax_dec.axhline(0, color='k', linewidth=0.6, linestyle='--')
    ax_ra.set_ylabel("RA residual ['']")
    ax_dec.set_ylabel("Dec residual ['']")
    ax_dec.set_xlabel('Year')
    plt.setp(ax_ra.get_xticklabels(), visible=False)
    _apply_date_formatter(ax_dec)

    h, l = ax_ra.get_legend_handles_labels()
    ax_leg.legend(h, l, loc='center', fontsize=9, ncol=len(h),
                  markerscale=1.5, frameon=True)
    fig.tight_layout()
    return fig


def _obs_by_id_figure(times_dt, ra, dec, mask_acc, mask_rej,
                      set_ids, set_offsets, title,
                      show_rejected=True, figsize=None,
                      axes_fontsize=None, tick_fontsize=None,
                      legend_fontsize=8, legend_observatory_names=False,
                      show_legend=True,
                      match_y_axes=False,
                      y_unit_label="''",
                      y_tick_step=None,
                      y_lim=None,
                      panel_hspace=0.08):
    """Shared implementation: RA/Dec scatter coloured by ID, legend panel on the right.

    Parameters
    ----------
    show_rejected : bool
        If True, rejected observations are overlaid as red × markers.
        If False, only accepted observations are plotted.

    Returns a single figure (plots left, 2-column legend right).
    """
    from matplotlib.lines import Line2D

    n_acc = int(mask_acc.sum());  n_rej = int(mask_rej.sum())

    if figsize is None:
        figsize = (FIG_W_DOUBLE * 1.6, 7.0)

    fig = plt.figure(figsize=figsize)
    if show_legend:
        gs     = fig.add_gridspec(2, 2, width_ratios=[3, 1],
                                  hspace=panel_hspace, wspace=0.04)
        ax_ra  = fig.add_subplot(gs[0, 0])
        ax_dec = fig.add_subplot(gs[1, 0], sharex=ax_ra)
        ax_leg = fig.add_subplot(gs[:, 1])
        ax_leg.axis('off')
    else:
        gs     = fig.add_gridspec(2, 1, hspace=panel_hspace)
        ax_ra  = fig.add_subplot(gs[0, 0])
        ax_dec = fig.add_subplot(gs[1, 0], sharex=ax_ra)
        ax_leg = None

    def _legend_label(sid: str) -> str:
        if legend_observatory_names:
            mpc_code = sid.split('_', 1)[0]
            return _get_observatory_name(mpc_code)
        return sid.replace('_', ' ')

    # Group ID indices by legend label so duplicates collapse to one entry.
    label_to_color = {}
    legend_order   = []
    for i, set_id in enumerate(set_ids):
        lbl = _legend_label(set_id)
        if lbl not in label_to_color:
            label_to_color[lbl] = _obs_color(i)
            legend_order.append(lbl)

    legend_handles = []
    for i, set_id in enumerate(set_ids):
        s, e  = int(set_offsets[i]), int(set_offsets[i + 1])
        col   = _obs_color(i)
        lbl   = _legend_label(set_id)
        t_s   = times_dt[s:e]
        ra_s  = ra[s:e];   dec_s = dec[s:e]
        m_a   = mask_acc[s:e];  m_r = mask_rej[s:e]
        if m_a.any():
            ax_ra.scatter(t_s[m_a],  ra_s[m_a],  s=6, color=col, alpha=0.7, zorder=2)
            ax_dec.scatter(t_s[m_a], dec_s[m_a], s=6, color=col, alpha=0.7, zorder=2)
            legend_handles.append(
                Line2D([0], [0], linestyle='none', marker='o', color=col,
                       markersize=5, label=lbl))
        if show_rejected and m_r.any():
            ax_ra.scatter(t_s[m_r],  ra_s[m_r],  s=28, color='red', alpha=0.9,
                          marker='x', zorder=5, linewidths=0.8)
            ax_dec.scatter(t_s[m_r], dec_s[m_r], s=28, color='red', alpha=0.9,
                           marker='x', zorder=5, linewidths=0.8)

    if legend_observatory_names:
        legend_handles = [
            Line2D([0], [0], linestyle='none', marker='o',
                   color=label_to_color[lbl], markersize=5, label=lbl)
            for lbl in legend_order
        ]
    if show_rejected:
        legend_handles.append(
            Line2D([0], [0], linestyle='none', marker='x', color='red',
                   markersize=6, markeredgewidth=0.8, label=f'rejected  (n={n_rej})'))

    ax_ra.axhline(0,  color='k', linewidth=0.6, linestyle='--')
    ax_dec.axhline(0, color='k', linewidth=0.6, linestyle='--')
    ax_ra.set_ylabel(f"RA residual [{y_unit_label}]",
                    **(({'fontsize': axes_fontsize}) if axes_fontsize is not None else {}))
    ax_dec.set_ylabel(f"Dec residual [{y_unit_label}]",
                     **(({'fontsize': axes_fontsize}) if axes_fontsize is not None else {}))
    if match_y_axes:
        lo_ra, hi_ra = ax_ra.get_ylim()
        lo_de, hi_de = ax_dec.get_ylim()
        lo, hi = min(lo_ra, lo_de), max(hi_ra, hi_de)
        ax_ra.set_ylim(lo, hi)
        ax_dec.set_ylim(lo, hi)
        # Lock both panels to the same tick locations.
        ax_dec.set_yticks(ax_ra.get_yticks())
        ax_dec.set_ylim(lo, hi)
    if y_lim is not None:
        ax_ra.set_ylim(*y_lim)
        ax_dec.set_ylim(*y_lim)
    if y_tick_step is not None:
        from matplotlib.ticker import MultipleLocator
        ax_ra.yaxis.set_major_locator(MultipleLocator(y_tick_step))
        ax_dec.yaxis.set_major_locator(MultipleLocator(y_tick_step))
    ax_dec.set_xlabel('Year',
                     **(({'fontsize': axes_fontsize}) if axes_fontsize is not None else {}))
    if tick_fontsize is not None:
        ax_ra.tick_params(axis='y', labelsize=tick_fontsize)
        ax_dec.tick_params(axis='both', labelsize=tick_fontsize)
    plt.setp(ax_ra.get_xticklabels(), visible=False)
    _apply_date_formatter(ax_dec)

    # ── Right-panel legend (1 column) ─────────────────────────────────────────
    if ax_leg is not None:
        ax_leg.legend(legend_handles, [h.get_label() for h in legend_handles],
                      loc='center left', fontsize=legend_fontsize, ncol=1,
                      markerscale=1.5, frameon=True, borderaxespad=0.2)
    fig.tight_layout()
    return fig


def mpl_obs_analysis_spice_by_id(
        sims, names, labels,
        data_path: str = 'Results/ObservationsAnalysis/obs_analysis_data.npy',
        title: str = r'O$-$C Residuals vs NEP097 — Included Files by ID',
        figsize=None):
    """RA / Dec SPICE residuals, coloured by obs-file ID.

    Legend placed in a panel to the right of the plots (2 columns).
    Rejected observations shown as red ×.
    """
    data = _load_obs_analysis(data_path)
    if data is None:
        return None
    return _obs_by_id_figure(
        data['times_dt'], data['ra_spice_arcsec'], data['dec_spice_arcsec'],
        data['mask_accepted'], data['mask_rejected'],
        list(data['set_ids']), data['set_offsets'],
        title, show_rejected=True, figsize=figsize,
    )


def mpl_obs_analysis_spice_by_id_accepted(
        sims, names, labels,
        data_path: str = 'Results/ObservationsAnalysis/obs_analysis_data.npy',
        title: str = r'O$-$C Residuals vs NEP097 — Accepted Observations by ID',
        figsize=None,
        axes_fontsize=None,
        tick_fontsize=None,
        legend_fontsize=8,
        legend_observatory_names=False,
        show_legend=True,
        match_y_axes=False,
        y_unit_label="''",
        y_tick_step=None,
        y_lim=None,
        panel_hspace=0.08):
    """RA / Dec SPICE residuals, coloured by obs-file ID — accepted only.

    Identical layout to mpl_obs_analysis_spice_by_id but rejected observations
    are omitted entirely (not marked with ×).
    """
    data = _load_obs_analysis(data_path)
    if data is None:
        return None
    return _obs_by_id_figure(
        data['times_dt'], data['ra_spice_arcsec'], data['dec_spice_arcsec'],
        data['mask_accepted'], data['mask_rejected'],
        list(data['set_ids']), data['set_offsets'],
        title, show_rejected=False, figsize=figsize,
        axes_fontsize=axes_fontsize,
        tick_fontsize=tick_fontsize,
        legend_fontsize=legend_fontsize,
        legend_observatory_names=legend_observatory_names,
        show_legend=show_legend,
        match_y_axes=match_y_axes,
        y_unit_label=y_unit_label,
        y_tick_step=y_tick_step,
        y_lim=y_lim,
        panel_hspace=panel_hspace,
    )


def mpl_obs_analysis_prop_by_id(
        sims, names, labels,
        data_path: str = 'Results/ObservationsAnalysis/obs_analysis_data.npy',
        title: str = r'O$-$C Residuals vs Numerical Propagation — Included Files by ID',
        figsize=None):
    """RA / Dec residuals vs the numerical propagation, coloured by obs-file ID.

    Uses ``ra_prop_arcsec`` / ``dec_prop_arcsec`` from obs_analysis_data.
    Same side-by-side layout (plots left, 1-col legend right).
    Observations outside the propagation arc are masked to NaN so they do
    not corrupt the axis scale.
    """
    data = _load_obs_analysis(data_path)
    if data is None:
        return None
    if 'ra_prop_arcsec' not in data:
        print('  SKIP: ra_prop_arcsec not found in obs_analysis_data')
        return None

    t_min   = float(data.get('prop_epoch_min', -np.inf))
    t_max   = float(data.get('prop_epoch_max',  np.inf))
    t_j2000 = np.asarray(data['times_j2000'], dtype=float)
    in_arc  = (t_j2000 >= t_min) & (t_j2000 <= t_max)

    ra_plot  = np.where(in_arc, np.asarray(data['ra_prop_arcsec'],  dtype=float), np.nan)
    dec_plot = np.where(in_arc, np.asarray(data['dec_prop_arcsec'], dtype=float), np.nan)
    mask_acc = data['mask_accepted'] & in_arc
    mask_rej = data['mask_rejected'] & in_arc

    n_outside = int((~in_arc).sum())
    extra = f'  [{n_outside} obs outside propagation arc hidden]' if n_outside else ''

    return _obs_by_id_figure(
        data['times_dt'], ra_plot, dec_plot,
        mask_acc, mask_rej,
        list(data['set_ids']), data['set_offsets'],
        title + extra, show_rejected=True, figsize=figsize,
    )


def mpl_obs_analysis_spice_filtered_highlight(
        sims, names, labels,
        data_path: str = 'Results/ObservationsAnalysis/obs_analysis_data.npy',
        title: str = r'O$-$C Residuals vs NEP097 — Accepted / Rejected',
        figsize=None):
    """RA / Dec SPICE residuals: accepted = blue, rejected = red ×.

    All included files shown without per-ID colouring so that the
    accepted / rejected split is immediately visible.
    """
    from matplotlib.lines import Line2D

    data = _load_obs_analysis(data_path)
    if data is None:
        return None

    times_dt = data['times_dt']
    ra       = data['ra_spice_arcsec']
    dec      = data['dec_spice_arcsec']
    mask_acc = data['mask_accepted']
    mask_rej = data['mask_rejected']

    _C_ACC = '#1f77b4'

    if figsize is None:
        figsize = (FIG_W_DOUBLE, 7.0)
    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(3, 1, height_ratios=[4, 4, 0.8], hspace=0.12)
    ax_ra  = fig.add_subplot(gs[0])
    ax_dec = fig.add_subplot(gs[1], sharex=ax_ra)
    ax_leg = fig.add_subplot(gs[2])
    ax_leg.axis('off')

    n_acc = int(mask_acc.sum());  n_rej = int(mask_rej.sum())

    if mask_acc.any():
        ax_ra.scatter(times_dt[mask_acc], ra[mask_acc],   s=5, color=_C_ACC, alpha=0.6,
                      label=f'accepted  (n={n_acc})', zorder=2)
        ax_dec.scatter(times_dt[mask_acc], dec[mask_acc], s=5, color=_C_ACC, alpha=0.6,
                       zorder=2)
    if mask_rej.any():
        ax_ra.scatter(times_dt[mask_rej], ra[mask_rej],   s=22, color='red', alpha=0.85,
                      marker='x', zorder=5, linewidths=0.8,
                      label=f'rejected  (n={n_rej})')
        ax_dec.scatter(times_dt[mask_rej], dec[mask_rej], s=22, color='red', alpha=0.85,
                       marker='x', zorder=5, linewidths=0.8)

    ax_ra.axhline(0,  color='k', linewidth=0.6, linestyle='--')
    ax_dec.axhline(0, color='k', linewidth=0.6, linestyle='--')
    ax_ra.set_ylabel("RA residual ['']")
    ax_dec.set_ylabel("Dec residual ['']")
    ax_dec.set_xlabel('Year')
    plt.setp(ax_ra.get_xticklabels(), visible=False)
    _apply_date_formatter(ax_dec)

    h, l = ax_ra.get_legend_handles_labels()
    ax_leg.legend(h, l, loc='center', fontsize=9, ncol=len(h),
                  markerscale=1.5, frameon=True)
    fig.tight_layout()
    return fig


def mpl_obs_analysis_spice_per_file(
        sims, names, labels,
        data_path: str = 'Results/ObservationsAnalysis/obs_analysis_data.npy',
        title_prefix: str = r'O$-$C Residuals vs NEP097',
        figsize=None):
    """RA / Dec SPICE residuals for each included file — one figure per file.

    Returns a list of (fig, set_id) tuples so the caller (main) saves each
    as a separate PDF.  Use ``subfolder`` in the config entry to route all
    these PDFs into a dedicated sub-directory.
    """
    data = _load_obs_analysis(data_path)
    if data is None:
        return None

    set_ids     = list(data['set_ids'])
    set_offsets = data['set_offsets']
    times_dt    = data['times_dt']
    ra          = data['ra_spice_arcsec']
    dec         = data['dec_spice_arcsec']
    mask_acc    = data['mask_accepted']
    mask_rej    = data['mask_rejected']

    if figsize is None:
        figsize = (FIG_W_DOUBLE, 5.5)

    figs = []
    for i, set_id in enumerate(set_ids):
        s, e  = int(set_offsets[i]), int(set_offsets[i + 1])
        col   = _obs_color(i)
        t_s   = times_dt[s:e]
        ra_s  = ra[s:e];    dec_s = dec[s:e]
        m_a   = mask_acc[s:e];  m_r = mask_rej[s:e]

        fig, (ax_ra, ax_dec) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                             gridspec_kw={'hspace': 0.08})
        if m_a.any():
            ax_ra.scatter(t_s[m_a],  ra_s[m_a],  s=8, color=col, alpha=0.8, zorder=2)
            ax_dec.scatter(t_s[m_a], dec_s[m_a], s=8, color=col, alpha=0.8, zorder=2)
        if m_r.any():
            ax_ra.scatter(t_s[m_r],  ra_s[m_r],  s=30, color='red', alpha=0.9,
                          marker='x', zorder=5, linewidths=1.0,
                          label=f'rejected (n={int(m_r.sum())})')
            ax_dec.scatter(t_s[m_r], dec_s[m_r], s=30, color='red', alpha=0.9,
                           marker='x', zorder=5, linewidths=1.0)
            ax_ra.legend(fontsize=8, loc='upper right')

        ax_ra.axhline(0,  color='k', linewidth=0.6, linestyle='--')
        ax_dec.axhline(0, color='k', linewidth=0.6, linestyle='--')
        ax_ra.set_ylabel("RA residual ['']")
        ax_dec.set_ylabel("Dec residual ['']")
        ax_dec.set_xlabel('Year')
        plt.setp(ax_ra.get_xticklabels(), visible=False)
        _apply_date_formatter(ax_dec)
        fig.tight_layout()
        figs.append((fig, set_id))

    return figs if figs else None


def mpl_obs_analysis_combined_count(
        sims, names, labels,
        data_path: str = 'Results/ObservationsAnalysis/obs_analysis_data.npy',
        bin_years: int = 1,
        title: str = 'Observation Count',
        figsize=None,
        which: str = 'both',
        axes_fontsize: int | None = None,
        tick_fontsize: int | None = None,
        legend_fontsize: int = 8,
        suppress_panel_titles: bool = False,
        legend_observatory_names: bool = False):
    """Combined count figure: stacked-by-file bar per year (top) + bar per file (bottom).

    Parameters
    ----------
    which : {'both', 'year', 'file'}
        'both'  → side-by-side per-year (left) + per-file (right) panel (default).
        'year'  → only the per-year stacked-bar panel.
        'file'  → only the per-file ranked-bar panel.

    Returns ``[(main_fig, 'count'), (legend_fig, 'legend')]`` so the colour
    legend is saved as a separate PDF and can be composed as a subfigure in
    the document independently of the count panels.
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    data = _load_obs_analysis(data_path)
    if data is None:
        return None

    set_ids     = list(data['set_ids'])
    set_offsets = data['set_offsets']
    times_dt    = data['times_dt']
    mask_acc    = data['mask_accepted']
    mask_rej    = data['mask_rejected']

    n_ids = len(set_ids)
    years = np.array([t.year for t in times_dt])
    yr_min = int(years.min());  yr_max = int(years.max())

    if bin_years == 1:
        bins = np.arange(yr_min, yr_max + 2)
    else:
        start = (yr_min // bin_years) * bin_years
        bins  = np.arange(start, yr_max + bin_years + 1, bin_years)

    centers = (bins[:-1] + bins[1:]) / 2.0
    widths  = np.diff(bins) * 0.85
    n_bins  = len(centers)

    # Per-file, per-bin accepted and rejected counts
    acc_per_file   = np.zeros((n_ids, n_bins), dtype=int)
    rej_per_file   = np.zeros((n_ids, n_bins), dtype=int)
    total_per_file = np.zeros(n_ids, dtype=int)
    for i in range(n_ids):
        s, e = int(set_offsets[i]), int(set_offsets[i + 1])
        yr_s = years[s:e]
        for k in range(n_bins):
            in_bin = (yr_s >= bins[k]) & (yr_s < bins[k + 1])
            acc_per_file[i, k] = int(mask_acc[s:e][in_bin].sum())
            rej_per_file[i, k] = int(mask_rej[s:e][in_bin].sum())
        total_per_file[i] = int(mask_acc[s:e].sum()) + int(mask_rej[s:e].sum())

    # ── Main figure (no legend) ───────────────────────────────────────────────
    if figsize is None:
        if which == 'both':
            figsize = (FIG_W_DOUBLE * 2.2, 5.5)
        else:
            figsize = (FIG_W_DOUBLE * 1.4, 4.5)

    fig = plt.figure(figsize=figsize)
    if which == 'both':
        gs    = fig.add_gridspec(1, 2, wspace=0.28)
        ax_yr = fig.add_subplot(gs[0])
        ax_id = fig.add_subplot(gs[1])
    elif which == 'year':
        ax_yr = fig.add_subplot(111)
        ax_id = None
    elif which == 'file':
        ax_yr = None
        ax_id = fig.add_subplot(111)
    else:
        raise ValueError(f"which must be one of 'both', 'year', 'file' (got {which!r})")

    bin_label = f'{bin_years}-year bins' if bin_years > 1 else 'annual'

    # Left: stacked bar per year
    if ax_yr is not None:
        bottom_acc = np.zeros(n_bins)
        for i in range(n_ids):
            col = _obs_color(i)
            ax_yr.bar(centers, acc_per_file[i], width=widths, bottom=bottom_acc,
                      color=col, alpha=0.85)
            bottom_acc += acc_per_file[i]

        total_per_bin = acc_per_file.sum(axis=0) + rej_per_file.sum(axis=0)
        acc_per_bin   = acc_per_file.sum(axis=0)
        rej_per_bin   = total_per_bin - acc_per_bin
        ax_yr.bar(centers, rej_per_bin, width=widths, bottom=acc_per_bin,
                  color='red', alpha=0.55, hatch='//',
                  label=f'rejected  (n={int(rej_per_bin.sum())})')

        ax_yr.set_xlabel('Year',
                         **(({'fontsize': axes_fontsize}) if axes_fontsize is not None else {}))
        ax_yr.set_ylabel('Number of observations',
                         **(({'fontsize': axes_fontsize}) if axes_fontsize is not None else {}))
        ax_yr.set_xlim(bins[0] - 0.5, bins[-1] + 0.5)
        if not suppress_panel_titles:
            ax_yr.set_title(f'{title} — per year  ({bin_label})')
        ax_yr.legend(fontsize=legend_fontsize, loc='upper left')
        if tick_fontsize is not None:
            ax_yr.tick_params(axis='both', labelsize=tick_fontsize)

    # Right: vertical bar per file, sorted largest → smallest
    if ax_id is not None:
        order       = np.argsort(total_per_file)[::-1]
        ids_sorted  = [set_ids[k].replace('_', ' ') for k in order]
        acc_sorted  = total_per_file[order] - rej_per_file.sum(axis=1)[order]
        rej_sorted  = rej_per_file.sum(axis=1)[order]
        cols_sorted = [_obs_color(k) for k in order]
        x_pos       = np.arange(len(ids_sorted))

        for k in range(len(x_pos)):
            ax_id.bar(x_pos[k], acc_sorted[k] + rej_sorted[k],
                      color=cols_sorted[k], alpha=0.85, width=0.7)
            if rej_sorted[k] > 0:
                ax_id.bar(x_pos[k], rej_sorted[k], bottom=acc_sorted[k],
                          color='red', alpha=0.55, hatch='//', width=0.7)

        xtick_fs = tick_fontsize if tick_fontsize is not None else 7
        ax_id.set_xticks(x_pos)
        ax_id.set_xticklabels(ids_sorted, rotation=45, ha='right', fontsize=xtick_fs)
        if tick_fontsize is not None:
            ax_id.tick_params(axis='y', labelsize=tick_fontsize)
        ax_id.set_ylabel('Number of observations',
                         **(({'fontsize': axes_fontsize}) if axes_fontsize is not None else {}))
        if not suppress_panel_titles:
            ax_id.set_title(f'{title} — per file')
        ax_id.set_ylim(0, total_per_file.max() * 1.15)
    fig.tight_layout()

    # ── Separate legend figure (1 column) ────────────────────────────────────
    def _id_legend_label(sid: str) -> str:
        if legend_observatory_names:
            mpc_code = sid.split('_', 1)[0]
            return _get_observatory_name(mpc_code)
        return sid.replace('_', ' ')

    leg_handles = [
        Line2D([0], [0], linestyle='none', marker='s',
               color=_obs_color(i), markersize=7, alpha=0.85,
               label=_id_legend_label(set_ids[i]))
        for i in range(n_ids)
    ]
    leg_handles.append(Patch(facecolor='red', alpha=0.55, hatch='//',
                             label=f'rejected  (n={int(rej_per_file.sum())})'))

    n_h     = len(leg_handles)
    fig_leg = plt.figure(figsize=(2.2, max(2.5, 0.32 * n_h + 0.4)))
    ax_l    = fig_leg.add_subplot(111)
    ax_l.axis('off')
    ax_l.legend(leg_handles, [h.get_label() for h in leg_handles],
                loc='center', fontsize=8, ncol=1,
                markerscale=1.2, frameon=True, borderaxespad=0.2)
    fig_leg.tight_layout()

    return [(fig, 'count'), (fig_leg, 'legend')]


def mpl_obs_analysis_spice_biased_by_id(
        sims, names, labels,
        data_path: str = 'Results/ObservationsAnalysis/obs_analysis_data.npy',
        title: str = r'O$-$C Residuals vs NEP097 — Bias-Corrected by ID',
        figsize=None):
    """RA / Dec SPICE residuals after manual Dec-bias correction, coloured by obs-file ID.

    Uses ``ra_spice_biased_arcsec`` / ``dec_spice_biased_arcsec`` from obs_analysis_data.
    Annotates the figure with the bias values that were applied.
    Same side-by-side layout (plots left, 1-col legend right).
    """
    data = _load_obs_analysis(data_path)
    if data is None:
        return None
    if 'ra_spice_biased_arcsec' not in data:
        print('  SKIP: ra_spice_biased_arcsec not found in obs_analysis_data '
              '(re-run Test_Observations.py)')
        return None

    bias_info = data.get('bias_applied_arcsec', {})
    bias_str  = ', '.join(f'{k}: {v:+.3f}"' for k, v in bias_info.items()) if bias_info else ''
    full_title = f'{title}\n{bias_str}' if bias_str else title

    return _obs_by_id_figure(
        data['times_dt'],
        np.asarray(data['ra_spice_biased_arcsec'],  dtype=float),
        np.asarray(data['dec_spice_biased_arcsec'], dtype=float),
        data['mask_accepted'], data['mask_rejected'],
        list(data['set_ids']), data['set_offsets'],
        full_title, show_rejected=True, figsize=figsize,
    )


def mpl_obs_analysis_spice_bias_overlay(
        sims, names, labels,
        data_path: str = 'Results/ObservationsAnalysis/obs_analysis_data.npy',
        title: str = r'O$-$C Residuals vs NEP097 — Bias Correction Overlay',
        figsize=None):
    """Overlay of unbiased (faded) and bias-corrected (solid) SPICE residuals.

    Both datasets are plotted for every obs-file ID using the same per-ID colour.
    Unbiased observations are shown as small faded circles; bias-corrected as
    larger solid circles on top.  For files not in the bias dict the two are
    identical and only the solid marker is visible.  Rejected observations are
    shown as red × using the mask from the unbiased run (unchanged by bias).

    The figure layout is identical to the by-ID figures: plots left, 1-col
    legend right with per-ID colour entries plus style entries.
    """
    from matplotlib.lines import Line2D

    data = _load_obs_analysis(data_path)
    if data is None:
        return None
    if 'ra_spice_biased_arcsec' not in data:
        print('  SKIP: ra_spice_biased_arcsec not found — re-run Test_Observations.py')
        return None

    set_ids     = list(data['set_ids'])
    set_offsets = data['set_offsets']
    times_dt    = data['times_dt']
    ra_ub       = data['ra_spice_arcsec']
    dec_ub      = data['dec_spice_arcsec']
    ra_b        = np.asarray(data['ra_spice_biased_arcsec'],  dtype=float)
    dec_b       = np.asarray(data['dec_spice_biased_arcsec'], dtype=float)
    mask_acc    = data['mask_accepted']
    mask_rej    = data['mask_rejected']
    bias_info   = data.get('bias_applied_arcsec', {})

    if figsize is None:
        figsize = (FIG_W_DOUBLE * 1.6, 7.0)

    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(2, 2, width_ratios=[3, 1], hspace=0.08, wspace=0.04)
    ax_ra  = fig.add_subplot(gs[0, 0])
    ax_dec = fig.add_subplot(gs[1, 0], sharex=ax_ra)
    ax_leg = fig.add_subplot(gs[:, 1])
    ax_leg.axis('off')

    # Two fixed colours — makes bias shift immediately visible regardless of file ID
    _C_UB  = '#1f77b4'   # unbiased: blue
    _C_B   = '#ff7f0e'   # biased:   orange

    n_rej = int(mask_rej.sum())

    # All observations (all file IDs merged, no per-ID coloring)
    for i in range(len(set_ids)):
        s, e = int(set_offsets[i]), int(set_offsets[i + 1])
        t_s  = times_dt[s:e]
        m_a  = mask_acc[s:e];  m_r = mask_rej[s:e]

        if m_a.any():
            ax_ra.scatter(t_s[m_a],  ra_ub[s:e][m_a],  s=5, color=_C_UB, alpha=0.45, zorder=2)
            ax_dec.scatter(t_s[m_a], dec_ub[s:e][m_a], s=5, color=_C_UB, alpha=0.45, zorder=2)
            ax_ra.scatter(t_s[m_a],  ra_b[s:e][m_a],   s=5, color=_C_B,  alpha=0.55, zorder=3)
            ax_dec.scatter(t_s[m_a], dec_b[s:e][m_a],  s=5, color=_C_B,  alpha=0.55, zorder=3)
        if m_r.any():
            ax_ra.scatter(t_s[m_r],  ra_b[s:e][m_r],   s=28, color='red', alpha=0.9,
                          marker='x', zorder=5, linewidths=0.8)
            ax_dec.scatter(t_s[m_r], dec_b[s:e][m_r],  s=28, color='red', alpha=0.9,
                           marker='x', zorder=5, linewidths=0.8)

    bias_str = ', '.join(f'{k}: {v:+.3f}$^{{\\prime\\prime}}$'
                         for k, v in bias_info.items()) if bias_info else ''
    leg_handles = [
        Line2D([0], [0], linestyle='none', marker='o', color=_C_UB,
               markersize=5, alpha=0.7, label='unbiased'),
        Line2D([0], [0], linestyle='none', marker='o', color=_C_B,
               markersize=5, alpha=0.7,
               label=f'bias-corrected{(" (" + bias_str + ")") if bias_str else ""}'),
        Line2D([0], [0], linestyle='none', marker='x', color='red',
               markersize=6, markeredgewidth=0.8, label=f'rejected  (n={n_rej})'),
    ]

    ax_ra.axhline(0,  color='k', linewidth=0.6, linestyle='--')
    ax_dec.axhline(0, color='k', linewidth=0.6, linestyle='--')
    ax_ra.set_ylabel("RA residual ['']")
    ax_dec.set_ylabel("Dec residual ['']")
    ax_dec.set_xlabel('Year')
    plt.setp(ax_ra.get_xticklabels(), visible=False)
    _apply_date_formatter(ax_dec)

    ax_leg.legend(leg_handles, [h.get_label() for h in leg_handles],
                  loc='center left', fontsize=8, ncol=1,
                  markerscale=1.5, frameon=True, borderaxespad=0.2)
    fig.tight_layout()
    return fig


def generate_rejected_obs_table(data_path: str, out_path: str):
    """Write a LaTeX table (or short sentence) listing all rejected observations.

    If all rejections belong to a single obs-file ID, writes a one-sentence
    .tex snippet instead of a full table.

    Parameters
    ----------
    data_path : path to obs_analysis_data.npy
    out_path  : output .tex file path
    """
    data = _load_obs_analysis(data_path)
    if data is None:
        print(f'  SKIP rejected-obs table: {data_path!r} not found')
        return

    set_ids     = list(data['set_ids'])
    set_offsets = data['set_offsets']
    times_dt    = data['times_dt']
    mask_rej    = data['mask_rejected']

    # Per-file rejected counts and epoch lists
    rej_info = {}   # set_id → list of datetime
    for i, sid in enumerate(set_ids):
        s, e = int(set_offsets[i]), int(set_offsets[i + 1])
        m_r  = mask_rej[s:e]
        if m_r.any():
            rej_info[sid] = list(times_dt[s:e][m_r])

    n_total_rej = int(mask_rej.sum())
    n_files_rej = len(rej_info)

    lines = []
    lines.append(r'% ============================================================')
    lines.append(r'% Rejected observations summary')
    lines.append(r'% ============================================================')
    lines.append('')

    if n_total_rej == 0:
        lines.append(r'No observations were rejected by the outlier filter.')

    elif n_files_rej == 1:
        sid     = next(iter(rej_info))
        dts     = rej_info[sid]
        yr_str  = ', '.join(str(d.year) for d in sorted(dts))
        sid_tex = sid.replace('_', r'\_')
        lines.append(
            rf'All {n_total_rej} rejected observations belong to dataset '
            rf'\texttt{{{sid_tex}}}'
            rf' (epochs: {yr_str}).'
        )

    else:
        lines.append(r'\begin{table}[htbp]')
        lines.append(r'  \centering')
        lines.append(
            r'  \caption{Observations rejected by the outlier filter, '
            r'grouped by dataset identifier.}'
        )
        lines.append(r'  \label{tab:rejected-obs}')
        lines.append(r'  \begin{tabular}{l c l}')
        lines.append(r'    \toprule')
        lines.append(r'    Dataset ID & $N_{\mathrm{rej}}$ & Epoch(s) \\')
        lines.append(r'    \midrule')
        for sid, dts in sorted(rej_info.items()):
            yr_str   = ', '.join(str(d.year) for d in sorted(dts))
            sid_tex  = sid.replace('_', r'\_')
            lines.append(f'    \\texttt{{{sid_tex}}} & {len(dts)} & {yr_str} \\\\')
        lines.append(r'    \bottomrule')
        lines.append(r'  \end{tabular}')
        lines.append(r'\end{table}')

    lines.append('')
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'  Saved: {out_path}  ({n_total_rej} rejected obs, {n_files_rej} file(s))')


# ============================================================================
# PER-FILE WEIGHT / RESIDUAL FIGURE FUNCTIONS
# ============================================================================

_RAD_TO_MAS_CONST = 206264806.0


def mpl_weight_uncertainty_per_file(sims, names, labels,
                                     sim_subset=None,
                                     title='Per-File Observation Uncertainty [mas]',
                                     figsize=None):
    """Bar chart: mean RA/Dec uncertainty [mas] = mean(1/sqrt(weight)) per file.

    One figure per simulation.  Returns list of (fig, safe_label).
    x axis = observation file ID, y axis = mean uncertainty in mas.
    """
    targets = sim_subset if sim_subset is not None else names
    figs = []

    for sn in targets:
        if sn not in sims:
            continue
        sd = sims[sn]
        if 'weight_info' not in sd:
            print(f"  SKIP weight_uncertainty_per_file for '{sn}': no weight_info.")
            continue
        wi  = sd['weight_info']
        lbl = labels[names.index(sn)] if sn in names else sn

        file_ids = sorted(wi['ref_point_id'].unique())
        x = np.arange(len(file_ids))

        unc_ra, unc_dec = [], []
        for fid in file_ids:
            mask  = wi['ref_point_id'] == fid
            w_ra  = wi.loc[mask, 'weight_ra'].values
            w_dec = wi.loc[mask, 'weight_dec'].values
            unc_ra.append(
                np.mean(1.0 / np.sqrt(np.maximum(w_ra,  1e-40))) * _RAD_TO_MAS_CONST)
            unc_dec.append(
                np.mean(1.0 / np.sqrt(np.maximum(w_dec, 1e-40))) * _RAD_TO_MAS_CONST)

        fig_w    = max(FIG_W_DOUBLE, 0.35 * len(file_ids))
        _figsize = figsize if figsize is not None else (fig_w, FIG_H_DEFAULT * 1.6)

        fig, (ax_ra, ax_dec) = plt.subplots(2, 1, figsize=_figsize, sharex=True)
        ax_ra.bar(x, unc_ra,  color='#1f77b4', alpha=0.8,
                  edgecolor='black', linewidth=0.4)
        ax_dec.bar(x, unc_dec, color='#d62728', alpha=0.8,
                   edgecolor='black', linewidth=0.4)

        ax_dec.set_xticks(x)
        ax_dec.set_xticklabels([f.replace('_', ' ') for f in file_ids],
                                rotation=45, ha='right', fontsize=7)
        ax_ra.set_ylabel('RA uncertainty [mas]')
        ax_dec.set_ylabel('Dec uncertainty [mas]')
        ax_ra.set_title('RA')
        ax_dec.set_title('Dec')

        safe = lbl.replace(' ', '_').replace('/', '-').replace('+', 'p').replace('.', '')
        fig.suptitle(f'{title} — {lbl}')
        fig.tight_layout()
        figs.append((fig, safe))

    return figs if figs else None


def mpl_weight_uncertainty_overlay(sims, names, labels,
                                    sim_subset=None,
                                    title='Per-File Uncertainty — All Weight Schemes [mas]',
                                    figsize=None):
    """Line plot: per-file mean uncertainty [mas] for all simulations overlaid.

    x axis = observation file ID, y axis = mean 1/sqrt(weight) in mas.
    """
    targets = sim_subset if sim_subset is not None else names
    valid = [
        (sn, labels[names.index(sn)] if sn in names else sn)
        for sn in targets
        if sn in sims and 'weight_info' in sims.get(sn, {})
    ]
    if not valid:
        print("  SKIP weight_uncertainty_overlay: no sims with weight_info.")
        return None

    wi0      = sims[valid[0][0]]['weight_info']
    file_ids = sorted(wi0['ref_point_id'].unique())
    x        = np.arange(len(file_ids))

    fig_w    = max(FIG_W_DOUBLE, 0.35 * len(file_ids))
    _figsize = figsize if figsize is not None else (fig_w, FIG_H_DEFAULT * 1.8)

    fig, (ax_ra, ax_dec) = plt.subplots(2, 1, figsize=_figsize, sharex=True)

    for j, (sn, lbl) in enumerate(valid):
        wi  = sims[sn]['weight_info']
        col = _SIM_COLORS.get(sn, _color(j))
        mk  = _marker(sn)

        unc_ra, unc_dec = [], []
        for fid in file_ids:
            mask = wi['ref_point_id'] == fid
            if mask.any():
                w_ra  = wi.loc[mask, 'weight_ra'].values
                w_dec = wi.loc[mask, 'weight_dec'].values
                unc_ra.append(
                    np.mean(1.0 / np.sqrt(np.maximum(w_ra,  1e-40))) * _RAD_TO_MAS_CONST)
                unc_dec.append(
                    np.mean(1.0 / np.sqrt(np.maximum(w_dec, 1e-40))) * _RAD_TO_MAS_CONST)
            else:
                unc_ra.append(np.nan)
                unc_dec.append(np.nan)

        ax_ra.plot(x, unc_ra,  color=col, marker=mk, linewidth=1.2,
                   markersize=5, label=lbl, zorder=2)
        ax_dec.plot(x, unc_dec, color=col, marker=mk, linewidth=1.2,
                    markersize=5, zorder=2)

    ax_dec.set_xticks(x)
    ax_dec.set_xticklabels([f.replace('_', ' ') for f in file_ids],
                            rotation=45, ha='right', fontsize=11)
    ax_ra.set_ylabel('RA uncertainty [mas]')
    ax_dec.set_ylabel('Dec uncertainty [mas]')
    ax_ra.set_title('RA')
    ax_dec.set_title('Dec')
    ax_ra.legend(fontsize=10, ncol=2, loc='upper right')

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def mpl_weight_uncertainty_per_timeframe(sims, names, labels,
                                          sim_subset=None,
                                          file_ids=None,
                                          title='Per-Timeframe Observation Uncertainty [mas]',
                                          figsize=None,
                                          show_suptitle=True,
                                          tick_fontsize=11):
    """Line plot: uncertainty (1/√weight) [mas] per timeframe for selected files.

    One figure per entry in file_ids.  Each figure has two panels (RA top, Dec
    bottom).  All sims in sim_subset are overlaid.

    A secondary y-axis (right side, grey) shows the number of observations per
    timeframe as a semi-transparent fill+step — providing context without
    cluttering the main axes.  Its top limit is set to max(nobs) × 4 so it
    occupies only the bottom quarter of the panel.

    Parameters
    ----------
    file_ids : list of str
        ref_point_id values to plot.  If None, all files in the first valid sim
        are plotted.
    """
    def _get_df(sd):
        return sd.get('residual_df', sd.get('weight_info'))

    targets = sim_subset if sim_subset is not None else names
    valid = [
        (sn, labels[names.index(sn)] if sn in names else sn)
        for sn in targets
        if sn in sims and _get_df(sims.get(sn, {})) is not None
    ]
    if not valid:
        print("  SKIP weight_uncertainty_per_timeframe: no valid sims.")
        return None

    if file_ids is None:
        df0 = _get_df(sims[valid[0][0]])
        file_ids = sorted(df0['ref_point_id'].unique())

    figs = []
    for fid in file_ids:
        all_tfs = sorted({
            tf
            for sn, _ in valid
            for tf in _get_df(sims[sn]).loc[
                _get_df(sims[sn])['ref_point_id'] == fid, 'timeframe'
            ].unique()
        })
        if not all_tfs:
            print(f"  SKIP weight_uncertainty_per_timeframe: {fid!r} not found.")
            continue

        tf_index = {tf: i for i, tf in enumerate(all_tfs)}
        x = np.arange(len(all_tfs))

        df0 = _get_df(sims[valid[0][0]])
        mask0 = df0['ref_point_id'] == fid
        nobs = [int((mask0 & (df0['timeframe'] == tf)).sum()) for tf in all_tfs]
        nobs_top = max(nobs) * 4

        fig_w = max(FIG_W_DOUBLE, 0.45 * len(all_tfs))
        _figsize = figsize if figsize is not None else (fig_w, FIG_H_DEFAULT * 1.6)
        fig, (ax_ra, ax_dec) = plt.subplots(2, 1, figsize=_figsize, sharex=True)

        for ax in (ax_ra, ax_dec):
            ax_r = ax.twinx()
            ax_r.fill_between(x, nobs, step='mid', alpha=0.12, color='grey', linewidth=0)
            ax_r.step(x, nobs, color='grey', linewidth=0.8, alpha=0.35, where='mid')
            ax_r.set_ylim(0, nobs_top)
            ax_r.set_ylabel('N obs', color='grey', fontsize=10)
            ax_r.tick_params(axis='y', labelcolor='grey', labelsize=9)
            ax_r.set_zorder(0)
            ax.set_zorder(1)
            ax.patch.set_visible(False)

        for sn, lbl in valid:
            df = _get_df(sims[sn])
            mask = df['ref_point_id'] == fid
            color = _SIM_COLORS.get(sn, None)
            ls = _SIM_LINESTYLE.get(sn, '-')
            mk = _marker(sn)

            unc_ra  = [np.nan] * len(all_tfs)
            unc_dec = [np.nan] * len(all_tfs)
            for tf in all_tfs:
                sub = df[mask & (df['timeframe'] == tf)]
                if len(sub) == 0:
                    continue
                w_ra  = sub['weight_ra'].dropna().values
                w_dec = sub['weight_dec'].dropna().values
                if len(w_ra) > 0 and np.all(w_ra > 0):
                    unc_ra[tf_index[tf]] = (
                        np.mean(1.0 / np.sqrt(w_ra)) * _RAD_TO_MAS_CONST)
                if len(w_dec) > 0 and np.all(w_dec > 0):
                    unc_dec[tf_index[tf]] = (
                        np.mean(1.0 / np.sqrt(w_dec)) * _RAD_TO_MAS_CONST)

            kw = dict(label=lbl, linestyle=ls, marker=mk, markersize=4,
                      linewidth=1.0, zorder=2)
            if color:
                kw['color'] = color
            ax_ra.plot(x,  unc_ra,  **kw)
            ax_dec.plot(x, unc_dec, **{**kw, 'label': '_nolegend_'})

        ax_ra.set_ylabel('RA uncertainty [mas]')
        ax_dec.set_ylabel('Dec uncertainty [mas]')
        ax_ra.set_title(f'RA  —  {fid.replace("_", " ")}')
        ax_dec.set_title(f'Dec  —  {fid.replace("_", " ")}')
        ax_dec.set_xticks(x)
        ax_dec.set_xticklabels([str(tf) for tf in all_tfs],
                                rotation=45, ha='right', fontsize=tick_fontsize)
        ax_dec.tick_params(axis='y', labelsize=tick_fontsize)
        ax_ra.tick_params(axis='y', labelsize=tick_fontsize)
        ax_dec.set_xlabel('Timeframe', fontsize=tick_fontsize + 1)
        ax_ra.legend(fontsize=10, loc='upper right',
                     bbox_to_anchor=(0.99, 0.99), borderaxespad=0.3)
        if show_suptitle and title:
            fig.suptitle(f'{title}  —  {fid.replace("_", " ")}')
        fig.tight_layout()

        safe = fid.replace(' ', '_').replace('/', '-')
        figs.append((fig, safe))

    return figs if figs else None


def _add_obs_density_twinx(ax, datetimes):
    """Overlay a gray yearly observation-count bar chart on a twin right y-axis.

    Bins observations by calendar year.  The twin axis is pushed to the back
    (zorder=0) so the scatter sits on top.
    """
    dt = pd.to_datetime(datetimes)
    year_counts = dt.dt.year.value_counts().sort_index()
    bar_centers = [pd.Timestamp(year=int(y), month=7, day=1)
                   for y in year_counts.index]
    counts = year_counts.values

    def _round_sig(x, sig=2):
        """Round x to `sig` significant figures (returns int)."""
        if x <= 0:
            return 0
        d = int(np.floor(np.log10(x))) - (sig - 1)
        return int(round(x, -d))

    max_c  = int(counts.max())
    min_c  = int(counts[counts > 0].min())
    t_max  = _round_sig(max_c)
    t_min  = _round_sig(min_c)
    # Two intermediate ticks evenly spaced between t_min and t_max
    mids   = [_round_sig(int(v)) for v in np.linspace(t_min, t_max, 4)[1:-1]]
    ticks  = sorted(set([0, t_min] + mids + [t_max]))

    from matplotlib.ticker import FixedLocator
    ax_r = ax.twinx()
    ax_r.bar(bar_centers, counts, width=300,   # 300-day width ≈ 1 year bar
             color='#888888', alpha=0.40, linewidth=0, zorder=0)
    ax_r.set_ylim(0, max_c * 4)               # push bars to bottom quarter
    ax_r.yaxis.set_major_locator(FixedLocator(ticks))
    ax_r.set_ylabel('Number of Observations', color='#888888', fontsize=12)
    ax_r.tick_params(axis='y', labelcolor='#888888', labelsize=11)
    ax_r.set_zorder(0)
    ax.set_zorder(1)
    ax.patch.set_visible(False)
    return ax_r


def mpl_weight_vs_datetime(sims, names, labels,
                            sim_subset=None,
                            title='Observation Weights vs Time',
                            figsize=None):
    """Scatter: weight_ra / weight_dec vs datetime — one figure per simulation.

    Two stacked panels (RA top, Dec bottom) with a gray yearly observation-count
    histogram overlaid on each panel via a secondary right y-axis.
    datetime on x-axis, weight [1/rad²] on log y-axis.
    Returns list of (fig, safe_label).
    """
    targets = sim_subset if sim_subset is not None else names
    valid = [
        (sn, labels[names.index(sn)] if sn in names else sn)
        for sn in targets
        if sn in sims and 'weight_info' in sims.get(sn, {})
    ]
    if not valid:
        print("  SKIP weight_vs_datetime: no sims with weight_info.")
        return None

    _figsize = figsize if figsize is not None else (FIG_W_DOUBLE * 1.4, FIG_H_DEFAULT * 1.8)
    figs = []

    for sn, lbl in valid:
        wi  = sims[sn]['weight_info']

        fig, (ax_ra, ax_dec) = plt.subplots(2, 1, figsize=_figsize, sharex=True)

        _add_obs_density_twinx(ax_ra,  wi['datetime'])
        _add_obs_density_twinx(ax_dec, wi['datetime'])

        ax_ra.scatter(wi['datetime'], wi['weight_ra'].values,
                      color='#1f77b4', s=18, alpha=0.7, linewidths=0)
        ax_dec.scatter(wi['datetime'], wi['weight_dec'].values,
                       color='#1f77b4', s=18, alpha=0.7, linewidths=0)

        for ax in (ax_ra, ax_dec):
            ax.set_yscale('log')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator(5))
            ax.tick_params(axis='x', labelsize=11, rotation=30)

        ax_ra.set_ylabel('Weight RA [1/rad\u00b2]')
        ax_dec.set_ylabel('Weight Dec [1/rad\u00b2]')
        ax_ra.set_title('RA')
        ax_dec.set_title('Dec')
        ax_dec.set_xlabel('Date')

        fig.suptitle(f'{title} \u2014 {lbl}')
        fig.tight_layout()

        safe = lbl.replace(' ', '_').replace('/', '-').replace('+', 'p').replace('.', '')
        figs.append((fig, safe))

    return figs if figs else None


def mpl_uncertainty_vs_datetime(sims, names, labels,
                                 sim_subset=None,
                                 title='\u03c3 = 1/\u221aweight vs Time [mas]',
                                 figsize=None):
    """Scatter: \u03c3_RA, \u03c3_Dec [mas] = 1/\u221aweight vs datetime — one figure per simulation.

    Two stacked panels (RA top, Dec bottom) with a gray yearly observation-count
    histogram overlaid on each panel via a secondary right y-axis.
    datetime on x-axis, uncertainty [mas] on linear y-axis.
    Returns list of (fig, safe_label).
    """
    targets = sim_subset if sim_subset is not None else names
    valid = [
        (sn, labels[names.index(sn)] if sn in names else sn)
        for sn in targets
        if sn in sims and 'weight_info' in sims.get(sn, {})
    ]
    if not valid:
        print("  SKIP uncertainty_vs_datetime: no sims with weight_info.")
        return None

    _figsize = figsize if figsize is not None else (FIG_W_DOUBLE * 1.4, FIG_H_DEFAULT * 1.8)
    figs = []

    for sn, lbl in valid:
        wi      = sims[sn]['weight_info']
        unc_ra  = (1.0 / np.sqrt(np.maximum(wi['weight_ra'].values,  1e-40))
                   * _RAD_TO_MAS_CONST)
        unc_dec = (1.0 / np.sqrt(np.maximum(wi['weight_dec'].values, 1e-40))
                   * _RAD_TO_MAS_CONST)

        fig, (ax_ra, ax_dec) = plt.subplots(2, 1, figsize=_figsize, sharex=True)

        ax_ra_r  = _add_obs_density_twinx(ax_ra,  wi['datetime'])
        ax_dec_r = _add_obs_density_twinx(ax_dec, wi['datetime'])
        ax_ra_r.grid(False)
        ax_dec_r.grid(False)

        ax_ra.scatter(wi['datetime'], unc_ra,
                      color='#1f77b4', s=18, alpha=0.7, linewidths=0)
        ax_dec.scatter(wi['datetime'], unc_dec,
                       color='#1f77b4', s=18, alpha=0.7, linewidths=0)

        from matplotlib.ticker import LogLocator, ScalarFormatter
        for ax in (ax_ra, ax_dec):
            ax.set_yscale('log')
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=[1, 2, 5], numticks=20))
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs=[3, 4, 6, 7, 8, 9], numticks=50))
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.grid(which='minor', axis='y', alpha=0.15, linewidth=0.5)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator(5))
            ax.tick_params(axis='x', labelsize=13, rotation=30)
            ax.tick_params(axis='y', labelsize=13)

        ax_ra.set_ylabel('\u03c3 RA [mas]', fontsize=13)
        ax_dec.set_ylabel('\u03c3 Dec [mas]', fontsize=13)
        ax_dec.set_xlabel('Date')

        fig.tight_layout()

        safe = lbl.replace(' ', '_').replace('/', '-').replace('+', 'p').replace('.', '')
        figs.append((fig, safe))

    return figs if figs else None


def mpl_rms_delta_per_file(sims, names, labels,
                            sim_subset=None,
                            file_exclude=None,
                            title='Per-File RMS Update (Initial \u2212 Final) [mas]',
                            figsize=None):
    """Line plot: (initial_rms \u2212 final_rms) per observation file, all sims overlaid.

    Positive values indicate improvement.  Two panels: RA (top) and Dec (bottom).
    A secondary right-hand axis shows N_obs per file as a semi-transparent
    grey fill+step (same style as mpl_weight_uncertainty_per_timeframe).

    Parameters
    ----------
    file_exclude : list of str, optional
        ref_point_id values to omit from the plot.
    """
    from matplotlib.ticker import MaxNLocator

    targets = sim_subset if sim_subset is not None else names
    valid   = [
        (sn, labels[names.index(sn)] if sn in names else sn)
        for sn in targets
        if sn in sims and 'residual_df' in sims.get(sn, {})
    ]
    if not valid:
        print("  SKIP rms_delta_per_file: no sims with residual_df.")
        return None

    exclude  = set(file_exclude or [])
    file_ids = sorted({
        rid
        for sn, _ in valid
        for rid in sims[sn]['residual_df']['ref_point_id'].unique()
        if rid not in exclude
    })
    x = np.arange(len(file_ids))

    df0      = sims[valid[0][0]]['residual_df']
    nobs     = [int((df0['ref_point_id'] == fid).sum()) for fid in file_ids]
    nobs_top = max(nobs) * 4

    fig_w    = max(FIG_W_DOUBLE, 0.35 * len(file_ids))
    _figsize = figsize if figsize is not None else (fig_w, FIG_H_DEFAULT * 1.6)
    fig, (ax_ra, ax_dec) = plt.subplots(2, 1, figsize=_figsize, sharex=True)

    for ax in (ax_ra, ax_dec):
        ax_r = ax.twinx()
        ax_r.fill_between(x, nobs, step='mid', alpha=0.12, color='grey', linewidth=0)
        ax_r.step(x, nobs, color='grey', linewidth=0.8, alpha=0.35, where='mid')
        ax_r.set_ylim(0, nobs_top)
        ax_r.set_ylabel('N obs', color='grey', fontsize=10)
        ax_r.tick_params(axis='y', labelcolor='grey', labelsize=9)
        ax_r.set_zorder(0)
        ax.set_zorder(1)
        ax.patch.set_visible(False)

    for sn, lbl in valid:
        df    = sims[sn]['residual_df']
        color = _SIM_COLORS.get(sn, None)
        ls    = _SIM_LINESTYLE.get(sn, '-')
        mk    = _marker(sn)

        delta_ra, delta_dec = [], []
        for fid in file_ids:
            mask  = df['ref_point_id'] == fid
            ra_i  = df.loc[mask, 'ra_residual_initial_mas'].dropna().values
            dec_i = df.loc[mask, 'dec_residual_initial_mas'].dropna().values
            ra_f  = df.loc[mask, 'ra_residual_final_mas'].dropna().values
            dec_f = df.loc[mask, 'dec_residual_final_mas'].dropna().values
            rms_ra_i  = np.sqrt(np.mean(ra_i  ** 2)) if len(ra_i)  > 0 else np.nan
            rms_dec_i = np.sqrt(np.mean(dec_i ** 2)) if len(dec_i) > 0 else np.nan
            rms_ra_f  = np.sqrt(np.mean(ra_f  ** 2)) if len(ra_f)  > 0 else np.nan
            rms_dec_f = np.sqrt(np.mean(dec_f ** 2)) if len(dec_f) > 0 else np.nan
            delta_ra.append(rms_ra_i  - rms_ra_f)
            delta_dec.append(rms_dec_i - rms_dec_f)

        kw = dict(label=lbl, linestyle=ls, marker=mk, markersize=3, linewidth=1.0,
                  zorder=2)
        if color:
            kw['color'] = color
        ax_ra.plot(x,  delta_ra,  **kw)
        ax_dec.plot(x, delta_dec, **{**kw, 'label': '_nolegend_'})

    for ax in (ax_ra, ax_dec):
        ax.axhline(0, color='black', linewidth=0.6, linestyle='--', zorder=3)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=10, integer=False))

    ax_ra.set_ylabel('\u0394RA rms [mas]')
    ax_dec.set_ylabel('\u0394Dec rms [mas]')
    ax_ra.set_title('RA rms update (initial \u2212 final)')
    ax_dec.set_title('Dec rms update (initial \u2212 final)')
    ax_dec.set_xticks(x)
    ax_dec.set_xticklabels([f.replace('_', ' ') for f in file_ids],
                            rotation=45, ha='right', fontsize=11)
    ax_dec.set_xlabel('Ref Point ID')
    ax_ra.legend(fontsize=10, loc='upper right',
                 bbox_to_anchor=(1.0, 1.0), borderaxespad=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def mpl_rms_per_file(sims, names, labels,
                     sim_subset=None,
                     title='Per-File Final Residual RMS [mas]',
                     figsize=None):
    """Line plot: final residual RMS [mas] per observation file, all sims overlaid.

    Mirrors the Dash _obs_residual_summary view at level='per_file',
    metric_type='rms', data_source='final'.

    Two panels: RA rms (top) and DEC rms (bottom).  All simulations drawn as
    separate lines on the same axes so schemes can be directly compared.

    Returns a single Figure (not a list).
    """
    targets = sim_subset if sim_subset is not None else names
    valid   = [
        (sn, labels[names.index(sn)] if sn in names else sn)
        for sn in targets
        if sn in sims and 'residual_df' in sims.get(sn, {})
    ]
    if not valid:
        print("  SKIP rms_per_file: no sims with residual_df.")
        return None

    # File IDs from the union of all valid sims (should be identical in practice).
    file_ids = sorted({
        rid
        for sn, _ in valid
        for rid in sims[sn]['residual_df']['ref_point_id'].unique()
    })
    x = np.arange(len(file_ids))

    fig_w    = max(FIG_W_DOUBLE, 0.35 * len(file_ids))
    _figsize = figsize if figsize is not None else (fig_w, FIG_H_DEFAULT * 1.6)
    fig, (ax_ra, ax_dec) = plt.subplots(2, 1, figsize=_figsize, sharex=True)

    for sn, lbl in valid:
        df    = sims[sn]['residual_df']
        color = _SIM_COLORS.get(sn, None)
        ls    = _SIM_LINESTYLE.get(sn, '-')
        mk    = _marker(sn)

        rms_ra, rms_dec = [], []
        for fid in file_ids:
            mask  = df['ref_point_id'] == fid
            ra_v  = df.loc[mask, 'ra_residual_final_mas'].dropna().values
            dec_v = df.loc[mask, 'dec_residual_final_mas'].dropna().values
            rms_ra.append(np.sqrt(np.mean(ra_v ** 2))   if len(ra_v)  > 0 else np.nan)
            rms_dec.append(np.sqrt(np.mean(dec_v ** 2)) if len(dec_v) > 0 else np.nan)

        kw = dict(label=lbl, linestyle=ls, marker=mk, markersize=3, linewidth=1.0)
        if color:
            kw['color'] = color
        ax_ra.plot(x,  rms_ra,  **kw)
        ax_dec.plot(x, rms_dec, **{**kw, 'label': '_nolegend_'})

    ax_ra.set_ylabel('RA rms [mas]')
    ax_dec.set_ylabel('DEC rms [mas]')
    ax_ra.set_title('RA rms [mas]')
    ax_dec.set_title('DEC rms [mas]')
    ax_dec.set_xticks(x)
    ax_dec.set_xticklabels([f.replace('_', ' ') for f in file_ids],
                            rotation=45, ha='right', fontsize=7)
    ax_dec.set_xlabel('Ref Point ID')
    ax_ra.legend(fontsize=7, loc='upper right',
                 bbox_to_anchor=(1.0, 1.0), borderaxespad=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def mpl_rms_initial_final_per_file(sims, names, labels,
                                    sim_subset=None,
                                    title='Per-File RMS: Initial vs Final [mas]',
                                    figsize=None):
    """Grouped bar chart: initial and final residual RMS [mas] per observation file.

    One figure per simulation in sim_subset.  Returns list of (fig, safe_label).
    x axis = observation file ID, y axis = sqrt(mean(residual²)) in mas.
    Light bars = initial, solid bars = final.
    """
    targets = sim_subset if sim_subset is not None else names
    figs = []

    for sn in targets:
        if sn not in sims:
            continue
        sd = sims[sn]
        if 'residual_df' not in sd:
            print(f"  SKIP rms_initial_final_per_file for '{sn}': no residual_df.")
            continue
        df  = sd['residual_df']
        lbl = labels[names.index(sn)] if sn in names else sn

        file_ids = sorted(df['ref_point_id'].unique())
        x = np.arange(len(file_ids))
        bar_w = 0.38

        rms_ra_i,  rms_dec_i  = [], []
        rms_ra_f,  rms_dec_f  = [], []
        for fid in file_ids:
            mask  = df['ref_point_id'] == fid
            ra_i  = df.loc[mask, 'ra_residual_initial_mas'].dropna().values
            dec_i = df.loc[mask, 'dec_residual_initial_mas'].dropna().values
            ra_f  = df.loc[mask, 'ra_residual_final_mas'].dropna().values
            dec_f = df.loc[mask, 'dec_residual_final_mas'].dropna().values
            rms_ra_i.append(np.sqrt(np.mean(ra_i ** 2))   if len(ra_i)  > 0 else np.nan)
            rms_dec_i.append(np.sqrt(np.mean(dec_i ** 2)) if len(dec_i) > 0 else np.nan)
            rms_ra_f.append(np.sqrt(np.mean(ra_f ** 2))   if len(ra_f)  > 0 else np.nan)
            rms_dec_f.append(np.sqrt(np.mean(dec_f ** 2)) if len(dec_f) > 0 else np.nan)

        fig_w    = max(FIG_W_DOUBLE, 0.35 * len(file_ids))
        _figsize = figsize if figsize is not None else (fig_w, FIG_H_DEFAULT * 1.6)

        fig, (ax_ra, ax_dec) = plt.subplots(2, 1, figsize=_figsize, sharex=True)

        ax_ra.bar(x - bar_w / 2, rms_ra_i,  width=bar_w, label='Initial',
                  color='#aec7e8', alpha=0.9, edgecolor='black', linewidth=0.4)
        ax_ra.bar(x + bar_w / 2, rms_ra_f,  width=bar_w, label='Final',
                  color='#1f77b4', alpha=0.9, edgecolor='black', linewidth=0.4)
        ax_dec.bar(x - bar_w / 2, rms_dec_i, width=bar_w,
                   color='#ffbb78', alpha=0.9, edgecolor='black', linewidth=0.4)
        ax_dec.bar(x + bar_w / 2, rms_dec_f, width=bar_w,
                   color='#d62728', alpha=0.9, edgecolor='black', linewidth=0.4)

        ax_dec.set_xticks(x)
        ax_dec.set_xticklabels([f.replace('_', ' ') for f in file_ids],
                                rotation=45, ha='right', fontsize=7)
        ax_ra.set_ylabel('RA RMS [mas]')
        ax_dec.set_ylabel('Dec RMS [mas]')
        ax_ra.set_title('RA')
        ax_dec.set_title('Dec')
        ax_ra.legend(fontsize=8)

        safe = lbl.replace(' ', '_').replace('/', '-').replace('+', 'p').replace('.', '')
        fig.suptitle(f'{title} — {lbl}')
        fig.tight_layout()
        figs.append((fig, safe))

    return figs if figs else None


def mpl_n_obs_per_timeframe(sims, names, labels,
                             sim_name=None,
                             title='N_obs / N_timeframes per File',
                             figsize=None):
    """Line plot: ratio of observations to timeframes per observation file.

    Mirrors the Dash 'nobs_per_tf' metric in _obs_residual_summary:
      ratio = n_obs_in_file / n_unique_timeframes_in_file

    This is a dataset-level property (same for all weight schemes), so only
    one simulation is needed.  Uses residual_df of sim_name (defaults to
    names[0]); falls back to weight_info if residual_df is absent.

    x axis = observation file ID (ref_point_id).
    y axis = nobs / n_tf  (ratio, dimensionless).
    """
    if sim_name is None and names:
        sim_name = names[0]

    sd = sims.get(sim_name, {})
    if 'residual_df' in sd:
        df       = sd['residual_df']
        id_col   = 'ref_point_id'
        tf_col   = 'timeframe'
    elif 'weight_info' in sd:
        df       = sd['weight_info']
        id_col   = 'ref_point_id'
        tf_col   = 'timeframe'
    else:
        print(f"  SKIP n_obs_per_timeframe: no residual_df/weight_info for {sim_name!r}.")
        return None

    file_ids = sorted(df[id_col].unique())
    ratios   = []
    for fid in file_ids:
        sub   = df[df[id_col] == fid]
        n_tf  = max(1, len(sub[tf_col].unique()))
        ratios.append(len(sub) / n_tf)

    n_obs    = [len(df[df[id_col] == fid]) for fid in file_ids]

    x        = np.arange(len(file_ids))
    fig_w    = max(FIG_W_DOUBLE, 0.35 * len(file_ids))
    _figsize = figsize if figsize is not None else (fig_w, FIG_H_DEFAULT * 2)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=_figsize, sharex=True)

    ax.plot(x, ratios, marker='o', markersize=4, linewidth=1.0, color='steelblue')
    ax.set_ylabel('nobs_per_tf', fontsize=12)
    ax.tick_params(axis='y', labelsize=11)
    ax.set_title(title)

    ax2.bar(x, n_obs, color='steelblue', alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f.replace('_', ' ') for f in file_ids],
                        rotation=45, ha='right', fontsize=11)
    ax2.set_ylabel('N observations', fontsize=12)
    ax2.set_xlabel('Ref Point ID', fontsize=12)
    ax2.tick_params(axis='y', labelsize=11)

    fig.tight_layout()
    return fig


# ============================================================================
# DISPATCH TABLE
# ============================================================================

_PLOT_REGISTRY = {
    'rms_compare':              mpl_rms_compare,
    'rms_compare_rsw':          mpl_rms_compare_rsw,
    'formal_rms_rsw':           mpl_formal_rms_rsw,
    'rsw_rms_ratio_grid':       mpl_rsw_rms_ratio_grid,
    'residual_timeseries_by_id': mpl_residual_timeseries_by_id,
    'residual_histogram_per_sim': mpl_residual_histogram_per_sim,
    'pole_model':          mpl_pole_model,
    'pole_model_diff':     mpl_pole_model_diff,
    'pole_model_compare':         mpl_pole_model_compare,
    'pole_uncertainty_validation': mpl_pole_uncertainty_validation,
    'rms_formal':          mpl_rms_formal,
    'rms_ratio':           mpl_rms_ratio,
    'rsw_ratio':           mpl_rsw_ratio,
    'rms_compare_multi':   mpl_rms_compare_multi,
    'gof_combined_multi':  mpl_gof_combined_multi,
    'gof_metric_multi':    mpl_gof_metric_multi,
    'formal_rms_multi':    mpl_formal_rms_multi,
    'rms_ratio_multi':     mpl_rms_ratio_multi,
    'rsw_compare':         mpl_rsw_compare,
    'formal_compare':      mpl_formal_compare,
    'rsw_stats':           mpl_rsw_stats,
    'gof':                 mpl_gof,
    'gof_combined':        mpl_gof_combined,
    'corr_heatmap':        mpl_corr_heatmap,
    'condition_numbers_txt': write_condition_numbers,
    'residual_histogram':  mpl_residual_histogram,
    'residual_timeseries': mpl_residual_timeseries,
    'param_state':         mpl_param_state,
    'param_rsw':           mpl_param_rsw,
    'param_pole_pos':      mpl_param_pole_pos,
    'param_pole_lib':      mpl_param_pole_lib,
    'rsw_initial':              mpl_rsw_initial,
    'rsw_initial_diff':         mpl_rsw_initial_diff,
    'rsw_initial_vs_final':     mpl_rsw_initial_vs_final,
    'rsw_with_zoom':       mpl_rsw_with_zoom,
    'formal_with_zoom':    mpl_formal_with_zoom,
    'rsw_with_formal':            mpl_rsw_with_formal,
    'rsw_with_formal_cloud':      mpl_rsw_with_formal_cloud,
    'rsw_and_formal_lines':       mpl_rsw_and_formal_lines,
    'param_gm':            mpl_param_gm,
    'param_pole_rate':     mpl_param_pole_rate,
    'param_sh':            mpl_param_sh,
    'legend':              mpl_legend,
    # ── Observational dataset ─────────────────────────────────────────────────
    'obs_spice_timeseries':   mpl_obs_spice_timeseries,
    'obs_initial_timeseries': mpl_obs_initial_timeseries,
    'obs_histogram_per_id':   mpl_obs_histogram_per_id,
    # ── obs_analysis_data.npy figures (from Test_Observations.py) ─────────────
    'obs_analysis_spice_timeseries':    mpl_obs_analysis_spice_timeseries,
    'obs_analysis_prop_timeseries':     mpl_obs_analysis_prop_timeseries,
    'obs_analysis_histogram':           mpl_obs_analysis_histogram,
    'obs_analysis_excluded_spice':      mpl_obs_analysis_excluded_spice_timeseries,
    'obs_analysis_all_obs_combined':    mpl_obs_analysis_all_obs_combined,
    'obs_analysis_n_obs_datetime':      mpl_obs_analysis_n_obs_datetime,
    'obs_analysis_n_obs_id':            mpl_obs_analysis_n_obs_id,
    # ── new obs-analysis figures ───────────────────────────────────────────────
    'obs_analysis_all_in_folder':            mpl_obs_analysis_all_in_folder,
    'obs_analysis_spice_by_id':              mpl_obs_analysis_spice_by_id,
    'obs_analysis_spice_by_id_accepted':     mpl_obs_analysis_spice_by_id_accepted,
    'obs_analysis_prop_by_id':               mpl_obs_analysis_prop_by_id,
    'obs_analysis_spice_biased_by_id':       mpl_obs_analysis_spice_biased_by_id,
    'obs_analysis_spice_bias_overlay':       mpl_obs_analysis_spice_bias_overlay,
    'obs_analysis_spice_filtered_highlight': mpl_obs_analysis_spice_filtered_highlight,
    'obs_analysis_spice_per_file':           mpl_obs_analysis_spice_per_file,
    'obs_analysis_combined_count':           mpl_obs_analysis_combined_count,
    # ── Per-file weight / residual figures ────────────────────────────────────
    'weight_uncertainty_per_file':       mpl_weight_uncertainty_per_file,
    'weight_uncertainty_overlay':        mpl_weight_uncertainty_overlay,
    'weight_uncertainty_per_timeframe':  mpl_weight_uncertainty_per_timeframe,
    'weight_vs_datetime':                mpl_weight_vs_datetime,
    'uncertainty_vs_datetime':           mpl_uncertainty_vs_datetime,
    'rms_per_file':                      mpl_rms_per_file,
    'rms_delta_per_file':                mpl_rms_delta_per_file,
    'rms_initial_final_per_file':        mpl_rms_initial_final_per_file,
    'n_obs_per_timeframe':               mpl_n_obs_per_timeframe,
}


# ============================================================================
# PARAMETER TABLE EXPORT
# ============================================================================

# Unit scales applied per parameter group for display.
_GROUP_SCALE = {
    'Position':        (1e-3,         'km'),
    'Velocity':        (1e-3,         'km/s'),
    'Pole Position':   (_RAD_TO_DEG,  'deg'),
    'Pole Rate':       (_RAD_TO_DEG,  'deg/yr'),
    'Pole Librations': (_RAD_TO_DEG,  'deg'),
}
_TARGET_GROUPS = {'Position', 'Velocity', 'Pole Position', 'Pole Librations'}


def generate_single_sim_table(sims, sim_name, names, out_path):
    """Write a LaTeX table (.tex) for one simulation (sim_name).

    Columns: Parameter | Group | Unit | IAU initial | FitPole initial
             | Final | Δ (final − IAU initial) | Δ [%]
    Rows cover all target parameter groups (state + pole + librations).

    Requires preamble: \\usepackage{booktabs,siunitx,graphicx}
    """

    def _esc(s):
        return s.replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')

    def _p2tex(lbl):
        _MAP = {
            'α₀': r'$\alpha_0$',       'δ₀': r'$\delta_0$',
            'α₁': r'$\alpha_1$',       'δ₁': r'$\delta_1$',
            'α̇₀': r'$\dot{\alpha}_0$', 'δ̇₀': r'$\dot{\delta}_0$',
        }
        return _MAP.get(lbl, _esc(lbl))

    def _num(v):
        return rf'\num{{{v:.5e}}}'

    sd = sims.get(sim_name, {})
    if 'parameter_history' not in sd or 'est_parameters' not in sd:
        print(f"  SKIP table for '{sim_name}': no parameter data.")
        return

    iau_ref_in, _ = _get_iau_reference(sims, names)

    # FitPole reference (first-seen across all non-IAUPole sims).
    sp_ref_in = {}
    for sn in names:
        if sn.startswith('IAUPole'):
            continue
        sd2 = sims.get(sn, {})
        if 'parameter_history' not in sd2 or 'est_parameters' not in sd2:
            continue
        ph2    = sd2['parameter_history']
        lbls2, _, _ = get_parameter_info(sd2['est_parameters'])
        for i, lbl in enumerate(lbls2):
            if lbl not in sp_ref_in:
                sp_ref_in[lbl] = ph2[i, 0]

    # Canonical parameter list from all sims (same approach as generate_parameter_tables).
    seen       = set()
    param_rows = []
    for sn in names:
        sd2 = sims.get(sn, {})
        if 'parameter_history' not in sd2 or 'est_parameters' not in sd2:
            continue
        lbls2, grps2, _ = get_parameter_info(sd2['est_parameters'])
        for lbl, grp in zip(lbls2, grps2):
            if grp not in _TARGET_GROUPS or lbl in seen:
                continue
            seen.add(lbl)
            sc, unit = _GROUP_SCALE.get(grp, (1.0, '---'))
            param_rows.append((lbl, grp, sc, unit))

    ph_sim    = sd['parameter_history']
    lbls_sim, _, _ = get_parameter_info(sd['est_parameters'])

    sim_label = _esc(sim_name.replace('_cov', ''))
    lines = []
    lines.append(r'% ============================================================')
    lines.append(rf'% Single-simulation detail table: {sim_name}')
    lines.append(r'% Preamble: \usepackage{booktabs,siunitx,graphicx}')
    lines.append(r'% ============================================================')
    lines.append('')
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'  \centering')
    lines.append(
        rf'  \caption{{Parameter values for simulation \texttt{{{sim_label}}}. '
        rf'IAU\textsubscript{{0}} and FitPole\textsubscript{{0}} are the respective '
        rf'initial values; Final is the estimated result; '
        rf'$\Delta = \text{{Final}} - \text{{IAU}}_0$; '
        rf'$\Delta[\%] = \Delta\,/\,|\text{{IAU}}_0| \times 100$.}}'
    )
    lines.append(rf'  \label{{tab:single_{_esc(sim_name)}}}')
    lines.append(r'  \resizebox{\textwidth}{!}{%')
    lines.append(r'  \begin{tabular}{llcrrrrrrr}')
    lines.append(r'    \toprule')
    lines.append(
        r'    Parameter & Group & Unit'
        r' & IAU$_0$ & FitPole$_0$ & Final'
        r' & $\Delta$ & $\Delta\,[\%]$ \\'
    )
    lines.append(r'    \midrule')

    prev_grp = None
    for lbl, grp, sc, unit in param_rows:
        if prev_grp is not None and grp != prev_grp:
            lines.append(r'    \midrule')
        prev_grp = grp

        lbl_tex = _p2tex(lbl)
        iau_raw = iau_ref_in.get(lbl)
        sp_raw  = sp_ref_in.get(lbl)

        # Final value from this sim (--- if not estimated).
        if lbl in lbls_sim:
            final_raw = ph_sim[lbls_sim.index(lbl), -1]
            final_s   = final_raw * sc
            final_str = _num(final_s)
        else:
            final_raw = None
            final_str = '---'

        if iau_raw is None:
            lines.append(f'    {lbl_tex} & {_esc(grp)} & {unit}'
                         r' & --- & --- & ' + final_str + r' & --- & --- \\')
            continue

        iau_s  = iau_raw * sc
        sp_str = _num(sp_raw * sc) if sp_raw is not None else '---'

        if final_raw is not None:
            diff  = final_s - iau_s
            pct   = diff / abs(iau_s) * 100 if iau_s != 0 else 0.0
            lines.append(
                f'    {lbl_tex} & {_esc(grp)} & {unit}'
                f' & {_num(iau_s)} & {sp_str} & {final_str}'
                rf' & {_num(diff)} & {pct:.2f} \\'
            )
        else:
            lines.append(
                f'    {lbl_tex} & {_esc(grp)} & {unit}'
                f' & {_num(iau_s)} & {sp_str} & --- & --- & --- \\'
            )

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'  }% end resizebox')
    lines.append(r'\end{table}')
    lines.append('')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved: {out_path}")


def generate_final_estimation_tables(sims, names, out_path,
                                     iau_sim='IAUPole_pole_lib_cov',
                                     fitpole_sim='SimPole_pole_lib_cov'):
    """Write two LaTeX tables comparing state+lib. IAU vs state+lib. Fit. results.

    Table 1 — IAUPole baseline:
        Columns: Parameter | Group | Unit | p0(IAU) | state+lib.(IAU) | state+lib.(Fit.) | Δ IAU | Δ Fit.
        Δ = final − iau_initial.  For unestimated params (α₀, δ₀) the model's
        own initial value is used in the estimation column, so Δ IAU = 0.

    Table 2 — FitPole baseline: same but Δ = final − fit_initial.

    Requires preamble: \\usepackage{booktabs,siunitx,graphicx}
    """

    # ── helpers ──────────────────────────────────────────────────────────────
    def _esc(s):
        return s.replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')

    def _p2tex(lbl):
        _MAP = {
            'α₀': r'$\alpha_0$',       'δ₀': r'$\delta_0$',
            'α₁': r'$\alpha_1$',       'δ₁': r'$\delta_1$',
            'α̇₀': r'$\dot{\alpha}_0$', 'δ̇₀': r'$\dot{\delta}_0$',
        }
        return _MAP.get(lbl, _esc(lbl))

    def _num(v):
        return rf'\num{{{v:.5e}}}'

    def _extract_all(sn):
        """Return (initial_dict, final_dict) — both keyed by param label."""
        sd = sims.get(sn, {})
        if 'parameter_history' not in sd or 'est_parameters' not in sd:
            return {}, {}
        ph   = sd['parameter_history']
        lbls, _, _ = get_parameter_info(sd['est_parameters'])
        return ({lbl: ph[i, 0]  for i, lbl in enumerate(lbls)},
                {lbl: ph[i, -1] for i, lbl in enumerate(lbls)})

    # Full initial values: use the most complete reference sim for each model.
    iau_ref_full, _ = _get_iau_reference(sims, names)
    # FitPole full initial: aggregate across all non-IAUPole sims (first-seen).
    fit_ref_full = {}
    for sn in names:
        if sn.startswith('IAUPole'):
            continue
        sd = sims.get(sn, {})
        if 'parameter_history' not in sd or 'est_parameters' not in sd:
            continue
        ph    = sd['parameter_history']
        lbls_, _, _ = get_parameter_info(sd['est_parameters'])
        for i, lbl in enumerate(lbls_):
            if lbl not in fit_ref_full:
                fit_ref_full[lbl] = ph[i, 0]

    # Final values for the two target sims (only params they estimated).
    iau_init, iau_final = _extract_all(iau_sim)
    fit_init, fit_final = _extract_all(fitpole_sim)

    if not iau_ref_full and not fit_ref_full:
        print("  SKIP generate_final_estimation_tables: no reference data.")
        return

    # ── canonical param list ──────────────────────────────────────────────────
    seen, param_rows = set(), []
    for sn in names:
        sd = sims.get(sn, {})
        if 'parameter_history' not in sd or 'est_parameters' not in sd:
            continue
        lbls_, grps_, _ = get_parameter_info(sd['est_parameters'])
        for lbl, grp in zip(lbls_, grps_):
            if grp not in _TARGET_GROUPS or lbl in seen:
                continue
            seen.add(lbl)
            sc, unit = _GROUP_SCALE.get(grp, (1.0, '---'))
            param_rows.append((lbl, grp, sc, unit))

    # For each param: resolved display values.
    # iau_col[lbl] = iau_final if estimated in iau_sim, else full IAU initial.
    # fit_col[lbl] = fit_final if estimated in fitpole_sim, else full Fit initial.
    def _col(lbl, final_d, full_ref):
        return final_d[lbl] if lbl in final_d else full_ref.get(lbl)

    lines = []
    lines.append(r'% ============================================================')
    lines.append(r'% Final estimation comparison tables')
    lines.append(r'% state+lib.(IAU): ' + iau_sim)
    lines.append(r'% state+lib.(Fit.): ' + fitpole_sim)
    lines.append(r'% Preamble: \usepackage{booktabs,siunitx,graphicx}')
    lines.append(r'% ============================================================')
    lines.append('')

    for tbl_idx, (baseline_label, baseline_ref) in enumerate([
        (r'$p_{\mathrm{IAU},0}$', iau_ref_full),
        (r'$p_{\mathrm{Fit},0}$', fit_ref_full),
    ], start=1):
        bname  = 'IAU' if tbl_idx == 1 else 'Fit'
        tlabel = f'tab:final_est_{bname.lower()}_baseline'

        if tbl_idx == 1:
            caption = (
                r'Estimated parameter values for \texttt{state+lib.}\ '
                r'simulations referenced to the IAU~2015 a~priori $p_{\mathrm{IAU},0}$. '
                r'$\Delta = p_{\mathrm{final}} - p_{\mathrm{IAU},0}$. '
                r'For unestimated parameters the model initial value is shown.'
            )
        else:
            caption = (
                r'Estimated parameter values for \texttt{state+lib.}\ '
                r'simulations referenced to the NEP097-fitted initial values '
                r'$p_{\mathrm{Fit},0}$. '
                r'$\Delta = p_{\mathrm{final}} - p_{\mathrm{Fit},0}$. '
                r'For unestimated parameters the model initial value is shown.'
            )

        lines.append(rf'% ---- Table {tbl_idx}: {bname} baseline ----')
        lines.append(r'\begin{table}[htbp]')
        lines.append(r'  \centering')
        lines.append(rf'  \caption{{{caption}}}')
        lines.append(rf'  \label{{{tlabel}}}')
        lines.append(r'  \resizebox{\textwidth}{!}{%')
        lines.append(r'  \begin{tabular}{llcrrrrr}')
        lines.append(r'    \toprule')
        lines.append(
            r'    Parameter & Group & Unit'
            rf' & {baseline_label}'
            r' & state+lib.\ (IAU) & state+lib.\ (Fit.)'
            r' & $\Delta$\,IAU & $\Delta$\,Fit. \\'
        )
        lines.append(r'    \midrule')

        prev_grp = None
        for lbl, grp, sc, unit in param_rows:
            if prev_grp is not None and grp != prev_grp:
                lines.append(r'    \midrule')
            prev_grp = grp

            lbl_tex  = _p2tex(lbl)
            base_raw = baseline_ref.get(lbl)

            if base_raw is None:
                lines.append(f'    {lbl_tex} & {_esc(grp)} & {unit}'
                             r' & --- & --- & --- & --- & --- \\')
                continue

            iau_v = _col(lbl, iau_final, iau_ref_full)
            fit_v = _col(lbl, fit_final, fit_ref_full)

            def _cell(v):
                return _num(v * sc) if v is not None else '---'

            def _delta(v, ref):
                if v is not None and ref is not None:
                    return _num((v - ref) * sc)
                return '---'

            lines.append(
                f'    {lbl_tex} & {_esc(grp)} & {unit}'
                f' & {_num(base_raw * sc)}'
                f' & {_cell(iau_v)} & {_cell(fit_v)}'
                rf' & {_delta(iau_v, base_raw)} & {_delta(fit_v, base_raw)} \\'
            )

        lines.append(r'    \bottomrule')
        lines.append(r'  \end{tabular}')
        lines.append(r'  }% end resizebox')
        lines.append(r'\end{table}')
        lines.append('')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved: {out_path}")


def generate_parameter_tables(sims, names, labels, out_path):
    """Write two LaTeX tables (.tex) for direct inclusion in Overleaf.

    Requires in the preamble:
        \\usepackage{booktabs}
        \\usepackage{siunitx}
        \\usepackage{graphicx}   % for \\resizebox

    Table 1 — Initial values: IAU initial, FitPole initial, Δ [unit], Δ [%].
               Covers ALL parameter groups (state + pole + librations).
    Table 2 — Split IAU / FitPole sub-tables.  Rows = parameters,
               columns = simulations.  Cells = final − IAU_initial in
               physical units (km, km/s, deg).
    """

    # ── small helpers ────────────────────────────────────────────────────────
    def _esc(s):
        """Escape LaTeX special characters."""
        return s.replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')

    def _p2tex(lbl):
        """Convert parameter label to LaTeX math."""
        _MAP = {
            'α₀': r'$\alpha_0$',     'δ₀': r'$\delta_0$',
            'α₁': r'$\alpha_1$',     'δ₁': r'$\delta_1$',
            'α̇₀': r'$\dot{\alpha}_0$', 'δ̇₀': r'$\dot{\delta}_0$',
        }
        return _MAP.get(lbl, _esc(lbl))

    def _num(v):
        """Format a number as \\num{} for siunitx."""
        return rf'\num{{{v:.5e}}}'

    def _pct(v):
        """Format a percentage to 2 decimal places."""
        return f'{v:.2f}'

    def _sim_short(lb):
        s = lb.replace('IAUPole_', '').replace('SimPole_', '').replace('FitPole_', '')
        return _esc(s)

    def _diff(raw_diff, sc):
        """Physical difference formatted with siunitx."""
        return rf'\num{{{raw_diff * sc:.5e}}}'

    # ── build canonical param list from ALL sims (fixes missing pole params) ──
    # iau_ref_in already aggregates from all IAUPole sims.
    iau_ref_in, _ = _get_iau_reference(sims, names)

    # FitPole reference: initial value for each param, first-seen across all non-IAUPole sims.
    sp_ref_in = {}
    for sn in names:
        if sn.startswith('IAUPole'):
            continue
        sd = sims.get(sn, {})
        if 'parameter_history' not in sd or 'est_parameters' not in sd:
            continue
        ph_     = sd['parameter_history']
        lbls_, _, _ = get_parameter_info(sd['est_parameters'])
        for i, lbl in enumerate(lbls_):
            if lbl not in sp_ref_in:
                sp_ref_in[lbl] = ph_[i, 0]

    # Iterate all sims in order to build the canonical param list.
    seen       = set()
    param_rows = []   # list of (lbl, grp, sc, unit)
    for sn in names:
        sd = sims.get(sn, {})
        if 'parameter_history' not in sd or 'est_parameters' not in sd:
            continue
        lbls_, grps_, _ = get_parameter_info(sd['est_parameters'])
        for lbl, grp in zip(lbls_, grps_):
            if grp not in _TARGET_GROUPS or lbl in seen:
                continue
            seen.add(lbl)
            sc, unit = _GROUP_SCALE.get(grp, (1.0, '---'))
            param_rows.append((lbl, grp, sc, unit))

    iau_names = [n for n in names if n.startswith('IAUPole')]
    sp_names  = [n for n in names if not n.startswith('IAUPole')]
    iau_lbls  = [labels[names.index(n)] for n in iau_names]
    sp_lbls   = [labels[names.index(n)] for n in sp_names]
    # Detect the FitPole prefix from actual sim names for captions.
    _fit_prefix = next(
        (n.split('_')[0] for n in sp_names if '_' in n),
        'FitPole'
    )

    lines = []
    lines.append(r'% ============================================================')
    lines.append(r'% Parameter tables — Neptune/Triton orbit estimation thesis')
    lines.append(r'% Preamble: \usepackage{booktabs,siunitx,graphicx}')
    lines.append(r'% ============================================================')
    lines.append('')

    # ── TABLE 1: IAU initial vs FitPole initial ───────────────────────────────
    lines.append(r'% ---- Table 1: initial parameter values (IAU vs FitPole) ----')
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'  \centering')
    lines.append(
        r'  \caption{Initial parameter values for the IAU and FitPole rotation models. '
        r'$\Delta = p_{\mathrm{FitPole,0}} - p_{\mathrm{IAU,0}}$ in the listed unit; '
        r'$\Delta[\%] = \Delta\,/\,|p_{\mathrm{IAU,0}}| \times 100$.}'
    )
    lines.append(r'  \label{tab:initial_params}')
    lines.append(r'  \resizebox{\textwidth}{!}{%')
    lines.append(r'  \begin{tabular}{llcrrrrr}')
    lines.append(r'    \toprule')
    lines.append(
        r'    Parameter & Group & Unit'
        r' & IAU initial & FitPole initial'
        r' & $\Delta$ & $\Delta\,[\%]$ \\'
    )
    lines.append(r'    \midrule')

    prev_grp = None
    for lbl, grp, sc, unit in param_rows:
        if prev_grp is not None and grp != prev_grp:
            lines.append(r'    \midrule')
        prev_grp = grp

        lbl_tex = _p2tex(lbl)
        iau_raw = iau_ref_in.get(lbl)
        sp_raw  = sp_ref_in.get(lbl)

        if iau_raw is None:
            lines.append(f'    {lbl_tex} & {_esc(grp)} & {unit}'
                         r' & --- & --- & --- & --- \\')
            continue

        iau_s = iau_raw * sc
        if sp_raw is not None:
            sp_s  = sp_raw * sc
            diff  = sp_s - iau_s
            pct   = diff / abs(iau_s) * 100 if iau_s != 0 else 0.0
            lines.append(
                f'    {lbl_tex} & {_esc(grp)} & {unit}'
                f' & {_num(iau_s)} & {_num(sp_s)}'
                rf' & {_num(diff)} & {pct:.2f} \\'
            )
        else:
            lines.append(f'    {lbl_tex} & {_esc(grp)} & {unit}'
                         f' & {_num(iau_s)} & --- & --- & --- \\')

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'  }% end resizebox')
    lines.append(r'\end{table}')
    lines.append('')

    # ── TABLE 2: final − IAU_initial in physical units ───────────────────────
    if not iau_ref_in:
        lines.append(r'% TABLE 2: no IAU reference data available.')
        lines.append('')
    else:
        subtables = [
            (iau_names, iau_lbls, 'IAU',
             r'IAU-model simulations',
             'tab:update_iau'),
            (sp_names, sp_lbls, _fit_prefix,
             rf'{_fit_prefix}-model simulations (all referenced to IAU initial)',
             f'tab:update_{_fit_prefix.lower()}'),
        ]

        for grp_names, grp_labels, grp_tag, caption_detail, tbl_label in subtables:
            if not grp_names:
                continue

            col_spec = 'lll' + 'r' * len(grp_names)

            lines.append(f'% ---- Table 2 ({grp_tag}): final - IAU\_initial [physical units] ----')
            lines.append(r'\begin{table}[htbp]')
            lines.append(r'  \centering')
            lines.append(
                rf'  \caption{{Parameter update $\Delta p = p_{{\mathrm{{final}}}} - '
                rf'p_{{\mathrm{{IAU,0}}}}$ in physical units for {caption_detail}.}}'
            )
            lines.append(rf'  \label{{{tbl_label}}}')
            lines.append(r'  \resizebox{\textwidth}{!}{%')
            lines.append(rf'  \begin{{tabular}}{{{col_spec}}}')
            lines.append(r'    \toprule')

            hdr_parts = ['Parameter', 'Group', 'Unit'] + [_sim_short(lb) for lb in grp_labels]
            lines.append('    ' + ' & '.join(hdr_parts) + r' \\')
            lines.append(r'    \midrule')

            prev_grp = None
            for lbl, grp, sc, unit in param_rows:
                if prev_grp is not None and grp != prev_grp:
                    lines.append(r'    \midrule')
                prev_grp = grp

                iau_ref  = iau_ref_in.get(lbl)
                row_parts = [_p2tex(lbl), _esc(grp), unit]

                for sn in grp_names:
                    if iau_ref is None:
                        row_parts.append('---')
                        continue
                    sd = sims.get(sn, {})
                    if 'parameter_history' not in sd or 'est_parameters' not in sd:
                        row_parts.append('---')
                        continue
                    ph_n         = sd['parameter_history']
                    lbls_n, _, _ = get_parameter_info(sd['est_parameters'])
                    if lbl not in lbls_n:
                        row_parts.append('---')
                        continue
                    final = ph_n[lbls_n.index(lbl), -1]
                    row_parts.append(_diff(final - iau_ref, sc))

                lines.append('    ' + ' & '.join(row_parts) + r' \\')

            lines.append(r'    \bottomrule')
            lines.append(rf'  \end{{tabular}}')
            lines.append(r'  }% end resizebox')
            lines.append(r'\end{table}')
            lines.append('')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved: {out_path}")


# ============================================================================
# INITIAL PARAMETER VALUES TABLE
# ============================================================================

def generate_initial_values_table(sims, cfg_dict, out_path):
    """Write a LaTeX table comparing initial parameter values for IAU vs Jacobson.

    Reads parameter_history[:, 0] (first iteration = initial values) from two
    reference simulations specified in cfg_dict:
      'iau_sim'      — covers state + GM + SH + full IAU pole parameters
      'jacobson_sim' — covers state + full Jacobson pole parameters (incl. deg-2)

    GM and SH are model-agnostic; if absent from the Jacobson sim the IAU
    initial value is shown in both columns (noted in the table caption).

    Requires LaTeX preamble: \\usepackage{booktabs,siunitx}
    """
    iau_sim_name = cfg_dict.get('iau_sim')
    jac_sim_name = cfg_dict.get('jacobson_sim')
    caption      = cfg_dict.get('caption', 'Initial parameter values.')
    label        = cfg_dict.get('label',   'tab:initial-params')

    sd_iau = sims.get(iau_sim_name, {})
    sd_jac = sims.get(jac_sim_name, {})

    if 'parameter_history' not in sd_iau or 'est_parameters' not in sd_iau:
        print(f"  SKIP initial_values_table: no parameter data for '{iau_sim_name}'.")
        return

    def _extract_initial(sd):
        if 'parameter_history' not in sd or 'est_parameters' not in sd:
            return {}
        ph   = sd['parameter_history']
        lbls, _, _ = get_parameter_info(sd['est_parameters'])
        return {lbl: ph[i, 0] for i, lbl in enumerate(lbls)}

    iau_vals = _extract_initial(sd_iau)
    jac_vals = _extract_initial(sd_jac)

    # Build the complete ordered parameter list: IAU params first, then any
    # Jacobson-only params (e.g. α₂, δ₂ degree-2 librations).
    all_params = []
    seen = set()
    for sd in [sd_iau, sd_jac]:
        if 'est_parameters' not in sd:
            continue
        lbls, grps, units = get_parameter_info(sd['est_parameters'])
        for lbl, grp, unit in zip(lbls, grps, units):
            if lbl in seen:
                continue
            seen.add(lbl)
            all_params.append((lbl, grp, unit))

    _SEC_PER_YR = 365.25 * 86400.0
    _SCALE_MAP = {
        'Position':            (1e-3,                        r'km'),
        'Velocity':            (1e-3,                        r'km\,s$^{-1}$'),
        'Gravity':             (1e-9,                        r'km$^3$\,s$^{-2}$'),
        'Pole Position':       (_RAD_TO_DEG,                 r'deg'),
        'Pole Rate':           (_RAD_TO_DEG * _SEC_PER_YR,  r'deg\,yr$^{-1}$'),
        'Pole Librations':     (_RAD_TO_DEG,                 r'deg'),
        'Spherical Harmonics': (1.0,                         r'---'),
    }

    def _p2tex(lbl):
        _MAP = {
            'X':     r'$X$',
            'Y':     r'$Y$',
            'Z':     r'$Z$',
            'VX':    r'$\dot{X}$',
            'VY':    r'$\dot{Y}$',
            'VZ':    r'$\dot{Z}$',
            'GM_Nep': r'$GM_\mathrm{N}$',
            'GM_Tri': r'$GM_\mathrm{T}$',
            'C₂₀':   r'$\bar{C}_{20}$',
            'C₄₀':   r'$\bar{C}_{40}$',
            'α₀':    r'$\alpha_0$',
            'δ₀':    r'$\delta_0$',
            'α̇₀':   r'$\dot{\alpha}_0$',
            'δ̇₀':   r'$\dot{\delta}_0$',
            'α₁':    r'$\alpha_1$',
            'δ₁':    r'$\delta_1$',
            'α₂':    r'$\alpha_2$',
            'δ₂':    r'$\delta_2$',
        }
        return _MAP.get(lbl, lbl)

    def _fmtval(v, sc):
        if v is None:
            return r'\text{---}'
        return rf'\num{{{v * sc:.6e}}}'

    lines = []
    lines.append(r'% ============================================================')
    lines.append(r'% Initial parameter values table — SimObs_ParameterAnalysis')
    lines.append(r'% Preamble: \usepackage{booktabs,siunitx}')
    lines.append(r'% ============================================================')
    lines.append('')
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'  \centering')
    lines.append(rf'  \caption{{{caption}}}')
    lines.append(rf'  \label{{{label}}}')
    lines.append(r'  \begin{tabular}{llcrr}')
    lines.append(r'    \toprule')
    lines.append(r'    Parameter & Group & Unit & IAU 2015 & Jacobson 2009 \\')
    lines.append(r'    \midrule')

    prev_grp = None
    for lbl, grp, _ in all_params:
        if prev_grp is not None and grp != prev_grp:
            lines.append(r'    \midrule')
        prev_grp = grp

        sc, unit_str = _SCALE_MAP.get(grp, (1.0, r'---'))
        iau_v = iau_vals.get(lbl)
        jac_v = jac_vals.get(lbl)

        # GM and SH are model-agnostic: use IAU value for the Jacobson column too.
        if grp in ('Gravity', 'Spherical Harmonics') and jac_v is None and iau_v is not None:
            jac_v = iau_v

        lines.append(
            f'    {_p2tex(lbl)} & {grp} & {unit_str}'
            f' & ${_fmtval(iau_v, sc)}$ & ${_fmtval(jac_v, sc)}$ \\\\'
        )

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table}')
    lines.append('')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved: {out_path}")


# ============================================================================
# GM PARAMETER ESTIMATION TABLE
# ============================================================================

def generate_gm_param_table(sims, cfg_dict, out_path):
    """Write a LaTeX table of GM estimation results for selected simulations.

    For each sim in cfg_dict['sims'], writes a sub-table with rows for each
    estimated GM parameter showing:
      Parameter | Initial Value (m³/s²) | Final Value (m³/s²) | Δ | Δ [%]

    Values are read directly from parameter_history using get_parameter_info
    to ensure correct index mapping (avoids off-by-one bugs).

    Requires LaTeX preamble: \\usepackage{booktabs,siunitx}
    """
    sim_names = cfg_dict.get('sims', [])
    caption   = cfg_dict.get('caption', 'GM parameter estimation results.')
    label     = cfg_dict.get('label',   'tab:gm-params')

    if not sim_names:
        print("  SKIP gm_param_table: no sims specified.")
        return

    def _p2tex(lbl):
        return {'GM_Nep': r'$GM_\mathrm{N}$', 'GM_Tri': r'$GM_\mathrm{T}$'}.get(lbl, lbl)

    def _num(v):
        return rf'\num{{{v:.6e}}}'

    all_blocks = []
    for sim_name in sim_names:
        sd = sims.get(sim_name, {})
        if 'parameter_history' not in sd or 'est_parameters' not in sd:
            print(f"  SKIP gm_param_table: no parameter data for '{sim_name}'.")
            continue

        ph   = sd['parameter_history']           # shape (n_params, n_iters)
        lbls, grps, _ = get_parameter_info(sd['est_parameters'])

        # Collect only GM rows with their correct indices.
        rows = []
        for idx, (lbl, grp) in enumerate(zip(lbls, grps)):
            if grp != 'Gravity':
                continue
            init_val  = ph[idx, 0]
            final_val = ph[idx, -1]
            delta     = final_val - init_val
            pct       = delta / abs(init_val) * 100 if init_val != 0 else 0.0
            rows.append((lbl, init_val, final_val, delta, pct))

        if not rows:
            continue
        all_blocks.append((sim_name, rows))

    if not all_blocks:
        print("  SKIP gm_param_table: no GM parameters found in any sim.")
        return

    lines = []
    lines.append(r'% ============================================================')
    lines.append(r'% GM parameter estimation table — SimObs_ParameterAnalysis')
    lines.append(r'% Preamble: \usepackage{booktabs,siunitx}')
    lines.append(r'% ============================================================')
    lines.append('')
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'  \centering')
    lines.append(rf'  \caption{{{caption}}}')
    lines.append(rf'  \label{{{label}}}')
    lines.append(r'  \begin{tabular}{llrrrr}')
    lines.append(r'    \toprule')
    lines.append(
        r'    Simulation & Parameter'
        r' & Initial ($m^3\,s^{-2}$)'
        r' & Final ($m^3\,s^{-2}$)'
        r' & $\Delta$'
        r' & $\Delta\,[\%]$ \\'
    )
    lines.append(r'    \midrule')

    for block_idx, (sim_name, rows) in enumerate(all_blocks):
        if block_idx > 0:
            lines.append(r'    \midrule')
        sim_label = sim_name.replace('_', r'\_')
        for row_idx, (lbl, init_v, final_v, delta, pct) in enumerate(rows):
            # Only print sim name on first row of each block.
            sim_cell = rf'\texttt{{{sim_label}}}' if row_idx == 0 else ''
            lines.append(
                f'    {sim_cell} & {_p2tex(lbl)}'
                f' & ${_num(init_v)}$'
                f' & ${_num(final_v)}$'
                f' & ${_num(delta)}$'
                f' & ${pct:.3f}\\%$ \\\\'
            )

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table}')
    lines.append('')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved: {out_path}")


# ============================================================================
# CONDITION NUMBER TABLE
# ============================================================================

def generate_condition_number_table(sims, names, labels, cfg_dict, out_path):
    """Write a LaTeX table of correlation-matrix condition numbers.

    For each sim in cfg_dict['sims'] the condition number of the stored
    correlation matrix (key 'correlations') is computed via
    numpy.linalg.cond().  A large value indicates near-linear parameter
    dependencies (ill-conditioned estimation).

    Columns: Simulation label | Estimated parameters | Condition number

    Requires LaTeX preamble: \\usepackage{booktabs,siunitx}
    """
    sim_names = cfg_dict.get('sims', [])
    caption   = cfg_dict.get('caption', 'Condition numbers of correlation matrices.')
    label     = cfg_dict.get('label',   'tab:condition-numbers')

    if not sim_names:
        print("  SKIP condition_number_table: no sims specified.")
        return

    def _esc(s):
        return s.replace('_', r'\_')

    def _param_summary(est_params):
        """Short human-readable list of estimated parameter groups."""
        _MAP = {
            'initial_state':                      'state',
            'GM_Neptune':                         r'$GM_\mathrm{N}$',
            'GM_Triton':                          r'$GM_\mathrm{T}$',
            'iau_rotation_model_pole':            r'$\alpha_0,\delta_0$',
            'iau_rotation_model_pole_rate':       r'$\dot{\alpha}_0,\dot{\delta}_0$',
            'iau_rotation_model_pole_librations': r'$\alpha_1,\delta_1$',
            'pole_librations_deg2':               r'$\alpha_2,\delta_2$',
            'spherical_harmonics':                r'$\bar{C}_{20},\bar{C}_{40}$',
        }
        return ', '.join(_MAP.get(p, _esc(p)) for p in est_params)

    rows = []
    for sn in sim_names:
        sd = sims.get(sn, {})
        if 'correlations' not in sd:
            print(f"  WARNING condition_number_table: no correlations for '{sn}', skipping.")
            continue
        corr = np.asarray(sd['correlations'])
        cond = float(np.linalg.cond(corr))
        lbl  = labels[names.index(sn)] if sn in names else _esc(sn)
        est  = _param_summary(sd.get('est_parameters', []))
        rows.append((lbl, est, cond))

    if not rows:
        print("  SKIP condition_number_table: no valid data found.")
        return

    lines = []
    lines.append(r'% ============================================================')
    lines.append(r'% Condition number table — correlation matrices')
    lines.append(r'% Preamble: \usepackage{booktabs,siunitx}')
    lines.append(r'% ============================================================')
    lines.append('')
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'  \centering')
    lines.append(rf'  \caption{{{caption}}}')
    lines.append(rf'  \label{{{label}}}')
    lines.append(r'  \begin{tabular}{llr}')
    lines.append(r'    \toprule')
    lines.append(r'    Simulation & Estimated parameters & Condition number \\')
    lines.append(r'    \midrule')
    for lbl, est, cond in rows:
        lines.append(rf'    {lbl} & {est} & $\num{{{cond:.4e}}}$ \\')
    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table}')
    lines.append('')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved: {out_path}")


# ============================================================================
# SH SUMMARY TABLE  (condition number + SH parameter update)
# ============================================================================

def generate_sh_summary_table(sims, names, labels, cfg_dict, out_path):
    """Write a LaTeX table combining the correlation-matrix condition number
    and the spherical-harmonic parameter updates for selected simulations.

    Columns:
        Simulation | Estimated parameters | cond(C) | ΔC̄₂₀ | ΔC̄₂₀ [%] | ΔC̄₄₀ | ΔC̄₄₀ [%]

    cfg_dict keys
    -------------
    sims    — list of sim names to include
    caption — LaTeX table caption
    label   — LaTeX \\label key

    Requires LaTeX preamble: \\usepackage{booktabs,siunitx}
    """
    sim_names = cfg_dict.get('sims', [])
    caption   = cfg_dict.get('caption', 'Spherical harmonics estimation summary.')
    label     = cfg_dict.get('label',   'tab:sh-summary')

    if not sim_names:
        print('  SKIP sh_summary_table: no sims specified.')
        return

    def _esc(s):
        return s.replace('_', r'\_')

    def _param_summary(est_params):
        _MAP = {
            'initial_state':       'state',
            'GM_Neptune':          r'$GM_\mathrm{N}$',
            'GM_Triton':           r'$GM_\mathrm{T}$',
            'spherical_harmonics': r'$\bar{C}_{20},\bar{C}_{40}$',
        }
        return ', '.join(_MAP.get(p, _esc(p)) for p in est_params)

    _SH_LABELS = {
        'C20': r'$\Delta\bar{C}_{20}$',
        'C40': r'$\Delta\bar{C}_{40}$',
    }

    rows = []
    for sn in sim_names:
        sd = sims.get(sn, {})
        if 'correlations' not in sd or 'parameter_history' not in sd:
            print(f"  WARNING sh_summary_table: missing data for '{sn}', skipping.")
            continue

        corr = np.asarray(sd['correlations'])
        cond = float(np.linalg.cond(corr))
        lbl  = labels[names.index(sn)] if sn in names else _esc(sn)
        est  = _param_summary(sd.get('est_parameters', []))

        ph   = sd['parameter_history']
        lbls, grps, _ = get_parameter_info(sd['est_parameters'])

        sh_updates = {}
        for idx, (plbl, grp) in enumerate(zip(lbls, grps)):
            if grp != 'Spherical Harmonics':
                continue
            init_val  = ph[idx, 0]
            final_val = ph[idx, -1]
            delta     = final_val - init_val
            pct       = delta / abs(init_val) * 100 if init_val != 0 else 0.0
            # Map generic labels to C20/C40 keys by order of appearance.
            key = 'C20' if 'C20' not in sh_updates else 'C40'
            sh_updates[key] = (delta, pct)

        c20_d, c20_p = sh_updates.get('C20', (float('nan'), float('nan')))
        c40_d, c40_p = sh_updates.get('C40', (float('nan'), float('nan')))
        rows.append((lbl, est, cond, c20_d, c20_p, c40_d, c40_p))

    if not rows:
        print('  SKIP sh_summary_table: no valid data found.')
        return

    def _num(v):
        if v != v:  # nan check
            return r'\text{---}'
        return rf'\num{{{v:.4e}}}'

    def _pct(v):
        if v != v:
            return r'\text{---}'
        return rf'\num{{{v:.2f}}}'

    lines = []
    lines.append(r'% ============================================================')
    lines.append(r'% SH summary table — condition number + parameter update')
    lines.append(r'% Preamble: \usepackage{booktabs,siunitx}')
    lines.append(r'% ============================================================')
    lines.append('')
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'  \centering')
    lines.append(rf'  \caption{{{caption}}}')
    lines.append(rf'  \label{{{label}}}')
    lines.append(r'  \resizebox{\textwidth}{!}{%')
    lines.append(r'  \begin{tabular}{llrrrrrr}')
    lines.append(r'    \toprule')
    lines.append(
        r'    Simulation & Estimated parameters & $\kappa(\mathbf{C})$'
        r' & $\Delta\bar{C}_{20}$ & $\Delta\bar{C}_{20}$\,[\%]'
        r' & $\Delta\bar{C}_{40}$ & $\Delta\bar{C}_{40}$\,[\%] \\'
    )
    lines.append(r'    \midrule')
    for lbl, est, cond, c20_d, c20_p, c40_d, c40_p in rows:
        lines.append(
            rf'    {lbl} & {est} & $\num{{{cond:.4e}}}$'
            rf' & ${_num(c20_d)}$ & ${_pct(c20_p)}$'
            rf' & ${_num(c40_d)}$ & ${_pct(c40_p)}$ \\'
        )
    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}}')
    lines.append(r'\end{table}')
    lines.append('')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved: {out_path}")


# ============================================================================
# POLE ESTIMATION RESULTS TABLE
# ============================================================================

def generate_pole_estimation_table(sims, cfg_dict, out_path):
    """Write a LaTeX table extending the initial-values table with final estimated
    values (and Δ wrt IAU initial) for selected result simulations.

    Columns: Parameter | Group | Unit | IAU 2015 | Jacobson 2009 |
             <result_sim_1 final> | Δ₁ | <result_sim_2 final> | Δ₂ | …

    cfg_dict keys
    -------------
    iau_sim      — sim providing initial IAU values (parameter_history[:, 0])
    jacobson_sim — sim providing initial Jacobson values
    result_sims  — list of (sim_name, col_label) tuples for extra columns
    caption, label — LaTeX caption / label

    Non-estimated parameters in a result sim show '---' for both Final and Δ.

    Requires LaTeX preamble: \\usepackage{booktabs,siunitx}
    """
    iau_sim_name = cfg_dict.get('iau_sim')
    jac_sim_name = cfg_dict.get('jacobson_sim')
    result_sims  = cfg_dict.get('result_sims', [])
    caption      = cfg_dict.get('caption', 'Pole estimation results.')
    label        = cfg_dict.get('label',   'tab:pole-estimation')

    sd_iau = sims.get(iau_sim_name, {})
    sd_jac = sims.get(jac_sim_name, {})

    if 'parameter_history' not in sd_iau or 'est_parameters' not in sd_iau:
        print(f"  SKIP pole_estimation_table: no parameter data for '{iau_sim_name}'.")
        return

    _SEC_PER_YR = 365.25 * 86400.0
    _SCALE_MAP = {
        'Position':            (1e-3,                        r'km'),
        'Velocity':            (1e-3,                        r'km\,s$^{-1}$'),
        'Gravity':             (1e-9,                        r'km$^3$\,s$^{-2}$'),
        'Pole Position':       (_RAD_TO_DEG,                 r'deg'),
        'Pole Rate':           (_RAD_TO_DEG * _SEC_PER_YR,  r'deg\,yr$^{-1}$'),
        'Pole Librations':     (_RAD_TO_DEG,                 r'deg'),
        'Spherical Harmonics': (1.0,                         r'---'),
    }

    _P2TEX = {
        'X':     r'$X$',        'Y':     r'$Y$',       'Z':     r'$Z$',
        'VX':    r'$\dot{X}$',  'VY':    r'$\dot{Y}$', 'VZ':    r'$\dot{Z}$',
        'GM_Nep': r'$GM_\mathrm{N}$',
        'GM_Tri': r'$GM_\mathrm{T}$',
        'C₂₀':   r'$\bar{C}_{20}$',
        'C₄₀':   r'$\bar{C}_{40}$',
        'α₀':    r'$\alpha_0$',   'δ₀':   r'$\delta_0$',
        'α̇₀':   r'$\dot{\alpha}_0$', 'δ̇₀':  r'$\dot{\delta}_0$',
        'α₁':    r'$\alpha_1$',   'δ₁':   r'$\delta_1$',
        'α₂':    r'$\alpha_2$',   'δ₂':   r'$\delta_2$',
    }

    def _p2tex(lbl):
        return _P2TEX.get(lbl, lbl)

    def _extract_vals(sd, col_idx):
        """Return dict lbl → value from parameter_history column col_idx."""
        if 'parameter_history' not in sd or 'est_parameters' not in sd:
            return {}
        ph   = sd['parameter_history']
        idx  = min(col_idx, ph.shape[1] - 1)
        lbls, _, _ = get_parameter_info(sd['est_parameters'])
        return {lbl: ph[i, idx] for i, lbl in enumerate(lbls)}

    iau_vals = _extract_vals(sd_iau, 0)
    jac_vals = _extract_vals(sd_jac, 0)

    # Build ordered parameter list from IAU sim (primary), then Jacobson-only extras
    all_params = []
    seen = set()
    for sd in [sd_iau, sd_jac]:
        if 'est_parameters' not in sd:
            continue
        lbls, grps, units = get_parameter_info(sd['est_parameters'])
        for lbl, grp, unit in zip(lbls, grps, units):
            if lbl in seen:
                continue
            seen.add(lbl)
            all_params.append((lbl, grp, unit))

    # Pre-extract final values for each result sim
    result_data = []
    for sim_name, col_label in result_sims:
        sd_r = sims.get(sim_name, {})
        final_vals = _extract_vals(sd_r, -1)
        result_data.append((col_label, final_vals))

    def _fmtval(v, sc):
        if v is None:
            return r'\text{---}'
        return rf'\num{{{v * sc:.6e}}}'

    def _fmtdelta(v_final, v_ref, sc):
        if v_final is None or v_ref is None:
            return r'\text{---}'
        delta = (v_final - v_ref) * sc
        return rf'\num{{{delta:.6e}}}'

    # Column spec: llcr r (rr)*
    n_result_cols = len(result_data)
    col_spec = 'llc' + 'r' * (2 + 2 * n_result_cols)

    # Header row
    header_parts = [r'Parameter', r'Group', r'Unit', r'IAU 2015', r'Jacobson 2009']
    for col_label, _ in result_data:
        header_parts.append(col_label)
        header_parts.append(r'$\Delta$')
    header_line = '    ' + ' & '.join(header_parts) + r' \\'

    lines = []
    lines.append(r'% ============================================================')
    lines.append(r'% Pole estimation results table — SimObs_ParameterAnalysis')
    lines.append(r'% Preamble: \usepackage{booktabs,siunitx}')
    lines.append(r'% ============================================================')
    lines.append('')
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'  \centering')
    lines.append(rf'  \caption{{{caption}}}')
    lines.append(rf'  \label{{{label}}}')
    lines.append(rf'  \begin{{tabular}}{{{col_spec}}}')
    lines.append(r'    \toprule')
    lines.append(header_line)
    lines.append(r'    \midrule')

    prev_grp = None
    for lbl, grp, _ in all_params:
        if prev_grp is not None and grp != prev_grp:
            lines.append(r'    \midrule')
        prev_grp = grp

        sc, unit_str = _SCALE_MAP.get(grp, (1.0, r'---'))
        iau_v = iau_vals.get(lbl)
        jac_v = jac_vals.get(lbl)

        # GM and SH are model-agnostic
        if grp in ('Gravity', 'Spherical Harmonics') and jac_v is None and iau_v is not None:
            jac_v = iau_v

        row_cells = [
            _p2tex(lbl),
            grp,
            unit_str,
            f'${_fmtval(iau_v, sc)}$',
            f'${_fmtval(jac_v, sc)}$',
        ]
        for _, final_vals in result_data:
            v_final = final_vals.get(lbl)
            row_cells.append(f'${_fmtval(v_final, sc)}$')
            row_cells.append(f'${_fmtdelta(v_final, iau_v, sc)}$')

        lines.append('    ' + ' & '.join(row_cells) + r' \\')

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table}')
    lines.append('')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved: {out_path}")


# ============================================================================
# NAMING CONVENTION TABLE
# ============================================================================

def generate_naming_table(selected_sims, sim_labels, sim_descriptions,
                           dataset_label, out_path):
    """Write a LaTeX table mapping figure labels to weight-scheme descriptions.

    Parameters
    ----------
    selected_sims    : list of str   — internal simulation keys (from config)
    sim_labels       : list of str   — short display labels used in figures
    sim_descriptions : dict          — sim_name → human-readable description
    dataset_label    : str           — used in the table caption / label
    out_path         : str           — destination .tex file path
    """
    if not sim_descriptions:
        print("  SKIP naming table: SIM_DESCRIPTIONS not defined in config.")
        return

    def _esc(s):
        """Escape special LaTeX characters in plain-text strings."""
        for ch, rep in [('_', r'\_'), ('&', r'\&'), ('%', r'\%'), ('#', r'\#')]:
            s = s.replace(ch, rep)
        return s

    safe_label = dataset_label.replace('_', '-').lower()
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'  \centering')
    lines.append(
        rf'  \caption{{Weight-scheme naming conventions used in figures for '
        rf'\texttt{{{_esc(dataset_label)}}}.}}'
    )
    lines.append(rf'  \label{{tab:weight-naming-{safe_label}}}')
    lines.append(r'  \begin{tabular}{l p{10cm}}')
    lines.append(r'    \toprule')
    lines.append(r'    \textbf{Label} & \textbf{Description} \\')
    lines.append(r'    \midrule')

    label_map = {}
    if sim_labels is not None and len(sim_labels) == len(selected_sims):
        label_map = dict(zip(selected_sims, sim_labels))

    for sn in selected_sims:
        lbl  = label_map.get(sn, sn)
        desc = sim_descriptions.get(sn, '—')
        lines.append(
            rf'    \texttt{{{_esc(lbl)}}} & {desc} \\'
        )

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table}')
    lines.append('')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved: {out_path}")


def generate_dataset_overview_table(rows, caption, label, out_path):
    """Write a 2-column LaTeX table: Name | Description for dataset-level naming.

    Parameters
    ----------
    rows     : list of (name, description) tuples
    caption  : str — table caption
    label    : str — LaTeX \\label{} argument (without braces)
    out_path : str — destination .tex file path
    """
    def _esc(s):
        for ch, rep in [('_', r'\_'), ('&', r'\&'), ('%', r'\%'), ('#', r'\#')]:
            s = s.replace(ch, rep)
        return s

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'  \centering')
    lines.append(rf'  \caption{{{caption}}}')
    lines.append(rf'  \label{{{label}}}')
    lines.append(r'  \begin{tabular}{l p{10cm}}')
    lines.append(r'    \toprule')
    lines.append(r'    \textbf{Name} & \textbf{Description} \\')
    lines.append(r'    \midrule')
    for name, desc in rows:
        lines.append(rf'    \texttt{{{_esc(name)}}} & {desc} \\')
    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table}')
    lines.append('')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved: {out_path}")


def generate_weight_scheme_table(table_rows, dataset_label, out_path):
    """Write a LaTeX overview table of weight-scheme naming conventions.

    Parameters
    ----------
    table_rows    : list of (label, weighting_scheme, comments)
    dataset_label : str — used in caption and \\label
    out_path      : str — destination .tex file path
    """
    if not table_rows:
        print("  SKIP weight-scheme table: SIM_TABLE_ROWS not defined in config.")
        return

    safe_label = dataset_label.replace('_', '-').lower()
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'  \centering')
    lines.append(
        rf'  \caption{{Overview of weight schemes used in the \texttt{{{dataset_label}}} analysis.}}'
    )
    lines.append(rf'  \label{{tab:weight-schemes-{safe_label}}}')
    lines.append(r'  \begin{tabular}{l l l}')
    lines.append(r'    \toprule')
    lines.append(r'    \textbf{Label} & \textbf{Weighting Scheme} & \textbf{Comments} \\')
    lines.append(r'    \midrule')

    for lbl, scheme, comments in table_rows:
        lines.append(rf'    \texttt{{{lbl}}} & {scheme} & {comments} \\')

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table}')
    lines.append('')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved: {out_path}")


# ============================================================================
# RESIDUAL STATISTICS TABLE
# ============================================================================

def generate_residual_stats_table(sims, names, labels, out_path,
                                   data_source='final',
                                   caption='Mean and standard deviation of final residuals per weighting scheme [mas].',
                                   label='tab:residual-stats'):
    """Write a LaTeX table: mean and std of residuals per sim, RA and DEC.

    Rows    = one per simulation.
    Columns = Scheme | RA mean | RA std | DEC mean | DEC std  (all in mas).

    data_source : 'final' or 'initial' — selects which residual columns to use.
    """
    ra_col  = 'ra_residual_final_mas'  if data_source == 'final' else 'ra_residual_initial_mas'
    dec_col = 'dec_residual_final_mas' if data_source == 'final' else 'dec_residual_initial_mas'

    rows = []
    for sn, lbl in zip(names, labels):
        df = sims.get(sn, {}).get('residual_df', None)
        if df is None:
            continue
        ra  = df[ra_col].dropna().values  if ra_col  in df.columns else np.array([])
        dec = df[dec_col].dropna().values if dec_col in df.columns else np.array([])
        if len(ra) == 0:
            continue
        rows.append((lbl, np.mean(ra), np.std(ra), np.mean(dec), np.std(dec)))

    if not rows:
        print("  SKIP residual_stats_table: no residual data found.")
        return

    def _esc(s):
        for ch, rep in [('_', r'\_'), ('&', r'\&'), ('%', r'\%'), ('#', r'\#')]:
            s = s.replace(ch, rep)
        return s

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'  \centering')
    lines.append(rf'  \caption{{{caption}}}')
    lines.append(rf'  \label{{{label}}}')
    lines.append(r'  \begin{tabular}{l r r r r}')
    lines.append(r'    \toprule')
    lines.append(r'    \textbf{Scheme} & \textbf{RA mean} & \textbf{RA std} & \textbf{DEC mean} & \textbf{DEC std} \\')
    lines.append(r'    \multicolumn{1}{l}{} & \multicolumn{4}{c}{[mas]} \\')
    lines.append(r'    \midrule')
    for lbl, ra_m, ra_s, dec_m, dec_s in rows:
        lines.append(
            rf'    \texttt{{{_esc(lbl)}}} & {ra_m:+.2f} & {ra_s:.2f} & {dec_m:+.2f} & {dec_s:.2f} \\'
        )
    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table}')
    lines.append('')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved: {out_path}")


# ============================================================================
# POLE COMPARISON TABLE
# ============================================================================

def generate_pole_comparison_table(sims, names, labels, pole_sims, out_path):
    """Write a LaTeX table comparing pole (and position) parameter estimates
    across a set of estimation variants.

    Layout
    ------
    Rows  : α₀, δ₀, α₁, δ₁ (when available), X, Y, Z
    Cols  : Parameter | Unit | Initial | {sim1}: Est. | Δ | % | {sim2}: … | …

    The initial column is taken from the first available sim's parameter_history[:,0].
    If a sim does not estimate a given parameter, the cell shows '---'.

    Parameters
    ----------
    pole_sims : list of str
        Subset of sim names to use as estimation-variant columns (e.g.
        ``['pole_pos_IAU', 'pole_lib_IAU', 'pole_pos_lib_IAU']``).
        Only sims present in *both* ``sims`` and ``names`` are used.
    """
    # ── local helpers ─────────────────────────────────────────────────────────
    def _esc(s):
        for ch, rep in [('_', r'\_'), ('&', r'\&'), ('%', r'\%'), ('#', r'\#')]:
            s = s.replace(ch, rep)
        return s

    _P2TEX = {
        'α₀':  r'$\alpha_0$',
        'δ₀':  r'$\delta_0$',
        'α₁':  r'$\alpha_1$',
        'δ₁':  r'$\delta_1$',
        'α̇₀':  r'$\dot{\alpha}_0$',
        'δ̇₀':  r'$\dot{\delta}_0$',
        'X':   r'$X$',
        'Y':   r'$Y$',
        'Z':   r'$Z$',
    }
    def _p2tex(lbl):
        return _P2TEX.get(lbl, _esc(lbl))

    def _num(v):
        return rf'\num{{{v:.5e}}}'

    # ── resolve sim list and labels ───────────────────────────────────────────
    valid_pole_sims = [sn for sn in pole_sims if sn in sims and sn in names]
    if not valid_pole_sims:
        print("  SKIP pole comparison table: none of POLE_TABLE_SIMS found in dataset.")
        return

    col_labels = [labels[names.index(sn)] for sn in valid_pole_sims]

    # ── collect initial parameter values (first sim that has each param) ──────
    # Groups and scaling for the rows we care about.
    _WANTED_GROUPS = {'Pole Position', 'Pole Librations', 'Position'}
    _SCALE = {
        'Pole Position':   (_RAD_TO_DEG, 'deg'),
        'Pole Librations': (_RAD_TO_DEG, 'deg'),
        'Position':        (1e-3,        'km'),
    }

    # Build the canonical row list from all pole_sims in order.
    seen       = set()
    param_rows = []   # (lbl, grp, scale, unit)
    for sn in valid_pole_sims:
        sd = sims.get(sn, {})
        if 'parameter_history' not in sd or 'est_parameters' not in sd:
            continue
        lbls, grps, _ = get_parameter_info(sd['est_parameters'])
        for lbl, grp in zip(lbls, grps):
            if grp not in _WANTED_GROUPS or lbl in seen:
                continue
            seen.add(lbl)
            sc, unit = _SCALE[grp]
            param_rows.append((lbl, grp, sc, unit))

    if not param_rows:
        print("  SKIP pole comparison table: no pole/position parameters found.")
        return

    # Build initial-value lookup (first-seen per label across all pole_sims).
    init_vals = {}   # lbl → raw value
    for sn in valid_pole_sims:
        sd = sims.get(sn, {})
        if 'parameter_history' not in sd or 'est_parameters' not in sd:
            continue
        ph    = sd['parameter_history']
        lbls, _, _ = get_parameter_info(sd['est_parameters'])
        for i, lbl in enumerate(lbls):
            if lbl not in init_vals:
                init_vals[lbl] = ph[i, 0]

    # Per-sim: final parameter values.
    final_vals = {}   # sn → {lbl: raw_final}
    for sn in valid_pole_sims:
        sd = sims.get(sn, {})
        fv = {}
        if 'parameter_history' in sd and 'est_parameters' in sd:
            ph    = sd['parameter_history']
            lbls, _, _ = get_parameter_info(sd['est_parameters'])
            for i, lbl in enumerate(lbls):
                fv[lbl] = ph[i, -1]
        final_vals[sn] = fv

    # ── build LaTeX ───────────────────────────────────────────────────────────
    n_est_cols = len(valid_pole_sims)
    # Each estimation sim contributes 3 columns: Est. | Δ | %
    n_total_cols = 3 + n_est_cols * 3   # Parameter + Unit + Initial + [Est|Δ|%] × n

    col_spec = 'l l r ' + ' '.join(['r r r'] * n_est_cols)

    lines = []
    lines.append(r'% ============================================================')
    lines.append(r'% Pole parameter comparison table — SimObs analysis')
    lines.append(r'% Preamble: \usepackage{booktabs,siunitx,graphicx}')
    lines.append(r'% ============================================================')
    lines.append('')
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'  \centering')
    lines.append(
        r'  \caption{Estimated pole and position parameters for selected SimObs variants. '
        r'$\Delta = \text{Est.} - \text{Initial}$; '
        r'$\Delta[\%] = \Delta / |\text{Initial}| \times 100$.}'
    )
    lines.append(r'  \label{tab:pole-comparison-simobs}')
    lines.append(r'  \resizebox{\textwidth}{!}{%')
    lines.append(f'  \\begin{{tabular}}{{{col_spec}}}')
    lines.append(r'    \toprule')

    # Header row 1: spanning column labels per sim
    header1_parts = [r'    \multicolumn{3}{l}{}']
    for col_lbl in col_labels:
        n_span = 3
        header1_parts.append(
            rf'& \multicolumn{{{n_span}}}{{c}}{{\texttt{{{_esc(col_lbl)}}}}}'
        )
    lines.append(' '.join(header1_parts) + r' \\')

    # Cmidrule underlines for each group of 3 cols
    crule_parts = []
    for k in range(n_est_cols):
        start = 4 + k * 3   # 1-based: cols 1-3 are Param, Unit, Initial
        end   = start + 2
        crule_parts.append(rf'\cmidrule(lr){{{start}-{end}}}')
    lines.append('    ' + ' '.join(crule_parts))

    # Header row 2: column names
    header2 = r'    Parameter & Unit & Initial'
    for _ in col_labels:
        header2 += r' & Est. & $\Delta$ & $\Delta\,[\%]$'
    header2 += r' \\'
    lines.append(header2)
    lines.append(r'    \midrule')

    prev_grp = None
    for lbl, grp, sc, unit in param_rows:
        if prev_grp is not None and grp != prev_grp:
            lines.append(r'    \midrule')
        prev_grp = grp

        lbl_tex  = _p2tex(lbl)
        init_raw = init_vals.get(lbl)
        init_str = _num(init_raw * sc) if init_raw is not None else '---'

        row = f'    {lbl_tex} & {unit} & {init_str}'
        for sn in valid_pole_sims:
            fin_raw = final_vals[sn].get(lbl)
            if fin_raw is None or init_raw is None:
                row += r' & --- & --- & ---'
            else:
                fin_s  = fin_raw  * sc
                init_s = init_raw * sc
                diff   = fin_s - init_s
                pct    = diff / abs(init_s) * 100 if init_s != 0 else 0.0
                row   += f' & {_num(fin_s)} & {_num(diff)} & {pct:.2f}'
        row += r' \\'
        lines.append(row)

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'  }% end resizebox')
    lines.append(r'\end{table}')
    lines.append('')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved: {out_path}")


# ============================================================================
# KEY RESULTS TABLE
# ============================================================================

def generate_key_results_table(sims, names, labels, cfg_dict, out_path):
    """Write a LaTeX table comparing key parameter estimates vs IAU and FitPole references.

    Rows  : Position (X, Y, Z) [km], Velocity (VX, VY, VZ) [km/s],
            Pole Librations (α₁, δ₁) [deg] — only groups present in the sims.
    Cols  : Parameter | Unit | IAU₀ | FP₀ | {sim}: Δ_IAU | Δ_FP  × n_sims

    IAU₀ is taken from the first IAUPole sim's initial parameter values.
    FP₀  is taken from the first SimPole sim's initial parameter values.

    Parameters
    ----------
    cfg_dict : dict
        Must contain ``key_sims`` — list of sim names (order preserved in columns).
        Optional: ``caption``, ``label``.
    """
    key_sims   = cfg_dict.get('key_sims', [])
    iau_prefix = 'IAUPole'
    fp_prefix  = 'SimPole'

    _P2TEX = {
        'α₀': r'$\alpha_0$', 'δ₀': r'$\delta_0$',
        'α₁': r'$\alpha_1$', 'δ₁': r'$\delta_1$',
        'α̇₀': r'$\dot{\alpha}_0$', 'δ̇₀': r'$\dot{\delta}_0$',
        'X':  r'$X$',  'Y':  r'$Y$',  'Z':  r'$Z$',
        'VX': r'$V_X$', 'VY': r'$V_Y$', 'VZ': r'$V_Z$',
    }
    def _p2tex(lbl):
        return _P2TEX.get(lbl, lbl)

    def _num(v):
        return rf'\num{{{v:.5e}}}'

    def _esc(s):
        for ch, rep in [('_', r'\_'), ('&', r'\&'), ('%', r'\%')]:
            s = s.replace(ch, rep)
        return s

    _WANTED_GROUPS = {'Position', 'Velocity', 'Pole Librations'}
    _SCALE = {
        'Position':        (1e-3,        'km'),
        'Velocity':        (1e-3,        'km/s'),
        'Pole Librations': (_RAD_TO_DEG, 'deg'),
    }

    valid_key_sims = [sn for sn in key_sims if sn in sims and sn in names]
    if not valid_key_sims:
        print("  SKIP key results table: none of key_sims found in dataset.")
        return

    col_labels = [labels[names.index(sn)] for sn in valid_key_sims]

    # Build canonical row list (union of parameters across all key sims).
    seen       = set()
    param_rows = []
    for sn in valid_key_sims:
        sd = sims.get(sn, {})
        if 'parameter_history' not in sd or 'est_parameters' not in sd:
            continue
        lbls, grps, _ = get_parameter_info(sd['est_parameters'])
        for lbl, grp in zip(lbls, grps):
            if grp not in _WANTED_GROUPS or lbl in seen:
                continue
            seen.add(lbl)
            sc, unit = _SCALE[grp]
            param_rows.append((lbl, grp, sc, unit))

    if not param_rows:
        print("  SKIP key results table: no relevant parameters found.")
        return

    # IAU initial reference: first IAUPole sim that has each label.
    iau_init = {}
    for sn in names:
        if not sn.startswith(iau_prefix):
            continue
        sd = sims.get(sn, {})
        if 'parameter_history' not in sd or 'est_parameters' not in sd:
            continue
        ph   = sd['parameter_history']
        lbls, _, _ = get_parameter_info(sd['est_parameters'])
        for i, lbl in enumerate(lbls):
            if lbl not in iau_init:
                iau_init[lbl] = ph[i, 0]

    # FitPole initial reference: first SimPole sim that has each label.
    fp_init = {}
    for sn in names:
        if not sn.startswith(fp_prefix):
            continue
        sd = sims.get(sn, {})
        if 'parameter_history' not in sd or 'est_parameters' not in sd:
            continue
        ph   = sd['parameter_history']
        lbls, _, _ = get_parameter_info(sd['est_parameters'])
        for i, lbl in enumerate(lbls):
            if lbl not in fp_init:
                fp_init[lbl] = ph[i, 0]

    # Per-sim final values.
    final_vals = {}
    for sn in valid_key_sims:
        sd = sims.get(sn, {})
        fv = {}
        if 'parameter_history' in sd and 'est_parameters' in sd:
            ph   = sd['parameter_history']
            lbls, _, _ = get_parameter_info(sd['est_parameters'])
            for i, lbl in enumerate(lbls):
                fv[lbl] = ph[i, -1]
        final_vals[sn] = fv

    # ── Build LaTeX ───────────────────────────────────────────────────────────
    n_sims    = len(valid_key_sims)
    # Columns: Parameter | Unit | IAU₀ | FP₀ | [Δ_IAU Δ_FP] × n_sims
    col_spec  = 'l l r r ' + ' '.join(['r r'] * n_sims)

    caption = cfg_dict.get(
        'caption',
        r'Key parameter estimates. '
        r'$\Delta_\mathrm{IAU} = \mathrm{Est.} - \mathrm{IAU}_0$; '
        r'$\Delta_\mathrm{FP}  = \mathrm{Est.} - \mathrm{FP}_0$. '
        r'Position in km, velocity in km\,s$^{-1}$, angles in degrees.'
    )
    label = cfg_dict.get('label', 'tab:key-results')

    lines = []
    lines.append(r'% ============================================================')
    lines.append(r'% Key results table — parameter differences vs IAU and FitPole references')
    lines.append(r'% Preamble: \usepackage{booktabs,siunitx,graphicx}')
    lines.append(r'% ============================================================')
    lines.append('')
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'  \centering')
    lines.append(f'  \\caption{{{caption}}}')
    lines.append(f'  \\label{{{label}}}')
    lines.append(r'  \resizebox{\textwidth}{!}{%')
    lines.append(f'  \\begin{{tabular}}{{{col_spec}}}')
    lines.append(r'    \toprule')

    # Header row 1: spanning column labels per estimation sim.
    header1_parts = [r'    \multicolumn{4}{l}{}']
    for col_lbl in col_labels:
        header1_parts.append(
            rf'& \multicolumn{{2}}{{c}}{{\texttt{{{_esc(col_lbl)}}}}}'
        )
    lines.append(' '.join(header1_parts) + r' \\')

    # Cmidrule underlines for each group of 2 columns.
    crule_parts = []
    for k in range(n_sims):
        start = 5 + k * 2   # 1-based: cols 1–4 are Param, Unit, IAU₀, FP₀
        end   = start + 1
        crule_parts.append(rf'\cmidrule(lr){{{start}-{end}}}')
    lines.append('    ' + ' '.join(crule_parts))

    # Header row 2.
    header2 = r'    Parameter & Unit & IAU$_0$ & FP$_0$'
    for _ in col_labels:
        header2 += r' & $\Delta_\mathrm{IAU}$ & $\Delta_\mathrm{FP}$'
    header2 += r' \\'
    lines.append(header2)
    lines.append(r'    \midrule')

    prev_grp = None
    for lbl, grp, sc, unit in param_rows:
        if prev_grp is not None and grp != prev_grp:
            lines.append(r'    \midrule')
        prev_grp = grp

        lbl_tex = _p2tex(lbl)
        iau_raw = iau_init.get(lbl)
        fp_raw  = fp_init.get(lbl)
        iau_str = _num(iau_raw * sc) if iau_raw is not None else '---'
        fp_str  = _num(fp_raw  * sc) if fp_raw  is not None else '---'

        row = f'    {lbl_tex} & {unit} & {iau_str} & {fp_str}'
        for sn in valid_key_sims:
            fin_raw = final_vals[sn].get(lbl)
            if fin_raw is None:
                row += r' & --- & ---'
            else:
                fin_s     = fin_raw * sc
                d_iau     = (fin_s - iau_raw * sc) if iau_raw is not None else None
                d_fp      = (fin_s - fp_raw  * sc) if fp_raw  is not None else None
                d_iau_str = _num(d_iau) if d_iau is not None else '---'
                d_fp_str  = _num(d_fp)  if d_fp  is not None else '---'
                row += f' & {d_iau_str} & {d_fp_str}'
        row += r' \\'
        lines.append(row)

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'  }% end resizebox')
    lines.append(r'\end{table}')
    lines.append('')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved: {out_path}")


# ============================================================================
# OBSERVATIONAL DATASET TABLE
# ============================================================================

def generate_obs_dataset_table(sims: dict,
                                names: list,
                                obs_folder: str,
                                raw_obs_folder: str = '',
                                obs_types_override: dict = None,
                                sim_for_initial: str = None,   # kept for API compat, unused
                                file_names_json: str = 'file_names.json',
                                caption: str = 'Observational dataset summary.',
                                label: str = 'tab:obs-dataset-summary',
                                out_path: str = 'table_obs_dataset.tex'):
    """Write a LaTeX table summarising the observational dataset per ID.

    Only observation sets listed in ``file_names_json`` are included.

    Columns
    -------
    Observatory              : observatory name from MPC, e.g. "U.S. Naval Observatory".
    NSDC Listing             : NSDC dataset identifier, e.g. "nm0077".
    MPC Code                 : 3-digit MPC observatory code.
    $N_\\mathrm{obs}$        : total number of astrometric observations.
    Obs. Type                : 'Rel.' (relative) or 'Abs.' (absolute).
    RMS$_\\mathrm{RA}$       : RMS of O-C RA  residual vs NEP097 ephemeris [arcsec].
    RMS$_\\mathrm{Dec}$      : RMS of O-C Dec residual vs NEP097 ephemeris [arcsec].

    Rows are sorted by MPC observatory code (numeric), then by NSDC listing ID.

    Requires LaTeX preamble: \\usepackage{booktabs,siunitx}
    """

    def _esc(s: str) -> str:
        for ch, rep in [('_', r'\_'), ('&', r'\&'), ('%', r'\%'), ('#', r'\#'),
                        ('$', r'\$')]:
            s = s.replace(ch, rep)
        return s

    # ── Derive valid IDs from file_names.json ─────────────────────────────────
    # Filename format: Triton_<mpc>_<nsdc>.csv  →  ref_point_id = "<mpc>_<nsdc>"
    allowed_ids: set[str] | None = None
    if file_names_json:
        _p = Path(file_names_json)
        if _p.exists():
            import json as _json
            _fnames = _json.loads(_p.read_text())
            allowed_ids = set()
            for fn in _fnames:
                stem = fn.replace('.csv', '')
                parts = stem.split('_')   # ['Triton', '<mpc>', '<nsdc>']
                if len(parts) >= 3:
                    allowed_ids.add(f'{parts[1]}_{parts[2]}')
        else:
            print(f'  WARN: file_names.json not found at {file_names_json!r} — '
                  'including all IDs from obs folder')

    # ── Load SPICE residuals ──────────────────────────────────────────────────
    spice_df = _load_spice_residuals_df(obs_folder)

    # ── Build per-ID rows ─────────────────────────────────────────────────────
    all_ids = sorted(spice_df['ref_point_id'].unique()) if not spice_df.empty else []

    # Filter to file_names.json entries only
    if allowed_ids is not None:
        all_ids = [rid for rid in all_ids if rid in allowed_ids]

    # Sort: numeric MPC code first, then listing ID.
    def _sort_key(rid):
        parts = rid.split('_')
        try:
            code = int(parts[0])
        except ValueError:
            code = 9999
        nsdc = parts[1] if len(parts) > 1 else ''
        return (code, nsdc)

    all_ids.sort(key=_sort_key)

    table_rows = []
    for rid in all_ids:
        parts    = rid.split('_')
        mpc_code = parts[0]
        nsdc_id  = parts[1] if len(parts) > 1 else ''

        obs_name = _get_observatory_name(mpc_code)
        obs_type = _get_obs_type(nsdc_id, raw_obs_folder, obs_types_override)

        rms_ra_spice = rms_dec_spice = float('nan')
        n_obs = 0
        if not spice_df.empty and rid in spice_df['ref_point_id'].values:
            sub = spice_df.loc[spice_df['ref_point_id'] == rid]
            n_obs         = len(sub)
            rms_ra_spice  = float(np.sqrt(np.nanmean(sub['ra_resid_arcsec'].values  ** 2)))
            rms_dec_spice = float(np.sqrt(np.nanmean(sub['dec_resid_arcsec'].values ** 2)))

        table_rows.append((obs_name, mpc_code, nsdc_id, n_obs, obs_type,
                           rms_ra_spice, rms_dec_spice))

    # ── Build LaTeX ───────────────────────────────────────────────────────────
    def _fmt(v):
        return f'{v:.3f}' if np.isfinite(v) else r'\multicolumn{1}{c}{---}'

    safe_label = label.replace('_', '-')
    lines = []
    lines.append(r'% ============================================================')
    lines.append(r'% Observational dataset summary table')
    lines.append(r'% Preamble: \usepackage{booktabs,siunitx}')
    lines.append(r'% ============================================================')
    lines.append('')
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'  \centering')
    lines.append(rf'  \caption{{{_esc(caption)}}}')
    lines.append(rf'  \label{{{safe_label}}}')
    lines.append(r'  \resizebox{\textwidth}{!}{%')
    lines.append(r'  \begin{tabular}{l l c c c r r}')
    lines.append(r'    \toprule')
    lines.append(
        r'    Observatory & NSDC Listing & MPC Code'
        r' & $N_{\mathrm{obs}}$ & Obs.\ Type'
        r' & \multicolumn{2}{c}{RMS O$-$C vs NEP097 [$^{\prime\prime}$]} \\'
    )
    lines.append(
        r'    & & & & & RA & Dec \\'
    )
    lines.append(r'    \midrule')

    prev_code = None
    for (obs_name, mpc_code, nsdc_id, n_obs, obs_type,
         rms_ra_sp, rms_dec_sp) in table_rows:
        if prev_code is not None and mpc_code != prev_code:
            lines.append(r'    \midrule')
        prev_code = mpc_code

        lines.append(
            f'    {_esc(obs_name)} & {nsdc_id} & {mpc_code}'
            f' & {n_obs} & {obs_type}'
            f' & {_fmt(rms_ra_sp)} & {_fmt(rms_dec_sp)} \\\\'
        )

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'  }% end resizebox')
    lines.append(r'\end{table}')
    lines.append('')

    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'  Saved: {out_path}  ({len(table_rows)} rows)')


# ============================================================================
# PARAMETER FORMAL ERRORS TABLE
# ============================================================================

def generate_param_formal_table(sims, sim_name, names, out_path):
    """Write a LaTeX table of estimated parameters with formal errors.

    How formal errors are obtained
    --------------------------------
    Formal errors are computed as ``np.sqrt(np.diag(sd['covariance']))``, where
    ``sd['covariance']`` is the full post-fit covariance matrix stored in the
    pickle (shape *n_params × n_params*).  The diagonal gives the variance per
    parameter; taking the square root yields the 1σ uncertainty in native units
    (metres for position, m/s for velocity, radians for angles).
    Indices align exactly with the labels returned by
    ``get_parameter_info(sd['est_parameters'])``.

    Table columns
    -------------
    Parameter | Group | Unit | IAU₀ | FP₀ | Final | ±1σ | Δ_IAU | Δ_FP

    Only the *estimated* parameters are listed (those in ``est_parameters``).
    IAU₀ comes from ``IAUPole_pole_pos_cov_pole_lib_cov`` initial iteration.
    FP₀  comes from ``SimPole_pole_pos_cov_pole_lib_cov`` initial iteration.
    """

    def _esc(s):
        return s.replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')

    def _p2tex(lbl):
        _MAP = {
            'α₀': r'$\alpha_0$',       'δ₀': r'$\delta_0$',
            'α₁': r'$\alpha_1$',       'δ₁': r'$\delta_1$',
            'α̇₀': r'$\dot{\alpha}_0$', 'δ̇₀': r'$\dot{\delta}_0$',
        }
        return _MAP.get(lbl, _esc(lbl))

    def _num(v, decimals=5):
        return rf'\num{{{v:.{decimals}e}}}'

    sd = sims.get(sim_name, {})
    if 'parameter_history' not in sd or 'est_parameters' not in sd:
        print(f"  SKIP param_formal_table for '{sim_name}': no parameter data.")
        return
    if 'covariance' not in sd:
        print(f"  SKIP param_formal_table for '{sim_name}': 'covariance' key missing.")
        return

    ph       = sd['parameter_history']                  # (n_params, n_iters)
    fe_raw   = np.sqrt(np.diag(sd['covariance']))       # (n_params,) 1σ in native units
    lbls, grps, _ = get_parameter_info(sd['est_parameters'])

    # ── IAU reference initial values (from IAUPole_pole_pos_cov_pole_lib_cov) ─
    iau_ref, _ = _get_iau_reference(sims, names)

    # ── FitPole reference initial values (from SimPole_pole_pos_cov_pole_lib_cov) ─
    fp_ref = {}
    REF_FP = 'SimPole_pole_pos_cov_pole_lib_cov'
    sd_fp  = sims.get(REF_FP, {})
    if 'parameter_history' in sd_fp and 'est_parameters' in sd_fp:
        ph_fp    = sd_fp['parameter_history']
        lbls_fp, _, _ = get_parameter_info(sd_fp['est_parameters'])
        for i, lbl in enumerate(lbls_fp):
            fp_ref[lbl] = ph_fp[i, 0]

    sim_safe = _esc(sim_name.replace('_cov', ''))
    lines = []
    lines.append(r'% ============================================================')
    lines.append(rf'% Estimated-parameter formal-error table: {sim_name}')
    lines.append(r'% Formal errors = sqrt(diag(sd[covariance])) — 1sigma in native units')
    lines.append(r'% Preamble: \usepackage{booktabs,siunitx,graphicx}')
    lines.append(r'% ============================================================')
    lines.append('')
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'  \centering')
    lines.append(
        rf'  \caption{{Estimated parameters for simulation \texttt{{{sim_safe}}}. '
        r'IAU$_0$ and FP$_0$ are the initial values for the IAU\,2015 and '
        r'NEP097-fitted pole models, respectively. '
        r'$\pm1\sigma$ is the formal error from the post-fit covariance matrix '
        r'(square root of the diagonal element). '
        r'$\Delta_\mathrm{IAU} = \mathrm{Final} - \mathrm{IAU}_0$; '
        r'$\Delta_\mathrm{FP}  = \mathrm{Final} - \mathrm{FP}_0$.}}'
    )
    lines.append(rf'  \label{{tab:param-formal-{_esc(sim_name)}}}')
    lines.append(r'  \resizebox{\textwidth}{!}{%')
    lines.append(r'  \begin{tabular}{llcrrrrrrr}')
    lines.append(r'    \toprule')
    lines.append(
        r'    Parameter & Group & Unit'
        r' & IAU$_0$ & FP$_0$ & Final & $\pm1\sigma$'
        r' & $\Delta_\mathrm{IAU}$ & $\Delta_\mathrm{FP}$ \\'
    )
    lines.append(r'    \midrule')

    prev_grp = None
    for i, (lbl, grp) in enumerate(zip(lbls, grps)):
        if grp not in _TARGET_GROUPS:
            continue
        if prev_grp is not None and grp != prev_grp:
            lines.append(r'    \midrule')
        prev_grp = grp

        sc, unit = _GROUP_SCALE.get(grp, (1.0, '---'))
        lbl_tex  = _p2tex(lbl)

        final_s  = ph[i, -1] * sc
        sigma_s  = fe_raw[i]  * sc   # formal error in display units
        iau_raw  = iau_ref.get(lbl)
        fp_raw   = fp_ref.get(lbl)

        iau_str  = _num(iau_raw * sc) if iau_raw is not None else '---'
        fp_str   = _num(fp_raw  * sc) if fp_raw  is not None else '---'

        d_iau_str = _num((ph[i, -1] - iau_raw) * sc) if iau_raw is not None else '---'
        d_fp_str  = _num((ph[i, -1] - fp_raw)  * sc) if fp_raw  is not None else '---'

        lines.append(
            f'    {lbl_tex} & {_esc(grp)} & {unit}'
            f' & {iau_str} & {fp_str}'
            f' & {_num(final_s)} & {_num(sigma_s)}'
            rf' & {d_iau_str} & {d_fp_str} \\'
        )

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}}%')
    lines.append(r'\end{table}')
    lines.append('')

    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'  Saved: {out_path}')


# ============================================================================
# OBSERVATION RESIDUALS SUMMARY TABLE
# ============================================================================

def generate_obs_residuals_table(sims, sim_name, out_path,
                                 raw_obs_folder=None,
                                 obs_types_override=None,
                                 caption=None,
                                 label='tab:obs-residuals'):
    """Write a LaTeX table of per-dataset RMS O-C residuals from the final estimation.

    Columns: Observatory | NSDC Listing | MPC Code | N_obs | Obs. Type
             | RMS O-C NEP097 RA [arcsec] | RMS O-C NEP097 Dec [arcsec]
             | RMS O-C Final RA [arcsec]  | RMS O-C Final Dec [arcsec]

    NEP097 residuals come from ``residual_df['ra_residual_initial_mas']`` and
    ``residual_df['dec_residual_initial_mas']`` (first iteration, before fitting).
    Final residuals come from ``residual_df['ra_residual_final_mas']`` and
    ``residual_df['dec_residual_final_mas']``.  Both converted to arcsec.
    The ``ref_point_id`` format is ``'{mpc_code}_{nsdc_id}'``, e.g. ``'689_nm0007'``.
    Observatory names are looked up from ``Observations/Observatories.txt`` via
    ``_get_observatory_name()``.  Rows are sorted by MPC code then NSDC listing.
    """

    def _esc(s):
        return str(s).replace('_', r'\_').replace('&', r'\&').replace('%', r'\%')

    sd = sims.get(sim_name, {})
    if 'residual_df' not in sd:
        print(f"  SKIP obs_residuals_table for '{sim_name}': no residual_df.")
        return

    df = sd['residual_df'].copy()
    # Convert mas → arcsec (final)
    df['ra_as']       = df['ra_residual_final_mas']   / 1000.0
    df['dec_as']      = df['dec_residual_final_mas']  / 1000.0
    # Convert mas → arcsec (initial = NEP097 baseline)
    df['ra_as_init']  = df['ra_residual_initial_mas'] / 1000.0
    df['dec_as_init'] = df['dec_residual_initial_mas'] / 1000.0

    # Aggregate per ref_point_id
    rows = []
    for rid, grp in df.groupby('ref_point_id'):
        parts    = str(rid).split('_', 1)
        mpc_code = parts[0] if len(parts) >= 1 else rid
        nsdc_id  = parts[1] if len(parts) >= 2 else '---'

        obs_name     = _get_observatory_name(mpc_code)
        obs_type     = _get_obs_type(nsdc_id,
                                     raw_obs_folder=raw_obs_folder,
                                     obs_types_override=obs_types_override or {})
        n_obs        = len(grp)
        rms_ra_init  = float(np.sqrt(np.mean(grp['ra_as_init']  ** 2)))
        rms_dec_init = float(np.sqrt(np.mean(grp['dec_as_init'] ** 2)))
        rms_ra       = float(np.sqrt(np.mean(grp['ra_as']  ** 2)))
        rms_dec      = float(np.sqrt(np.mean(grp['dec_as'] ** 2)))
        rows.append((mpc_code, nsdc_id, obs_name, obs_type, n_obs,
                     rms_ra_init, rms_dec_init, rms_ra, rms_dec))

    # Sort by MPC code (numeric if possible), then NSDC id
    def _sort_key(r):
        try:
            return (int(r[0]), r[1])
        except ValueError:
            return (0, r[0] + r[1])
    rows.sort(key=_sort_key)

    if caption is None:
        sim_safe = sim_name.replace('_cov', '').replace('_', r'\_')
        caption = (
            rf'RMS observed minus computed (O$-$C) residuals per observation dataset. '
            rf'NEP097 column uses pre-estimation residuals (initial iteration); '
            rf'Final column uses post-estimation residuals (\texttt{{{sim_safe}}}). '
            r'MPC Code is the three-digit Minor Planet Center observatory code. '
            r'$N_\mathrm{obs}$ is the number of astrometric observations in the dataset. '
            r"Obs.\ Type indicates relative (Rel.) or absolute (Abs.) astrometry. "
            r"All RMS values in arcseconds $[^{\prime\prime}]$."
        )

    lines = []
    lines.append(r'% ============================================================')
    lines.append(rf'% Observation residuals table: {sim_name}')
    lines.append(r'% Preamble: \usepackage{booktabs}')
    lines.append(r'% ============================================================')
    lines.append('')
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'  \centering')
    lines.append(rf'  \caption{{{caption}}}')
    lines.append(rf'  \label{{{label}}}')
    lines.append(r'  \begin{tabular}{llrrlrrrr}')
    lines.append(r'    \toprule')
    lines.append(
        r'    Observatory & NSDC Listing & MPC Code'
        r' & $N_\mathrm{obs}$ & Obs.\ Type'
        r' & \multicolumn{2}{c}{RMS O$-$C NEP097 [$^{\prime\prime}$]}'
        r' & \multicolumn{2}{c}{RMS O$-$C Final [$^{\prime\prime}$]} \\'
    )
    lines.append(r'    \cmidrule(lr){6-7} \cmidrule(lr){8-9}')
    lines.append(r'    & & & & & RA & Dec & RA & Dec \\')
    lines.append(r'    \midrule')

    prev_mpc = None
    for mpc_code, nsdc_id, obs_name, obs_type, n_obs, \
            rms_ra_init, rms_dec_init, rms_ra, rms_dec in rows:
        if prev_mpc is not None and mpc_code != prev_mpc:
            lines.append(r'    \addlinespace')
        prev_mpc = mpc_code
        lines.append(
            f'    {_esc(obs_name)} & {_esc(nsdc_id)} & {mpc_code}'
            f' & {n_obs} & {obs_type}'
            rf' & {rms_ra_init:.3f} & {rms_dec_init:.3f}'
            rf' & {rms_ra:.3f} & {rms_dec:.3f} \\'
        )

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table}')
    lines.append('')

    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'  Saved: {out_path}  ({len(rows)} datasets)')


# ============================================================================
# MAIN
# ============================================================================

def main():
    sims, all_names = _get_data()
    names, labels   = _get_sims_and_labels(sims, all_names)

    # When EXPORT_DATASETS / EXPORT_FIGURES_FILTER is in effect, the active
    # config's primary dataset may not be loaded — that's fine for figures
    # that pull their own data via `from_config`.  Only abort if the user
    # ran with no filters at all (legacy behaviour).
    _filtered_run = bool(os.environ.get('EXPORT_DATASETS') or
                          os.environ.get('EXPORT_FIGURES_FILTER') or
                          os.environ.get('EXPORT_SIMS_FILTER'))
    if not names:
        if not _filtered_run:
            sys.exit("ERROR: No simulations found.  Check SELECTED_SIMS and DATA_FILES.")
        print("  NOTE: primary dataset has no sims under the active filter — "
              "from_config-based figures will still render.")

    # Create timestamped output directory.
    timestamp  = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir    = os.path.join(OUTPUT_DIR, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Exporting {len(FIGURES_TO_EXPORT)} figure(s) | "
          f"{len(names)} simulation(s):")
    for n, l in zip(names, labels):
        print(f"  {n!r:50s}  →  {l!r}")
    print(f"Output directory: {out_dir}\n")

    # Optional substring filter on subfolder/fig_label.  Set
    # EXPORT_FIGURES_FILTER=11_pole_model_compare to render just one block.
    _figs_filter_env = os.environ.get('EXPORT_FIGURES_FILTER')
    _figs_filter = ([x.strip() for x in _figs_filter_env.split(',') if x.strip()]
                    if _figs_filter_env else None)
    if _figs_filter:
        print(f"[EXPORT_FIGURES_FILTER={_figs_filter_env}] only rendering "
              f"figures whose subfolder/fig_label contains any of these.")

    for spec in FIGURES_TO_EXPORT:
        fn_name = spec[0]
        kwargs  = dict(spec[1]) if len(spec) > 1 else {}

        # Filter by EXPORT_GROUPS if set.
        if _EXPORT_GROUPS is not None:
            fig_group = kwargs.get('group')
            if fig_group not in _EXPORT_GROUPS:
                continue

        # Filter by EXPORT_FIGURES_FILTER if set.
        if _figs_filter:
            _hay = ((kwargs.get('subfolder') or '') + ' ' +
                    (kwargs.get('fig_label') or '') + ' ' + fn_name)
            if not any(tok in _hay for tok in _figs_filter):
                continue

        if SKIP_TIMESERIES and fn_name in _SLOW_FIGURE_TYPES:
            print(f"  SKIP (SKIP_TIMESERIES=True): {fn_name}")
            continue

        fn = _PLOT_REGISTRY.get(fn_name)
        if fn is None:
            print(f"  SKIP: unknown function '{fn_name}'")
            continue

        # fig_label, subfolder, group are routing/naming keys — pop before passing to fn.
        fig_label = kwargs.pop('fig_label', None)
        subfolder = kwargs.pop('subfolder', None)
        kwargs.pop('group', None)  # consumed for filtering; not passed to plot functions
        if _SUPPRESS_TITLES:
            kwargs['title'] = ''
        suffix    = (f'_{fig_label}' if fig_label
                     else (f'_{kwargs.get("variant", "")}' if kwargs.get('variant') else ''))

        # from_config: load sims/names/labels from a different ExportConfig for this figure.
        from_config = kwargs.pop('from_config', None)
        if from_config is not None:
            _fc_sims, _fc_names, _fc_labels, _, _, _ = _load_dataset_for_config(from_config)
        else:
            _fc_sims, _fc_names, _fc_labels = sims, names, labels

        # Resolve output directory (create subfolder on demand).
        if subfolder:
            fig_out_dir = os.path.join(out_dir, subfolder)
            os.makedirs(fig_out_dir, exist_ok=True)
        else:
            fig_out_dir = out_dir

        # condition_numbers_txt writes a text file and needs no PDF saving.
        if fn_name == 'condition_numbers_txt':
            txt_path = os.path.join(fig_out_dir, 'condition_numbers.txt')
            fn(_fc_sims, _fc_names, txt_path)
            continue

        # corr_heatmap operates on a single sim, not a list.
        if fn_name == 'corr_heatmap':
            sim_name = kwargs.pop('sim_name', None)
            if sim_name is None:
                sim_name = next(
                    (n for n in _fc_names if 'correlations' in _fc_sims.get(n, {})),
                    None)
            if sim_name is None:
                print("  SKIP: no simulation with correlation data.")
                continue
            fig = fn(_fc_sims, sim_name, **kwargs)
        else:
            fig = fn(_fc_sims, _fc_names, _fc_labels, **kwargs)

        if fig is None:
            print(f"  SKIP: {fn_name} returned None (data missing?)")
            continue

        # Multi-dataset functions return list of (fig, label) — save each separately.
        if isinstance(fig, list):
            for sub_fig, sub_label in fig:
                safe_label = sub_label.replace(' ', '_').replace('/', '-')
                out_path = os.path.join(fig_out_dir,
                                        f'{fn_name}{suffix}_{safe_label}.pdf')
                with PdfPages(out_path) as pdf:
                    pdf.savefig(sub_fig, bbox_inches='tight')
                plt.close(sub_fig)
                print(f"  Saved: {out_path}")
            continue

        out_path = os.path.join(fig_out_dir, f'{fn_name}{suffix}.pdf')
        with PdfPages(out_path) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path}")

    # Write parameter comparison tables (LaTeX).
    table_path = os.path.join(out_dir, 'parameter_tables.tex')
    generate_parameter_tables(sims, names, labels, table_path)

    # Write single-simulation detail tables (LaTeX).
    for sim_name in SINGLE_SIM_TABLES:
        safe  = sim_name.replace('_cov', '').replace('_', '-')
        tpath = os.path.join(out_dir, f'table_single_{safe}.tex')
        generate_single_sim_table(sims, sim_name, names, tpath)

    # Write naming-convention table if config provides SIM_DESCRIPTIONS.
    _sim_descriptions = getattr(_cfg, 'SIM_DESCRIPTIONS', {})
    if _sim_descriptions:
        naming_path = os.path.join(out_dir, 'table_naming_conventions.tex')
        generate_naming_table(
            selected_sims    = _cfg.SELECTED_SIMS or list(all_names),
            sim_labels       = _cfg.SIM_LABELS,
            sim_descriptions = _sim_descriptions,
            dataset_label    = DATASET_LABEL,
            out_path         = naming_path,
        )

    # Write weight-scheme overview table if config provides SIM_TABLE_ROWS.
    _table_rows = getattr(_cfg, 'SIM_TABLE_ROWS', [])
    if _table_rows:
        ws_path = os.path.join(out_dir, 'table_weight_schemes.tex')
        generate_weight_scheme_table(
            table_rows    = _table_rows,
            dataset_label = DATASET_LABEL,
            out_path      = ws_path,
        )

    # Write extra naming-convention tables (EXTRA_NAMING_TABLES — one per dataset).
    _extra_naming = getattr(_cfg, 'EXTRA_NAMING_TABLES', [])
    for _ent in _extra_naming:
        _en_subfolder = _ent.get('subfolder', None)
        if _en_subfolder:
            _en_dir = os.path.join(out_dir, _en_subfolder)
            os.makedirs(_en_dir, exist_ok=True)
        else:
            _en_dir = out_dir
        _en_filename = _ent.get('out_filename', 'table_naming_conventions_extra.tex')
        _en_path = os.path.join(_en_dir, _en_filename)
        generate_naming_table(
            selected_sims    = _ent['selected_sims'],
            sim_labels       = _ent.get('sim_labels'),
            sim_descriptions = _ent['sim_descriptions'],
            dataset_label    = _ent.get('dataset_label', DATASET_LABEL),
            out_path         = _en_path,
        )

    # Write dataset overview table (DATASET_OVERVIEW_TABLE).
    _dataset_overview = getattr(_cfg, 'DATASET_OVERVIEW_TABLE', None)
    if _dataset_overview:
        _do_subfolder = _dataset_overview.get('subfolder', None)
        if _do_subfolder:
            _do_dir = os.path.join(out_dir, _do_subfolder)
            os.makedirs(_do_dir, exist_ok=True)
        else:
            _do_dir = out_dir
        _do_path = os.path.join(_do_dir, 'table_dataset_overview.tex')
        generate_dataset_overview_table(
            rows     = _dataset_overview['rows'],
            caption  = _dataset_overview.get('caption', 'Dataset overview.'),
            label    = _dataset_overview.get('label', 'tab:dataset-overview'),
            out_path = _do_path,
        )

    # Write residual statistics table (RESIDUAL_STATS_TABLE).
    _res_stats_cfg = getattr(_cfg, 'RESIDUAL_STATS_TABLE', None)
    if _res_stats_cfg:
        _rst_subfolder = _res_stats_cfg.get('subfolder', None) if isinstance(_res_stats_cfg, dict) else None
        if _rst_subfolder:
            _rst_dir = os.path.join(out_dir, _rst_subfolder)
            os.makedirs(_rst_dir, exist_ok=True)
        else:
            _rst_dir = out_dir
        rst_path = os.path.join(_rst_dir, 'table_residual_stats.tex')
        generate_residual_stats_table(
            sims        = sims,
            names       = names,
            labels      = labels,
            out_path    = rst_path,
            data_source = _res_stats_cfg.get('data_source', 'final') if isinstance(_res_stats_cfg, dict) else 'final',
            caption     = _res_stats_cfg.get('caption', 'Mean and standard deviation of final residuals per weighting scheme [mas].') if isinstance(_res_stats_cfg, dict) else 'Mean and standard deviation of final residuals per weighting scheme [mas].',
            label       = _res_stats_cfg.get('label', 'tab:residual-stats') if isinstance(_res_stats_cfg, dict) else 'tab:residual-stats',
        )

    # Write final estimation comparison tables if config provides FINAL_ESTIMATION_TABLES.
    _final_est = getattr(_cfg, 'FINAL_ESTIMATION_TABLES', None)
    if _final_est:
        import pathlib
        _fe_dir  = pathlib.Path(out_dir) / 'FinalEstimation'
        _fe_dir.mkdir(parents=True, exist_ok=True)
        _fe_path = _fe_dir / 'final_estimation_params.tex'
        generate_final_estimation_tables(
            sims,
            names,
            str(_fe_path),
            iau_sim=_final_est.get('iau_sim',     'IAUPole_pole_lib_cov'),
            fitpole_sim=_final_est.get('fitpole_sim', 'SimPole_pole_lib_cov'),
        )

    # Write pole comparison table if config provides POLE_TABLE_SIMS.
    _pole_table_sims = getattr(_cfg, 'POLE_TABLE_SIMS', [])
    if _pole_table_sims:
        pole_tpath = os.path.join(out_dir, 'table_pole_comparison.tex')
        generate_pole_comparison_table(
            sims      = sims,
            names     = names,
            labels    = labels,
            pole_sims = _pole_table_sims,
            out_path  = pole_tpath,
        )

    # Write key results table if config provides KEY_RESULTS_CONFIG.
    _key_results_cfg = getattr(_cfg, 'KEY_RESULTS_CONFIG', None)
    if _key_results_cfg:
        import pathlib as _pathlib
        _kr_subfolder = _key_results_cfg.get('subfolder', '08_params/KeyResults')
        _kr_dir = _pathlib.Path(out_dir) / _kr_subfolder
        _kr_dir.mkdir(parents=True, exist_ok=True)
        _kr_path = _kr_dir / 'key_results_params.tex'
        generate_key_results_table(
            sims     = sims,
            names    = names,
            labels   = labels,
            cfg_dict = _key_results_cfg,
            out_path = str(_kr_path),
        )

    # Write observational dataset summary table if config provides OBS_DATASET_TABLE.
    _obs_table_cfg = getattr(_cfg, 'OBS_DATASET_TABLE', None)
    if _obs_table_cfg:
        obs_tpath = os.path.join(out_dir, 'table_obs_dataset.tex')
        generate_obs_dataset_table(
            sims               = sims,
            names              = names,
            obs_folder         = _obs_table_cfg.get('obs_folder', 'Observations/AllModernJ2000'),
            raw_obs_folder     = _obs_table_cfg.get('raw_obs_folder', ''),
            obs_types_override = _obs_table_cfg.get('obs_types', {}),
            file_names_json    = _obs_table_cfg.get('file_names_json', 'file_names.json'),
            caption            = _obs_table_cfg.get('caption', 'Observational dataset summary.'),
            label              = _obs_table_cfg.get('label', 'tab:obs-dataset-summary'),
            out_path           = obs_tpath,
        )

    # Write rejected-observations table if OBS_ANALYSIS_DATA is configured.
    _oa_data = getattr(_cfg, 'OBS_ANALYSIS_DATA', None)
    if _oa_data:
        rej_tpath = os.path.join(out_dir, 'table_rejected_obs.tex')
        generate_rejected_obs_table(data_path=_oa_data, out_path=rej_tpath)

    # Write initial parameter values table if config provides INITIAL_VALUES_TABLE.
    _init_vals_cfg = getattr(_cfg, 'INITIAL_VALUES_TABLE', None)
    if _init_vals_cfg:
        init_tpath = os.path.join(out_dir, 'table_initial_params.tex')
        generate_initial_values_table(
            sims     = sims,
            cfg_dict = _init_vals_cfg,
            out_path = init_tpath,
        )

    # Write GM parameter estimation table if config provides GM_PARAM_TABLE.
    _gm_table_cfg = getattr(_cfg, 'GM_PARAM_TABLE', None)
    if _gm_table_cfg:
        gm_tpath = os.path.join(out_dir, 'table_gm_params.tex')
        generate_gm_param_table(
            sims     = sims,
            cfg_dict = _gm_table_cfg,
            out_path = gm_tpath,
        )

    # Write parameter formal errors table if config provides PARAM_FORMAL_TABLE.
    _param_formal_cfg = getattr(_cfg, 'PARAM_FORMAL_TABLE', None)
    if _param_formal_cfg:
        import pathlib as _pfpathlib
        _pf_subfolder = _param_formal_cfg.get('subfolder', 'FinalEstimation')
        _pf_dir = _pfpathlib.Path(out_dir) / _pf_subfolder
        _pf_dir.mkdir(parents=True, exist_ok=True)
        _pf_sim = _param_formal_cfg.get('sim_name', 'SimPole_pole_lib_cov')
        _pf_path = _pf_dir / 'table_param_formal.tex'
        generate_param_formal_table(sims, _pf_sim, names, str(_pf_path))

    # Write obs residuals table if config provides OBS_RESIDUALS_TABLE.
    _obs_res_cfg = getattr(_cfg, 'OBS_RESIDUALS_TABLE', None)
    if _obs_res_cfg:
        import pathlib as _orpathlib
        _or_subfolder = _obs_res_cfg.get('subfolder', 'FinalEstimation')
        _or_dir = _orpathlib.Path(out_dir) / _or_subfolder
        _or_dir.mkdir(parents=True, exist_ok=True)
        _or_sim = _obs_res_cfg.get('sim_name', 'SimPole_pole_lib_cov')
        _or_path = _or_dir / 'table_obs_residuals.tex'
        generate_obs_residuals_table(
            sims               = sims,
            sim_name           = _or_sim,
            out_path           = str(_or_path),
            raw_obs_folder     = _obs_res_cfg.get('raw_obs_folder', None),
            obs_types_override = _obs_res_cfg.get('obs_types_override', None),
            caption            = _obs_res_cfg.get('caption', None),
            label              = _obs_res_cfg.get('label', 'tab:obs-residuals'),
        )

    # Write condition number table if config provides CONDITION_NUMBER_TABLE.
    _cond_cfg = getattr(_cfg, 'CONDITION_NUMBER_TABLE', None)
    if _cond_cfg:
        cond_tpath = os.path.join(out_dir, 'table_condition_numbers.tex')
        generate_condition_number_table(
            sims     = sims,
            names    = names,
            labels   = labels,
            cfg_dict = _cond_cfg,
            out_path = cond_tpath,
        )

    # Write SH summary table if config provides SH_SUMMARY_TABLE.
    _sh_summary_cfg = getattr(_cfg, 'SH_SUMMARY_TABLE', None)
    if _sh_summary_cfg:
        sh_summary_tpath = os.path.join(out_dir, 'table_sh_summary.tex')
        generate_sh_summary_table(
            sims     = sims,
            names    = names,
            labels   = labels,
            cfg_dict = _sh_summary_cfg,
            out_path = sh_summary_tpath,
        )

    # Write pole estimation results table if config provides POLE_ESTIMATION_TABLE.
    _pole_est_cfg = getattr(_cfg, 'POLE_ESTIMATION_TABLE', None)
    if _pole_est_cfg:
        pole_est_tpath = os.path.join(out_dir, 'table_pole_estimation.tex')
        generate_pole_estimation_table(
            sims     = sims,
            cfg_dict = _pole_est_cfg,
            out_path = pole_est_tpath,
        )

    print(f"\nDone.  Output → {out_dir}")


if __name__ == '__main__':
    main()
