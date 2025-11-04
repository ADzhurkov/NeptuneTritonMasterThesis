import numpy as np
from pathlib import Path
from typing import Dict
from datetime import datetime, timedelta
import sys
import json
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import pickle
import os 

from datetime import datetime, timedelta
import matplotlib.dates as mdates


from tudatpy.interface import spice

from tudatpy.astro import time_conversion, element_conversion,frame_conversion
from tudatpy.astro.time_conversion import DateTime


# Get the path to the directory containing this file
current_dir = Path(__file__).resolve().parent

# Append the HelperFunctions directory
sys.path.append(str(current_dir / "HelperFunctions"))

import ProcessingUtils
import PropFuncs
import FigUtils
import ObsFunc
import nsdc

J2000_EPOCH = datetime(2000, 1, 1, 12, 0, 0)

def make_timestamped_folder(base_path="Results"):
    folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_path = Path(base_path) / folder_name
    full_path.mkdir(parents=True, exist_ok=True)
    return full_path

def load_npy_files(folder_path: str) -> Dict[str, np.ndarray]:
    """
    Load all .npy files from a specified folder.
    
    Parameters:
    -----------
    folder_path : str
        Path to the folder containing .npy files
    
    Returns:
    --------
    dict
        Dictionary with filenames (without extension) as keys and numpy arrays as values
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder_path}")
    
    # Find all .npy files
    npy_files = folder.glob("*.npy")
    
    # Load each file into a dictionary
    data = {}
    for file_path in npy_files:
        try:
            array = np.load(file_path)
            # Use filename without extension as key
            data[file_path.stem] = array
            print(f"Loaded: {file_path.name} with shape {array.shape}")
        except Exception as e:
            print(f"Warning loading {file_path.name}: {e}")
            
            try:
                print("Trying with allow_pickle=True...")
                array = np.load(file_path,allow_pickle=True)
                data[file_path.stem] = array
                print(f"Loaded: {file_path.name} with shape {array.shape}")
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")
    if not data:
        print(f"No .npy files found in {folder_path}")
    
    return data

def create_observations_dataframe(observations, observation_set_ids, observation_times, 
                                 residual_history=None):
    """
    Create a DataFrame from observations with variable lengths.
    
    Parameters:
    -----------
    observations : list of lists (or numpy array with dtype=object)
        Each inner list contains flattened RA/DEC pairs [ra1, dec1, ra2, dec2, ...]
    observation_set_ids : list
        IDs corresponding to each inner list in observations
    observation_times : list or array
        Single flattened list of all observation times (one per RA/DEC pair)
    residual_history : numpy array, optional
        Shape (n_runs, n_obs, 3) where last dimension is [time, ra_residual, dec_residual]
        If provided, adds residuals from first and last runs to DataFrame
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns: 'id', 'ra', 'dec', 'times', and optionally residual columns
    """
    data = []
    time_idx = 0
    
    for i, obs_list in enumerate(observations):
        obs_id = observation_set_ids[i]
        
        # Reshape flattened RA/DEC into pairs
        # obs_list = [ra1, dec1, ra2, dec2, ...] -> [[ra1, dec1], [ra2, dec2], ...]
        obs_array = np.array(obs_list).reshape(-1, 2)
        
        for ra, dec in obs_array:
            row = {
                'id': obs_id,
                'observatory': obs_id.split("_")[0],
                'ra': ra,
                'dec': dec,
                'times': observation_times[time_idx]
            }
            
            # Add residuals if provided
            if residual_history is not None:
                # First run (index 0)
                row['residual_ra_first'] = residual_history[0, time_idx, 1]
                row['residual_dec_first'] = residual_history[0, time_idx, 2]
                
                # Last run (index -1)
                row['residual_ra_last'] = residual_history[-1, time_idx, 1]
                row['residual_dec_last'] = residual_history[-1, time_idx, 2]
            
            data.append(row)
            time_idx += 1
    
    df = pd.DataFrame(data)
    df = df.sort_values('id')
    return df

#--------------------------------------------------------------------------------------------------
# PLOTTING FUNCTIONS
#--------------------------------------------------------------------------------------------------
def create_color_mapping(df):
    """
    Create a consistent color mapping for all files in the dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with 'id' column
    
    Returns:
    --------
    dict : Dictionary mapping file number to color
    """
    file_numbers = sorted(df['id'].unique())
    n_files = len(file_numbers)
    
    # Generate colors
    if n_files <= 20:
        colors_array = plt.cm.tab20(np.linspace(0, 1, 20))
    else:
        colors_array = plt.cm.tab20b(np.linspace(0, 1, n_files))
    
    # Create a dictionary mapping file number to color
    file_colors = {file_nr: colors_array[i] for i, file_nr in enumerate(file_numbers)}
    
    return file_colors

def plot_observation_analysis(df, file_colors=None, title_suffix=""):
    """
    Create stacked histogram and bar chart of observations over time.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with columns: 'id', 'observatory', 'times'...
    file_colors : dict, optional
        Dictionary mapping file number to color. If None, creates new mapping.
    title_suffix : str
        Additional text to add to plot titles
    
    Returns:
    --------
    dict : Color mapping used in the plot
    fig  : Figure to save to specific folder later
    """
    file_numbers = sorted(df['id'].unique())
    n_files = len(file_numbers)
    
    if n_files == 0:
        print("No data to plot!")
        return file_colors
    
    # Use provided color mapping or create new one
    if file_colors is None:
        file_colors = create_color_mapping(df)
    
    # Explode the dataframe so each observation time gets its own row
    rows = []
    for _, row in df.iterrows():
        # Convert Tudat time (seconds since J2000) to datetime
        time_datetime = J2000_EPOCH + timedelta(seconds=float(row['times']))
        rows.append({
            'id': row['id'],
            'observatory': row['observatory'],
            'time': time_datetime
        })

    df_exploded = pd.DataFrame(rows)
    
    # Create a period column for monthly bins
    df_exploded['month'] = df_exploded['time'].dt.to_period('M')
    
    # Count observations per file per month
    monthly_counts = df_exploded.groupby(['month', 'id']).size().unstack(fill_value=0)
    
    # Calculate n_observations for the bar plot
    id_counts = df.groupby('id').size().reset_index(name='n_observations')



    # Create figure with two subplots (stacked vertically)
    fig = plt.figure(figsize=(18, 12))
    
    # Adjust subplot positioning to leave room for legend
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3,
                          left=0.08, right=0.82, top=0.95, bottom=0.08)
    
    # Top subplot: Stacked histogram
    ax1 = fig.add_subplot(gs[0])
    
    # Get colors in the correct order for the stacked plot (only for files in current data)
    plot_colors = [file_colors[nr] for nr in monthly_counts.columns]
    
    monthly_counts.plot(
        kind='bar',
        stacked=True,
        ax=ax1,
        color=plot_colors,
        width=1.0,
        legend=False
    )
    
    ax1.set_xlabel('Time (Month)', fontsize=12)
    ax1.set_ylabel('Number of Observations', fontsize=12)
    title = f'Observation Count Over Time by File{title_suffix}'
    ax1.set_title(title, fontsize=14)
    
    # Format x-axis to show fewer labels
    n_labels = 20
    tick_positions = np.linspace(0, len(monthly_counts) - 1, n_labels, dtype=int)
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels([str(monthly_counts.index[i]) for i in tick_positions], 
                         rotation=45, ha='right')
    
    # Create legend outside the plot area (only for files in current data)
    handles = [Patch(facecolor=file_colors[nr], label=f'File {nr}') 
               for nr in file_numbers]
    ax1.legend(handles=handles, 
               bbox_to_anchor=(1.02, 1), 
               loc='upper left',
               ncol=1,
               fontsize=8,
               frameon=True)
    
    # Bottom subplot: Bar chart of observations per file
    ax2 = fig.add_subplot(gs[1])
    
    # Use the same colors for each file
    bar_colors = [file_colors[id] for id in id_counts['id']]

    ax2.bar(id_counts['id'], id_counts['n_observations'], color=bar_colors)
    ax2.set_xlabel('File Number', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Number of Observations per File', fontsize=14)
    ax2.tick_params(axis='x', rotation=45)
    

    return fig, file_colors



def calculate_mean_std(df):
    """
    Create bar charts showing mean and std for RA and DEC per file.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with columns: 'id', 'times', 'residuals_ra_first', 'residuals_dec_first', 'residuals_ra_last', 'residuals_dec_last'

    Returns:
    --------
    df : pd.DataFrame
        Dataframe with columns: id  ra_first_mean  ra_first_std  dec_first_mean  dec_first_std  ra_last_mean  ra_last_std  dec_last_mean  dec_last_std
    """
    residual_stats = df.groupby('id').agg({
    'residual_ra_first': ['mean', 'std'],
    'residual_dec_first': ['mean', 'std'],
    'residual_ra_last': ['mean', 'std'],
    'residual_dec_last': ['mean', 'std']
    }).reset_index()

    # Flatten column names
    residual_stats.columns = ['id', 'ra_first_mean', 'ra_first_std', 
                            'dec_first_mean', 'dec_first_std',
                            'ra_last_mean', 'ra_last_std',
                            'dec_last_mean', 'dec_last_std']

    return residual_stats



def plot_mean_std (df,file_colors=None,title_suffix = ""):
    """
    Create bar charts showing mean and std for RA and DEC per file.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with columns: id  ra_first_mean  ra_first_std  dec_first_mean  dec_first_std  ra_last_mean  ra_last_std  dec_last_mean  dec_last_std
    file_colors : dict, optional
        Dictionary mapping file number to color. If None, creates new mapping.
    title_suffix : str
        Additional text to add to plot titles.
    
    Returns:
    --------
    dict  : Color mapping used in the plot
    fig1 : Figure of the initial propagation
    fig2 : Figure of the last propagation
    """

    # Use provided color mapping or create new one
    if file_colors is None:
        file_colors = create_color_mapping(df)
    

    # Get colors for each ID
    bar_colors = [file_colors[id] for id in df['id']]


    # Create figure with 2 rows and 2 columns (RA and DEC for mean and std)
    fig1, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Mean RA per file
    axes[0, 0].bar(df['id'], df['ra_first_mean'], color=bar_colors)
    axes[0, 0].set_xlabel('File ID')
    axes[0, 0].set_ylabel('Mean RA [arcseconds]')
    axes[0, 0].set_title(f'Mean RA Residuals per File{title_suffix} Initila')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Std RA per file
    axes[0, 1].bar(df['id'], df['ra_first_std'], color=bar_colors)
    axes[0, 1].set_xlabel('File ID')
    axes[0, 1].set_ylabel('Std RA [arcseconds]')
    axes[0, 1].set_title(f'Standard Deviation RA per File{title_suffix} Initial')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Mean DEC per file
    axes[1, 0].bar(df['id'], df['dec_first_mean'], color=bar_colors)
    axes[1, 0].set_xlabel('File ID')
    axes[1, 0].set_ylabel('Mean DEC [arcseconds]')
    axes[1, 0].set_title(f'Mean DEC Residuals per File{title_suffix} Initial')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Std DEC per file
    axes[1, 1].bar(df['id'], df['ra_first_std'], color=bar_colors)
    axes[1, 1].set_xlabel('File ID')
    axes[1, 1].set_ylabel('Std DEC [arcseconds]')
    axes[1, 1].set_title(f'Standard Deviation DEC per File{title_suffix} Initial')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()


    # Create figure with 2 rows and 2 columns (RA and DEC for mean and std)
    fig2, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Mean RA per file
    axes[0, 0].bar(df['id'], df['ra_last_mean'], color=bar_colors)
    axes[0, 0].set_xlabel('File ID')
    axes[0, 0].set_ylabel('Mean RA [arcseconds]')
    axes[0, 0].set_title(f'Mean RA Residuals per File{title_suffix} Last')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Std RA per file
    axes[0, 1].bar(df['id'], df['ra_last_std'], color=bar_colors)
    axes[0, 1].set_xlabel('File ID')
    axes[0, 1].set_ylabel('Std RA [arcseconds]')
    axes[0, 1].set_title(f'Standard Deviation RA per File{title_suffix} Last')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Mean DEC per file
    axes[1, 0].bar(df['id'], df['dec_last_mean'], color=bar_colors)
    axes[1, 0].set_xlabel('File ID')
    axes[1, 0].set_ylabel('Mean DEC [arcseconds]')
    axes[1, 0].set_title(f'Mean DEC Residuals per File{title_suffix}')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Std DEC per file
    axes[1, 1].bar(df['id'], df['ra_last_std'], color=bar_colors)
    axes[1, 1].set_xlabel('File ID')
    axes[1, 1].set_ylabel('Std DEC [arcseconds]')
    axes[1, 1].set_title(f'Standard Deviation DEC per File{title_suffix} Last')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    

    
    return fig1,fig2, file_colors

def overlay_residual_stats(df, file_colors, fig=None, axes=None, 
                          label='Final', alpha=0.5, edgecolor='black', 
                          linewidth=1.5, hatch='///'):
    """
    Overlay residual statistics on existing or new figure.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns: 'id', 'residual_ra_first', 'residual_dec_first',
        'residual_ra_last', 'residual_dec_last'
    file_colors : dict
        Dictionary mapping file IDs to RGBA color arrays
    fig : matplotlib.figure.Figure, optional
        Existing figure to overlay on. If None, creates new figure
    axes : array of matplotlib.axes.Axes, optional
        Array of 4 axes (2x2) to plot on. If None, creates new axes
    label : str, optional
        Label for the overlaid data (default: 'Final')
    alpha : float, optional
        Transparency of overlaid bars (default: 0.5)
    edgecolor : str, optional
        Edge color for overlaid bars (default: 'black')
    linewidth : float, optional
        Edge line width (default: 1.5)
    hatch : str, optional
        Hatch pattern for overlaid bars (default: '///' for diagonal lines)
        Options: '///', '\\\\\\', '|||', '---', '+++', 'xxx', '...', '**', 'oo', 'OO'
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    axes : array of matplotlib.axes.Axes
        Array of 4 axes
    """
    # Calculate statistics per ID
    residual_stats = df.groupby('id').agg({
        'residual_ra_first': ['mean', 'std'],
        'residual_dec_first': ['mean', 'std'],
        'residual_ra_last': ['mean', 'std'],
        'residual_dec_last': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    residual_stats.columns = ['id', 'ra_first_mean', 'ra_first_std', 
                              'dec_first_mean', 'dec_first_std',
                              'ra_last_mean', 'ra_last_std',
                              'dec_last_mean', 'dec_last_std']
    
    # Keep the order from the dataframe (first occurrence of each ID)
    id_order = df['id'].drop_duplicates().tolist()
    residual_stats['id'] = pd.Categorical(residual_stats['id'], 
                                          categories=id_order, 
                                          ordered=True)
    residual_stats = residual_stats.sort_values('id')
    
    # Get colors for each ID
    bar_colors = [file_colors[id] for id in residual_stats['id']]
    
    # Create figure if not provided
    if fig is None or axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        is_new_figure = True
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else axes
        is_new_figure = False
    
    x = np.arange(len(residual_stats))
    width = 0.6
    
    # If new figure, plot the "Initial" data first
    if is_new_figure:
        # RA Mean Initial
        axes[0].bar(x, residual_stats['ra_first_mean'], width, 
                   label='Initial', alpha=0.8, color=bar_colors)
        axes[0].set_xlabel('File ID', fontsize=12)
        axes[0].set_ylabel('Mean RA Residual (arcsec)', fontsize=12)
        axes[0].set_title('Mean RA Residuals per File', fontsize=14)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(residual_stats['id'], rotation=45, ha='right')
        axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[0].grid(axis='y', alpha=0.3)
        
        # RA Std Initial
        axes[1].bar(x, residual_stats['ra_first_std'], width, 
                   label='Initial', alpha=0.8, color=bar_colors)
        axes[1].set_xlabel('File ID', fontsize=12)
        axes[1].set_ylabel('Std RA (arcsec)', fontsize=12)
        axes[1].set_title('Standard Deviation RA per File', fontsize=14)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(residual_stats['id'], rotation=45, ha='right')
        axes[1].grid(axis='y', alpha=0.3)
        
        # DEC Mean Initial
        axes[2].bar(x, residual_stats['dec_first_mean'], width, 
                   label='Initial', alpha=0.8, color=bar_colors)
        axes[2].set_xlabel('File ID', fontsize=12)
        axes[2].set_ylabel('Mean DEC Residual (arcsec)', fontsize=12)
        axes[2].set_title('Mean DEC Residuals per File', fontsize=14)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(residual_stats['id'], rotation=45, ha='right')
        axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[2].grid(axis='y', alpha=0.3)
        
        # DEC Std Initial
        axes[3].bar(x, residual_stats['dec_first_std'], width, 
                   label='Initial', alpha=0.8, color=bar_colors)
        axes[3].set_xlabel('File ID', fontsize=12)
        axes[3].set_ylabel('Std DEC (arcsec)', fontsize=12)
        axes[3].set_title('Standard Deviation DEC per File', fontsize=14)
        axes[3].set_xticks(x)
        axes[3].set_xticklabels(residual_stats['id'], rotation=45, ha='right')
        axes[3].grid(axis='y', alpha=0.3)
    
    # Overlay the "Final" or other data with hatch pattern
    axes[0].bar(x, residual_stats['ra_last_mean'], width, 
               label=label, alpha=alpha, color=bar_colors, 
               edgecolor=edgecolor, linewidth=linewidth, hatch='///')
    axes[0].legend()
    
    axes[1].bar(x, residual_stats['ra_last_std'], width, 
               label=label, alpha=alpha, color=bar_colors, 
               edgecolor=edgecolor, linewidth=linewidth, hatch='///')
    axes[1].legend()
    
    axes[2].bar(x, residual_stats['dec_last_mean'], width, 
               label=label, alpha=alpha, color=bar_colors, 
               edgecolor=edgecolor, linewidth=linewidth, hatch='///')
    axes[2].legend()
    
    axes[3].bar(x, residual_stats['dec_last_std'], width, 
               label=label, alpha=alpha, color=bar_colors, 
               edgecolor=edgecolor, linewidth=linewidth, hatch='///')
    axes[3].legend()
    
    plt.tight_layout()
    
    return fig, axes


def plot_RA_DEC_residuals_colored(df, file_colors=None, labels=["final", "initial"], save_path=None):
    """
    Plot initial and final RA and DEC residuals side by side for multiple files.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: 'id', 'observatory', 'times', 
        'residual_ra_first', 'residual_dec_first', 
        'residual_ra_last', 'residual_dec_last'
    file_colors : dict, optional
        Dictionary mapping file IDs (e.g., '119_nm0017') to RGBA color arrays.
        If None, uses default colors.
    labels : list of str
        Labels for [final, initial] residuals
    save_path : str or None, optional
        If provided, saves the figure to this path
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """

    fig, ax = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True, sharex=True)

    # Precompute datetime once for all rows
    df = df.copy()
    df["datetime"] = [J2000_EPOCH + timedelta(seconds=t) for t in df["times"]]

    # Group by file ID for efficient plotting
    grouped = df.groupby("id")

    for file_id, group in grouped:
        color = file_colors.get(file_id, None) if file_colors else None

        # --- RA plot ---
        ax[0].scatter(group["datetime"], group["residual_ra_last"],
                      color=color, alpha=0.6, s=10, label=labels[0])
        #ax[0].scatter(group["datetime"], group["residual_ra_first"],
        #              color=color, alpha=0.3, s=10, marker='x', label=labels[1])

        # --- DEC plot ---
        ax[1].scatter(group["datetime"], group["residual_dec_last"],
                      color=color, alpha=0.6, s=10)
        #ax[1].scatter(group["datetime"], group["residual_dec_first"],
        #              color=color, alpha=0.3, s=10, marker='x')

    # --- Formatting for both subplots ---
    for axis, ylabel in zip(ax, ["simulated - observed RA [arcseconds]",
                                 "simulated - observed DEC [arcseconds]"]):
        axis.set_xlabel("Observation epoch")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        axis.axhline(y=0, color='k', linestyle='--', alpha=0.3)

        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(formatter)

    # Add legend only once if few files
    if len(grouped) <= 10:
        ax[0].legend(loc='best', fontsize=8)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig



def plot_RA_DEC_residuals(
    df,
    df2=None,
    file_colors=None,
    labels=("final", "initial"),
    dataset_labels=("set 1", "set 2"),
    fig=None,
    ax=None,
    save_path=None,
):
    """
    Plot initial and final RA and DEC residuals side by side for multiple files.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: 'id', 'observatory', 'times',
        'residual_ra_first', 'residual_dec_first',
        'residual_ra_last', 'residual_dec_last'
    df2 : pd.DataFrame or None, optional
        Optional second DataFrame with the same structure to overlay on the same axes.
    file_colors : dict, optional
        Dictionary mapping file IDs (e.g., '119_nm0017') to RGBA color arrays.
        If None, uses default colors.
    labels : tuple(str, str)
        Labels for [final, initial] residuals (applies to both datasets).
    dataset_labels : tuple(str, str)
        Dataset names used in the legend for df and df2, respectively.
    fig, ax : matplotlib Figure and Axes or None
        If provided, plot onto these axes; otherwise create new (1x2) axes.
    save_path : str or None, optional
        If provided, saves the figure to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """

    def _prepare(_df):
        _df = _df.copy()
        if "datetime" not in _df.columns:
            _df["datetime"] = [J2000_EPOCH + timedelta(seconds=t) for t in _df["times"]]
        return _df

    def _plot_one(_ax, _df, ds_label, alpha_main=0.7, alpha_init=0.35, marker_init="x"):
        # final
        _ax[0].scatter(_df["datetime"], _df["residual_ra_last"],
                       alpha=alpha_main, s=10, label=f"{ds_label} ({labels[0]})")
        _ax[1].scatter(_df["datetime"], _df["residual_dec_last"],
                       alpha=alpha_main, s=10, label=f"{ds_label} ({labels[0]})")
        # initial
        #_ax[0].scatter(_df["datetime"], _df["residual_ra_first"],
        #               alpha=alpha_init, s=10, marker=marker_init, label=f"{ds_label} ({labels[1]})")
        #_ax[1].scatter(_df["datetime"], _df["residual_dec_first"],
        #               alpha=alpha_init, s=10, marker=marker_init, label=f"{ds_label} ({labels[1]})")

    # Create or reuse axes
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True, sharex=True)

        # Common formatting (do once if we created axes)
        for axis, ylabel in zip(ax, ["simulated - observed RA [arcseconds]",
                                     "simulated - observed DEC [arcseconds]"]):
            axis.set_xlabel("Observation epoch")
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.3)
            axis.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            axis.xaxis.set_major_locator(locator)
            axis.xaxis.set_major_formatter(formatter)

    # Plot df
    df = _prepare(df)
    _plot_one(ax, df, dataset_labels[0])

    # Optional overlay df2
    if df2 is not None:
        df2 = _prepare(df2)
        # Use slightly different alpha/marker so overlays are distinguishable
        _plot_one(ax, df2, dataset_labels[1], alpha_main=0.9, alpha_init=0.5, marker_init="+")

    # Legend on RA axis (covers both datasets and both states)
    ax[0].legend(loc='best', fontsize=8, ncol=2)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig, ax

def plot_individual_RA_DEC_residuals(df, file_colors=None, labels=["Final", "Initial"], 
                                    show_initial=False):
    """
    Create separate RA and DEC residual plots for each unique ID.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: 'id', 'observatory', 'times', 
        'residual_ra_first', 'residual_dec_first', 
        'residual_ra_last', 'residual_dec_last'
    file_colors : dict, optional
        Dictionary mapping file IDs (e.g., '119_nm0017') to RGBA color arrays.
        If None, uses default colors.
    labels : list of str
        Labels for [final, initial] residuals. Default: ["Final", "Initial"]
    save_folder : str or None, optional
        If provided, saves each figure to this folder with filename based on ID
    show_initial : bool, optional
        If True, also plots initial residuals. Default: True
    
    Returns
    -------
    figs : dict
        Dictionary mapping file IDs to their figure objects
    """
    # J2000 epoch
    J2000_EPOCH = datetime(2000, 1, 1, 12, 0, 0)
    
    # Get unique IDs
    unique_ids = df['id'].unique()
    
    # Store figures
    figs = {}
    
    for file_id in unique_ids:
        # Filter data for this ID
        df_id = df[df['id'] == file_id].copy()
        
        # Convert Tudat time to datetime
        df_id['datetime'] = df_id['times'].apply(
            lambda t: J2000_EPOCH + timedelta(seconds=float(t))
        )
        
        # Sort by time
        df_id = df_id.sort_values('datetime')
        
        # Get color for this file
        color = file_colors.get(file_id, 'C0') if file_colors else 'C0'
        
        # Create figure
        fig, ax = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        
        # --- RA plot ---
        ax[0].scatter(df_id['datetime'], df_id['residual_ra_last'], 
                     color=color, alpha=0.7, s=30, label=labels[0])
        
        if show_initial:
            ax[0].scatter(df_id['datetime'], df_id['residual_ra_first'], 
                         color=color, alpha=0.3, marker='x', s=30, 
                         label=labels[1])
        
        ax[0].set_xlabel('Observation Epoch', fontsize=12)
        ax[0].set_ylabel('Simulated - Observed RA [arcseconds]', fontsize=12)
        ax[0].set_title(f'RA Residuals: {file_id}', fontsize=14)
        ax[0].grid(True, alpha=0.3)
        ax[0].axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
        ax[0].legend(loc='best')
        
        # Format x-axis
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax[0].xaxis.set_major_locator(locator)
        ax[0].xaxis.set_major_formatter(formatter)
        
        # --- DEC plot ---
        ax[1].scatter(df_id['datetime'], df_id['residual_dec_last'], 
                     color=color, alpha=0.7, s=30, label=labels[0])
        
        if show_initial:
            ax[1].scatter(df_id['datetime'], df_id['residual_dec_first'], 
                         color=color, alpha=0.3, marker='x', s=30,
                         label=labels[1])
        
        ax[1].set_xlabel('Observation Epoch', fontsize=12)
        ax[1].set_ylabel('Simulated - Observed DEC [arcseconds]', fontsize=12)
        ax[1].set_title(f'DEC Residuals: {file_id}', fontsize=14)
        ax[1].grid(True, alpha=0.3)
        ax[1].axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
        ax[1].legend(loc='best')
        
        # Format x-axis
        ax[1].xaxis.set_major_locator(locator)
        ax[1].xaxis.set_major_formatter(formatter)
        

        # Store figure
        figs[file_id] = fig
    
    return figs

if __name__ == "__main__":
    #############################################################################################
    # LOAD SPICE KERNELS
    ##############################################################################################

    print("Starting Post Processing...")

    # Path to the current script
    current_dir = Path(__file__).resolve().parent

    # Kernel folder 
    kernel_folder = "Kernels" #current_dir.parent / 

    #kernel_folder = "/Kernels/"
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



    ##############################################################################################
    # CREATE ENVIRONMENT  
    ##############################################################################################

    # Define temporal scope of the simulation - equal to the time JUICE will spend in orbit around Jupiter
    simulation_start_epoch = DateTime(1960, 1,  1).epoch()
    simulation_end_epoch   = DateTime(2025, 1, 1).epoch()
    global_frame_origin = 'SSB'
    global_frame_orientation = 'ECLIPJ2000'

    #--------------------------------------------------------------------------------------------
    # ENVIORONMENT SETTINGS 
    #--------------------------------------------------------------------------------------------
    settings_env = dict()
    settings_env["start_epoch"] = simulation_start_epoch
    settings_env["end_epoch"] = simulation_end_epoch
    settings_env["bodies"] = ['Sun','Jupiter', 'Saturn','Neptune','Triton','Uranus','Mercury','Venus','Mars','Earth']
    settings_env["global_frame_origin"] = global_frame_origin
    settings_env["global_frame_orientation"] = global_frame_orientation
    settings_env["interpolator_triton_cadance"] = 60*8
    settings_env["neptune_extended_gravity"] = "Jacobson2009"


    body_settings,system_of_bodies = PropFuncs.Create_Env(settings_env)

    #--------------------------------------------------------------------------------------------
    # LOAD OBSERVATIONS
    #--------------------------------------------------------------------------------------------
    with open("file_names.json", "r") as f:
        file_names_loaded = json.load(f)

    # Extract IDs from filenames (remove 'Triton_' prefix and '.csv' suffix)
    ordered_ids = [f.replace('Triton_', '').replace('.csv', '') for f in file_names_loaded]


    data = []
    observations,observations_settings,observation_set_ids = ObsFunc.LoadObservations("Observations/ProcessedOutliers/",system_of_bodies,file_names_loaded)

    print("Loaded tudat observations...")
    #--------------------------------------------------------------------------------------------
    # LOAD ESTIMATION RESULTS
    #--------------------------------------------------------------------------------------------
    arrays = load_npy_files("/home/atn/Documents/Year 5/Thesis/Github/NeptuneTritonMasterThesis/Results/BetterFigs/Outliers/First")
    
    print("Loaded numpy arrays from estimation...")
    nr_observations = []
    for list in arrays['residuals_sorted']:
        nr_observations.append(len(list)/2)
    nr_observations_check = []

    for list in observations.get_observations():
        nr_observations_check.append(len(list)/2) 

    df = create_observations_dataframe(
        arrays['observations_sorted'], 
        ordered_ids,
        arrays['observation_times'],
        arrays['residual_history_arcseconds'])
    
    #Load results without outliers
    arrays = load_npy_files("Results/BetterFigs/FirstFitNoWeights/EstimationTest7_State_only")
    
    print("Loaded previous numpy arrays from estimation...")
    nr_observations = []
    for list in arrays['residuals_sorted']:
        nr_observations.append(len(list)/2)
    nr_observations_check = []

    for list in observations.get_observations():
        nr_observations_check.append(len(list)/2) 

    df_initial = create_observations_dataframe(
        arrays['observations_sorted'], 
        ordered_ids,
        arrays['observation_times'],
        arrays['residual_history_arcseconds'])
    
    



    #--------------------------------------------------------------------------------------------
    # MAKE FIGURES
    #--------------------------------------------------------------------------------------------

    # Load unique id colors 
    with open('file_colors.pkl', 'rb') as f:
        file_colors = pickle.load(f)


    print("Making figures...")
    out_dir = make_timestamped_folder("Results/BetterFigs/Outliers/PostProcessing/First")

    

    fig, _ = plot_observation_analysis(df, file_colors=file_colors, title_suffix="All Data")
    fig.savefig(out_dir / "Colored_Count.pdf")
    # fig_rms = FigUtils.Residuals_RMS(residual_history_arcseconds)
    # fig_rms.savefig(out_dir / "Residual_RMS.pdf")

    residual_df = calculate_mean_std(df)
    fig_first,fig_last,_ = plot_mean_std(residual_df,file_colors=file_colors,title_suffix = "")
    fig_first.savefig(out_dir / "Colored_mean_std_Initial.pdf")
    fig_last.savefig(out_dir / "Colored_mean_std_Last.pdf")

    fig, axes = overlay_residual_stats(df, file_colors)
    fig.savefig(out_dir / "Colored_mean_std_overlayed.pdf")


    print("Created and saved all other figures...")

    fig_estimation_residuals,ax = plot_RA_DEC_residuals(df,df_initial, file_colors=file_colors, dataset_labels=("removed outliers", "with outliers"))
    fig_estimation_residuals.savefig(out_dir / "Residuals_time.pdf")

    fig_estimation_residuals_colored = plot_RA_DEC_residuals_colored(df, file_colors=file_colors, labels=["final", "initial"])
    fig_estimation_residuals_colored.savefig(out_dir / "Residuals_time_colored.pdf")

    individual_figs = plot_individual_RA_DEC_residuals(df,file_colors) 

    for file_id, fig in individual_figs.items():
        fig.savefig(out_dir / (file_id + '_residuals_time.pdf'))



    outliers = df_initial[~df_initial['times'].isin(df['times'])].copy()
    fig_outlier_residuals_colored = plot_RA_DEC_residuals_colored(outliers, file_colors=file_colors, labels=["final", "initial"])
    fig_outlier_residuals_colored.savefig(out_dir / "Residuals_time_colored_outliers.pdf")

    fig_estimation_residuals,ax = plot_RA_DEC_residuals(df,outliers, file_colors=file_colors, dataset_labels=("removed outliers", "outliers only"))
    fig_estimation_residuals.savefig(out_dir / "Residuals_time_vs_outliers.pdf")


    #--------------------------------------------------------------------------------------------
    # COMPUTE WEIGHTS
    #--------------------------------------------------------------------------------------------


    std_per_id = df.groupby("id")[["residual_ra_last", "residual_dec_last"]].std().rename(
    columns={"residual_ra_last": "std_ra", "residual_dec_last": "std_dec"})
    
    weights = 1.0 / (std_per_id ** 2)
    weights.columns = ["weight_ra", "weight_dec"]
    
    weights.to_csv(out_dir / "weights.txt", sep="\t", float_format="%.8f")


    # Assign weights
    observations,observations_settings,observation_set_ids = ObsFunc.LoadObservations("Observations/ProcessedOutliers/",system_of_bodies,file_names_loaded,weights = weights)

    print("end")