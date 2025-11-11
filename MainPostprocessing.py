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

import tudatpy



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


def plot_individual_RA_DEC_residuals(
    df,
    file_colors=None,
    labels=("Final", "Initial"),
    show_initial=False,
    show_timeframes=False,
    gap_hours=4.0,
    frame_line_kwargs=None,
):
    """
    Create separate RA and DEC residual plots for each unique ID.
    Optionally draw vertical lines BETWEEN time frames (midpoint of large gaps).

    Parameters
    ----------
    df : pd.DataFrame
        Columns: 'id','observatory','times',
                 'residual_ra_first','residual_dec_first',
                 'residual_ra_last','residual_dec_last'
        'times' should be seconds since J2000 (float).
    file_colors : dict, optional
        Map file_id -> color (e.g. {'119_nm0017': 'C0'})
    labels : (str, str)
        Labels for [final, initial] residuals (order matters).
    show_initial : bool
        If True, also plot initial residuals.
    show_timeframes : bool
        If True, draw vertical lines between time frames.
    gap_hours : float
        Minimum gap that defines a split between frames (default 4.0h).
    frame_line_kwargs : dict or None
        Matplotlib kwargs for the vertical lines (e.g., {'color':'k','ls':':'})

    Returns
    -------
    figs : dict
        {file_id: figure}
    """
    # Defaults for the split lines
    if frame_line_kwargs is None:
        frame_line_kwargs = dict(color='k', linestyle=':', linewidth=1.0, alpha=0.35)

    # J2000 epoch
    J2000_EPOCH = datetime(2000, 1, 1, 12, 0, 0)
    gap_sec = float(gap_hours) * 3600.0

    # Get unique IDs
    unique_ids = df['id'].unique()

    figs = {}

    for file_id in unique_ids:
        # Filter & copy, convert time
        df_id = df[df['id'] == file_id].copy()
        df_id['datetime'] = df_id['times'].astype(float).apply(
            lambda t: J2000_EPOCH + timedelta(seconds=t)
        )
        df_id = df_id.sort_values('datetime')

        # Pick color
        color = file_colors.get(file_id, 'C0') if file_colors else 'C0'

        # Create figure
        fig, ax = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

        # === RA ===
        ax[0].scatter(df_id['datetime'], df_id['residual_ra_last'],
                      color=color, alpha=0.7, s=30, label=labels[0])
        if show_initial:
            ax[0].scatter(df_id['datetime'], df_id['residual_ra_first'],
                          color=color, alpha=0.35, marker='x', s=30, label=labels[1])

        ax[0].set_xlabel('Observation Epoch', fontsize=12)
        ax[0].set_ylabel('Simulated - Observed RA [arcseconds]', fontsize=12)
        ax[0].set_title(f'RA Residuals: {file_id}', fontsize=14)
        ax[0].grid(True, alpha=0.3)
        ax[0].axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
        ax[0].legend(loc='best')

        # === DEC ===
        ax[1].scatter(df_id['datetime'], df_id['residual_dec_last'],
                      color=color, alpha=0.7, s=30, label=labels[0])
        if show_initial:
            ax[1].scatter(df_id['datetime'], df_id['residual_dec_first'],
                          color=color, alpha=0.35, marker='x', s=30, label=labels[1])

        ax[1].set_xlabel('Observation Epoch', fontsize=12)
        ax[1].set_ylabel('Simulated - Observed DEC [arcseconds]', fontsize=12)
        ax[1].set_title(f'DEC Residuals: {file_id}', fontsize=14)
        ax[1].grid(True, alpha=0.3)
        ax[1].axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
        ax[1].legend(loc='best')

        # Shared time formatting
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        for a in ax:
            a.xaxis.set_major_locator(locator)
            a.xaxis.set_major_formatter(formatter)

        # === Optional: draw vertical lines BETWEEN time frames ===
        if show_timeframes and len(df_id) >= 2:
            t = df_id['datetime'].to_numpy()
            # diffs in seconds between consecutive observations
            dt_sec = np.diff(df_id['times'].astype(float).to_numpy())
            # indices where gap >= threshold
            cut_idx = np.where(dt_sec >= gap_sec)[0]
            # draw a line at the midpoint between the two timestamps that straddle the gap
            for i in cut_idx:
                mid_dt = t[i] + (t[i+1] - t[i]) / 2
                for a in ax:
                    a.axvline(mid_dt, **frame_line_kwargs)

        figs[file_id] = fig

    return figs



def plot_individual_RA_DEC_residuals_demeaned(
    df,
    file_colors=None,
    labels=("Final (demeaned)", "Initial (demeaned)"),
    show_initial=False,
):
    """
    For each ID, subtract the per-file mean from each observation (RA/DEC),
    then plot time series panels for RA and DEC.

    Parameters
    ----------
    df : pd.DataFrame
        Requires columns:
        'id','observatory','times',
        'residual_ra_first','residual_dec_first',
        'residual_ra_last','residual_dec_last'
    file_colors : dict, optional
        Map file_id -> color (e.g. {'119_nm0017':'C0'})
    labels : (str, str)
        Legend labels for (final, initial) traces.
    show_initial : bool
        If True, also plot demeaned initial residuals.

    Returns
    -------
    figs : dict
        {file_id: figure}
    """
    # Work on a copy
    df = df.copy()

    # J2000 epoch → datetime
    J2000_EPOCH = datetime(2000, 1, 1, 12, 0, 0)
    df["datetime"] = df["times"].astype(float).apply(
        lambda t: J2000_EPOCH + timedelta(seconds=t)
    )

    # --- demean per file (ID) ---
    # final residuals
    df["ra_last_demean"]  = df["residual_ra_last"] - df.groupby("id")["residual_ra_last"].transform("mean")
    df["dec_last_demean"] = df["residual_dec_last"] - df.groupby("id")["residual_dec_last"].transform("mean")
    # initial residuals (only if you plan to plot them)
    if show_initial:
        df["ra_first_demean"]  = df["residual_ra_first"] - df.groupby("id")["residual_ra_first"].transform("mean")
        df["dec_first_demean"] = df["residual_dec_first"] - df.groupby("id")["residual_dec_first"].transform("mean")

    figs = {}
    unique_ids = df["id"].unique()

    for file_id in unique_ids:
        d = df[df["id"] == file_id].sort_values("datetime")

        color = (file_colors.get(file_id, "C0") if file_colors else "C0")

        fig, ax = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

        # --- RA ---
        ax[0].scatter(d["datetime"], d["ra_last_demean"], color=color, alpha=0.8, s=30, label=labels[0])
        if show_initial:
            ax[0].scatter(d["datetime"], d["ra_first_demean"], color=color, alpha=0.35, marker="x", s=30, label=labels[1])

        ax[0].set_xlabel("Observation Epoch", fontsize=12)
        ax[0].set_ylabel("RA residual − mean [arcsec]", fontsize=12)
        ax[0].set_title(f"RA Residuals (demeaned): {file_id}", fontsize=14)
        ax[0].grid(True, alpha=0.3)
        ax[0].axhline(0, color="k", linestyle="--", alpha=0.5, linewidth=1)
        ax[0].legend(loc="best")

        # --- DEC ---
        ax[1].scatter(d["datetime"], d["dec_last_demean"], color=color, alpha=0.8, s=30, label=labels[0])
        if show_initial:
            ax[1].scatter(d["datetime"], d["dec_first_demean"], color=color, alpha=0.35, marker="x", s=30, label=labels[1])

        ax[1].set_xlabel("Observation Epoch", fontsize=12)
        ax[1].set_ylabel("DEC residual − mean [arcsec]", fontsize=12)
        ax[1].set_title(f"DEC Residuals (demeaned): {file_id}", fontsize=14)
        ax[1].grid(True, alpha=0.3)
        ax[1].axhline(0, color="k", linestyle="--", alpha=0.5, linewidth=1)
        ax[1].legend(loc="best")

        # nice time formatting
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        for a in ax:
            a.xaxis.set_major_locator(locator)
            a.xaxis.set_major_formatter(formatter)

        figs[file_id] = fig

    return figs






def plot_formal_errors(initial, final, sigma, title="Change from Initial with Formal Errors"):
    pos_diff, vel_diff = final[:3] - initial[:3], final[3:] - initial[3:]
    pos_err, vel_err = sigma[:3], sigma[3:]
    labels = ['X', 'Y', 'Z']

    fig, axes = plt.subplots(1, 2, figsize=(6, 7), sharex=False)
    fig.suptitle(title, fontsize=14, y=0.98)

    def plot_single_axis(ax, diffs, errs, ylabel):
        for i, label in enumerate(labels):
            # horizontal offset so columns don't overlap
            x = i  
            ax.errorbar(x, diffs[i], yerr=errs[i], fmt='s', color='C1', capsize=4, label=None)
            ax.scatter(x, 0.0, marker='o', color='C0', zorder=3)  # initial at 0
            # vertical line at 0 for reference
            ax.plot([x, x], [0, diffs[i]], color='gray', lw=0.8, alpha=0.4)

        # make it look like a single vertical axis
        ax.set_xlim(-0.5, len(labels)-0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis='y', alpha=0.3)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        # remove x-axis line (optional, keeps only vertical)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='x', bottom=False)

    # Position (left)
    plot_single_axis(axes[0], pos_diff, pos_err, 'Δ meters')
    axes[0].set_title('Position')

    # Velocity (right)
    plot_single_axis(axes[1], vel_diff, vel_err, 'Δ m/s')
    axes[1].set_title('Velocity')

    # One legend for both
    handles = [
        plt.Line2D([], [], marker='o', color='C0', linestyle='None', label='Initial (0)'),
        plt.Line2D([], [], marker='s', color='C1', linestyle='None', label='Final ±σ')
    ]
    #fig.legend(handles=handles, ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    from matplotlib.ticker import ScalarFormatter

    # --- scientific notation for POSITION y-axis ---
    sf = ScalarFormatter(useMathText=True)
    sf.set_powerlimits((0, 0))  # always use scientific notation
    axes[0].yaxis.set_major_formatter(sf)
    axes[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    axes[0].yaxis.get_offset_text().set_va('bottom')  # nicer offset label placement

    # --- legend above the plots (not covering the title) ---
    handles = [
        plt.Line2D([], [], marker='o', color='C0', linestyle='None', label='Initial (0)'),
        plt.Line2D([], [], marker='s', color='C1', linestyle='None', label='Final ±σ'),
    ]
    fig.legend(
        handles=handles,
        loc='upper center',
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.95)  # slightly below the suptitle
    )

    # give room for the legend + title
    plt.tight_layout(rect=[0, 0, 1, 0.93])




    return fig



def split_into_timeframes(
    df: pd.DataFrame,
    gap_hours: float = 4.0,
    time_col: str = "times",              # seconds since J2000
    make_datetime: bool = False,
    min_obs_per_frame: int = 10,
    merge_last_short_back: bool = False,
    summary_cols: list[str] = ["residual_ra_last", "residual_dec_last"],       # e.g. ["residual_ra_last", "residual_dec_last"]
):
    """
    Split each id into time frames using a gap threshold, but only break if the
    current frame already has >= min_obs_per_frame. Optionally merge a short
    final frame back. Adds per-frame mean/std for requested columns in 'summary'.
    """
    gap_sec = float(gap_hours) * 3600.0
    df_frames = df.copy().sort_values(["id", time_col]).reset_index(drop=True)

    if make_datetime and "datetime" not in df_frames.columns:
        J2000_EPOCH = datetime(2000, 1, 1, 12, 0, 0)
        df_frames["datetime"] = df_frames[time_col].astype(float).apply(
            lambda t: J2000_EPOCH + timedelta(seconds=float(t))
        )

    def assign_with_min_size(g):
        t = g[time_col].astype(float).to_numpy()
        n = len(t)
        if n == 0:
            return pd.Series([], dtype=int, index=g.index)
        diffs = np.diff(t)
        is_break = np.concatenate(([False], diffs >= gap_sec))
        frame_ids = np.zeros(n, dtype=int)
        cur_frame, cur_count = 0, 1
        for i in range(1, n):
            if is_break[i] and cur_count >= max(1, min_obs_per_frame):
                cur_frame += 1
                cur_count = 1
            else:
                cur_count += 1
            frame_ids[i] = cur_frame
        if merge_last_short_back and cur_count < max(1, min_obs_per_frame) and cur_frame > 0:
            frame_ids[frame_ids == cur_frame] = cur_frame - 1
        return pd.Series(frame_ids, index=g.index, dtype=int)

    df_frames["timeframe"] = df_frames.groupby("id", group_keys=False).apply(assign_with_min_size)

    # ---------- Build per-frame summary with mean/std ----------
    base = (
        df_frames.groupby(["id", "timeframe"])
        .agg(start_sec=(time_col, "min"),
             end_sec=(time_col, "max"),
             n_obs=(time_col, "count"))
        .reset_index()
    )
    base["duration_hours"] = (base["end_sec"] - base["start_sec"]) / 3600.0

    # Add per-frame means/stds for requested columns
    if summary_cols:
        stats = (
            df_frames.groupby(["id", "timeframe"])[summary_cols]
            .agg(["mean", "std"])  # std uses ddof=1 (sample std)
        )
        # Flatten MultiIndex columns: ("residual_ra_last","mean") -> "mean_residual_ra_last"
        stats.columns = [f"{func}_{col}" for col, func in stats.columns.swaplevel(0,1)]
        stats = stats.reset_index()
        summary = base.merge(stats, on=["id", "timeframe"], how="left")
    else:
        summary = base

    if make_datetime:
        J2000_EPOCH = datetime(2000, 1, 1, 12, 0, 0)
        summary["start_dt"] = summary["start_sec"].apply(lambda s: J2000_EPOCH + timedelta(seconds=float(s)))
        summary["end_dt"]   = summary["end_sec"].apply(lambda s: J2000_EPOCH + timedelta(seconds=float(s)))

    return df_frames, summary

def plot_timeframe_residuals(
    df,
    file_colors=None,              # e.g. {'119_nm0017': 'C0'}
    ylab_ra="RA residual [arcsec]",
    ylab_dec="DEC residual [arcsec]",
    min_obs_per_frame=1,
    save_folder=None,
):
    """
    Plot RA/DEC residuals for each timeframe of each ID.
    The dataframe must already include:
        - 'id'
        - 'timeframe'
        - 'datetime' (datetime64)
        - 'ra_last_plot', 'dec_last_plot'  (residuals to plot)

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed data including timeframe and residual columns.
    file_colors : dict, optional
        Mapping of id -> color.
    ylab_ra, ylab_dec : str
        Axis labels for RA/DEC.
    min_obs_per_frame : int
        Skip timeframes with fewer than this number of observations.
    save_folder : str or Path, optional
        If provided, save each figure as PNG.

    Returns
    -------
    figs : dict
        Dictionary {(id, timeframe): figure}.
    """
    figs = {}
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

    if save_folder is not None:
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)

    for (file_id, tf), d in df.groupby(["id", "timeframe"]):
        if len(d) < min_obs_per_frame:
            continue

        color = file_colors.get(file_id, "C0") if file_colors else "C0"
        d = d.sort_values("datetime")

        fig, ax = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

        # --- RA ---
        ax[0].scatter(d["datetime"], d["residual_ra_last"], color=color, alpha=0.8, s=30, label="Final")
        ax[0].axhline(0, color="k", ls="--", lw=1, alpha=0.5)
        ax[0].set_xlabel("Observation Epoch")
        ax[0].set_ylabel(ylab_ra)
        ax[0].set_title(f"RA • {file_id} • frame {tf} (n={len(d)})")
        ax[0].grid(True, axis="y", alpha=0.3)
        ax[0].xaxis.set_major_locator(locator)
        ax[0].xaxis.set_major_formatter(formatter)

        # --- DEC ---
        ax[1].scatter(d["datetime"], d["residual_dec_last"], color=color, alpha=0.8, s=30, label="Final")
        ax[1].axhline(0, color="k", ls="--", lw=1, alpha=0.5)
        ax[1].set_xlabel("Observation Epoch")
        ax[1].set_ylabel(ylab_dec)
        ax[1].set_title(f"DEC • {file_id} • frame {tf} (n={len(d)})")
        ax[1].grid(True, axis="y", alpha=0.3)
        ax[1].xaxis.set_major_locator(locator)
        ax[1].xaxis.set_major_formatter(formatter)



        # ----- Fix the x axis limits & ticks per frame -----
        t0, t1 = d["datetime"].min(), d["datetime"].max()
        if t0 == t1:
            pad = timedelta(hours=2)
            xlo, xhi = t0 - pad, t1 + pad
        else:
            pad = (t1 - t0) * 0.05
            xlo, xhi = t0 - pad, t1 + pad

        for a in ax:
            a.set_xlim(xlo, xhi)
            a.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
            a.xaxis.set_major_formatter(mdates.ConciseDateFormatter(a.xaxis.get_major_locator()))
            a.grid(True, axis="y", alpha=0.3)

        # ----- Safer suptitle without overlap -----
        suptxt = f"Timeframe {tf} • {file_id}\n{t0:%Y-%m-%d %H:%M} → {t1:%Y-%m-%d %H:%M} (Δ={t1-t0})"
        fig.suptitle(suptxt, fontsize=14)  # with constrained_layout this is fine
        # If you keep tight_layout instead:
        # plt.tight_layout(rect=[0, 0, 1, 0.92])


        # # --- Suptitle ---
        # t0, t1 = d["datetime"].min(), d["datetime"].max()
        # dur = (t1 - t0)
        # fig.suptitle(
        #     f"Timeframe {tf} • {file_id}\n{t0:%Y-%m-%d %H:%M} → {t1:%Y-%m-%d %H:%M}  (Δ={dur})",
        #     y=1.02,
        #     fontsize=14,
        # )

        # # --- Save if requested ---
        # if save_folder is not None:
        #     fname = f"{file_id}_frame{tf:02d}.png"
        #     fig.savefig(save_folder / fname, dpi=200, bbox_inches="tight")

        figs[(file_id, tf)] = fig

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
    arrays = load_npy_files("Results/BetterFigs/Weights/FirstWeights")
    
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
    arrays2 = load_npy_files("Results/BetterFigs/Outliers/First")
    
    print("Loaded previous numpy arrays from estimation...")
    nr_observations = []
    for list in arrays2['residuals_sorted']:
        nr_observations.append(len(list)/2)
    nr_observations_check = []

    for list in observations.get_observations():
        nr_observations_check.append(len(list)/2) 

    df_initial = create_observations_dataframe(
        arrays2['observations_sorted'], 
        ordered_ids,
        arrays2['observation_times'],
        arrays2['residual_history_arcseconds'])
    
    
    #--------------------------------------------------------------------------------------------
    # MAKE RSW differences
    #--------------------------------------------------------------------------------------------
    state_history_without_outliers = arrays['state_history_array']
    state_history_with_outliers = arrays2['state_history_array']
    time_column = state_history_without_outliers[:, [0]]  
    #states_SPICE_RSW = ProcessingUtils.rotate_inertial_3_to_rsw(time_column, states_SPICE[:,0:3], state_history_array)
    states_without_outliers_RSW = ProcessingUtils.rotate_inertial_3_to_rsw(time_column, state_history_without_outliers[:,1:4], state_history_without_outliers)
    states_with_outliers_RSW = ProcessingUtils.rotate_inertial_3_to_rsw(time_column, state_history_with_outliers[:,1:4], state_history_without_outliers)

    diff_RSW = (states_without_outliers_RSW - states_with_outliers_RSW)/1e3
    #--------------------------------------------------------------------------------------------
    # MAKE FIGURES
    #--------------------------------------------------------------------------------------------

    # Load unique id colors 
    with open('file_colors.pkl', 'rb') as f:
        file_colors = pickle.load(f)

    print("Making figures...")
    out_dir = make_timestamped_folder("Results/BetterFigs/Weights/PostProcessing")

    #--------------------------------------------------------------------------------------------

    final  = arrays['final_paramaters']      # note the key name as provided
    initial = arrays['initial_paramaters']
    sigma   = arrays['formal_errors']

    fig_formal_errors = plot_formal_errors(initial,final,sigma)
    fig_formal_errors.savefig(out_dir / "Formal_Errors.pdf")
    #--------------------------------------------------------------------------------------------

    fig_final_sim_SPICE_rsw = FigUtils.Residuals_RSW(diff_RSW, time_column,type="difference",title="RSW Difference Without vs With Weights")
    fig_final_sim_SPICE_rsw.savefig(out_dir / "RSW_Diff_with_without_weights.pdf")

    #--------------------------------------------------------------------------------------------

    fig, _ = plot_observation_analysis(df, file_colors=file_colors, title_suffix="All Data")
    fig.savefig(out_dir / "Colored_Count.pdf")
    # fig_rms = FigUtils.Residuals_RMS(residual_history_arcseconds)
    # fig_rms.savefig(out_dir / "Residual_RMS.pdf")

    #--------------------------------------------------------------------------------------------

    residual_df = calculate_mean_std(df)
    fig_first,fig_last,_ = plot_mean_std(residual_df,file_colors=file_colors,title_suffix = "")
    fig_first.savefig(out_dir / "Colored_mean_std_Initial.pdf")
    fig_last.savefig(out_dir / "Colored_mean_std_Last.pdf")

    fig, axes = overlay_residual_stats(df, file_colors)
    fig.savefig(out_dir / "Colored_mean_std_overlayed.pdf")


    print("Created and saved all other figures...")
    #--------------------------------------------------------------------------------------------

    fig_estimation_residuals,ax = plot_RA_DEC_residuals(df,df_initial, file_colors=file_colors, dataset_labels=("removed outliers", "with outliers"))
    fig_estimation_residuals.savefig(out_dir / "Residuals_time.pdf")

    fig_estimation_residuals_colored = plot_RA_DEC_residuals_colored(df, file_colors=file_colors, labels=["final", "initial"])
    fig_estimation_residuals_colored.savefig(out_dir / "Residuals_time_colored.pdf")

    #individual_figs = plot_individual_RA_DEC_residuals(df,file_colors) 
    #individual_figs_demeaned = plot_individual_RA_DEC_residuals_demeaned(df,file_colors)

    df_frames, summary = split_into_timeframes(df, gap_hours=4, time_col="times", make_datetime=True)

    individual_figs_timeframes =  plot_timeframe_residuals(df_frames,file_colors=file_colors)

    for file_id, fig in individual_figs_timeframes.items():
        fig.savefig(out_dir / (file_id[0] + "_" + str(file_id[1]) + '_residuals_time_timeframes.pdf'))


    # for file_id, fig in individual_figs_demeaned.items():
    #     fig.savefig(out_dir / (file_id + '_residuals_time_demeaned.pdf'))




    outliers = df_initial[~df_initial['times'].isin(df['times'])].copy()
    fig_outlier_residuals_colored = plot_RA_DEC_residuals_colored(outliers, file_colors=file_colors, labels=["final", "initial"])
    fig_outlier_residuals_colored.savefig(out_dir / "Residuals_time_colored_outliers.pdf")

    fig_estimation_residuals,ax = plot_RA_DEC_residuals(df,df_initial, file_colors=file_colors, dataset_labels=("weights", "no weights"))
    fig_estimation_residuals.savefig(out_dir / "Residuals_time_weights_no_weights.pdf")


    #--------------------------------------------------------------------------------------------
    # COMPUTE WEIGHTS
    #--------------------------------------------------------------------------------------------


    #Compute std in arcseconds
    std_per_id = df.groupby("id")[["residual_ra_last", "residual_dec_last"]].std().rename(
        columns={"residual_ra_last": "std_ra", "residual_dec_last": "std_dec"}
    )



    # Convert to radians
    arcsec_to_rad = np.pi / (180.0 * 3600.0)
    std_per_id_rad = std_per_id * arcsec_to_rad

    # Compute weights = 1 / σ²
    weights = 1.0 / (std_per_id_rad ** 2)
    weights.columns = ["weight_ra", "weight_dec"]

    # Save to file
    weights.to_csv(out_dir / "weights.txt", sep="\t", float_format="%.8e")


    #Compute weight per id and timeframe:
    # --- Compute per-frame std in arcseconds ---
    min_sigma_arcsec = 0.0   # e.g. 1e-6 if you want a floor

    sigma_ra = summary['residual_ra_last_std'].astype(float).copy()
    sigma_dec = summary['residual_dec_last_std'].astype(float).copy()

    if min_sigma_arcsec > 0:
        sigma_ra = sigma_ra.clip(lower=min_sigma_arcsec)
        sigma_dec = sigma_dec.clip(lower=min_sigma_arcsec)

    # arcsec -> rad
    arcsec_to_rad = np.pi / (180.0 * 3600.0)
    sigma_ra_rad  = sigma_ra * arcsec_to_rad
    sigma_dec_rad = sigma_dec * arcsec_to_rad

    # weights = 1 / σ²
    summary['weight_ra']  = 1.0 / (sigma_ra_rad ** 2)
    summary['weight_dec'] = 1.0 / (sigma_dec_rad ** 2)

    # Replace inf with NaN if any σ==0 survived
    summary.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Compute per-id mean weights (ignoring NaNs)
    mean_weights = (
            summary.groupby("id")[["weight_ra", "weight_dec"]]
            .transform(lambda x: x.fillna(x.mean()))
            )

    # Replace NaNs in the original dataframe with these means
    summary[["weight_ra", "weight_dec"]] = mean_weights

    summary.to_csv(out_dir / "summary.txt", sep="\t", float_format="%.8e")


    # Assign weights
    observations,observations_settings,observation_set_ids = ObsFunc.LoadObservations(
            "Observations/ProcessedOutliers/",
            system_of_bodies,file_names_loaded,
            weights = summary,
            timeframe_weights=True)

    min_time = float(row["start_sec"]-10)  # seconds since J2000
    max_time = float(row["end_sec"]+10)

    # Create parser for this time interval
    parser = estimation.observations.observations_processing.observation_parser(
        (min_time, max_time)
    )

    simulation_start_epoch = DateTime(1986, 1,  1).epoch() #2006, 8,  27 1963, 3,  4  
    simulation_end_epoch   = DateTime(2020, 1, 1).epoch()

    print("end")