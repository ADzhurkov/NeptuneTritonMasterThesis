import numpy as np
from pathlib import Path
from typing import Dict
from datetime import datetime, timedelta
import sys
import json
import pandas as pd
import matplotlib
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import matplotlib.cm as cm
from matplotlib.lines import Line2D

import pickle
import os 

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.dates as mdates

import math



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
                                 residual_history=None,best_iteration=None):
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
        If provided, adds residuals from first and best runs to DataFrame
    best_iteration : int, optional
        The run for which the residuals were the lowest
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
                
                # Best run (at best_iteration index )
                row['residual_ra_best'] = residual_history[best_iteration, time_idx, 1]
                row['residual_dec_best'] = residual_history[best_iteration, time_idx, 2]
            
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
    Create stacked histogram and bar chart of observations over time (by year).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with columns: 'id', 'observatory', 'times'
    file_colors : dict, optional
        Dictionary mapping file number to color
    title_suffix : str
        Additional text for plot titles
    
    Returns:
    --------
    tuple : (fig, file_colors)
    """
    file_numbers = sorted(df['id'].unique())
    
    if len(file_numbers) == 0:
        print("No data to plot!")
        return None, file_colors
    
    # Create color mapping if not provided
    if file_colors is None:
        file_colors = create_color_mapping(df)
    
    # Step 1: Assign year to each observation
    observations = []
    for _, row in df.iterrows():
        time_datetime = J2000_EPOCH + timedelta(seconds=float(row['times']))
        year = time_datetime.year
        observations.append({
            'id': row['id'],
            'year': year
        })
    
    obs_df = pd.DataFrame(observations)
    
    # Step 2: Count observations per year per file
    counts = obs_df.groupby(['year', 'id']).size().reset_index(name='count')
    
    # Step 3: Create all years from 1963 to 2025
    all_years = list(range(1963, 2026))
    
    # Step 4: Fill in missing years with zeros
    complete_data = []
    for year in all_years:
        for file_id in file_numbers:
            existing = counts[(counts['year'] == year) & (counts['id'] == file_id)]
            if len(existing) > 0:
                count_val = existing['count'].values[0]
            else:
                count_val = 0
            
            complete_data.append({
                'year': year,
                'id': file_id,
                'count': count_val
            })
    
    complete_df = pd.DataFrame(complete_data)
    
    # Step 5: Pivot to get years as rows, file ids as columns
    yearly_counts = complete_df.pivot(index='year', columns='id', values='count')
    
    # Count total observations per file for bottom plot
    id_counts = df.groupby('id').size().reset_index(name='n_observations')
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), height_ratios=[2, 1],
                                    gridspec_kw={'hspace': 0.3, 'left': 0.08, 
                                                 'right': 0.82, 'top': 0.95, 'bottom': 0.08})
    
    # Top plot: Stacked bar chart by year
    plot_colors = [file_colors[nr] for nr in yearly_counts.columns]
    yearly_counts.plot(kind='bar', stacked=True, ax=ax1, color=plot_colors, 
                       width=1.0, legend=False)
    
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Number of Observations', fontsize=12)
    ax1.set_title(f'Observation Count Over Time by File{title_suffix}', fontsize=14)
    
    # Format x-axis to show every few years
    all_tick_labels = list(yearly_counts.index)
    years_to_show = all_tick_labels[::5]  # Show every 5th year
    tick_positions = [i for i in range(len(all_tick_labels)) if all_tick_labels[i] in years_to_show]
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels([all_tick_labels[i] for i in tick_positions], rotation=45, ha='right')
    ax1.set_ylim(top=3000)
    # Legend
    handles = [Patch(facecolor=file_colors[nr], label=f'File {nr}') 
               for nr in file_numbers]
    ax1.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc='upper left',
              ncol=1, fontsize=8, frameon=True)
    
    # Bottom plot: Total observations per file
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
    'residual_ra_best': ['mean', 'std'],
    'residual_dec_best': ['mean', 'std']
    }).reset_index()

    # Flatten column names
    residual_stats.columns = ['id', 'ra_first_mean', 'ra_first_std', 
                            'dec_first_mean', 'dec_first_std',
                            'ra_last_mean', 'ra_last_std',
                            'dec_last_mean', 'dec_last_std']

    return residual_stats

def plot_mean_std(df, file_colors=None, title_suffix=""):
    """
    Create bar charts showing mean and std for RA and DEC per file.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with columns: id, ra_first_mean, ra_first_std, dec_first_mean, 
        dec_first_std, ra_last_mean, ra_last_std, dec_last_mean, dec_last_std
    file_colors : dict, optional
        Dictionary mapping file number to color. If None, creates new mapping.
    title_suffix : str
        Additional text to add to plot titles.
    
    Returns:
    --------
    fig1 : Figure of the initial propagation
    fig2 : Figure of the last propagation
    file_colors : dict - Color mapping used in the plot
    """
    # Use provided color mapping or create new one
    if file_colors is None:
        file_colors = create_color_mapping(df)
    
    # Get colors for each ID
    bar_colors = [file_colors[id] for id in df['id']]
    
    # Set up x-axis positions
    x = np.arange(len(df))
    width = 0.6
    
    # Create figure 1 for Initial propagation (2 rows x 2 columns)
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Mean RA per file (Initial)
    axes1[0, 0].bar(x, df['ra_first_mean'], width, alpha=0.8, color=bar_colors)
    axes1[0, 0].set_xlabel('File ID', fontsize=12)
    axes1[0, 0].set_ylabel('Mean RA Residual (arcsec)', fontsize=12)
    axes1[0, 0].set_title(f'Mean RA Residuals per File{title_suffix}', fontsize=14)
    axes1[0, 0].set_xticks(x)
    axes1[0, 0].set_xticklabels(df['id'], rotation=45, ha='right')
    axes1[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes1[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Std RA per file (Initial)
    axes1[0, 1].bar(x, df['ra_first_std'], width, alpha=0.8, color=bar_colors)
    axes1[0, 1].set_xlabel('File ID', fontsize=12)
    axes1[0, 1].set_ylabel('Std RA (arcsec)', fontsize=12)
    axes1[0, 1].set_title(f'Standard Deviation RA per File{title_suffix}', fontsize=14)
    axes1[0, 1].set_xticks(x)
    axes1[0, 1].set_xticklabels(df['id'], rotation=45, ha='right')
    axes1[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Mean DEC per file (Initial)
    axes1[1, 0].bar(x, df['dec_first_mean'], width, alpha=0.8, color=bar_colors)
    axes1[1, 0].set_xlabel('File ID', fontsize=12)
    axes1[1, 0].set_ylabel('Mean DEC Residual (arcsec)', fontsize=12)
    axes1[1, 0].set_title(f'Mean DEC Residuals per File{title_suffix}', fontsize=14)
    axes1[1, 0].set_xticks(x)
    axes1[1, 0].set_xticklabels(df['id'], rotation=45, ha='right')
    axes1[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes1[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Std DEC per file (Initial) - FIXED: was using ra_first_std instead of dec_first_std
    axes1[1, 1].bar(x, df['dec_first_std'], width, alpha=0.8, color=bar_colors)
    axes1[1, 1].set_xlabel('File ID', fontsize=12)
    axes1[1, 1].set_ylabel('Std DEC (arcsec)', fontsize=12)
    axes1[1, 1].set_title(f'Standard Deviation DEC per File{title_suffix}', fontsize=14)
    axes1[1, 1].set_xticks(x)
    axes1[1, 1].set_xticklabels(df['id'], rotation=45, ha='right')
    axes1[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Create figure 2 for Last propagation (2 rows x 2 columns)
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Mean RA per file (Last)
    axes2[0, 0].bar(x, df['ra_last_mean'], width, alpha=0.8, color=bar_colors)
    axes2[0, 0].set_xlabel('File ID', fontsize=12)
    axes2[0, 0].set_ylabel('Mean RA Residual (arcsec)', fontsize=12)
    axes2[0, 0].set_title(f'Mean RA Residuals per File{title_suffix}', fontsize=14)
    axes2[0, 0].set_xticks(x)
    axes2[0, 0].set_xticklabels(df['id'], rotation=45, ha='right')
    axes2[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes2[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Std RA per file (Last)
    axes2[0, 1].bar(x, df['ra_last_std'], width, alpha=0.8, color=bar_colors)
    axes2[0, 1].set_xlabel('File ID', fontsize=12)
    axes2[0, 1].set_ylabel('Std RA (arcsec)', fontsize=12)
    axes2[0, 1].set_title(f'Standard Deviation RA per File{title_suffix}', fontsize=14)
    axes2[0, 1].set_xticks(x)
    axes2[0, 1].set_xticklabels(df['id'], rotation=45, ha='right')
    axes2[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Mean DEC per file (Last)
    axes2[1, 0].bar(x, df['dec_last_mean'], width, alpha=0.8, color=bar_colors)
    axes2[1, 0].set_xlabel('File ID', fontsize=12)
    axes2[1, 0].set_ylabel('Mean DEC Residual (arcsec)', fontsize=12)
    axes2[1, 0].set_title(f'Mean DEC Residuals per File{title_suffix}', fontsize=14)
    axes2[1, 0].set_xticks(x)
    axes2[1, 0].set_xticklabels(df['id'], rotation=45, ha='right')
    axes2[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes2[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Std DEC per file (Last) - FIXED: was using ra_last_std instead of dec_last_std
    axes2[1, 1].bar(x, df['dec_last_std'], width, alpha=0.8, color=bar_colors)
    axes2[1, 1].set_xlabel('File ID', fontsize=12)
    axes2[1, 1].set_ylabel('Std DEC (arcsec)', fontsize=12)
    axes2[1, 1].set_title(f'Standard Deviation DEC per File{title_suffix}', fontsize=14)
    axes2[1, 1].set_xticks(x)
    axes2[1, 1].set_xticklabels(df['id'], rotation=45, ha='right')
    axes2[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    return fig1, fig2, file_colors

def overlay_residual_stats(df, file_colors, fig=None, axes=None, 
                          label='Final', alpha=0.5, edgecolor='black', 
                          linewidth=1.5, hatch='///'):
    """
    Overlay residual statistics on existing or new figure.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns: 'id', 'residual_ra_first', 'residual_dec_first',
        'residual_ra_best', 'residual_dec_best'
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
        'residual_ra_best': ['mean', 'std'],
        'residual_dec_best': ['mean', 'std']
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
        'residual_ra_best', 'residual_dec_best'
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
        ax[0].scatter(group["datetime"], group["residual_ra_best"],
                      color=color, alpha=0.6, s=10, label=labels[0])
        #ax[0].scatter(group["datetime"], group["residual_ra_first"],
        #              color=color, alpha=0.3, s=10, marker='x', label=labels[1])

        # --- DEC plot ---
        ax[1].scatter(group["datetime"], group["residual_dec_best"],
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
        'residual_ra_best', 'residual_dec_best'
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
        _ax[0].scatter(_df["datetime"], _df["residual_ra_best"],
                       alpha=alpha_main, s=10, label=f"{ds_label} ({labels[0]})")
        _ax[1].scatter(_df["datetime"], _df["residual_dec_best"],
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
                 'residual_ra_best','residual_dec_best'
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
        ax[0].scatter(df_id['datetime'], df_id['residual_ra_best'],
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
        ax[1].scatter(df_id['datetime'], df_id['residual_dec_best'],
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

def plot_formal_errors(initial, final, sigma, title="Difference wrt NEP097"):
    pos_diff, vel_diff = (final[:3] - initial[:3])/1e3, final[3:] - initial[3:]
    pos_err, vel_err = sigma[:3]/1e3, sigma[3:]
    labels = ['X', 'Y', 'Z']

    fig, axes = plt.subplots(1, 2, figsize=(6, 7), sharex=False)
    fig.suptitle(title, fontsize=14, y=0.98)

    def plot_single_axis(ax, diffs, errs, ylabel):
        for i, label in enumerate(labels):
            # horizontal offset so columns don't overlap
            x = i  
            ax.errorbar(x, diffs[i], yerr=errs[i], fmt='s', color='C1', capsize=4, label=None)
            ax.axhline(y=0,alpha=0.9,c='black',linestyle='--')
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
    plot_single_axis(axes[0], pos_diff, pos_err, 'Δ km')
    axes[0].set_title('Difference in Position')

    # Velocity (right)
    plot_single_axis(axes[1], vel_diff, vel_err, 'Δ m/s')
    axes[1].set_title('Difference in Velocity')

    # # One legend for both
    # handles = [
    #     #plt.Line2D([], [], marker='o', color='C0', linestyle='None', label='Initial (0)'),
    #     plt.Line2D([], [], marker='s', color='C1', linestyle='None', label='Estimation ±σ')
    # ]
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
        #plt.Line2D([], [], marker='o', color='C0', linestyle='None', label='Initial (0)'),
        plt.Line2D([], [], marker='s', color='C1', linestyle='None', label='Estimation ±σ'),
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
    time_col: str = "times",
    min_obs_per_frame: int = 1,
    summary_cols: list = None
):
    """
    Split observations into timeframes based on time gaps.
    
    Parameters:
    -----------
    df : DataFrame with columns including 'id' and time_col
    gap_hours : Break into new timeframe if gap exceeds this (hours)
    time_col : Column name containing time in seconds since J2000
    min_obs_per_frame : Minimum observations before allowing a break
    summary_cols : Columns to calculate mean/std for each timeframe
    
    Returns:
    --------
    df_with_frames : Original df with 'timeframe' column added
    summary : Summary statistics per (id, timeframe)
    """
    
    if summary_cols is None:
        summary_cols = ["residual_ra_best", "residual_dec_best"]
    
    gap_seconds = gap_hours * 3600.0
    
    # Sort by id and time
    df_sorted = df.copy().sort_values(["id", time_col]).reset_index(drop=True)
    
    def assign_timeframes(group):
        """Assign timeframe number to each observation in the group."""
        times = group[time_col].values
        n = len(times)
        
        timeframes = np.zeros(n, dtype=int)
        
        current_frame = 0
        obs_in_current_frame = 1
        
        for i in range(1, n):
            time_gap = times[i] - times[i-1]
            
            # Should we start a new frame?
            # Yes, if gap is large
            if time_gap >= gap_seconds and obs_in_current_frame >= min_obs_per_frame:
                current_frame += 1
                obs_in_current_frame = 1
            else:
                obs_in_current_frame += 1
            
            timeframes[i] = current_frame
        
        return pd.Series(timeframes, index=group.index)
    
    # Apply to each id separately
    df_sorted["timeframe"] = df_sorted.groupby("id", group_keys=False).apply(assign_timeframes)
    
    # Create summary for each (id, timeframe) combination
    summary = create_summary(df_sorted, time_col, summary_cols)
    
    return df_sorted, summary

def create_summary(df, time_col, summary_cols):
    """Create summary statistics for each (id, timeframe)."""
    
    # Basic statistics
    summary = df.groupby(["id", "timeframe"]).agg(
        start_sec=(time_col, "min"),
        end_sec=(time_col, "max"),
        n_obs=(time_col, "count")
    ).reset_index()
    
    # Calculate duration
    summary["duration_hours"] = (summary["end_sec"] - summary["start_sec"]) / 3600.0
    
    # Add mean and std for requested columns
    for col in summary_cols:
        if col in df.columns:
            stats = (
                    df.groupby(["id", "timeframe"])[col]
                            .agg(
                            mean="mean",
                            std="std",
                            rms=lambda s: np.sqrt(np.mean(s**2))
                        )
                        .reset_index())

            stats.columns = [
                    "id",
                    "timeframe",
                    f"{col}_mean",
                    f"{col}_std",
                    f"{col}_rms",]

            summary = summary.merge(stats, on=["id", "timeframe"], how="left")
    
    # Add datetime columns
    J2000_EPOCH = datetime(2000, 1, 1, 12, 0, 0)
    summary["start_dt"] = summary["start_sec"].apply(
        lambda s: J2000_EPOCH + timedelta(seconds=s)
    )
    summary["end_dt"] = summary["end_sec"].apply(
        lambda s: J2000_EPOCH + timedelta(seconds=s)
    )
    print("kur")
    summary.rename(columns={'residual_ra_best_mean': 'residual_ra_mean'}, inplace=True)
    summary.rename(columns={'residual_ra_best_std': 'residual_ra_std'}, inplace=True)
    summary.rename(columns={'residual_ra_best_rms': 'residual_ra_rms'}, inplace=True)
    summary.rename(columns={'residual_dec_best_mean': 'residual_dec_mean'}, inplace=True)
    summary.rename(columns={'residual_dec_best_std': 'residual_dec_std'}, inplace=True)
    summary.rename(columns={'residual_dec_best_rms': 'residual_dec_rms'}, inplace=True)
    return summary

def plot_average_weight(summary_or_weights: pd.DataFrame, *, split_by_id: bool = False,
            selected_key = None,logscale=True,title=None,label_y=None):
    """
    Takes:
        summary_or_weights: pandas DataFrame either summary or weight like. Contains per timeframe or per id weights/stds
        split_by_id: Default = False, plots either all in one figure or one fig per file id
        selected_key: Default = None, select a different key from the dataframe to plot (usually count)

    Flexible plotter:
      • If time info exists (start_dt/end_dt or start_sec/end_sec):
          -> plots average weight ((RA+DEC)/2) at per-frame midpoints (time series).
      • Else (only per-id weights):
          -> plots bar chart of average weight per id.

    Returns:
      - dict of {id: Figure} if split_by_id=True (time-series mode),
      - a single Figure otherwise.

    Accepted inputs:
      - Columns required: 'weight_ra', 'weight_dec'
      - Optional time columns (choose one set):
          * 'start_dt' & 'end_dt'  (datetime64)
          * 'start_sec' & 'end_sec' (float seconds since J2000)
      - Optional 'id' column OR index named 'id'; if missing, index is used.
    """
    df = summary_or_weights.copy()

    # Normalize id
    if "id" not in df.columns:
        if df.index.name == "id":
            df = df.reset_index()
        else:
            df = df.reset_index().rename(columns={"index": "id"})

    # Basic checks
    need = {"id", "weight_rmse_ra", "weight_rmse_dec","mean_weight_rmse"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")

    # Compute average weight
    #df["weight_mean"] = (df["weight_rmse_ra"] + df["weight_rmse_dec"]) / 2.0

    has_dt = {"start_dt", "end_dt"} <= set(df.columns)
    has_sec = {"start_sec", "end_sec"} <= set(df.columns)


    # ---- TIME-SERIES MODE (summary-like input) ----
    if has_dt or has_sec:
        # Midpoint datetime
        if has_dt:
            df["mid_dt"] = df["start_dt"] + (df["end_dt"] - df["start_dt"]) / 2
        else:
            J2000 = datetime(2000, 1, 1, 12, 0, 0)
            start_dt = df["start_sec"].astype(float).map(lambda s: J2000 + timedelta(seconds=s))
            delta = (df["end_sec"] - df["start_sec"]).astype(float).map(lambda s: timedelta(seconds=s/2))
            df["mid_dt"] = start_dt + delta

        if split_by_id:
            figs = {}
            for file_id, g in df.groupby("id", sort=False):
                fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
                g = g.sort_values("mid_dt")

                if selected_key is not None:
                    ax.plot(g["mid_dt"], g[selected_key], marker="o", linestyle="-", alpha=0.8)
                else:
                    ax.plot(g["mid_dt"], g["mean_weight_rmse"], marker="o", linestyle="-", alpha=0.8)
                
                if logscale == True:  
                    ax.set_yscale("log")
   
                if label_y == None:
                    ax.set_ylabel("Average Weight [1/rad²]")
                else:
                    ax.set_ylabel(label_y)
            
                if title is None:
                    ax.set_title(f"Average weight vs time — {file_id}")
                else:
                    ax.set_title(title + f" {file_id}")
                
                ax.set_xlabel("Time")
                ax.grid(True, alpha=0.25)
                figs[file_id] = fig
            return figs
        else:
            fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
            
            for file_id, g in df.groupby("id", sort=False):
                
                g = g.sort_values("mid_dt")

                if selected_key is not None:
                    ax.plot(g["mid_dt"], g[selected_key], marker="o", linestyle="-", alpha=0.8, label=str(file_id))
                else:
                    ax.plot(g["mid_dt"], g["mean_weight_rmse"], marker="o", linestyle="-", alpha=0.8, label=str(file_id))
                    
            if logscale == True:  
                ax.set_yscale("log")
   
            ax.set_xlabel("Time")
            if label_y == None:
                ax.set_ylabel("Average Weight [1/rad²]")
            else:
                ax.set_ylabel(label_y)
            
            if title is None:
                ax.set_title("Average weight vs time (per-frame midpoint)")
            else:
                ax.set_title(title)
                #ax.set_ylabel("Number of observations [-]")
            ax.legend(title="ID", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
            ax.grid(True, alpha=0.25)
            return fig

    # ---- PER-ID BAR MODE (simple weights input) ----
    # Aggregate if there are multiple rows per id (e.g., multiple frames without time info)
    agg = df.groupby("id", sort=False)["weight_mean"].mean().reset_index()
    # Preserve RA/DEC bars too if present uniquely per id; otherwise average them as well
    ra = df.groupby("id", sort=False)["weight_ra"].mean().reset_index()
    de = df.groupby("id", sort=False)["weight_dec"].mean().reset_index()
    m = agg.merge(ra, on="id").merge(de, on="id")

    ids = m["id"].astype(str).to_numpy()
    x = np.arange(len(ids), dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    width = 0.36
    ax.bar(x - width/2, m["weight_ra"].to_numpy(), width, label="RA", alpha=0.85)
    ax.bar(x + width/2, m["weight_dec"].to_numpy(), width, label="DEC", alpha=0.85)

    ax.set_yscale("log")
    ax.set_xticks(x, ids, rotation=0)
    ax.set_xlim(-0.75, (x.max() + 0.75) if x.size else 1)
    ax.set_xlabel("ID")
    ax.set_ylabel("Weight [1/rad²]")
    ax.set_title("Average weight per ID")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    return fig

def plot_average_weight_vs_time_from_weights(
    weights: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    split_by_id: bool = False
):
    """
    Use per-ID weights (weights) + per-frame times (summary) to make
    the SAME time-series plot as the summary-based version.

    weights: DataFrame with columns ['id','weight_ra','weight_dec'] or index 'id'
    summary: DataFrame with frame rows and either:
             ['start_dt','end_dt'] OR ['start_sec','end_sec'], plus 'id'

    Returns:
      dict[id, Figure] if split_by_id=True, else a single Figure.
    """
    # ---- normalize inputs ----
    w = weights.copy()
    if "id" not in w.columns:
        if w.index.name == "id":
            w = w.reset_index()
        else:
            w = w.reset_index().rename(columns={"index": "id"})
    need_w = {"id", "weight_ra", "weight_dec"}
    miss_w = need_w - set(w.columns)
    if miss_w:
        raise ValueError(f"'weights' missing columns: {sorted(miss_w)}")

    s = summary.copy()
    if "id" not in s.columns:
        if s.index.name == "id":
            s = s.reset_index()
        else:
            s = s.reset_index().rename(columns={"index": "id"})

    # Keep only time + id from summary
    cols_keep = ["id"]
    if {"start_dt", "end_dt"} <= set(s.columns):
        cols_keep += ["start_dt", "end_dt"]
        has_dt = True
    elif {"start_sec", "end_sec"} <= set(s.columns):
        cols_keep += ["start_sec", "end_sec"]
        has_dt = False
    else:
        raise ValueError("summary must have start_dt/end_dt or start_sec/end_sec")

    s = s[cols_keep].copy()

    # ---- broadcast weights to each frame via merge ----
    df = s.merge(w[["id", "weight_ra", "weight_dec"]], on="id", how="inner")
    if df.empty:
        raise ValueError("No overlapping ids between weights and summary.")

    # ---- compute frame midpoints ----
    if has_dt:
        df["mid_dt"] = df["start_dt"] + (df["end_dt"] - df["start_dt"]) / 2
    else:
        J2000 = datetime(2000, 1, 1, 12, 0, 0)
        start_dt = df["start_sec"].astype(float).map(lambda s: J2000 + timedelta(seconds=s))
        half_dt  = (df["end_sec"] - df["start_sec"]).astype(float).map(lambda s: timedelta(seconds=s/2))
        df["mid_dt"] = start_dt + half_dt

    # ---- average weight ----
    df["weight_mean"] = (df["weight_ra"] + df["weight_dec"]) / 2.0

    # ---- plotting (same style as your original) ----
    if split_by_id:
        figs = {}
        for file_id, g in df.groupby("id", sort=False):
            fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
            g = g.sort_values("mid_dt")
            ax.plot(g["mid_dt"], g["weight_mean"], marker="o", linestyle="-", alpha=0.8)
            ax.set_yscale("log")
            ax.set_xlabel("Time")
            ax.set_ylabel("Average Weight [1/rad²]")
            ax.set_title(f"Average weight vs time — {file_id}")
            ax.grid(True, alpha=0.25)
            figs[file_id] = fig
        return figs
    else:
        fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
        for file_id, g in df.groupby("id", sort=False):
            g = g.sort_values("mid_dt")
            ax.plot(g["mid_dt"], g["weight_mean"], marker="o", linestyle="-", alpha=0.8, label=str(file_id))
        ax.set_yscale("log")
        ax.set_xlabel("Time")
        ax.set_ylabel("Average Weight [1/rad²]")
        ax.set_title("Average weight vs time (per-frame midpoint)")
        ax.legend(title="ID", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
        ax.grid(True, alpha=0.25)
        return fig

def plot_residual_std(summary: pd.DataFrame, split_by_id: bool = False):
    """
    Plot standard deviations of residuals over time.
    
    Parameters:
    -----------
    summary : pd.DataFrame
        Must contain:
        - 'id': observation file identifier
        - 'residual_ra_std': RA standard deviation
        - 'residual_dec_std': DEC standard deviation
        - Time columns (one of):
            * 'start_dt' & 'end_dt' (datetime)
            * 'start_sec' & 'end_sec' (seconds since J2000)
    
    split_by_id : bool
        If True, creates separate figure for each ID
        If False, plots all IDs on one figure
    
    Returns:
    --------
    dict of {id: Figure} if split_by_id=True
    single Figure if split_by_id=False
    """
    
    df = summary.copy()
    
    # Check required columns
    required = {'id', 'residual_ra_std', 'residual_dec_std'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Calculate midpoint times
    df = add_midpoint_time(df)
    
    # Sort by time
    df = df.sort_values(['id', 'mid_dt'])
    
    # Create plots
    if split_by_id:
        return plot_separate_figures(df)
    else:
        return plot_combined_figure(df)

def add_midpoint_time(df):
    """Add midpoint datetime column."""
    
    # Check which time columns exist
    has_datetime = 'start_dt' in df.columns and 'end_dt' in df.columns
    has_seconds = 'start_sec' in df.columns and 'end_sec' in df.columns
    
    if has_datetime:
        # Calculate midpoint from datetime columns
        df['mid_dt'] = df['start_dt'] + (df['end_dt'] - df['start_dt']) / 2
    
    elif has_seconds:
        # Convert seconds to datetime and calculate midpoint
        J2000 = datetime(2000, 1, 1, 12, 0, 0)
        df['mid_dt'] = df.apply(
            lambda row: J2000 + timedelta(seconds=(row['start_sec'] + row['end_sec']) / 2),
            axis=1
        )
    
    else:
        raise ValueError("Need either (start_dt, end_dt) or (start_sec, end_sec) columns")
    
    return df

def plot_separate_figures(df):
    """Create one figure per ID."""
    
    figs = {}
    
    for file_id in df['id'].unique():
        df_id = df[df['id'] == file_id]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), 
                                       sharex=True, constrained_layout=True)
        
        # RA std
        ax1.plot(df_id['mid_dt'], df_id['residual_ra_std'], 
                marker='o', linestyle='-', color='C0', alpha=0.7)
        ax1.set_ylabel('RA Std [arcsec]', fontsize=11)
        ax1.set_title(f'Residual Standard Deviations — {file_id}', fontsize=13)
        ax1.grid(True, alpha=0.3)
        
        # DEC std
        ax2.plot(df_id['mid_dt'], df_id['residual_dec_std'], 
                marker='o', linestyle='-', color='C1', alpha=0.7)
        ax2.set_xlabel('Time', fontsize=11)
        ax2.set_ylabel('DEC Std [arcsec]', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        figs[file_id] = fig
    
    return figs

def plot_combined_figure(df):
    """Create one figure with all IDs."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                   sharex=True, constrained_layout=True)
    
    for file_id in df['id'].unique():
        df_id = df[df['id'] == file_id]
        
        # RA std
        ax1.plot(df_id['mid_dt'], df_id['residual_ra_std'], 
                marker='o', linestyle='-', alpha=0.7, label=str(file_id))
       
        # DEC std
        ax2.plot(df_id['mid_dt'], df_id['residual_dec_std'], 
                marker='o', linestyle='-', alpha=0.7, label=str(file_id))
    
    ax1.set_yscale('log')
    ax1.set_ylabel('RA Std [arcsec]', fontsize=11)
    ax1.set_title('Residual Standard Deviations Over Time', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(title='ID', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    ax2.set_yscale('log')
    ax2.set_xlabel('Time', fontsize=11)
    ax2.set_ylabel('DEC Std [arcsec]', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(title='ID', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    return fig

def plot_residuals(df, summary=None, plot_individual_frames=False):
    """
    Plot RA/DEC residuals for each observation ID with per-timeframe subplots.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain: id, datetime, residual_ra_best, residual_dec_best, timeframe
    summary : pd.DataFrame, optional
        Summary statistics with: id, timeframe, start_dt, residual_ra_std, residual_dec_std
    plot_individual_frames : bool, optional
        If True, creates individual subplots for each timeframe (default: False)
    
    Returns
    -------
    dict
        {id: figure} for each unique ID
    """
    figs = {}

    if 'datetime' not in df.keys():
        J2000_EPOCH = datetime(2000, 1, 1, 12, 0, 0)
        df['datetime'] = df['times'].astype(float).apply(
            lambda t: J2000_EPOCH + timedelta(seconds=t))

    for obs_id in df['id'].unique():
        # Get data for this ID
        data = df[df['id'] == obs_id].sort_values('datetime')
        
        # Get timeframe info
        timeframes = data.groupby('timeframe')
        n_frames = len(timeframes)
        colors = cm.get_cmap('Pastel1', n_frames)
        
        # Calculate overall statistics for this ID
        overall_ra_std = data['residual_ra_best'].std()
        overall_dec_std = data['residual_dec_best'].std()
        
        # Determine number of rows: main + summary (if provided) + per-timeframe (if enabled)
        has_summary = summary is not None and obs_id in summary['id'].values
        n_rows = 1 + (1 if has_summary else 0) + (n_frames if plot_individual_frames else 0)
        
        # Create figure
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 + 3*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Main plots (top row)
        ax1, ax2 = axes[0]
        
        # Plot RA residuals
        ax1.scatter(data['datetime'], data['residual_ra_best'], s=30, alpha=0.9)
        ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('RA Residual [arcsec]')
        ax1.set_title(f'RA Residuals: {obs_id}')
        ax1.grid(alpha=0.5)
        
        # Plot DEC residuals
        ax2.scatter(data['datetime'], data['residual_dec_best'], s=30, alpha=0.9)
        ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('DEC Residual [arcsec]')
        ax2.set_title(f'DEC Residuals: {obs_id}')
        ax2.grid(alpha=0.5)
        
        # Add vertical lines at timeframe midpoints
        legend_items = []
        current_row = 1
        
        for i, (tf_id, tf_data) in enumerate(timeframes):
            start = tf_data['datetime'].min()
            end = tf_data['datetime'].max()
            midpoint = start + (end - start) / 2
            color = 'black'
            
            # Draw vertical line at midpoint
            ax1.axvline(midpoint, color=color, linestyle='-', linewidth=2, alpha=0.3)
            ax2.axvline(midpoint, color=color, linestyle='-', linewidth=2, alpha=0.3)
            
            # # Create legend entry
            # legend_items.append((
            #     Line2D([0], [0], color=color, linewidth=2, alpha=0.7),
            #     f'Frame {tf_id} (n={len(tf_data)})'
            # ))
        
        # Add legend to main plots
        #handles, labels = zip(*legend_items)
        #ax2.legend(handles, labels, loc='upper right')
        
        # Plot summary statistics if provided
        if has_summary:
            sum_data = summary[summary['id'] == obs_id].sort_values('start_dt')
            
            ax_sum_ra = axes[current_row, 0]
            ax_sum_dec = axes[current_row, 1]
            
            # RA std over time
            ax_sum_ra.plot(sum_data['start_dt'], sum_data['residual_ra_std'], 
                          marker='o', linewidth=2, markersize=8)
            ax_sum_ra.axhline(overall_ra_std, color='red', linestyle='--', 
                             linewidth=2, alpha=0.7, label=f'Overall Std: {overall_ra_std:.4f}"')
            ax_sum_ra.set_xlabel('Date')
            ax_sum_ra.set_ylabel('RA Residual Std [arcsec]')
            ax_sum_ra.set_title(f'RA Standard Deviation per Timeframe')
            ax_sum_ra.grid(alpha=0.5)
            #ax_sum_ra.legend()
            
            # DEC std over time
            ax_sum_dec.plot(sum_data['start_dt'], sum_data['residual_dec_std'], 
                           marker='o', linewidth=2, markersize=8)
            ax_sum_dec.axhline(overall_dec_std, color='red', linestyle='--', 
                              linewidth=2, alpha=0.7, label=f'Overall Std: {overall_dec_std:.4f}"')
            ax_sum_dec.set_xlabel('Date')
            ax_sum_dec.set_ylabel('DEC Residual Std [arcsec]')
            ax_sum_dec.set_title(f'DEC Standard Deviation per Timeframe')
            ax_sum_dec.grid(alpha=0.5)
            #ax_sum_dec.legend()
            
            # Match x-axis with main plots
            ax_sum_ra.set_xlim(ax1.get_xlim())
            ax_sum_dec.set_xlim(ax2.get_xlim())
            
            # Add vertical lines at timeframe midpoints
            for i, (tf_id, tf_data) in enumerate(timeframes):
                start = tf_data['datetime'].min()
                end = tf_data['datetime'].max()
                midpoint = start + (end - start) / 2
                color = 'black'
                
                ax_sum_ra.axvline(midpoint, color=color, linestyle='-', linewidth=2, alpha=0.3)
                ax_sum_dec.axvline(midpoint, color=color, linestyle='-', linewidth=2, alpha=0.3)
            
            # Format dates
            ax_sum_ra.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
            ax_sum_dec.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
            
            current_row += 1
        
        # Plot individual timeframes if enabled
        if plot_individual_frames:
            for i, (tf_id, tf_data) in enumerate(timeframes):
                start = tf_data['datetime'].min()
                end = tf_data['datetime'].max()
                color = colors(i)
                
                # Plot individual timeframe in row below
                row_idx = current_row + i
                ax_ra = axes[row_idx, 0]
                ax_dec = axes[row_idx, 1]
                
                # Calculate duration
                duration = end - start
                
                # RA subplot for this timeframe
                ax_ra.scatter(tf_data['datetime'], tf_data['residual_ra_best'], 
                             s=30, alpha=0.9, color=color)
                ax_ra.axhline(0, color='black', linestyle='--', alpha=0.5)
                ax_ra.set_xlabel('Date')
                ax_ra.set_ylabel('RA Residual [arcsec]')
                ax_ra.set_title(f'Frame {tf_id} - RA (n={len(tf_data)}, duration: {duration})')
                ax_ra.grid(alpha=0.5)
                
                # DEC subplot for this timeframe
                ax_dec.scatter(tf_data['datetime'], tf_data['residual_dec_best'], 
                              s=30, alpha=0.9, color=color)
                ax_dec.axhline(0, color='black', linestyle='--', alpha=0.5)
                ax_dec.set_xlabel('Date')
                ax_dec.set_ylabel('DEC Residual [arcsec]')
                ax_dec.set_title(f'Frame {tf_id} - DEC (n={len(tf_data)}, duration: {duration})')
                ax_dec.grid(alpha=0.5)
                
                # Format dates
                ax_ra.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
                ax_dec.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
        
        # Format dates on main plots
        ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
        ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
        
        fig.tight_layout()
        figs[obs_id] = fig
    
    return figs

def main(file_names_loaded,simulation_path,min_sigma_arcsec = 0.01  ):
    """
    A function to create a 'summary' pandas dataframe with multiple types of observational weights.
    
    Parameters
    ----------
    file_names_loaded : list of str
        A list of observation file names 
    simulation_path : str 
        Path of the simulation from which the weights are generated
    min_sigma_arcsec : float (optional)
        A minimum value for the estimated 'accuracy' of an observation in arcseconds (set to 10mas)
    
    Returns
    -------
    pd.DataFrame
        Includes many things that need to be documented....

    """

    #--------------------------------------------------------------------------------------------
    # REQUIRES A .json OR LIST WITH THE LOADED OBSERVATIONS FILES
    #--------------------------------------------------------------------------------------------

    # with open("file_names.json", "r") as f:
    #     file_names_loaded = json.load(f)

    # Extract IDs from filenames (remove 'Triton_' prefix and '.csv' suffix)
    ordered_ids = [f.replace('Triton_', '').replace('.csv', '') for f in file_names_loaded]



    #--------------------------------------------------------------------------------------------
    # LOAD ESTIMATION RESULTS
    #--------------------------------------------------------------------------------------------
    #"Results/PoleEstimationRealObservations/Hybrid_Weights/initial_state_only_Jacobson2009"
    arrays = load_npy_files(simulation_path) 
    
    print("Loaded numpy arrays from estimation...")

    df = create_observations_dataframe(
        arrays['observations_sorted'], 
        ordered_ids,
        arrays['observation_times'],
        arrays['residual_history_arcseconds'],
        arrays['best_iteration'])
    

    #Create timeframes 
    df_frames, summary = split_into_timeframes(df, gap_hours=4)



    #--------------------------------------------------------------------------------------------
    # COMPUTE ID WEIGHTS
    #--------------------------------------------------------------------------------------------
    print("Computing ID weights...")

    #Compute std in arcseconds
    std_per_id = df.groupby("id")[["residual_ra_best", "residual_dec_best"]].std().rename(
        columns={"residual_ra_best": "std_ra", "residual_dec_best": "std_dec"}
    )

    # Compute RMSE in arcseconds
    rmse_per_id = (
        df.groupby("id")[["residual_ra_best", "residual_dec_best"]]
        .apply(lambda g: np.sqrt((g**2).mean()))
        .rename(columns={"residual_ra_best": "rmse_ra", "residual_dec_best": "rmse_dec"})
    )


    # Files with 1 Id (NaN stds) have an STD assigned as 1 arcsecond
    std_per_id = std_per_id.fillna(1)

    # Convert to radians
    arcsec_to_rad = np.pi / (180.0 * 3600.0)
    std_per_id_rad = std_per_id * arcsec_to_rad
    rmse_per_id_rad = rmse_per_id * arcsec_to_rad

    # Compute weights = 1 / σ² std or rmse
    weights_std = 1.0 / (std_per_id_rad ** 2)
    weights_std.columns = ["weight_ra", "weight_dec"]

    weights_rmse = 1.0 / (rmse_per_id_rad ** 2)
    weights_rmse.columns = ["weight_ra", "weight_dec"]


    # Don't save in this function
    # Save to file
    # weights_std.to_csv(out_dir / "weights_per_id_std.txt", sep="\t", float_format="%.8e")
    # weights_rmse.to_csv(out_dir / "weights_per_id_rmse.txt", sep="\t", float_format="%.8e")
    

    #--------------------------------------------------------------------------------------------
    # COMPUTE TIMEFRAME WEIGHTS
    #--------------------------------------------------------------------------------------------
    replacement_for_id = {}
    usage_stats = {}

    DEFAULT = 1  # fallback if an id has no valid stds

    # Replace STD NaNs with appropriate values 
    for id, g in summary.groupby("id"):
        total_obs = g["n_obs"].sum()

        # masks for valid stds
        ra_mask = g["residual_ra_std"].notna()
        dec_mask = g["residual_dec_std"].notna()

        # weighted sums and totals (ignore NaNs)
        ra_weight_sum = (g.loc[ra_mask, "residual_ra_std"] * g.loc[ra_mask, "n_obs"]).sum()
        ra_n_used = g.loc[ra_mask, "n_obs"].sum()
        dec_weight_sum = (g.loc[dec_mask, "residual_dec_std"] * g.loc[dec_mask, "n_obs"]).sum()
        dec_n_used = g.loc[dec_mask, "n_obs"].sum()

        # weighted means (fallback to weights DataFrame if nothing valid)
        ra_mean = ra_weight_sum / ra_n_used if ra_n_used > 0 else std_per_id.loc[id,'std_ra']
        dec_mean = dec_weight_sum / dec_n_used if dec_n_used > 0 else std_per_id.loc[id,'std_dec']

        replacement_for_id[id] = {
            "residual_ra_std": float(ra_mean),
            "residual_dec_std": float(dec_mean),
        }

        usage_stats[id] = {
            "total_obs": int(total_obs),
            "ra_used": int(ra_n_used),
            "ra_excluded": int(total_obs - ra_n_used),
            "dec_used": int(dec_n_used),
            "dec_excluded": int(total_obs - dec_n_used),
        }

    # Useful stats to check if replacement of NaNs makes sense
    usage_df = pd.DataFrame.from_dict(usage_stats, orient='index')


    def fill_two_cols(group):
        rep = replacement_for_id[group.name]
        group[["residual_ra_std","residual_dec_std"]] = (
            group[["residual_ra_std","residual_dec_std"]].fillna(rep)
        )
        return group


    summary = summary.groupby("id", group_keys=False).apply(fill_two_cols)

    #--------------------------------------------------------------------------------------------
    # Created tabulated weights
    #--------------------------------------------------------------------------------------------
    
    #Add rmse and std per id to summary
    summary = summary.merge(
    rmse_per_id.rename(columns={'rmse_ra': 'rmse_ra_id', 'rmse_dec': 'rmse_dec_id'}),
    on='id',
    how='left')

    summary = summary.merge(
    std_per_id.rename(columns={'std_ra': 'std_ra_id', 'std_dec': 'std_dec_id'}),
    on='id',
    how='left')

    # Add ID weights to summary
    summary = summary.merge(
    weights_rmse.rename(columns={'weight_ra': 'weight_rmse_ra_id', 'weight_dec': 'weight_rmse_dec_id'}),
    on='id',
    how='left')

    summary = summary.merge(
    weights_std.rename(columns={'weight_ra': 'weight_std_ra_id', 'weight_dec': 'weight_std_dec_id'}),
    on='id',
    how='left')

    # Minimum accuracy
    #min_sigma_arcsec = 0.005   # 5mas minimum

    
    sigma_ra = summary['residual_ra_std'].astype(float).copy()
    sigma_dec = summary['residual_dec_std'].astype(float).copy()

    rmse_ra = summary['residual_ra_rms'].astype(float).copy()
    rmse_dec = summary['residual_dec_rms'].astype(float).copy()
    
    if min_sigma_arcsec > 0:
        sigma_ra = sigma_ra.clip(lower=min_sigma_arcsec)
        sigma_dec = sigma_dec.clip(lower=min_sigma_arcsec)
        
        rmse_ra = rmse_ra.clip(lower=min_sigma_arcsec)
        rmse_dec = rmse_dec.clip(lower=min_sigma_arcsec)
        

    summary['residual_ra_std'] = sigma_ra
    summary['residual_dec_std'] = sigma_dec
    
    summary['residual_ra_rms'] = rmse_ra
    summary['residual_dec_rms'] = rmse_dec
    
    #average of tf and id RMSE
    summary['residual_ra_rms_tf_id'] = (summary['residual_ra_rms'] +  summary['rmse_ra_id'])/2
    summary['residual_dec_rms_tf_id'] = (summary['residual_dec_rms'] +  summary['rmse_dec_id'])/2


    #--------------------------------------------------------------------------------------------
    # Compute arcsec to rad

    arcsec_to_rad = np.pi / (180.0 * 3600.0)

    #STD weights
    sigma_ra_rad  = sigma_ra * arcsec_to_rad
    sigma_dec_rad = sigma_dec * arcsec_to_rad

    summary['weight_std_ra']  = 1.0 / (sigma_ra_rad ** 2)
    summary['weight_std_dec'] = 1.0 / (sigma_dec_rad ** 2)

    #RMS weights
    rmse_ra_rad  = summary['residual_ra_rms'] * arcsec_to_rad
    rmse_dec_rad = summary['residual_ra_rms'] * arcsec_to_rad

    summary['weight_rmse_ra']  = 1.0 / (rmse_ra_rad ** 2)
    summary['weight_rmse_dec'] = 1.0 / (rmse_dec_rad ** 2)

    #Hybrid RMS weights TF ID
    rmse_ra_rad  = summary['residual_ra_rms_tf_id'] * arcsec_to_rad
    rmse_dec_rad = summary['residual_dec_rms_tf_id'] * arcsec_to_rad

    summary['weight_rmse_tf_id_ra']  = 1.0 / (rmse_ra_rad ** 2)
    summary['weight_rmse_tf_id_dec'] = 1.0 / (rmse_dec_rad ** 2)



    #--------------------------------------------------------------------------------------------
    # Compute mean weight
    
    summary['mean_weight_std'] = summary[['weight_std_ra', 'weight_std_dec']].mean(axis=1)
    summary['mean_weight_rmse'] = summary[['weight_rmse_ra', 'weight_rmse_dec']].mean(axis=1)
    

    # Create descale per night weights std
    summary['weight_std_ra_scaled'] = summary['weight_std_ra'] / np.sqrt(summary['n_obs'])
    summary['weight_std_dec_scaled'] = summary['weight_std_dec'] / np.sqrt(summary['n_obs'])

    summary['mean_weight_std_scaled'] = summary[['weight_std_ra_scaled', 'weight_std_dec_scaled']].mean(axis=1)

    # Create descale per night weights rmse
    summary['weight_rmse_ra_scaled'] = summary['weight_rmse_ra'] / np.sqrt(summary['n_obs'])
    summary['weight_rmse_dec_scaled'] = summary['weight_rmse_dec'] / np.sqrt(summary['n_obs'])

    summary['mean_weight_rmse_scaled'] = summary[['weight_rmse_ra_scaled', 'weight_rmse_dec_scaled']].mean(axis=1)


    # Create descaled per night weights rmse ID 
    summary['weight_rmse_ra_id_scaled'] = summary['weight_rmse_ra_id'] / np.sqrt(summary['n_obs'])
    summary['weight_rmse_dec_id_scaled'] = summary['weight_rmse_dec_id'] / np.sqrt(summary['n_obs'])

    summary['mean_weight_rmse_id_scaled'] = summary[['weight_rmse_ra_id_scaled', 'weight_rmse_dec_id_scaled']].mean(axis=1)

    # Create descaled per night weights std ID 
    summary['weight_std_ra_id_scaled'] = summary['weight_std_ra_id'] / np.sqrt(summary['n_obs'])
    summary['weight_std_dec_id_scaled'] = summary['weight_std_dec_id'] / np.sqrt(summary['n_obs'])

    summary['mean_weight_std_id_scaled'] = summary[['weight_std_ra_id_scaled', 'weight_std_dec_id_scaled']].mean(axis=1)



    # Create descaled per night weights rmse Hybrid ID TF 
    summary['weight_rmse_ra_tf_id_scaled'] = summary['weight_rmse_tf_id_ra'] / np.sqrt(summary['n_obs'])
    summary['weight_rmse_dec_tf_id_scaled'] = summary['weight_rmse_tf_id_dec'] / np.sqrt(summary['n_obs'])

    summary['mean_weight_rmse_tf_id_scaled'] = summary[['weight_rmse_ra_tf_id_scaled', 'weight_rmse_dec_tf_id_scaled']].mean(axis=1)




    # Compute difference between std and rmse metric
    summary["ra_diff_id"] = summary["rmse_ra_id"] - summary["std_ra_id"]
    summary["dec_diff_id"] = summary["rmse_dec_id"] - summary["std_dec_id"]





    # Compute differences for timeframes
    summary["ra_diff"] = summary["residual_ra_rms"] - summary["residual_ra_std"]
    summary["dec_diff"] = summary["residual_dec_rms"] - summary["residual_dec_std"]

    return summary
    #Dont save in this function
    # summary.to_csv(out_dir / "summary.txt", sep="\t", float_format="%.8e")







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
        #"nep105.bsp",
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
    simulation_start_epoch = DateTime(1963, 1,  1).epoch()
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
    settings_env['Neptune_rot_model_type'] = 'IAU2015' 

    body_settings,system_of_bodies = PropFuncs.Create_Env(settings_env)

    #--------------------------------------------------------------------------------------------
    # LOAD OBSERVATIONS
    #--------------------------------------------------------------------------------------------
    with open("file_names.json", "r") as f:
        file_names_loaded = json.load(f)

    # Extract IDs from filenames (remove 'Triton_' prefix and '.csv' suffix)
    ordered_ids = [f.replace('Triton_', '').replace('.csv', '') for f in file_names_loaded]


    data = []
    observations,observations_settings,observation_set_ids = ObsFunc.LoadObservations("Observations/AllModernECLIPJ2000",system_of_bodies,file_names_loaded)

    print("Loaded tudat observations...")
    
    #--------------------------------------------------------------------------------------------
    # THIS FUNCTION GENERATES THE DATAFRAME WITH WEIGHTS
    #--------------------------------------------------------------------------------------------

    summary = main()


    #--------------------------------------------------------------------------------------------
    # ALL PLOTTING MOVED HERE AND PROBABLY NOT WORKING
    #--------------------------------------------------------------------------------------------
    
    # nr_observations = []
    # for list1 in arrays['residuals_sorted']:
    #     nr_observations.append(len(list1)/2)
    # nr_observations_check = []

    # for list1 in observations.get_observations():
    #     nr_observations_check.append(len(list1)/2) 



    #Load results without outliers
    # arrays2 = load_npy_files("Results/AllModernObservations/TimeFrameWeights_50mas_min")
    
    # print("Loaded previous numpy arrays from estimation...")
    # nr_observations = []
    # for list1 in arrays2['residuals_sorted']:
    #     nr_observations.append(len(list1)/2)
    # nr_observations_check = []

    # for list1 in observations.get_observations():
    #     nr_observations_check.append(len(list1)/2) 

    # df_initial = create_observations_dataframe(
    #     arrays2['observations_sorted'], 
    #     ordered_ids,
    #     arrays2['observation_times'],
    #     arrays2['residual_history_arcseconds'],
    #     arrays2)
    
    
    #--------------------------------------------------------------------------------------------
    # MAKE RSW differences
    #--------------------------------------------------------------------------------------------
    # state_history_without_outliers = arrays['state_history_array']
    # state_history_with_outliers = arrays2['state_history_array']
    # time_column = state_history_without_outliers[:, [0]]  
    # #states_SPICE_RSW = ProcessingUtils.rotate_inertial_3_to_rsw(time_column, states_SPICE[:,0:3], state_history_array)
    # states_without_outliers_RSW = ProcessingUtils.rotate_inertial_3_to_rsw(time_column, state_history_without_outliers[:,1:4], state_history_without_outliers)
    # states_with_outliers_RSW = ProcessingUtils.rotate_inertial_3_to_rsw(time_column, state_history_with_outliers[:,1:4], state_history_without_outliers)

    # diff_RSW = (states_without_outliers_RSW - states_with_outliers_RSW)/1e3

        ##############################################################################################
    # PLOT FIGURES  
    ##############################################################################################
    
    #make folder
    # Load unique id colors (new one needs to be made)
    # with open('file_colors.pkl', 'rb') as f:
    #     file_colors = pickle.load(f)

    
    out_dir = make_timestamped_folder("Results/AllModernJ2000Frame/PostProcessing")
    
    make_figures = False
    if make_figures == True:
        print("Making figures...")
        #--------------------------------------------------------------------------------------------
        #Formal Errors
        final  = arrays['final_paramaters']      # note the key name as provided
        initial = arrays['initial_paramaters']
        sigma   = arrays['formal_errors']

        fig_formal_errors = plot_formal_errors(initial,final,sigma)
        fig_formal_errors.savefig(out_dir / "Formal_Errors.pdf")
        #--------------------------------------------------------------------------------------------
        #RSW Difference
        fig_final_sim_SPICE_rsw = FigUtils.Residuals_RSW(diff_RSW, time_column,type="difference",title="RSW Difference Weights vs No Weights")
        fig_final_sim_SPICE_rsw.savefig(out_dir / "RSW_Diff_with_without_weights.pdf")

        #--------------------------------------------------------------------------------------------

        fig,file_colors = plot_observation_analysis(df, title_suffix="All Data")
        fig.savefig(out_dir / "Colored_Count.pdf")
        # fig_rms = FigUtils.Residuals_RMS(residual_history_arcseconds)
        # fig_rms.savefig(out_dir / "Residual_RMS.pdf")

        # Define the file numbers from the screenshot
        file_numbers = ['nm0002', 'nm0003', 'nm0004', 'nm0006', 'nm0008', 'nm0009', 'nm0010']

        # Method 1: Filter rows that contain any of these file numbers
        df_included = df[df['id'].str.contains('|'.join(file_numbers))]

        # Method 2: Filter rows that DON'T contain any of these file numbers
        df_excluded = df[~df['id'].str.contains('|'.join(file_numbers))]

        fig,file_colors = plot_observation_analysis(df_included,file_colors=file_colors,title_suffix="Relative Data")
        fig.savefig(out_dir / "Relative_Colored_Count.pdf")

        fig,file_colors = plot_observation_analysis(df_excluded,file_colors=file_colors,title_suffix="Absolute Data")
        fig.savefig(out_dir / "Absolute_Colored_Count.pdf")


        #--------------------------------------------------------------------------------------------

        residual_df = calculate_mean_std(df)
        fig_first,fig_last,_ = plot_mean_std(residual_df,file_colors=file_colors,title_suffix = "")
        fig_first.savefig(out_dir / "Colored_mean_std_Initial.pdf")
        fig_last.savefig(out_dir / "Colored_mean_std_Last.pdf")

        fig, axes = overlay_residual_stats(df, file_colors)
        fig.savefig(out_dir / "Colored_mean_std_overlayed.pdf")


        print("Created and saved all other figures...")
        #--------------------------------------------------------------------------------------------

        # fig_estimation_residuals,ax = plot_RA_DEC_residuals(df,df_initial, file_colors=file_colors, dataset_labels=("removed outliers", "with outliers"))
        # fig_estimation_residuals.savefig(out_dir / "Residuals_time.pdf")

        fig_estimation_residuals_colored = plot_RA_DEC_residuals_colored(df, file_colors=file_colors, labels=["final", "initial"])
        fig_estimation_residuals_colored.savefig(out_dir / "Residuals_time_colored.pdf")


        fig_estimation_residuals,ax = plot_RA_DEC_residuals(df,df_initial, file_colors=file_colors, dataset_labels=("more data", "previous data"))
        fig_estimation_residuals.savefig(out_dir / "Residuals_new_data_set_vs_previous.pdf")


        #individual_figs = plot_individual_RA_DEC_residuals(df,file_colors) 
        #individual_figs_demeaned = plot_individual_RA_DEC_residuals_demeaned(df,file_colors)
        #--------------------------------------------------------------------------------------------
        make_residual_figures = False
        if make_residual_figures == True:
            figs = plot_residuals(df_frames,summary,plot_individual_frames=False)

            for file_id, fig in figs.items():
                fig.savefig(out_dir / (file_id + '_residuals_time_timeframes.pdf'))
            #--------------------------------------------------------------------------------------------

        # individual_figs_timeframes =  plot_timeframe_residuals(df_frames,file_colors=file_colors)

        # for file_id, fig in individual_figs_timeframes.items():
        #     fig.savefig(out_dir / (file_id[0] + "_" + str(file_id[1]) + '_residuals_time_timeframes.pdf'))


        # for file_id, fig in individual_figs_demeaned.items():
        #     fig.savefig(out_dir / (file_id + '_residuals_time_demeaned.pdf'))


        # outliers = df_initial[~df_initial['times'].isin(df['times'])].copy()
        # fig_outlier_residuals_colored = plot_RA_DEC_residuals_colored(outliers, file_colors=file_colors, labels=["final", "initial"])
        # fig_outlier_residuals_colored.savefig(out_dir / "Residuals_time_colored_outliers.pdf")
    else:
        print("Plotting is turned off. Continuing...")



        #--------------------------------------------------------------------------------------------
    # Check if weights can be assigned from summary
    #--------------------------------------------------------------------------------------------
    # observations,observations_settings,observation_set_ids = ObsFunc.LoadObservations(
    #         "Observations/RelativeObservations/",
    #         system_of_bodies,file_names_loaded,
    #         weights = summary,
    #         timeframe_weights=True,
    #         per_night_weights_hybrid=True)
    #--------------------------------------------------------------------------------------------
    # Plot Weight Figures
    #--------------------------------------------------------------------------------------------
    make_weight_figures = False
    if make_weight_figures == True:
        print("plotting weight figures...")

        #----------------------------
        # weight plots
        #----------------------------
        fig_weights_per_time = plot_average_weight(summary, split_by_id=False)
        #fig_weights_per_file = plot_average_weight_vs_time_from_weights(weights, summary, split_by_id=False)
        fig_std_per_timeframe = plot_residual_std(summary)
        fig_per_night_weights = plot_average_weight(summary,
                split_by_id=False,selected_key="mean_weight_rmse_scaled",title="Descaled Weight per timeframe (night)")



        fig_weights_per_time.savefig(out_dir / "fig_weights_per_time.pdf")
        #fig_weights_per_file.savefig(out_dir / "fig_weights_per_id.pdf")
        fig_std_per_timeframe.savefig(out_dir / "fig_std_per_timeframe.pdf")
        fig_per_night_weights.savefig(out_dir / "fig_weights_per_night.pdf")
        
        #----------------------------
        # Plot number of observations
        #----------------------------
        fig_count_per_timeframe = plot_average_weight(
                summary, 
                split_by_id = False,
                selected_key = "n_obs",
                logscale=False,
                title="Number of Observations per Timeframe",
                label_y="Number of observations [-]")
        fig_count_per_timeframe.savefig(out_dir / "fig_count_per_timeframe.pdf")
        #----------------------------
        # Plot differences per timeframe
        #----------------------------
        ra_diff_per_timeframe = plot_average_weight(
                summary,
                split_by_id = False,
                selected_key = "ra_diff",
                logscale=False,
                title="RA STD - RMS difference per Timeframe",
                label_y="STD - RMS [arcs]")

        dec_diff_per_timeframe = plot_average_weight(
                    summary,
                    split_by_id = False,
                    selected_key = "dec_diff",
                    logscale=False,
                    title="DEC STD - RMS difference per Timeframe",
                    label_y = "STD - RMS [arcs]")
        
        ra_diff_per_timeframe.savefig(out_dir / "diff_timeframe_rmse_std_ra.pdf")
        dec_diff_per_timeframe.savefig(out_dir / "diff_timeframe_rmse_std_dec.pdf")

        #----------------------------
        # Plot differences per id
        #----------------------------
        ra_diff_per_id = plot_average_weight(
                summary,
                split_by_id = True,
                selected_key = "ra_diff_id",
                logscale=False,
                title="RA STD - RMS difference per ID",
                label_y="STD - RMS [arcs]")

        dec_diff_per_id = plot_average_weight(
                    summary,
                    split_by_id = True,
                    selected_key = "dec_diff_id",
                    logscale=False,
                    title="DEC STD - RMS difference per ID",
                    label_y = "STD - RMS [arcs]")

        ra_diff_per_id.savefig(out_dir / "diff_id_rmse_std_ra.pdf")
        dec_diff_per_id.savefig(out_dir / "diff_id_rmse_std_dec.pdf")

        # for file_id, fig in ra_diff_per_id.items():
        #     fig.savefig(out_dir / (file_id + 'diff_id_rmse_std_ra.pdf'))
        # for file_id, fig in dec_diff_per_id.items():
        #     fig.savefig(out_dir / (file_id + 'diff_id_rmse_std_dec.pdf'))
        
        #----------------------------
        # Plot differences RMSE id vs timeframe
        #----------------------------
        # Compute differences for timeframes
        summary["ra_diff_id_tf"] = summary["residual_ra_rms"] - summary["rmse_ra_id"]
        summary["dec_diff_id_tf"] = summary["residual_dec_rms"] - summary["rmse_dec_id"]



        ra_diff_per_id_tf = plot_average_weight(
                summary,
                split_by_id = False,
                selected_key = "ra_diff_id_tf",
                logscale=False,
                title="RA ID - TF for RMSE",
                label_y="Diff RMSE [arcs]")

        dec_diff_per_id_tf = plot_average_weight(
                    summary,
                    split_by_id = False,
                    selected_key = "dec_diff_id_tf",
                    logscale=False,
                    title="DEC ID - TF for RMSE",
                    label_y = "RMSE [arcs]")



        ra_diff_per_id_tf.savefig(out_dir / "diff_id_tf_rmse_ra.pdf")
        dec_diff_per_id_tf.savefig(out_dir / "diff_id_tf_rmse_dec.pdf")



        fig_weights_id_rmse_norm = plot_average_weight(summary,
        split_by_id=False,selected_key="mean_weight_rmse_id",title="RMSE Weight per ID (night)")


        fig_weights_id_rmse = plot_average_weight(summary,
        split_by_id=False,selected_key="mean_weight_rmse_id_scaled",title="Descaled RMSE Weight per ID (night)")


        fig_weights_id_std = plot_average_weight(summary,
        split_by_id=False,selected_key="mean_weight_std_id_scaled",title="Descaled STD Weight per ID (night)")


        fig_weights_id_rmse_norm.savefig(out_dir / "id_rmse_weight_not_descaled.pdf")
        fig_weights_id_rmse.savefig(out_dir / "id_rmse_weight.pdf")
        fig_weight_id_std.savefig(out_dir / "id_std_weight.pdf")


    else:
        print("plotting disabled, not making weight figures.")


    print("end")