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

# def plot_observation_analysis(df, file_colors=None, title_suffix=""):
#     """
#     Create stacked histogram and bar chart of observations over time.
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Dataframe with columns: 'id', 'observatory', 'times'...
#     file_colors : dict, optional
#         Dictionary mapping file number to color. If None, creates new mapping.
#     title_suffix : str
#         Additional text to add to plot titles
    
#     Returns:
#     --------
#     dict : Color mapping used in the plot
#     fig  : Figure to save to specific folder later
#     """
#     file_numbers = sorted(df['id'].unique())
#     n_files = len(file_numbers)
    
#     if n_files == 0:
#         print("No data to plot!")
#         return file_colors
    
#     # Use provided color mapping or create new one
#     if file_colors is None:
#         file_colors = create_color_mapping(df)
    
#     # Explode the dataframe so each observation time gets its own row
#     rows = []
#     for _, row in df.iterrows():
#         # Convert Tudat time (seconds since J2000) to datetime
#         time_datetime = J2000_EPOCH + timedelta(seconds=float(row['times']))
#         rows.append({
#             'id': row['id'],
#             'observatory': row['observatory'],
#             'time': time_datetime
#         })

#     df_exploded = pd.DataFrame(rows)
    
#     # Create a period column for monthly bins
#     df_exploded['month'] = df_exploded['time'].dt.to_period('M')
    
#     # Count observations per file per month
#     monthly_counts = df_exploded.groupby(['month', 'id']).size().unstack(fill_value=0)
    
#     # Calculate n_observations for the bar plot
#     id_counts = df.groupby('id').size().reset_index(name='n_observations')



#     # Create figure with two subplots (stacked vertically)
#     fig = plt.figure(figsize=(18, 12))
    
#     # Adjust subplot positioning to leave room for legend
#     gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3,
#                           left=0.08, right=0.82, top=0.95, bottom=0.08)
    
#     # Top subplot: Stacked histogram
#     ax1 = fig.add_subplot(gs[0])
    
#     # Get colors in the correct order for the stacked plot (only for files in current data)
#     plot_colors = [file_colors[nr] for nr in monthly_counts.columns]
    
#     monthly_counts.plot(
#         kind='bar',
#         stacked=True,
#         ax=ax1,
#         color=plot_colors,
#         width=1.0,
#         legend=False
#     )
    
#     ax1.set_xlabel('Time (Month)', fontsize=12)
#     ax1.set_ylabel('Number of Observations', fontsize=12)
#     title = f'Observation Count Over Time by File{title_suffix}'
#     ax1.set_title(title, fontsize=14)
    
#     # Format x-axis to show fewer labels
#     n_labels = 20
#     tick_positions = np.linspace(0, len(monthly_counts) - 1, n_labels, dtype=int)
#     ax1.set_xticks(tick_positions)
#     ax1.set_xticklabels([str(monthly_counts.index[i]) for i in tick_positions], 
#                          rotation=45, ha='right')
    
#     # Create legend outside the plot area (only for files in current data)
#     handles = [Patch(facecolor=file_colors[nr], label=f'File {nr}') 
#                for nr in file_numbers]
#     ax1.legend(handles=handles, 
#                bbox_to_anchor=(1.02, 1), 
#                loc='upper left',
#                ncol=1,
#                fontsize=8,
#                frameon=True)

#     # Set hard x-axis limits
#     xlim_start = pd.Timestamp('1963-01-01').to_period('M')
#     xlim_end = pd.Timestamp('2025-01-01').to_period('M')
#     #ax1.set_xlim(-0.5, (xlim_end - xlim_start).n + 0.5)
      
#     # Bottom subplot: Bar chart of observations per file
#     ax2 = fig.add_subplot(gs[1])
    
#     # Use the same colors for each file
#     bar_colors = [file_colors[id] for id in id_counts['id']]

#     ax2.bar(id_counts['id'], id_counts['n_observations'], color=bar_colors)
#     ax2.set_xlabel('File Number', fontsize=12)
#     ax2.set_ylabel('Count', fontsize=12)
#     ax2.set_title('Number of Observations per File', fontsize=14)
#     ax2.tick_params(axis='x', rotation=45)
    

#     return fig, file_colors

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
    'residual_ra_last': ['mean', 'std'],
    'residual_dec_last': ['mean', 'std']
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



# def split_into_timeframes(
#     df: pd.DataFrame,
#     gap_hours: float = 4.0,
#     time_col: str = "times",              # seconds since J2000
#     make_datetime: bool = False,
#     min_obs_per_frame: int = 1,
#     merge_last_short_back: bool = False,
#     summary_cols: list[str] = ["residual_ra_last", "residual_dec_last"],       # e.g. ["residual_ra_last", "residual_dec_last"]
# ):
#     """
#     Split each id into time frames using a gap threshold, but only break if the
#     current frame already has >= min_obs_per_frame. Optionally merge a short
#     final frame back. Adds per-frame mean/std for requested columns in 'summary'.
#     """
#     gap_sec = float(gap_hours) * 3600.0
#     df_frames = df.copy().sort_values(["id", time_col]).reset_index(drop=True)

#     if make_datetime and "datetime" not in df_frames.columns:
#         J2000_EPOCH = datetime(2000, 1, 1, 12, 0, 0)
#         df_frames["datetime"] = df_frames[time_col].astype(float).apply(
#             lambda t: J2000_EPOCH + timedelta(seconds=float(t))
#         )

#     def assign_with_min_size(g):
#         t = g[time_col].astype(float).to_numpy()
#         n = len(t)
#         if n == 0:
#             return pd.Series([], dtype=int, index=g.index)
#         diffs = np.diff(t)
#         is_break = np.concatenate(([False], diffs >= gap_sec))
#         frame_ids = np.zeros(n, dtype=int)
#         cur_frame, cur_count = 0, 1
#         for i in range(1, n):
#             if is_break[i] and cur_count >= max(1, min_obs_per_frame):
#                 cur_frame += 1
#                 cur_count = 1
#             else:
#                 cur_count += 1
#             frame_ids[i] = cur_frame
#         if merge_last_short_back and cur_count < max(1, min_obs_per_frame) and cur_frame > 0:
#             frame_ids[frame_ids == cur_frame] = cur_frame - 1
#         return pd.Series(frame_ids, index=g.index, dtype=int)

#     df_frames["timeframe"] = df_frames.groupby("id", group_keys=False).apply(assign_with_min_size)

#     # ---------- Build per-frame summary with mean/std ----------
#     base = (
#         df_frames.groupby(["id", "timeframe"])
#         .agg(start_sec=(time_col, "min"),
#              end_sec=(time_col, "max"),
#              n_obs=(time_col, "count"))
#         .reset_index()
#     )
#     base["duration_hours"] = (base["end_sec"] - base["start_sec"]) / 3600.0

#     # Add per-frame means/stds for requested columns
#     if summary_cols:
#         stats = (
#             df_frames.groupby(["id", "timeframe"])[summary_cols]
#             .agg(["mean", "std"])  # std uses ddof=1 (sample std)
#         )
#         # Flatten MultiIndex columns: ("residual_ra_last","mean") -> "mean_residual_ra_last"
#         stats.columns = [f"{func}_{col}" for col, func in stats.columns.swaplevel(0,1)]
#         stats = stats.reset_index()
#         summary = base.merge(stats, on=["id", "timeframe"], how="left")
#     else:
#         summary = base

#     if make_datetime:
#         J2000_EPOCH = datetime(2000, 1, 1, 12, 0, 0)
#         summary["start_dt"] = summary["start_sec"].apply(lambda s: J2000_EPOCH + timedelta(seconds=float(s)))
#         summary["end_dt"]   = summary["end_sec"].apply(lambda s: J2000_EPOCH + timedelta(seconds=float(s)))

#     return df_frames, summary



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
        summary_cols = ["residual_ra_last", "residual_dec_last"]
    
    # Convert gap to seconds
    gap_seconds = gap_hours * 3600.0
    
    # Sort by id and time
    df_sorted = df.copy().sort_values(["id", time_col]).reset_index(drop=True)
    
    # Assign timeframe numbers to each observation
    def assign_timeframes(group):
        """Assign timeframe number to each observation in the group."""
        times = group[time_col].values
        n = len(times)
        
        # Start with all observations in frame 0
        timeframes = np.zeros(n, dtype=int)
        
        current_frame = 0
        obs_in_current_frame = 1
        
        for i in range(1, n):
            time_gap = times[i] - times[i-1]
            
            # Should we start a new frame?
            # Yes, if gap is large AND current frame has enough observations
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
            stats = df.groupby(["id", "timeframe"])[col].agg(["mean", "std"]).reset_index()
            stats.columns = ["id", "timeframe", f"{col}_mean", f"{col}_std"]
            summary = summary.merge(stats, on=["id", "timeframe"], how="left")
    
    # Add datetime columns
    J2000_EPOCH = datetime(2000, 1, 1, 12, 0, 0)
    summary["start_dt"] = summary["start_sec"].apply(
        lambda s: J2000_EPOCH + timedelta(seconds=s)
    )
    summary["end_dt"] = summary["end_sec"].apply(
        lambda s: J2000_EPOCH + timedelta(seconds=s)
    )
    
    return summary


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


def plot_residuals_from_df_frames(
    df_frames: pd.DataFrame,
    *,
    file_colors: dict | None = None,
    labels: tuple[str, str] = ("Final", "Initial"),
    show_initial: bool = False,
    id_col: str = "id",
    timeframe_col: str = "timeframe",
    dt_col: str = "datetime",         # expects datetime already
    time_col: str = "times",          # optional, only used if datetime missing
    frame_alpha: float = 1,
    cmap_name: str = "Pastel1",
):
    """
    Plot RA/DEC residuals using df_frames (which already includes timeframe info).

    Parameters
    ----------
    df_frames : pd.DataFrame
        Must contain columns:
          - id_col (str)
          - timeframe_col (int or category)
          - datetime (datetime) or times (seconds since J2000)
          - residual_ra_last, residual_dec_last
        Optionally residual_ra_first, residual_dec_first.

    file_colors : dict, optional
        Map {id -> color} for scatter points.
    labels : (str, str)
        Labels for [final, initial] residuals.
    show_initial : bool
        If True, plot initial residuals (if available).
    id_col : str
        Column name for observation set identifier.
    timeframe_col : str
        Column name for the timeframe index.
    dt_col : str
        Datetime column name (if not present, one will be derived from `time_col`).
    frame_alpha : float
        Transparency for shaded timeframe backgrounds.
    cmap_name : str
        Matplotlib colormap name for timeframe shading.

    Returns
    -------
    figs : dict
        {id_value: matplotlib.figure.Figure}
    """
    # J2000 reference (only used if datetime missing)
    J2000_EPOCH = datetime(2000, 1, 1, 12, 0, 0)

    def ensure_datetime(df):
        if dt_col in df.columns:
            return df[dt_col]
        else:
            return df[time_col].astype(float).map(lambda s: J2000_EPOCH + timedelta(seconds=float(s)))

    figs = {}
    unique_ids = pd.unique(df_frames[id_col])

    for file_id in unique_ids:
        dfi = df_frames[df_frames[id_col] == file_id].copy()
        if dfi.empty:
            continue

        dfi = dfi.sort_values(time_col)
        dfi["__dt__"] = ensure_datetime(dfi)

        # Determine number of timeframes and their ranges
        grouped = dfi.groupby(timeframe_col, group_keys=True)
        timeframes = []
        for tf_id, g in grouped:
            tmin, tmax = g["__dt__"].min(), g["__dt__"].max()
            n_obs = len(g)
            timeframes.append((tf_id, tmin, tmax, n_obs))

        n_frames = len(timeframes)
        cmap = mpl.cm.get_cmap(cmap_name, max(n_frames, 3))

        # Color for points
        color = (file_colors or {}).get(file_id, "C0")

        # Create figure
        fig, ax = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

        # === RA residuals ===
        ax[0].scatter(dfi["__dt__"], dfi["residual_ra_last"], color=color,
                      alpha=0.7, s=30, label=labels[0])
        if show_initial and "residual_ra_first" in dfi.columns:
            ax[0].scatter(dfi["__dt__"], dfi["residual_ra_first"], color=color,
                          alpha=0.35, marker="x", s=30, label=labels[1])
        ax[0].set_xlabel("Observation Epoch", fontsize=12)
        ax[0].set_ylabel("Simulated - Observed RA [arcseconds]", fontsize=12)
        ax[0].set_title(f"RA Residuals: {file_id}", fontsize=14)
        ax[0].grid(True, alpha=0.3)
        ax[0].axhline(y=0, color="k", linestyle="--", alpha=0.5, linewidth=1)

        # === DEC residuals ===
        ax[1].scatter(dfi["__dt__"], dfi["residual_dec_last"], color=color,
                      alpha=0.7, s=30, label=labels[0])
        if show_initial and "residual_dec_first" in dfi.columns:
            ax[1].scatter(dfi["__dt__"], dfi["residual_dec_first"], color=color,
                          alpha=0.35, marker="x", s=30, label=labels[1])
        ax[1].set_xlabel("Observation Epoch", fontsize=12)
        ax[1].set_ylabel("Simulated - Observed DEC [arcseconds]", fontsize=12)
        ax[1].set_title(f"DEC Residuals: {file_id}", fontsize=14)
        ax[1].grid(True, alpha=0.3)
        ax[1].axhline(y=0, color="k", linestyle="--", alpha=0.5, linewidth=1)

        # Date formatting
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        for a in ax:
            a.xaxis.set_major_locator(locator)
            a.xaxis.set_major_formatter(formatter)

        # === Shade background per timeframe ===
        frame_handles, frame_labels = [], []
        for i, (tf_id, tmin, tmax, n_obs) in enumerate(timeframes):
            fc = cmap(i)
            for a in ax:
                a.axvspan(tmin, tmax, facecolor=fc, alpha=frame_alpha, zorder=0, linewidth=0)
            frame_handles.append(Patch(facecolor=fc, edgecolor="none", alpha=frame_alpha))
            frame_labels.append(f"Frame {tf_id} (n={n_obs})")
        
        #from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    #    # pick the 1–2 shortest frames
    #     N = 1
    #     short = sorted(timeframes, key=lambda r: (r[2]-r[1]))[:min(N, len(timeframes))]
    #     for tf_id, tmin, tmax, n_obs in short:
    #         diff = tmax - tmin
    #         iax = add_short_frame_inset(ax[0], dfi, tmin-diff, tmax+diff,
    #                                     y_col="residual_ra_last",
    #                                     show_initial=show_initial,
    #                                     y0_col="residual_ra_first")
    #         # Optional caption inside the inset:
    #         iax.set_title(f"F{tf_id} (n={n_obs})", fontsize=8, pad=2)


        # === Legends ===
        # # Merge scatter + timeframe legends on RA axis
        h_pts, l_pts = ax[0].get_legend_handles_labels()
        if frame_handles:
            ax[0].legend(h_pts + frame_handles, l_pts + frame_labels, loc="best")
        else:
            ax[0].legend(loc="best")

        # Only scatter legend for DEC
        h_pts_dec, l_pts_dec = ax[1].get_legend_handles_labels()
        #ax[1].legend(h_pts_dec, l_pts_dec, loc="best")

        # after you build h_pts/l_pts and frame_handles/frame_labels:
        handles = h_pts + frame_handles
        labels  = l_pts + frame_labels

        # Put legend outside on the right of RA axis
        ax[1].legend(
            handles, labels,
            loc="right",
            bbox_to_anchor=(1.5, 0.5),   # push just outside the axes
            borderaxespad=0.0,
            frameon=True
        )

        # Tighten layout so the outside legend fits
        #fig.tight_layout()


        figs[file_id] = fig

    return figs


from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.dates as mdates


def add_short_frame_inset(ax, dfi, tmin, tmax, *,
                          y_col="residual_ra_last",
                          show_initial=False, y0_col="residual_ra_first"):
    # Create inset anchored inside the parent axes (upper-right corner)
    iax = inset_axes(
        ax, width="34%", height="34%",
        loc="upper right",
        bbox_to_anchor=(0, 0, 1, 1),      # inside the axes box
        bbox_transform=ax.transAxes,
        borderpad=0.6
    )

    m = (dfi["__dt__"] >= tmin) & (dfi["__dt__"] <= tmax)
    iax.scatter(dfi.loc[m, "__dt__"], dfi.loc[m, y_col], s=18, alpha=0.9)
    if show_initial and y0_col in dfi.columns:
        iax.scatter(dfi.loc[m, "__dt__"], dfi.loc[m, y0_col], s=18, alpha=0.5, marker="x")

    # Limit with a little padding
    pad = (tmax - tmin) / 6
    iax.set_xlim(tmin - pad, tmax + pad)

    # Tight y-lims with padding
    y = dfi.loc[m, y_col]
    if len(y) >= 1:
        yr = (y.max() - y.min()) if len(y) > 1 else 1.0
        ypad = 0.2 * (yr if yr > 0 else 1.0)
        iax.set_ylim(y.min() - ypad, y.max() + ypad)

    # Use a light date formatter with few ticks so labels don’t collide
    locator = mdates.AutoDateLocator(minticks=3, maxticks=4)
    iax.xaxis.set_major_locator(locator)
    iax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    iax.tick_params(axis="x", labelsize=8)
    iax.tick_params(axis="y", labelsize=8)
    iax.grid(True, alpha=0.25)

    # Title with small pad so it doesn't overlap the frame
    iax.set_title("", fontsize=9, pad=2)
    return iax



import math
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

def plot_weights_per_id_timeframe(summary, *, facet_by_id=True):
    """
    Plot the RA and DEC weights per timeframe for each ID.

    Parameters
    ----------
    summary : pd.DataFrame
        Must include ['id', 'timeframe', 'weight_ra', 'weight_dec'].
    facet_by_id : bool, default True
        If True, return {id: fig}; else one combined figure.

    Returns
    -------
    dict[str, matplotlib.figure.Figure] | matplotlib.figure.Figure
    """
    df = summary.copy()
    required = {"id", "timeframe", "weight_ra", "weight_dec"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"summary missing columns: {sorted(missing)}")

    df = df.sort_values(["id", "timeframe"]).reset_index(drop=True)

    # internal helper for one ID
    def _plot_one(ax, g, title):
        tfs = g["timeframe"].astype(int).to_numpy()
        x = np.arange(len(tfs))
        width = 0.36
        ax.bar(x - width/2, g["weight_ra"], width, label="RA", alpha=0.85)
        ax.bar(x + width/2, g["weight_dec"], width, label="DEC", alpha=0.85)
        ax.set_yscale("log")
        ax.set_xlabel("Timeframe")
        ax.set_ylabel("Weight [1/rad²]")
        ax.set_title(title)
        ax.set_xticks(x, [str(tf) for tf in tfs])
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="upper right")

    # ---- plot per ID ----
    if facet_by_id:
        figs = {}
        for file_id, g in df.groupby("id", sort=False):
            fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
            _plot_one(ax, g, f"Weights per timeframe — {file_id}")
            figs[file_id] = fig
        return figs

    # ---- combined plot ----
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    x_positions = []
    weights_ra = []
    weights_dec = []
    labels = []

    x = 0.0
    gap = 1.0  # spacing between IDs
    for file_id, g in df.groupby("id", sort=False):
        g = g.sort_values("timeframe")
        n = len(g)
        xs = x + np.arange(n, dtype=float)
        x_positions.append(xs)
        weights_ra.append(g["weight_ra"].to_numpy())
        weights_dec.append(g["weight_dec"].to_numpy())
        labels.extend([f"{file_id}\nTF {tf}" for tf in g["timeframe"]])
        x = xs[-1] + gap + 1.0

    if not x_positions:
        raise ValueError("No data to plot.")

    X = np.concatenate(x_positions)
    RA = np.concatenate(weights_ra)
    DE = np.concatenate(weights_dec)

    width = 0.36
    ax.bar(X - width/2, RA, width, label="RA", alpha=0.85)
    ax.bar(X + width/2, DE, width, label="DEC", alpha=0.85)
    ax.set_yscale("log")
    ax.set_xticks(X, labels, rotation=0)
    ax.set_xlim(-0.75, X.max() + 0.75)
    ax.set_xlabel("ID / Timeframe")
    ax.set_ylabel("Weight [1/rad²]")
    ax.set_title("Weights per ID & timeframe")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.autofmt_xdate(rotation=0)

    return fig

# def plot_average_weight_vs_time(summary, *, split_by_id=False):
#     """
#     Plots average weight ((RA+DEC)/2) at each frame midpoint time.
#     Prefers start_dt/end_dt; falls back to start_sec/end_sec.
#     Returns:
#       - dict of {id: Figure} if split_by_id=True
#       - a single Figure otherwise
#     """
#     df = summary.copy()
#     need = {"id", "weight_ra", "weight_dec"}
#     if need - set(df.columns):
#         raise ValueError(f"summary missing columns: {sorted(need - set(df.columns))}")

#     # Build midpoint datetime
#     if {"start_dt", "end_dt"} <= set(df.columns):
#         df["mid_dt"] = df["start_dt"] + (df["end_dt"] - df["start_dt"]) / 2
#     else:
#         if {"start_sec", "end_sec"} - set(df.columns):
#             raise ValueError("Need start_dt/end_dt or start_sec/end_sec to compute time midpoints.")
#         J2000 = datetime(2000, 1, 1, 12, 0, 0)
#         df["mid_dt"] = df["start_sec"].astype(float).map(lambda s: J2000 + timedelta(seconds=s)) + \
#                        (df["end_sec"] - df["start_sec"]).astype(float).map(lambda s: timedelta(seconds=s/2))

#     df["weight_mean"] = (df["weight_ra"] + df["weight_dec"]) / 2.0

#     if split_by_id:
#         figs = {}
#         for file_id, g in df.groupby("id", sort=False):
#             fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
#             g = g.sort_values("mid_dt")
#             ax.plot(g["mid_dt"], g["weight_mean"], marker="o", linestyle="-", alpha=0.8)
#             ax.set_yscale("log")
#             ax.set_xlabel("Time")
#             ax.set_ylabel("Average Weight [1/rad²]")
#             ax.set_title(f"Average weight vs time — {file_id}")
#             ax.grid(True, alpha=0.25)
#             figs[file_id] = fig
#         return figs
#     else:
#         fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
#         for file_id, g in df.groupby("id", sort=False):
#             g = g.sort_values("mid_dt")
#             ax.plot(g["mid_dt"], g["weight_mean"], marker="o", linestyle="-", alpha=0.8, label=str(file_id))
#         ax.set_yscale("log")
#         ax.set_xlabel("Time")
#         ax.set_ylabel("Average Weight [1/rad²]")
#         ax.set_title("Average weight vs time (per-frame midpoint)")
#         ax.legend(title="ID", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
#         ax.grid(True, alpha=0.25)
#         return fig



def plot_average_weight(summary_or_weights: pd.DataFrame, *, split_by_id: bool = False,
            selected_key = None,logscale=True,title=None):
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
    need = {"id", "weight_ra", "weight_dec"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")

    # Compute average weight
    df["weight_mean"] = (df["weight_ra"] + df["weight_dec"]) / 2.0

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

                if selected_key is not None:
                    ax.plot(g["mid_dt"], g[selected_key], marker="o", linestyle="-", alpha=0.8, label=str(file_id))
                else:
                    ax.plot(g["mid_dt"], g["weight_mean"], marker="o", linestyle="-", alpha=0.8, label=str(file_id))
                    
            if logscale == True:  
                ax.set_yscale("log")
   
            ax.set_xlabel("Time")
            ax.set_ylabel("Average Weight [1/rad²]")
            if title is None:
                ax.set_title("Average weight vs time (per-frame midpoint)")
            else:
                ax.set_title(title)
                ax.set_ylabel("Number of observations [-]")
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
        - 'residual_ra_last_std': RA standard deviation
        - 'residual_dec_last_std': DEC standard deviation
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
    required = {'id', 'residual_ra_last_std', 'residual_dec_last_std'}
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
        ax1.plot(df_id['mid_dt'], df_id['residual_ra_last_std'], 
                marker='o', linestyle='-', color='C0', alpha=0.7)
        ax1.set_ylabel('RA Std [arcsec]', fontsize=11)
        ax1.set_title(f'Residual Standard Deviations — {file_id}', fontsize=13)
        ax1.grid(True, alpha=0.3)
        
        # DEC std
        ax2.plot(df_id['mid_dt'], df_id['residual_dec_last_std'], 
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
        ax1.plot(df_id['mid_dt'], df_id['residual_ra_last_std'], 
                marker='o', linestyle='-', alpha=0.7, label=str(file_id))
       
        # DEC std
        ax2.plot(df_id['mid_dt'], df_id['residual_dec_last_std'], 
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
        Must contain: id, datetime, residual_ra_last, residual_dec_last, timeframe
    summary : pd.DataFrame, optional
        Summary statistics with: id, timeframe, start_dt, residual_ra_last_std, residual_dec_last_std
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
        overall_ra_std = data['residual_ra_last'].std()
        overall_dec_std = data['residual_dec_last'].std()
        
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
        ax1.scatter(data['datetime'], data['residual_ra_last'], s=30, alpha=0.9)
        ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('RA Residual [arcsec]')
        ax1.set_title(f'RA Residuals: {obs_id}')
        ax1.grid(alpha=0.5)
        
        # Plot DEC residuals
        ax2.scatter(data['datetime'], data['residual_dec_last'], s=30, alpha=0.9)
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
            ax_sum_ra.plot(sum_data['start_dt'], sum_data['residual_ra_last_std'], 
                          marker='o', linewidth=2, markersize=8)
            ax_sum_ra.axhline(overall_ra_std, color='red', linestyle='--', 
                             linewidth=2, alpha=0.7, label=f'Overall Std: {overall_ra_std:.4f}"')
            ax_sum_ra.set_xlabel('Date')
            ax_sum_ra.set_ylabel('RA Residual Std [arcsec]')
            ax_sum_ra.set_title(f'RA Standard Deviation per Timeframe')
            ax_sum_ra.grid(alpha=0.5)
            #ax_sum_ra.legend()
            
            # DEC std over time
            ax_sum_dec.plot(sum_data['start_dt'], sum_data['residual_dec_last_std'], 
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
                ax_ra.scatter(tf_data['datetime'], tf_data['residual_ra_last'], 
                             s=30, alpha=0.9, color=color)
                ax_ra.axhline(0, color='black', linestyle='--', alpha=0.5)
                ax_ra.set_xlabel('Date')
                ax_ra.set_ylabel('RA Residual [arcsec]')
                ax_ra.set_title(f'Frame {tf_id} - RA (n={len(tf_data)}, duration: {duration})')
                ax_ra.grid(alpha=0.5)
                
                # DEC subplot for this timeframe
                ax_dec.scatter(tf_data['datetime'], tf_data['residual_dec_last'], 
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
    observations,observations_settings,observation_set_ids = ObsFunc.LoadObservations("Observations/MoreObservationsNovember/",system_of_bodies,file_names_loaded)

    print("Loaded tudat observations...")
    #--------------------------------------------------------------------------------------------
    # LOAD ESTIMATION RESULTS
    #--------------------------------------------------------------------------------------------
    arrays = load_npy_files("Results/BetterFigs/AllModernObservations/First") 
    
    print("Loaded numpy arrays from estimation...")
    nr_observations = []
    for list1 in arrays['residuals_sorted']:
        nr_observations.append(len(list1)/2)
    nr_observations_check = []

    for list1 in observations.get_observations():
        nr_observations_check.append(len(list1)/2) 

    df = create_observations_dataframe(
        arrays['observations_sorted'], 
        ordered_ids,
        arrays['observation_times'],
        arrays['residual_history_arcseconds'])
    
    #Load results without outliers
    arrays2 = load_npy_files("Results/BetterFigs/AllModernObservations/TimeFrameWeights_50mas_min")
    
    print("Loaded previous numpy arrays from estimation...")
    nr_observations = []
    for list1 in arrays2['residuals_sorted']:
        nr_observations.append(len(list1)/2)
    nr_observations_check = []

    for list1 in observations.get_observations():
        nr_observations_check.append(len(list1)/2) 

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


    
    #Create timeframes 
    df_frames, summary = split_into_timeframes(df, gap_hours=4)


    ##############################################################################################
    # PLOT FIGURES  
    ##############################################################################################
    
    #make folder
    # Load unique id colors (new one needs to be made)
    # with open('file_colors.pkl', 'rb') as f:
    #     file_colors = pickle.load(f)

    
    out_dir = make_timestamped_folder("Results/BetterFigs/AllModernObservations/PostProcessing")
    
    make_figures = True
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
        make_residual_figures = True
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
    # COMPUTE WEIGHTS
    #--------------------------------------------------------------------------------------------
    print("Computing weights...")

    #Compute std in arcseconds
    std_per_id = df.groupby("id")[["residual_ra_last", "residual_dec_last"]].std().rename(
        columns={"residual_ra_last": "std_ra", "residual_dec_last": "std_dec"}
    )

    # Files with 1 Id (NaN stds) have an STD assigned as 1 arcsecond
    std_per_id = std_per_id.fillna(1)

    # Convert to radians
    arcsec_to_rad = np.pi / (180.0 * 3600.0)
    std_per_id_rad = std_per_id * arcsec_to_rad

    # Compute weights = 1 / σ²
    weights = 1.0 / (std_per_id_rad ** 2)
    weights.columns = ["weight_ra", "weight_dec"]

    weight_for_arcsec = 1/arcsec_to_rad**2
    #weights.fillna(weight_for_arcsec, inplace=True)

    # Save to file
    weights.to_csv(out_dir / "weights.txt", sep="\t", float_format="%.8e")


    #--------------------------------------------------------------------------------------------
    # Replace NaNs stds of frames with 1 observation
    #--------------------------------------------------------------------------------------------
    replacement_for_id = {}
    usage_stats = {}

    DEFAULT = 1  # fallback if an id has no valid stds


    for id, g in summary.groupby("id"):
        total_obs = g["n_obs"].sum()

        # masks for valid stds
        ra_mask = g["residual_ra_last_std"].notna()
        dec_mask = g["residual_dec_last_std"].notna()

        # weighted sums and totals (ignore NaNs)
        ra_weight_sum = (g.loc[ra_mask, "residual_ra_last_std"] * g.loc[ra_mask, "n_obs"]).sum()
        ra_n_used = g.loc[ra_mask, "n_obs"].sum()
        dec_weight_sum = (g.loc[dec_mask, "residual_dec_last_std"] * g.loc[dec_mask, "n_obs"]).sum()
        dec_n_used = g.loc[dec_mask, "n_obs"].sum()

        # weighted means (fallback to weights DataFrame if nothing valid)
        ra_mean = ra_weight_sum / ra_n_used if ra_n_used > 0 else std_per_id.loc[id,'std_ra']
        dec_mean = dec_weight_sum / dec_n_used if dec_n_used > 0 else std_per_id.loc[id,'std_dec']

        replacement_for_id[id] = {
            "residual_ra_last_std": float(ra_mean),
            "residual_dec_last_std": float(dec_mean),
        }

        usage_stats[id] = {
            "total_obs": int(total_obs),
            "ra_used": int(ra_n_used),
            "ra_excluded": int(total_obs - ra_n_used),
            "dec_used": int(dec_n_used),
            "dec_excluded": int(total_obs - dec_n_used),
        }

    usage_df = pd.DataFrame.from_dict(usage_stats, orient='index')


    def fill_two_cols(group):
        rep = replacement_for_id[group.name]
        group[["residual_ra_last_std","residual_dec_last_std"]] = (
            group[["residual_ra_last_std","residual_dec_last_std"]].fillna(rep)
        )
        return group


    summary = summary.groupby("id", group_keys=False).apply(fill_two_cols)

    #--------------------------------------------------------------------------------------------
    # Created tabulated weights
    #--------------------------------------------------------------------------------------------
    #Compute weight per id and timeframe:
    min_sigma_arcsec = 0.05   # 10mas minimum



    sigma_ra = summary['residual_ra_last_std'].astype(float).copy()
    sigma_dec = summary['residual_dec_last_std'].astype(float).copy()

    if min_sigma_arcsec > 0:
        sigma_ra = sigma_ra.clip(lower=min_sigma_arcsec)
        sigma_dec = sigma_dec.clip(lower=min_sigma_arcsec)

    summary['residual_ra_last_std'] = sigma_ra
    summary['residual_dec_last_std'] = sigma_dec
    # arcsec -> rad
    arcsec_to_rad = np.pi / (180.0 * 3600.0)
    sigma_ra_rad  = sigma_ra * arcsec_to_rad
    sigma_dec_rad = sigma_dec * arcsec_to_rad

    # weights = 1 / σ²
    summary['weight_ra']  = 1.0 / (sigma_ra_rad ** 2)
    summary['weight_dec'] = 1.0 / (sigma_dec_rad ** 2)

    # Replace inf with NaN if any σ==0 survived (No NaNs should exist)
    #summary.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Compute per-id mean weights (ignoring NaNs)
    mean_weights = (
            summary.groupby("id")[["weight_ra", "weight_dec"]]
            .transform(lambda x: x.fillna(x.mean()))
            )

    # Replace NaNs in the original dataframe with these means
    #summary[["weight_ra", "weight_dec"]] = mean_weights

    summary['mean_weight'] = summary[['weight_ra', 'weight_dec']].mean(axis=1)
    
    #summary.loc[summary['mean_weight'].isna(), ['weight_ra', 'weight_dec', 'mean_weight']] = 1e10
    
    # Created scaled weights
    summary['weight_ra_scaled'] = summary['weight_ra'] / np.sqrt(summary['n_obs'])
    summary['weight_dec_scaled'] = summary['weight_dec'] / np.sqrt(summary['n_obs'])
    summary['mean_weight_scaled'] = summary[['weight_ra_scaled', 'weight_dec_scaled']].mean(axis=1)

    summary.to_csv(out_dir / "summary.txt", sep="\t", float_format="%.8e")


    
    # Assign weights
    observations,observations_settings,observation_set_ids = ObsFunc.LoadObservations(
            "Observations/RelativeObservations/",
            system_of_bodies,file_names_loaded,
            weights = summary,
            timeframe_weights=True)



    #tabulated_weights = []
    # for id_name in summary['id'].unique():
    #     group = summary[summary['id'] == id_name]
    #     expanded = group.loc[group.index.repeat(group['n_obs'])]
    #     weights = expanded['mean_weight'].values
    #     tabulated_weights.append(weights)

    # np.save(out_dir / 'tabulated_weights.npy', np.array(tabulated_weights, dtype=object))





    #--------------------------------------------------------------------------------------------
    # Plot Weight Figures
    #--------------------------------------------------------------------------------------------
    make_weight_figures = True
    if make_weight_figures == True:
        print("plotting weight figures...")
        fig_weights_per_time = plot_average_weight(summary, split_by_id=False)

        fig_weights_per_file = plot_average_weight_vs_time_from_weights(weights, summary, split_by_id=False)

        fig_std_per_timeframe = plot_residual_std(summary)

        fig_count_per_timeframe = plot_average_weight(summary, split_by_id = False,selected_key = "n_obs",logscale=False,title="Number of Observations per Timeframe")

        fig_per_night_weights = plot_average_weight(summary,split_by_id=False,selected_key="mean_weight_scaled",title="Descaled Weight per timeframe (night)")

        fig_weights_per_time.savefig(out_dir / "fig_weights_per_time.pdf")
        fig_weights_per_file.savefig(out_dir / "fig_weights_per_id.pdf")
        fig_std_per_timeframe.savefig(out_dir / "fig_std_per_timeframe.pdf")
        fig_count_per_timeframe.savefig(out_dir / "fig_count_per_timeframe.pdf")
        fig_per_night_weights.savefig(out_dir / "fig_weights_per_night.pdf")
        
    else:
        print("plotting disabled, not making weight figures.")


    print("end")