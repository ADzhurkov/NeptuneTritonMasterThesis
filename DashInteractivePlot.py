import dash
from dash import dcc, html, callback, Output, Input
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def epoch_to_datetime(epoch_seconds):
    """Convert seconds since J2000 to datetime"""
    j2000 = datetime(2000, 1, 1, 12, 0, 0)
    return j2000 + timedelta(seconds=epoch_seconds)

def rad_to_mas(radians):
    """Convert radians to milliarcseconds"""
    return radians * 206264806.0


#file_path = 'Results/PoleEstimationRealObservations/WeightTest_CASE1/initial_state_tf_weights/observation_weights.csv'
file_path = 'Results/PoleEstimationRealObservations/WeightTest_CASE1/initial_state_hybrid_weights/observation_weights.csv'
#file_path = 'Results/PoleEstimationRealObservations/WeightTest_CASE1/initial_state_hybrid_old_weights/observation_weights.csv'
#file_path = 'Results/PoleEstimationRealObservations/WeightTest_CASE1/initial_state_id_weights/observation_weights.csv'



weight_info = pd.read_csv(file_path)

# Assuming weight_info is your DataFrame - convert time to datetime
weight_info['datetime'] = weight_info['time'].apply(epoch_to_datetime)

# Convert residuals and rmse_id from radians to milliarcseconds
weight_info['ra_residual_mas'] = rad_to_mas(weight_info['ra_residual'])
weight_info['dec_residual_mas'] = rad_to_mas(weight_info['dec_residual'])
weight_info['ra_rmse_id_mas'] = rad_to_mas(weight_info['ra_rmse_id'])
weight_info['dec_rmse_id_mas'] = rad_to_mas(weight_info['dec_rmse_id'])

# Compute rmse per timeframe from weights
weight_info['ra_rmse_tf'] = 1.0 / np.sqrt(weight_info['weight_ra'])
weight_info['dec_rmse_tf'] = 1.0 / np.sqrt(weight_info['weight_dec'])
weight_info['ra_rmse_tf_mas'] = rad_to_mas(weight_info['ra_rmse_tf'])
weight_info['dec_rmse_tf_mas'] = rad_to_mas(weight_info['dec_rmse_tf'])

# Use Plotly's built-in qualitative color palettes
color_palette = (
    px.colors.qualitative.Dark24 + 
    px.colors.qualitative.Light24 + 
    px.colors.qualitative.Alphabet
)

def generate_extra_colors(n, start_index=0):
    """Generate additional distinct colors as hex codes"""
    import colorsys
    colors = []
    for i in range(n):
        hue = (start_index + i) / n
        rgb = colorsys.hls_to_rgb(hue, 0.5, 0.7)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        colors.append(hex_color)
    return colors

max_timeframes = weight_info['timeframe'].nunique()
if max_timeframes > len(color_palette):
    extra_colors = generate_extra_colors(max_timeframes - len(color_palette), len(color_palette))
    color_palette = color_palette + extra_colors

# Get unique values
ref_point_ids = sorted(weight_info['ref_point_id'].unique().tolist())
ref_point_ids_with_all = ['ALL FILES'] + ref_point_ids

# Create a color mapping for ref_point_ids (used in "ALL FILES" mode)
ref_point_color_map = {ref_id: color_palette[i % len(color_palette)] for i, ref_id in enumerate(ref_point_ids)}

# Y-axis options
y_options_ra = [
    {'label': 'RA Residual [mas]', 'value': 'ra_residual_mas'},
    {'label': 'RA RMSE ID [mas]', 'value': 'ra_rmse_id_mas'},
    {'label': 'RA RMSE Timeframe [mas]', 'value': 'ra_rmse_tf_mas'},
    {'label': 'RA Residual [rad]', 'value': 'ra_residual'},
    {'label': 'RA RMSE ID [rad]', 'value': 'ra_rmse_id'},
    {'label': 'Weight RA', 'value': 'weight_ra'},
]
y_options_dec = [
    {'label': 'DEC Residual [mas]', 'value': 'dec_residual_mas'},
    {'label': 'DEC RMSE ID [mas]', 'value': 'dec_rmse_id_mas'},
    {'label': 'DEC RMSE Timeframe [mas]', 'value': 'dec_rmse_tf_mas'},
    {'label': 'DEC Residual [rad]', 'value': 'dec_residual'},
    {'label': 'DEC RMSE ID [rad]', 'value': 'dec_rmse_id'},
    {'label': 'Weight DEC', 'value': 'weight_dec'},
]

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Weight Info Analysis", style={'textAlign': 'center'}),
    
    html.Div([
        # Row 1: Main dropdowns
        html.Div([
            # Dropdown for ref_point_id
            html.Div([
                html.Label("Select Reference Point ID:"),
                dcc.Dropdown(
                    id='ref-point-dropdown',
                    options=[{'label': ref_id, 'value': ref_id} for ref_id in ref_point_ids_with_all],
                    value=ref_point_ids[0],
                    clearable=False,
                    style={'width': '250px'}
                ),
            ], style={'display': 'inline-block', 'marginRight': '30px'}),
            
            # Dropdown for RA y-axis
            html.Div([
                html.Label("RA Y-Axis:"),
                dcc.Dropdown(
                    id='y-axis-ra-dropdown',
                    options=y_options_ra,
                    value='ra_residual_mas',
                    clearable=False,
                    style={'width': '250px'}
                ),
            ], style={'display': 'inline-block', 'marginRight': '30px'}),
            
            # Dropdown for DEC y-axis
            html.Div([
                html.Label("DEC Y-Axis:"),
                dcc.Dropdown(
                    id='y-axis-dec-dropdown',
                    options=y_options_dec,
                    value='dec_residual_mas',
                    clearable=False,
                    style={'width': '250px'}
                ),
            ], style={'display': 'inline-block'}),
        ], style={'marginBottom': '15px'}),
        
        # Row 2: Toggle options
        html.Div([
            html.Div([
                dcc.Checklist(
                    id='show-timeframe-lines',
                    options=[{'label': ' Show timeframe boundaries', 'value': 'show'}],
                    value=[],
                    style={'display': 'inline-block', 'marginRight': '30px'}
                ),
            ], style={'display': 'inline-block'}),
            
            html.Div([
                html.Label("Line style:", style={'marginRight': '10px'}),
                dcc.RadioItems(
                    id='line-style',
                    options=[
                        {'label': 'Solid', 'value': 'solid'},
                        {'label': 'Dashed', 'value': 'dash'},
                        {'label': 'Dotted', 'value': 'dot'},
                    ],
                    value='dash',
                    inline=True,
                    style={'display': 'inline-block'}
                ),
            ], style={'display': 'inline-block', 'marginLeft': '20px'}),
            
            html.Div([
                dcc.Checklist(
                    id='log-y-axis',
                    options=[{'label': ' Log Y-Axis', 'value': 'log'}],
                    value=[],
                    style={'display': 'inline-block', 'marginLeft': '30px'}
                ),
            ], style={'display': 'inline-block'}),
        ], style={'marginBottom': '15px'}),
        
        # Row 3: Highlight dropdown (only visible when "ALL FILES" is selected)
        html.Div([
            html.Div([
                html.Label("Highlight File (ALL FILES mode):", style={'marginRight': '10px'}),
                dcc.Dropdown(
                    id='highlight-dropdown',
                    options=[{'label': 'None', 'value': 'none'}] + [{'label': ref_id, 'value': ref_id} for ref_id in ref_point_ids],
                    value='none',
                    clearable=False,
                    style={'width': '250px'}
                ),
            ], style={'display': 'inline-block'}),
        ], id='highlight-row', style={'display': 'none'}),
        
    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'marginBottom': '20px'}),
    
    # Graph
    dcc.Graph(
        id='main-graph',
        style={'height': '800px'},
        config={
            'scrollZoom': True,
            'displayModeBar': True,
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'weight_info_plot',
                'height': 800,
                'width': 1600,
                'scale': 2
            }
        }
    ),
    
    # Info panel
    html.Div(id='info-panel', style={'padding': '20px', 'backgroundColor': '#e9ecef', 'marginTop': '10px'})
])


@callback(
    Output('highlight-row', 'style'),
    Input('ref-point-dropdown', 'value'),
)
def toggle_highlight_dropdown(ref_point_id):
    """Show/hide the highlight dropdown based on whether ALL FILES is selected"""
    if ref_point_id == 'ALL FILES':
        return {'display': 'block', 'marginTop': '15px'}
    return {'display': 'none'}


@callback(
    Output('log-y-axis', 'value'),
    Input('y-axis-ra-dropdown', 'value'),
    Input('y-axis-dec-dropdown', 'value'),
)
def auto_enable_log_for_weights(y_col_ra, y_col_dec):
    """Automatically enable log scale when weight columns are selected"""
    if 'weight' in y_col_ra.lower() or 'weight' in y_col_dec.lower():
        return ['log']
    return []


@callback(
    Output('main-graph', 'figure'),
    Output('info-panel', 'children'),
    Input('ref-point-dropdown', 'value'),
    Input('y-axis-ra-dropdown', 'value'),
    Input('y-axis-dec-dropdown', 'value'),
    Input('show-timeframe-lines', 'value'),
    Input('line-style', 'value'),
    Input('log-y-axis', 'value'),
    Input('highlight-dropdown', 'value'),
)
def update_graph(ref_point_id, y_col_ra, y_col_dec, show_tf_lines, line_style, log_y, highlight_file):
    # Determine if we're in "ALL FILES" mode
    all_files_mode = ref_point_id == 'ALL FILES'
    
    if all_files_mode:
        df_filtered = weight_info.copy()
    else:
        df_filtered = weight_info[weight_info['ref_point_id'] == ref_point_id].copy()
    
    n_points = len(df_filtered)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[f'RA ({y_col_ra})', f'DEC ({y_col_dec})'],
        shared_xaxes=True,
        vertical_spacing=0.08
    )
    
    if all_files_mode:
        # In ALL FILES mode, color by ref_point_id
        unique_ref_points = df_filtered['ref_point_id'].unique()
        n_files = len(unique_ref_points)
        
        for ref_id in unique_ref_points:
            df_ref = df_filtered[df_filtered['ref_point_id'] == ref_id]
            color = ref_point_color_map[ref_id]
            
            # Determine opacity and size based on highlight
            if highlight_file == 'none':
                opacity = 0.7
                size = 5
            elif highlight_file == ref_id:
                opacity = 1.0
                size = 8
            else:
                opacity = 0.2
                size = 4
            
            # Create hover text
            hover_text = [
                f"<b>File:</b> {ref_id}<br>"
                f"<b>Timeframe:</b> {row['timeframe']}<br>"
                f"<b>Obs Index:</b> {row['obs_index']}<br>"
                f"<b>Global Obs Index:</b> {row['global_obs_index']}<br>"
                f"<b>Time:</b> {row['datetime'].strftime('%Y-%m-%d %H:%M:%S')}<br>"
                f"<b>RA Residual:</b> {row['ra_residual_mas']:.4f} mas<br>"
                f"<b>DEC Residual:</b> {row['dec_residual_mas']:.4f} mas<br>"
                f"<b>RA RMSE ID:</b> {row['ra_rmse_id_mas']:.4f} mas<br>"
                f"<b>DEC RMSE ID:</b> {row['dec_rmse_id_mas']:.4f} mas<br>"
                f"<b>RA RMSE TF:</b> {row['ra_rmse_tf_mas']:.4f} mas<br>"
                f"<b>DEC RMSE TF:</b> {row['dec_rmse_tf_mas']:.4f} mas<br>"
                f"<b>Weight RA:</b> {row['weight_ra']:.6e}<br>"
                f"<b>Weight DEC:</b> {row['weight_dec']:.6e}"
                for _, row in df_ref.iterrows()
            ]
            
            # RA subplot
            fig.add_trace(
                go.Scattergl(
                    x=df_ref['datetime'],
                    y=df_ref[y_col_ra],
                    mode='markers',
                    name=ref_id,
                    marker=dict(color=color, size=size, opacity=opacity),
                    legendgroup=ref_id,
                    hovertemplate='%{customdata}<extra></extra>',
                    customdata=hover_text
                ),
                row=1, col=1
            )
            
            # DEC subplot
            fig.add_trace(
                go.Scattergl(
                    x=df_ref['datetime'],
                    y=df_ref[y_col_dec],
                    mode='markers',
                    name=ref_id,
                    marker=dict(color=color, size=size, opacity=opacity),
                    legendgroup=ref_id,
                    showlegend=False,
                    hovertemplate='%{customdata}<extra></extra>',
                    customdata=hover_text
                ),
                row=2, col=1
            )
        
        title_text = 'All Reference Points'
        if highlight_file != 'none':
            title_text += f' (Highlighting: {highlight_file})'
        
        # Info for all files mode
        date_range = f"{df_filtered['datetime'].min().strftime('%Y-%m-%d')} to {df_filtered['datetime'].max().strftime('%Y-%m-%d')}"
        info_content = [
            html.B("Data Summary: "),
            f"{n_points} points | {n_files} files | Date range: {date_range}",
            html.Br(),
            html.B("Controls: "),
            "Scroll to zoom | Click+drag to pan | Double-click to reset | Use range slider at bottom for overview",
            html.Br(),
            html.B("Tip: "),
            "Use the 'Highlight File' dropdown to focus on a specific file while seeing all data for context",
            html.Br(),
            html.B("Units: "),
            "1 mas (milliarcsecond) = 4.848e-9 rad"
        ]
        
    else:
        # Single file mode - color by timeframe
        timeframes = sorted(df_filtered['timeframe'].unique())
        n_timeframes = len(timeframes)
        tf_to_color = {tf: i for i, tf in enumerate(timeframes)}
        
        # Calculate timeframe boundaries for vertical lines
        tf_boundaries = []
        if 'show' in show_tf_lines:
            for tf in timeframes:
                df_tf = df_filtered[df_filtered['timeframe'] == tf]
                tf_start = df_tf['datetime'].min()
                tf_end = df_tf['datetime'].max()
                tf_boundaries.append({
                    'timeframe': tf,
                    'start': tf_start,
                    'end': tf_end,
                    'color': color_palette[tf_to_color[tf] % len(color_palette)]
                })
        
        # Add traces for each timeframe
        for tf in timeframes:
            df_tf = df_filtered[df_filtered['timeframe'] == tf]
            color = color_palette[tf_to_color[tf] % len(color_palette)]
            
            hover_text = [
                f"<b>Timeframe:</b> {tf}<br>"
                f"<b>Obs Index:</b> {row['obs_index']}<br>"
                f"<b>Global Obs Index:</b> {row['global_obs_index']}<br>"
                f"<b>Time:</b> {row['datetime'].strftime('%Y-%m-%d %H:%M:%S')}<br>"
                f"<b>RA Residual:</b> {row['ra_residual_mas']:.4f} mas<br>"
                f"<b>DEC Residual:</b> {row['dec_residual_mas']:.4f} mas<br>"
                f"<b>RA RMSE ID:</b> {row['ra_rmse_id_mas']:.4f} mas<br>"
                f"<b>DEC RMSE ID:</b> {row['dec_rmse_id_mas']:.4f} mas<br>"
                f"<b>RA RMSE TF:</b> {row['ra_rmse_tf_mas']:.4f} mas<br>"
                f"<b>DEC RMSE TF:</b> {row['dec_rmse_tf_mas']:.4f} mas<br>"
                f"<b>Weight RA:</b> {row['weight_ra']:.6e}<br>"
                f"<b>Weight DEC:</b> {row['weight_dec']:.6e}"
                for _, row in df_tf.iterrows()
            ]
            
            # RA subplot
            fig.add_trace(
                go.Scattergl(
                    x=df_tf['datetime'],
                    y=df_tf[y_col_ra],
                    mode='markers',
                    name=f'TF {tf}',
                    marker=dict(color=color, size=6),
                    legendgroup=f'tf{tf}',
                    hovertemplate='%{customdata}<extra></extra>',
                    customdata=hover_text
                ),
                row=1, col=1
            )
            
            # DEC subplot
            fig.add_trace(
                go.Scattergl(
                    x=df_tf['datetime'],
                    y=df_tf[y_col_dec],
                    mode='markers',
                    name=f'TF {tf}',
                    marker=dict(color=color, size=6),
                    legendgroup=f'tf{tf}',
                    showlegend=False,
                    hovertemplate='%{customdata}<extra></extra>',
                    customdata=hover_text
                ),
                row=2, col=1
            )
        
        # Add vertical lines for timeframe boundaries
        if 'show' in show_tf_lines:
            for boundary in tf_boundaries:
                fig.add_vline(
                    x=boundary['start'],
                    line=dict(color=boundary['color'], width=1, dash=line_style),
                    row=1, col=1,
                    opacity=0.7
                )
                fig.add_vline(
                    x=boundary['end'],
                    line=dict(color=boundary['color'], width=1, dash=line_style),
                    row=1, col=1,
                    opacity=0.7
                )
                fig.add_vline(
                    x=boundary['start'],
                    line=dict(color=boundary['color'], width=1, dash=line_style),
                    row=2, col=1,
                    opacity=0.7
                )
                fig.add_vline(
                    x=boundary['end'],
                    line=dict(color=boundary['color'], width=1, dash=line_style),
                    row=2, col=1,
                    opacity=0.7
                )
        
        title_text = f'Reference Point: {ref_point_id}'
        
        # Info for single file mode
        date_range = f"{df_filtered['datetime'].min().strftime('%Y-%m-%d')} to {df_filtered['datetime'].max().strftime('%Y-%m-%d')}"
        info_content = [
            html.B("Data Summary: "),
            f"{n_points} points | {n_timeframes} timeframes | Date range: {date_range}",
            html.Br(),
            html.B("Controls: "),
            "Scroll to zoom | Click+drag to pan | Double-click to reset | Use range slider at bottom for overview",
            html.Br(),
            html.B("Units: "),
            "1 mas (milliarcsecond) = 4.848e-9 rad"
        ]
    
    # Determine y-axis type
    y_axis_type = 'log' if 'log' in log_y else 'linear'
    
    # Determine y-axis labels with units
    def get_ylabel(col_name):
        if col_name.endswith('_mas'):
            return col_name.replace('_mas', '') + ' [mas]'
        elif 'weight' in col_name.lower():
            return col_name
        else:
            return col_name + ' [rad]'
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            font=dict(size=20)
        ),
        hovermode='closest',
        legend=dict(
            title='Files' if all_files_mode else 'Timeframes',
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='lightgray',
            borderwidth=1,
            itemsizing='constant',
            tracegroupgap=2
        ),
        margin=dict(r=150),
        
        xaxis2=dict(
            rangeslider=dict(visible=True, thickness=0.05),
            type='date',
            title='DateTime'
        ),
        
        yaxis=dict(title=get_ylabel(y_col_ra), exponentformat='e', type=y_axis_type),
        yaxis2=dict(title=get_ylabel(y_col_dec), exponentformat='e', type=y_axis_type),
        
        uirevision=ref_point_id
    )
    
    # Update axes for both subplots
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikecolor='gray',
        spikethickness=1
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikecolor='gray',
        spikethickness=1
    )
    
    return fig, info_content


if __name__ == '__main__':
    app.run(debug=True, port=8050)