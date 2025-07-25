import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import base64
from io import BytesIO
import pandas as pd
import requests
from dash import Dash, html, dcc, dash_table, Input, Output, State
from MAC_module import run_mac
import plotly.graph_objects as go
from plotly.subplots import make_subplots


color_dict = {"Fastball": "red", "Breaking": "blue", "Breaking1": "blue", "Breaking2": "cyan", "Offspeed": "green"}

# === CONFIG ===
base_path = "./output"
os.makedirs(base_path, exist_ok=True)

# Dropbox URLs - Replace these with your actual Dropbox direct download links
NCAA_DROPBOX_URL = "https://www.dropbox.com/scl/fi/xayqylfb2d8wnqr4p5jua/CCBL_current.parquet?rlkey=e1mqyzpgvp68iq1w1j171q3i6&st=ilngpt9n&dl=1"
CCBL_DROPBOX_URL = "https://www.dropbox.com/scl/fi/xayqylfb2d8wnqr4p5jua/CCBL_current.parquet?rlkey=e1mqyzpgvp68iq1w1j171q3i6&st=ilngpt9n&dl=1"

def download_file_from_dropbox(url, local_filename):
    """Download file from Dropbox if not already cached locally"""
    if os.path.exists(local_filename):
        print(f"Loading cached {local_filename}")
        return local_filename
    
    print(f"Downloading {local_filename} from Dropbox...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Save locally for faster subsequent loads
        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Successfully downloaded {local_filename}")
        return local_filename
        
    except Exception as e:
        print(f"Error downloading file {local_filename}: {e}")
        raise

def load_combined_data():
    """Load both NCAA and CCBL data from Dropbox"""
    try:
        # Download files if not cached
        ncaa_file = download_file_from_dropbox(NCAA_DROPBOX_URL, "NCAA_final.parquet")
        ccbl_file = download_file_from_dropbox(CCBL_DROPBOX_URL, "CCBL_current.parquet")
        
        # Load parquet files
        ncaa_df = pd.read_parquet(ncaa_file)
        ccbl_df = pd.read_parquet(ccbl_file)
        
        print(f"Loaded NCAA data: {len(ncaa_df)} rows")
        print(f"Loaded CCBL data: {len(ccbl_df)} rows")
        
        # Combine datasets
        combined_df = pd.concat([ncaa_df, ccbl_df], ignore_index=True)
        print(f"Combined dataset: {len(combined_df)} rows")
        
        return combined_df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        # Return empty DataFrame as fallback
        return pd.DataFrame()

# Load data on app startup
print("Loading data from Dropbox...")
df_all = load_combined_data()

if df_all.empty:
    print("WARNING: No data loaded. App may not function properly.")
else:
    print(f"Successfully loaded {len(df_all)} total rows")

# === APP ===
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "MAC Interactive Visualizer"

app.layout = html.Div([
    html.H2("MAC Matchup Calculator", style={"textAlign": "center"}),

    html.Label("Select Pitcher:"),
    dcc.Dropdown(
        id="pitcher-dropdown", 
        options=[{"label": p, "value": p} for p in sorted(df_all["Pitcher"].dropna().unique())] if not df_all.empty else [],
        placeholder="Select a pitcher..." if not df_all.empty else "Loading data..."
    ),

    html.Label("Select Hitters:"),
    dcc.Dropdown(
        id="hitter-dropdown", 
        options=[{"label": h, "value": h} for h in sorted(df_all["Batter"].dropna().unique())] if not df_all.empty else [],
        multi=True,
        placeholder="Select hitters..." if not df_all.empty else "Loading data..."
    ),

    html.Button("Run Matchup", id="run-button", n_clicks=0),
    html.Div(id="output-status", style={"marginTop": "10px"}),
    html.Hr(),
    html.Div(id="main-visual-output"),
])


def compute_heatmap_stats(df, metric_col, min_samples=3):
    import numpy as np
    from scipy import ndimage
    from scipy.interpolate import griddata

    valid = df[["PlateLocSide", "PlateLocHeight", metric_col]].dropna()
    if len(valid) < min_samples:
        return None, None, None

    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(0.5, 4.5, 100)
    X, Y = np.meshgrid(x_range, y_range)

    try:
        points = valid[["PlateLocSide", "PlateLocHeight"]].values
        values = valid[metric_col].values
        Z = griddata(points, values, (X, Y), method='linear', fill_value=0)

        if len(valid) < 10:
            sigma = 0.5
        elif len(valid) < 25:
            sigma = 1.0
        else:
            sigma = 1.5

        Z_smooth = ndimage.gaussian_filter(Z, sigma=sigma, mode='constant', cval=0)

        mask = np.zeros_like(Z_smooth)
        for i in range(len(x_range)):
            for j in range(len(y_range)):
                dist = np.sqrt((points[:, 0] - x_range[i])**2 + (points[:, 1] - y_range[j])**2)
                if np.min(dist) < 0.8:
                    mask[j, i] = 1

        Z_smooth *= mask
        Z_smooth[Z_smooth < 0.01] = 0
        return x_range, y_range, Z_smooth
    except Exception as e:
        print(f"Heatmap error: {e}")
        return None, None, None

def generate_mpl_heatmap_grid(df, selected_hitter):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import base64
    from io import BytesIO
    import numpy as np

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    metrics = [("WhiffFlag", "Whiff Rate"), ("HardHitFlag", "Hard Hit Rate"), ("wOBA_result", "wOBA")]
    pitch_groups = ["Fastball", "Breaking", "Offspeed"]

    for i, (metric, title) in enumerate(metrics):
        for j, group in enumerate(pitch_groups):
            ax = axes[i, j]
            subset = df[df["PitchGroup"] == group].copy()

            if len(subset) == 0:
                ax.text(0, 2.75, "No Data", ha='center', va='center', fontsize=12)
            else:
                x_range, y_range, z = compute_heatmap_stats(subset, metric)

                if z is not None and np.any(z > 0):
                    X, Y = np.meshgrid(x_range, y_range)

                    if metric == "wOBA_result":
                        actual_max = np.max(z[z > 0])
                        vmax = min(actual_max * 1.1, 1.8)
                        vmin = 0
                        cmap = "RdYlBu_r"
                        levels = np.linspace(vmin, vmax, 20)
                    else:
                        vmin, vmax = 0, 1
                        cmap = "RdYlBu_r"
                        levels = np.linspace(0, 1, 20)

                    cs = ax.contourf(X, Y, z, levels=levels, cmap=cmap,
                                     vmin=vmin, vmax=vmax, alpha=0.8, extend='both')
                    ax.contour(X, Y, z, levels=levels[::4], colors='white',
                               linewidths=0.5, alpha=0.3)

                    if j == 2:
                        cbar = fig.colorbar(cs, ax=ax, shrink=0.6)
                        cbar.set_label(title, rotation=270, labelpad=15)
                else:
                    valid = subset[["PlateLocSide", "PlateLocHeight", metric]].dropna()
                    if not valid.empty:
                        if metric in ["WhiffFlag", "HardHitFlag"]:
                            colors = ['lightblue' if x == 0 else 'red' for x in valid[metric]]
                            ax.scatter(valid["PlateLocSide"], valid["PlateLocHeight"],
                                       c=colors, s=40, alpha=0.7, edgecolors='black')
                        else:
                            vmin, vmax = 0, 1 if metric != "wOBA_result" else min(valid[metric].max() * 1.1, 1.8)
                            sc = ax.scatter(valid["PlateLocSide"], valid["PlateLocHeight"],
                                            c=valid[metric], cmap="RdYlBu_r", s=60,
                                            edgecolors="black", alpha=0.8,
                                            vmin=0, vmax=vmax)
                            if j == 2:
                                fig.colorbar(sc, ax=ax, shrink=0.6)

            ax.add_patch(Rectangle((-0.83, 1.5), 1.66, 1.8775, linewidth=2.5,
                                   edgecolor='black', facecolor='none'))

            ax.set_xlim([-1.5, 1.5])
            ax.set_ylim([1.0, 4.0])
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("#f8f9fa")

            if j == 0:
                ax.set_ylabel(title, fontsize=12, fontweight='bold')
            if i == 2:
                ax.set_xlabel(group, fontsize=12, fontweight='bold')

    fig.suptitle(f"Zone-Level Heat Maps for {selected_hitter}", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


# === CALLBACK ===
@app.callback(
    Output("output-status", "children"),
    Output("main-visual-output", "children"),
    Input("run-button", "n_clicks"),
    State("pitcher-dropdown", "value"),
    State("hitter-dropdown", "value")
)
def run_mac_pipeline(n_clicks, pitcher_name, hitter_list):
    if n_clicks == 0 or not pitcher_name or not hitter_list:
        return "", None

    if df_all.empty:
        return "❌ Error: No data available. Please check data loading.", None

    try:
        run_mac(pitcher_name, hitter_list, df_all, base_path)
        last_first = pitcher_name.replace(", ", "_")
        summary_path = os.path.join(base_path, f"{last_first}_summary.csv")
        breakdown_path = os.path.join(base_path, f"{last_first}_group_breakdown.csv")
        movement_path = os.path.join(base_path, f"{last_first}_pitch_level_filtered.csv")

        summary_df = pd.read_csv(summary_path)
        breakdown_df = pd.read_csv(breakdown_path)
        movement_df = pd.read_csv(movement_path)

        summary_df = summary_df[summary_df["Batter"].isin(hitter_list)].copy()
        breakdown_df = breakdown_df[breakdown_df["Batter"].isin(hitter_list)].copy()
        movement_df = movement_df[movement_df["Batter"].isin(hitter_list)].copy()

        for df in [summary_df, breakdown_df, movement_df]:
            df["Batter"] = pd.Categorical(df["Batter"], categories=hitter_list, ordered=True)
            df.sort_values("Batter", inplace=True)

        # CREATE PLOTLY SCATTER PLOT (REPLACE THE fig_data SECTION)
        fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scatter"}]])

        # Add summary points with full hover info
        for _, row in summary_df.iterrows():
            fig.add_trace(go.Scatter(
                x=[row["Batter"]],
                y=[row["RV/100"]],
                mode="markers",
                marker=dict(size=20, color="black"),
                hovertemplate=(
                    f"<b>{row['Batter']}</b><br>"
                    f"RV/100: {row['RV/100']}<br>"
                    f"wOBA: {row['wOBA']}<br>"
                    f"AVG: {row['AVG']}<br>"
                    f"Whiff%: {row['Whiff%']}<br>"
                    f"SwStr%: {row['SwStr%']}<br>"
                    f"HH%: {row['HH%']}<br>"
                    f"GB%: {row['GB%']}<br>"
                    f"ExitVelo: {row['ExitVelo']}<br>"
                    f"Launch: {row['Launch']}<br>"
                    f"Pitches: {row['Pitches']}<br>"
                    f"InPlay: {row['InPlay']}<extra></extra>"
                ),
                showlegend=False
            ))

        # Add breakdown points with full hover info
        for _, row in breakdown_df.iterrows():
            fig.add_trace(go.Scatter(
                x=[row["Batter"]],
                y=[row["RV/100"]],
                mode="markers+text",
                marker=dict(size=14, color=color_dict.get(row["PitchGroup"], "gray")),
                text=[f"{int(row['Pitches'])}P"],
                textposition="top center",
                textfont=dict(size=10, color="black"),
                hovertemplate=(
                    f"<b>{row['Batter']}</b><br>"
                    f"PitchGroup: {row['PitchGroup']}<br>"
                    f"RV/100: {row['RV/100']}<br>"
                    f"wOBA: {row['wOBA']}<br>"
                    f"AVG: {row['AVG']}<br>"
                    f"Whiff%: {row['Whiff%']}<br>"
                    f"SwStr%: {row['SwStr%']}<br>"
                    f"HH%: {row['HH%']}<br>"
                    f"GB%: {row['GB%']}<br>"
                    f"ExitVelo: {row['ExitVelo']}<br>"
                    f"Launch: {row['Launch']}<br>"
                    f"Pitches: {row['Pitches']}<br>"
                    f"InPlay: {row['InPlay']}<extra></extra>"
                ),
                showlegend=False
            ))

        fig.update_layout(
            height=700,
            title=f"Expected Matchup RV/100 + Hitter Summary: {pitcher_name} - - - Note that the black indicates expected weighted performance and the red, blue, and green dots represent fastballs, breaking balls, and offspeed, respectively",
            yaxis_title="RV/100",
            template="simple_white",
            xaxis=dict(
                tickmode="array",
                tickvals=list(range(len(summary_df))),
                ticktext=summary_df["Batter"].tolist(),
                tickangle=45
            )
        )

        # Add annotations for better/worse indicators
        fig.add_annotation(
            xref="paper", yref="y",
            x=1.02, y=summary_df["RV/100"].max() + 1 if not summary_df.empty else 5,
            text="↑ Better for Hitters",
            showarrow=False,
            font=dict(size=20, color="green"),
            align="left"
        )

        fig.add_annotation(
            xref="paper", yref="y",
            x=1.02, y=summary_df["RV/100"].min() - 1 if not summary_df.empty else -5,
            text="↓ Worse for Hitters",
            showarrow=False,
            font=dict(size=20, color="red"),
            align="left"
        )

        fig_table = dash_table.DataTable(
            data=summary_df.to_dict("records"),
            columns=[{"name": col, "id": col} for col in summary_df.columns],
            style_table={"overflowX": "auto", "marginTop": "20px"},
            style_cell={"textAlign": "center"}
        )

        heatmap_dropdown = dcc.Dropdown(
            id="heatmap-hitter-dropdown",
            options=[{"label": h, "value": h} for h in hitter_list],
            value=hitter_list[0],
            style={"width": "60%"}
        )
                # === Movement Plot ===
        movement_df_filtered = movement_df[(movement_df["HorzBreak"].between(-50, 50)) & (movement_df["InducedVertBreak"].between(-50, 50))]
        movement_fig = go.Figure()
        for pitch_type, color in color_dict.items():
            pitch_df = movement_df_filtered[movement_df_filtered["PitchGroup"] == pitch_type]
            if not pitch_df.empty:
                movement_fig.add_trace(go.Scatter(
                    x=pitch_df["HorzBreak"],
                    y=pitch_df["InducedVertBreak"],
                    mode="markers",
                    marker=dict(color=color, size=10),
                    name=pitch_type,
                    customdata=pitch_df[["Batter", "RelSpeed", "SpinRate"]],
                    hovertemplate="<b>%{customdata[0]}</b><br>"
                                  "HB: %{x}<br>"
                                  "IVB: %{y}<br>"
                                  "RelSpeed: %{customdata[1]} mph<br>"
                                  "SpinRate: %{customdata[2]}<extra></extra>"
                ))
        movement_fig.update_layout(
            title="Pitch Movement (HorzBreak vs. InducedVertBreak)",
            xaxis=dict(title="Horizontal Break", range=[-30, 30]),
            yaxis=dict(title="Induced Vertical Break", range=[-30, 30], scaleanchor="x", scaleratio=1),
            template="simple_white",
            height=600,
            width=1000
        )

        return f"✅ CSV generated for {pitcher_name} vs {len(hitter_list)} hitters.", html.Div([
            dcc.Graph(figure=fig),  # Main scatter plot
            html.Hr(),
            html.H3("Full Stats Table"),
            fig_table,
            html.Hr(),
            html.H3("Select Hitter for Zone Heatmap:"),
            heatmap_dropdown,
            html.Div(id="heatmap-image-output"),
            html.Hr(),
            html.H3("Pitch Movement Chart"),
            dcc.Graph(figure=movement_fig)  # Add the movement chart here
        ])

    except Exception as e:
        return f"❌ Error: {str(e)}", None


@app.callback(
    Output("heatmap-image-output", "children"),
    Input("heatmap-hitter-dropdown", "value"),
    State("pitcher-dropdown", "value")
)
def render_heatmap(hitter_name, pitcher_name):
    if not hitter_name or not pitcher_name:
        return html.Div("No data available.")

    last_first = pitcher_name.replace(", ", "_")
    movement_path = os.path.join(base_path, f"{last_first}_pitch_level_filtered.csv")
    
    if not os.path.exists(movement_path):
        return html.Div("Please run a matchup analysis first.")
    
    df = pd.read_csv(movement_path)
    df = df[df["Batter"] == hitter_name].copy()

    df["WhiffFlag"] = (df["PitchCall"] == "StrikeSwinging").astype(int)
    df["HardHitFlag"] = ((df["ExitSpeed"] >= 95) & df["ExitSpeed"].notna()).astype(int)

    encoded_img = generate_mpl_heatmap_grid(df, hitter_name)
    return html.Img(src=encoded_img, style={"width": "100%", "maxWidth": "1200px", "marginTop": "20px"})

# Production server configuration
if __name__ == "__main__":
    # For local development
    if os.getenv('RENDER'):
        # Production settings for Render
        app.run_server(
            host='0.0.0.0', 
            port=int(os.environ.get('PORT', 10000)), 
            debug=False
        )
    else:
        # Local development settings
        app.run_server(debug=True, port=8052)
