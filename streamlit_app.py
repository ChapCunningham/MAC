import streamlit as st
import traceback

try:
    import os
    #import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import base64
    from io import BytesIO
    from scipy import ndimage
    from scipy.interpolate import griddata
    from MAC_module import run_mac
    import tempfile
    import requests
    import pyarrow

    st.success("All imports successful!")

except Exception as e:
    st.error("Error during startup.")
    st.text(traceback.format_exc())
    st.stop()

# === CONFIG ===
st.set_page_config(
    page_title="MAC Matchup Calculator",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color dictionary for pitch types
color_dict = {"Fastball": "red", "Breaking": "blue", "Breaking1": "blue", "Breaking2": "cyan", "Offspeed": "green"}

# Data paths (update these to your actual paths)
# NCAA_PARQUET = "NCAA_final.parquet"   # https://www.dropbox.com/scl/fi/zozfzz75hamjsx5amp65b/NCAA_final.parquet?rlkey=nalex56psi9rj62fnyo5jhqt5&st=zm9f3dbm&dl=1
CCBL_PARQUET = "CCBL_current.parquet"
base_path = "./output"
os.makedirs(base_path, exist_ok=True)



@st.cache_data
def load_combined_data():
    try:
        # Download the NCAA file from Dropbox
        DROPBOX_NCAA_URL = "https://www.dropbox.com/scl/fi/zozfzz75hamjsx5amp65b/NCAA_final.parquet?rlkey=nalex56psi9rj62fnyo5jhqt5&st=zm9f3dbm&dl=1"
        with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp_file:
            response = requests.get(DROPBOX_NCAA_URL)
            if response.status_code != 200:
                st.error("Failed to download NCAA data from Dropbox.")
                return pd.DataFrame()
            tmp_file.write(response.content)
            tmp_file.flush()
            ncaa_df = pd.read_parquet(tmp_file.name, engine = "pyarrow")

        ccbl_df = pd.read_parquet(CCBL_PARQUET, engine = "pyarrow")
        return pd.concat([ncaa_df, ccbl_df], ignore_index=True)
    except Exception as e:
        st.error(f"Data load error: {e}")
        return pd.DataFrame()


def compute_heatmap_stats(df, metric_col, min_samples=3):
    """Compute heatmap statistics for zone visualization"""
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
        st.error(f"Heatmap error: {e}")
        return None, None, None

def generate_mpl_heatmap_grid(df, selected_hitter):
    """Generate matplotlib heatmap grid"""
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

def create_scatter_plot(summary_df, breakdown_df, pitcher_name):
    """Create the main RV/100 scatter plot"""
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
        title=f"Expected Matchup RV/100 + Hitter Summary: {pitcher_name}<br><sub>Black dots = weighted performance | Red/Blue/Green = Fastball/Breaking/Offspeed</sub>",
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

    return fig

def create_movement_plot(movement_df):
    """Create pitch movement scatter plot"""
    movement_df_filtered = movement_df[(movement_df["HorzBreak"].between(-50, 50)) & 
                                      (movement_df["InducedVertBreak"].between(-50, 50))]
    
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
    
    return movement_fig

# === MAIN APP ===
def main():
    st.title("⚾ MAC Matchup Calculator")
    st.markdown("---")

    with st.spinner("Loading data..."):
        try:
            if not os.path.exists(CCBL_PARQUET):
                st.error(f"Missing file: {CCBL_PARQUET}")
                st.stop()

            df_all = load_combined_data()
            if df_all.empty:
                st.error("Data load returned an empty DataFrame.")
                st.stop()
            else:
                st.success(f"Loaded {len(df_all)} rows.")
        except Exception as e:
            st.error("❌ Exception while loading data.")
            st.text(traceback.format_exc())
            st.stop()


    
    
    # Sidebar for inputs
    st.sidebar.header("Select Matchup")
    
    # Pitcher selection
    pitcher_options = sorted(df_all["Pitcher"].dropna().unique())
    selected_pitcher = st.sidebar.selectbox(
        "Select Pitcher:",
        options=pitcher_options,
        index=0 if pitcher_options else None
    )
    
    # Hitter selection
    hitter_options = sorted(df_all["Batter"].dropna().unique())
    selected_hitters = st.sidebar.multiselect(
        "Select Hitters:",
        options=hitter_options,
        help="Select hitters in lineup order"
    )
    
    # Run analysis button
    run_analysis = st.sidebar.button("🚀 Run Matchup Analysis", type="primary")
    
    if run_analysis and selected_pitcher and selected_hitters:
        
        with st.spinner("Running MAC analysis..."):
            try:
                # Run the MAC module
                run_mac(selected_pitcher, selected_hitters, df_all, base_path)
                
                # Load results
                last_first = selected_pitcher.replace(", ", "_")
                summary_path = os.path.join(base_path, f"{last_first}_summary.csv")
                breakdown_path = os.path.join(base_path, f"{last_first}_group_breakdown.csv")
                movement_path = os.path.join(base_path, f"{last_first}_pitch_level_filtered.csv")
                
                summary_df = pd.read_csv(summary_path)
                breakdown_df = pd.read_csv(breakdown_path)
                movement_df = pd.read_csv(movement_path)
                
                # Filter for selected hitters and maintain order
                summary_df = summary_df[summary_df["Batter"].isin(selected_hitters)].copy()
                breakdown_df = breakdown_df[breakdown_df["Batter"].isin(selected_hitters)].copy()
                movement_df = movement_df[movement_df["Batter"].isin(selected_hitters)].copy()
                
                for df in [summary_df, breakdown_df, movement_df]:
                    df["Batter"] = pd.Categorical(df["Batter"], categories=selected_hitters, ordered=True)
                    df.sort_values("Batter", inplace=True)
                
                st.success(f"✅ Analysis complete for {selected_pitcher} vs {len(selected_hitters)} hitters!")
                
                # Display results
                st.markdown("## 📊 RV/100 Performance Analysis")
                
                # Main scatter plot
                scatter_fig = create_scatter_plot(summary_df, breakdown_df, selected_pitcher)
                st.plotly_chart(scatter_fig, use_container_width=True)
                
                # Statistics table
                st.markdown("## 📋 Detailed Statistics")
                st.dataframe(summary_df, use_container_width=True)
                
                # Heatmap section
                st.markdown("## 🎯 Zone Analysis Heatmaps")
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    selected_hitter_heatmap = st.selectbox(
                        "Select hitter for zone heatmap:",
                        options=selected_hitters
                    )
                
                if selected_hitter_heatmap:
                    # Filter movement data for selected hitter
                    hitter_movement_df = movement_df[movement_df["Batter"] == selected_hitter_heatmap].copy()
                    
                    # Add required flags
                    hitter_movement_df["WhiffFlag"] = (hitter_movement_df["PitchCall"] == "StrikeSwinging").astype(int)
                    hitter_movement_df["HardHitFlag"] = ((hitter_movement_df["ExitSpeed"] >= 95) & 
                                                        hitter_movement_df["ExitSpeed"].notna()).astype(int)
                    
                    # Generate and display heatmap
                    encoded_img = generate_mpl_heatmap_grid(hitter_movement_df, selected_hitter_heatmap)
                    st.markdown(f"<img src='{encoded_img}' style='width: 100%; max-width: 1200px;'>", 
                               unsafe_allow_html=True)
                
                # Movement plot
                st.markdown("## 🌪️ Pitch Movement Analysis")
                movement_fig = create_movement_plot(movement_df)
                st.plotly_chart(movement_fig, use_container_width=True)
                

            except Exception as e:
                st.error("❌ Error running analysis.")
                st.text(traceback.format_exc())
                return

    elif run_analysis:
        st.warning("Please select both a pitcher and at least one hitter.")
    
    # Instructions
    if not run_analysis:
        st.markdown("""
        ## How to Use This App
        
        1. **Select a Pitcher** from the dropdown in the sidebar
        2. **Select Hitters** you want to analyze (multiple selection allowed)
        3. **Click "Run Matchup Analysis"** to generate the report
        
        The analysis will show:
        - **RV/100 Performance**: Expected run value per 100 pitches
        - **Detailed Statistics**: Complete breakdown by pitch type
        - **Zone Heatmaps**: Visual representation of performance by location
        - **Movement Charts**: Pitch movement patterns
        """)

if __name__ == "__main__":
    main()
