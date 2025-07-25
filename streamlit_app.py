import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import requests
from io import BytesIO
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage
from scipy.interpolate import griddata
import base64
import tempfile
import shutil

# Import the original MAC module
from MAC_module import run_mac

# Configure Streamlit
st.set_page_config(
    page_title="MAC Baseball Analytics",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
color_dict = {"Fastball": "red", "Breaking": "blue", "Offspeed": "green", "Unknown": "gray"}

class DatabaseManager:
    def __init__(self, db_path="baseball_data.db"):
        self.db_path = db_path
        self.ensure_database_exists()
    
    def ensure_database_exists(self):
        """Download and create database if it doesn't exist"""
        if not os.path.exists(self.db_path):
            st.info("📂 Setting up database for first time use...")
            self.create_database_from_dropbox()
    
    def create_database_from_dropbox(self):
        """Create database from both NCAA and CCBL data (both from Dropbox)"""
        try:
            progress_bar = st.progress(0)
            st.info("📂 Downloading NCAA data from Dropbox...")
            
            # Download NCAA data
            ncaa_url = "https://www.dropbox.com/scl/fi/c5jpffe349ejtboynvbab/NCAA_final_compressed.parquet?rlkey=u9q96ge9z5aenb2ttnecb46uo&st=co7eyva9&dl=1"
            response = requests.get(ncaa_url, timeout=300)
            response.raise_for_status()
            progress_bar.progress(30)
            
            ncaa_df = pd.read_parquet(BytesIO(response.content))
            st.success(f"✅ NCAA data loaded: {len(ncaa_df):,} rows")
            progress_bar.progress(50)
            
            # Download CCBL data (if available)
            st.info("📂 Downloading CCBL data from Dropbox...")
            try:
                ccbl_url = "https://www.dropbox.com/scl/fi/xayqylfb2d8wnqr4p5jua/CCBL_current.parquet?rlkey=e1mqyzpgvp68iq1w1j171q3i6&st=d88z1j0u&dl=1"
                ccbl_response = requests.get(ccbl_url, timeout=180)
                ccbl_response.raise_for_status()
                
                ccbl_df = pd.read_parquet(BytesIO(ccbl_response.content))
                st.success(f"✅ CCBL data loaded: {len(ccbl_df):,} rows")
                df = pd.concat([ncaa_df, ccbl_df], ignore_index=True)
                st.success(f"✅ Combined dataset: {len(df):,} rows")
                
            except Exception as e:
                st.warning(f"⚠️ Could not load CCBL data: {e}")
                st.info("Using NCAA data only")
                df = ncaa_df
            
            progress_bar.progress(70)
            
            # Create SQLite database
            conn = sqlite3.connect(self.db_path)
            df.to_sql('pitches', conn, if_exists='replace', index=False)
            progress_bar.progress(85)
            
            # Create indexes
            cursor = conn.cursor()
            cursor.execute("CREATE INDEX idx_pitcher ON pitches(Pitcher)")
            cursor.execute("CREATE INDEX idx_batter ON pitches(Batter)")
            cursor.execute("CREATE INDEX idx_pitcher_batter ON pitches(Pitcher, Batter)")
            
            # Create summary tables
            cursor.execute("""
                CREATE TABLE pitcher_summary AS
                SELECT 
                    Pitcher,
                    COUNT(*) as total_pitches,
                    AVG(RelSpeed) as avg_speed,
                    AVG(InducedVertBreak) as avg_ivb,
                    AVG(HorzBreak) as avg_hb,
                    AVG(SpinRate) as avg_spin
                FROM pitches 
                WHERE RelSpeed IS NOT NULL AND InducedVertBreak IS NOT NULL 
                  AND HorzBreak IS NOT NULL AND SpinRate IS NOT NULL
                GROUP BY Pitcher
                HAVING COUNT(*) >= 10
            """)
            
            conn.commit()
            conn.close()
            progress_bar.progress(100)
            st.success("✅ Database created successfully!")
            
            # Store the combined dataframe for MAC module use
            self.df_all = df
            
        except Exception as e:
            st.error(f"❌ Error creating database: {e}")
            raise
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def get_combined_dataframe(self):
        """Get the full combined dataframe for MAC analysis"""
        if hasattr(self, 'df_all'):
            return self.df_all
        else:
            # If not in memory, reconstruct from database
            conn = self.get_connection()
            df = pd.read_sql_query("SELECT * FROM pitches", conn)
            conn.close()
            return df
    
    def get_pitchers(self):
        """Get list of pitchers with sufficient data"""
        conn = self.get_connection()
        query = "SELECT Pitcher FROM pitcher_summary ORDER BY Pitcher"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df['Pitcher'].tolist()
    
    def get_batters(self):
        """Get list of batters"""
        conn = self.get_connection()
        query = "SELECT DISTINCT Batter FROM pitches WHERE Batter IS NOT NULL ORDER BY Batter"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df['Batter'].tolist()

def compute_heatmap_stats(df, metric_col, min_samples=3):
    """Compute heatmap statistics for zone analysis"""
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

def generate_zone_heatmap(df, selected_hitter):
    """Generate zone-level heatmap for a specific hitter"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    metrics = [("WhiffFlag", "Whiff Rate"), ("HardHitFlag", "Hard Hit Rate"), ("wOBA_result", "wOBA")]
    pitch_groups = ["Fastball", "Breaking", "Offspeed"]

    # Add flags to dataframe
    df["WhiffFlag"] = (df["PitchCall"] == "StrikeSwinging").astype(int)
    df["HardHitFlag"] = ((df["ExitSpeed"] >= 95) & df["ExitSpeed"].notna()).astype(int)

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

            # Add strike zone
            strike_zone_rect = patches.Rectangle((-0.83, 1.5), 1.66, 1.8775, linewidth=2.5,
                                               edgecolor='black', facecolor='none')
            ax.add_patch(strike_zone_rect)

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

    # Convert to base64 for Streamlit
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def create_comprehensive_visualization(summary_df, breakdown_df, pitcher_name):
    """Create comprehensive visualization matching the Dash app style"""
    fig = make_subplots(rows=1, cols=1)
    
    # Add summary points with comprehensive hover info
    for _, row in summary_df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["Batter"]],
            y=[row["RV/100"]],
            mode="markers",
            marker=dict(size=20, color="black"),
            name="Overall",
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
    
    # Add breakdown points with comprehensive hover info
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
    
    return fig

def create_movement_chart(movement_df):
    """Create pitch movement chart matching Dash app style"""
    movement_df_filtered = movement_df[
        (movement_df["HorzBreak"].between(-50, 50)) & 
        (movement_df["InducedVertBreak"].between(-50, 50))
    ]
    
    fig = go.Figure()
    
    for pitch_type, color in color_dict.items():
        pitch_df = movement_df_filtered[movement_df_filtered["PitchGroup"] == pitch_type]
        if not pitch_df.empty:
            fig.add_trace(go.Scatter(
                x=pitch_df["HorzBreak"],
                y=pitch_df["InducedVertBreak"],
                mode="markers",
                marker=dict(color=color, size=10, opacity=0.7),
                name=pitch_type,
                customdata=pitch_df[["Batter", "RelSpeed", "SpinRate"]],
                hovertemplate="<b>%{customdata[0]}</b><br>"
                              "HB: %{x}<br>"
                              "IVB: %{y}<br>"
                              "RelSpeed: %{customdata[1]} mph<br>"
                              "SpinRate: %{customdata[2]}<extra></extra>"
            ))
    
    fig.update_layout(
        title="Pitch Movement (HorzBreak vs. InducedVertBreak)",
        xaxis=dict(title="Horizontal Break", range=[-30, 30]),
        yaxis=dict(title="Induced Vertical Break", range=[-30, 30], scaleanchor="x", scaleratio=1),
        template="simple_white",
        height=600,
        width=1000
    )
    
    return fig

# Initialize database manager
@st.cache_resource
def get_database_manager():
    return DatabaseManager()

def main():
    st.title("⚾ MAC Baseball Analytics")
    st.markdown("**Streamlit Version** - Using original MAC_module with beautiful visualizations")
    
    # Initialize database
    try:
        db_manager = get_database_manager()
    except Exception as e:
        st.error(f"❌ Could not initialize database: {e}")
        st.stop()
    
    # Sidebar info
    with st.sidebar:
        st.header("🗄️ Database Info")
        if os.path.exists("baseball_data.db"):
            db_size = os.path.getsize("baseball_data.db") / 1024**2
            st.metric("Database Size", f"{db_size:.1f}MB")
            st.success("✅ Database ready")
        
        if st.button("🔄 Refresh Database"):
            if os.path.exists("baseball_data.db"):
                os.remove("baseball_data.db")
            st.cache_resource.clear()
            st.rerun()
        
        # Add analysis parameters
        st.header("🎯 Analysis Parameters")
        st.metric("Distance Threshold", "0.6")
        st.info("Pitches within this similarity distance to the pitcher's arsenal are included in the analysis.")
    
    # Get available options
    with st.spinner("Loading available players..."):
        try:
            available_pitchers = db_manager.get_pitchers()
            available_batters = db_manager.get_batters()
        except Exception as e:
            st.error(f"❌ Error loading players: {e}")
            st.stop()
    
    # Display stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Available Pitchers", len(available_pitchers))
    with col2:
        st.metric("Available Batters", len(available_batters))
    
    st.markdown("---")
    
    # Selection interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🥎 Select Pitcher")
        selected_pitcher = st.selectbox(
            "Choose a pitcher:",
            available_pitchers,
            index=0 if available_pitchers else None
        )
    
    with col2:
        st.subheader("🏏 Select Hitters")
        selected_hitters = st.multiselect(
            "Choose hitters:",
            available_batters,
            default=available_batters[:3] if len(available_batters) >= 3 else available_batters[:1]
        )
    
    # Analysis
    if st.button("🚀 Run MAC Analysis", type="primary", use_container_width=True):
        if not selected_pitcher or not selected_hitters:
            st.warning("Please select both a pitcher and at least one hitter.")
            return
        
        with st.spinner(f"Running MAC analysis for {selected_pitcher} vs {len(selected_hitters)} hitters..."):
            try:
                # Get the combined dataframe
                df_all = db_manager.get_combined_dataframe()
                
                # Create temporary output directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Run the original MAC module
                    run_mac(selected_pitcher, selected_hitters, df_all, temp_dir)
                    
                    # Read the results
                    last_first = selected_pitcher.replace(", ", "_")
                    summary_path = os.path.join(temp_dir, f"{last_first}_summary.csv")
                    breakdown_path = os.path.join(temp_dir, f"{last_first}_group_breakdown.csv")
                    movement_path = os.path.join(temp_dir, f"{last_first}_pitch_level_filtered.csv")
                    
                    if not all(os.path.exists(p) for p in [summary_path, breakdown_path, movement_path]):
                        st.error("❌ MAC analysis failed to generate expected output files")
                        return
                    
                    summary_df = pd.read_csv(summary_path)
                    breakdown_df = pd.read_csv(breakdown_path)
                    movement_df = pd.read_csv(movement_path)
                    
                    # Filter for selected hitters
                    summary_df = summary_df[summary_df["Batter"].isin(selected_hitters)].copy()
                    breakdown_df = breakdown_df[breakdown_df["Batter"].isin(selected_hitters)].copy()
                    movement_df = movement_df[movement_df["Batter"].isin(selected_hitters)].copy()
                    
                    # Sort by hitter order
                    for df in [summary_df, breakdown_df, movement_df]:
                        df["Batter"] = pd.Categorical(df["Batter"], categories=selected_hitters, ordered=True)
                        df.sort_values("Batter", inplace=True)
                    
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
                import traceback
                st.error(traceback.format_exc())
                return
        
        if summary_df is not None and not summary_df.empty:
            st.success("✅ MAC Analysis complete!")
            
            # Main visualization
            fig = create_comprehensive_visualization(summary_df, breakdown_df, selected_pitcher)
            st.plotly_chart(fig, use_container_width=True)
            
            # Data tables
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Full Stats Table")
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.subheader("🎯 Pitch Group Breakdown")
                st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
            
            # Movement chart
            st.subheader("🌪️ Pitch Movement Chart")
            movement_fig = create_movement_chart(movement_df)
            st.plotly_chart(movement_fig, use_container_width=True)
            
            # Zone analysis
            st.subheader("🎯 Select Hitter for Zone Heatmap:")
            selected_hitter_heatmap = st.selectbox(
                "Choose hitter for detailed zone analysis:",
                selected_hitters,
                key="heatmap_hitter"
            )
            
            if selected_hitter_heatmap:
                with st.spinner(f"Generating zone heatmap for {selected_hitter_heatmap}..."):
                    # Filter data for selected hitter
                    hitter_data = movement_df[movement_df["Batter"] == selected_hitter_heatmap].copy()
                    
                    if not hitter_data.empty:
                        heatmap_img = generate_zone_heatmap(hitter_data, selected_hitter_heatmap)
                        st.markdown(f"<img src='{heatmap_img}' style='width: 100%; max-width: 1200px;'>", 
                                  unsafe_allow_html=True)
                    else:
                        st.warning(f"No data available for {selected_hitter_heatmap} zone analysis.")
            
            # Coverage analysis
            st.subheader("📈 Coverage Matrix")
            coverage_matrix = pd.DataFrame(
                index=selected_hitters, 
                columns=["Fastball", "Breaking", "Offspeed"]
            ).fillna(0)
            
            for hitter in selected_hitters:
                for group in ["Fastball", "Breaking", "Offspeed"]:
                    matches = movement_df[
                        (movement_df["Batter"] == hitter) &
                        (movement_df["PitchGroup"] == group)
                    ]
                    coverage_matrix.loc[hitter, group] = len(matches)
            
            st.dataframe(coverage_matrix.astype(int), use_container_width=True)
            st.info("📊 Coverage Matrix shows pitch counts within distance threshold for each hitter vs pitch group combination")
            
            # Downloads
            st.subheader("💾 Download Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_summary = summary_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Summary",
                    csv_summary,
                    f"{selected_pitcher.replace(', ', '_')}_summary.csv",
                    "text/csv"
                )
            
            with col2:
                csv_breakdown = breakdown_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Breakdown",
                    csv_breakdown,
                    f"{selected_pitcher.replace(', ', '_')}_breakdown.csv",
                    "text/csv"
                )
            
            with col3:
                csv_movement = movement_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Pitch Data",
                    csv_movement,
                    f"{selected_pitcher.replace(', ', '_')}_pitch_level.csv",
                    "text/csv"
                )
            
            # Analysis insights
            st.subheader("🔍 Analysis Insights")
            
            # Calculate insights
            best_matchup = summary_df.loc[summary_df["RV/100"].idxmin(), "Batter"] if not summary_df.empty else "N/A"
            worst_matchup = summary_df.loc[summary_df["RV/100"].idxmax(), "Batter"] if not summary_df.empty else "N/A"
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Best Matchup (Pitcher)",
                    best_matchup,
                    f"RV/100: {summary_df['RV/100'].min():.2f}" if not summary_df.empty else "N/A"
                )
            
            with col2:
                st.metric(
                    "Worst Matchup (Pitcher)",
                    worst_matchup,
                    f"RV/100: {summary_df['RV/100'].max():.2f}" if not summary_df.empty else "N/A"
                )
            
            with col3:
                avg_rv = summary_df["RV/100"].mean() if not summary_df.empty else 0
                st.metric(
                    "Average RV/100",
                    f"{avg_rv:.2f}",
                    "Lower is better for pitcher"
                )
        
        else:
            st.warning("❌ No sufficient data found for this matchup.")

if __name__ == "__main__":
    main()
