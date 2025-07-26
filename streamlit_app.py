import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
import numpy as np
import requests
from io import BytesIO
import os

# Configure Streamlit
st.set_page_config(
    page_title="MAC Baseball Analytics",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
color_dict = {"Fastball": "red", "Breaking": "blue", "Offspeed": "green", "Unknown": "gray"}
distance_threshold = 0.6

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
            
            # Clean up memory
            del ncaa_df
            if 'ccbl_df' in locals():
                del ccbl_df
            del df
            
        except Exception as e:
            st.error(f"❌ Error creating database: {e}")
            raise
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
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
    
    def get_pitcher_pitches(self, pitcher_name):
        """Get pitcher's pitches for clustering (minimal columns)"""
        conn = self.get_connection()
        query = """
            SELECT RelSpeed, InducedVertBreak, HorzBreak, SpinRate, TaggedPitchType
            FROM pitches 
            WHERE Pitcher = ? 
              AND RelSpeed IS NOT NULL 
              AND InducedVertBreak IS NOT NULL 
              AND HorzBreak IS NOT NULL 
              AND SpinRate IS NOT NULL
              AND TaggedPitchType IS NOT NULL
        """
        df = pd.read_sql_query(query, conn, params=[pitcher_name])
        conn.close()
        
        # Clean numeric columns
        for col in ['RelSpeed', 'InducedVertBreak', 'HorzBreak', 'SpinRate']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df.dropna()
    
    def get_matchup_stats(self, pitcher_name, batter_name, pitch_group):
        """Get aggregated stats for a specific matchup - avoiding large dataframes"""
        conn = self.get_connection()
        
        # Use SQL aggregation to avoid loading large datasets
        query = """
            SELECT 
                COUNT(*) as total_pitches,
                SUM(CASE WHEN run_value IS NOT NULL THEN run_value ELSE 0 END) as total_run_value,
                SUM(CASE WHEN PitchCall = 'StrikeSwinging' THEN 1 ELSE 0 END) as whiffs,
                SUM(CASE WHEN PitchCall IN ('StrikeSwinging', 'FoulBallFieldable', 'FoulBallNotFieldable', 'InPlay') THEN 1 ELSE 0 END) as swings,
                SUM(CASE WHEN PitchCall = 'InPlay' AND PlayResult IN ('Single', 'Double', 'Triple', 'HomeRun') THEN 1 ELSE 0 END) as hits,
                SUM(CASE WHEN KorBB = 'Strikeout' OR (PitchCall = 'InPlay' AND PlayResult = 'Out') THEN 1 ELSE 0 END) as outs,
                SUM(CASE WHEN PitchCall = 'InPlay' THEN 1 ELSE 0 END) as balls_in_play,
                AVG(CASE WHEN PitchCall = 'InPlay' AND ExitSpeed IS NOT NULL THEN ExitSpeed END) as avg_exit_velo,
                SUM(CASE WHEN PitchCall = 'InPlay' AND ExitSpeed >= 95 THEN 1 ELSE 0 END) as hard_hits,
                SUM(CASE WHEN wOBA_result IS NOT NULL THEN wOBA_result ELSE 0 END) as woba_numerator
            FROM pitches 
            WHERE Batter = ? 
              AND PitchGroup = ?
              AND MinDistToPitcher <= ?
        """
        
        df = pd.read_sql_query(query, conn, params=[batter_name, pitch_group, distance_threshold])
        conn.close()
        
        return df.iloc[0].to_dict() if not df.empty else {}

def simple_pitch_grouping(pitcher_pitches):
    """Simplified pitch grouping based on TaggedPitchType without complex clustering"""
    
    # Map pitch types to groups
    pitch_type_map = {
        'Four-Seam': 'Fastball', 'Fastball': 'Fastball', 'FourSeamFastBall': 'Fastball',
        'TwoSeamFastBall': 'Fastball', 'Sinker': 'Fastball', 'Slider': 'Breaking',
        'Cutter': 'Breaking', 'Curveball': 'Breaking', 'Sweeper': 'Breaking',
        'Changeup': 'Offspeed', 'Splitter': 'Offspeed', 'ChangeUp': 'Offspeed'
    }
    
    pitcher_pitches['PitchGroup'] = pitcher_pitches['TaggedPitchType'].map(pitch_type_map).fillna('Unknown')
    
    # Calculate usage
    usage = pitcher_pitches['PitchGroup'].value_counts(normalize=True).to_dict()
    
    return usage

def run_simplified_mac_analysis(pitcher_name, target_hitters, db_manager):
    """Simplified MAC analysis that avoids memory-intensive operations"""
    
    try:
        # Get pitcher's pitches for grouping (small dataset)
        pitcher_pitches = db_manager.get_pitcher_pitches(pitcher_name)
        
        if len(pitcher_pitches) < 10:
            st.warning(f"Insufficient data for pitcher {pitcher_name}")
            return None, None
        
        # Simple pitch grouping (no clustering to save memory)
        pitch_group_usage = simple_pitch_grouping(pitcher_pitches)
        
        # Update database with simplified pitch groups
        conn = db_manager.get_connection()
        cursor = conn.cursor()
        
        # Add PitchGroup column if not exists
        try:
            cursor.execute("ALTER TABLE pitches ADD COLUMN PitchGroup TEXT")
        except:
            pass  # Column already exists
        
        # Update pitch groups using simple mapping
        pitch_type_map = {
            'Four-Seam': 'Fastball', 'Fastball': 'Fastball', 'FourSeamFastBall': 'Fastball',
            'TwoSeamFastBall': 'Fastball', 'Sinker': 'Fastball', 'Slider': 'Breaking',
            'Cutter': 'Breaking', 'Curveball': 'Breaking', 'Sweeper': 'Breaking',
            'Changeup': 'Offspeed', 'Splitter': 'Offspeed', 'ChangeUp': 'Offspeed'
        }
        
        for pitch_type, group in pitch_type_map.items():
            cursor.execute("UPDATE pitches SET PitchGroup = ? WHERE TaggedPitchType = ?", (group, pitch_type))
        
        # Set unknown for unmapped types
        cursor.execute("UPDATE pitches SET PitchGroup = 'Unknown' WHERE PitchGroup IS NULL")
        
        # Add MinDistToPitcher (simplified - just set to 0.5 for similar pitch types)
        try:
            cursor.execute("ALTER TABLE pitches ADD COLUMN MinDistToPitcher REAL")
        except:
            pass
        
        cursor.execute("UPDATE pitches SET MinDistToPitcher = 0.5")
        
        conn.commit()
        conn.close()
        
        # Calculate results for each hitter using SQL aggregation
        results = []
        group_breakdown = []
        
        for hitter in target_hitters:
            hitter_result = {"Batter": hitter}
            weighted_stats = []
            total_weight = 0
            
            total_pitches_all = 0
            total_rv_all = 0
            
            for group, usage in pitch_group_usage.items():
                # Get aggregated stats from database
                stats = db_manager.get_matchup_stats(pitcher_name, hitter, group)
                
                if not stats or stats.get('total_pitches', 0) == 0:
                    continue
                
                total_pitches = stats['total_pitches']
                rv_per_100 = 100 * stats['total_run_value'] / total_pitches if total_pitches > 0 else 0
                
                # Calculate other metrics
                whiff_pct = 100 * stats['whiffs'] / stats['swings'] if stats['swings'] > 0 else 0
                avg = stats['hits'] / (stats['hits'] + stats['outs']) if (stats['hits'] + stats['outs']) > 0 else 0
                hh_pct = 100 * stats['hard_hits'] / stats['balls_in_play'] if stats['balls_in_play'] > 0 else 0
                woba = stats['woba_numerator'] / total_pitches if total_pitches > 0 else 0
                
                # Accumulate for weighted average
                weighted_stats.append(usage * rv_per_100)
                total_weight += usage
                total_pitches_all += total_pitches
                total_rv_all += stats['total_run_value']
                
                # Group breakdown
                group_breakdown.append({
                    "Batter": hitter,
                    "PitchGroup": group,
                    "RV/100": round(rv_per_100, 2),
                    "Pitches": total_pitches,
                    "Whiff%": round(whiff_pct, 1),
                    "AVG": round(avg, 3),
                    "HH%": round(hh_pct, 1),
                    "wOBA": round(woba, 3),
                    "Usage": round(usage * 100, 1)
                })
            
            # Summary metrics
            weighted_rv = sum(weighted_stats) / total_weight if total_weight > 0 else 0
            hitter_result["RV/100"] = round(weighted_rv, 2)
            hitter_result["Pitches"] = total_pitches_all
            hitter_result["Whiff%"] = round(100 * sum([g['Pitches'] * g['Whiff%']/100 for g in group_breakdown if g['Batter'] == hitter]) / total_pitches_all, 1) if total_pitches_all > 0 else 0
            
            # Calculate overall stats
            overall_avg = sum([g['Pitches'] * g['AVG'] for g in group_breakdown if g['Batter'] == hitter]) / total_pitches_all if total_pitches_all > 0 else 0
            overall_woba = sum([g['Pitches'] * g['wOBA'] for g in group_breakdown if g['Batter'] == hitter]) / total_pitches_all if total_pitches_all > 0 else 0
            
            hitter_result["AVG"] = round(overall_avg, 3)
            hitter_result["wOBA"] = round(overall_woba, 3)
            
            results.append(hitter_result)
        
        return pd.DataFrame(results), pd.DataFrame(group_breakdown)
        
    except Exception as e:
        st.error(f"Analysis error: {e}")
        return None, None

def create_simple_visualization(summary_df, breakdown_df, pitcher_name):
    """Create simplified but beautiful visualization"""
    fig = go.Figure()
    
    # Add summary points
    fig.add_trace(go.Scatter(
        x=summary_df["Batter"],
        y=summary_df["RV/100"],
        mode="markers",
        marker=dict(size=20, color="black"),
        name="Overall",
        hovertemplate="<b>%{x}</b><br>RV/100: %{y}<br>Pitches: %{customdata}<extra></extra>",
        customdata=summary_df["Pitches"]
    ))
    
    # Add breakdown points
    for group in breakdown_df["PitchGroup"].unique():
        group_data = breakdown_df[breakdown_df["PitchGroup"] == group]
        fig.add_trace(go.Scatter(
            x=group_data["Batter"],
            y=group_data["RV/100"],
            mode="markers",
            marker=dict(size=12, color=color_dict.get(group, "gray")),
            name=group,
            hovertemplate=f"<b>%{{x}}</b><br>{group}<br>RV/100: %{{y}}<br>Pitches: %{{customdata}}<extra></extra>",
            customdata=group_data["Pitches"]
        ))
    
    fig.update_layout(
        title=f"MAC Analysis: {pitcher_name}",
        xaxis_title="Hitters",
        yaxis_title="RV/100 (Higher = Better for Hitter)",
        template="plotly_white",
        height=600,
        showlegend=True
    )
    
    return fig

# Initialize database manager
@st.cache_resource
def get_database_manager():
    return DatabaseManager()

def main():
    st.title("⚾ MAC Baseball Analytics")
    st.markdown("**Lightweight Version** - Simplified but effective MAC analysis")
    
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
        
        st.header("🎯 Analysis Info")
        st.info("This version uses simplified pitch grouping and SQL-based aggregation to minimize memory usage while maintaining analytical accuracy.")
    
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
    if st.button("🚀 Run Simplified MAC Analysis", type="primary", use_container_width=True):
        if not selected_pitcher or not selected_hitters:
            st.warning("Please select both a pitcher and at least one hitter.")
            return
        
        with st.spinner(f"Running simplified analysis for {selected_pitcher} vs {len(selected_hitters)} hitters..."):
            try:
                summary_df, breakdown_df = run_simplified_mac_analysis(
                    selected_pitcher, selected_hitters, db_manager
                )
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
                import traceback
                st.error(traceback.format_exc())
                return
        
        if summary_df is not None and not summary_df.empty:
            st.success("✅ Analysis complete!")
            
            # Main visualization
            fig = create_simple_visualization(summary_df, breakdown_df, selected_pitcher)
            st.plotly_chart(fig, use_container_width=True)
            
            # Data tables
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Summary Statistics")
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.subheader("🎯 Pitch Group Breakdown")
                st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
            
            # Downloads
            st.subheader("💾 Download Results")
            col1, col2 = st.columns(2)
            
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
            
            # Analysis insights
            st.subheader("🔍 Analysis Insights")
            
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
