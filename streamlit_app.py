import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from kneed import KneeLocator
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
        """Create database from both NCAA (Dropbox) and CCBL (GitHub) data"""
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
            st.info("📂 Downloading CCBL data from GitHub...")
            try:
                ccbl_url = "https://github.com/yourusername/yourrepo/raw/main/CCBL_current.parquet"  # UPDATE THIS URL
                ccbl_response = requests.get(ccbl_url, timeout=180)
                if ccbl_response.status_code == 200:
                    ccbl_df = pd.read_parquet(BytesIO(ccbl_response.content))
                    st.success(f"✅ CCBL data loaded: {len(ccbl_df):,} rows")
                    
                    # Combine datasets
                    df = pd.concat([ncaa_df, ccbl_df], ignore_index=True)
                    st.success(f"✅ Combined dataset: {len(df):,} rows")
                else:
                    st.warning("⚠️ CCBL data not found, using NCAA only")
                    df = ncaa_df
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
    
    def get_pitcher_data(self, pitcher_name):
        """Get all data for a specific pitcher"""
        conn = self.get_connection()
        query = """
            SELECT * FROM pitches 
            WHERE Pitcher = ? 
              AND RelSpeed IS NOT NULL 
              AND InducedVertBreak IS NOT NULL 
              AND HorzBreak IS NOT NULL 
              AND SpinRate IS NOT NULL
        """
        df = pd.read_sql_query(query, conn, params=[pitcher_name])
        conn.close()
        return df
    
    def get_matchup_data(self, pitcher_name, batter_names):
        """Get relevant data for matchup analysis - MEMORY EFFICIENT"""
        conn = self.get_connection()
        
        # Convert batter names to SQL IN clause
        placeholders = ','.join(['?' for _ in batter_names])
        
        # Get only the data we need for analysis
        query = f"""
            SELECT 
                Pitcher, Batter, TaggedPitchType, PitchCall, PlayResult, KorBB,
                RelSpeed, InducedVertBreak, HorzBreak, SpinRate, RelHeight, RelSide,
                PlateLocSide, PlateLocHeight, ExitSpeed, Angle, run_value, wOBA_result
            FROM pitches 
            WHERE (Pitcher = ? OR Batter IN ({placeholders}))
              AND RelSpeed IS NOT NULL 
              AND InducedVertBreak IS NOT NULL 
              AND HorzBreak IS NOT NULL 
              AND SpinRate IS NOT NULL
              AND Pitcher IS NOT NULL 
              AND Batter IS NOT NULL
        """
        
        params = [pitcher_name] + batter_names
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        # Clean numeric columns
        numeric_cols = ['RelSpeed', 'InducedVertBreak', 'HorzBreak', 'SpinRate', 
                       'RelHeight', 'RelSide', 'PlateLocSide', 'PlateLocHeight', 
                       'ExitSpeed', 'Angle', 'run_value', 'wOBA_result']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df

# Initialize database manager
@st.cache_resource
def get_database_manager():
    return DatabaseManager()

def run_mac_analysis_efficient(pitcher_name, target_hitters, db_manager):
    """Memory-efficient MAC analysis using database queries"""
    
    # Get only the data we need
    df = db_manager.get_matchup_data(pitcher_name, target_hitters)
    
    if df.empty:
        return None, None, None
    
    # Get pitcher's data for clustering
    pitcher_pitches = df[df["Pitcher"] == pitcher_name].copy()
    if len(pitcher_pitches) < 10:
        st.warning(f"Insufficient data for pitcher {pitcher_name} ({len(pitcher_pitches)} pitches)")
        return None, None, None
    
    # Add wOBA if missing
    if 'wOBA_result' not in df.columns or df['wOBA_result'].isna().all():
        df['wOBA_result'] = 0.0
        woba_weights = {
            'Walk': 0.673, 'HitByPitch': 0.718, 'Single': 0.949,
            'Double': 1.483, 'Triple': 1.963, 'HomeRun': 2.571
        }
        
        for result, weight in woba_weights.items():
            if result == 'Walk':
                df.loc[df['KorBB'] == 'Walk', 'wOBA_result'] = weight
            elif result == 'HitByPitch':
                df.loc[df['PitchCall'] == 'HitByPitch', 'wOBA_result'] = weight
            elif result in ['Single', 'Double', 'Triple', 'HomeRun']:
                df.loc[df['PlayResult'] == result, 'wOBA_result'] = weight
    
    # Clustering features
    clustering_features = ['RelSpeed', 'InducedVertBreak', 'HorzBreak', 'SpinRate']
    scanning_features = clustering_features + ['RelHeight', 'RelSide']
    
    # Cluster pitcher's arsenal
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(pitcher_pitches[clustering_features])
    
    # Find optimal clusters
    max_k = min(6, len(pitcher_pitches) // 10 + 1)
    if max_k < 2:
        optimal_k = 1
        pitcher_pitches['PitchCluster'] = 0
    else:
        bic_scores = []
        ks = range(1, max_k)
        for k in ks:
            gmm = GaussianMixture(n_components=k, random_state=42, max_iter=50)
            gmm.fit(X_cluster)
            bic_scores.append(gmm.bic(X_cluster))
        
        knee = KneeLocator(ks, bic_scores, curve='convex', direction='decreasing')
        optimal_k = knee.elbow or 2
        
        # Fit final model
        best_gmm = GaussianMixture(n_components=optimal_k, random_state=42, max_iter=50)
        pitcher_pitches['PitchCluster'] = best_gmm.fit_predict(X_cluster)
    
    # Assign pitch groups
    autopitchtype_to_group = {
        'Four-Seam': 'Fastball', 'Fastball': 'Fastball', 'TwoSeamFastBall': 'Fastball',
        'Sinker': 'Fastball', 'Slider': 'Breaking', 'Cutter': 'Breaking',
        'Curveball': 'Breaking', 'Sweeper': 'Breaking', 'Changeup': 'Offspeed',
        'Splitter': 'Offspeed', 'ChangeUp': 'Offspeed'
    }
    
    # Map clusters to pitch groups
    cluster_to_type = {}
    if optimal_k == 1:
        # Single cluster case
        if 'TaggedPitchType' in pitcher_pitches.columns:
            most_common_type = pitcher_pitches['TaggedPitchType'].mode()
            if len(most_common_type) > 0:
                pitch_group = autopitchtype_to_group.get(most_common_type.iloc[0], 'Unknown')
            else:
                pitch_group = 'Unknown'
        else:
            pitch_group = 'Unknown'
        cluster_to_type[0] = pitch_group
    else:
        for cluster in range(optimal_k):
            cluster_data = pitcher_pitches[pitcher_pitches['PitchCluster'] == cluster]
            if len(cluster_data) > 0 and 'TaggedPitchType' in cluster_data.columns:
                type_counts = cluster_data['TaggedPitchType'].value_counts()
                if not type_counts.empty:
                    most_common_type = type_counts.idxmax()
                    pitch_group = autopitchtype_to_group.get(most_common_type, 'Unknown')
                else:
                    pitch_group = 'Unknown'
            else:
                pitch_group = 'Unknown'
            cluster_to_type[cluster] = pitch_group
    
    pitcher_pitches['PitchGroup'] = pitcher_pitches['PitchCluster'].map(cluster_to_type)
    
    # Calculate pitch group usage
    pitch_group_usage = pitcher_pitches['PitchGroup'].value_counts(normalize=True).to_dict()
    
    # Calculate similarity for full dataset (memory efficient)
    from sklearn.metrics.pairwise import euclidean_distances
    
    # Process in smaller batches to avoid memory issues
    batch_size = 1000
    df['MinDistToPitcher'] = np.inf
    
    scaler_all = StandardScaler()
    pitcher_scaled = scaler_all.fit_transform(pitcher_pitches[scanning_features])
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        batch_scaled = scaler_all.transform(batch[scanning_features])
        distances = euclidean_distances(batch_scaled, pitcher_scaled)
        df.iloc[i:i+batch_size, df.columns.get_loc('MinDistToPitcher')] = distances.min(axis=1)
    
    # Assign pitch groups to full dataset
    if optimal_k > 1:
        df_scaled = scaler.transform(df[clustering_features])
        df['PitchCluster'] = best_gmm.predict(df_scaled)
    else:
        df['PitchCluster'] = 0
    
    df['PitchGroup'] = df['PitchCluster'].map(cluster_to_type)
    
    # Perform matchup analysis
    results = []
    group_breakdown = []
    
    swing_calls = ["StrikeSwinging", "FoulBallFieldable", "FoulBallNotFieldable", "InPlay"]
    
    for hitter in target_hitters:
        hitter_result = {"Batter": hitter}
        weighted_stats = []
        total_weight = 0
        
        total_pitches_seen = 0
        total_whiffs_seen = 0
        total_swings_seen = 0
        total_hits = 0
        total_outs = 0
        total_woba_num = 0
        total_woba_den = 0
        
        for group, usage in pitch_group_usage.items():
            # Get matchup data for this pitch group
            group_pitches = df[
                (df["Batter"] == hitter) &
                (df["PitchGroup"] == group) &
                (df["MinDistToPitcher"] <= distance_threshold)
            ].copy()
            
            if group_pitches.empty:
                continue
            
            # Calculate metrics
            total_pitches = len(group_pitches)
            total_run_value = group_pitches["run_value"].sum()
            rv_per_100 = 100 * total_run_value / total_pitches if total_pitches > 0 else 0
            
            # Other stats
            swings = group_pitches["PitchCall"].isin(swing_calls).sum()
            whiffs = (group_pitches["PitchCall"] == "StrikeSwinging").sum()
            
            # Hits and outs
            hits = group_pitches["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun"]).sum() if 'PlayResult' in group_pitches.columns else 0
            strikeouts = (group_pitches["KorBB"] == "Strikeout").sum() if 'KorBB' in group_pitches.columns else 0
            
            # wOBA
            woba_num = group_pitches["wOBA_result"].sum()
            woba_den = total_pitches
            
            # Accumulate
            total_pitches_seen += total_pitches
            total_swings_seen += swings
            total_whiffs_seen += whiffs
            total_hits += hits
            total_outs += strikeouts
            total_woba_num += woba_num
            total_woba_den += woba_den
            
            weighted_stats.append(usage * rv_per_100)
            total_weight += usage
            
            # Group breakdown
            group_breakdown.append({
                "Batter": hitter,
                "PitchGroup": group,
                "RV/100": round(rv_per_100, 2),
                "Pitches": total_pitches,
                "Whiff%": round(100 * whiffs / swings, 1) if swings > 0 else 0,
                "Usage": round(usage * 100, 1)
            })
        
        # Summary
        weighted_rv = sum(weighted_stats) / total_weight if total_weight > 0 else 0
        hitter_result["RV/100"] = round(weighted_rv, 2)
        hitter_result["Pitches"] = total_pitches_seen
        hitter_result["Whiff%"] = round(100 * total_whiffs_seen / total_swings_seen, 1) if total_swings_seen > 0 else 0
        hitter_result["AVG"] = round(total_hits / (total_hits + total_outs), 3) if (total_hits + total_outs) > 0 else 0
        hitter_result["wOBA"] = round(total_woba_num / total_woba_den, 3) if total_woba_den > 0 else 0
        
        results.append(hitter_result)
    
    return pd.DataFrame(results), pd.DataFrame(group_breakdown), df

def create_visualization(summary_df, breakdown_df, pitcher_name):
    """Create the main visualization"""
    fig = make_subplots(rows=1, cols=1)
    
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

def main():
    st.title("⚾ MAC Baseball Analytics")
    st.markdown("**Memory-Efficient Version** - Using SQLite database for optimal performance")
    
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
        
        with st.spinner(f"Analyzing {selected_pitcher} vs {len(selected_hitters)} hitters..."):
            try:
                summary_df, breakdown_df, _ = run_mac_analysis_efficient(
                    selected_pitcher, selected_hitters, db_manager
                )
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
                return
        
        if summary_df is not None and not summary_df.empty:
            st.success("✅ Analysis complete!")
            
            # Visualization
            fig = create_visualization(summary_df, breakdown_df, selected_pitcher)
            st.plotly_chart(fig, use_container_width=True)
            
            # Data tables
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Summary Statistics")
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.subheader("Pitch Group Breakdown")
                st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
            
            # Downloads
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
        else:
            st.warning("❌ No sufficient data found for this matchup.")

if __name__ == "__main__":
    main()
