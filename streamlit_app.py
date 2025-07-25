import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import ndimage
from scipy.interpolate import griddata
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from kneed import KneeLocator
import os

# Configure Streamlit
st.set_page_config(
    page_title="MAC Baseball Analytics",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main .block-container {
    max-width: 1400px;
    padding-top: 1rem;
}
.stSelectbox > div > div > select {
    background-color: #f0f2f6;
}
div[data-testid="metric-container"] {
    background-color: #f0f2f6;
    border: 1px solid #cccccc;
    padding: 5% 5% 5% 10%;
    border-radius: 5px;
    color: rgb(30, 103, 119);
    overflow-wrap: break-word;
}
</style>
""", unsafe_allow_html=True)

# Constants
color_dict = {"Fastball": "red", "Breaking": "blue", "Offspeed": "green", "Unknown": "gray"}
distance_threshold = 0.6
strike_zone = {"top": 3.3775, "bottom": 1.5, "left": -0.83083, "right": 0.83083}
swing_calls = ["StrikeSwinging", "FoulBallFieldable", "FoulBallNotFieldable", "InPlay"]

@st.cache_data(ttl=3600, show_spinner=True)
def load_data_from_dropbox():
    """Load data from Dropbox with caching"""
    
    # Convert Dropbox share link to direct download link
    ncaa_url = "https://www.dropbox.com/scl/fi/c5jpffe349ejtboynvbab/NCAA_final_compressed.parquet?rlkey=u9q96ge9z5aenb2ttnecb46uo&st=k5cuysoi&dl=1"  # Changed dl=0 to dl=1
    
    try:
        st.info("📂 Loading NCAA data from Dropbox...")
        response = requests.get(ncaa_url, timeout=300)  # 5 minute timeout
        response.raise_for_status()
        
        ncaa_df = pd.read_parquet(BytesIO(response.content))
        st.success(f"✅ Loaded NCAA data: {len(ncaa_df):,} pitches")
        
        # For now, just use NCAA data. You can add CCBL later if needed
        return ncaa_df
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.info("Using sample data for demonstration...")
        return create_sample_data()

def create_sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    
    pitchers = [f"Smith, John", f"Johnson, Mike", f"Williams, Dave"] * 10
    batters = [f"Batter_{i}" for i in range(50)]
    
    data = []
    for _ in range(2000):
        data.append({
            'Pitcher': np.random.choice(pitchers),
            'Batter': np.random.choice(batters),
            'RelSpeed': np.random.normal(88, 5),
            'InducedVertBreak': np.random.normal(0, 8),
            'HorzBreak': np.random.normal(0, 8),
            'SpinRate': np.random.normal(2200, 300),
            'RelHeight': np.random.normal(6, 0.5),
            'RelSide': np.random.normal(0, 0.3),
            'run_value': np.random.normal(-0.05, 0.3),
            'TaggedPitchType': np.random.choice(['Fastball', 'Slider', 'Changeup', 'Curveball']),
            'PitchCall': np.random.choice(['BallCalled', 'StrikeSwinging', 'InPlay', 'StrikeCalled']),
            'PlayResult': np.random.choice(['Out', 'Single', 'Double', 'Triple', 'HomeRun', None]),
            'KorBB': np.random.choice(['Strikeout', 'Walk', None, None, None]),
            'ExitSpeed': np.random.normal(85, 10),
            'Angle': np.random.normal(15, 20),
            'PlateLocSide': np.random.normal(0, 1),
            'PlateLocHeight': np.random.normal(2.5, 0.8),
            'wOBA_result': np.random.uniform(0, 2)
        })
    
    return pd.DataFrame(data)

def clean_numeric_column(series):
    """Convert a series to numeric, replacing non-numeric values with NaN"""
    return pd.to_numeric(series, errors='coerce')

def run_mac_analysis(pitcher_name, target_hitters, df_all):
    """Run MAC analysis on the data"""
    
    # Filter for relevant data
    df = df_all.copy()
    
    # Get pitcher's data for clustering
    pitcher_pitches = df[df["Pitcher"] == pitcher_name].copy()
    if pitcher_pitches.empty:
        return None, None, None
    
    # Clean numeric columns
    numeric_columns = [
        'RelSpeed', 'InducedVertBreak', 'HorzBreak', 'SpinRate', 'RelHeight', 'RelSide',
        'run_value', 'ExitSpeed', 'Angle', 'PlateLocHeight', 'PlateLocSide'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col])
            pitcher_pitches[col] = clean_numeric_column(pitcher_pitches[col])
    
    # Add wOBA if missing
    if 'wOBA_result' not in df.columns:
        df['wOBA_result'] = 0.0
        woba_weights = {
            'Walk': 0.673, 'HitByPitch': 0.718, 'Single': 0.949,
            'Double': 1.483, 'Triple': 1.963, 'HomeRun': 2.571
        }
        
        if 'KorBB' in df.columns:
            df.loc[df['KorBB'] == 'Walk', 'wOBA_result'] = woba_weights['Walk']
        if 'PitchCall' in df.columns:
            df.loc[df['PitchCall'] == 'HitByPitch', 'wOBA_result'] = woba_weights['HitByPitch']
        if 'PlayResult' in df.columns:
            for result, weight in woba_weights.items():
                if result in ['Single', 'Double', 'Triple', 'HomeRun']:
                    df.loc[df['PlayResult'] == result, 'wOBA_result'] = weight
    
    # Feature sets
    scanning_features = ['RelSpeed', 'InducedVertBreak', 'HorzBreak', 'SpinRate', 'RelHeight', 'RelSide']
    clustering_features = ['RelSpeed', 'InducedVertBreak', 'HorzBreak', 'SpinRate']
    
    # Drop rows with missing critical data
    df = df.dropna(subset=scanning_features + ["Pitcher", "Batter"])
    pitcher_pitches = pitcher_pitches.dropna(subset=scanning_features + ["Pitcher", "Batter"])
    
    if len(pitcher_pitches) < 10:
        st.warning(f"Not enough data for pitcher {pitcher_name} ({len(pitcher_pitches)} pitches)")
        return None, None, None
    
    # Cluster pitcher's arsenal
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(pitcher_pitches[clustering_features])
    
    # Find optimal clusters
    max_k = min(6, len(pitcher_pitches) // 5 + 1)
    if max_k < 2:
        optimal_k = 1
    else:
        bic_scores = []
        ks = range(1, max_k)
        for k in ks:
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(X_cluster)
            bic_scores.append(gmm.bic(X_cluster))
        
        knee = KneeLocator(ks, bic_scores, curve='convex', direction='decreasing')
        optimal_k = knee.elbow or 2
    
    # Fit final model
    best_gmm = GaussianMixture(n_components=optimal_k, random_state=42)
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
    for cluster in pitcher_pitches['PitchCluster'].unique():
        cluster_data = pitcher_pitches[pitcher_pitches['PitchCluster'] == cluster]
        if 'TaggedPitchType' in cluster_data.columns:
            type_counts = cluster_data['TaggedPitchType'].value_counts()
            if not type_counts.empty:
                most_common_type = type_counts.idxmax()
                pitch_group = autopitchtype_to_group.get(most_common_type, 'Unknown')
                cluster_to_type[cluster] = pitch_group
            else:
                cluster_to_type[cluster] = 'Unknown'
        else:
            cluster_to_type[cluster] = 'Unknown'
    
    pitcher_pitches['PitchGroup'] = pitcher_pitches['PitchCluster'].map(cluster_to_type)
    
    # Calculate pitch group usage
    pitch_group_usage = pitcher_pitches['PitchGroup'].value_counts(normalize=True).to_dict()
    
    # Tag full dataset with similarity to pitcher
    from sklearn.metrics.pairwise import euclidean_distances
    scaler_all = StandardScaler()
    df_scaled = scaler_all.fit_transform(df[scanning_features])
    X_pitcher_full = scaler_all.transform(pitcher_pitches[scanning_features])
    distances = euclidean_distances(df_scaled, X_pitcher_full)
    df['MinDistToPitcher'] = distances.min(axis=1)
    
    # Assign pitch groups to full dataset
    df_subset_scaled = scaler.transform(df[clustering_features])
    df['PitchCluster'] = best_gmm.predict(df_subset_scaled)
    df['PitchGroup'] = df['PitchCluster'].map(cluster_to_type)
    
    # Perform matchup analysis
    results = []
    group_breakdown = []
    
    for hitter in target_hitters:
        hitter_result = {"Batter": hitter}
        weighted_stats = []
        total_weight = 0
        
        # Aggregated stats for summary
        total_pitches_seen = 0
        total_swings_seen = 0
        total_whiffs_seen = 0
        total_hits = 0
        total_outs = 0
        total_woba_num = 0
        total_woba_den = 0
        
        for group, usage in pitch_group_usage.items():
            # Get matchup data
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
            
            # Calculate other stats
            swings = group_pitches["PitchCall"].isin(swing_calls).sum()
            whiffs = (group_pitches["PitchCall"] == "StrikeSwinging").sum()
            
            # Hits and outs for AVG
            if 'PlayResult' in group_pitches.columns:
                hits = group_pitches["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun"]).sum()
            else:
                hits = 0
            
            if 'KorBB' in group_pitches.columns:
                strikeouts = (group_pitches["KorBB"] == "Strikeout").sum()
            else:
                strikeouts = 0
            
            outs = strikeouts  # Simplified
            
            # wOBA calculation
            woba_num = group_pitches["wOBA_result"].sum()
            woba_den = total_pitches
            
            # Accumulate
            total_pitches_seen += total_pitches
            total_swings_seen += swings
            total_whiffs_seen += whiffs
            total_hits += hits
            total_outs += outs
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
        
        # Summary stats
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
    st.title("⚾ MAC Baseball Matchup Calculator")
    st.markdown("**Machup Analysis Calculator** - Analyze pitcher vs hitter matchups using advanced metrics")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Dataset Info")
        if st.button("🔄 Reload Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Load data
    with st.spinner("Loading baseball data..."):
        df_all = load_data_from_dropbox()
    
    if df_all.empty:
        st.error("Could not load data")
        return
    
    # Dataset info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Pitches", f"{len(df_all):,}")
    with col2:
        st.metric("Pitchers", f"{df_all['Pitcher'].nunique():,}")
    with col3:
        st.metric("Batters", f"{df_all['Batter'].nunique():,}")
    
    st.markdown("---")
    
    # Selection interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🥎 Select Pitcher")
        available_pitchers = sorted(df_all['Pitcher'].dropna().unique())
        selected_pitcher = st.selectbox(
            "Choose a pitcher:",
            available_pitchers,
            index=0 if available_pitchers else None
        )
    
    with col2:
        st.subheader("🏏 Select Hitters")
        available_hitters = sorted(df_all['Batter'].dropna().unique())
        selected_hitters = st.multiselect(
            "Choose hitters:",
            available_hitters,
            default=available_hitters[:5] if len(available_hitters) >= 5 else available_hitters
        )
    
    # Analysis button
    if st.button("🚀 Run MAC Analysis", type="primary", use_container_width=True):
        if not selected_pitcher or not selected_hitters:
            st.warning("Please select both a pitcher and at least one hitter.")
            return
        
        with st.spinner(f"Analyzing {selected_pitcher} vs {len(selected_hitters)} hitters..."):
            summary_df, breakdown_df, analyzed_df = run_mac_analysis(
                selected_pitcher, selected_hitters, df_all
            )
        
        if summary_df is not None and not summary_df.empty:
            st.success("✅ Analysis complete!")
            
            # Results
            st.subheader("📊 Matchup Results")
            
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
            
            # Download
            csv_summary = summary_df.to_csv(index=False)
            csv_breakdown = breakdown_df.to_csv(index=False)
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📥 Download Summary CSV",
                    csv_summary,
                    f"{selected_pitcher.replace(', ', '_')}_summary.csv",
                    "text/csv"
                )
            with col2:
                st.download_button(
                    "📥 Download Breakdown CSV", 
                    csv_breakdown,
                    f"{selected_pitcher.replace(', ', '_')}_breakdown.csv",
                    "text/csv"
                )
        
        else:
            st.warning("No sufficient data found for this matchup.")

if __name__ == "__main__":
    main()
