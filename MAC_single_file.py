import os
import sys
import gc
print("Starting streamlined MAC app...")
sys.stdout.flush()

# === CORE IMPORTS (ONLY ONCE) ===
import numpy as np
import pandas as pd
import requests
from dash import Dash, html, dcc, dash_table, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Delay heavy imports until needed
matplotlib = None
plt = None
sns = None
sklearn_imports = None

def lazy_import_heavy_libraries():
    """Import heavy libraries only when needed"""
    global matplotlib, plt, sns, sklearn_imports
    
    if matplotlib is None:
        print("Loading heavy libraries...")
        sys.stdout.flush()
        
        import matplotlib
        matplotlib.use("Agg")  # Use non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        import seaborn as sns
        from sklearn.preprocessing import StandardScaler
        from sklearn.mixture import GaussianMixture
        from sklearn.metrics.pairwise import euclidean_distances
        from kneed import KneeLocator
        from scipy import ndimage
        from scipy.interpolate import griddata
        
        sklearn_imports = {
            'StandardScaler': StandardScaler,
            'GaussianMixture': GaussianMixture,
            'euclidean_distances': euclidean_distances,
            'KneeLocator': KneeLocator,
            'ndimage': ndimage,
            'griddata': griddata,
            'Rectangle': Rectangle
        }
        
        print("✅ Heavy libraries loaded")
        sys.stdout.flush()

print("✅ Core imports successful")
sys.stdout.flush()

# === CONSTANTS ===
color_dict = {"Fastball": "red", "Breaking": "blue", "Breaking1": "blue", "Breaking2": "cyan", "Offspeed": "green"}
base_path = "./output"
os.makedirs(base_path, exist_ok=True)

distance_threshold = 0.6
strike_zone = {"top": 3.3775, "bottom": 1.5, "left": -0.83083, "right": 0.83083}
swing_calls = ["StrikeSwinging", "FoulBallFieldable", "FoulBallNotFieldable", "InPlay"]

# Dropbox URLs
NCAA_DROPBOX_URL = "https://www.dropbox.com/scl/fi/yqn49nclkdqcb6zl5jbcl/NCAA_final.parquet?rlkey=sn7gnopkfppqkiw2xwrzoxzj9&st=jdcrvw2p&dl=1"
CCBL_DROPBOX_URL = "https://www.dropbox.com/scl/fi/h3d8e8tqjg69vxb5z8kj3/CCBL_current.parquet?rlkey=9zpe9hn4a29jzj3lcsj7e0sxv&st=k7rdcj8u&dl=1"

# === UTILITY FUNCTIONS ===
def get_memory_usage():
    """Get current memory usage in MB"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0

def clean_numeric_column(series):
    """Convert a series to numeric, replacing non-numeric values with NaN"""
    return pd.to_numeric(series, errors='coerce')

def download_and_optimize_parquet(url, local_filename):
    """Download and load optimized parquet"""
    if os.path.exists(local_filename):
        print(f"✅ Using cached {local_filename}")
        sys.stdout.flush()
    else:
        print(f"⬇️ Downloading {local_filename}...")
        sys.stdout.flush()
        
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    
    # Load with memory optimization
    df = pd.read_parquet(local_filename)
    
    # Optimize dtypes
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].nunique() / len(df) < 0.1:
            df[col] = df[col].astype('category')
        elif df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
    
    print(f"✅ Loaded {local_filename}: {len(df)} rows")
    sys.stdout.flush()
    return df

# === STREAMLINED MAC FUNCTION ===
def run_mac_streamlined(pitcher_name, target_hitters, df_all, output_dir):
    """Streamlined MAC function with heavy imports only when needed"""
    lazy_import_heavy_libraries()  # Load heavy libraries now
    
    # Use sklearn imports
    StandardScaler = sklearn_imports['StandardScaler']
    GaussianMixture = sklearn_imports['GaussianMixture']
    euclidean_distances = sklearn_imports['euclidean_distances']
    KneeLocator = sklearn_imports['KneeLocator']
    
    df = df_all.copy()
    
    if df.empty:
        print(f"No data found in dataset")
        return

    # Filter for pitcher's data only for clustering
    pitcher_pitches = df[df["Pitcher"] == pitcher_name].copy()
    if pitcher_pitches.empty:
        print(f"No pitches found for pitcher: {pitcher_name}")
        return

    try:
        # Clean numeric columns
        numeric_columns = [
            'RelSpeed', 'InducedVertBreak', 'HorzBreak', 'SpinRate', 'RelHeight', 'RelSide',
            'run_value', 'RunsScored', 'OutsOnPlay', 'ExitSpeed', 'Angle', 'PlateLocHeight', 'PlateLocSide'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = clean_numeric_column(df[col])
        
        # Required columns check
        required_cols = ['RelSpeed', 'InducedVertBreak', 'HorzBreak', 'SpinRate', 'RelHeight', 'RelSide', 'TaggedPitchType']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
       
        # wOBA setup
        total_runs = df['RunsScored'].sum()
        strikeouts = (df['KorBB'] == 'Strikeout').sum()
        outs_on_play = df['OutsOnPlay'].fillna(0).sum()
        total_outs = strikeouts + outs_on_play
        r_out = total_runs / total_outs

        woba_weights = {
            'Walk': 0.673, 'HitByPitch': 0.718, 'Single': 0.949,
            'Double': 1.483, 'Triple': 1.963, 'HomeRun': 2.571
        }

        # Assign wOBA result values
        if 'wOBA_result' not in df.columns:
            df['wOBA_result'] = 0.0
            df.loc[df['KorBB'] == 'Walk', 'wOBA_result'] = woba_weights['Walk']
            df.loc[df['PitchCall'] == 'HitByPitch', 'wOBA_result'] = woba_weights['HitByPitch']
            df.loc[df['PlayResult'] == 'Single', 'wOBA_result'] = woba_weights['Single']
            df.loc[df['PlayResult'] == 'Double', 'wOBA_result'] = woba_weights['Double']
            df.loc[df['PlayResult'] == 'Triple', 'wOBA_result'] = woba_weights['Triple']
            df.loc[df['PlayResult'] == 'HomeRun', 'wOBA_result'] = woba_weights['HomeRun']
        else:
            df['wOBA_result'] = clean_numeric_column(df['wOBA_result'])

        # Feature sets
        scanning_features = ['RelSpeed', 'InducedVertBreak', 'HorzBreak', 'SpinRate', 'RelHeight', 'RelSide']
        clustering_features = ['RelSpeed', 'InducedVertBreak', 'HorzBreak', 'SpinRate']

        df = df.dropna(subset=scanning_features + ["Pitcher", "Batter"])
        pitcher_pitches = pitcher_pitches.dropna(subset=scanning_features + ["Pitcher", "Batter"])

        # Clustering
        scaler = StandardScaler()
        X_cluster = scaler.fit_transform(pitcher_pitches[clustering_features])

        # BIC optimization
        bic_scores = []
        ks = range(1, min(8, len(pitcher_pitches)//10 + 1))  # Limit clusters based on data size
        for k in ks:
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(X_cluster)
            bic_scores.append(gmm.bic(X_cluster))

        knee = KneeLocator(ks, bic_scores, curve='convex', direction='decreasing')
        optimal_k = knee.elbow or 2

        best_gmm = GaussianMixture(n_components=optimal_k, random_state=42)
        pitcher_pitches['PitchCluster'] = best_gmm.fit_predict(X_cluster)

        # Assign PitchGroup
        autopitchtype_to_group = {
            'Four-Seam': 'Fastball', 'Fastball': 'Fastball', 'FourSeamFastBall': 'Fastball',
            'TwoSeamFastBall': 'Fastball', 'Sinker': 'Fastball', 'Slider': 'Breaking',
            'Cutter': 'Breaking', 'Curveball': 'Breaking', 'Sweeper': 'Breaking',
            'Changeup': 'Offspeed', 'Splitter': 'Offspeed', 'ChangeUp': 'Offspeed'
        }

        pitcher_pitches = pitcher_pitches.dropna(subset=["TaggedPitchType"])

        cluster_to_type = {}
        for cluster in pitcher_pitches['PitchCluster'].unique():
            cluster_data = pitcher_pitches[pitcher_pitches['PitchCluster'] == cluster]
            type_counts = cluster_data['TaggedPitchType'].value_counts()
            if type_counts.empty:
                cluster_to_type[cluster] = 'Unknown'
                continue
            most_common_type = type_counts.idxmax()
            pitch_group = autopitchtype_to_group.get(most_common_type, 'Unknown')
            cluster_to_type[cluster] = pitch_group

        pitcher_pitches['PitchGroup'] = pitcher_pitches['PitchCluster'].map(cluster_to_type)
        pitch_group_usage = pitcher_pitches['PitchGroup'].value_counts(normalize=True).to_dict()

        # Tag full dataset
        scaler_all = StandardScaler()
        df_scaled = scaler_all.fit_transform(df[scanning_features])
        X_pitcher_full = scaler_all.transform(pitcher_pitches[scanning_features])
        distances = euclidean_distances(df_scaled, X_pitcher_full)
        df['MinDistToPitcher'] = distances.min(axis=1)

        df_subset_scaled = scaler.transform(df[clustering_features])
        df['PitchCluster'] = best_gmm.predict(df_subset_scaled)
        df['PitchGroup'] = df['PitchCluster'].map(cluster_to_type)

        # Simplified matchup analysis (to save memory)
        results = []
        for hitter in target_hitters:
            hitter_data = df[
                (df["Batter"] == hitter) & 
                (df["MinDistToPitcher"] <= distance_threshold)
            ]
            
            if not hitter_data.empty:
                rv_per_100 = 100 * hitter_data["run_value"].sum() / len(hitter_data)
                results.append({
                    "Batter": hitter,
                    "RV/100": round(rv_per_100, 2),
                    "Pitches": len(hitter_data)
                })

        # Save results
        os.makedirs(output_dir, exist_ok=True)
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(os.path.join(output_dir, f"{pitcher_name.replace(', ', '_')}_summary.csv"), index=False)
        
        print(f"✅ Basic analysis complete for {pitcher_name}")
        return True

    except Exception as e:
        print(f"[run_mac] ERROR: {e}")
        return False

# === GLOBAL DATA STORAGE ===
df_all = None

def get_data():
    """Lazy load data"""
    global df_all
    if df_all is None:
        print(f"Memory before loading: {get_memory_usage():.1f}MB")
        sys.stdout.flush()
        
        ncaa_df = download_and_optimize_parquet(NCAA_DROPBOX_URL, "NCAA_final.parquet")
        ccbl_df = download_and_optimize_parquet(CCBL_DROPBOX_URL, "CCBL_current.parquet")
        
        df_all = pd.concat([ncaa_df, ccbl_df], ignore_index=True)
        
        # Clean up
        del ncaa_df, ccbl_df
        gc.collect()
        
        print(f"✅ Data loaded: {len(df_all)} rows, Memory: {get_memory_usage():.1f}MB")
        sys.stdout.flush()
    
    return df_all

# === DASH APP ===
print("🌐 Starting Dash app...")
sys.stdout.flush()

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "MAC Interactive Visualizer"

app.layout = html.Div([
    html.H2("MAC Matchup Calculator", style={"textAlign": "center"}),
    html.Button("Load Data", id="load-button", n_clicks=0),
    html.Div(id="status"),
    
    html.Div(id="controls", style={"display": "none"}, children=[
        html.Label("Select Pitcher:"),
        dcc.Dropdown(id="pitcher-dropdown"),
        html.Label("Select Hitters:"),
        dcc.Dropdown(id="hitter-dropdown", multi=True),
        html.Button("Run Analysis", id="run-button", n_clicks=0),
        html.Div(id="results")
    ])
])

@app.callback(
    [Output("status", "children"), Output("controls", "style"),
     Output("pitcher-dropdown", "options"), Output("hitter-dropdown", "options")],
    Input("load-button", "n_clicks")
)
def load_data_callback(n_clicks):
    if n_clicks == 0:
        return f"Memory: {get_memory_usage():.1f}MB. Click to load data.", {"display": "none"}, [], []
    
    try:
        data = get_data()
        pitcher_opts = [{"label": p, "value": p} for p in sorted(data["Pitcher"].dropna().unique())]
        hitter_opts = [{"label": h, "value": h} for h in sorted(data["Batter"].dropna().unique())]
        
        return f"✅ Loaded {len(data)} rows. Memory: {get_memory_usage():.1f}MB", {"display": "block"}, pitcher_opts, hitter_opts
    except Exception as e:
        return f"❌ Error: {e}", {"display": "none"}, [], []

@app.callback(
    Output("results", "children"),
    Input("run-button", "n_clicks"),
    State("pitcher-dropdown", "value"),
    State("hitter-dropdown", "value")
)
def run_analysis(n_clicks, pitcher, hitters):
    if n_clicks == 0 or not pitcher or not hitters:
        return "Select pitcher and hitters, then click Run Analysis"
    
    try:
        data = get_data()
        success = run_mac_streamlined(pitcher, hitters, data, base_path)
        if success:
            return f"✅ Analysis complete! Memory: {get_memory_usage():.1f}MB"
        else:
            return "❌ Analysis failed"
    except Exception as e:
        return f"❌ Error: {e}"

if __name__ == "__main__":
    print(f"🚀 Server starting. Memory: {get_memory_usage():.1f}MB")
    sys.stdout.flush()
    
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
