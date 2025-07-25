import streamlit as st
import traceback
import sys
import os
import gc  # For garbage collection

# Add debug info at the very beginning
st.write("🐛 **Debug Info:**")
st.write(f"Python version: {sys.version}")
st.write(f"Current working directory: {os.getcwd()}")

try:
    # Test imports one by one
    st.write("Testing imports...")
    
    import pandas as pd
    st.write("✅ pandas imported")
    
    import numpy as np
    st.write("✅ numpy imported")
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    st.write("✅ plotly imported")
    
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    st.write("✅ matplotlib imported")
    
    import base64
    from io import BytesIO
    st.write("✅ base64 and io imported")
    
    from scipy import ndimage
    from scipy.interpolate import griddata
    st.write("✅ scipy imported")
    
    # Test MAC_module import separately
    try:
        from MAC_module import run_mac
        st.write("✅ MAC_module imported successfully")
    except Exception as mac_error:
        st.error(f"❌ MAC_module import failed: {mac_error}")
        run_mac = None
    
    import tempfile
    st.write("✅ tempfile imported")
    
    import requests
    st.write("✅ requests imported")
    
    # Test pyarrow import
    try:
        import pyarrow
        st.write("✅ pyarrow imported")
    except Exception as arrow_error:
        st.warning(f"⚠️ pyarrow not available: {arrow_error}")
        pyarrow = None

    st.success("✅ All imports successful!")

except Exception as e:
    st.error("❌ Error during imports:")
    st.text(traceback.format_exc())
    st.stop()

# === CONFIG ===
try:
    st.set_page_config(
        page_title="MAC Matchup Calculator",
        page_icon="⚾",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception as e:
    st.error(f"❌ Page config error: {e}")

# Color dictionary for pitch types
color_dict = {"Fastball": "red", "Breaking": "blue", "Breaking1": "blue", "Breaking2": "cyan", "Offspeed": "green"}

# Data paths
base_path = "./output"
try:
    os.makedirs(base_path, exist_ok=True)
except Exception as e:
    st.error(f"❌ Could not create output directory: {e}")

@st.cache_data(ttl=3600)  # Cache for 1 hour
def download_ncaa_data():
    """Download NCAA data with better error handling and memory management"""
    try:
        st.write("📥 Starting NCAA download...")
        DROPBOX_NCAA_URL = "https://www.dropbox.com/scl/fi/zozfzz75hamjsx5amp65b/NCAA_final.parquet?rlkey=nalex56psi9rj62fnyo5jhqt5&st=zm9f3dbm&dl=1"
        
        # Use streaming download for large files
        response = requests.get(DROPBOX_NCAA_URL, stream=True, timeout=30)
        
        if response.status_code != 200:
            raise Exception(f"Download failed with status code: {response.status_code}")
        
        # Get file size if available
        content_length = response.headers.get('content-length')
        if content_length:
            file_size_mb = int(content_length) / (1024 * 1024)
            st.write(f"📊 File size: {file_size_mb:.1f} MB")
        
        # Stream download to temporary file
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
            downloaded = 0
            chunk_size = 8192  # 8KB chunks
            
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    tmp_file.write(chunk)
                    downloaded += len(chunk)
                    
                    # Update progress every 1MB
                    if downloaded % (1024 * 1024) == 0:
                        st.write(f"📥 Downloaded: {downloaded / (1024 * 1024):.1f} MB")
            
            tmp_file.flush()
            tmp_path = tmp_file.name
        
        st.write("✅ NCAA download complete")
        return tmp_path
        
    except requests.exceptions.Timeout:
        raise Exception("Download timed out - try again")
    except Exception as e:
        raise Exception(f"Download error: {str(e)}")

@st.cache_data(ttl=3600)
def load_ncaa_data(file_path):
    """Load NCAA data with memory optimization"""
    try:
        st.write("📊 Loading NCAA parquet file...")
        
        # Try different engines
        engines_to_try = ['pyarrow', 'fastparquet']
        
        for engine in engines_to_try:
            try:
                if engine == 'pyarrow' and pyarrow is None:
                    continue
                    
                st.write(f"🔧 Trying {engine} engine...")
                
                # Load with memory optimization
                ncaa_df = pd.read_parquet(
                    file_path, 
                    engine=engine,
                    # Only load essential columns to save memory
                    columns=None  # Load all for now, but we could specify key columns
                )
                
                st.write(f"✅ NCAA data loaded with {engine}: {len(ncaa_df)} rows, {len(ncaa_df.columns)} columns")
                st.write(f"📊 Memory usage: {ncaa_df.memory_usage(deep=True).sum() / (1024*1024):.1f} MB")
                
                # Clean up temp file
                try:
                    os.unlink(file_path)
                except:
                    pass
                
                return ncaa_df
                
            except Exception as engine_error:
                st.warning(f"⚠️ {engine} failed: {engine_error}")
                continue
        
        raise Exception("All parquet engines failed")
        
    except Exception as e:
        # Clean up temp file on error
        try:
            os.unlink(file_path)
        except:
            pass
        raise e

@st.cache_data(ttl=3600)
def load_ccbl_data():
    """Load CCBL data"""
    CCBL_PARQUET = "CCBL_current.parquet"
    
    if not os.path.exists(CCBL_PARQUET):
        raise FileNotFoundError(f"CCBL file not found: {CCBL_PARQUET}")
    
    try:
        # Try pyarrow first, then fastparquet
        engines_to_try = ['pyarrow', 'fastparquet'] if pyarrow else ['fastparquet']
        
        for engine in engines_to_try:
            try:
                ccbl_df = pd.read_parquet(CCBL_PARQUET, engine=engine)
                st.write(f"✅ CCBL data loaded with {engine}: {len(ccbl_df)} rows")
                return ccbl_df
            except Exception as engine_error:
                st.warning(f"⚠️ CCBL {engine} failed: {engine_error}")
                continue
        
        raise Exception("All CCBL parquet engines failed")
        
    except Exception as e:
        raise Exception(f"CCBL load error: {e}")

@st.cache_data(ttl=3600)
def load_combined_data():
    """Load and combine data with better memory management"""
    try:
        # Force garbage collection before starting
        gc.collect()
        
        # Step 1: Download NCAA data
        ncaa_path = download_ncaa_data()
        
        # Step 2: Load NCAA data
        ncaa_df = load_ncaa_data(ncaa_path)
        
        # Step 3: Load CCBL data
        ccbl_df = load_ccbl_data()
        
        # Step 4: Combine data
        st.write("🔄 Combining datasets...")
        combined_df = pd.concat([ncaa_df, ccbl_df], ignore_index=True)
        
        # Clean up individual dataframes to free memory
        del ncaa_df, ccbl_df
        gc.collect()
        
        st.write(f"✅ Combined data: {len(combined_df)} rows, {len(combined_df.columns)} columns")
        st.write(f"📊 Total memory usage: {combined_df.memory_usage(deep=True).sum() / (1024*1024):.1f} MB")
        
        return combined_df
        
    except Exception as e:
        st.error(f"❌ Data load error: {e}")
        st.text(traceback.format_exc())
        # Force cleanup on error
        gc.collect()
        return pd.DataFrame()

# Simplified functions (removing some to save space)
def create_simple_scatter_plot(summary_df, pitcher_name):
    """Simplified scatter plot"""
    try:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=summary_df["Batter"],
            y=summary_df["RV/100"],
            mode="markers",
            marker=dict(size=15, color="blue"),
            text=summary_df["Batter"],
            hovertemplate="<b>%{text}</b><br>RV/100: %{y}<extra></extra>"
        ))
        
        fig.update_layout(
            title=f"RV/100 Analysis: {pitcher_name}",
            xaxis_title="Batter",
            yaxis_title="RV/100",
            height=500
        )
        
        return fig
    except Exception as e:
        st.error(f"Plot error: {e}")
        return None

# === MAIN APP ===
def main():
    st.title("⚾ MAC Matchup Calculator")
    st.markdown("---")

    # Add memory usage info
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        st.write(f"💾 Current memory usage: {memory_mb:.1f} MB")
    except:
        pass

    # Load data with progress tracking
    data_load_error = None
    df_all = pd.DataFrame()
    
    # Add a button to control data loading
    if st.button("📥 Load Data", type="primary"):
        with st.spinner("Loading data (this may take a few minutes)..."):
            try:
                df_all = load_combined_data()
                if df_all.empty:
                    data_load_error = "Data load returned empty DataFrame"
                else:
                    st.session_state['data_loaded'] = True
                    st.session_state['df_all'] = df_all
            except Exception as e:
                data_load_error = f"Data load failed: {str(e)}"
                st.text(traceback.format_exc())
    
    # Check if data is already loaded
    if 'data_loaded' in st.session_state and 'df_all' in st.session_state:
        df_all = st.session_state['df_all']
        st.success(f"✅ Data loaded: {len(df_all)} rows")
    elif not data_load_error:
        st.info("👆 Click 'Load Data' to start")
        return
    
    if data_load_error:
        st.error(data_load_error)
        return

    # Check if we have MAC module
    if run_mac is None:
        st.error("❌ MAC_module not available")
        return

    # Rest of the app (simplified)
    st.sidebar.header("Select Matchup")
    
    # Pitcher selection
    if len(df_all) > 0:
        pitcher_options = sorted(df_all["Pitcher"].dropna().unique())
        selected_pitcher = st.sidebar.selectbox(
            "Select Pitcher:",
            options=pitcher_options[:100],  # Limit to first 100 to avoid memory issues
            index=0 if pitcher_options else None
        )
        
        # Hitter selection  
        hitter_options = sorted(df_all["Batter"].dropna().unique())
        selected_hitters = st.sidebar.multiselect(
            "Select Hitters:",
            options=hitter_options[:100],  # Limit to first 100
            help="Select hitters in lineup order"
        )
        
        # Run analysis button
        if st.sidebar.button("🚀 Run Analysis", type="primary"):
            if selected_pitcher and selected_hitters:
                with st.spinner("Running analysis..."):
                    try:
                        run_mac(selected_pitcher, selected_hitters, df_all, base_path)
                        st.success("✅ Analysis complete!")
                        
                        # Load and display simple results
                        last_first = selected_pitcher.replace(", ", "_")
                        summary_path = os.path.join(base_path, f"{last_first}_summary.csv")
                        
                        if os.path.exists(summary_path):
                            summary_df = pd.read_csv(summary_path)
                            st.dataframe(summary_df)
                            
                            # Simple plot
                            fig = create_simple_scatter_plot(summary_df, selected_pitcher)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Analysis error: {e}")
                        st.text(traceback.format_exc())
            else:
                st.warning("Please select pitcher and hitters")

if __name__ == "__main__":
    main()
