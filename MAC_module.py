
#### REFACTORED MAC MODULE === DO NOT RUN THIS DIRECTLY ===
#### To use: call run_mac(pitcher_name, target_hitters, df_all, output_dir)

#### FIXED MAC MODULE === DO NOT RUN THIS DIRECTLY ===
#### To use: call run_mac(pitcher_name, target_hitters, df_all, output_dir)

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from kneed import KneeLocator

# === CONSTANTS ===
distance_threshold = 0.6
strike_zone = {"top": 3.3775, "bottom": 1.5, "left": -0.83083, "right": 0.83083}
swing_calls = ["StrikeSwinging", "FoulBallFieldable", "FoulBallNotFieldable", "InPlay"]

def clean_numeric_column(series):
    """Convert a series to numeric, replacing non-numeric values with NaN"""
    return pd.to_numeric(series, errors='coerce')

def run_mac(pitcher_name, target_hitters, df_all, output_dir):
    # *** CRITICAL FIX: Use full dataset, not just pitcher's data ***
    df = df_all.copy()  # Use FULL dataset, not filtered by pitcher
    
    if df.empty:
        print(f"No data found in dataset")
        return

    # Filter for pitcher's data only for clustering
    pitcher_pitches = df[df["Pitcher"] == pitcher_name].copy()
    if pitcher_pitches.empty:
        print(f"No pitches found for pitcher: {pitcher_name}")
        return

    try:
        # === Clean numeric columns first ===
        numeric_columns = [
            'RelSpeed', 'InducedVertBreak', 'HorzBreak', 'SpinRate', 'RelHeight', 'RelSide',
            'run_value', 'RunsScored', 'OutsOnPlay', 'ExitSpeed', 'Angle', 'PlateLocHeight', 'PlateLocSide'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                original_type = df[col].dtype
                df[col] = clean_numeric_column(df[col])
                print(f"Cleaned {col}: {original_type} -> {df[col].dtype}")
        
        # === Check for required columns ===
        required_cols = ['RelSpeed', 'InducedVertBreak', 'HorzBreak', 'SpinRate', 'RelHeight', 'RelSide', 'TaggedPitchType']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
       
        # === Compute League Run Environment for wOBA ===
        total_runs = df['RunsScored'].sum()
        strikeouts = (df['KorBB'] == 'Strikeout').sum()
        outs_on_play = df['OutsOnPlay'].fillna(0).sum()
        total_outs = strikeouts + outs_on_play
        r_out = total_runs / total_outs

        # === Scaled wOBA weights ===
        woba_weights = {
            'Walk': 0.673,
            'HitByPitch': 0.718,
            'Single': 0.949,
            'Double': 1.483,
            'Triple': 1.963,
            'HomeRun': 2.571
        }

        # === Assign wOBA result values to each pitch ===
        if 'wOBA_result' not in df.columns:
            df['wOBA_result'] = 0.0  # Initialize
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

        # === Scale features and cluster pitcher's arsenal ===
        # Step 1: Fit using clustering features on pitcher's data
        scaler = StandardScaler()
        X_cluster = scaler.fit_transform(pitcher_pitches[clustering_features])

        # Step 2: Run BIC loop to find optimal number of clusters
        bic_scores = []
        ks = range(1, 10)
        for k in ks:
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(X_cluster)
            bic_scores.append(gmm.bic(X_cluster))

        # Step 3: Find the "elbow" (knee point)
        knee = KneeLocator(ks, bic_scores, curve='convex', direction='decreasing')
        optimal_k = knee.elbow or 2  # fallback to 2 if no elbow found
        
        # Step 4: Fit final GMM using optimal_k and assign cluster labels
        best_gmm = GaussianMixture(n_components=optimal_k, random_state=42)
        pitcher_pitches['PitchCluster'] = best_gmm.fit_predict(X_cluster)

        # === Assign PitchGroup using TaggedPitchType majority ===
        autopitchtype_to_group = {
            'Four-Seam': 'Fastball',
            'Fastball': 'Fastball',
            'FourSeamFastBall': 'Fastball',
            'TwoSeamFastBall': 'Fastball',
            'Sinker': 'Fastball',
            'Slider': 'Breaking',
            'Cutter': 'Breaking',
            'Curveball': 'Breaking',
            'Sweeper': 'Breaking',
            'Changeup': 'Offspeed',
            'Splitter': 'Offspeed',
            'ChangeUp': 'Offspeed'
        }

        # Handle missing TaggedPitchType if any
        pitcher_pitches = pitcher_pitches.dropna(subset=["TaggedPitchType"])

        # Compute most common TaggedPitchType for each cluster
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

        # === Compute pitch group usage ===
        pitch_group_usage = pitcher_pitches['PitchGroup'].value_counts(normalize=True).to_dict()

        # *** CRITICAL FIX: Tag FULL dataset with MinDistToPitcher ***
        from sklearn.metrics.pairwise import euclidean_distances
        scaler_all = StandardScaler()
        df_scaled = scaler_all.fit_transform(df[scanning_features])  # FULL dataset
        X_pitcher_full = scaler_all.transform(pitcher_pitches[scanning_features])
        distances = euclidean_distances(df_scaled, X_pitcher_full)
        df['MinDistToPitcher'] = distances.min(axis=1)

        # === Assign PitchGroup to entire dataset using cluster model ===
        df_subset_scaled = scaler.transform(df[clustering_features])
        df['PitchCluster'] = best_gmm.predict(df_subset_scaled)
        df['PitchGroup'] = df['PitchCluster'].map(cluster_to_type)

        # === Matchup scoring ===
        results = []
        group_breakdown = []

        for hitter in target_hitters:
            hitter_result = {"Batter": hitter}
            weighted_stats = []
            total_weight = 0

            # Initialize accumulators for summary
            total_pitches_seen = 0
            total_swings_seen = 0
            total_whiffs_seen = 0
            total_ev_sum = 0
            total_la_sum = 0
            total_hard_hits = 0
            total_gbs = 0
            total_bips = 0
            total_hits = 0
            total_outs = 0
            total_woba_num = 0
            total_woba_den = 0

            for group, usage in pitch_group_usage.items():
                # *** NOW using full dataset for matchup analysis ***
                group_pitches = df[
                    (df["Batter"] == hitter) &
                    (df["PitchGroup"] == group) &
                    (df["MinDistToPitcher"] <= distance_threshold)
                ].copy()

                if group_pitches.empty:
                    continue

                # Clean plate location columns for zone calculation
                group_pitches['PlateLocHeight'] = clean_numeric_column(group_pitches['PlateLocHeight'])
                group_pitches['PlateLocSide'] = clean_numeric_column(group_pitches['PlateLocSide'])

                group_pitches["InZone"] = (
                    (group_pitches["PlateLocHeight"] >= strike_zone["bottom"]) &
                    (group_pitches["PlateLocHeight"] <= strike_zone["top"]) &
                    (group_pitches["PlateLocSide"] >= strike_zone["left"]) &
                    (group_pitches["PlateLocSide"] <= strike_zone["right"])
                )
                group_pitches["Swung"] = group_pitches["PitchCall"].isin(swing_calls)
                group_pitches["Whiff"] = group_pitches["PitchCall"] == "StrikeSwinging"
                group_pitches["IsInPlay"] = group_pitches['PitchCall'].isin(["InPlay"])

                total_pitches = len(group_pitches)
                total_swings = group_pitches["Swung"].sum()
                total_whiffs = group_pitches["Whiff"].sum()
                total_run_value = group_pitches["run_value"].sum()

                # Clean exit speed and angle columns
                group_pitches['ExitSpeed'] = clean_numeric_column(group_pitches['ExitSpeed'])
                group_pitches['Angle'] = clean_numeric_column(group_pitches['Angle'])
                
                exit_velo = group_pitches["ExitSpeed"].mean()
                launch_angle = group_pitches["Angle"].mean()

                balls_in_play = group_pitches[group_pitches["IsInPlay"]]
                num_ground_balls = (balls_in_play["Angle"] < 10).sum()
                gb_percent = round(100 * num_ground_balls / len(balls_in_play), 1) if len(balls_in_play) > 0 else np.nan
                num_hard_hits = (balls_in_play["ExitSpeed"] >= 95).sum()
                hh_percent = round(100 * num_hard_hits / len(balls_in_play), 1) if len(balls_in_play) > 0 else np.nan

                rv_per_100 = 100 * total_run_value / total_pitches if total_pitches > 0 else 0
                weighted_stats.append(usage * rv_per_100)
                total_weight += usage

                # === Calculate AVG for this group ===
                hit_mask = (
                    (group_pitches["PitchCall"] == "InPlay") &
                    (group_pitches["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun"]))
                )
                hits = hit_mask.sum()

                out_mask = (
                    ((group_pitches["KorBB"].isin(["Strikeout", "Walk"])) |
                    ((group_pitches["PitchCall"] == "InPlay") & (group_pitches["PlayResult"] == "Out"))) &
                    (group_pitches["PlayResult"] != "Sacrifice")
                )
                outs = out_mask.sum()

                avg = round(hits / (hits + outs), 3) if (hits + outs) > 0 else np.nan

                # Accumulate full pitch data for summary
                total_pitches_seen += total_pitches
                total_swings_seen += total_swings
                total_whiffs_seen += total_whiffs
                total_bips += len(balls_in_play)
                total_hits += hits
                total_outs += outs
                if not np.isnan(exit_velo):
                    total_ev_sum += exit_velo * len(balls_in_play)
                if not np.isnan(launch_angle):
                    total_la_sum += launch_angle * len(balls_in_play)

                if not np.isnan(num_hard_hits):
                    total_hard_hits += num_hard_hits
                if not np.isnan(num_ground_balls):
                    total_gbs += num_ground_balls

                # Compute wOBA for this group
                plate_ending = group_pitches[
                    (group_pitches["KorBB"].isin(["Strikeout", "Walk"])) |
                    (group_pitches["PitchCall"].isin(["InPlay", "HitByPitch"]))
                ]

                group_woba_numerator = plate_ending["wOBA_result"].sum()
                group_woba_denominator = len(plate_ending)
                group_woba = round(group_woba_numerator / group_woba_denominator, 3) if group_woba_denominator > 0 else np.nan

                # Accumulate for summary-level wOBA
                total_woba_num += group_woba_numerator
                total_woba_den += group_woba_denominator

                group_breakdown.append({
                    "Batter": hitter,
                    "PitchGroup": group,
                    "AVG": avg,
                    "RV/100": round(rv_per_100, 2),
                    "Whiff%": round(100 * total_whiffs / total_swings, 1) if total_swings > 0 else np.nan,
                    "SwStr%": round(100 * total_whiffs / total_pitches, 1) if total_pitches > 0 else np.nan,
                    "HH%": hh_percent,
                    "ExitVelo": round(exit_velo, 1) if not np.isnan(exit_velo) else np.nan,
                    "Launch": round(launch_angle, 1) if not np.isnan(launch_angle) else np.nan,
                    "GB%": gb_percent,
                    "UsageWeight": round(usage, 2),
                    "Pitches": total_pitches,
                    "InPlay": len(balls_in_play),
                    "wOBA": group_woba,
                })

            weighted_rv = sum(weighted_stats) / total_weight if total_weight > 0 else np.nan
            hitter_result["RV/100"] = round(weighted_rv, 2)
            hitter_result["AVG"] = round(total_hits / (total_hits + total_outs), 3) if (total_hits + total_outs) > 0 else np.nan
            hitter_result["Whiff%"] = round(100 * total_whiffs_seen / total_swings_seen, 1) if total_swings_seen > 0 else np.nan
            hitter_result["SwStr%"] = round(100 * total_whiffs_seen / total_pitches_seen, 1) if total_pitches_seen > 0 else np.nan
            hitter_result["ExitVelo"] = round(total_ev_sum / total_bips, 1) if total_bips > 0 else np.nan
            hitter_result["Launch"] = round(total_la_sum / total_bips, 1) if total_bips > 0 else np.nan
            hitter_result["HH%"] = round(100 * total_hard_hits / total_bips, 1) if total_bips > 0 else np.nan
            hitter_result["GB%"] = round(100 * total_gbs / total_bips, 1) if total_bips > 0 else np.nan
            hitter_result["Pitches"] = total_pitches_seen
            hitter_result["InPlay"] = total_bips
            hitter_result["wOBA"] = round(total_woba_num / total_woba_den, 3) if total_woba_den > 0 else np.nan

            results.append(hitter_result)

        # === Output summary and breakdown ===
        os.makedirs(output_dir, exist_ok=True)
        sum_df = pd.DataFrame(results)
        breakdown_df = pd.DataFrame(group_breakdown)

        sum_df.to_csv(os.path.join(output_dir, f"{pitcher_name.replace(', ', '_')}_summary.csv"), index=False)
        breakdown_df.to_csv(os.path.join(output_dir, f"{pitcher_name.replace(', ', '_')}_group_breakdown.csv"), index=False)

        print("✅ Matchup summary and breakdown files saved.")

        # === Save filtered pitch-level data for interactive visual ===
        filtered_df = df[
            (df["Batter"].isin(target_hitters)) &
            (df["MinDistToPitcher"] <= distance_threshold)
        ].copy()

        # Save only relevant columns for movement plotting
        cols_to_keep = [
            "Pitcher", "PitcherThrows","Batter", "PitchGroup", "RelSpeed", "InducedVertBreak", "HorzBreak","SpinRate", "RelHeight",
            "PlateLocSide", "PlateLocHeight", "ExitSpeed", "Angle", "PitchCall", "PlayResult", "wOBA_result"
        ]
        filtered_df[cols_to_keep].to_csv(
            os.path.join(output_dir, f"{pitcher_name.replace(', ', '_')}_pitch_level_filtered.csv"),
            index=False
        )

        # === Coverage Analysis ===
        coverage_matrix = pd.DataFrame(index=target_hitters, columns=["Fastball", "Breaking", "Offspeed"]).fillna(0)

        for hitter in target_hitters:
            for group in ["Fastball", "Breaking", "Offspeed"]:
                matches = df[
                    (df["Batter"] == hitter) &
                    (df["PitchGroup"] == group) &
                    (df["MinDistToPitcher"] <= distance_threshold)
                ]
                coverage_matrix.loc[hitter, group] = len(matches)

        # === Visualize: Clusters vs Groups vs TaggedPitchType ===
        os.makedirs(output_dir, exist_ok=True)

        # Define consistent color palettes
        cluster_palette = sns.color_palette("tab10", n_colors=optimal_k)
        group_palette = {
            'Fastball': 'red',
            'Breaking': 'blue',
            'Offspeed': 'green',
            'Unknown': 'gray'
        }
        autopitch_palette = sns.color_palette("tab20", n_colors=pitcher_pitches['TaggedPitchType'].nunique())

        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharex=True, sharey=True)

        # Plot 1: Clusters
        sns.scatterplot(
            data=pitcher_pitches,
            x='HorzBreak',
            y='InducedVertBreak',
            hue='PitchCluster',
            palette=cluster_palette,
            ax=axes[0],
            legend='full'
        )
        axes[0].set_title("Pitch Movement by Cluster")
        axes[0].set_xlabel("Horizontal Break")
        axes[0].set_ylabel("Induced Vertical Break")

        # Plot 2: Pitch Groups
        sns.scatterplot(
            data=pitcher_pitches,
            x='HorzBreak',
            y='InducedVertBreak',
            hue='PitchGroup',
            palette=group_palette,
            ax=axes[1],
            legend='full'
        )
        axes[1].set_title("Pitch Movement by Pitch Group")
        axes[1].set_xlabel("Horizontal Break")
        axes[1].set_ylabel("Induced Vertical Break")

        similar_pitches = df[df['MinDistToPitcher'] <= distance_threshold]
        # Plot 3: TaggedPitchType
        sns.scatterplot(
            data=similar_pitches,
            x='HorzBreak',
            y='InducedVertBreak',
            hue='TaggedPitchType',
            palette=autopitch_palette,
            ax=axes[2],
            legend='full'
        )
        axes[2].set_title("Pitch Movement by TaggedPitchType")
        axes[2].set_xlabel("Horizontal Break")
        axes[2].set_ylabel("Induced Vertical Break")

        # === Set same x/y limits and square aspect ratio for all ===
        x_limits = (-25, 25)
        y_limits = (-25, 25)
        for ax in axes:
            ax.set_xlim(x_limits)
            ax.set_ylim(y_limits)
            ax.set_aspect('equal', adjustable='box')

        # Finalize layout and save
        plt.tight_layout()
        plt.suptitle(f"{pitcher_name} — Pitch Movement Clustering Comparison", y=1.05, fontsize=16)
        plt.savefig(os.path.join(output_dir, f"{pitcher_name.replace(', ', '_')}_Cluster_Group_TaggedPitchType.png"), bbox_inches='tight')
        plt.show()

        # Print clean coverage summary
        print("\n🎯 Coverage Matrix (pitch counts within distance threshold):\n")
        print(coverage_matrix.astype(int))

        print("✅ Using clustering features:", clustering_features)
        print("✅ Using scanning features:", scanning_features)
    
    except Exception as e:
        print(f"[run_mac] ERROR: {e}")





   #####################
    ### END OF MAC_3 PASTING
   #####################




    pass  # Replace with actual logic
