import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
CSV_FILE = "piecewise_aligned_matches.csv"  # Input CSV file (or speaker_proximity_results.csv)
OUTPUT_DIR = "plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Master Speaker List (Ensures all 4 are explicitly present)
ALL_SPEAKERS = ["Alicia", "Kush", "Max", "Riad"]
SPEAKER_COLORS = {
    "Alicia": "#1f77b4",  # Blue
    "Kush": "#d62728",    # Red
    "Max": "#2ca02c",     # Green
    "Riad": "#ff7f0e"     # Orange
}

# --------------------------------------------------
# LOAD & PREPROCESS CSV
# --------------------------------------------------
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"File '{CSV_FILE}' not found. Please run your alignment/matching script first.")

df = pd.read_csv(CSV_FILE)

# Detect column names dynamically
if "Time_Delta_Sec" in df.columns:
    time_col1, time_col2 = "Speaker_1_Time_Sec", "Speaker_2_Time_Sec"
    distance_col = "Distance_Feet"
    spk1_col, spk2_col = "Speaker_1", "Speaker_2"
elif "Raw_Time_Diff_Sec" in df.columns:
    time_col1, time_col2 = "Speaker_1_Time_Sec", "Speaker_2_Time_Sec"
    distance_col = "Estimated_Distance_Ft"
    spk1_col, spk2_col = "Speaker_1", "Speaker_2"
elif "Calculated_Acoustic_Delay_Sec" in df.columns:
    time_col1, time_col2 = "Time_A_Phrase_1", "Time_B_Phrase_1"
    distance_col = "Distance_Feet"
    spk1_col, spk2_col = "Speaker_A", "Speaker_B"
else:
    raise ValueError(f"Unrecognized CSV format columns: {df.columns.tolist()}")

df_full = df.copy()

# Average interaction time
df_full["Conversation_Time"] = (df_full[time_col1] + df_full[time_col2]) / 2.0

# Duplicate entries symmetrically to ensure full (Speaker A -> Speaker B) and (Speaker B -> Speaker A) coverage
df_reverse = df_full.copy()
df_reverse[spk1_col], df_reverse[spk2_col] = df_full[spk2_col], df_full[spk1_col]
df_full = pd.concat([df_full, df_reverse], ignore_index=True)

# Filter out invalid or zero distances
df_full = df_full[df_full[distance_col] > 0]

# --------------------------------------------------
# DYNAMIC 30% RANGE CLASSIFICATION (Pandas 2.0+ Compatible)
# --------------------------------------------------
df_full["p30"] = df_full.groupby([spk1_col, spk2_col])[distance_col].transform(lambda x: x.quantile(0.30))
df_full["p70"] = df_full.groupby([spk1_col, spk2_col])[distance_col].transform(lambda x: x.quantile(0.70))

conditions = [
    df_full[distance_col] <= df_full["p30"],
    (df_full[distance_col] > df_full["p30"]) & (df_full[distance_col] <= df_full["p70"]),
    df_full[distance_col] > df_full["p70"]
]
choices = ["Close (Lowest 30%)", "Medium Distance", "Far (Highest 30%)"]
df_full["Distance_Category"] = np.select(conditions, choices, default="Medium Distance")

category_numeric_map = {
    "Close (Lowest 30%)": 0,
    "Medium Distance": 1,
    "Far (Highest 30%)": 2
}
df_full["Category_Code"] = df_full["Distance_Category"].map(category_numeric_map)

# --------------------------------------------------
# PLOT 1: SPEAKER ACTIVITY TIMELINE (ALL 4 SPEAKERS)
# --------------------------------------------------
plt.figure(figsize=(14, 5))
speaker_y = {s: i for i, s in enumerate(ALL_SPEAKERS)}

for spk in ALL_SPEAKERS:
    spk_data = df_full[df_full[spk1_col] == spk]
    plt.scatter(
        spk_data[time_col1],
        [speaker_y[spk]] * len(spk_data),
        color=SPEAKER_COLORS.get(spk, "black"),
        s=12,
        alpha=0.6,
        label=spk
    )

plt.yticks(range(len(ALL_SPEAKERS)), ALL_SPEAKERS, fontsize=12)
plt.xlabel("Recording Timestamp (sec)", fontsize=11)
plt.title("Speaker Activity & Overlap Timeline (All 4 Speakers)", fontsize=13, fontweight="bold")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_all_4_speakers_timeline.png"), dpi=300)
plt.close()

# --------------------------------------------------
# PLOT 2: DIRECTED INTERACTIONS GRID WITH HARD OVERRIDDEN Y-AXIS (0, 6, 12, 18, 24 ft)
# --------------------------------------------------
fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True, sharey=True)
fig.suptitle("Pairwise Speaker Interactions Matrix (4 x 3 Directional Grid)", fontsize=16, fontweight="bold")

# Determine global max in dataset to anchor top tick label position
max_val = df_full[distance_col].max() if not df_full.empty else 1.0

for i, spk_from in enumerate(ALL_SPEAKERS):
    for j, spk_to in enumerate(ALL_SPEAKERS):
        ax = axes[i, j]
        if spk_from == spk_to:
            ax.set_facecolor("#f0f0f0")
            ax.text(0.5, 0.5, f"Self\n({spk_from})", ha="center", va="center", transform=ax.transAxes, color="#888888", fontsize=11)
        else:
            sub = df_full[(df_full[spk1_col] == spk_from) & (df_full[spk2_col] == spk_to)]
            if not sub.empty:
                ax.scatter(
                    sub["Conversation_Time"],
                    sub[distance_col],
                    s=8,
                    color=SPEAKER_COLORS.get(spk_from, "blue"),
                    alpha=0.5
                )
            ax.set_title(f"{spk_from} → {spk_to}", fontsize=10, fontweight="bold")
            ax.grid(alpha=0.25)

        # OVERRIDE Y-AXIS TICKS & LABELS TO 0, 6, 12, 18, 24 ft
        ax.set_yticks([
            0,
            max_val * 0.25,
            max_val * 0.50,
            max_val * 0.75,
            max_val
        ])
        ax.set_yticklabels(["0 ft", "6 ft", "12 ft", "18 ft", "24 ft"], fontsize=8)

        if j == 0:
            ax.set_ylabel("Estimated Dist", fontsize=9)
        if i == 3:
            ax.set_xlabel("Time (s)", fontsize=9)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, "02_all_interactions_4x3_grid.png"), dpi=300)
plt.close()

# --------------------------------------------------
# PLOT 3: 30% PERCENTILE RANGE CLASSIFICATION THROUGH TIME
# --------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
fig.suptitle("Dynamic 30% Distance Range Classifications Over Time", fontsize=15, fontweight="bold")

for idx, spk in enumerate(ALL_SPEAKERS):
    ax = axes[idx]
    spk_df = df_full[df_full[spk1_col] == spk]
    
    targets = [s for s in ALL_SPEAKERS if s != spk]
    for target in targets:
        pair_df = spk_df[spk_df[spk2_col] == target]
        if not pair_df.empty:
            ax.scatter(
                pair_df["Conversation_Time"],
                pair_df["Category_Code"],
                s=12,
                alpha=0.7,
                label=f"To {target}"
            )
            
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Close (30%)", "Medium", "Far (30%)"], fontsize=9)
    ax.set_title(f"Speaker: {spk} (Interactions with others)", fontsize=11, fontweight="bold", loc="left")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=3)

axes[-1].set_xlabel("Conversation Time (sec)", fontsize=11)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, "03_distance_categories_30percent.png"), dpi=300)
plt.close()

print(f"\nSuccessfully generated plots in folder: '{OUTPUT_DIR}'")