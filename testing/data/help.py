import matplotlib.pyplot as plt
import pandas as pd

# Load CSV
df = pd.read_csv("cdc_flu_data.csv")

# Use weighted ILI
df["ILI_plot"] = df["%WEIGHTED ILI"]

# Keep only season weeks (weeks 40-52 of year Y, 1-20 of year Y+1)
df = df[df["WEEK"].between(40, 52) | df["WEEK"].between(1, 20)].copy()

# Assign season_year: week 40-52 → current year, week 1-20 → previous year
df["season_year"] = df.apply(
    lambda row: row["YEAR"] if row["WEEK"] >= 40 else row["YEAR"] - 1, axis=1
)

# Map weeks to 0-based season_week (0-32)
df["season_week"] = df.apply(
    lambda row: row["WEEK"] - 40 if row["WEEK"] >= 40 else row["WEEK"] + 12, axis=1
)

# Select middle 12 seasons
seasons = list(range(2010, 2024))

# Prepare figure
n_rows, n_cols = 3, 5
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12), sharex=True, sharey=True)
axes = axes.flatten()

# Define colors for regions
regions = sorted(df["REGION"].unique())
colors = plt.cm.tab10.colors  # Up to 10 colors
region_colors = {region: colors[i % 10] for i, region in enumerate(regions)}

# Plot
for ax, season in zip(axes, seasons):
    for region in regions:
        mask = (df["season_year"] == season) & (df["REGION"] == region)
        df_region = df.loc[mask].copy()
        df_region = df_region.sort_values("season_week")
        ax.plot(
            df_region["season_week"],
            df_region["ILI_plot"],
            color=region_colors[region],
            alpha=0.7,
            linewidth=1.5,
        )

    ax.set_title(f"Season {season}", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.3)

# Set shared labels
for ax in axes[-n_cols:]:
    ax.set_xlabel("Week (season index 0 = week 40)", fontsize=12)
for ax in axes[::n_cols]:
    ax.set_ylabel("% Weighted ILI", fontsize=12)

plt.suptitle("Influenza Incidence (% Weighted ILI) by Season and Region", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("data.png")
plt.show()
