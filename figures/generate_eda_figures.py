import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
DATA_PATH = "data/processed/crimes_cleaned.csv"
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

# Make column names consistent just in case
df.columns = df.columns.str.strip()

# -----------------------------
# 1. Crime counts by primary type
# -----------------------------
top_types = df["primary_type"].value_counts().head(10)

plt.figure(figsize=(8, 5))
top_types.sort_values().plot(kind="barh")
plt.title("Top Crime Types")
plt.xlabel("Number of Incidents")
plt.ylabel("Primary Type")
plt.tight_layout()
plt.savefig(FIG_DIR / "eda_primary_type_counts.png", dpi=300)
plt.close()

# -----------------------------
# 2. Crime counts by district
# -----------------------------
district_counts = df["district"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
district_counts.plot(kind="bar")
plt.title("Crime Incidents by Police District")
plt.xlabel("District")
plt.ylabel("Number of Incidents")
plt.tight_layout()
plt.savefig(FIG_DIR / "eda_district_counts.png", dpi=300)
plt.close()

# -----------------------------
# 3. Crime frequency by hour
# -----------------------------
hour_counts = df["hour"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
hour_counts.plot(kind="line", marker="o")
plt.title("Crime Incidents by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Incidents")
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig(FIG_DIR / "eda_hourly_pattern.png", dpi=300)
plt.close()

# -----------------------------
# 4. Crime frequency by day of week
# -----------------------------
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

if "day_of_week" in df.columns:
    day_counts = df["day_of_week"].value_counts().reindex(day_order)
else:
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.day_name()
    day_counts = df["day_of_week"].value_counts().reindex(day_order)

plt.figure(figsize=(8, 5))
day_counts.plot(kind="bar")
plt.title("Crime Incidents by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Number of Incidents")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(FIG_DIR / "eda_day_of_week.png", dpi=300)
plt.close()

# -----------------------------
# 5. Arrest outcome distribution
# -----------------------------
arrest_counts = df["arrest"].value_counts()

# Optional: make labels cleaner if values are True/False
arrest_counts.index = arrest_counts.index.map(lambda x: "Arrest" if x in [True, "True", 1, "1"] else "No Arrest")

plt.figure(figsize=(6, 5))
arrest_counts.plot(kind="bar")
plt.title("Arrest Outcome Distribution")
plt.xlabel("Arrest Outcome")
plt.ylabel("Number of Incidents")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIG_DIR / "eda_arrest_distribution.png", dpi=300)
plt.close()

# -----------------------------
# 6. Domestic incident distribution
# -----------------------------
domestic_counts = df["domestic"].value_counts()

# Optional: make labels cleaner if values are True/False
domestic_counts.index = domestic_counts.index.map(lambda x: "Domestic" if x in [True, "True", 1, "1"] else "Non-Domestic")

plt.figure(figsize=(6, 5))
domestic_counts.plot(kind="bar")
plt.title("Domestic vs. Non-Domestic Incidents")
plt.xlabel("Domestic Status")
plt.ylabel("Number of Incidents")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIG_DIR / "eda_domestic_distribution.png", dpi=300)
plt.close()

print("EDA figures saved in the figures folder.")