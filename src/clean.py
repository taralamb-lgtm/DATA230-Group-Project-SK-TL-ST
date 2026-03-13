import pandas as pd

# File path
file_path = r"C:\Users\taral\Documents\GitHub\DATA230-Group-Project-SK-TL-ST\data\raw\chicago_crime.xlsx"

# Load dataset
df = pd.read_excel(file_path)

print("Raw shape:", df.shape)

# Standardize column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Convert date column (change if yours is named differently)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Drop rows without date
df = df.dropna(subset=["date"])

# Create new features
df["hour"] = df["date"].dt.hour
df["day_of_week"] = df["date"].dt.day_name()
df["month"] = df["date"].dt.month
df["is_weekend"] = df["date"].dt.dayofweek.isin([5,6]).astype(int)

# Remove duplicates
df = df.drop_duplicates()

print("Cleaned shape:", df.shape)

# Save cleaned dataset
df.to_csv(r"C:\Users\taral\Documents\GitHub\DATA230-Group-Project-SK-TL-ST\data\processed\crimes_cleaned.csv", index=False)

print("Cleaning complete. File saved to data/processed/")