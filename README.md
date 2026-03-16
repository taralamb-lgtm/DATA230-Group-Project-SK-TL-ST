# Chicago Crime Data Visualization Project

## Dataset Source
This project uses data from the City of Chicago Open Data Portal.

Dataset:
https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2/about_data

The dataset contains reported crime incidents in Chicago from 2001 to the present, including details such as crime type, location, arrest status, and timestamps.

---

## Data Extraction

For this project, we extracted a subset of the dataset and stored it locally for analysis.

The processed dataset (`crimes_cleaned.csv`) contains crime records from:

- November 2025
- December 2025
- January 2026
- February 2026

Total records in the extracted dataset: **62,723**

---

## Project Goal

The goal of this project is to create visualizations and dashboards to analyze crime patterns in Chicago.

We will analyze trends such as:

- Crime frequency over time
- Crimes by hour of the day
- Crimes by day of the week
- Distribution of crime types
- Arrest rates across crime categories
- Geographic crime patterns

---

## Project Structure

DATA230-Group-Project

data/
- raw/
  - chicago_crime.xlsx
- processed/
  - crimes_cleaned.csv

src/
- clean.py

requirements.txt
README.md

---

## Data Processing

The script `src/clean.py` performs the following steps:

- Loads the raw dataset
- Standardizes column names
- Converts date columns
- Creates additional features:
  - hour
  - day_of_week
  - month
  - is_weekend
- Removes duplicate records
- Saves the cleaned dataset

---

## Technologies Used

- Python
- Pandas
- RAPIDS
- Tableau

---

## Future Work

We will use the processed dataset to create visualizations and dashboards to explore crime patterns and trends in Chicago.