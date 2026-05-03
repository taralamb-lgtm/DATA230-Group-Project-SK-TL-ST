# Chicago Crime Data Visualization Project

## Dataset Source

This project uses data from the City of Chicago Open Data Portal.

Dataset:  
https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2/about_data

The dataset contains reported crime incidents in Chicago from 2001 to the present, including details such as crime type, location, arrest status, and timestamps.

---

## Project Goal

The goal of this project is to analyze Chicago crime patterns and build interactive dashboards that help explain crime trends, arrest outcomes, clustering patterns, and classification model predictions.

This project includes descriptive analytics, clustering analysis, and a machine learning classification workflow focused on predicting whether an arrest occurred.

Key analysis areas include:

- Crime frequency by type
- Crime records by district
- Crimes by hour of day
- Crimes by day of week
- Arrest outcomes
- Predicted arrest outcomes
- Classification model performance
- Crime pattern grouping through clustering

---

## Data Extraction

For this project, a subset of the Chicago crime dataset was extracted and processed locally.

The processed dataset `crimes_cleaned.csv` contains crime records from:

- November 2025
- December 2025
- January 2026
- February 2026

Total records in the extracted dataset: **62,723**

---

## Project Structure

```text
DATA230-Group-Project/

data/
  raw/
    chicago_crime.xlsx
  processed/
    crimes_cleaned.csv

notebooks/
  Chicago Crime Analysis_ Districts, Arrest Rat...
  Chicago Crime Overview_ Type, Time & Tre...
  eda.ipynb
  rapids_viz.ipynb

outputs/
  classification_metrics.json
  confusion_matrix.csv
  feature_importance.csv
  prediction_outputs.csv
  roc_curve.csv

src/
  clean.py
  train_classification_model.py

streamlit/
  streamlit_arrest_dashboard.py

Power BI/
  cluster_summary_for_powerbi.csv
  cluster_top_types_for_powerbi.csv
  crime_arrest_classification_dashboard.pbix
  crime_clusters_for_powerbi.csv
  DATA230_Clusterdash.pbix
  powerbi.md

requirements.txt
README.md
```

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
- Saves the cleaned dataset as `data/processed/crimes_cleaned.csv`

---

## Classification Model

A classification model was created to predict whether a crime incident resulted in an arrest.

The classification workflow uses features such as:

- primary_type
- district
- hour
- day_of_week
- domestic

The target variable is:

- arrest

Classification model file:

```text
src/train_classification_model.py
```

The model outputs were saved in the `outputs/` folder, including:

- `classification_metrics.json`
- `confusion_matrix.csv`
- `feature_importance.csv`
- `prediction_outputs.csv`
- `roc_curve.csv`

The exported prediction file `outputs/prediction_outputs.csv` was also used to build the Power BI classification dashboard.

---

## Streamlit Dashboard

The Streamlit dashboard provides an interactive classification model view.

It includes:

- Model performance metrics
- Confusion matrix
- ROC curve
- Feature importance
- Prediction output exploration

Streamlit app file:

```text
streamlit/streamlit_arrest_dashboard.py
```

---

## Power BI Dashboards

Power BI was used to create presentation-ready dashboards for both classification and clustering analysis.

### Classification Power BI Dashboard

The classification Power BI dashboard was created using:

```text
outputs/prediction_outputs.csv
```

The dashboard contains two pages:

#### Page 1: Model Performance Overview

This page summarizes classification performance with:

- Total records
- Accuracy
- Correct predictions
- Incorrect predictions
- Actual vs predicted arrest outcomes
- Prediction accuracy breakdown
- Confusion matrix summary

#### Page 2: Prediction Pattern Insights

This page focuses on prediction patterns across crime-related fields, including:

- Predicted arrest outcomes by crime type
- Crime records by district
- Crime records by hour
- Crime records by day of week
- Interactive slicers for:
  - primary_type
  - district
  - domestic

Power BI file:

```text
Power BI/crime_arrest_classification_dashboard.pbix
```

### Clustering Power BI Dashboard

A clustering Power BI dashboard was also added to support exploratory crime pattern analysis.

The clustering dashboard uses Power BI-ready clustering files, including:

- `cluster_summary_for_powerbi.csv`
- `cluster_top_types_for_powerbi.csv`
- `crime_clusters_for_powerbi.csv`

The clustering dashboard helps show grouped crime patterns and supports visual analysis of how crime incidents cluster based on selected features.

Power BI file:

```text
Power BI/DATA230_Clusterdash.pbix
```

---

## Work Completed After Mid-Presentation

After the mid-presentation, the project was expanded from descriptive crime visualization into additional machine learning and dashboard workflows.

Additional work completed includes:

- Built an arrest classification model to predict whether a crime incident resulted in an arrest
- Exported classification model outputs for dashboard use, including:
  - classification metrics
  - confusion matrix
  - feature importance
  - ROC curve data
  - prediction outputs
- Created a Streamlit classification dashboard to display model performance, ROC curve, feature importance, and prediction results
- Created a Power BI classification dashboard using `outputs/prediction_outputs.csv`
- Added Power BI classification pages for:
  - model performance overview
  - prediction pattern insights
- Added interactive Power BI slicers for crime type, district, and domestic crime status
- Added the clustering Power BI dashboard to support crime pattern grouping and exploratory analysis
- Organized the final workflow so Streamlit focuses on detailed model analysis while Power BI focuses on presentation-ready insights

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Power BI
- Tableau
- RAPIDS

---

## Current Status

Completed project components:

- Data cleaning and preprocessing
- Descriptive crime analysis
- Classification model training
- Classification output exports
- Streamlit classification dashboard
- Power BI classification dashboard
- Power BI clustering dashboard

Remaining work mainly includes dashboard layout polishing, final screenshots, final documentation review, and presentation preparation.