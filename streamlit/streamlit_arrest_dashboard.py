import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Crime Arrest Prediction System", layout="wide")

st.title("Crime Arrest Prediction System (Interactive)")
st.markdown(
    "Use the input panel to estimate whether a reported crime is likely to result in an arrest."
)

@st.cache_data
def load_data():
    file_path = "data/processed/crimes_cleaned.csv"
    df = pd.read_csv(file_path)

    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    if ("hour" not in df.columns or "day_of_week" not in df.columns) and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "hour" not in df.columns:
            df["hour"] = df["date"].dt.hour
        if "day_of_week" not in df.columns:
            df["day_of_week"] = df["date"].dt.day_name()

    required_cols = ["primary_type", "district", "hour", "day_of_week", "domestic", "arrest"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.write("Available columns in your dataset:")
        st.write(list(df.columns))
        st.stop()

    df = df[required_cols].copy()
    df = df.dropna()

    df["primary_type"] = df["primary_type"].astype(str).str.strip()
    df["district"] = df["district"].astype(str).str.strip()
    df["day_of_week"] = df["day_of_week"].astype(str).str.strip()

    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    df = df.dropna(subset=["hour"])
    df["hour"] = df["hour"].astype(int)

    df["domestic"] = df["domestic"].astype(str).str.strip().str.lower()
    df["domestic"] = df["domestic"].replace({
        "true": "Yes",
        "false": "No",
        "1": "Yes",
        "0": "No",
        "yes": "Yes",
        "no": "No"
    })

    if str(df["arrest"].dtype) == "bool":
        df["arrest"] = df["arrest"].astype(int)
    else:
        df["arrest"] = df["arrest"].astype(str).str.strip().str.lower()
        df["arrest"] = df["arrest"].replace({
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
            "yes": 1,
            "no": 0
        })
        df["arrest"] = pd.to_numeric(df["arrest"], errors="coerce")
        df = df.dropna(subset=["arrest"])
        df["arrest"] = df["arrest"].astype(int)

    return df


df = load_data()

@st.cache_resource
def train_model(data):
    X = data[["primary_type", "district", "hour", "day_of_week", "domestic"]]
    y = data["arrest"]

    categorical_features = ["primary_type", "district", "day_of_week", "domestic"]
    numeric_features = ["hour"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features)
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced"
            ))
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy


model, accuracy = train_model(df)

st.sidebar.header("Crime Input Panel")

crime_types = ["All"] + sorted(df["primary_type"].dropna().unique().tolist())
districts = sorted(df["district"].dropna().unique().tolist())
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
domestic_options = ["Yes", "No"]

crime_type = st.sidebar.selectbox("Crime Type", crime_types)
district = st.sidebar.selectbox("District", districts)
hour = st.sidebar.slider("Hour", 0, 23, 12)
day = st.sidebar.selectbox("Day", days)
domestic = st.sidebar.selectbox("Domestic Case", domestic_options)

default_chart_crime_type = df["primary_type"].mode()[0]
chart_crime_type = default_chart_crime_type if crime_type == "All" else crime_type

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Total Records", f"{len(df):,}")
with m2:
    st.metric("Model Accuracy", f"{accuracy:.2%}")
with m3:
    st.metric("Dataset Arrest Rate", f"{df['arrest'].mean():.2%}")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Selected Inputs")
    display_df = pd.DataFrame({
        "Crime Type": [crime_type],
        "District": [district],
        "Hour": [hour],
        "Day": [day],
        "Domestic": [domestic]
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)

with col2:
    st.subheader("Prediction Result")

    if st.button("Predict Arrest", use_container_width=True):
        if crime_type == "All":
            all_types = sorted(df["primary_type"].dropna().unique().tolist())
            temp_df = pd.DataFrame({
                "primary_type": all_types,
                "district": [district] * len(all_types),
                "hour": [hour] * len(all_types),
                "day_of_week": [day] * len(all_types),
                "domestic": [domestic] * len(all_types)
            })
            probs = model.predict_proba(temp_df)
            prob_yes = probs[:, 1].mean() * 100
            prob_no = probs[:, 0].mean() * 100
            predicted_label = "YES" if prob_yes >= prob_no else "NO"
            st.info("Using average probabilities across all crime types.")
        else:
            input_df = pd.DataFrame({
                "primary_type": [crime_type],
                "district": [district],
                "hour": [hour],
                "day_of_week": [day],
                "domestic": [domestic]
            })

            probs = model.predict_proba(input_df)[0]
            prob_no = probs[0] * 100
            prob_yes = probs[1] * 100
            predicted_label = "YES" if prob_yes >= prob_no else "NO"

        st.metric("Predicted Arrest", predicted_label)
        st.metric("Probability of YES", f"{prob_yes:.2f}%")
        st.metric("Probability of NO", f"{prob_no:.2f}%")

        if predicted_label == "YES":
            st.success("This case has a higher predicted likelihood of arrest.")
        else:
            st.warning("This case has a higher predicted likelihood of no arrest.")
    else:
        st.info("Click the button to generate a prediction.")

st.markdown("---")
st.header("Machine Learning Insights")

if crime_type == "All":
    st.info(f"For the charts below, 'All' uses the most common crime type: {chart_crime_type}.")

row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

with row1_col1:
    st.subheader("1. What-If Analysis by Hour")

    hour_test_df = pd.DataFrame({
        "primary_type": [chart_crime_type] * 24,
        "district": [district] * 24,
        "hour": list(range(24)),
        "day_of_week": [day] * 24,
        "domestic": [domestic] * 24
    })

    hour_probs = model.predict_proba(hour_test_df)[:, 1] * 100

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(list(range(24)), hour_probs, marker="o")
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Predicted Arrest Probability (%)")
    ax1.set_title("Predicted Probability Across Hours")
    ax1.set_xticks(range(0, 24, 4))
    st.pyplot(fig1)

with row1_col2:
    st.subheader("2. Scenario Comparison")

    alt_hour = hour + 6 if hour <= 17 else hour - 6
    alt_domestic = "No" if domestic == "Yes" else "Yes"

    scenario_df = pd.DataFrame({
        "primary_type": [chart_crime_type, chart_crime_type, chart_crime_type],
        "district": [district, district, district],
        "hour": [hour, alt_hour, hour],
        "day_of_week": [day, day, day],
        "domestic": [domestic, domestic, alt_domestic]
    })

    scenario_probs = model.predict_proba(scenario_df)[:, 1] * 100

    scenario_labels = [
        "Current",
        f"Hour {alt_hour}",
        f"Domestic {alt_domestic}"
    ]

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(scenario_labels, scenario_probs)
    ax2.set_ylabel("Predicted Arrest Probability (%)")
    ax2.set_title("Scenario Comparison")
    plt.xticks(rotation=15)
    st.pyplot(fig2)

with row2_col1:
    st.subheader("3. Feature Importance")

    rf_model = model.named_steps["classifier"]
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf_model.feature_importances_
    }).sort_values("Importance", ascending=False).head(8)

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.barh(feature_importance_df["Feature"][::-1], feature_importance_df["Importance"][::-1])
    ax3.set_xlabel("Importance Score")
    ax3.set_ylabel("Feature")
    ax3.set_title("Top Model Features")
    st.pyplot(fig3)

with row2_col2:
    st.subheader("4. Probability by Day of Week")

    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    day_test_df = pd.DataFrame({
        "primary_type": [chart_crime_type] * 7,
        "district": [district] * 7,
        "hour": [hour] * 7,
        "day_of_week": days_order,
        "domestic": [domestic] * 7
    })

    day_probs = model.predict_proba(day_test_df)[:, 1] * 100

    fig4, ax4 = plt.subplots(figsize=(6, 4))
    ax4.bar(days_order, day_probs)
    ax4.set_ylabel("Predicted Arrest Probability (%)")
    ax4.set_title("Predicted Probability Across Days")
    plt.xticks(rotation=30)
    st.pyplot(fig4)

st.markdown("---")
st.subheader("Dashboard Summary")
st.write(
    """
    This Streamlit dashboard is designed for predictive analysis rather than descriptive reporting.
    Unlike the Tableau dashboards used in the mid-presentation, this dashboard supports interactive
    decision-making through arrest prediction, what-if analysis, scenario comparison, feature importance,
    and day-of-week probability comparison.
    """
)