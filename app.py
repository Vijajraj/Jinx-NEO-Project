import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np
import folium
from streamlit_folium import folium_static
import plotly.express as px
import requests
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load config (secrets) — your nasa_api_key and dataset filename
nasa_api_key = st.secrets["nasa_api_key"]
dataset_file = st.secrets["dataset_file"]

# Load dataset
if os.path.exists(dataset_file):
    data = pd.read_csv(dataset_file)
else:
    st.error("⚠️ Dataset not found. Please check your config.")
    st.stop()

# Check for existing model and scaler
if os.path.exists("best_model.pkl") and os.path.exists("best_scaler.pkl"):
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("best_scaler.pkl")
else:
    model = None
    scaler = None

# Streamlit config
st.set_page_config(page_title="Jinx: NEO Hazard Predictor", layout="wide")
st.title("🚀 Jinx: Near-Earth Object (NEO) Hazard Prediction AI")
st.markdown("---")

# Sidebar retrain button
if st.sidebar.button("📥 Fetch New Data & Retrain"):
    with st.spinner("Fetching and retraining model…"):

        # Fetch new data from NASA API
        url = f"https://api.nasa.gov/neo/rest/v1/neo/browse?api_key={nasa_api_key}"
        response = requests.get(url)

        if response.status_code == 200:
            new_data = pd.DataFrame(response.json()['near_earth_objects'])
            new_data.to_csv(dataset_file, index=False)
            data = new_data
            st.success("✅ New data fetched and saved!")

            # Retrain model
            X = data[['estimated_diameter_min', 'estimated_diameter_max', 'kilometers_per_second', 'miss_distance_kilometers']].copy()
            X.columns = ['diameter_min', 'diameter_max', 'velocity_kms', 'miss_distance_km']

            # Fill missing if any
            X = X.fillna(0)

            y = data['is_potentially_hazardous_asteroid'].astype(int)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = LogisticRegression(max_iter=500)
            model.fit(X_scaled, y)

            joblib.dump(model, "best_model.pkl")
            joblib.dump(scaler, "best_scaler.pkl")

            st.success("✅ Model retrained and saved!")
        else:
            st.error(f"❌ NASA API fetch failed: {response.status_code} {response.text}")

st.sidebar.markdown("---")

# Sidebar filters
st.sidebar.header("🔍 Filters")
min_diameter = st.sidebar.slider(
    "Minimum Diameter (km)",
    float(data['estimated_diameter_min'].min()),
    float(data['estimated_diameter_min'].max()),
    0.1
)

hazard_option = st.sidebar.selectbox(
    "Filter by Hazard Status",
    ['All', 'Hazardous', 'Safe']
)

# Filter data
filtered_data = data[data['estimated_diameter_min'] >= min_diameter]
if hazard_option == 'Hazardous':
    filtered_data = filtered_data[filtered_data['is_potentially_hazardous_asteroid'] == True]
elif hazard_option == 'Safe':
    filtered_data = filtered_data[filtered_data['is_potentially_hazardous_asteroid'] == False]

st.write(f"**Total NEOs after filter:** {len(filtered_data)}")
st.dataframe(filtered_data)

# Charts
st.markdown("---")
st.subheader("📊 NEO Hazard Status Distribution")
hazard_counts = data['is_potentially_hazardous_asteroid'].value_counts().rename({False: 'Safe', True: 'Hazardous'})
st.bar_chart(hazard_counts)

st.subheader("📈 Average Velocity by Hazard Status")
avg_velocity = data.groupby('is_potentially_hazardous_asteroid')['kilometers_per_second'].mean().rename({False: 'Safe', True: 'Hazardous'})
st.line_chart(avg_velocity)

st.subheader("📊 Velocity vs Miss Distance")
plt.figure(figsize=(8, 5))
sns.scatterplot(data=filtered_data, x='kilometers_per_second', y='miss_distance_kilometers',
                hue='is_potentially_hazardous_asteroid', palette=['green', 'red'])
plt.xlabel("Velocity (km/s)")
plt.ylabel("Miss Distance (km)")
plt.title("Velocity vs Miss Distance by Hazard Status")
st.pyplot(plt)

# Map
st.subheader("🌐 NEO Miss Distance Map")
m = folium.Map(location=[0, 0], zoom_start=2)
for _, row in filtered_data.iterrows():
    lat = np.random.uniform(-90, 90)
    lon = np.random.uniform(-180, 180)
    popup_info = f"{row['name']}<br>Miss Distance: {row['miss_distance_kilometers']} km"
    color = 'red' if row['is_potentially_hazardous_asteroid'] else 'green'
    folium.CircleMarker(location=[lat, lon], radius=5, popup=popup_info, color=color, fill=True, fill_opacity=0.7).add_to(m)
folium_static(m, width=900, height=500)

# Prediction section
st.markdown("---")
st.subheader("🚀 Predict if a NEO is Hazardous")

diameter_min = st.number_input("Estimated Diameter Min (km)", min_value=0.0, value=0.5)
diameter_max = st.number_input("Estimated Diameter Max (km)", min_value=0.0, value=1.5)
velocity = st.number_input("Velocity (km/s)", min_value=0.0, value=20.0)
miss_distance = st.number_input("Miss Distance (km)", min_value=0.0, value=750000.0)

if st.button("Predict Hazard Status"):
    if model and scaler:
        X_new = np.array([[diameter_min, diameter_max, velocity, miss_distance]])
        X_new_scaled = scaler.transform(X_new)
        prediction = model.predict(X_new_scaled)
        if prediction[0] == 1:
            st.error("⚠️ This NEO is predicted to be **Hazardous**!")
        else:
            st.success("✅ This NEO is predicted to be **Safe**.")
    else:
        st.error("⚠️ Model not trained yet. Please use the sidebar button to retrain.")

st.markdown("---")
st.caption("👨‍💻 Developed by Vijayraj S | AI-DS | Chennai Institute of Technology")

     




