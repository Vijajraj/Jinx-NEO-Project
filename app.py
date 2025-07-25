import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np
import folium
from streamlit_folium import folium_static
import plotly.express as px

# Load dataset
data = pd.read_csv("final_neo_dataset.csv")

# Load model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")


# Streamlit page config
st.set_page_config(page_title="Jinx: NEO Hazard Predictor", layout="wide")
st.title("🚀 Jinx: Near-Earth Object (NEO) Hazard Prediction AI")
st.markdown("---")

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Diameter slider based on corrected column name
min_diameter = st.sidebar.slider(
    "Minimum Diameter (km)",
    float(data['diameter_min'].min()),
    float(data['diameter_min'].max()),
    0.1
)

hazard_option = st.sidebar.selectbox(
    "Filter by Hazard Status",
    ['All', 'Hazardous', 'Safe']
)

# Filter dataset
filtered_data = data[data['diameter_min'] >= min_diameter]

if hazard_option == 'Hazardous':
    filtered_data = filtered_data[filtered_data['hazardous'] == 1]
elif hazard_option == 'Safe':
    filtered_data = filtered_data[filtered_data['hazardous'] == 0]

st.write(f"**Total NEOs after filter:** {len(filtered_data)}")
st.dataframe(filtered_data)
st.markdown("---")

# Bar Chart — Hazard Status Distribution
st.subheader("📊 NEO Hazard Status Distribution")
hazard_counts = data['hazardous'].value_counts().rename({0: 'Safe', 1: 'Hazardous'})
st.bar_chart(hazard_counts)

# Line Chart — Average Velocity by Hazard Status
st.subheader("📈 Average Velocity by Hazard Status")
avg_velocity = data.groupby('hazardous')['velocity_kms'].mean().rename({0: 'Safe', 1: 'Hazardous'})
st.line_chart(avg_velocity)

# Scatter Plot — Velocity vs Miss Distance
st.subheader("📊 Velocity vs Miss Distance Scatterplot")
plt.figure(figsize=(8, 5))
sns.scatterplot(data=filtered_data, x='velocity_kms', y='miss_distance_km', hue='hazardous', palette=['green', 'red'])
plt.xlabel("Velocity (km/s)")
plt.ylabel("Miss Distance (km)")
plt.title("Velocity vs Miss Distance by Hazard Status")
st.pyplot(plt)

# Interactive Map
st.subheader("🌐 NEO Miss Distance Map (Simulated Locations)")
m = folium.Map(location=[0, 0], zoom_start=2)

for _, row in filtered_data.iterrows():
    lat = np.random.uniform(-90, 90)
    lon = np.random.uniform(-180, 180)
    popup_info = f"{row['name']}<br>Miss Distance: {row['miss_distance_km']:.2f} km<br>Velocity: {row['velocity_kms']:.2f} km/s"
    color = 'red' if row['hazardous'] == 1 else 'green'
    folium.CircleMarker(
        location=[lat, lon],
        radius=5,
        popup=popup_info,
        color=color,
        fill=True,
        fill_opacity=0.7
    ).add_to(m)

folium_static(m, width=900, height=500)

# Animated Velocity Trend
st.subheader("📈 Animated NEO Velocity Trend (Simulated Dates)")
if 'date' not in data.columns:
    np.random.seed(42)
    random_dates = pd.date_range("2024-06-01", periods=len(data), freq='H')
    data['date'] = random_dates

fig = px.line(
    data.sort_values("date"),
    x="date",
    y="velocity_kms",
    title="NEO Velocity Over Time",
    animation_frame=data['date'].dt.date.astype(str),
    range_y=[0, data['velocity_kms'].max() + 10],
    color='hazardous',
    labels={'hazardous': 'Hazardous (1=Yes, 0=No)'}
)
st.plotly_chart(fig, use_container_width=True)

# Prediction section
st.markdown("---")
st.subheader("🚀 Predict if a NEO is Hazardous")

diameter_min_input = st.number_input("Estimated Diameter Min (km)", min_value=0.0, value=0.5)
diameter_max_input = st.number_input("Estimated Diameter Max (km)", min_value=0.0, value=1.5)
velocity_input = st.number_input("Velocity (km/s)", min_value=0.0, value=20.0)
miss_distance_input = st.number_input("Miss Distance (km)", min_value=0.0, value=750000.0)

if st.button("Predict Hazard Status"):
    X_new = np.array([[diameter_min_input, diameter_max_input, velocity_input, miss_distance_input]])
    X_new_scaled = scaler.transform(X_new)
    prediction = model.predict(X_new_scaled)
    if prediction[0] == 1:
        st.error("⚠️ This NEO is predicted to be **Hazardous**!")
    else:
        st.success("✅ This NEO is predicted to be **Safe**.")

st.markdown("---")
st.caption("👨‍💻 Developed by Vijayraj S | AI-DS | Chennai Institute of Technology")


   



