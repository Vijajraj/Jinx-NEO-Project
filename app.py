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

# Set Streamlit page config
st.set_page_config(page_title="Jinx: NEO Hazard Predictor", layout="wide")

# Load dataset
data = pd.read_csv('final_neo_dataset.csv')

# Load model and scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Title
st.title("🚀 Jinx: Near-Earth Object (NEO) Hazard Prediction AI")
st.markdown("---")

# Sidebar filters
st.sidebar.header("🔍 Filters")

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

filtered_data = data[data['diameter_min'] >= min_diameter]

if hazard_option == 'Hazardous':
    filtered_data = filtered_data[filtered_data['hazardous'] == 1]
elif hazard_option == 'Safe':
    filtered_data = filtered_data[filtered_data['hazardous'] == 0]

st.write(f"**Total NEOs after filter:** {len(filtered_data)}")

st.dataframe(filtered_data)

st.markdown("---")

# Bar chart — Hazard Status Distribution
st.subheader("📊 NEO Hazard Status Distribution")
hazard_counts = data['hazardous'].value_counts().rename({0: 'Safe', 1: 'Hazardous'})
st.bar_chart(hazard_counts)

# Line chart — Average Velocity by Hazard Status
st.subheader("📈 Average Velocity by Hazard Status")
avg_velocity = data.groupby('hazardous')['velocity_kms'].mean().rename({0: 'Safe', 1: 'Hazardous'})
st.line_chart(avg_velocity)

# Scatter Plot — Velocity vs Miss Distance
st.subheader("📊 Velocity vs Miss Distance Scatterplot")

plt.figure(figsize=(8,5))
sns.scatterplot(data=filtered_data, x='velocity_kms', y='miss_distance_km', hue='hazardous', palette=['green', 'red'])
plt.xlabel("Velocity (km/s)")
plt.ylabel("Miss Distance (km)")
plt.title("Velocity vs Miss Distance by Hazard Status")
st.pyplot(plt)

# Interactive Map — NEO Miss Distances
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

# NEO Hazard Prediction
st.markdown("---")
st.subheader("🚀 Predict if a NEO is Hazardous")

diameter_min = st.number_input("Estimated Diameter Min (km)", min_value=0.0, value=0.5)
diameter_max = st.number_input("Estimated Diameter Max (km)", min_value=0.0, value=1.5)
velocity = st.number_input("Velocity (km/s)", min_value=0.0, value=20.0)
miss_distance = st.number_input("Miss Distance (km)", min_value=0.0, value=750000.0)

if st.button("Predict Hazard Status"):
    X_new = np.array([[diameter_min, diameter_max, velocity, miss_distance]])
    X_new_scaled = scaler.transform(X_new)
    prediction = model.predict(X_new_scaled)
    if prediction[0] == 1:
        st.error("⚠️ This NEO is predicted to be **Hazardous**!")
    else:
        st.success("✅ This NEO is predicted to be **Safe**.")

# Hugging Face Q&A Section
st.markdown("---")
st.subheader("❓ Ask Jinx AI: NEO Q&A")

# Hugging Face API Config
hf_token = st.secrets["huggingface"]["api_token"]
headers = {
    "Authorization": f"Bearer {hf_token}"
}
api_url = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"

def ask_question(question, context):
    payload = {
        "inputs": {
            "question": question,
            "context": context
        }
    }
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["answer"]
    else:
        return f"Error: {response.status_code} - {response.text}"

context = st.text_area("📝 Provide your context text here", 
    "Near-Earth Objects (NEOs) are comets and asteroids nudged by gravitational attractions of nearby planets into orbits that allow them to enter Earth's neighborhood."
)

question = st.text_input("🔍 Ask your question")

if st.button("Get Answer"):
    if question.strip() and context.strip():
        answer = ask_question(question, context)
        st.write(f"**📝 Answer:** {answer}")
    else:
        st.error("Please provide both a question and context.")

st.markdown("---")
st.caption("👨‍💻 Developed by Vijayraj S | AI-DS | Chennai Institute of Technology")


  







   

