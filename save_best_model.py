import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import os

# Define absolute paths
project_path = r"C:\NEO_Project"
dataset_path = os.path.join(project_path, "final_neo_dataset.csv")
models_dir = os.path.join(project_path, "models")
model_path = os.path.join(models_dir, "best_model.pkl")
scaler_path = os.path.join(models_dir, "scaler.pkl")

# Load final dataset
data = pd.read_csv(dataset_path)

# Features and target
X = data[['diameter_min', 'diameter_max', 'velocity_kms', 'miss_distance_km']]
y = data['hazardous']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict and show accuracy
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained. Accuracy on test set: {accuracy:.2f}")

# Ensure models directory exists
os.makedirs(models_dir, exist_ok=True)

# Save model and scaler with absolute paths
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print("✅ Improved model and scaler saved successfully in 'models' folder.")
