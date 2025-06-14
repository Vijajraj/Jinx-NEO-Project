# Import required libraries
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# NASA API key
API_KEY = '4b6fTfYjn62ulMxmvlxd0j5tbMq2lXkdXA4m7Try'

# API URL to fetch NEO data for a date range
url = f'https://api.nasa.gov/neo/rest/v1/feed?start_date=2024-06-01&end_date=2024-06-07&api_key={API_KEY}'

# Fetching the data
response = requests.get(url)
data = response.json()

# Extract NEO details
neo_list = []
for date in data['near_earth_objects']:
    for neo in data['near_earth_objects'][date]:
        if neo['close_approach_data']:
            neo_list.append({
                'name': neo['name'],
                'diameter_min': neo['estimated_diameter']['kilometers']['estimated_diameter_min'],
                'diameter_max': neo['estimated_diameter']['kilometers']['estimated_diameter_max'],
                'velocity_kms': float(neo['close_approach_data'][0]['relative_velocity']['kilometers_per_second']),
                'miss_distance_km': float(neo['close_approach_data'][0]['miss_distance']['kilometers']),
                'hazardous': 1 if neo['is_potentially_hazardous_asteroid'] else 0
            })

# Convert to DataFrame
df = pd.DataFrame(neo_list)

# Save DataFrame to CSV
df.to_csv('neo_data.csv', index=False)
print("\nData saved to neo_data.csv")

# Add average diameter feature
df['average_diameter'] = (df['diameter_min'] + df['diameter_max']) / 2

# Categorize threat level based on average diameter & hazardous status
def hazard_level(row):
    if row['hazardous'] == 0:
        return 'Safe'
    elif row['average_diameter'] > 0.3:
        return 'Severe Threat'
    else:
        return 'Moderate Threat'

df['threat_level'] = df.apply(hazard_level, axis=1)

# EDA Visualizations
sns.countplot(x='hazardous', data=df)
plt.title('Hazardous vs Non-Hazardous NEOs')
plt.show()

sns.countplot(x='threat_level', data=df)
plt.title('NEO Threat Levels')
plt.show()

sns.scatterplot(x='velocity_kms', y='miss_distance_km', hue='hazardous', data=df)
plt.title('Velocity vs Miss Distance')
plt.show()

sns.heatmap(df.corr(), annot=True)
plt.title('Feature Correlation Heatmap')
plt.show()

# Prepare features and target
X = df[['diameter_min', 'diameter_max', 'velocity_kms', 'miss_distance_km', 'average_diameter']]
y = df['hazardous']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy: {:.2f}%".format(accuracy * 100))

# Classification Report
report = classification_report(y_test, y_pred)
print("\nClassification Report:\n", report)

# ROC Curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', label='ROC Curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Save project summary to text file
with open("project_summary.txt", "w") as file:
    file.write(f"Random Forest Model Accuracy: {accuracy * 100:.2f}%\n\n")
    file.write("Classification Report:\n")
    file.write(report)
    file.write("\nThreat Level Distribution:\n")
    file.write(df['threat_level'].value_counts().to_string())

print("\nProject summary saved as project_summary.txt")
