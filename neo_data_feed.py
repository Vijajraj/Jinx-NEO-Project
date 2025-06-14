import requests
import pandas as pd

# NASA API Key
API_KEY = '4b6fTfYjn62ulMxmvlxd0j5tbMq2lXkdXA4m7Try'

# Date ranges (max 7 days)
date_ranges = [
    ('2024-06-01', '2024-06-07'),
    ('2024-06-08', '2024-06-14'),
    ('2024-06-15', '2024-06-21'),
    ('2024-06-22', '2024-06-28'),
    ('2024-06-29', '2024-06-30')
]



feed_data = []

for start_date, end_date in date_ranges:
    print(f"\nFetching data from {start_date} to {end_date}...")
    
    url = f'https://api.nasa.gov/neo/rest/v1/feed?start_date={start_date}&end_date={end_date}&api_key={API_KEY}'
    response = requests.get(url)
    
    # Check if response was successful
    if response.status_code != 200:
        print(f"⚠️ Failed to fetch data: {response.status_code} {response.text}")
        continue
    
    data = response.json()

    # Check if 'near_earth_objects' exists
    if 'near_earth_objects' not in data:
        print("⚠️ No 'near_earth_objects' key found in response.")
        print(data)  # Print the full response for debugging
        continue

    for date in data['near_earth_objects']:
        for neo in data['near_earth_objects'][date]:
            if neo['close_approach_data']:
                feed_data.append({
                    'name': neo['name'],
                    'diameter_min': neo['estimated_diameter']['kilometers']['estimated_diameter_min'],
                    'diameter_max': neo['estimated_diameter']['kilometers']['estimated_diameter_max'],
                    'velocity_kms': float(neo['close_approach_data'][0]['relative_velocity']['kilometers_per_second']),
                    'miss_distance_km': float(neo['close_approach_data'][0]['miss_distance']['kilometers']),
                    'hazardous': 1 if neo['is_potentially_hazardous_asteroid'] else 0
                })

print(f"\n✅ Total NEOs collected: {len(feed_data)}")

# Save to CSV if we have data
if feed_data:
    df_feed = pd.DataFrame(feed_data)
    df_feed.to_csv('neo_feed_data.csv', index=False)
    print("✅ Data saved to neo_feed_data.csv successfully.")
else:
    print("⚠️ No data collected.")
