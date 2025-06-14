import requests
import pandas as pd

# NASA API Key
API_KEY = '4b6fTfYjn62ulMxmvlxd0j5tbMq2lXkdXA4m7Try'

# Base Browse API URL
base_url = f'https://api.nasa.gov/neo/rest/v1/neo/browse?api_key={API_KEY}'

# Number of pages to fetch
total_pages = 5  # you can increase this

# Empty list to store NEO details
browse_data = []

for page in range(total_pages):
    print(f"\nFetching page {page + 1} of {total_pages}...")

    url = f"{base_url}&page={page}&size=50"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"⚠️ Failed to fetch page {page + 1}: {response.status_code}")
        continue

    data = response.json()

    for neo in data['near_earth_objects']:
        if neo['close_approach_data']:
            browse_data.append({
                'name': neo['name'],
                'diameter_min': neo['estimated_diameter']['kilometers']['estimated_diameter_min'],
                'diameter_max': neo['estimated_diameter']['kilometers']['estimated_diameter_max'],
                'hazardous': 1 if neo['is_potentially_hazardous_asteroid'] else 0,
                'neo_reference_id': neo['neo_reference_id'],
                'nasa_jpl_url': neo['nasa_jpl_url']
            })

print(f"\n✅ Total NEOs fetched from Browse API: {len(browse_data)}")

# Save to CSV
if browse_data:
    df_browse = pd.DataFrame(browse_data)
    df_browse.to_csv('neo_browse_data.csv', index=False)
    print("✅ Data saved to neo_browse_data.csv successfully.")
else:
    print("⚠️ No data collected.")
