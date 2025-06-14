import pandas as pd

# Load both CSV files
feed_df = pd.read_csv('neo_feed_data.csv')
browse_df = pd.read_csv('neo_browse_data.csv')

# Show number of records in each dataset
print(f"📊 Feed data records: {len(feed_df)}")
print(f"📊 Browse data records: {len(browse_df)}")

# Combine the two datasets into one
merged_df = pd.concat([feed_df, browse_df], ignore_index=True)

# Remove duplicate NEO names (if any)
merged_df.drop_duplicates(subset=['name'], inplace=True)

# Show total after merge
print(f"✅ Total records after merging: {len(merged_df)}")

# Save merged data to CSV
merged_df.to_csv('merged_neo_data.csv', index=False)
print("✅ Merged data saved to merged_neo_data.csv successfully.")
