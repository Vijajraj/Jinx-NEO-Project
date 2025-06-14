import pandas as pd

# Load feed data
feed_df = pd.read_csv('neo_feed_data.csv')
print(f"📊 Feed data records: {len(feed_df)}")

# Load browse data
browse_df = pd.read_csv('neo_browse_data.csv')
print(f"📊 Browse data records: {len(browse_df)}")

# Merge both datasets
merged_df = pd.concat([feed_df, browse_df], ignore_index=True)

# Remove duplicate names (if any)
merged_df.drop_duplicates(subset=['name'], inplace=True)

# Save final combined data
merged_df.to_csv('final_neo_dataset.csv', index=False)
print(f"✅ Final merged records: {len(merged_df)}")
print("✅ Final dataset saved to final_neo_dataset.csv")
