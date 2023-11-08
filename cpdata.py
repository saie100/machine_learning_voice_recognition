import pandas as pd
import shutil
import os

# Replace 'metadata.csv' with your actual CSV file's name
metadata_csv = "metadata_osr.csv"
# The destination directory
destination_dir = "data"

# Make sure that the destination directory exists, if not, create it
# os.makedirs(destination_dir, exist_ok=True)

# Read the CSV file
metadata_df = pd.read_csv(metadata_csv)

# Copy each file to the destination directory
for index, row in metadata_df.iterrows():
    # The source path for the current file
    source_path = row["relative_path"]
    # The target path for the current file
    target_path = os.path.join(destination_dir, source_path)

    # Make sure the source file exists
    if os.path.exists(source_path):
        # Copy the file
        shutil.copy2(source_path, target_path)
        print(f"Copied: {source_path} to {target_path}")
    else:
        print(f"File not found: {source_path}")

print("File copying process completed.")
