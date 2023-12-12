import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file
df = pd.read_csv("metadata_osr.csv")

# Shuffle the dataframe
df = df.sample(frac=1).reset_index(drop=True)

# Split the data into 80% training and 20% inference
train_df, inference_df = train_test_split(df, test_size=0.20)

# Save the split dataframes to new CSV files
train_df.to_csv("metadata_osr_training.csv", index=False)
inference_df.to_csv("metadata_osr_inference.csv", index=False)
