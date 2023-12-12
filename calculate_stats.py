import pandas as pd
import matplotlib.pyplot as plt


def calculate_training_stats(csv_file, output_csv_file):
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Check if the required columns are in the dataframe
    if "accuracy" in df.columns and "loss" in df.columns and "model" in df.columns:
        # Group by 'model' and calculate mean and standard deviation
        stats = df.groupby("model").agg({"accuracy": ["mean", "std"]})
        stats.columns = ["_".join(col).strip() for col in stats.columns.values]
        stats.reset_index(inplace=True)
        stats.rename(
            columns={
                "model": "model",
                "accuracy_mean": "mean_accuracy",
                "accuracy_std": "std_accuracy",
            },
            inplace=True,
        )

        # Write the statistics to a new CSV file
        stats.to_csv(output_csv_file, index=False)
        return stats
    else:
        return None


def calculate_inference_stats(csv_file, output_csv_file):
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Check if the required columns are in the dataframe
    if "accuracy" in df.columns and "model" in df.columns:
        # Group by 'model' and calculate mean and standard deviation
        stats = df.groupby("model").agg({"accuracy": ["mean", "std"]})
        stats.columns = ["_".join(col).strip() for col in stats.columns.values]
        stats.reset_index(inplace=True)
        stats.rename(
            columns={
                "model": "model",
                "accuracy_mean": "mean_accuracy",
                "accuracy_std": "std_accuracy",
            },
            inplace=True,
        )

        # Write the statistics to a new CSV file
        stats.to_csv(output_csv_file, index=False)
        return stats
    else:
        return None


def write_to_file(data, header, csv_file):
    file_exists = os.path.isfile(csv_file)

    # Open the file in append mode ('a')
    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)

        # If the file is newly created, write the header first
        if not file_exists:
            writer.writerow(header)  # Adjust column names as needed

        # Write the data row
        writer.writerow(data)


def print_image(csv_file_path, image_file_path):
    df = pd.read_csv(csv_file_path)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    syracuse_orange = "#D44500"  # Orange
    syracuse_blue = "#212B6D"  # Blue
    # Iterate over the DataFrame and plot each bar with a specific color
    for index, row in df.iterrows():
        color = "skyblue" if "train" in row["model"] else "lightcoral"
        bar = ax.bar(
            row["model"],
            row["mean_accuracy"],
            yerr=row["std_accuracy"],
            capsize=5,
            color=color,
        )

        # Add value label on top of the bar
        yval = row["mean_accuracy"]
        ax.text(
            row["model"], yval, round(yval, 3), va="bottom", ha="center", fontsize=8
        )

    ax.set_title("Mean Accuracy with Standard Deviation")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.8, 1.05 * df["mean_accuracy"].max())

    plt.tight_layout()

    # Save the figure to the specified path
    plt.savefig(image_file_path)


print("\nTRAINING")
# Replace 'your_file_path.csv' with the path to your CSV file
csv_file_training = "models/results.csv"
csv_file_training_stats = "models/results_stats.csv"
image_file_training_path = "models/results_training.png"
training_stats = calculate_training_stats(
    csv_file=csv_file_training, output_csv_file=csv_file_training_stats
)
print_image(
    csv_file_path=csv_file_training_stats, image_file_path=image_file_training_path
)

print(training_stats)

print("\nINFERENCE")
# Replace 'your_file_path.csv' with the path to your CSV file
csv_file_inference = "models/results_inference.csv"
csv_file_inference_stats = "models/results_inference_stats.csv"
image_file_path_inference = "models/results_inference.png"
inference_stats = calculate_inference_stats(
    csv_file=csv_file_inference, output_csv_file=csv_file_inference_stats
)
print_image(
    csv_file_path=csv_file_inference_stats, image_file_path=image_file_path_inference
)

print(inference_stats)


df1 = pd.read_csv(csv_file_training_stats)
df2 = pd.read_csv(csv_file_inference_stats)

df1["model"] = df1["model"] + " train"
df2["model"] = df2["model"] + " infer"

# Combine the two DataFrames
combined_df = pd.concat([df1, df2], ignore_index=True)

# Write the combined DataFrame to a new CSV file
output_csv_file = "models/combined_results.csv"
combined_df.to_csv(output_csv_file, index=False)

print_image(
    csv_file_path=output_csv_file,
    image_file_path="models/combined_results.png",
)
