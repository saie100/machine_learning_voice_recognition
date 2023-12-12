import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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

    # Define colors for 'train' and 'infer'
    colors = {"train": "skyblue", "infer": "lightcoral"}

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))

    # Unique models for plotting
    # models = df["model"].unique()
    # types = df["type"].unique()
    models = ["relu", "tanh", "sigmoid"]
    types = ["train", "infer"]
    # Width of a bar
    bar_width = 0.35
    # Iterate over models
    for i, model in enumerate(models):
        for j, t in enumerate(types):
            # Filter data for each model and type
            data = df[(df["model"] == model) & (df["type"] == t)]

            # Calculate mean accuracy and standard deviation
            mean_accuracy = data["mean_accuracy"].mean()
            std_accuracy = data["std_accuracy"].mean()

            # Position of the bar
            pos = i - bar_width / 2 + j * bar_width

            # Plot the bar
            bar = ax.bar(
                pos,
                mean_accuracy,
                yerr=std_accuracy,
                width=bar_width,
                capsize=5,
                color=colors[t],
                label=t if i == 0 and j == 0 else "",
            )
            # Add value label at the bottom of the bar
            ax.text(
                pos,
                0.8,  # Adjust this value as needed
                f"{round(mean_accuracy, 3)}",
                va="bottom",
                ha="center",
                fontsize=8,
                color="black",  # Optional: Change text color if needed
            )

    # Set labels and title
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models)
    ax.set_title("Mean Accuracy with Standard Deviation by Model and Type")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.8, 1.05)

    # Add legend
    ax.legend(["train", "infer"], loc="upper right")

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
df1 = pd.read_csv(csv_file_training_stats)
df1["type"] = "train"

# print_image(
#     csv_file_path=csv_file_training_stats, image_file_path=image_file_training_path
# )

print(training_stats)

print("\nINFERENCE")
# Replace 'your_file_path.csv' with the path to your CSV file
csv_file_inference = "models/results_inference.csv"
csv_file_inference_stats = "models/results_inference_stats.csv"
image_file_path_inference = "models/results_inference.png"
inference_stats = calculate_inference_stats(
    csv_file=csv_file_inference, output_csv_file=csv_file_inference_stats
)
df2 = pd.read_csv(csv_file_inference_stats)
df2["type"] = "infer"
# print_image(
#     csv_file_path=csv_file_inference_stats, image_file_path=image_file_path_inference
# )

print(inference_stats)

# Combine the two DataFrames
combined_df = pd.concat([df1, df2], ignore_index=True)

# # Write the combined DataFrame to a new CSV file
output_csv_file = "models/combined_results.csv"
combined_df.to_csv(output_csv_file, index=False)

print_image(
    csv_file_path=output_csv_file,
    image_file_path="models/combined_results.png",
)
