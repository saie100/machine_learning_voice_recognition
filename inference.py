import os
import torch
import pandas as pd
from SoundDS import SoundDS
from AudioUtil import AudioUtil
from train_model import AudioClassifier
from audio_processing import processing

  
# ----------------------------
# Inference
# ----------------------------
def inference (model, val_dl):
  correct_prediction = 0
  total_prediction = 0

  # Disable gradient updates
  with torch.no_grad():
    for data in val_dl:
      # Get the input features and target labels, and put them on the GPU
      inputs, labels = data[0].to(device), data[1].to(device)

      # Normalize the inputs
      inputs_m, inputs_s = inputs.mean(), inputs.std()
      inputs = (inputs - inputs_m) / inputs_s

      # Get predictions
      outputs = model(inputs)

      # Get the predicted class with the highest score
      _, prediction = torch.max(outputs,1)
      # Count of predictions that matched the target label
      correct_prediction += (prediction == labels).sum().item()
      total_prediction += prediction.shape[0]
    
  acc = correct_prediction/total_prediction
  print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')





def main():
  # ----------------------------
  # Prepare inference data from Metadata file
  # ----------------------------

  global device
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
  # Read metadata file
  metadata_file = 'metadata.csv'
  df = pd.read_csv(metadata_file)
  df.head()

  # Take relevant columns
  df = df[['relative_path', 'classID']]
  df.head()

  # process the raw data and place it in processed_audio directory
  processing([metadata_file])

  current_directory = os.getcwd() + "/processed_audio/"
  myds = SoundDS(df, current_directory)

  # Create validation data loaders and load model
  val_dl = torch.utils.data.DataLoader(myds, batch_size=4, shuffle=True)
  PATH = "machine_learning_model.pth"
  model = torch.load(PATH)

  # Run inference on trained model with the validation set
  inference(model, val_dl)



if __name__ == '__main__':
   main()