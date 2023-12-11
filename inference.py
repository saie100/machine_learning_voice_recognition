import os
import torch
import pandas as pd
from ImageDS import ImageDS
from AudioUtil import AudioUtil
from train_model import AudioClassifier
from audio_processing import processing
  
# ----------------------------
# Inference
# ----------------------------
def inference (model, val_dl):
  correct_prediction = 0
  total_prediction = 0

  label_to_name = {0 : 'Dan', 1 : 'David', 2 : 'Derek'}

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
      prediction_l = prediction.numpy()[0]
      model_l = labels.numpy()[0]
      #print(prediction_l)
      #print(model_l)

      print(f"\n\nMODEL PREDICTS: {label_to_name[prediction_l]}")
      print(f"LABEL MARKERD AS: {label_to_name[model_l]}\n\n")

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
  metadata_file = 'testing_metadata.csv'

  df = pd.read_csv(metadata_file)
  df.head()

  # Take relevant columns
  df = df[['relative_path', 'classID']]
  df.head()

  # process the raw data and place it in images directory
  processing([metadata_file])
  
  df['relative_path'] = df['relative_path'].str.replace('.wav', '.wav.png', regex=False)
  #df['relative_path'] = df['relative_path'].str.replace('.flac', '.flav.png', regex=False)

  current_directory = os.getcwd() + "/images/"
  myds = ImageDS(df, current_directory)

  # Create validation data loaders and load model
  val_dl = torch.utils.data.DataLoader(myds, batch_size=1, shuffle=True)
  PATH = "machine_learning_model.pth"
  model = torch.load(PATH)

  # Run inference on trained model with the validation set
  inference(model, val_dl)



if __name__ == '__main__':
   main()