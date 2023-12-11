import os
import sys
import torch
import wave
import pyaudio
import pandas as pd
from ImageDS import ImageDS
from AudioUtil import AudioUtil
from torch.utils.data import random_split
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
      
      print(f"\n\nMODEL PREDICTS: {prediction}")
      print(f"LABEL MARKERD AS: {labels}\n\n")

      # Count of predictions that matched the target label
      correct_prediction += (prediction == labels).sum().item()
      total_prediction += prediction.shape[0]
    
  acc = correct_prediction/total_prediction
  print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')


def getLiveRecording(num_of_runs=1):
    #CHUNK = 1024
    CHUNK = 48000 * 3
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 48000 
    RECORD_SECONDS = 3

    p = pyaudio.PyAudio()

    ## This code block to print out your audio device index and info
    ## use audio device index as the input_device_index argument for p.open method
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    print(info)
    print(numdevices)
    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))

    input_ = input(f"Enter \'y\' to begin\n")
    while(input_ != "y"):
        if(input_ == "n" or input_ == "N"):
            print("ending program...")
            sys.exit()

        print(f"You entered \'{input_}\'")
        input_ = input(f"Enter \'y\' to begin or enter \'n\' to end program\n")

    ## This code block starts the recording of audio
    for index in range(num_of_runs):
        WAVE_OUTPUT_FILENAME = f"live_voice/output{index}.wav"
        
        stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=6)
        
        print(f"* recording number {index}")
        frames = []

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        print(f"* done recording {index}")
        stream.stop_stream()
        stream.close()

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

    p.terminate()




def main():
    # ----------------------------
    # Prepare inference data from Metadata file
    # ----------------------------

    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Read metadata file
    metadata_file = 'live_metadata.csv'
    df = pd.read_csv(metadata_file)
    df.head()

    # Take relevant columns
    df = df[['relative_path', 'classID']]
    df.head()

    
    current_directory = os.getcwd()

    try:
        os.remove(f"{current_directory}/processed_audio/{df['relative_path'][0]}")  
    except:
        print(f"File do not exist: {current_directory}/processed_audio/{df['relative_path'][0]}")
    try:
        os.remove(f"{current_directory}/{df['relative_path'][0]}")  
    except:
        print(f"File do not exist: {current_directory}/{df['relative_path'][0]}")
    
    df['relative_path'] = df['relative_path'].str.replace('.wav', '.wav.png', regex=False)
    df['relative_path'] = df['relative_path'].str.replace('.flac', '.flav.png', regex=False)

    try:
        os.remove(f"{current_directory}/images/{df['relative_path'][0]}")
    except:
        print(f"File do not exist: {current_directory}/images/{df['relative_path'][0]}")

    current_directory = os.getcwd() + "/images/"
    myds = ImageDS(df, current_directory)

    # get live recording from user
    getLiveRecording(num_of_runs=1)
    
    # process the raw data and place it in processed_audio directory
    processing([metadata_file])

    # Create validation data loaders and load model
    val_dl = torch.utils.data.DataLoader(myds, batch_size=1, shuffle=False)
    PATH = "machine_learning_model.pth"
    model = torch.load(PATH)

    # Run inference on trained model with the validation set
    inference(model, val_dl)



if __name__ == '__main__':
    main()