import os
import torch
import pandas as pd
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn import init
from ImageDS import ImageDS
from audio_processing import processing
from audio_processing import plot_spectrogram
# importing OpenCV(cv2) module
import cv2

# Read metadata file
processing_metadata = 'training_metadata.csv'
df = pd.read_csv(processing_metadata)
df.head()

# Take relevant columns
df = df[['relative_path', 'classID']]
df.head()
df['relative_path'] = df['relative_path'].str.replace('.wav', '.wav.png', regex=False)

current_directory = os.getcwd() + "/images/"
myds = ImageDS(df, current_directory)
# Create training data loaders
train_dl = torch.utils.data.DataLoader(myds, batch_size=1, shuffle=True)

image, label = next(iter(train_dl))
print(image.shape)
#print(image)

conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5, stride=1)
pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
conv3 = nn.Conv2d(in_channels=12, out_channels=6, kernel_size=5, stride=1)

fc1 = nn.Linear(6 * 245 * 94, 120)
fc2 = nn.Linear(120, 128)
fc3 = nn.Linear(128, 3)

x = conv1(image)
print(x.shape)
x = pool1(x)
print(x.shape)
x = conv2(x)
print(x.shape)
x = pool2(x)
print(x.shape)
x = conv3(x)
print(x.shape)

#x = x.view(x.shape[0], -1)
x = x.view(-1, x.shape[0])
print(x.shape)
