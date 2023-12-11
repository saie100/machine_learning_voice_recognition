import os
import torch
import pandas as pd
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn import init
from ImageDS import ImageDS
from audio_processing import processing

# ----------------------------
# Audio Classification Model
# ----------------------------
class AudioClassifier(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()

        # Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=(2, 2), padding=(2, 2))
        self.bn1 = nn.BatchNorm2d(8)
        
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=(2, 2), padding=(2, 2))
        self.bn2 = nn.BatchNorm2d(16)
        
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=(2, 2), padding=(2, 2))
        self.bn3 = nn.BatchNorm2d(32)
        
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=(2, 2), padding=(2, 2))                
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1)
        self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=1)
        
        # Linear Layer
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=16, out_features=3)


    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional layer
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        x = F.relu(self.conv5(x))
        x = self.mpool1(x)
        x = F.relu(self.conv6(x))

        # flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x) 

        # Final output
        return x


# ----------------------------
# Training Loop
# ----------------------------
def training(model, train_dl, num_epochs, learning_rate=0.001):
  # Loss Function, Optimizer and Scheduler
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate * 1.5,
                                                steps_per_epoch=int(len(train_dl)),
                                                epochs=num_epochs,
                                                anneal_strategy='linear')

  # Repeat for each epoch
  for epoch in range(num_epochs):
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0

    # Repeat for each batch in the training set
    for i, data in enumerate(train_dl):
        # Get the input features and target labels, and put them on the GPU
        inputs, labels = data[0].to(device), data[1].to(device)

        # Normalize the inputs
        inputs_m, inputs_s = inputs.mean(), inputs.std()
        inputs = (inputs - inputs_m) / inputs_s
        #print(inputs.shape)
        # Zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Keep stats for Loss and Accuracy
        running_loss += loss.item()

        # Get the predicted class with the highest score
        _, prediction = torch.max(outputs,1)
        # Count of predictions that matched the target label
        correct_prediction += (prediction == labels).sum().item()
        total_prediction += prediction.shape[0]

        #if i % 10 == 0:    # print every 10 mini-batches
        #    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
    
    # Print stats at the end of the epoch
    num_batches = len(train_dl)
    avg_loss = running_loss / num_batches
    acc = correct_prediction/total_prediction
    print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')

  print('Finished Training')
  

def main():
  # ----------------------------
  # Prepare training data from Metadata file
  # ----------------------------

  # Read metadata file
  metadata_file = 'training_metadata.csv'
  df = pd.read_csv(metadata_file)
  df.head()

  # Take relevant columns
  df = df[['relative_path', 'classID']]
  df.head()

  # process the raw data and place it in images directory
  processing([metadata_file], True)

  df['relative_path'] = df['relative_path'].str.replace('.wav', '.wav.png', regex=False)
  #df['relative_path'] = df['relative_path'].str.replace('.flac', '.flav.png', regex=False) do not use .flac files only use .wav

  current_directory = os.getcwd() + "/images/"
  myds = ImageDS(df, current_directory)

  # Create training data loaders
  train_dl = torch.utils.data.DataLoader(myds, batch_size=16, shuffle=True)
  
  # Create the model and put it on the GPU if available
  myModel = AudioClassifier()
  global device
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  myModel = myModel.to(device)
  # Check that it is on Cuda
  next(myModel.parameters()).device

  num_epochs=13   # increase num of epochs until there isn't much change in validation loss
  learning_rate = .001

  training(myModel, train_dl, num_epochs, learning_rate)

  # save model 
  PATH = "machine_learning_model.pth"
  torch.save(myModel, PATH)


if __name__ == '__main__':
   main()
