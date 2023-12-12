from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from AudioUtil import AudioUtil
import torchaudio
import random

# importing OpenCV(cv2) module
import cv2


# ----------------------------
# Image Dataset
# ----------------------------
class ImageDS(Dataset):
    def __init__(self, df, data_path):
        self.df = df
        self.data_path = str(data_path)

    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return len(self.df)

    # ----------------------------
    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, idx):
        # Absolute file path of the audio file - concatenate the audio directory with
        # the relative path
        image_file = self.data_path + self.df.loc[idx, "relative_path"]
        # Get the Class ID
        class_id = self.df.loc[idx, "classID"]
        # print(image_file)
        img = cv2.imread(image_file)

        transform = transforms.Compose([transforms.ToTensor()])

        img_tensor = transform(img)
        return img_tensor, class_id
