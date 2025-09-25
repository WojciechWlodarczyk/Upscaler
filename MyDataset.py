import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from torch.utils.data import Subset, Dataset, DataLoader


input_dir_train = 'D:/mojeAI/MyUpscalerDataSet/half_FHD'
target_dir_train = 'D:/mojeAI/MyUpscalerDataSet/FHD'

input_dir_test = 'D:/mojeAI/MyUpscalerDataSet/half_FHD_test'
target_dir_test = 'D:/mojeAI/MyUpscalerDataSet/FHD_test'

input_dir_final_test = 'D:/mojeAI/MyUpscalerDataSet/half_FHD_final_test'
target_dir_final_test = 'D:/mojeAI/MyUpscalerDataSet/FHD_final_test'

class ImagePairDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)])
        self.target_files = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir)])
        self.input_transform = transforms.Compose([
            transforms.Resize((540, 960)),
            transforms.ToTensor()
        ])
        self.target_transform = transforms.Compose([
            transforms.Resize((1080, 1920)),
            transforms.ToTensor()
        ])

    def __len__(self):
    #    print("Długość datasetu:", len(self.input_files), len(self.target_files))
        return len(self.input_files)

    def __getitem__(self, idx):
        input_img = Image.open(self.input_files[idx]).convert('RGB')
        target_img = Image.open(self.target_files[idx]).convert('RGB')
        input_tensor = self.input_transform(input_img)
        target_tensor = self.target_transform(target_img)
        return input_tensor, target_tensor


def GetTrainDataset():
    return ImagePairDataset(input_dir_train, target_dir_train)

def GetTestDataset():
    return ImagePairDataset(input_dir_test, target_dir_test)

def GetFinalTestDataset():
    return ImagePairDataset(input_dir_final_test, target_dir_final_test)

