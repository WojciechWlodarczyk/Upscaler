import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from torch.utils.data import Subset, Dataset, DataLoader
from pathlib import Path

from Model_17.Model_17 import CNNUpscaler, ResidualBlock
# from Model_14.Model_14 import RRDBNet, SEBlock, DenseResidualBlock, RRDB, Discriminator, VGGFeatureExtractor
import TrainModel
import TestModel

model_name = 'Model_17'

if __name__ == "__main__":
    print(__name__)
    model = CNNUpscaler()
    TrainModel.train(model, model_name, 1)

def RerunTraining(modelName, best_test_loss, final_epoch):
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)

    print('RerunTraining ' + modelName)
    model = torch.load(modelName, weights_only=False)
    TrainModel.train(model, model_name, 1, final_epoch + 1, best_test_loss, True)
