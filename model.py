import pandas as pd 
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn as nn
from tqdm import tqdm
from torchvision import utils,datasets
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts
from torchsummary import summary
import logging
import os 

class SCNNB(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
                                        nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2)
                                    )
        
        self.layer2 = nn.Sequential(
                                        nn.Conv2d(in_channels=32, out_channels=64,kernel_size=3),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2)
                                    )
        self.flt = nn.Flatten()
        self.FC = nn.Sequential    (    nn.Linear(1600,1280),
                                        nn.ReLU(),
                                        nn.Dropout(p = 0.5), 
                                        nn.Linear(1280,10),
                                        nn.Softmax(dim = 1)
                                   )

        self.initialize_weights() 

    def initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                # Apply Xavier initialization to Conv2d layers
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.constant_(layer.bias, 0.0)                              
        
    def  forward(self,x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flt(x)
        x = self.FC(x)
        return x 
