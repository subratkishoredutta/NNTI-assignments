# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 15:35:18 2023

@author: Asus
"""

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics import F1Score
import matplotlib.pyplot as plt


class Model(nn.Module):
    def __init__(self, input_dim, channels, kernel_size,out_dim,num_layers):
        super(Model, self).__init__() 
        self.input_dim = input_dim
        self.channels = channels
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.k_size = kernel_size 
        layers=[]
        layers.append(nn.Conv2d(in_channels=self.input_dim, out_channels=self.channels,kernel_size=self.k_size,padding='same',stride=1,bias=True))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
        for num in range(self.num_layers):
            layers.append(nn.Conv2d(in_channels=self.channels, out_channels=self.channels,kernel_size=self.k_size,padding='same',stride=1,bias=True))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(self.channels*int(32/(2**(self.num_layers+1)))**2,350,bias=True))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(350,self.out_dim,bias=True))
        layers.append(nn.Softmax())
        self.layers = nn.Sequential(*layers)
    def __call__(self,x):
        out=self.layers(x)
        return out
        
def datagen(split,batch_size):
    transform_fn = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.7,), (0.7,)),])

    svhn_train = datasets.SVHN(root='./data', split=split, download=True, transform=transform_fn)
    train_dl = torch.utils.data.DataLoader(svhn_train, batch_size=batch_size, shuffle=True)
    return train_dl
