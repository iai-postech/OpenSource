import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
import albumentations
import cv2
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from datadiversity import *

class LambdaLR:
    
    # input: current epoch
    # output: weight of learning rate decay
    
    def __init__(self, n_epoch, decay_start_epoch):
        self.n_epoch = n_epoch
        self.decay_start_epoch = decay_start_epoch
        
    def step(self, epoch):
        return 1.0 - max(0, epoch - self.decay_start_epoch) / (self.n_epoch - self.decay_start_epoch)

        
class ReplayBuffer:
    
    # input: image
    # outpu: buffer class of historical images
    
    def __init__(self, max_size = 50):
        self.max_size = max_size
        self.data = []
        
    def push_and_pop(self, data):
        to_return = []
        
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
                
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        
        return torch.cat(to_return)
    
def weights_init_normal(m):
    
    # input: layer
    # output: layer's weight and bias initialization
    
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
 
class LambdaLR:
    
    # input: current epoch
    # output: weight of learning rate decay
    
    def __init__(self, n_epoch, decay_start_epoch):
        self.n_epoch = n_epoch
        self.decay_start_epoch = decay_start_epoch
        
    def step(self, epoch):
        return 1.0 - max(0, epoch - self.decay_start_epoch) / (self.n_epoch - self.decay_start_epoch)
     

class ResidualBlock(nn.Module):
    
    # input: x
    # output: skip-connected x
    
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(in_channels, in_channels, kernel_size = 3),
                                   nn.InstanceNorm2d(in_channels),
                                   nn.ReLU(inplace = True),
                                   nn.ReflectionPad2d(1),
                                   nn.Conv2d(in_channels, in_channels, kernel_size = 3),
                                   nn.InstanceNorm2d(in_channels))
        
    def forward(self, x):
        return x + self.block(x)
    
class DiscriminatorBlock(nn.Module):
    
    # input: x
    # output: processed x
    
    def __init__(self, in_channels, out_channels, normalize):
        super(DiscriminatorBlock, self).__init__()
        
        if normalize:
            self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1),
                                       nn.InstanceNorm2d(out_channels),
                                       nn.LeakyReLU(0.2, inplace = True))
        else:
            self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1),
                                       nn.LeakyReLU(0.2, inplace = True))
    
    def forward(self, x):
        return self.block(x)
    
class GeneratorResNet(nn.Module):
    
    # input: image
    # output: translated image
    
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()
        
        out_channels = 64
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_shape[0], out_channels, kernel_size = 7),
                 nn.InstanceNorm2d(out_channels),
                 nn.ReLU(inplace = True)]
        in_channels = out_channels
        
        # Down-Sampling
        for _ in range(2):
            out_channels *= 2
            model += [nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1),
                      nn.InstanceNorm2d(out_channels),
                      nn.ReLU(inplace = True)]
            in_channels = out_channels
            
        # Bottle-neck
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_channels)]
        
        # Up-Sampling
        for _ in range(2):
            out_channels //= 2
            model += [nn.Upsample(scale_factor = 2),
                      nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                      nn.InstanceNorm2d(out_channels),
                      nn.ReLU(inplace = True)]
            in_channels = out_channels
        
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(out_channels, input_shape[0], kernel_size = 7),
                  nn.Sigmoid()]
        
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        return self.model(x)
    
class Discriminator(nn.Module):
    
    # input: translated image
    # output: discrimination of real or fake
    
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        
        channels, height, width = input_shape
        
        model = [DiscriminatorBlock(channels, 64, normalize = False),
                 DiscriminatorBlock(64, 128, normalize = True),
                 DiscriminatorBlock(128, 256, normalize = True),
                 DiscriminatorBlock(256, 512, normalize = True)]
        
        model += [nn.ZeroPad2d((1, 0, 1, 0)),
                 nn.Conv2d(512, 1, kernel_size = 4, padding = 1)]
        
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        return self.model(x)