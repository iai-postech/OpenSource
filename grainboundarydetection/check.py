import torch
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

#import for module
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
import albumentations
import cv2
from glob import glob
from PIL import Image

#import for trainig
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

#import for own module
from datadiversity import *
from DiffAugment_pytorch import *
from consistencyimage import *
from cyclegan import *


class ImageDataset(Dataset):
    
    # input: image files
    # output: dataset class of transformed images
    
    def __init__(self, Name, transforms_ = None):
        self.transform = transforms_
        self.Name = Name
        self.domain_A_file = "/mnt/disk1/jaejun/grain_boundary_detection/data_files/trainA/magnesium_169.jpg"
        self.domain_B_file = glob('/mnt/disk2/thresh/thresh/*.*')
        
    def __getitem__(self, index):
        domain_A = Image.open(self.domain_A_file)
        domain_B = Image.open(self.domain_B_file[random.randint(0, len(self.domain_B_file) - 1)])
        
        if domain_A.mode == "RGB":
            domain_A = to_gray(domain_A)
        if domain_B.mode == "RGB":
            domain_B = to_gray(domain_B)
        
        domain_A = self.transform(domain_A)
        domain_B = self.transform(domain_B)
        
        return {"A": domain_A, "B": domain_B}
        
    def __len__(self):
        return max(len(self.domain_A_file), len(self.domain_B_file))
    
transforms_ = transforms.Compose([
    transforms.RandomCrop((256, 256)),
    transforms.RandomResizedCrop(size = 256, scale = (0.75, 1.), ratio = (0.75, 1.33)),
    transforms.ToTensor()
])

train_dataset = ImageDataset(Name = "magnesium_169.jpg",transforms_ = transforms_)
val_dataset = ImageDataset(Name = "magnesium_169.jpg",transforms_ = transforms_)


# build dataloader class
train_dataloader = DataLoader(train_dataset, batch_size = 1, shuffle = True, num_workers = 0)
val_dataloader = DataLoader(val_dataset, batch_size = 1, shuffle = True, num_workers = 0)


for batch in val_dataloader:
    real_A_ = batch["A"].cuda()
    real_B_ = batch["B"].cuda()
    # real_A = make_grid(real_A_, nrow= 2, normalize=False)
    # real_B = make_grid(real_B_, nrow = 2, normalize=False)
    # img_grid = torch.cat((real_A,real_B),2)
    # img_grid = img_grid.cpu()
    break

plt.imshow(real_B_.squeeze(0).squeeze(0).detach().cpu().numpy())
plt.show()
# missing_B = random_rand_cutout(real_B_,min=8,max=32)
# torch_image = random_rand_cutout(real_B_,min=8,max=32)
# pil_image = tensor_to_pil(torch_image[0])
# plt.imshow(pil_image,"gray")
# plt.show()
# line_missing_B = line_cutout(real_B_)
# pil_image = tensor_to_pil(line_missing_B[0])
# plt.imshow(pil_image,"gray")
# plt.show()