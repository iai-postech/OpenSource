import cv2
import albumentations
from PIL import Image
import numpy as np
import random

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from DiffAugment_pytorch import *

def to_gray(image):
    
    # input: gray scale image
    # output: RGB scale image
    
    rgb_image = Image.new("L", image.size)
    rgb_image.paste(image)
    
    return rgb_image

def tensor_to_pil(img):
    return transforms.ToPILImage()(img)

def pil_to_tensor(img):
    return transforms.ToTensor()(img)

''' common data augmentation '''
def random_zoom_in(img):
    return transforms.RandomResizedCrop(size = 256, scale = (0.5, 1.), ratio = (1., 1.))(img)

def random_zoom_out(img):
    img = np.array(img)
    scale = np.random.rand() * 0.5 + 0.3
    zoomed_img = cv2.resize(img, None, fx = scale, fy = scale, interpolation = cv2.INTER_AREA)
    
    top_border = int((256 - zoomed_img.shape[0]) / 2)
    bottom_border = int(256 - zoomed_img.shape[0] - top_border)
    left_border = int((256 - zoomed_img.shape[1]) / 2)
    right_border = int(256 - zoomed_img.shape[1] - left_border)
    
    zoomed_img = cv2.copyMakeBorder(zoomed_img.copy(), top_border, bottom_border, left_border, right_border, cv2.BORDER_CONSTANT, value = 0)
    return Image.fromarray(zoomed_img)


def random_gaussian_blur(img):
    radius = np.random.rand() + 0.5
    return img.filter(ImageFilter.GaussianBlur(radius))

''' experimental data augmentation '''
def random_brightness(img):
    return transforms.ColorJitter(brightness =  (0.75, 1.25))(img)

def random_contrast(img):
    return transforms.ColorJitter(contrast =  (0.75, 2))(img)

''' simulation data augmentation '''
def random_missing_boundary(img):
    n_holes = np.random.randint(8,32)
    max_h_size = np.random.randint(8,32)
    max_w_size = np.random.randint(8,32)
    cutout_img = albumentations.Cutout(num_holes = n_holes, max_h_size = max_h_size, max_w_size = max_w_size, fill_value = 255, p = 1.)(image = np.array(img))['image']
    return Image.fromarray(cutout_img)

def random_precipitation(img):
    img = np.array(img)
    n_precipitation = np.random.randint(10,20)
    for _ in range(n_precipitation):
        center = (int(np.random.rand(1)[0] * img.shape[0]), int(np.random.rand(1)[0] * img.shape[1]))
        axes = (int(np.random.rand(1)[0] * 3 + 1), int(np.random.rand(1)[0] * 2 + 2))
        img = cv2.ellipse(img, center, axes, 0, 0, 360, [0, 0, 0], -1)
    return Image.fromarray(img)

def random_scratch(img):
    img = np.array(img)
    n_scratch = np.random.randint(2,3)
    for _ in range(n_scratch):
        start_angle = np.random.randint(360)
        end_angle = start_angle + 360
        img = cv2.ellipse(img, (int(np.random.rand(1)[0] * img.shape[0]), int(np.random.rand(1)[0] * img.shape[1])), (int(np.random.rand(1)[0] * 60 + 80), int(np.random.rand(1)[0] * 1.0)), start_angle, start_angle, end_angle, [0, 0, 0], -1)
    return Image.fromarray(img)

def random_rand_cutout(x,max,min):
    n_holes = np.random.randint(min,max)
    for i in range(n_holes):
        ratio = np.random.uniform(1/32,1/8)
        x = rand_cutout(x,ratio=ratio)
    return x    

def line_cutout(x):
    mask = torch.ones(x.size(0),x.size(1),x.size(2),x.size(3),device = x.device)
    for i in range(25):
        ratio = 1/8
        cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
        offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
        grid_batch, grid_channel, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(x.size(1), dtype=torch.long, device=x.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
        grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
        # print(x[grid_batch,grid_channel,grid_x,grid_y].le(0.90))
        # print(x[grid_batch,grid_channel,grid_x,grid_y])
        mask[grid_batch,grid_channel,grid_x,grid_y] = x[grid_batch,grid_channel,grid_x,grid_y].le(0.20).float()
    x= x*mask
    return x

def line_le_cutout(x):
    mask = torch.ones(x.size(0),x.size(1),x.size(2),x.size(3),device = x.device)
    for i in range(25):
        ratio = 1/8
        cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
        offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
        grid_batch, grid_channel, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(x.size(1), dtype=torch.long, device=x.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
        grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
        # print(x[grid_batch,grid_channel,grid_x,grid_y].le(0.90))
        # print(x[grid_batch,grid_channel,grid_x,grid_y])
        
        le_threshold = random.uniform(0.20,0.40)
        mask[grid_batch,grid_channel,grid_x,grid_y] = x[grid_batch,grid_channel,grid_x,grid_y].le(le_threshold).float()
    x= x*mask
    return x