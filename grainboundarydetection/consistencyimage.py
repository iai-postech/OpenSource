import cv2
import albumentations
from datadiversity import tensor_to_pil,pil_to_tensor
from datadiversity import random_zoom_out,random_zoom_in
from datadiversity import random_gaussian_blur,random_brightness
from datadiversity import random_contrast,random_missing_boundary
from datadiversity import random_precipitation
from datadiversity import random_scratch

'''
for consistency regularization 
｜D(CR(x))-D(X)｜

'''
def transforms_zoom_out(img):
    
    img = tensor_to_pil(img)
    
    img = random_zoom_out(img)

    img = pil_to_tensor(img)
    
    return img

def transforms_zoom_in(img):
    
    img = tensor_to_pil(img)
    
    img = random_zoom_in(img)

    img = pil_to_tensor(img)
    
    return img

def transforms_gaussian_blur(img):
    
    img = tensor_to_pil(img)
    
    img = random_gaussian_blur(img)

    img = pil_to_tensor(img)
    
    return img

def transforms_brightness(img):
    
    img = tensor_to_pil(img)
    
    img = random_brightness(img)

    img = pil_to_tensor(img)
    
    return img

def transforms_contrast(img):
    
    img = tensor_to_pil(img)
    
    img = random_contrast(img)

    img = pil_to_tensor(img)
    
    return img

def transforms_missing_boundary(img):

    img = tensor_to_pil(img)
    
    img = random_missing_boundary(img)
    
    img = pil_to_tensor(img)
    
    return img

def transforms_precipitation(img):

    img = tensor_to_pil(img)
    
    img = random_precipitation(img)

    img = pil_to_tensor(img)
    
    return img

def transforms_scratch(img):

    img = tensor_to_pil(img)
    
    img = random_scratch(img)

    img = pil_to_tensor(img)
    
    return img