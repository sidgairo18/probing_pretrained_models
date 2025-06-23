#!/usr/bin/python                                                                             
# Author: Siddhartha Gairola (sgairola at mpi-inf dot mpg dot de)                             
                                                                                              
import os                                                                                     
import random                                                                                 
import numpy as np                                                                            
import cv2                                                                                    
                                                                                              
import torch                                                                                  
from torch.utils.data import Dataset                                                          
import torchvision.transforms.functional as F                                                 
import torch.nn.functional as nn_F                                                            
from PIL import Image                                                                         
                                                                                              
class image_loader(Dataset):                                                                  
                                                                                              
    def __init__(self, data_file, data_root, input_transforms=None):           
        self.image_names = []                                                                 
        self.labels = []                                                                      
        f = open(data_file)                                                                   
        for line in f:                                                                        
            line = line.strip().split(' ')                                                    
            self.image_names.append(data_root+line[0])                                        
            self.labels.append(int(line[1]))                                                  
        self.input_transforms = input_transforms                                              
                                                                                              
    def __getitem__(self, index):                                                             
                                                                                              
        image_name = self.image_names[index]                                                  
        label = self.labels[index]                                                            
        with open(image_name, "rb") as f:                                                     
            image = Image.open(f).convert("RGB")                                              
                                                                                              
        if self.input_transforms is not None:                                                 
            image = self.input_transforms(image)                                              
                                                                                              
        return image, label                                                                   
                                                                                              
    def __len__(self):                                                                        
        return len(self.labels)
