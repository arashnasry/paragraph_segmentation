import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image 

import cv2
from matplotlib import pyplot as plt
from numpy import random
import os
###############################################################################3


th=180
kx=13
ky=7
itr=3
area_thresh=10000
resize=(2500,2500)
intersection_thresh = .5
input_folder = 'new folder'
output_folder = 'output'

class HotDogClassifier(nn.Module):
    def __init__(self):
        super(HotDogClassifier, self).__init__()
        self.block1 = self.conv_block(c_in=1, c_out=256, dropout=0.1, kernel_size=5, stride=1, padding=2)
        self.block2 = self.conv_block(c_in=256, c_out=128, dropout=0.1, kernel_size=3, stride=1, padding=1)
        self.block3 = self.conv_block(c_in=128, c_out=64, dropout=0.1, kernel_size=3, stride=1, padding=1)
        self.lastcnn = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=56, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        x = self.block1(x)
        x = self.maxpool(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.maxpool(x)
        x = self.lastcnn(x)
        return x
    def conv_block(self, c_in, c_out, dropout,  **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )
        return seq_block
os.makedirs(output_folder,exist_ok=True)
load_cnn=torch.load(r"C:\Users\Batman\Desktop\New folder (3)\arash.pth",map_location='cpu')

#%%
images_list = os.listdir(input_folder)
images_list = [item for item in images_list if '.png' in item or '.jpg' in item or '.jpeg' in item]
for image_path in images_list:
    image_name = image_path.split('.')[0]
    img = cv2.imread(os.path.join(input_folder,image_path), 0)
    plane  =  img.copy()
    dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    result = 255 - cv2.absdiff(plane, bg_img)
    result_norm = cv2.normalize(result,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    
    #%%
    
    
    image=cv2.resize(result_norm,(resize))
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
        gray =image.copy()
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY_INV )[1]
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx,ky))
    dilate = cv2.dilate(thresh, kernel, iterations=itr)
    
    #%%
    
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts=[]
    for c in contours:
        area = cv2.contourArea(c)
        if area>area_thresh:
            cnts.append(c)
    #%%
    angles = []
    for contour in cnts:
        rect = cv2.minAreaRect(contour)
        angle = rect[-1]
        if angle < -45:
            angle = 90 + angle
        else:
        	angle = angle
        print(angle)
        angles.append(angle)
    
    res = plt.hist(angles)
    rotate_angle = (res[1][res[0].argmax()] + res[1][res[0].argmax()+1])/2
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
    	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    #%%
    try:
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    except:
        gray =rotated.copy()
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY_INV )[1]
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx,ky))
    dilate = cv2.dilate(thresh, kernel, iterations=itr)
    
    
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #%%
    image_transforms = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]),
        "test": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    #%%
    cnts=[]

    for c in contours:
        area = cv2.contourArea(c)
        x,y,w,h = cv2.boundingRect(c)
     

        if area>area_thresh:
            if  area > intersection_thresh*h*w:
                cnts.append(c)
                
   
   

    final=[]
    tasvir=[]
    for i,main_contour in enumerate(cnts):
        canvas = rotated.copy()
        for j,rest_contour in enumerate(cnts):
            if i == j:
                continue
            else:
                cv2.drawContours(canvas, cnts, j, color=(255,255,255), thickness=-1)
        x,y,w,h = cv2.boundingRect(main_contour)
        ROIx = canvas[y:y+h, x:x+w]

        im_pil = Image.fromarray(ROIx).convert('RGB')
        im=image_transforms['test'](im_pil).unsqueeze(0)
        example=im.to(device)
        pred=load_cnn(example).argmax().squeeze()
        if pred==1:
                final.append(main_contour)

                print(pred)
        if pred==0:
                tasvir.append(main_contour)

                print(pred)

    save_path = os.path.join(output_folder,f'{image_name}_{i:04d}.jpg')
    cv2.drawContours(rotated, final, -1, color=(0,0,255), thickness=2)
    cv2.imwrite(f'{image_name}.jpg',rotated)
    


