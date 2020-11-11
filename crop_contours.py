import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import random
import os
th=180
kx=13
ky=7
itr=3
area_thresh=10000
resize=(2500,2500)
intersection_thresh = .5
input_folder = 'images'
output_folder = 'output'
os.makedirs(output_folder,exist_ok=True)
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
    # cv2.namedWindow('canvas',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('canvas', 500, 600)
    angles = []
    # canvas = thresh.copy()
    for contour in cnts:
        rect = cv2.minAreaRect(contour)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # canvas = cv2.drawContours(canvas,[box],0,255,2)
        
        angle = rect[-1]
        if angle < -45:
            angle = 90 + angle
        else:
        	angle = angle
        print(angle)
        angles.append(angle)
    # cv2.imshow('canvas',canvas)
    # cv2.waitKey()    
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
    
    #%%
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts=[]
    
    for c in contours:
        area = cv2.contourArea(c)
        x,y,w,h = cv2.boundingRect(c)
        
        if area>area_thresh:
            if area > intersection_thresh*h*w:
                cnts.append(c)
                
    cnts_draw = cv2.drawContours(rotated.copy(),cnts,-1,80,2)
    cv2.imwrite(f'{image_name}.jpg',cnts_draw)
    #%%
    for i,main_contour in enumerate(cnts):
        canvas = rotated.copy()
        for j,rest_contour in enumerate(cnts):
            if i == j:
                continue
            else:
                cv2.drawContours(canvas, cnts, j, color=(255,255,255), thickness=-1)
        x,y,w,h = cv2.boundingRect(main_contour)
        ROIx = canvas[y:y+h, x:x+w]
        save_path = os.path.join(output_folder,f'{image_name}_{i:04d}.jpg')
        cv2.imwrite(save_path,ROIx)




