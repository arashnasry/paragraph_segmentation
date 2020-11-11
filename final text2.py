import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import random

th=180
kx=13
ky=16
itr=3
resize=(2500,2500)
########################################################################################################

img = cv2.imread('003.jpg', -1)

rgb_planes = cv2.split(img)

result_planes = []
result_norm_planes = []
for plane in rgb_planes:
    dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(plane, bg_img)
    norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    result_planes.append(diff_img)
    result_norm_planes.append(norm_img)

result = cv2.merge(result_planes)
result_norm = cv2.merge(result_norm_planes)


########################################################################################################


image=cv2.resize(result_norm,(resize))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3,3), 0)
thresh = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY_INV )[1]

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx,ky))
dilate = cv2.dilate(thresh, kernel, iterations=itr)

########################################################################################################

contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts=[]
for c in contours:
    area = cv2.contourArea(c)
    if area>10000:
        cnts.append(c)
########################################################################################################
for a in cnts:
    
    for c in cnts:
        if cv2.contourArea(c) == cv2.contourArea(a):
            pass
        else:
            x,y,w,h = cv2.boundingRect(c)
            ROI = image[y:y+h, x:x+w]
            ROI[:] = (255, 255, 255)
    x,y,w,h = cv2.boundingRect(a)
    ROIx = image[y:y+h, x:x+w]
    x1 = random.randint(10000)
    cv2.imwrite(f'aa{x1}aa.jpg',ROIx)

    
########################################################################################################

#aa=cv2.drawContours(image, cnts, -1, (0,255,0), 3)

# cv2.imshow('thresh', thresh)
# cv2.imshow('dilate', dilate)
# cv2.imshow('image', image)
# #cv2.imwrite('3.jpg',aa)
# cv2.waitKey()
plt.imshow(image, 'gray'),plt.show()
########################################################################################################
