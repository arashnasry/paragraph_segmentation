import cv2
import numpy as np
from matplotlib import pyplot as plt
th=160
kx=4
ky=14
itr=3
resize=(2000,2000)
# Load image, grayscale, Gaussian blur, Otsu's threshold
image = cv2.imread('book_inside_35.jpg')
image=cv2.resize(image,(resize))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7,7), 0)
thresh = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY_INV )[1]

# Create rectangular structuring element and dilate
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kx,ky))
dilate = cv2.dilate(thresh, kernel, iterations=itr)

# Find contours and draw rectangle
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
z,contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts=[]
for c in contours:
    area = cv2.contourArea(c)
    if area>10000:
        cnts.append(c)
# print(len(cnts[0]))
# for c in cnts:
#     x,y,w,h = cv2.boundingRect(c)
#     #cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
#     pts = np.array([[x,y],[x+w,y],[x,y+h],[x+w,y+h]], np.int32)
#     pts = pts.reshape((-1,1,2))
#     img = cv2.polylines(image,[pts],True,(0,255,255))

cv2.drawContours(image, cnts, -1, (0,255,0), 3)

cv2.imshow('thresh', thresh)
cv2.imshow('dilate', dilate)
cv2.imshow('image', image)
cv2.waitKey()
plt.imshow(image, 'gray'),plt.show()