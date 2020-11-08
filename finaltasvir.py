import cv2
import numpy as np
from matplotlib import pyplot as plt
th=170
kx=15
ky=15
itr=4
resize=(2500,2500)
# Load image, grayscale, Gaussian blur, Otsu's threshold
#image = cv2.imread('book_inside_16.jpg')

image = cv2.imread('book_inside_20.jpg',0)
dilated_img = cv2.dilate(image, np.ones((7,7), np.uint8)) 
bg_img = cv2.medianBlur(dilated_img, 21)
diff_img = 255 - cv2.absdiff(image, bg_img)
norm_img = diff_img.copy() # Needed for 3.x compatibility
cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
_, thr_img = cv2.threshold(norm_img, 230, 0, cv2.THRESH_TRUNC)
xx=cv2.normalize(thr_img, thr_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
#plt.imshow(xx, 'gray'),plt.show()



image=cv2.resize(image,(resize))
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(image, (5,5), 0)
thresh = cv2.threshold(blur, th, 255, cv2.THRESH_BINARY_INV )[1]

# Create rectangular structuring element and dilate
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx,ky))
#dilate = cv2.dilate(thresh, kernel, iterations=itr)

# Find contours and draw rectangle
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
z,contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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

a=cv2.drawContours(image, cnts, -1, (0,255,0), 3)

# cv2.imshow('thresh', thresh)
# cv2.imshow('dilate', dilate)
# cv2.imshow('image', image)
#cv2.imwrite('3.jpg',aa)
cv2.waitKey()
plt.imshow(image, 'gray'),plt.show()