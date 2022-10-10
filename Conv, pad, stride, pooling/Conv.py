# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 08:05:28 2022

@author: Nguyễn Vũ Hải
"""

import cv2
import numpy as np
import imutils

image = cv2.imread('HaiImage.jpg')
image = imutils.resize(image, 400)
# Print error message if image is null
if image is None:
    print('Could not read image')

# Apply identity kernel
kernel1 = np.array([[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]])

identity = cv2.filter2D(src=image, ddepth=-1, kernel=kernel1)

cv2.imshow('Original', image)
cv2.imshow('Identity', identity)
    
cv2.waitKey()
# cv2.imwrite('identity.jpg', identity)
cv2.destroyAllWindows()

# Apply blurring kernel
kernel2 = np.ones((5, 5), np.float32) / 25
img = cv2.filter2D(src=image, ddepth=-1, kernel=kernel2)

cv2.imshow('Original', image)
cv2.imshow('Kernel Blur', img)
    
cv2.waitKey()
cv2.destroyAllWindows()