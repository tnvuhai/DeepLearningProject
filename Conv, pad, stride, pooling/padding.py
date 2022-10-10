# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 08:37:51 2022

@author: mamto
"""


import cv2
import imutils
# path
  
# Reading an image in default mode
image = cv2.imread("HaiImage.jpg")
image = imutils.resize(image, 300)

# Window name in which image is displayed

# Using cv2.copyMakeBorder() method
image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value = 0)
image2 = cv2.copyMakeBorder(image, 100, 100, 50, 50, cv2.BORDER_REFLECT)

# Displaying the image
cv2.imshow("Black Border", image)
cv2.imshow("Reflect Border", image2)
cv2.waitKey()
cv2.destroyAllWindows()