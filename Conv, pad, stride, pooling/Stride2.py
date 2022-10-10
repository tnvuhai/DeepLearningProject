# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 10:07:20 2022

@author: mamto
"""

import cv2
import numpy as np

def apply_sliding_window(img, kernel, padding=0, stride=1):
    h, w = img.shape[:2]
    
    img_p = np.zeros([h+2*padding, w+2*padding])
    img_p[padding:padding+h, padding:padding+w] = img
    
    kernel = np.array(kernel)
    assert len(kernel.shape) == 2 and kernel.shape[0] == kernel.shape[1] # square kernel
    assert kernel.shape[0] % 2 != 0 # kernel size is odd number

    k_size = kernel.shape[0]
    k_half = int(k_size/2)
    
    y_pos = [v for idx, v in enumerate(list(range(k_half, h-k_half))) if idx % stride == 0]
    x_pos = [v for idx, v in enumerate(list(range(k_half, w-k_half))) if idx % stride == 0]
    
    new_img = np.zeros([len(y_pos), len(x_pos)])
    for new_y, y in enumerate(y_pos):
        for new_x, x in enumerate(x_pos):
            if k_half == 0:
                pixel_val = img_p[y, x] * kernel # element-wise multiply
            else:
                pixel_val = np.sum(img_p[y-k_half:y-k_half+k_size, x-k_half:x-k_half+k_size] * kernel) # dot product: https://minhng.info/toan-hoc/y-nghia-tich-vo-huong.html
            new_img[new_y, new_x] = pixel_val
    
    return new_img

def apply_sliding_window_on_3_channels(img, kernel, padding=0, stride=1):
    layer_blue = apply_sliding_window(img[:,:,0], kernel, padding, stride)
    layer_green = apply_sliding_window(img[:,:,1], kernel, padding, stride)
    layer_red = apply_sliding_window(img[:,:,2], kernel, padding, stride)
    
    new_img = np.zeros(list(layer_blue.shape) + [3])
    new_img[:,:,0], new_img[:,:,1], new_img[:,:,2] = layer_blue, layer_green, layer_red
    return new_img

if __name__ == "__main__":
    img = cv2.imread("HaiImage.jpg")
    
    new_img = apply_sliding_window_on_3_channels(img, kernel=[[1]], padding=0, stride=2)
    
    cv2.imshow('img_7_new.jpg', new_img)
    print('Shape img_7.jpg:', img.shape)
    print('Shape img_7_new.jpg:', new_img.shape)
    print('Saved new image @ img_7_new.jpg')
    cv2.waitKey(1)
    print('------------')
    
    lighten_blur_img = apply_sliding_window_on_3_channels(img, kernel=[[0.33, 0.33, 0.33], [0.33, 0.33, 0.33], [0.33, 0.33, 0.33]], padding=1, stride=1)
    cv2.imshow('img_7_lighten_blur.jpg', lighten_blur_img)
    print('Shape img_7.jpg:', img.shape)
    print('Shape img_7_lighten_blur.jpg:', lighten_blur_img.shape)
    print('Saved new image @ img_7_lighten_blur.jpg')
    cv2.waitKey(1)