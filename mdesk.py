# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 16:44:27 2017

@author: sakurai
"""


import numpy as np
import cv2

if __name__ == '__main__':
    width = 1280
    height = 800 - 22 # - title bar height

    # create image
    image = np.ones((height, width, 3), dtype=np.float32)
    image[:10, :10] = 0  # black at top-left corner
    image[height - 10:, :10] = [1, 0, 0]  # blue at bottom-left
    image[:10, width - 10:] = [0, 1, 0]  # green at top-right
    image[height - 10:, width - 10:] = [0, 0, 1]  # red at bottom-right

    window_name = 'projector'
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, 1680, 0)
    cv2.imshow(window_name, image)

    cam = cv2.VideoCapture(1)
    while True:
        ret_val, img = cam.read()
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27: break
    
    cv2.destroyAllWindows()
