# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 16:44:27 2017

@author: sakurai
"""


import numpy as np
import cv2

if __name__ == '__main__':
    width = 1280
    height = 720

    # create image
    image = np.ones((height, width, 3), dtype=np.float32)
    image[:10, :10] = 0  # black at top-left corner
    image[height - 10:, :10] = [1, 0, 0]  # blue at bottom-left
    image[:10, width - 10:] = [0, 1, 0]  # green at top-right
    image[height - 10:, width - 10:] = [0, 0, 1]  # red at bottom-right

    window_name = 'projector'
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, 1500, -50)
    cv2.imshow(window_name, image)
    cv2.waitKey()
    cv2.destroyAllWindows()
