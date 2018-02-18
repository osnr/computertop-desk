# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 16:44:27 2017

@author: sakurai
"""


import numpy as np
import time
import cv2

if __name__ == '__main__':
    width = 1280
    height = 800 - 22 # - title bar height

    # create calibration outline
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[0:height, 0:10] = [1, 0, 0]  # blue on left
    image[0:height, (width - 10):width] = [0, 1, 0]  # green on right
    image[0:10, 0:width] = [0, 0, 1]  # red on top
    image[(height - 10):height, 0:width] = [0, 0, 1]  # ?? on bottom

    margin = 200
    project_pts = [
        (margin, margin),
        (width - margin, margin),
        (margin, height - margin),
        (width - margin, height - margin)
    ]
    for idx, project_pt in enumerate(project_pts):
        cv2.circle(image, project_pt, 10, (0, 255, 0) if idx != 3 else (0, 255, 255), 10)

    window_name = 'projector'
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, 1680, 0)
    cv2.imshow(window_name, image)

    print("Press key to begin calibration.")
    cv2.waitKey()

    cam = cv2.VideoCapture(1)

    camera_width = 1920
    camera_height = 1080
    camera_pts = []
    def calibration_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            camera_pts.append([x * (camera_width / width), y * (camera_height / height)])
            print(camera_pts)


    ret_val, img = cam.read()
    cv2.imshow('camera calibrator', cv2.resize(img, (width, height)))
    cv2.setMouseCallback("camera calibrator", calibration_click)

    while True:
        # ret_val, img = cam.read()
        # cv2.imshow('camera calibrator', cv2.resize(img, (width, height)))
        if cv2.waitKey(1) == 27: break

    M, status = cv2.findHomography(np.float32(camera_pts), np.float32(project_pts))
    print(M)

    while True:
        ret_val, img = cam.read()
        rows, cols, ch = img.shape
        warped = cv2.warpPerspective(img, M, (cols, rows))
        cv2.imshow('projector', warped[0:height, 0:width])
        if cv2.waitKey(1) == 27: break

    cv2.destroyAllWindows()
