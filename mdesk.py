# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 16:44:27 2017

@author: sakurai
"""


import numpy as np
import time
import pickle
import cv2

from mss import mss

def calibrate():
    print("Press key to begin calibration.")
    cv2.waitKey()

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

    return camera_pts

def get_new(old):
    new = np.zeros(old.shape, np.uint8)
    return new

def paperdet(orig):
    # these constants are carefully picked
    MORPH = 9
    CANNY = 84
    HOUGH = 25

    img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

    _, img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
    cv2.imshow('thres', img)

    cv2.GaussianBlur(img, (3,3), 0, img)

    # this is to recognize white on white
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(MORPH,MORPH))
    dilated = cv2.dilate(img, kernel)

    edges = cv2.Canny(dilated, 0, CANNY, apertureSize=3)

    # lines = cv2.HoughLinesP(edges, 1,  3.14/180, HOUGH)
    # for line in lines[0]:
    #      cv2.line(edges, (line[0], line[1]), (line[2], line[3]),
    #                      (255,0,0), 2, 8)

    new = get_new(orig)

    # finding contours
    _, contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    contours = filter(lambda cont: cv2.arcLength(cont, False) > 100, contours)
    contours = filter(lambda cont: cv2.contourArea(cont) > 10000, contours)

    # simplify contours down to polygons
    rects = []
    for cont in contours:
        rect = cv2.approxPolyDP(cont, 40, True).copy().reshape(-1, 2)
        if len(rect) == 4: rects.append(rect)

    # # that's basically it
    # cv2.drawContours(orig, rects,-1,(0,255,0),1)

    # show only contours

    cv2.drawContours(new, rects,-1,(255,0,255),1)
    return new, rects


if __name__ == '__main__':
    width = 1280
    height = 800 - 22 # - title bar height

    sct = mss()

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

    cam = cv2.VideoCapture(1)

    camera_width = 1920
    camera_height = 1080
    try:
        camera_pts = pickle.load(open("camera_pts.p", "rb"))
    except:
        camera_pts = calibrate()
        pickle.dump(camera_pts, open("camera_pts.p", "wb"))

    M, status = cv2.findHomography(np.float32(camera_pts), np.float32(project_pts))
    iM, _ = cv2.findHomography(np.float32(project_pts), np.float32(camera_pts))

    while True:
        ret_val, img = cam.read()

        rows, cols, ch = img.shape
        warped = cv2.warpPerspective(img, M, (cols, rows))[0:height, 0:width]
        # cv2.imshow('img', warped)

        det, rects = paperdet(warped)

        if len(rects) >= 1:
            rect = rects[0]
            sct_img = np.array(sct.grab({ 'top': 0, 'left': 800, 'width': 200, 'height': 200}))
            moments = cv2.moments(rect)
            Mw = cv2.getAffineTransform(
                np.float32([(0, 0), (200, 0), (100, 100)]),
                np.float32([rect[0], rect[1], (moments['m10']/moments['m00'], moments['m01']/moments['m00'])]))

            sct_img_warped = cv2.warpAffine(sct_img, Mw, (width, height))

            cv2.imshow('projector', sct_img_warped)

        cv2.imshow('det', det)
        if cv2.waitKey(1) == 27: break

    cv2.destroyAllWindows()
