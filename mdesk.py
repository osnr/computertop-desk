# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 16:44:27 2017

@author: sakurai
"""

import numpy as np
import time
import pickle
import cv2
import subprocess
import re

from mss import mss

MIN_THRES = 200
SHOW_THRES = False

KBD_YELLOW_LOWER = [100, 160, 200]
KBD_YELLOW_UPPER = [160, 255, 255]

find_windows = subprocess.Popen("swift ./find_windows.swift", shell=True, stdout=subprocess.PIPE)
window_regex = re.compile(r'(.*)\t\(([\d\.]+), ([\d\.]+), ([\d\.]+), ([\d\.]+)\)')
windows = []
for line in find_windows.stdout.readlines():
    line_parts = window_regex.match(line.decode("utf-8"))
    title, x, y, width, height = line_parts.groups()
    # if ("Google" in title or "fish" in title or "Stack Overflow" in title or "Untitled 3" in title) and float(height) < 1000:
    if float(height) < 600:
        windows.append({
            'title': title,
            'left': float(x),
            'top': float(y),
            'width': float(width),
            'height': float(height),
            'last_centroid': np.float32([0, 0]),
            'drawing': None
        })
print(len(windows), windows)

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
    img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

    cv2.GaussianBlur(img, (3,3), 0, img)
    
    _, img = cv2.threshold(img, MIN_THRES, 255, cv2.THRESH_BINARY)
    if SHOW_THRES:
        cv2.imshow('thres', img)

    #cv2.UMat(np.zeros((HEIGHT, WIDTH, 3), np.uint8))
    # new = get_new(orig)

    # CANNY = 84
    # edges = cv2.Canny(img, 0, CANNY, apertureSize=3)

    # finding contours
    _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    contours = filter(lambda cont: cv2.arcLength(cont, False) > 100, contours)
    contours = filter(lambda cont: cv2.contourArea(cont) > 10000, contours)

    # simplify contours down to polygons
    rects = []
    for cont in contours:
        rect = cv2.approxPolyDP(cont, 80, True).copy().reshape(-1, 2)
        if len(rect) == 4: rects.append(rect)

    # cv2.drawContours(new, rects,-1,(255,0,255),1)

    # for rectIdx, rect in enumerate(rects):
    #     for i in range(4):
    #         cv2.putText(new, str(i), (rect[i][0] + 3, rect[i][1] + 3), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # cv2.putText(new, str(rectIdx), (rect[0][0], rect[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return rects


def centroid(rect):
    moments = cv2.moments(rect)
    return np.float32([moments['m10']/moments['m00'], moments['m01']/moments['m00']])

def kbddet(img):
    lower = np.array(KBD_YELLOW_LOWER, dtype='uint8')
    upper = np.array(KBD_YELLOW_UPPER, dtype='uint8')
    mask = cv2.inRange(img, lower, upper)

    # new = get_new(img)
    # _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = list(filter(lambda cont: cv2.contourArea(cont) > 10, contours))

    # cv2.drawContours(new, contours, -1, (255,0,255),1)
    # cv2.imshow('new', new)

    # if len(contours) != 3:
    #     raise ValueError("Weird # of contours, can't find KBD")

    # points = [centroid(cont) for cont in contours]
    # points_by_x = sorted(points, key=lambda pt: pt[0])

    # TODO: Return start, end of whisker

    return centroid(mask)

WIDTH = 1280
HEIGHT = 800 - 22 # - title bar height

if __name__ == '__main__':
    sct = mss()

    # create calibration outline
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
    image[0:HEIGHT, 0:10] = [1, 0, 0]  # blue on left
    image[0:HEIGHT, (WIDTH - 10):WIDTH] = [0, 1, 0]  # green on right
    image[0:10, 0:WIDTH] = [0, 0, 1]  # red on top
    image[(HEIGHT - 10):HEIGHT, 0:WIDTH] = [0, 0, 1]  # ?? on bottom

    margin = 200
    project_pts = [
        (margin, margin),
        (WIDTH - margin, margin),
        (margin, HEIGHT - margin),
        (WIDTH - margin, HEIGHT - margin)
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
    # M = cv2.getAffineTransform(np.float32(camera_pts[0:3]), np.float32(project_pts[0:3]))
    # iM, _ = cv2.findHomography(np.float32(project_pts), np.float32(camera_pts))

    kbd_centroid = None
    kbd_closest_window = None
    while True:
        ret_val, img = cam.read()
        rows, cols, ch = img.shape
        # img = cv2.UMat(img)

        warped = cv2.warpPerspective(img, M, (cols, rows))[0:HEIGHT, 0:WIDTH]
        # warped = cv2.warpAffine(img, M, (cols, rows))[0:HEIGHT, 0:WIDTH]

        # cv2.imshow('projector', warped)
        # if cv2.waitKey(1) == 27: break
        # continue

        try:
            kbd_centroid = kbddet(warped)
        except: pass

        rects = paperdet(warped)
        # det = det.get()

        det = np.zeros_like(warped)

        for window in windows:
            window["drawing"] = None

        for rectIdx, rect in enumerate(rects):
            cen = centroid(rect)
            try:
                window = sorted([window for window in windows if window["drawing"] is None], key=lambda w: np.linalg.norm(cen - w["last_centroid"]))[0]
                window["drawing"] = rect
                window["last_centroid"] = cen
            except:
                print("Excess window!")
                window = windows[0]

            sct_img = np.array(sct.grab(window))[:,:,:3]
            sct_height, sct_width, _ = sct_img.shape
            # sct_img = cv2.UMat(sct_img)

            # max_dist = 0
            # max_pt = None
            # for i in range(1, 4):
            #     dist = np.linalg.norm(rect[i] - rect[0])
            #     if dist > max_dist:
            #         max_dist = dist
            #         max_pt = rect[i]

            moments = cv2.moments(rect)

            WMARGIN = 20
            Mw, _ = cv2.findHomography(
                np.float32([
                    (-WMARGIN, -WMARGIN),
                    (-WMARGIN, sct_height + WMARGIN),
                    (sct_width + WMARGIN, sct_height + WMARGIN),
                    (sct_width + WMARGIN, -WMARGIN)
                ]),
                np.float32([rect[0], rect[1], rect[2], rect[3]]))
                # np.float32([rect[0], (moments['m10']/moments['m00'], moments['m01']/moments['m00']), rect[2], rect[3]]))

            sct_img_warped = cv2.warpPerspective(sct_img, Mw, (WIDTH, HEIGHT))

            det += sct_img_warped # .get()

        try:
            if kbd_centroid is not None:
                # cv2.circle(det, (kbd_centroid[0], kbd_centroid[1]), 10, (255, 0, 255), 10)

                new_kbd_closest_window = sorted([window for window in windows if window["drawing"] is not None], key=lambda w: np.linalg.norm(kbd_centroid - w["last_centroid"]))[0]
                kbd_closest_rect = new_kbd_closest_window["drawing"]
                cv2.drawContours(det, [kbd_closest_rect], -1, (255,0,255),1)

                if new_kbd_closest_window != kbd_closest_window:
                    kbd_closest_window = new_kbd_closest_window
                    cmd = "/Users/osnr/Code/mdesk/mdesk/click -x {} -y {}".format(int(kbd_closest_window["left"] + 10), int(kbd_closest_window["top"] + 70))
                    # subprocess.run(cmd, shell=True)
                    kbd_closest_window_file = open("focus_closest_window.sh", "w")
                    kbd_closest_window_file.write("echo hi > hi.txt\n" + cmd)
                    kbd_closest_window_file.close()
        except: pass

        cv2.imshow('projector', det)
        if cv2.waitKey(1) == 27: break

    cv2.destroyAllWindows()
