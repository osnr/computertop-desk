import cv2
import numpy as np

CANNY = 84

def get_new(old):
    new = np.zeros(old.shape, np.uint8)
    return new

cam = cv2.VideoCapture(1)
while True:
    ret_val, orig = cam.read()
    # img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    # cv2.GaussianBlur(img, (3,3), 0, img)
    # edges = cv2.Canny(img, 0, CANNY, apertureSize=3)

    # _, contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    # contours = filter(lambda cont: cv2.arcLength(cont, False) > 100, contours)
    # # contours = filter(lambda cont: cv2.contourArea(cont) > 10000, contours)

    # rects = []
    # for cont in contours:
    #     rect = cv2.approxPolyDP(cont, 40, True).copy().reshape(-1, 2)
    #     rects.append(rect)

    # new = get_new(orig)

    # cv2.drawContours(new, rects ,-1,(255,0,255),1)

    # cv2.imshow('new', new)

    # lower = np.array([25, 146, 190], dtype='uint8')
    # upper = np.array([62, 174, 250], dtype='uint8')
    lower = np.array([100, 160, 200], dtype='uint8')
    upper = np.array([160, 255, 255], dtype='uint8')
    mask = cv2.inRange(orig, lower, upper)

    cv2.imshow('img', orig)
    cv2.imshow('output', mask)

    if cv2.waitKey(1) == 27: break
