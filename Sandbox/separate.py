from __future__ import division
from __future__ import print_function

import cv2 as cv
import numpy as np

# Create an image
r = 100
src = cv.imread("AA.png", 0)
# Create a sequence of points to make a contour

contours, _ = cv.findContours(src, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# Calculate the distances to the contour
raw_dist = np.empty(src.shape, dtype=np.float32)
for i in range(src.shape[0]):
    for j in range(src.shape[1]):
        raw_dist[i, j] = cv.pointPolygonTest(contours[0], (j, i), True)
minVal, maxVal, _, maxDistPt = cv.minMaxLoc(raw_dist)
cv.imshow("A", src)
cv.waitKey(0)
src = src[maxDistPt[1] + 50:, :]
src[0, :] = 255
cv.imshow("A", src)
cv.waitKey(0)
contours, _ = cv.findContours(src, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# Calculate the distances to the contour
raw_dist = np.empty(src.shape, dtype=np.float32)
for i in range(src.shape[0]):
    for j in range(src.shape[1]):
        raw_dist[i, j] = cv.pointPolygonTest(contours[0], (j, i), True)
minVal, maxVal, _, maxDistPt = cv.minMaxLoc(raw_dist)
minVal = abs(minVal)
maxVal = abs(maxVal)
# Depicting the  distances graphically
drawing = np.zeros((src.shape[0], src.shape[1], 3), dtype=np.uint8)
for i in range(src.shape[0]):
    for j in range(src.shape[1]):
        # if raw_dist[i, j] < 0:
        #     drawing[i, j, 0] = 255 - abs(raw_dist[i, j]) * 255 / minVal
        if raw_dist[i, j] > 0:
            drawing[i, j, 2] = 255 - raw_dist[i, j] * 255 / maxVal
        else:
            drawing[i, j, 0] = 255
            drawing[i, j, 1] = 255
            drawing[i, j, 2] = 255
cv.circle(drawing, maxDistPt, int(maxVal), (255, 255, 255), 1, cv.LINE_8, 0)
cv.imshow('Source', src)
cv.imshow('Distance and inscribed circle', drawing)
cv.waitKey()
