import cv2
import numpy as np

img = cv2.imread("m3.jpg")
while True:
    # a = input("Max sat: ")
    imgH = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # background_lower = np.array([165, 75, 105],
    #                             dtype="uint8")
    # background_upper = np.array([180, 255, 255],
    #                             dtype="uint8")
    background_lower = np.array([75, 75, 105],
                                dtype="uint8")
    background_upper = np.array([180, 255, 255],
                                dtype="uint8")
    # print(imgH)
    mask_B = cv2.inRange(imgH, background_lower, background_upper)
    mask_inv_B = cv2.bitwise_not(mask_B)
    kernel = np.ones((3, 3), np.uint8)
    # Remove small noise dots outside piece. (May not need.)
    # TODO: implement morphology FP.
    mask_inv_B = cv2.morphologyEx(mask_inv_B, cv2.MORPH_OPEN, kernel)
    # Remove small noise dots inside piece. (May not need.)
    mask_inv_B = cv2.morphologyEx(mask_inv_B, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("B", mask_inv_B)
    cv2.waitKey(0)
    edges = cv2.Canny(mask_inv_B, 30, 200)
    cv2.imshow("B", edges)
    cv2.waitKey(0)
    cv2.imwrite("AA.png", edges)
    dist = cv2.distanceTransform(mask_inv_B, cv2.DIST_L2, 3)
    loc = np.unravel_index(np.argmax(dist), dist.shape)
    y, x = loc
    # cv2.circle(dist, (x, y), 3, (0, 0, 0), 7)
    # cv2.imshow("C", dist)
    # cv2.waitKey(0)
