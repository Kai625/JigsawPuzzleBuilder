import cv2
import numpy as np


def getPerscpective(pts1, pts2):
    A = np.array([[pts1[0][0], pts1[0][1], 1, 0, 0, 0, -pts1[0][0] * pts2[0][0], -pts1[0][1] * pts2[0][0]],
                  [pts1[1][0], pts1[1][1], 1, 0, 0, 0, -pts1[1][0] * pts2[1][0], -pts1[1][1] * pts2[1][0]],
                  [pts1[2][0], pts1[2][1], 1, 0, 0, 0, -pts1[2][0] * pts2[2][0], -pts1[2][1] * pts2[2][0]],
                  [pts1[3][0], pts1[3][1], 1, 0, 0, 0, -pts1[3][0] * pts2[3][0], -pts1[3][1] * pts2[3][0]],
                  [0, 0, 0, pts1[0][0], pts1[0][1], 1, -pts1[0][0] * pts2[0][1], -pts1[0][1] * pts2[0][1]],
                  [0, 0, 0, pts1[1][0], pts1[1][1], 1, -pts1[1][0] * pts2[1][1], -pts1[1][1] * pts2[1][1]],
                  [0, 0, 0, pts1[2][0], pts1[2][1], 1, -pts1[2][0] * pts2[2][1], -pts1[2][1] * pts2[2][1]],
                  [0, 0, 0, pts1[3][0], pts1[3][1], 1, -pts1[3][0] * pts2[3][1], -pts1[3][1] * pts2[3][1]]])
    B = np.array([[pts2[0][0], pts2[1][0], pts2[2][0], pts2[3][0], pts2[0][1], pts2[1][1], pts2[2][1], pts2[3][1]]]).T
    x = np.linalg.solve(A, B)
    rotationMatrix = np.append(x, 1)
    return rotationMatrix.reshape(3, 3)


img = cv2.imread("opencv_frame_0.png")
img2 = img.copy()
c = 0
while c < 1:
    c += 1
    # a = input("Max sat: ")
    imgH = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    background_lower = np.array([165, 100, 165],
                                dtype="uint8")
    background_upper = np.array([180, 255, 255],
                                dtype="uint8")
    # background_lower = np.array([75, 75, 105],
    #                             dtype="uint8")
    # background_upper = np.array([180, 255, 255],
    #                             dtype="uint8")
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
    cv2.imwrite("AA.png", mask_inv_B)
    # edges = cv2.Canny(mask_inv_B, 30, 200)
    # dist = cv2.distanceTransform(mask_inv_B, cv2.DIST_L2, 3)
    # loc = np.unravel_index(np.argmax(dist), dist.shape)
    # y, x = loc
    # cv2.circle(dist, (x, y), 3, (0, 0, 0), 7)
    # cv2.imshow("C", dist)
    # cv2.waitKey(0)
    contours, hierarchy = cv2.findContours(mask_inv_B, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Only get contours where there are are no parents, thus inside contour.
    # hierarchy[0][i][3]  == -1 is the white (no mask) contours.
    contours = [contours[i] for i in range(len(contours)) if hierarchy[0][i][3] == -1]
    contours = [c for c in contours if 2000 < cv2.contourArea(c) < 4000]
    # Display contours
    contouredImage = cv2.drawContours(img, contours, -1, (0, 255, 75), 2)
    cv2.imshow("Contoured_Image.png", contouredImage)
    cv2.waitKey(0)
    cv2.imwrite("BB.png", contouredImage)
    M = cv2.moments(contours[0])
    cX1 = int(M["m10"] / M["m00"])
    cY1 = int(M["m01"] / M["m00"])
    cv2.circle(contouredImage, (cX1, cY1), 7, (255, 255, 255), -1)
    M = cv2.moments(contours[1])
    cX2 = int(M["m10"] / M["m00"])
    cY2 = int(M["m01"] / M["m00"])
    cv2.circle(contouredImage, (cX2, cY2), 7, (255, 255, 255), -1)
    M = cv2.moments(contours[2])
    cX3 = int(M["m10"] / M["m00"])
    cY3 = int(M["m01"] / M["m00"])
    cv2.circle(contouredImage, (cX3, cY3), 7, (255, 255, 255), -1)
    M = cv2.moments(contours[3])
    cX4 = int(M["m10"] / M["m00"])
    cY4 = int(M["m01"] / M["m00"])
    cv2.circle(contouredImage, (cX4, cY4), 7, (255, 255, 255), -1)
    cv2.imwrite("CC.png", contouredImage)
    width, height = 1200, 2600
    pts1 = np.float32([[cX1, cY1], [cX3, cY3], [cX2, cY2], [cX4, cY4]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = getPerscpective(pts1, pts2)
    # matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img2 = cv2.warpPerspective(img2, matrix, (width, height))
    cv2.imwrite("PP.png", img2)
    # a = np.array([[1, 2], [3, 4]])
    # b = np.array([10, 22])
    # x = np.linalg.solve(a, b)
    # print(x)
