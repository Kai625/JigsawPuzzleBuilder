"""
Author:         Mr. H. van der Westhuizen
Date opened:    11 September 2021
Student Number: u18141235
Project number: HG1

The basic image handling file is used to do basic image
manipulation, characteristic extraction and writing.
"""
import os

import cv2
import numpy as np


def saveResult(saveName, img):
    """
        This method saves images to results
        Input: N/A
        Output: RGB and GRAY scale images of the puzzle pieces.
    """
    percentage = 500 / img.shape[1]
    width = int(img.shape[1] * percentage)
    height = int(img.shape[0] * percentage)

    # dsize
    dsize = (width, height)
    resizedForDisplay = cv2.resize(img, dsize)
    cv2.imwrite(os.path.abspath("Results" + "/" + saveName), resizedForDisplay)


def get_test_images(configObj):
    """
    This method returns the RGB and GRAY images of the puzzle pieces.
    Input: N/A
    Output: RGB and GRAY scale images of the puzzle pieces.
    """
    if configObj.TESTING:
        rgbImage = cv2.imread(os.path.abspath("Resources" + "/" + configObj.RGB_IMAGE), 1)
        grayImage = cv2.imread(os.path.abspath("Resources" + "/" + configObj.GRAY_IMAGE), 1)
        # rgbImage, grayImage = shiftPerspective(rgbImage, grayImage)
        grayImage = cv2.cvtColor(grayImage, cv2.COLOR_BGR2HSV)
        grayImage = grayImage[:, :, 2]
        saveResult("RGBImage.png", rgbImage)
        saveResult("GrayImage.png", grayImage)
        return rgbImage, grayImage


def shiftPerspective(RGBImage, GrayImage):
    width, height = 315, 736
    pts1 = np.float32([[256, 4], [679, 5], [249, 1009], [679, 1006]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    RGBImage = cv2.warpPerspective(RGBImage, matrix, (width, height))
    GrayImage = cv2.warpPerspective(GrayImage, matrix, (width, height))
    return RGBImage, GrayImage


def show_save(name, img, fx=1, fy=1):
    """
    Save the image to the results folder.
    Input: Image name, image matrix, x scale, y scale.
    Output: Saved and displayed image.
    """
    cv2.imshow(name, cv2.resize(img, dsize=None, fx=fx, fy=fy))
    cv2.imwrite(RESULTS_PATH + name + ".png", img)
    cv2.waitKey()


def create_mask(image, configObj):
    """
    This method creates a binary mask of the image and reduces noise.
    Input: Gray scale image.
    Output: Altered Gray scale image, binary image [0,1] of the mask.
    """
    # # All values above the SELECTED_THRESHOLD is pulled down to 0.
    # image[image >= configObj.MASK_TRESHOLD] = 0
    # # All values above the SELECTED_THRESHOLD is pushed up to 255.
    # image[image >= 1] = 255
    # kernel = np.ones((3, 3), np.uint8)
    # # Remove small noise dots outside piece. (May not need.)
    # # TODO: implement morphology FP.
    # image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    # # Remove small noise dots inside piece. (May not need.)
    # image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    # binaryImage = image.copy()
    # # Turn the mask into binary matrix.
    # binaryImage[binaryImage >= 1] = 1
    #
    # return image, binaryImage

    imgH = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    background_lower = np.array([165, 75, 105],
                                dtype="uint8")
    background_upper = np.array([180, 255, 255],
                                dtype="uint8")
    # print(imgH)
    mask_B = cv2.inRange(imgH, background_lower, background_upper)
    mask_inv_B = cv2.bitwise_not(mask_B)
    cv2.imshow("B", mask_inv_B)
    cv2.waitKey(0)
    binaryImage = mask_inv_B.copy()
    binaryImage[binaryImage >= 1] = 1
    return mask_inv_B, binaryImage


def apply_mask_rgb(image, binaryMatrix):
    """
    Apply mask to a copy of the RGB image.
    Input: Original RGB image, binary matrix
    Output: RGB image with mask applied.
    """
    # Must make the binary matrix 3D.
    mask = cv2.cvtColor(binaryMatrix, cv2.COLOR_GRAY2RGB)
    # If 0 then it will be black and 1 will stay the same.
    maskedImage = image * mask
    return maskedImage


def get_image_block(image, x, y, w, h, configObj):
    """
    Input: Image and dimensions of block.
    Output: Image cropped according to dimensions.
    """
    return image[y - configObj.PIECE_MARGIN: y + h + configObj.PIECE_MARGIN,
           x - configObj.PIECE_MARGIN: x + w + configObj.PIECE_MARGIN]


def isolate_Piece(grayPiece, RGBPiece):
    """"
    Remove all the small objects around the piece be making use of the area of the piece.
    Input: Non isolated piece.
    Output: Isolated piece
    """
    # Find the contours of all the object in the image.
    contours, _ = cv2.findContours(grayPiece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # The largest contour must be the piece itself
    max_c = max(contours, key=cv2.contourArea)

    for c in contours:
        # If the contour is smaller than the piece image. The
        if cv2.contourArea(c) < cv2.contourArea(max_c):
            # Blacken out any contours smaller than the piece itself.
            cv2.drawContours(RGBPiece, [c], -1, (0, 0, 0), thickness=cv2.FILLED)
            cv2.drawContours(grayPiece, [c], -1, 0, thickness=cv2.FILLED)
    return RGBPiece, grayPiece


def createChain(connections):
    """
    This method is used to chain the 4 corners together, so pair one holds (1,2)
    and pair 3 hold (2,3) this rearrange the chains. Sets pair 3 as 2.
    Input: Corners
    Output: Chains connecting corners.
    """
    chains = []
    chainLinks = connections.copy()
    while len(chainLinks) > 0:
        currentLink = chainLinks.pop(0)
        currentChain = [currentLink]
        while True:
            neighbour = [pair for pair in chainLinks if pair[0] == currentLink[1]]
            if len(neighbour) == 0:
                chains.append(currentChain)
                break
            else:
                currentLink = neighbour[0]
                currentChain.append(currentLink)
                chainLinks.remove(currentLink)
    return chains


def distance_from_line(corner, point):
    x0, y0 = point[0], point[1]
    x1, y1 = corner[0], corner[1]
    return (y1 * x0 - x1 * y0) / (np.sqrt(x1 ** 2 + y1 ** 2))


def compare_piece_edges(firstEdgeMatrix, secondEdgeMatrix):
    """
    The method compares the edges of the two pieces and checks how well they fit
    each other by making use of the binary edge matrix.
    Input: Edges of two different pieces.
    Output: Score the fitness score.
    """
    # Create the mask frame that is used to compare the 2 images.
    compareMask = (max(firstEdgeMatrix.shape[0], secondEdgeMatrix.shape[0]), firstEdgeMatrix.shape[1])
    # 0 filled mask.
    compareMask = np.zeros(compareMask, np.uint8)
    # Fill the first half with 1's this is needed since the one edgem any be taller and will
    # alter the result.
    compareMask[:, :firstEdgeMatrix.shape[1] // 2] = 1

    frame1 = compareMask
    frame2 = compareMask.copy()
    # The first edge matrix is copied onto the frame.
    frame1[:firstEdgeMatrix.shape[0], :firstEdgeMatrix.shape[1]] = firstEdgeMatrix
    # The second edge matrix is copied onto the frame.
    frame2[:secondEdgeMatrix.shape[0], :secondEdgeMatrix.shape[1]] = secondEdgeMatrix

    # Flip the second piece. over the x axis then y axis.
    frame2 = cv2.flip(frame2, 1)
    frame2 = cv2.flip(frame2, 0)

    kernel_erode = np.ones((1, 9), np.uint8)
    kernel_dilate = np.ones((4, 1), np.uint8)
    frame2 = cv2.erode(frame2, kernel_erode, iterations=1)
    frame2 = cv2.dilate(frame2, kernel_dilate, iterations=1)

    # Check how well they fit in together. Check the overlapping part,
    # or check the none connection.
    xored = cv2.bitwise_xor(frame1, frame2)
    xored = 1 - xored
    # The lower the score the better.
    score = np.sum(xored)
    return score
