"""
Author:         Mr. H. van der Westhuizen
Date opened:    11 September 2021
Student Number: u18141235
Project number: HG1

The following file extract the pieces from the frame and
create Piece objects for each piece.
"""
import cv2
import numpy as np

from PuzzleSolving.basic_image_handler import get_image_block, saveResult
from PuzzleSolving.piece import Piece


def piece_extraction(maskedImage, originalGrayImage, binaryImage, configObj):
    """
    The pieces are extracted from the frame.
    Input: Masked RGB image, masked Gray scale image and binary matrix.
    Output: Array of Piece objects, image that display all characteristics.
    """

    # Create copy of the RGB masked image.
    matrix = maskedImage.copy()
    # Initialise the piece object array.
    pieces = []
    # Piece ID
    ID = 0

    # TODO : Implement contour FP
    # Find the contours of the frame.
    contours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Only get contours where there are are no parents, thus inside contour.
    # hierarchy[0][i][3]  == -1 is the white (no mask) contours.
    contours = [contours[i] for i in range(len(contours)) if hierarchy[0][i][3] == -1]

    # Display contours
    contouredImage = cv2.drawContours(matrix, contours, -1, (0, 255, 75), 2)
    saveResult("Contoured_Image.png", contouredImage)

    # Create a tuple of the contours array and the area amount of the contour.
    # also get rid of all the small contours of the image.
    contours = [(c, cv2.contourArea(c)) for c in contours if cv2.contourArea(c) > configObj.CONTOUR_AREA_THRESHOLD]

    # TODO: Overlapping

    # Use the contours to identify the pieces
    addFiles = ["Contoured_Image.png"]
    for c, area in contours:
        # Use the bounding box to extract the pieces.
        # TODO : bounding box FP
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(matrix, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract an image of the piece using the bounding box dimensions.
        pieceRGBImageBlock = get_image_block(maskedImage.copy(), x, y, w, h, configObj)
        pieceGrayImageBlock = get_image_block(originalGrayImage.copy(), x, y, w, h, configObj)

        # Display each of the blocks that are extracted.
        saveResult("Extracted_Piece_%d.png" % ID, pieceRGBImageBlock)
        addFiles.append("Extracted_Piece_%d.png" % ID)

        # Initialise piece object.
        pieces.append(
            Piece(pieceRGBImageBlock, pieceGrayImageBlock, ID,
                  np.array([x - configObj.PIECE_MARGIN, y - configObj.PIECE_MARGIN]), configObj))

        # Get and place centroid on the frame.
        centroid = tuple(pieces[ID].get_pickup().tolist())
        cv2.circle(matrix, centroid, 10, [0, 0, 255], -1)

        # Label piece and add the corners
        cv2.putText(matrix, str(ID), centroid, cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255))
        for corner in pieces[ID].get_real_corners():
            cv2.circle(matrix, tuple(corner.tolist()), 10, [255, 0, 0], -1)

        ID += 1

    configObj.stages.append(str(3) + " Piece extraction")
    configObj.stageFiles.append(addFiles)
    return matrix, pieces
