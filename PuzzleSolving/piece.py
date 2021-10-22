"""
Author:         Mr. H. van der Westhuizen
Date opened:    12 September 2021
Student Number: u18141235
Project number: HG1

The following file holds the Piece object and is responsible for gathering
and storing all the piece characteristics.
"""
import itertools
import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate
from scipy.signal import find_peaks

from PuzzleSolving.basic_image_handler import isolate_Piece, createChain, distance_from_line, \
    compare_piece_edges, saveResult


class Piece(object):
    def __init__(self, originalRGBPiece, originalGrayPiece, ID, relativePosition, configObj):
        """
        Gather all the information of the piece using the cropped images provided.
        Input: Cropped image of the piece.
        """
        # General attributes.
        self._originalRGBPiece = originalRGBPiece
        self._originalGrayPiece = originalGrayPiece
        self._ID = ID
        self._pieceName = "Piece %d" % ID
        self._relativePosition = relativePosition

        # Isolate the piece by removing any small object around the piece.
        self._originalRGBPiece, self._originalGrayPiece = isolate_Piece(self._originalGrayPiece, self._originalRGBPiece)
        # Make a copy for displaying only.
        self._pieceDisplay = originalRGBPiece.copy()

        # Position attributes
        self._theta = 0
        self._centroid = self._find_centroid()
        self._pickupLocation = self._find_pickup_location()

        saveResult("Position_Points_Piece_%d.png" % ID, self._pieceDisplay)

        # Find the edge of the entire piece and extract the matrix of points.
        self._erodedEdgeMatrix, self._realEdgeMatrix = self._find_edges()

        # Find the corners of the piece.
        self._corners, self._cornerAngles = self._find_corners(configObj)

        # Get the separate edges using the corners and edge matrix.
        # The colour edge must use the eroded image since it will ensure all the background is gone.
        self._splitColourEdges = self.divide_edges(self._erodedEdgeMatrix)
        self._splitShapeEdges = self.divide_edges(self._realEdgeMatrix)

        # Create a binary matrix image that is split by the edge.
        self._binaryEdgeImage = self.create_shape_matrix()
        self._color_vectors = self.create_colour_vector()

        # Evaluate the edges of the piece and create classification array.
        self._puzzleEdge, self._pieceEdgeArray = self._puzzle_edges()
        self._cornerPiece, self._edgePiece = self._setPieceType()

    def __repr__(self):
        # Represents the piece object externally.
        return self._pieceName

    def get_ID(self):
        # Get the ID of the piece.
        return self._ID

    def get_piece_edge_array(self):
        return self._pieceEdgeArray

    def get_binary_edge_image(self):
        return self._binaryEdgeImage

    def get_pickup(self):
        return self._pickupLocation + self._relativePosition

    def get_real_corners(self):
        return [np.array(corner) + self._relativePosition for corner in self._corners]

    def is_puzzle_corner(self):
        return self._cornerPiece

    def is_puzzle_edge(self):
        return self._edgePiece

    def get_piece_edge_indices(self):
        return [idx for idx in range(len(self._pieceEdgeArray)) if self._pieceEdgeArray[idx] == 0]

    def get_piece_slot_indices(self):
        return [idx for idx in range(len(self._pieceEdgeArray)) if self._pieceEdgeArray[idx] == -1]

    def get_piece_tab_indices(self):
        return [idx for idx in range(len(self._pieceEdgeArray)) if self._pieceEdgeArray[idx] == 1]

    def get_piece_connectors_indices(self):
        return [idx for idx in range(len(self._pieceEdgeArray)) if self._pieceEdgeArray[idx] != 0]

    def _find_centroid(self):
        """
        Indicate where the centroid of the piece is and label it.
        Input: Isolated piece image
        Output: Image with identified centroid.
        """
        # TODO : get centroid FP
        _, _, _, centroids = cv2.connectedComponentsWithStats(self._originalGrayPiece)
        # First centroid is of the black area thus centroid[1].
        centroid = tuple(list(centroids[1].astype(int)))

        cv2.circle(self._pieceDisplay, centroid, 3, (0, 255, 0), 5)
        # Label each piece according to center and ID.
        cv2.putText(self._pieceDisplay, str(self._ID), centroid, cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255))

        return centroid

    def _find_pickup_location(self):
        """
        Indicate where the center of gravity of the piece is and label it.
        Input: Isolated piece image
        Output: Image with identified pickup location.
        """
        # This gets the relative center of the white binary image.
        # TODO : distance Transform FP
        gravity = cv2.distanceTransform(self._originalGrayPiece, cv2.DIST_L2, 3)

        # The max value represents center of gravity of the piece.
        loc = np.unravel_index(np.argmax(gravity), self._originalGrayPiece.shape)
        y, x = loc

        # Draw the pickup location.
        cv2.circle(self._pieceDisplay, (x, y), 3, (0, 0, 255), 7)
        return x, y

    def _find_edges(self):
        """
        Find the edge of the original and eroded image.
        Input: Isolated piece image
        Output: Edge matrix of the eroded and original image, only return points on the edge.
        """
        # Erode the image slightly to ensure the image within the edge is clear
        # used manly for the colour section.
        # TODO : Erode and Canny
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(self._originalGrayPiece, kernel)
        # Get the edge using canny. So the canny edge is the pixel edge
        # of the piece.
        erodedEdge = cv2.Canny(eroded, 100, 255)
        # Get the y and x of the edge alone.
        erodedEdgePositions = np.where(erodedEdge != [0])
        # Create array of (x,y) of the edge coordinates.
        erodedEdgeMatrix = np.array(list(zip(erodedEdgePositions[1], erodedEdgePositions[0])))

        # Second method using the Gray image itself.
        # Get the edge using canny
        realEdge = cv2.Canny(self._originalGrayPiece, 100, 255)
        # Get the y and x of the edge alone.
        realEdgePositions = np.where(realEdge != [0])
        # Create tuple of (x,y) of the edge coordinates.
        realEdgeMatrix = np.array(list(zip(realEdgePositions[1], realEdgePositions[0])))

        # Display both edges.
        copyDisplay = self._pieceDisplay.copy()
        copyDisplay[erodedEdgePositions] = [0, 255, 255]
        copyDisplay[realEdgePositions] = [255, 255, 0]
        saveResult("Both_Edges_%d.png" % self._ID, copyDisplay)

        return erodedEdgeMatrix, realEdgeMatrix

    def _find_corners(self, constantObj):
        """
        The following method is used to find the corners of the pieces by making use of angles,
        and distance peaks from the center.
        Input: Isolated piece image.
        Output: Piece corner array and angel size between the corner and centroid.
        """

        edgeInformation = []
        # Tuple of point, distance from centroid and angle between centroid and point -pi till pi (rad).
        for point in self._realEdgeMatrix:
            edgeInformation.append((point, np.linalg.norm(point - np.array(self._centroid)),
                                    math.atan2(point[1] - self._centroid[1], point[0] - self._centroid[0])))

        # Sort edge information by angle since it is then in a circle.
        edgeInformation.sort(key=lambda x: x[2])

        # Create separate copies of the edge information.
        points = np.array([edge[0] for edge in edgeInformation])
        distances = np.array([edge[1] for edge in edgeInformation])
        angles = np.array([edge[2] for edge in edgeInformation])

        length = len(angles)
        distances = np.lib.pad(distances, (0, length // 16), 'wrap')
        angles = np.lib.pad(angles, (0, length // 16), 'wrap')
        angles[length:] += 2 * np.pi

        # Get the local peak distance from the centroid.
        peaks, _ = find_peaks(distances, prominence=(5), threshold=(0, 5), width=(20), distance=(10))

        plt.figure(figsize=(5.5, 4))
        plt.plot(angles, distances)
        plt.plot(angles[peaks], distances[peaks], "x")
        plt.plot(angles, distances)
        plt.ylabel("Distance(pixels)")
        plt.xlabel("Radians(rad)")
        plt.grid()
        plt.savefig(os.path.abspath("Results" + "/" + "Peaks_%d.png" % self._ID))
        plt.close()

        # Revert angles.
        angles[length:] -= 2 * np.pi

        # Create tuple of the peaks, and the angle at that peak.
        possibleCorners = list(zip(peaks, angles[peaks]))

        # I love this, pair each of the possible corners with another used to get angle between them.
        pairs = list(itertools.combinations(possibleCorners, 2))

        # The index of the first peak, second peak, difference in angle.
        pairs = [(a1[0] % length, a2[0] % length, abs(a1[1] - a2[1])) for a1, a2 in pairs]

        # Sort the pairs swap the two possible peaks if needed.
        pairs = [(a1, a2, diff) if a1 < a2 else (a2, a1, diff) for a1, a2, diff in pairs]

        # Remove pairs that pair same points and double pairs.
        equivalents = [pair for pair in pairs if pair[2] == 0]
        to_remove = [pair[1] for pair in equivalents]
        pairs = [pair for pair in pairs if pair[1] not in to_remove]

        if len(to_remove):
            print(self._pieceName, ":\tRemoved Pairs: ", to_remove)

        # Sort pairs by the difference closeness to 90 degrees.
        bestPairs = sorted(pairs, key=lambda x: abs(x[2] - np.pi / 2))[:5]
        # The angel should be between 63 and 117 degrees
        bestPairs = [x for x in bestPairs if abs(x[2] - np.pi / 2) < ((np.pi / 2) * constantObj.CORNER_ANGLE_TRESHHOLD)]
        # Need to sort to ensure the first chain corner is top left.
        bestPairs = sorted(bestPairs, key=lambda x: x[0])

        # Chain the corners by using the peak index points.
        chains = createChain(bestPairs)
        # The max chain is 3 as it connects 4 corners with 3 lines.
        max_chain = max(chains, key=len)
        bestPairs = max_chain
        corners = list(set([pair[0] for pair in bestPairs] + [pair[1] for pair in bestPairs]))[:4]

        # Get the x and y coordinates of the corners.
        corners = points[corners].tolist()
        corners.sort(
            key=lambda x:
            math.atan2(x[1] - self._centroid[1], x[0] - self._centroid[0])
        )

        # Label the corners
        for index, i in enumerate(corners):
            x, y = i
            cv2.circle(self._pieceDisplay, (x, y), 3, color=(255, 0, 0), thickness=-1)
            cv2.putText(self._pieceDisplay, str(index), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255))

        saveResult("Corners_Piece_%d.png" % self._ID, self._pieceDisplay)

        # Get the angles of the corners, used to get separate the edges.
        corner_angles = []
        for point in corners:
            corner_angles.append((math.atan2(point[1] - self._centroid[1], point[0] - self._centroid[0])))

        return corners, corner_angles

    def divide_edges(self, edges):
        """
        The edges are devided into point arrays that are splt using the angles of the corners,
        as the angle of the corners act as the boundaries of the edges. Think of it as a circle,
        with 4 dots. As the angle increases it moves from 1 dot till the other.
        Input: Corners, and edge matrix.
        Output: Separate Edges as a 2D array.
        """
        # Get the angles of each of the edge points, the angles are used as the check.
        edge_angles = [
            ((math.atan2(point[1] - self._centroid[1], point[0] - self._centroid[0])), point)
            for point in edges
        ]
        edge_angles.sort(key=lambda x: x[0])

        # Initialise divided edge array.
        divided_edges = [[]]
        # Corner's angles are retrieved as they are used as the break points of each edge.
        # Inf is used to ensure it reaches the final point of the edge matrix.
        corner_angles = self._cornerAngles + [np.inf]
        edgeCounter = 0

        # Check if the points angle is between two points (smaller than the next point).
        for angle, edge in edge_angles:
            if angle > corner_angles[edgeCounter]:
                edgeCounter += 1
                divided_edges.append([])
            divided_edges[edgeCounter].append(edge)

        # Since the angle of the point can start in the middle of the edge one must
        # append the points of the 2 partial edges.
        if len(divided_edges) > 4:
            divided_edges[4] += divided_edges[0]
            divided_edges.pop(0)

        for index, edge_class in enumerate(divided_edges):
            for edge in edge_class:
                x, y = tuple(edge)
                cv2.circle(self._pieceDisplay, (x, y), 2, color=(50 * index + 50, 100, 50 * index + 50), thickness=-1)

        saveResult("Divided_Edges_Piece_%d.png" % self._ID, self._pieceDisplay)

        return divided_edges

    def create_shape_matrix(self):
        """
        Create a binary matrix that is split using the edge of the piece.
        Input: The divided edges.
        Output: 2D matrix that is divided using the edge.
        """
        edgeMatrix = []
        counter = 0
        indx = 0
        for corner in self._corners:
            # Get the length difference in x and y.
            edgeLength = np.array(self._corners[(counter + 1) % 4]) - np.array(corner)
            # Get the edge.
            givenEdge = self._splitShapeEdges[counter]
            counter += 1
            # Distance of the point from the end.
            givenEdge = [point - np.array(corner) for point in givenEdge]

            # The x distance from the edge.
            distanceEdge = [distance_from_line(edgeLength, point) for point in givenEdge]
            rSquared = [point[0] ** 2 + point[1] ** 2 for point in givenEdge]
            normedXS = []
            # The y distance from the edge.
            for i in range(len(distanceEdge)):
                normedXS.append(np.sqrt(rSquared[i] - distanceEdge[i] ** 2))

            middle = 200
            shape = (int(max(normedXS)) + 1, 2 * middle + 1)

            mask = np.zeros(shape)
            for y, x in zip(normedXS, distanceEdge):
                mask[int(y)][int(x) + middle] = 255
                mask = mask.astype(np.uint8)
            kernel = np.ones((4, 4), np.uint8)
            # Need to dilate the edge to ensure flood fill only fills,
            # the left half.
            mask = cv2.dilate(mask, kernel)
            leftFill = mask.copy()
            h, w = mask.shape[:2]
            mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(leftFill, mask, (0, 0), 255)
            leftFill = cv2.erode(leftFill, kernel)
            # Make the edge matrix a binary image. for future comparison.
            leftFill[leftFill > 1] = 1

            saveResult("Example_Edge_Matrix_Edge_%d.png" % indx, leftFill * 255)

            indx += 1
            edgeMatrix.append(leftFill.astype(np.uint8))

        return edgeMatrix

    def create_colour_vector(self):
        color_vectors = []

        # sort edges by curve
        for color_edge in self._splitColourEdges:
            color_edges_curve = self.make_curve(np.array(color_edge))

            x_s = [edge[1] for edge in color_edges_curve]
            y_s = [edge[0] for edge in color_edges_curve]
            values = self._originalRGBPiece[(x_s, y_s)]
            color_vectors.append(values)

        return color_vectors

    def make_curve(self, cord_array):
        xmax = np.max(cord_array[:, 0])
        ymax = np.max(cord_array[:, 1])
        cord_matrix = np.zeros((xmax + 1, ymax + 1))

        for cord in cord_array:
            cord_matrix[cord[0], cord[1]] = 1

        cord = cord_array[0]
        cnt = 0
        results = np.zeros(cord_array.shape)
        while cnt + 1 < len(cord_array):
            cord_matrix[cord[0], cord[1]] = 0
            min = 10000
            mincord = None

            radius = 1
            while True:
                for i in range(cord[0] - radius, cord[0] + radius + 1):
                    for j in range(cord[1] - radius, cord[1] + radius + 1):
                        if not (i < 0 or i >= len(cord_matrix) or j < 0 or j >= len(cord_matrix[0])):
                            if cord_matrix[i, j] == 1:
                                diq = np.linalg.norm(cord - np.array([i, j]))
                                if diq < min:
                                    min = diq
                                    mincord = np.array([i, j])

                if mincord is None:
                    radius *= 2
                else:
                    break
            results[cnt, :] = cord
            cord = mincord
            cnt += 1
        return results.astype(dtype=np.int)

    def _puzzle_edges(self):
        """
        Identify if the piece has a straight edge and what other edges it has.
        Input: Edge matrix
        Output: puzzle edge score
         -1 is a slot, 0 is an edge, 1 is a tab
         [-1,0,1]
        """
        puzzle_edges = []
        pieceEdgeArray = []
        # Get all the edges of the piece
        for edgeMatrix in self._binaryEdgeImage:
            frame = np.zeros(edgeMatrix.shape, np.uint8)

            # Used to check if their is an indent.
            leftCheck = np.ones((edgeMatrix.shape[0], edgeMatrix.shape[1] // 2), np.uint8)
            # Used to check if their is an tab.
            rightCheck = np.zeros((edgeMatrix.shape[0], edgeMatrix.shape[1] // 2 + 1), np.uint8)
            # Used to check if their is an edge.
            frame[:, :edgeMatrix.shape[1] // 2] = 1

            # Get the amount of non overlapping area for the slot.
            XORSlot = np.bitwise_xor(leftCheck, edgeMatrix[:, :edgeMatrix.shape[1] // 2])
            # Get the amount of non overlapping area for the tab.
            XORTabs = np.bitwise_xor(rightCheck, edgeMatrix[:, edgeMatrix.shape[1] // 2:])
            XOREdge = np.bitwise_xor(frame, edgeMatrix)

            # Add up all the points that differ.
            scoreSlot = np.sum(XORSlot)
            scoreTab = np.sum(XORTabs)
            scoreEdge = np.sum(XOREdge)

            puzzle_edges.append(scoreEdge < frame.shape[0] * 8)
            # The largest score indicates the type of edge.
            if scoreEdge < frame.shape[0] * 8:
                pieceEdgeArray.append(0)
            elif scoreSlot > scoreTab:
                pieceEdgeArray.append(-1)
            else:
                pieceEdgeArray.append(1)

        return puzzle_edges, pieceEdgeArray

    def _setPieceType(self):
        """
        Check is the piece is a corner piece or an edge.
        Input: Piece edge Indication array.
        Output: Corner or edge check
        """
        edgeCounter = 0
        for i in self._pieceEdgeArray:
            if i == 0:
                edgeCounter += 1
        return edgeCounter == 2, edgeCounter == 1

    def compare_edge_to_piece(self, idx1, other, method):
        """
        Compare the edges of the target piece with the edges of the supply.
        Only get the score of piece edges that can connect like tab and slot.
        Input: The target edge and supplied pieces.
        Output: List of scores for each edge.
        """

        scores = []
        # For each edge of the second piece.

        for idx2 in range(len(self._pieceEdgeArray)):
            # Only get fitness score of the edges that can fit like a tab and slot. Saves time.
            if (self._pieceEdgeArray[idx1] == -1 and other._pieceEdgeArray[idx2] == 1) or (
                    self._pieceEdgeArray[idx1] == 1 and other._pieceEdgeArray[idx2] == -1):
                score = 0

                if method == 0:
                    score = compare_piece_edges(self._binaryEdgeImage[idx1], other._binaryEdgeImage[idx2])

                elif method == 1:
                    score = self.compare_edges_colour(self._color_vectors[idx1], other._color_vectors[idx2])
                scores.append((idx2, score))
        scores.sort(key=lambda x: x[1])
        return scores

    def compare_edges_colour(self, color_vector_1, color_vector_2):
        # flip color vector 2
        color_vector_2 = cv2.flip(color_vector_2, 0)
        '''
        :param edge1: (N, 1, 3) array of RGB colors
        :param edge2: (M, 1, 3) array of RGB colors
        :return: best correlation in offset window
        '''
        edge1 = np.reshape(color_vector_1, (color_vector_1.shape[0], 1, color_vector_1.shape[1]))
        edge1 = cv2.cvtColor(edge1, cv2.COLOR_BGR2Lab)
        edge2 = np.reshape(color_vector_2, (color_vector_2.shape[0], 1, color_vector_2.shape[1]))
        edge2 = cv2.cvtColor(edge2, cv2.COLOR_BGR2Lab)
        # make edge1 the longer
        if color_vector_1.shape[0] < color_vector_2.shape[0]:
            edge1, edge2 = edge2, edge1
        # cut off the lightness
        # edge1 = edge1[:, :, 1:]
        # edge2 = edge2[:, :, 1:]

        # normalize and centerize (mean = 0) edges before CC
        nmedge1 = (edge1 - np.mean(edge1, axis=(0, 1), keepdims=True)) / \
                  np.linalg.norm(edge1 - np.mean(edge1, axis=(0, 1), keepdims=True),
                                 axis=(0, 1), keepdims=True)
        nmedge2 = (edge2 - np.mean(edge2, axis=(0, 1), keepdims=True)) / \
                  np.linalg.norm(edge2 - np.mean(edge2, axis=(0, 1), keepdims=True),
                                 axis=(0, 1), keepdims=True)
        # cross correlate
        cc = correlate(nmedge1, nmedge2, 'valid')
        cc_tot = cc.reshape(cc.shape[0])
        score1mean = np.mean(cc_tot)
        # score1max = np.max(cc_tot)  # if len(cc_tot) > 1 else cc_tot[0]
        return -score1mean

    def display_real_piece(self):
        big_pic = np.zeros((12, 12, 3)).astype(dtype=np.uint8)
        r_y, r_x = self._relativePosition
        general = self._originalRGBPiece.copy()

        big_pic[r_x: r_x + general.shape[0], r_y: r_y + general.shape[1]] = general
        return big_pic
