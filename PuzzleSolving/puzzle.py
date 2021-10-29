"""
Author:         Mr. H. van der Westhuizen
Date opened:    13 September 2021
Student Number: u18141235
Project number: HG1

The following file hold the Puzzle object and is used to do all the
building, virtually.
"""
import copy
import cv2
from constant_parameters import *
import numpy as np
import operator


class Puzzle(object):

    def __init__(self, pieces, constObj):
        self.constObj = constObj
        self._method = self.constObj.METHOD
        self._pieces = pieces
        self._cornerPieces = [piece for piece in pieces if piece.is_puzzle_corner()]
        if len(self._cornerPieces) != 4:
            self.solvable = False
            return None
        self.solvable = True
        self._edgesPieces = [piece for piece in pieces if piece.is_puzzle_edge()]
        self._regularPieces = [
            piece for piece in pieces
            if not piece.is_puzzle_edge() and not piece.is_puzzle_corner()
        ]
        self._width, self._height = self.get_dimensions()

        print("The puzzle is of dimensions %d x %d" % (int(self._width), int(self._height)))

        # Holds the puzzle matrix.
        self._finalPuzzle = []
        self._connectedPuzzle = []

    def get_dimensions(self):
        # Use the number of edges to determine the puzzle size.
        total_edges = len(self._edgesPieces) + 2 * len(self._cornerPieces)

        p4 = total_edges / 4
        delta = np.sqrt(p4 ** 2 - len(self._pieces))

        return p4 + delta, p4 - delta

    def init_puzzle_builder(self):
        """
        The puzzle building process starts at the puzzle's corners.
        If one corner fails it tries the other corner until it
        has tested all four.
        """
        corners = self._cornerPieces
        for corner in corners[0:]:
            self._finalPuzzle = []
            if self.build_new_puzzle(corner):
                break
            else:
                print("Trying different corner!!!!!!!!!!!!!!!")
        for i in self._finalPuzzle:
            for j in i:
                print(j[0])

    def build_new_puzzle(self, startPiece):
        """
        This is the umbrella method that connects the rows, and new row pieces.
        It returns true if the puzzle is complete.
        Input: Corner piece
        Output: Completed puzzle checker
        """
        # Get new set of pieces for the current puzzle.
        remainingPieces = copy.copy(self._pieces)
        rowCounter = 0

        # Start from the corner and build the first row.
        # Get index of the connection points.
        pieceEdges = startPiece.get_piece_edge_indices()
        pieceConnectors = startPiece.get_piece_connectors_indices()
        if pieceConnectors[0] == 0 and pieceConnectors[1] == 3:
            pieceConnectors.reverse()

        # Get the edge that will be used to add new piece to the row.
        rowEdgeConnection = pieceConnectors.pop(0)
        # Get the edge that will connect to the next row.
        newRowEdgeConnection = pieceConnectors.pop(0)

        # Start the puzzle building with the corner (row by row)
        currentRow, isFilled, failedConnection = self.fill_first_row(remainingPieces, startPiece, rowEdgeConnection,
                                                                     newRowEdgeConnection, True, False, None, None)

        # Check if the row is completed and if the connection is valid.
        if (len(currentRow) == self._width or len(currentRow) == self._height) and not failedConnection:
            print("Row complete")
        else:
            return False

        # Append the row to the puzzle.
        self._finalPuzzle.append(currentRow)
        rowLength = len(currentRow)

        while len(remainingPieces) > 0:
            lastRow = len(remainingPieces) <= rowLength

            # The last row must start with corner.
            if lastRow:
                suppliedPieces = [p for p in remainingPieces if p.is_puzzle_corner()]
            else:
                suppliedPieces = [p for p in remainingPieces if p.is_puzzle_edge()]

            if suppliedPieces is None:
                return False

            print("Moving to next row: Finding Partner for: %s edge %d" % (
                self._finalPuzzle[rowCounter][0][0],
                self._finalPuzzle[rowCounter][0][2]))

            firstRowPiece, firstRowPieceEdge = self.get_new_row_start_piece(self._finalPuzzle[rowCounter][0][0],
                                                                            self._finalPuzzle[rowCounter][0][2],
                                                                            suppliedPieces)
            if firstRowPiece is None:
                return False

            currentRow, stopped, failedConnection = self.fill_first_row(remainingPieces, firstRowPiece,
                                                                        firstRowPieceEdge,
                                                                        newRowEdgeConnection, False, lastRow,
                                                                        self._finalPuzzle[rowCounter], rowLength)

            if (len(currentRow) == self._width or len(currentRow) == self._height) and not failedConnection:
                print("Row complete")
            else:
                return False

            self._finalPuzzle.append(currentRow)
            rowCounter += 1
            if len(remainingPieces) == 0:
                return True

    def fill_first_row(self, remainingPieces, firstPiece, rowEdgeConnection, newRowEdgeConnection, firstRow, lastRow,
                       prevRow, rowLength):
        # Row uses the piece, the edge that connects the other piece in the row
        # the edge that connects to the next row's piece. The First row has no edge connected to prev row.
        # BOT, LEFT, TOP, RIGHT
        if firstRow:
            row = [(firstPiece, rowEdgeConnection, newRowEdgeConnection, None, None)]
        elif lastRow:
            row = [(firstPiece, (rowEdgeConnection + 1) % 4, None, None, rowEdgeConnection)]
            rowEdgeConnection = (rowEdgeConnection + 1) % 4
        else:
            row = [(firstPiece, (rowEdgeConnection + 1) % 4, (rowEdgeConnection + 2) % 4, None, rowEdgeConnection)]
            rowEdgeConnection = (rowEdgeConnection + 1) % 4

        # Remove the starting row piece.
        remainingPieces.remove(firstPiece)
        rowPosition = 1

        # Keep track of the current and previous piece.
        currentPiece = firstPiece
        currentEdge = rowEdgeConnection
        prevPiece = firstPiece
        prevEdge = rowEdgeConnection

        continueRow = True
        while continueRow:
            # The score holder for both the shape and colour rankings.
            methodScoredPieces = []

            if firstRow or lastRow:
                # Possible pieces that can connect
                suppliedPieces = [p for p in remainingPieces if p.is_puzzle_corner() or p.is_puzzle_edge()]
            else:
                suppliedPieces = [p for p in remainingPieces if not p.is_puzzle_corner()]

            if firstRow:
                # The first row only checks the connetions between the row connecting edges.
                for m in self.constObj.METHOD:
                    # Get the score of the connection, either by edge or colour.
                    scoredPieces = [(x, currentPiece.compare_edge_to_piece(currentEdge, x, m)) for x in suppliedPieces]
                    # Remove any scoredPieces that can not connect
                    scoredPieces = [p for p in scoredPieces if len(p[1]) > 0]

                    # Reformat the score for easy comparison.
                    scoredPieces = [(piece, edge, score) for piece, data in scoredPieces for edge, score in data]
                    # Append the two method's scores.
                    methodScoredPieces.append(scoredPieces)
            else:
                methodScoredPieces = []
                scoredPiecesTogether = []
                for m in self.constObj.METHOD:
                    scoredPiecesBot = [(x, currentPiece.compare_edge_to_piece(currentEdge, x, m)) for x in
                                       suppliedPieces]
                    scoredPiecesRight = [
                        (x, prevRow[rowPosition][0].compare_edge_to_piece(prevRow[rowPosition][2], x, m))
                        for x in suppliedPieces]
                    scoredPiecesBot = [p for p in scoredPiecesBot if len(p[1]) > 0]
                    scoredPiecesRight = [p for p in scoredPiecesRight if len(p[1]) > 0]
                    scoredPiecesTogether = []
                    for i in range(len(scoredPiecesBot)):
                        piece1, data1 = scoredPiecesBot[i]
                        piece2, data2 = scoredPiecesRight[i]
                        if piece1 == piece2:
                            # create new dataset
                            for edge1, score1 in data1:
                                find_edge = (edge1 + 1) % 4
                                nexts = [(edge2, score2) for edge2, score2 in data2 if edge2 == find_edge]
                                if len(nexts):
                                    scoredPiecesTogether.append((piece1, edge1, score1 + nexts[0][1]))
                    methodScoredPieces.append(scoredPiecesTogether)
            if len(methodScoredPieces[0]) == 0:
                return row, True, True
            # Chose the edge with the best score
            newPiece, newPieceConnectingEdge, score = self.choose_best_method(methodScoredPieces)

            # Indicate the edge that will connect to the next row's pieces.
            newPieceNewRowConnection = (newPieceConnectingEdge + 3) % 4
            if not lastRow:
                if newPiece.get_piece_edge_array()[newPieceNewRowConnection] == 0:
                    return row, True, True

            print("Found Piece! %s is connected to %s by edges (%d, %d)" % (
                prevPiece,
                newPiece,
                prevEdge,
                newPieceConnectingEdge
            ))
            if firstRow:
                row.append((
                    newPiece, (newPieceConnectingEdge + 2) % 4, newPieceNewRowConnection, newPieceConnectingEdge,
                    None))
            elif lastRow:
                row.append((
                    newPiece, (newPieceConnectingEdge + 2) % 4, None, newPieceConnectingEdge,
                    (newPieceConnectingEdge + 1) % 4))
            else:
                row.append((
                    newPiece, (newPieceConnectingEdge + 2) % 4, (newPieceConnectingEdge + 3) % 4,
                    newPieceConnectingEdge,
                    (newPieceConnectingEdge + 1) % 4))

            remainingPieces.remove(newPiece)

            # Move to the next piece
            prevPiece = newPiece
            prevEdge = (newPieceConnectingEdge + 2) % 4
            currentPiece = newPiece
            currentEdge = (newPieceConnectingEdge + 2) % 4
            rowPosition += 1

            if currentPiece.is_puzzle_corner() or (
                    not firstRow and not lastRow and currentPiece.is_puzzle_edge()):
                continueRow = False
        return row, True, False

    def choose_best_method(self, bestPiecesMethods):
        """
        The method compares the scores of all the edges and compares the rankings.
        All the scores are ranked and the rank of the edge and colour ranks are added
        per edge.
        """
        ranking = []
        # Sort according to score for each method.
        for i, bestPieces in enumerate(bestPiecesMethods):
            k = 0
            ranking.append([])
            bestPieces.sort(key=lambda x: x[2])
            # Rank from best score till worst.
            for j, pieceData in enumerate(bestPieces):
                if i == 0 and j != 0:
                    if pieceData[2] - prevData[2] >= 500:
                        k += 1
                elif i == 1 and j != 0:
                    if abs(pieceData[2] - prevData[2]) >= 0.5:
                        k += 1
                ranking[i].append((pieceData[0], pieceData[1], j + k))
                prevData = pieceData

        # Rearrange the rank set to ensure the edges are aligned.
        for i in range(len(ranking)):
            ranking[i] = sorted(ranking[i], key=lambda x: (x[0].get_ID(), x[1]))

        # Add the score of the two methods together, might use weights.
        final_ranking = []
        for i in range(len(ranking[0])):
            score = sum([rank[i][2] for rank in ranking])
            final_ranking.append((ranking[0][i][0], ranking[0][i][1], score))

        # Return the piece and connecting edge of the best option.
        return min(final_ranking, key=lambda x: x[2])

    def get_new_row_start_piece(self, prevRowPiece, connectingEdge, suppliedPieces):
        """
        This method is very similar to the row completion score system.
        """
        methodScoredPieces = []
        for m in self.constObj.METHOD:
            scoredPieces = [(x, prevRowPiece.compare_edge_to_piece(connectingEdge, x, m)) for x in suppliedPieces]
            scoredPieces = [p for p in scoredPieces if len(p[1]) > 0]

            scoredPieces = [(piece, edge, score) for piece, data in scoredPieces for edge, score in data]
            methodScoredPieces.append(scoredPieces)
        if len(methodScoredPieces[0]) == 0:
            return None, None
        newPiece, newPieceConnectingEdge, score = self.choose_best_method(methodScoredPieces)
        return newPiece, newPieceConnectingEdge

    def connect(self):
        displayPieces = []
        for i, row in enumerate(self._finalPuzzle):
            displayPieces.append([])
            for p, e, _, _, _ in row:
                displayPieces[i].append((p, (e + 2) % 4))
        self._connectedPuzzle, connectedImages = connect_puzzle.get_solved_puzzle_img(displayPieces)

        for index, image in enumerate(connectedImages):
            cv2.imshow("big pic", cv2.resize(image, (0, 0), fx=0.5, fy=0.5))
            cv2.imwrite(".\\image_processing\\results\\videos\\%d.png" % index, image)
            cv2.waitKey(100)
        cv2.waitKey(0)

        # print data from the connected puzzle
        img = connectedImages[-1]
        for row in self._connectedPuzzle:
            for piece, center, angle in row:
                cv2.circle(img, (int(center[0]), int(center[1])), 5, (255, 0, 0), -1)
        cv2.imshow("mat", img)
        cv2.waitKey(0)
