import json
import os


class Constants:

    def __init__(self):
        self.basePath = os.path.abspath("Resources")
        self.resultBasePath = os.path.abspath("Results")
        self.TESTING = True
        self.stages = []
        self.stageFiles = []
        self.readParameters()

    def writeParameters(self):
        # Config parameters
        data = {"FULLDISPLAY": self.FULLDISPLAY,
                "MINIMUMDISPLAY": self.MINIMUMDISPLAY,
                "PIECEDISPLAY": self.PIECEDISPLAY,
                "METHOD": self.METHOD,
                "MASK": self.MASK_TRESHOLD,
                "AREA": self.CONTOUR_AREA_THRESHOLD,
                "PIECEMARGIN": self.PIECE_MARGIN,
                "CORNERANGLE": self.CORNER_ANGLE_TRESHHOLD,
                "RGBIMAGE": self.RGB_IMAGE,
                "GRAYIMAGE": self.GRAY_IMAGE}
        variables = open("Configuration/data.json", "w")
        json.dump(data, variables, indent=4)
        variables.close()

    def readParameters(self):
        # Config parameters
        json_file = open("Configuration/data.json")
        variables = json.load(json_file)
        self.FULLDISPLAY = variables["FULLDISPLAY"]
        self.MINIMUMDISPLAY = variables["MINIMUMDISPLAY"]
        self.PIECEDISPLAY = variables["PIECEDISPLAY"]
        self.METHOD = variables["METHOD"]
        self.MASK_TRESHOLD = variables["MASK"]
        self.CONTOUR_AREA_THRESHOLD = variables["AREA"]
        self.PIECE_MARGIN = variables["PIECEMARGIN"]
        self.CORNER_ANGLE_TRESHHOLD = variables["CORNERANGLE"]
        self.RGB_IMAGE = variables["RGBIMAGE"]
        self.GRAY_IMAGE = variables["GRAYIMAGE"]
        json_file.close()
