import os

import PySimpleGUI as sg

from GUI.config_window import Config_window
from GUI.test_window import Test_window
from PuzzleSolving.basic_image_handler import get_test_images, create_mask, apply_mask_rgb, saveResult
from PuzzleSolving.piece_builder import piece_extraction
from constant_parameters import Constants


class Window_handler:
    def __init__(self, mainMenu, givenTheme):
        sg.theme(givenTheme)
        self.mainMenuObj = mainMenu
        self.mainMenuWindow = mainMenu.getWindow()
        self.currentWindow = mainMenu.getWindow()
        self.constant_parameters = Constants()
        self.configObj = Config_window("DarkAmber", self.constant_parameters)
        self.display_window()
        self.RGBImage = None
        self.grayImage = None
        self.maskedImage = None
        self.binaryMatrix = None

    def showConfigWindow(self):
        configWindow = self.configObj.getWindow()
        while True:
            configEvents, configValues = configWindow.read()
            if configEvents == 'BUTTON-RETURNMAIN' or configEvents == sg.WIN_CLOSED:
                configWindow.close()
                self.mainMenuWindow.UnHide()
                break
            elif configEvents == "BUTTON-SUBMITGONFIG":
                self.constant_parameters.MASK_TRESHOLD = int(configWindow['INPUT-MASK'].get())
                self.constant_parameters.CONTOUR_AREA_THRESHOLD = int(configWindow['INPUT-AREA'].get())
                self.constant_parameters.CORNER_ANGLE_TRESHHOLD = float(configWindow['INPUT-CORNERANGLE'].get())
                self.constant_parameters.RGB_IMAGE = configWindow['INPUT-RGBIMAGE'].get()
                self.constant_parameters.GRAY_IMAGE = configWindow['INPUT-GRAYIMAGE'].get()
                self.constant_parameters.writeParameters()

    def display_window(self):
        configActive = False
        while True:
            mainEvents, mainValues = self.mainMenuWindow.read(timeout=100)
            if mainEvents == sg.WIN_CLOSED:  # if user closes window or clicks cancel
                break
            elif mainEvents == "BUTTON-CONFIG":
                self.mainMenuWindow.Hide()
                self.showConfigWindow()
            elif mainEvents == "BUTTON-TEST":
                self.mainMenuWindow.Hide()
                self.showTestWindow()

    def addStageFiles(self):
        if os.path.exists(self.constant_parameters.resultBasePath):
            files = os.listdir(self.constant_parameters.resultBasePath)
            newFilesAdded = list(filter(lambda x: "Position" in x, files))
            self.constant_parameters.stageFiles.append(newFilesAdded)
            newFilesAdded = list(filter(lambda x: "Both_Edge" in x, files))
            self.constant_parameters.stageFiles.append(newFilesAdded)
            newFilesAdded = list(filter(lambda x: "Peaks" in x, files))
            self.constant_parameters.stageFiles.append(newFilesAdded)
            newFilesAdded = list(filter(lambda x: "Corners" in x, files))
            self.constant_parameters.stageFiles.append(newFilesAdded)
            newFilesAdded = list(filter(lambda x: "Divided" in x, files))
            self.constant_parameters.stageFiles.append(newFilesAdded)
            newFilesAdded = list(filter(lambda x: "Example" in x, files))
            self.constant_parameters.stageFiles.append(newFilesAdded)
            newFilesAdded = list(filter(lambda x: "Characterised" in x, files))
            self.constant_parameters.stageFiles.append(newFilesAdded)

    def showTestWindow(self):
        self.mainMenuWindow.close()
        counter = 1
        stages = []
        testWindow = Test_window("DarkAmber", self.constant_parameters).getFirstWindow()
        while True:
            testEvent, testValues = testWindow.read()
            if testEvent == 'BUTTON-RETURNMAIN' or testEvent == sg.WIN_CLOSED:
                testWindow.close()
                break
            elif testEvent == "BUTTON-NEXT":
                if counter == 1:
                    self.RGBImage, self.grayImage = get_test_images(self.constant_parameters)
                    self.constant_parameters.stages.append(str(counter) + " Input Image")
                    self.constant_parameters.stageFiles.append(["RGBImage.png", "GrayImage.png"])
                    testWindow["LIST-FILES"].update(
                        self.constant_parameters.stageFiles[0])
                    testWindow["LIST-STAGES"].update(self.constant_parameters.stages)
                if counter == 2:
                    # Create the mask that removes the background and reduces the noise.
                    self.grayImage, self.binaryMatrix = create_mask(self.RGBImage, self.constant_parameters)

                    # Use binary matrix to apply the mask to the RGB image.
                    self.maskedImage = apply_mask_rgb(self.RGBImage, self.binaryMatrix)
                    saveResult("Mask.png", self.grayImage)
                    saveResult("MaskedRGBImage.png", self.maskedImage)
                    self.constant_parameters.stageFiles.append(["Mask.png", "MaskedRGBImage.png"])
                    self.constant_parameters.stages.append(str(counter) + " Apply mask")
                    testWindow["LIST-STAGES"].update(self.constant_parameters.stages)
                if counter == 3:
                    puzzleCharacterised, pieces = piece_extraction(self.maskedImage, self.grayImage, self.binaryMatrix,
                                                                   self.constant_parameters)
                    saveResult("Characterised_Pieces.png", puzzleCharacterised)
                    self.constant_parameters.stages.append(str(4) + " Piece centroid and pickup location")
                    self.constant_parameters.stages.append(str(5) + " Piece edges")
                    self.constant_parameters.stages.append(str(6) + " Corner peaks")
                    self.constant_parameters.stages.append(str(7) + " Piece corners")
                    self.constant_parameters.stages.append(str(8) + " Divided edges")
                    self.constant_parameters.stages.append(str(9) + " Example edge matrix")
                    self.constant_parameters.stages.append(str(10) + " Fully divined frame")
                    testWindow["LIST-STAGES"].update(self.constant_parameters.stages)
                    self.addStageFiles()
                    counter = 10

                counter += 1

            elif testEvent == "LIST-FILES" and len(testWindow["LIST-FILES"].get()) != 0:
                testWindow["IMAGE-SHOWN"].update(
                    self.constant_parameters.resultBasePath + "\\" + testWindow["LIST-FILES"].get()[0])

            elif testEvent == "LIST-STAGES" and len(testWindow["LIST-STAGES"].get()) != 0:
                stageIndex = self.constant_parameters.stages.index(testWindow["LIST-STAGES"].get()[0])
                testWindow["LIST-FILES"].update(self.constant_parameters.stageFiles[stageIndex])
                testWindow["TEXT-CURRENT"].update(
                    "Current stage: " + testWindow["LIST-STAGES"].get()[0][2:])
