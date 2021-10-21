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
                    self.grayImage, self.binaryMatrix = create_mask(self.grayImage, self.constant_parameters)

                    # Use binary matrix to apply the mask to the RGB image.
                    self.maskedImage = apply_mask_rgb(self.RGBImage, self.binaryMatrix)
                    saveResult("Mask.png", self.grayImage)
                    saveResult("MaskedRGBImage.png", self.maskedImage)
                    self.constant_parameters.stageFiles.append(["Mask.png", "MaskedRGBImage.png"])
                    self.constant_parameters.stages.append(str(counter) + " Apply mask")
                    testWindow["LIST-STAGES"].update(self.constant_parameters.stages)
                if counter == 3:
                    piece_extraction(self.maskedImage, self.grayImage, self.binaryMatrix,
                                     self.constant_parameters)
                    testWindow["LIST-STAGES"].update(self.constant_parameters.stages)
                counter += 1

            elif testEvent == "LIST-FILES" and len(testWindow["LIST-FILES"].get()) != 0:
                testWindow["IMAGE-SHOWN"].update(
                    self.constant_parameters.resultBasePath + "\\" + testWindow["LIST-FILES"].get()[0])

            elif testEvent == "LIST-STAGES" and len(testWindow["LIST-STAGES"].get()) != 0:
                stageIndex = self.constant_parameters.stages.index(testWindow["LIST-STAGES"].get()[0])
                testWindow["LIST-FILES"].update(self.constant_parameters.stageFiles[stageIndex])
