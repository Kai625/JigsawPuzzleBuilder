import PySimpleGUI as sg


class Config_window:
    def __init__(self, givenTheme, configObject):
        sg.theme(givenTheme)
        self.configObject = configObject
        self.configLayout = self.setup_config_layout()

    def setup_config_layout(self):
        configLayout = [[sg.Text("Configuration parameters", justification="center", size=(int(400 / 8), 1),
                                 font="Comic 16 bold")],
                        [sg.HorizontalSeparator()],
                        [sg.T("Mask threshold: ", pad=(20, 0), font="Comic 12 bold", size=(14, 1)),
                         sg.Input(default_text=self.configObject.MASK_TRESHOLD, key="INPUT-MASK", size=(5, 1))],
                        [sg.T("Area threshold: ", pad=(20, 0), font="Comic 12 bold", size=(14, 1)),
                         sg.Input(default_text=self.configObject.CONTOUR_AREA_THRESHOLD, key="INPUT-AREA",
                                  size=(5, 1))],
                        [sg.T("Corner angle: ", pad=(20, 0), font="Comic 12 bold", size=(14, 1)),
                         sg.Input(default_text=self.configObject.CORNER_ANGLE_TRESHHOLD, key="INPUT-CORNERANGLE",
                                  size=(5, 1))],
                        [sg.T("RGB image: ", pad=(20, 0), font="Comic 12 bold", size=(14, 1)),
                         sg.Input(default_text=self.configObject.RGB_IMAGE, key="INPUT-RGBIMAGE",
                                  size=(30, 1))],
                        [sg.T("Gray image: ", pad=(20, 0), font="Comic 12 bold", size=(14, 1)),
                         sg.Input(default_text=self.configObject.GRAY_IMAGE, key="INPUT-GRAYIMAGE",
                                  size=(30, 1))],
                        [sg.Button("SUBMIT", key="BUTTON-SUBMITGONFIG", enable_events=True, pad=(10, 0),
                                   font="Comic 8 bold",
                                   size=(10, 1))],
                        [sg.Button("RETURN", key="BUTTON-RETURNMAIN", enable_events=True, pad=(10, 0),
                                   font="Comic 8 bold",
                                   size=(10, 1))]]
        return configLayout

    def getWindow(self):
        layout = self.setup_config_layout()
        window = sg.Window('Window Title', layout=layout, size=(400, 600), margins=(10, 10))
        return window
