"""
Author:         Mr. H. van der Westhuizen
Date opened:    11 September 2021
Student Number: u18141235
Project number: HG1

The main menu is used as the entry point for the program and the GUI frames.
The GUI is built using PySimpleGUI
"""
import PySimpleGUI as sg


class Main_menu:
    def __init__(self, givenTheme):
        sg.theme(givenTheme)
        self.mainMenuWindow = self.setup_main_menu()

    def setup_main_menu(self):
        mainMenuLayout = [
            [sg.Text("Jigsaw puzzle building robot", justification="center", size=(int(400 / 8), 1),
                     font="Comic 16 bold")],
            [sg.Text("ERP 420", justification="center", size=(int(400 / 8), 1),
                     font="Comic 12 bold")],
            [sg.Text("Project: HG1", justification="center", size=(int(400 / 8), 1),
                     font="Comic 12 bold")],
            [sg.T(" ")],
            [sg.Button("Start", "center", key="BUTTON-START", enable_events=True, size=(10, 1),
                       pad=(160, 0), font="Comic 8 bold")],
            [sg.Checkbox("Step through", key="CHECK-STEP", enable_events=True, pad=(140, 0), font="Comic 8 bold")],
            [sg.T(" ")],
            [sg.Button("Controller", key="BUTTON-CONTROL", enable_events=True, pad=(0, 0), font="Comic 8 bold",
                       size=(10, 1)),
             sg.Button("Test", key="BUTTON-TEST", enable_events=True, pad=(78, 0), font="Comic 8 bold", size=(10, 1)),
             sg.Button("Config", key="BUTTON-CONFIG", enable_events=True, pad=(0, 0), font="Comic 8 bold",
                       size=(10, 1))]]

        window = sg.Window('Window Title', layout=mainMenuLayout, size=(400, 235), margins=(10, 10))
        return window

    def getWindow(self):
        return self.mainMenuWindow
