from GUI.main_menu import Main_menu
from GUI.window_handler import Window_handler

if __name__ == '__main__':
    theme = "DarkAmber"
    mainMenu = Main_menu(theme)
    Window_handler(mainMenu, theme)
