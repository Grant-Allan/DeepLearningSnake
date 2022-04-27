from game import StartMenu
from pygame import quit as pyg_quit


if __name__ == "__main__":
    start_menu = StartMenu()
    start_menu.main_menu()
    pyg_quit()
