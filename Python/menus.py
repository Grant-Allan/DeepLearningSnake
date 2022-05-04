from game import BackgroundSnake
from run import RunGame
from helper import TILE_SIZE, WHITE, GRAY, BLACK, RED, GREEN2, GREEN3, WIDTH, HEIGHT, MARGIN

from pygame import RESIZABLE as pyg_RESIZABLE
from pygame import QUIT as pyg_QUIT
from pygame import quit as pyg_quit
from pygame import MOUSEBUTTONDOWN as pyg_MOUSEBUTTONDOWN
from pygame import KEYDOWN as pyg_KEYDOWN
from pygame import K_RETURN as pyg_K_RETURN
from pygame import K_BACKSPACE as pyg_K_BACKSPACE
from pygame import font as pyg_font
from pygame import display as pyg_display
from pygame.mouse import get_pos as pyg_mouse_get_pos
from pygame.event import get as pyg_get
from pygame.transform import scale as pyg_scale
from pygame.draw import rect as pyg_rect


# Initialize pygame modules as needed
pyg_font.init()

# Fonts
FONT_SIZE = int(TILE_SIZE*1.5)
TITLE_FONT_SIZE = int(TILE_SIZE*3)
try:
    FONT = pyg_font.Font("arial.ttf", FONT_SIZE)
    TITLE_FONT = pyg_font.Font("arial.ttf", TITLE_FONT_SIZE)
except:
    FONT = pyg_font.Font("arial", FONT_SIZE)
    TITLE_FONT = pyg_font.Font("arial", TITLE_FONT_SIZE)



class StartMenu():
    '''
    A main menu to open the game into, where you can select
    your game mode and change various settings.
    '''
    def __init__(self):
        # Initialize input data
        self.width = WIDTH
        self.height = HEIGHT
        self.margin = MARGIN

        # Initialze display
        self.true_display = pyg_display.set_mode((self.width, self.height+self.margin), pyg_RESIZABLE)
        self.false_display = self.true_display.copy()
        pyg_display.set_caption("Snake")

        # Set initial scale
        tw, th = self.true_display.get_size()
        fw, fh = self.false_display.get_size()
        self.w_scale = tw/fw
        self.h_scale = th/fh

        # Create background snake object
        self.bg_snake = BackgroundSnake(self.false_display)

        # Load menu values
        try:
            with open(r"./Resources/HumanGameSettings.txt", "r") as file:
                self.HumanGameSettings = file.readlines()
        except:
            self.HumanGameSettings = ["10"] # FPS
            with open(r"./Resources/HumanGameSettings.txt", "w") as file:
                file.write(''.join(self.HumanGameSettings))
        try:
            with open(r"./Resources/DQN_Settings.txt", "r") as file:
                self.DQN_Settings = file.readlines()
        except:
            self.DQN_Settings = ["100", "120"] # FPS, Episodes
            with open(r"./Resources/DQN_Settings.txt", "w") as file:
                file.write(''.join(self.DQN_Settings))
        try:
            with open(r"./Resources/DGA_Settings.txt", "r") as file:
                self.DGA_Settings = file.readlines()
        except:
            self.DGA_Settings = ["50", "20", "250"] # FPS, Agents, Generations
            with open(r"./Resources/DGA_Settings.txt", "w") as file:
                file.write(''.join(self.DGA_Settings))


    def main_menu(self):
        while True:
            # Update width and height scale
            tw, th = self.true_display.get_size()
            fw, fh = self.false_display.get_size()
            self.w_scale = tw/fw
            self.h_scale = th/fh

            # Get current mouse position
            mouse_pos = pyg_mouse_get_pos()

            # Set placement values
            # NGMB = New Game Menu Button
            NGMB_text = "New Game"
            NGMB_x_check, NGMB_y_check, NGMB_pos, NGMB_size = self.button_values(NGMB_text, 60, mouse_pos)

            # SMB = Settings Menu Button
            SMB_text = "Settings"
            SMB_x_check, SMB_y_check, SMB_pos, SMB_size = self.button_values(SMB_text, NGMB_pos[1], mouse_pos)

            # QB = Quit Button
            QB_text = "Quit"
            QB_x_check, QB_y_check, QB_pos, QB_size = self.button_values(QB_text, SMB_pos[1], mouse_pos)

            # Black out previous display
            self.false_display.fill(BLACK)

            # Run the background snake
            self.false_display = self.bg_snake.play_step()

            # Display title
            t_x, t_y = TITLE_FONT.size("Snake")
            text = TITLE_FONT.render("Snake", True, GREEN3)
            self.false_display.blit(text, [self.width//2 - t_x//2, t_y//3])

            # Draw buttons
            self.draw_button(NGMB_text, NGMB_x_check, NGMB_y_check, NGMB_pos, NGMB_size)
            self.draw_button(SMB_text, SMB_x_check, SMB_y_check, SMB_pos, SMB_size)
            self.draw_button(QB_text, QB_x_check, QB_y_check, QB_pos, QB_size)

            # Update display
            self.true_display.blit(pyg_scale(self.false_display, self.true_display.get_size()), (0, 0))
            pyg_display.flip()

            # Get player input
            for event in pyg_get():
                # Check for exiting out of window
                if event.type == pyg_QUIT:
                    pyg_quit()
                    quit()
                # Check for if a button is pressed
                elif event.type == pyg_MOUSEBUTTONDOWN:
                    # Enter game selection menu
                    if NGMB_x_check and NGMB_y_check:
                        self.game_type_selection_menu()
                    # Enter settings menu
                    elif SMB_x_check and SMB_y_check:
                        self.settings_menu()
                    # Quit
                    elif QB_x_check and QB_y_check:
                        pyg_quit()
                        quit()


    def game_type_selection_menu(self):
        '''
        Allow a person to select either playing as a human, having
        a single agent play, or a population of agents play.
        '''
        while True:
            # Update width and height scale
            tw, th = self.true_display.get_size()
            fw, fh = self.false_display.get_size()
            self.w_scale = tw/fw
            self.h_scale = th/fh
            
            # Get current mouse position
            mouse_pos = pyg_mouse_get_pos()

            # Set placement values
            # NG = Normal Game
            NG_text = "Normal Game"
            NG_x_check, NG_y_check, NG_pos, NG_size = self.button_values(NG_text, 60, mouse_pos)

            # DQN = Deep Q Learning
            DQN_text = "Deep Q Learning"
            DQN_x_check, DQN_y_check, DQN_pos, DQN_size = self.button_values(DQN_text, NG_pos[1], mouse_pos)

            # DGA = Deep Genetic Algorithm
            DGA_text = "Deep Genetic Algorithm"
            DGA_x_check, DGA_y_check, DGA_pos, DGA_size = self.button_values(DGA_text, DQN_pos[1], mouse_pos)

            # BB = Back Button
            BB_text = "Back"
            BB_x_check, BB_y_check, BB_pos, BB_size = self.button_values(BB_text, DGA_pos[1], mouse_pos)

            # Black out previous display
            self.false_display.fill(BLACK)

            # Run the background snake
            self.false_display = self.bg_snake.play_step()

            # Display menu title
            t_x, t_y = TITLE_FONT.size("Select Game Type")
            text = TITLE_FONT.render("Select Game Type", True, GREEN3)
            self.false_display.blit(text, [self.width//2 - t_x//2, t_y//3])

            # Draw buttons
            self.draw_button(NG_text, NG_x_check, NG_y_check, NG_pos, NG_size)
            self.draw_button(DQN_text, DQN_x_check, DQN_y_check, DQN_pos, DQN_size)
            self.draw_button(DGA_text, DGA_x_check, DGA_y_check, DGA_pos, DGA_size)
            self.draw_button(BB_text, BB_x_check, BB_y_check, BB_pos, BB_size)

            # Update display
            self.true_display.blit(pyg_scale(self.false_display, self.true_display.get_size()), (0, 0))
            pyg_display.flip()

            # Get player input
            for event in pyg_get():
                # Check for exiting out of window
                if event.type == pyg_QUIT:
                    pyg_quit()
                    quit()
                # Check for if a button is pressed
                elif event.type == pyg_MOUSEBUTTONDOWN:
                    # Start normal game
                    if NG_x_check and NG_y_check:
                        run_game = RunGame()
                        run_game.run_human()
                        self.main_menu()
                    # Start game with a single agent
                    elif DQN_x_check and DQN_y_check:
                        run_game = RunGame()
                        run_game.run_dqn()
                        self.main_menu()
                    # Start game with a population of agents
                    elif DGA_x_check and DGA_y_check:
                        run_game = RunGame()
                        run_game.run_dga()
                        self.main_menu()
                    # Back to main menu
                    elif BB_x_check and BB_y_check:
                        self.main_menu()


    def settings_menu(self):
        '''
        Allow a person to decide the frame rate, number of of episodes,
        population size, model values, etc.
        '''
        while True:
            # Update width and height scale
            tw, th = self.true_display.get_size()
            fw, fh = self.false_display.get_size()
            self.w_scale = tw/fw
            self.h_scale = th/fh
            
            # Get current mouse position
            mouse_pos = pyg_mouse_get_pos()

            # Set placement values
            # NG = Normal Game
            NG_text = "Human Game Settings"
            NG_x_check, NG_y_check, NG_pos, NG_size = self.button_values(NG_text, 60, mouse_pos)

            # DQN = Deep Q Learning
            DQN_text = "Deep Q Learning Settings"
            DQN_x_check, DQN_y_check, DQN_pos, DQN_size = self.button_values(DQN_text, NG_pos[1], mouse_pos)

            # DGA = Population of Agents
            DGA_text = "Deep Genetic Algorithm Settings"
            DGA_x_check, DGA_y_check, DGA_pos, DGA_size = self.button_values(DGA_text, DQN_pos[1], mouse_pos)

            # BB = Back Button
            BB_text = "Back"
            BB_x_check, BB_y_check, BB_pos, BB_size = self.button_values(BB_text, DGA_pos[1], mouse_pos)

            # Black out previous display
            self.false_display.fill(BLACK)

            # Run the background snake
            self.false_display = self.bg_snake.play_step()

            # Display menu title
            t_x, t_y = TITLE_FONT.size("Settings Menu")
            text = TITLE_FONT.render("Settings Menu", True, GREEN3)
            self.false_display.blit(text, [self.width//2 - t_x//2, t_y//3])

            # Draw buttons
            self.draw_button(NG_text, NG_x_check, NG_y_check, NG_pos, NG_size)
            self.draw_button(DQN_text, DQN_x_check, DQN_y_check, DQN_pos, DQN_size)
            self.draw_button(DGA_text, DGA_x_check, DGA_y_check, DGA_pos, DGA_size)
            self.draw_button(BB_text, BB_x_check, BB_y_check, BB_pos, BB_size)

            # Update display
            self.true_display.blit(pyg_scale(self.false_display, self.true_display.get_size()), (0, 0))
            pyg_display.flip()

            # Get player input
            for event in pyg_get():
                # Check for exiting out of window
                if event.type == pyg_QUIT:
                    pyg_quit()
                    quit()
                # Check for if a button is pressed
                elif event.type == pyg_MOUSEBUTTONDOWN:
                    # Start normal game
                    if NG_x_check and NG_y_check:
                        self.human_settings()
                    # Start game with a single agent
                    elif DQN_x_check and DQN_y_check:
                        self.DQL_settings()
                    # Start game with a population of agents
                    elif DGA_x_check and DGA_y_check:
                        self.DGA_settings()
                    # Back to main menu
                    elif BB_x_check and BB_y_check:
                        self.main_menu()

    
    def human_settings(self):
        ''' Set settings for playing as a human. '''
        active = "" # typing boolean
        input_text = "" # box text

        while True:
            # Update width and height scale
            tw, th = self.true_display.get_size()
            fw, fh = self.false_display.get_size()
            self.w_scale = tw/fw
            self.h_scale = th/fh
            
            # Black out previous display
            self.false_display.fill(BLACK)

            # Run the background snake
            self.false_display = self.bg_snake.play_step()

            # Display menu title
            t_x, t_y = TITLE_FONT.size("Normal Game Settings")
            text = TITLE_FONT.render("Normal Game Settings", True, GREEN3)
            self.false_display.blit(text, [self.width//2 - t_x//2, t_y//3])

            # Get current mouse position
            mouse_pos = pyg_mouse_get_pos()

            # Settings
            FPS_x_check, FPS_y_check, FPS_box_size = self.draw_option("HG", "FPS: ", 0, 45, "FPS", active, mouse_pos, input_text, center=True)

            # BB = Back Button
            BB_text = "Back"
            BB_x_check, BB_y_check, BB_pos, BB_size = self.button_values(BB_text, (self.height - self.margin - TILE_SIZE), mouse_pos)
            self.draw_button(BB_text, BB_x_check, BB_y_check, BB_pos, BB_size)

            # Update display
            self.true_display.blit(pyg_scale(self.false_display, self.true_display.get_size()), (0, 0))
            pyg_display.flip()

            # Get player input
            for event in pyg_get():
                # Check for exiting out of window
                if event.type == pyg_QUIT:
                    pyg_quit()
                    quit()
                # Check for if a button is pressed
                elif event.type == pyg_MOUSEBUTTONDOWN:
                    # Reset active
                    active = ""

                    # Back to main menu
                    if BB_x_check and BB_y_check:
                        self.settings_menu()

                    # Check to see if you clicked into a text box
                    if FPS_x_check and FPS_y_check:
                        active = "FPS"
                
                # Check for typing
                if active=="FPS":
                    if event.type == pyg_KEYDOWN:
                        if event.key == pyg_K_RETURN:
                            with open(r"./Resources/HumanGameSettings.txt", "w") as file:
                                self.HumanGameSettings[0] = input_text
                                file.write(''.join(self.HumanGameSettings))
                            input_text = ""
                            active = ""
                        elif event.key == pyg_K_BACKSPACE:
                            input_text = input_text[:-1]
                        else:
                            x, _ = FONT.size(input_text)
                            if x < FPS_box_size[0]:
                                input_text += event.unicode
                        self.HumanGameSettings[0] = input_text
    

    def DQL_settings(self):
        ''' Set settings for a deep q learning session. '''
        active = "" # typing check
        input_text = "" # box text

        while True:
            # Update width and height scale
            tw, th = self.true_display.get_size()
            fw, fh = self.false_display.get_size()
            self.w_scale = tw/fw
            self.h_scale = th/fh
            
            # Black out previous display
            self.false_display.fill(BLACK)

            # Run the background snake
            self.false_display = self.bg_snake.play_step()

            # Display menu title
            t_x, t_y = TITLE_FONT.size("DQL Settings")
            text = TITLE_FONT.render("DQL Settings", True, GREEN3)
            self.false_display.blit(text, [self.width//2 - t_x//2, t_y//3])

            # Get current mouse position
            mouse_pos = pyg_mouse_get_pos()

            # Settings
            FPS_x_check, FPS_y_check, FPS_box_size = self.draw_option("DQN", "FPS: ", 0, 65, "FPS", active, mouse_pos, input_text)
            EP_x_check, EP_y_check, EP_box_size = self.draw_option("DQN", "Episodes: ", 1, 65, "EP", active, mouse_pos, input_text)

            # BB = Back Button
            BB_text = "Back"
            BB_x_check, BB_y_check, BB_pos, BB_size = self.button_values(BB_text, (self.height - self.margin - TILE_SIZE), mouse_pos)
            self.draw_button(BB_text, BB_x_check, BB_y_check, BB_pos, BB_size)

            # Update display
            self.true_display.blit(pyg_scale(self.false_display, self.true_display.get_size()), (0, 0))
            pyg_display.flip()

            # Get player input
            for event in pyg_get():
                # Check for exiting out of window
                if event.type == pyg_QUIT:
                    pyg_quit()
                    quit()
                # Check for if a button is pressed
                elif event.type == pyg_MOUSEBUTTONDOWN:
                    # Reset active
                    active = ""

                    # Back to main menu
                    if BB_x_check and BB_y_check:
                        self.settings_menu()

                    # Check to see if you clicked into a text box
                    if FPS_x_check and FPS_y_check:
                        active = "FPS"
                    elif EP_x_check and EP_y_check:
                        active = "EP"
                
                # Check for typing
                if active=="FPS":
                    if event.type == pyg_KEYDOWN:
                        if event.key == pyg_K_RETURN:
                            with open(r"./Resources/DQN_Settings.txt", "w") as file:
                                self.DQN_Settings[0] = input_text
                                file.write(''.join(self.DQN_Settings))
                            input_text = ""
                            active = ""
                        elif event.key == pyg_K_BACKSPACE:
                            input_text = input_text[:-1]
                        else:
                            x, _ = FONT.size(input_text)
                            if x < FPS_box_size[0]:
                                input_text += event.unicode
                        self.DQN_Settings[0] = input_text
                elif active=="EP":
                    if event.type == pyg_KEYDOWN:
                        if event.key == pyg_K_RETURN:
                            with open(r"./Resources/DQN_Settings.txt", "w") as file:
                                self.DQN_Settings[1] = input_text
                                file.write(''.join(self.DQN_Settings))
                            input_text = ""
                            active = ""
                        elif event.key == pyg_K_BACKSPACE:
                            input_text = input_text[:-1]
                        else:
                            x, _ = FONT.size(input_text)
                            if x < EP_box_size[0]:
                                input_text += event.unicode
                        self.DQN_Settings[1] = input_text


    def DGA_settings(self):
        ''' Set settings for a depp genetic algorithm session. '''
        active = "" # typing check
        input_text = "" # box text

        while True:
            # Update width and height scale
            tw, th = self.true_display.get_size()
            fw, fh = self.false_display.get_size()
            self.w_scale = tw/fw
            self.h_scale = th/fh
            
            # Black out previous display
            self.false_display.fill(BLACK)

            # Run the background snake
            self.false_display = self.bg_snake.play_step()

            # Display menu title
            t_x, t_y = TITLE_FONT.size("DGA Settings")
            text = TITLE_FONT.render("DGA Settings", True, GREEN3)
            self.false_display.blit(text, [self.width//2 - t_x//2, t_y//3])

            # Get current mouse position
            mouse_pos = pyg_mouse_get_pos()

            # Settings
            FPS_x_check, FPS_y_check, FPS_box_size = self.draw_option("DGA", "FPS: ", 0, 65, "FPS", active, mouse_pos, input_text)
            POP_x_check, POP_y_check, POP_box_size = self.draw_option("DGA", "Population Size: ", 1, 65, "POP", active, mouse_pos, input_text)
            NoG_x_check, NoG_y_check, NoG_box_size = self.draw_option("DGA", "Number of Generations: ", 2, 65, "NoG", active, mouse_pos, input_text)

            # BB = Back Button
            BB_text = "Back"
            BB_x_check, BB_y_check, BB_pos, BB_size = self.button_values(BB_text, (self.height - self.margin - TILE_SIZE), mouse_pos)
            self.draw_button(BB_text, BB_x_check, BB_y_check, BB_pos, BB_size)

            # Update display
            self.true_display.blit(pyg_scale(self.false_display, self.true_display.get_size()), (0, 0))
            pyg_display.flip()

            # Get player input
            for event in pyg_get():
                # Check for exiting out of window
                if event.type == pyg_QUIT:
                    pyg_quit()
                    quit()
                # Check for if a button is pressed
                elif event.type == pyg_MOUSEBUTTONDOWN:
                    # Reset active
                    active = ""

                    # Back to main menu
                    if BB_x_check and BB_y_check:
                        self.settings_menu()

                    # Check to see if you clicked into a text box
                    if FPS_x_check and FPS_y_check:
                        active = "FPS"
                    elif POP_x_check and POP_y_check:
                        active = "POP"
                    elif NoG_x_check and NoG_y_check:
                        active = "NoG"
                
                # Check for typing
                if active=="FPS":
                    if event.type == pyg_KEYDOWN:
                        if event.key == pyg_K_RETURN:
                            with open(r"./Resources/DGA_Settings.txt", "w") as file:
                                self.DGA_Settings[0] = input_text
                                file.write(''.join(self.DGA_Settings))
                            input_text = ""
                            active = ""
                        elif event.key == pyg_K_BACKSPACE:
                            input_text = input_text[:-1]
                        else:
                            x, _ = FONT.size(input_text)
                            if x < FPS_box_size[0]:
                                input_text += event.unicode
                        self.DGA_Settings[0] = input_text
                elif active=="POP":
                    if event.type == pyg_KEYDOWN:
                        if event.key == pyg_K_RETURN:
                            with open(r"./Resources/DGA_Settings.txt", "w") as file:
                                self.DGA_Settings[1] = input_text
                                file.write(''.join(self.DGA_Settings))
                            input_text = ""
                            active = ""
                        elif event.key == pyg_K_BACKSPACE:
                            input_text = input_text[:-1]
                        else:
                            x, _ = FONT.size(input_text)
                            if x < POP_box_size[0]:
                                input_text += event.unicode
                        self.DGA_Settings[1] = input_text
                elif active=="NoG":
                    if event.type == pyg_KEYDOWN:
                        if event.key == pyg_K_RETURN:
                            with open(r"./Resources/DGA_Settings.txt", "w") as file:
                                self.DGA_Settings[2] = input_text
                                file.write(''.join(self.DGA_Settings))
                            input_text = ""
                            active = ""
                        elif event.key == pyg_K_BACKSPACE:
                            input_text = input_text[:-1]
                        else:
                            x, _ = FONT.size(input_text)
                            if x < NoG_box_size[0]:
                                input_text += event.unicode
                        self.DGA_Settings[2] = input_text


    def button_values(self, text, y_start, mouse_pos):
        width, height = FONT.size(text)
        x = self.width//2 - width//2
        y = y_start + int(1.5*height)
        x_check = self.w_scale*(x) <= mouse_pos[0] <= self.w_scale*(x+width)
        y_check = self.h_scale*(y) <= mouse_pos[1] <= self.h_scale*(y+height)
        return x_check, y_check, (x, y), (width, height)


    def draw_button(self, button_text, x_check, y_check, postion, size):
        # If they're hovering over the button
        if x_check and y_check:
            # Fill button area
            #pyg_rect(self.false_display, GRAY, [postion[0], postion[1]+5, size[0]+10, size[1]+5])

            # Place text
            text = FONT.render(button_text, True, RED)
            self.false_display.blit(text, [postion[0]+5, postion[1]+5])
        # Standard colors
        else:
            # Fill button area
            #pyg_rect(self.false_display, RED, [postion[0], postion[1]+5, size[0]+10, size[1]+5])

            # Place text
            text = FONT.render(button_text, True, WHITE)
            self.false_display.blit(text, [postion[0]+5, postion[1]+5])
    

    def draw_option(self, menu, option_text, position, box_x, active_checker, active, mouse_pos, input_text, center=False):
        # Get width, height, x, y
        w, h = FONT.size(option_text)
        x = TILE_SIZE*7 if not center else self.width//2 - w//2 - box_x//2
        y = TILE_SIZE*(7 + (1.5*position))

        # Text display
        text = FONT.render(option_text, True, GREEN2)
        self.false_display.blit(text, [x, y])

        # Input box
        box_pos = (x+w, y-5)
        box_size = (box_x, h-10)
        x_check = box_pos[0] <= mouse_pos[0] <= box_pos[0]+box_size[0]
        y_check = box_pos[1] <= mouse_pos[1] <= box_pos[1]+box_size[1]

        self.input_box(input_text, active, active_checker, menu, box_pos, box_size)
        return x_check, y_check, box_size
    

    def input_box(self, button_text, active, active_checker, menu, postion, size):
        if active==active_checker:
            # Fill text area
            pyg_rect(self.false_display, GRAY, [postion[0], postion[1]+7, size[0]+10, size[1]+5])
            
        else:
            # Get settings folder
            if menu=="HG":
                lines = self.HumanGameSettings
            elif menu=="DQN":
                lines = self.DQN_Settings
            elif menu=="DGA":
                lines = self.DGA_Settings

            # Get file path
            if active_checker=="FPS":
                button_text = lines[0].strip('\n')
            elif active_checker=="EP" or active_checker=="POP":
                button_text = lines[1].strip('\n')
            elif active_checker=="NoG":
                button_text = lines[2].strip('\n')

        # Place text
        text = FONT.render(button_text, True, WHITE)
        self.false_display.blit(text, [postion[0]+2, postion[1]+3])
