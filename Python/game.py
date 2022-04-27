from agent import Agent
from genetics import GeneticAlgorithm
from helper import Plotter, Direction, Point, TILE_SIZE, WHITE, GRAY, SLATE_GRAY, DIM_GRAY, BLACK, RED, GREEN1, GREEN2

from os import makedirs as os_makedirs
from os.path import exists as os_exists

from random import randint as rand_randint
from numpy import array_equal as np_array_equal
from numpy import round as np_round

from pygame import init as pyg_init
from pygame import RESIZABLE as pyg_RESIZABLE
from pygame import QUIT as pyg_QUIT
from pygame import quit as pyg_quit
from pygame import MOUSEBUTTONDOWN as pyg_MOUSEBUTTONDOWN
from pygame import KEYDOWN as pyg_KEYDOWN
from pygame import K_UP as pyg_K_UP
from pygame import K_LEFT as pyg_K_LEFT
from pygame import K_DOWN as pyg_K_DOWN
from pygame import K_RIGHT as pyg_K_RIGHT
from pygame import K_w as pyg_K_w
from pygame import K_a as pyg_K_a
from pygame import K_s as pyg_K_s
from pygame import K_d as pyg_K_d
from pygame.font import Font as pyg_Font
from pygame.mouse import get_pos as pyg_mouse_get_pos
from pygame.time import Clock as pyg_Clock
from pygame.event import get as pyg_get
from pygame.transform import scale as pyg_scale
from pygame.display import set_mode as pyg_set_mode
from pygame.display import set_caption as pyg_set_caption
from pygame.display import flip as pyg_flip
from pygame.draw import rect as pyg_rect
from pygame.draw import line as pyg_line


# Initialize pygame
pyg_init()

# Fonts
FONT_SIZE = int(TILE_SIZE*1.5)
TITLE_FONT_SIZE = int(TILE_SIZE*3)
try:
    FONT = pyg_Font("arial.ttf", FONT_SIZE)
    TITLE_FONT = pyg_Font("arial.ttf", TITLE_FONT_SIZE)
except:
    FONT = pyg_Font("arial", FONT_SIZE)
    TITLE_FONT = pyg_Font("arial", TITLE_FONT_SIZE)



class StartMenu():
    '''
    A main menu to open the game into, where you can select
    your game mode and change various settings.
    '''
    def __init__(self, tiles_wide=32, tiles_high=24, tiles_margin=4):
        # Initialize input data
        self.width = tiles_wide*TILE_SIZE
        self.height = tiles_high*TILE_SIZE
        self.margin = tiles_margin*TILE_SIZE

        # Initialze display
        self.true_display = pyg_set_mode((self.width, self.height+self.margin), pyg_RESIZABLE)
        self.false_display = self.true_display.copy()
        pyg_set_caption("Snake")

        # Create background snake object
        self.bg_snake = BackgroundSnake(self.width, self.height, self.margin, self.false_display)


    def main_menu(self):
        while True:
            # Get current mouse position
            mouse_pos = pyg_mouse_get_pos()

            # Set placement values
            # NGMB = New Game Menu Button
            NGMB_text = "New Game"
            NGMB_width, NGMB_height = FONT.size(NGMB_text)
            NGMB_x = self.width//2 - NGMB_width//2
            NGMB_y = TILE_SIZE*5
            NGMB_x_check = NGMB_x <= mouse_pos[0] <= NGMB_x+NGMB_width
            NGMB_y_check = NGMB_y <= mouse_pos[1] <= NGMB_y+NGMB_height

            # SMB = Settings Menu Button
            SMB_text = "Settings"
            SMB_width, SMB_height = FONT.size(SMB_text)
            SMB_x = self.width//2 - SMB_width//2
            SMB_y = NGMB_y + int(1.5*SMB_height)
            SMB_x_check = SMB_x <= mouse_pos[0] <= SMB_x+SMB_width
            SMB_y_check = SMB_y <= mouse_pos[1] <= SMB_y+SMB_height

            # QB = Quit Button
            QB_text = "Quit"
            QB_width, QB_height = FONT.size(QB_text)
            QB_x = self.width//2 - QB_width//2
            QB_y = SMB_y + int(1.5*QB_height)
            QB_x_check = QB_x <= mouse_pos[0] <= QB_x+QB_width
            QB_y_check = QB_y <= mouse_pos[1] <= QB_y+QB_height

            # Black out previous display
            self.false_display.fill(BLACK)

            # Run the background snake
            self.false_display = self.bg_snake.play_step()

            # Display title
            t_x, t_y = TITLE_FONT.size("Snake")
            text = TITLE_FONT.render("Snake", True, GREEN2)
            self.false_display.blit(text, [self.width//2 - t_x//2, t_y//3])

            # Draw buttons
            self.draw_button(NGMB_text,
                             NGMB_x_check,
                             NGMB_y_check,
                             (NGMB_x, NGMB_y),
                             (NGMB_width, NGMB_height))

            self.draw_button(SMB_text,
                             SMB_x_check,
                             SMB_y_check,
                             (SMB_x, SMB_y),
                             (SMB_width, SMB_height))
            
            self.draw_button(QB_text,
                             QB_x_check,
                             QB_y_check,
                             (QB_x, QB_y),
                             (QB_width, QB_height))

            # Update display
            self.true_display.blit(pyg_scale(self.false_display, self.true_display.get_rect().size), (0, 0))
            pyg_flip()

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
            # Get current mouse position
            mouse_pos = pyg_mouse_get_pos()

            # Set placement values
            # NG = Normal Game
            NG_text = "Normal Game"
            NG_width, NG_height = FONT.size(NG_text)
            NG_x = self.width//2 - NG_width//2
            NG_y = 100
            NG_x_check = NG_x <= mouse_pos[0] <= NG_x+NG_width
            NG_y_check = NG_y <= mouse_pos[1] <= NG_y+NG_height

            # SA = Single Agent
            SA_text = "Single Agent"
            SA_width, SA_height = FONT.size(SA_text)
            SA_x = self.width//2 - SA_width//2
            SA_y = NG_y + int(1.5*SA_height)
            SA_x_check = SA_x <= mouse_pos[0] <= SA_x+SA_width
            SA_y_check = SA_y <= mouse_pos[1] <= SA_y+SA_height

            # PoA = Population of Agents
            PoA_text = "Multiple Agents"
            PoA_width, PoA_height = FONT.size(PoA_text)
            PoA_x = self.width//2 - PoA_width//2
            PoA_y = SA_y + int(1.5*PoA_height)
            PoA_x_check = PoA_x <= mouse_pos[0] <= PoA_x+PoA_width
            PoA_y_check = PoA_y <= mouse_pos[1] <= PoA_y+PoA_height

            # BB = Back Button
            BB_text = "Back"
            BB_width, BB_height = FONT.size(BB_text)
            BB_x = self.width//2 - BB_width//2
            BB_y = PoA_y + int(1.5*BB_height)
            BB_x_check = BB_x <= mouse_pos[0] <= BB_x+BB_width
            BB_y_check = BB_y <= mouse_pos[1] <= BB_y+BB_height

            # Black out previous display
            self.false_display.fill(BLACK)

            # Run the background snake
            self.false_display = self.bg_snake.play_step()

            # Display menu title
            t_x, t_y = TITLE_FONT.size("Select Game Type")
            text = TITLE_FONT.render("Select Game Type", True, GREEN2)
            self.false_display.blit(text, [self.width//2 - t_x//2, t_y//3])

            # Draw buttons
            self.draw_button(NG_text,
                             NG_x_check,
                             NG_y_check,
                             (NG_x, NG_y),
                             (NG_width, NG_height))

            self.draw_button(SA_text,
                             SA_x_check,
                             SA_y_check,
                             (SA_x, SA_y),
                             (SA_width, SA_height))
            
            self.draw_button(PoA_text,
                             PoA_x_check,
                             PoA_y_check,
                             (PoA_x, PoA_y),
                             (PoA_width, PoA_height))
            
            self.draw_button(BB_text,
                             BB_x_check,
                             BB_y_check,
                             (BB_x, BB_y),
                             (BB_width, BB_height))

            # Update display
            self.true_display.blit(pyg_scale(self.false_display, self.true_display.get_rect().size), (0, 0))
            pyg_flip()

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
                        run_game = RunGame(self.width, self.height, self.margin)
                        run_game.run_human(fps=10)
                    # Start game with a single agent
                    elif SA_x_check and SA_y_check:
                        run_game = RunGame(self.width, self.height, self.margin)
                        run_game.run_dqn()
                    # Start game with a population of agents
                    elif PoA_x_check and PoA_y_check:
                        run_game = RunGame(self.width, self.height, self.margin)
                        run_game.run_grl()
                    # Back to main menu
                    elif BB_x_check and BB_y_check:
                        self.main_menu()


    def settings_menu(self):
        pass


    def draw_button(self, button_text, x_check, y_check, postion, size):
        # If they're hovering over the button
        if x_check and y_check:
            # Fill button area
            #pyg_rect(self.false_display, GRAY, [postion[0], postion[1]+5, size[0]+10, size[1]+5])
            #pyg_rect(self.false_display, DIM_GRAY, [postion[0], postion[1]+5, size[0]+10, size[1]+5], 1)

            # Place text
            text = FONT.render(button_text, True, RED)
            self.false_display.blit(text, [postion[0]+5, postion[1]+5])
        # Standard colors
        else:
            # Fill button area
            #pyg_rect(self.false_display, SLATE_GRAY, [postion[0], postion[1]+5, size[0]+10, size[1]+5])
            #pyg_rect(self.false_display, DIM_GRAY, [postion[0], postion[1]+5, size[0]+10, size[1]+5], 1)

            # Place text
            text = FONT.render(button_text, True, WHITE)
            self.false_display.blit(text, [postion[0]+5, postion[1]+5])



class BackgroundSnake():
    ''' The base logic for the game itself. '''
    def __init__(self, width, height, margin, false_display, fps=15):
        # Initialize input data
        self.fps = fps
        self.width = width
        self.height = height
        self.margin = margin

        # Initialze display
        self.false_display = false_display
        self.clock = pyg_Clock()

        # Initialize agent
        self.agent = Agent(model_path=r"./resources/background_model.h5")

        # Initialize game values
        self.reset()
        self.death_counter = 0


    def reset(self):
        ''' Reset/Initialize base game state. '''
        # Default starting direction
        self.direction = Direction.UP

        # Set head, then add it to the snake, along with two
        # other body blocks
        self.head = Point(self.width//2, self.height//2)
        self.snake = [self.head,
                      Point(self.head.x, self.head.y+TILE_SIZE),
                      Point(self.head.x, self.head.y+(2*TILE_SIZE))]

        # Initialize score and food
        self.score = 0
        self._food_gen()
        self.frame_iteration = 0


    def _food_gen(self):
        ''' Randomly place food on the map. '''
        x = rand_randint(0, (self.width-TILE_SIZE) // TILE_SIZE) * TILE_SIZE
        y = rand_randint(0, (self.height-TILE_SIZE) // TILE_SIZE) * TILE_SIZE
        self.food = Point(x, y)

        # Check for conflicting values
        if self.food in self.snake:
            self._food_gen()


    def play_step(self):
        ''' Run a frame of the game. '''
        self.frame_iteration += 1

        # Set-up for getting the state
        head = self.snake[0]
        point_l = Point(head.x - TILE_SIZE, head.y)
        point_r = Point(head.x + TILE_SIZE, head.y)
        point_u = Point(head.x, head.y - TILE_SIZE)
        point_d = Point(head.x, head.y + TILE_SIZE)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        # List of states, using binary checks to fill values
        state = [
            # Danger straight
            (dir_r and self.is_collision(point_r)) or
            (dir_l and self.is_collision(point_l)) or
            (dir_u and self.is_collision(point_u)) or
            (dir_d and self.is_collision(point_d)),

            # Danger right
            (dir_u and self.is_collision(point_r)) or
            (dir_d and self.is_collision(point_l)) or
            (dir_l and self.is_collision(point_u)) or
            (dir_r and self.is_collision(point_d)),

            # Danger left
            (dir_d and self.is_collision(point_r)) or
            (dir_u and self.is_collision(point_l)) or
            (dir_r and self.is_collision(point_u)) or
            (dir_l and self.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            self.food.x < head.x,  # food left
            self.food.x > head.x,  # food right
            self.food.y < head.y,  # food up
            self.food.y > head.y   # food down
        ]

        # Get action from the agent
        action = self.agent.get_action(state)

        # Move
        self._move(action) # Update the head
        self.snake.insert(0, self.head)

        # Check if game over
        # If the snake hasn't made enough progress, it's executed
        if self.is_collision() or (self.frame_iteration > 125*len(self.snake)):
            self.reset()
            self.death_counter += 1
            return self.false_display

        # Place new food or just move
        if self.head == self.food:
            self.score += 1
            self._food_gen()
        else:
            self.snake.pop()

        # Update ui and clock
        self._update_ui()
        self.clock.tick(self.fps)

        return self.false_display


    def is_collision(self, block=None):
        ''' Check for collision against a wall or the snake's body. '''
        if block is None:
            block = self.head
        # Hits boundary
        if block.x > self.width - TILE_SIZE or block.x < 0 or block.y > self.height - TILE_SIZE or block.y < 0:
            return True
        # Hits itself
        if block in self.snake[1:]:
            return True
        return False


    def _update_ui(self):
        ''' Update the game screen. '''
        # Draw out the snake block by block
        x, y = self.snake[0]
        pyg_rect(self.false_display, GREEN2, [x, y, TILE_SIZE, TILE_SIZE])
        pyg_rect(self.false_display, GREEN1, [x, y, TILE_SIZE, TILE_SIZE], 1)
        for x, y in self.snake[1:]:
            pyg_rect(self.false_display, GREEN1, [x, y, TILE_SIZE, TILE_SIZE])
            pyg_rect(self.false_display, GREEN2, [x, y, TILE_SIZE, TILE_SIZE], 1)

        # Draw the food block
        pyg_rect(self.false_display, RED, [self.food.x, self.food.y, TILE_SIZE, TILE_SIZE])

        # Draw a line for the margin
        pyg_line(self.false_display, WHITE, (0, self.height), (self.width, self.height), width=2)

        # Show the current score
        t_x, _ = FONT.size(f"Score: {self.score}")
        text = FONT.render(f"Score: {self.score}", True, WHITE)
        self.false_display.blit(text, [self.width//2 - t_x//2, int(self.height+(TILE_SIZE//4))])

        # Show the current death count
        t_x, _ = FONT.size(f"Deaths: {self.death_counter}")
        text = FONT.render(f"Deaths: {self.death_counter}", True, WHITE)
        self.false_display.blit(text, [self.width//2 - t_x//2, int(self.height+((TILE_SIZE//4)+(self.margin//2)))])


    def _move(self, action):
        '''
        Choose a new direction from straight, right, or left, where
        straight is to continue the current direction and right and
        left are to turn in either direction from the perspective
        of what direction the snake is currently heading.
        '''
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        # No change (straight)
        if np_array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        # Right turn r -> d -> l -> u
        elif np_array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        # Left turn r -> u -> l -> d
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        # Set new direction to the class variable
        self.direction = new_dir

        # Update the head coordinates
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += TILE_SIZE
        elif self.direction == Direction.LEFT:
            x -= TILE_SIZE
        elif self.direction == Direction.DOWN:
            y += TILE_SIZE
        elif self.direction == Direction.UP:
            y -= TILE_SIZE
        self.head = Point(x, y)



class SnakeGameAI():
    ''' The base logic for the game itself. '''
    def __init__(self, width, height, margin, fps=100):
        # Initialize input data
        self.fps = fps
        self.width = width
        self.height = height
        self.margin = margin

        # Initialze display
        self.true_display = pyg_set_mode((self.width, self.height + self.margin), pyg_RESIZABLE)
        self.false_display = self.true_display.copy()
        self.clock = pyg_Clock()

        # Initialize game values
        self.reset()
        self.generation = 0
        self.agent_num = 0
        self.top_score = 0
        self.agent_episode = 0
        self.mean_score = 0.0
        self.total_score = 0

        # Reward values
        self.food_reward = 10
        self.death_reward = -10

        # Colors
        self.color1 = (0, 0, 0)
        self.color2 = (0, 0, 0)


    def reset(self):
        ''' Reset/Initialize base game state. '''
        # Default starting direction
        self.direction = Direction.UP

        # Set head, then add it to the snake, along with two
        # other body blocks
        self.head = Point(self.width//2, self.height//2)
        self.snake = [self.head,
                      Point(self.head.x, self.head.y+TILE_SIZE),
                      Point(self.head.x, self.head.y+(2*TILE_SIZE))]

        # Initialize score and food
        self.score = 0
        self._food_gen()
        self.frame_iteration = 0


    def _food_gen(self):
        ''' Randomly place food on the map. '''
        x = rand_randint(0, (self.width-TILE_SIZE) // TILE_SIZE) * TILE_SIZE
        y = rand_randint(0, (self.height-TILE_SIZE) // TILE_SIZE) * TILE_SIZE
        self.food = Point(x, y)

        # Check for conflicting values
        if self.food in self.snake:
            self._food_gen()


    def play_step(self, action):
        ''' Run a frame of the game. '''
        self.frame_iteration += 1
        reward = 0

        # Check for if the game has been closed
        for event in pyg_get():
            if event.type == pyg_QUIT:
                pyg_quit()
                quit()

        # Move
        self._move(action) # Update the head
        self.snake.insert(0, self.head)

        # Check if game over
        # If the snake hasn't made enough progress, it's executed
        game_over = False
        if self.is_collision() or (self.frame_iteration > 125*len(self.snake)):
            game_over = True
            reward = self.death_reward
            return reward, game_over, self.score

        # Place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = self.food_reward
            self._food_gen()
        else:
            self.snake.pop()

        # Update ui and clock
        self._update_ui()
        self.clock.tick(self.fps)

        # Return values for the agent to process
        return reward, game_over, self.score


    def is_collision(self, block=None):
        ''' Check for collision against a wall or the snake's body. '''
        if block is None:
            block = self.head
        # Hits boundary
        if block.x > self.width - TILE_SIZE or block.x < 0 or block.y > self.height - TILE_SIZE or block.y < 0:
            return True
        # Hits itself
        if block in self.snake[1:]:
            return True
        return False


    def _update_ui(self):
        ''' Update the game screen. '''
        self.false_display.fill(BLACK)

        # Draw out the snake block by block
        #x, y = self.snake[0]
        #pyg_rect(self.false_display, self.color2, [x, y, TILE_SIZE, TILE_SIZE])
        #pyg_rect(self.false_display, self.color1, [x, y, TILE_SIZE, TILE_SIZE], 1)
        for x, y in self.snake[1:]:
            pyg_rect(self.false_display, self.color1, [x, y, TILE_SIZE, TILE_SIZE])
            pyg_rect(self.false_display, self.color2, [x, y, TILE_SIZE, TILE_SIZE], 1)

        # Draw the food block
        pyg_rect(self.false_display, RED, [self.food.x, self.food.y, TILE_SIZE, TILE_SIZE])

        # Draw a line for the margin
        pyg_line(self.false_display, WHITE, (0, self.height), (self.width, self.height), width=2)

        # Show the current agent
        text = FONT.render(f"Agent {self.agent_num}", True, WHITE)
        self.false_display.blit(text, [0, int(self.height+(TILE_SIZE//4))])

        # Show the current episode
        text = FONT.render(f"Episode: {self.agent_episode}", True, WHITE)
        self.false_display.blit(text, [TILE_SIZE*9, int(self.height+(TILE_SIZE//4))])

        # Show the current generation
        text = FONT.render(f"Generation: {self.generation}", True, WHITE)
        self.false_display.blit(text, [TILE_SIZE*20, int(self.height+(TILE_SIZE//4))])

        # Show the current score
        text = FONT.render(f"Score: {self.score}", True, WHITE)
        self.false_display.blit(text, [0, int(self.height+((TILE_SIZE//4)+(self.margin//2)))])

        # Show the highest score
        text = FONT.render(f"Top Score: {self.top_score}", True, WHITE)
        self.false_display.blit(text, [TILE_SIZE*9, int(self.height+((TILE_SIZE//4)+(self.margin//2)))])

        # Show the mean score
        text = FONT.render(f"Mean: {self.mean_score}", True, WHITE)
        self.false_display.blit(text, [TILE_SIZE*20, int(self.height+((TILE_SIZE//4)+(self.margin//2)))])

        # Update the display
        self.true_display.blit(pyg_scale(self.false_display, self.true_display.get_rect().size), (0, 0))
        pyg_flip()


    def _move(self, action):
        '''
        Choose a new direction from straight, right, or left, where
        straight is to continue the current direction and right and
        left are to turn in either direction from the perspective
        of what direction the snake is currently heading.
        '''
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        # No change (straight)
        if np_array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        # Right turn r -> d -> l -> u
        elif np_array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        # Left turn r -> u -> l -> d
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        # Set new direction to the class variable
        self.direction = new_dir

        # Update the head coordinates
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += TILE_SIZE
        elif self.direction == Direction.LEFT:
            x -= TILE_SIZE
        elif self.direction == Direction.DOWN:
            y += TILE_SIZE
        elif self.direction == Direction.UP:
            y -= TILE_SIZE
        self.head = Point(x, y)



class SnakeGameHuman():
    ''' This is the class to run if you're a human wanting to play snake. '''
    def __init__(self, width, height, margin, fps=10):
        # Initialize input data
        self.fps = fps
        self.width = width
        self.height = height
        self.margin = margin

        # Initialize internal values
        self.score = 0
        self.direction = Direction.UP
        self.wait = True

        # Initialze display
        self.true_display = pyg_set_mode((self.width, self.height + self.margin), pyg_RESIZABLE)
        self.false_display = self.true_display.copy()
        pyg_set_caption("Snake")
        self.clock = pyg_Clock()

        # Set head, then add it to the snake, along with two
        # other body blocks
        self.head = Point(self.width//2, self.height//2)
        self.snake = [self.head,
                      Point(self.head.x, self.head.y+TILE_SIZE),
                      Point(self.head.x, self.head.y+(2*TILE_SIZE))]

        # Generate first food block
        self._food_gen()


    def _food_gen(self):
        ''' Randomly place food on the map. '''
        x = rand_randint(0, (self.width-TILE_SIZE) // TILE_SIZE) * TILE_SIZE
        y = rand_randint(0, (self.height-TILE_SIZE) // TILE_SIZE) * TILE_SIZE
        self.food = Point(x, y)

        # Check for conflicting values
        if self.food in self.snake:
            self._food_gen()


    def play_step(self):
        ''' Run a frame of the game. '''
        # Get player input
        for event in pyg_get():
            if event.type == pyg_QUIT:
                pyg_quit()
                quit()
            if event.type == pyg_KEYDOWN:
                self.wait = False
                # Can use arrow keys or WASD
                # Prevents a person from turning the opposite direction right into the snake body
                if ((event.key == pyg_K_UP) or (event.key == pyg_K_w)) and (self.direction != Direction.DOWN):
                    self.direction = Direction.UP
                elif ((event.key == pyg_K_LEFT) or (event.key == pyg_K_a)) and (self.direction != Direction.RIGHT):
                    self.direction = Direction.LEFT
                elif ((event.key == pyg_K_DOWN) or (event.key == pyg_K_s)) and (self.direction != Direction.UP):
                    self.direction = Direction.DOWN
                elif ((event.key == pyg_K_RIGHT) or (event.key == pyg_K_d)) and (self.direction != Direction.LEFT):
                    self.direction = Direction.RIGHT

        # Skip if there's no input
        if self.wait:
            self._update_ui()
            self.clock.tick(self.fps)
            return False, 0

        # Move
        self._move() # Update the head
        self.snake.insert(0, self.head)

        # Check if game over
        game_over = False
        if self.is_collision():
            game_over = True
            return game_over, self.score

        # Place new food or just move
        if self.head == self.food:
            self.score += 1
            self._food_gen()
        else:
            self.snake.pop()

        # Update ui and clock
        self._update_ui()
        self.clock.tick(self.fps)

        # Return values for the agent to process
        return game_over, self.score


    def is_collision(self, block=None):
        ''' Check for collision against a wall or the snake's body. '''
        if block is None:
            block = self.head
        # Hits boundary
        if block.x > self.width - TILE_SIZE or block.x < 0 or block.y > self.height - TILE_SIZE or block.y < 0:
            return True
        # Hits itself
        if block in self.snake[1:]:
            return True
        return False


    def _update_ui(self):
        ''' Update the game screen. '''
        self.false_display.fill(BLACK)

        # Draw out the snake block by block
        #x, y = self.snake[0]
        #pyg_rect(self.false_display, GREEN2, [x, y, TILE_SIZE, TILE_SIZE])
        #pyg_rect(self.false_display, GREEN1, [x, y, TILE_SIZE, TILE_SIZE], 1)
        for x, y in self.snake[1:]:
            pyg_rect(self.false_display, GREEN1, [x, y, TILE_SIZE, TILE_SIZE])
            pyg_rect(self.false_display, GREEN2, [x, y, TILE_SIZE, TILE_SIZE], 1)

        # Draw the food block
        pyg_rect(self.false_display, RED, [self.food.x, self.food.y, TILE_SIZE, TILE_SIZE])

        # Draw a line for the margin
        pyg_line(self.false_display, WHITE, (0, self.height), (self.width, self.height), width=2)

        # Show the current score
        t_x, _ = FONT.size(f"Score: {self.score}")
        text = FONT.render(f"Score: {self.score}", True, WHITE)
        self.false_display.blit(text, [self.width//2 - t_x//2, int(self.height+(TILE_SIZE//4)) + TILE_SIZE])

        # Update the display
        self.true_display.blit(pyg_scale(self.false_display, self.true_display.get_rect().size), (0, 0))
        pyg_flip()


    def _move(self):
        ''' Set new direction (or continue with the current one). '''
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += TILE_SIZE
        elif self.direction == Direction.LEFT:
            x -= TILE_SIZE
        elif self.direction == Direction.DOWN:
            y += TILE_SIZE
        elif self.direction == Direction.UP:
            y -= TILE_SIZE
        self.head = Point(x, y)



class RunGame():
    '''
    Controller class for running the snake game as a human,
    for training a single DQN snake, or a population of DQN
    snakes in tandem with a deep genetic algorithm.
    '''
    def __init__(self, width, height, margin):
        # Initialize input data
        self.width = width
        self.height = height
        self.margin = margin


    def run_human(self, fps=10):
        ''' Run the snake game in a way that a human can play. '''
        # Create game object
        game = SnakeGameHuman(self.width, self.height, self.margin, fps=fps)

        # Game loop
        while True:
            game_over, score = game.play_step()
            if game_over: break
        print(f"\nFinal Score: {score}\n")


    def run_dqn(self, fps=100, max_episodes=5000):
        ''' Run a single deep Q learning snake. '''
        # Set internal variables
        self.max_episodes = max_episodes

        # Create objects
        agent = Agent()
        plotter = Plotter(single_agent=True)
        self.game = SnakeGameAI(self.width, self.height, self.margin, fps=fps)

        # Set colors
        self.game.color1 = agent.color1
        self.game.color2 = agent.color2

        # Set aggregate data lists
        self.agent_scores, self.agent_mean_scores = [], []

        # Set score aggregate for this agent
        self.agent_score = 0

        # Run for set number of episodes (adjusting to start at ep 1)
        for cur_episode in range(1, self.max_episodes+1):
            # Episode variables
            agent.episode = cur_episode
            self.game.agent_episode = cur_episode

            # Episode loop
            agent = self._run_episode(agent, single_agent=True)

            # Plot data
            plotter.plot_single_agent(self.agent_scores, self.agent_mean_scores)
        
        if not os_exists(r"./models"):
            os_makedirs(r"./models")
        if not os_exists(r"./models/single_model_({}).h5".format(self.max_episodes)):
            agent.model.save(r"./models/single_model_({}).h5".format(self.max_episodes))


    def run_grl(self, fps=100, population_size=1, max_episodes=10, max_generations=10):
        ''' Run a session of genetic reinforcement learning. '''
        # Set internal variables
        self.population_size = population_size
        self.max_episodes = max_episodes
        self.max_generations = max_generations

        # Create class objects
        self.agents = [Agent() for i in range(self.population_size)]
        self.game = SnakeGameAI(self.width, self.height, self.margin, fps=fps)
        self.genetics = GeneticAlgorithm()
        self.plotter = Plotter()

        # Set aggregate data lists
        self.all_scores, self.all_mean_scores = [], []
        self.all_episodes = 0

        # Run for set number of generations (adjusting to start at gen 1)
        for cur_gen in range(1, self.max_generations+1):
            self._run_generation(cur_gen)

        # Save session's graph
        self.plotter.save_session()


    def _run_generation(self, cur_gen):
        ''' Process an entire population of agents. '''
        # Reset generation data
        self.game.generation = cur_gen

        # Set score aggregate for this generation
        self.gen_score = 0

        # Reset generation data lists
        self.gen_scores, self.gen_mean_scores = [], []
        self.gen_episodes = 0

        # Run for set number of generations
        self.agents = self._run_agent()

        # Save generation's graph
        self.plotter.save_gen(cur_gen)

        # Make new population
        self.agents = self.genetics.breed_population(self.agents)


    def _run_agent(self):
        ''' Run an agent, whether on it's own or as part of a population. '''
        for agent_num, agent in enumerate(self.agents):
            # Set colors
            self.game.color1 = agent.color1
            self.game.color2 = agent.color2

            # Set agent number
            self.game.agent_num = agent_num+1

            # Reset agent data lists
            self.agent_scores, self.agent_mean_scores = [], []

            # Set score aggregate for this agent
            self.agent_score = 0

            # Run for set number of episodes (adjusting to start at ep 1)
            for cur_episode in range(1, self.max_episodes+1):
                # Episode variables
                agent.episode = cur_episode
                self.game.agent_episode = cur_episode
                self.all_episodes += 1
                self.gen_episodes += 1

                # Episode loop
                agent = self._run_episode(agent)

                # Record data
                self._record_data()
            
            # Update the agent in the population
            self.agents[agent_num] = agent
            

        # Save agent's graph
        self.plotter.save_agent(self.game.generation, agent_num)


    def _run_episode(self, agent, single_agent=False):
        ''' Run an episode of the game. '''
        run = True
        while run:
            # Get old state
            state_old = agent.get_state(self.game)

            # Get move
            final_move = agent.get_action(state_old)

            # Perform move and get new state
            reward, done, score = self.game.play_step(final_move)
            state_new = agent.get_state(self.game)

            # Train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # Remember
            agent.remember(state_old, final_move, reward, state_new, done)

            # Snake died
            if done:
                run = False

                # Train long memory
                self.game.reset()
                agent.train_long_memory()

                # Update agent's internal score if needed
                if score > agent.top_score:
                    agent.top_score = score

                if score > self.game.top_score:
                    self.game.top_score = score

                if not single_agent:
                    # Save model if it's the best (and update top score)
                    if score > self.game.top_score:
                        if not os_exists(r"./models"):
                            os_makedirs(r"./models")
                        if not os_exists(r"./models/pop{}-eps{}-gens{}".format(self.population_size, self.max_episodes, self.max_generations)):
                            agent.model.save(r"./models/pop{}-eps{}-gens{}".format(self.population_size, self.max_episodes, self.max_generations))

                        if not os_exists(r"./models/model_gen{}_({}-{}-{}).h5".format(self.game.generation, self.population_size, self.max_episodes, self.max_generations)):
                            agent.model.save(r"./models/model_gen{}_({}-{}-{}).h5".format(self.game.generation, self.population_size, self.max_episodes, self.max_generations))

                    # Update aggregate data
                    self.game.total_score += score
                    self.game.mean_score = np_round((self.game.total_score / self.all_episodes), 3)
                    self.all_scores.append(score)
                    self.all_mean_scores.append(self.game.mean_score)

                    # Updte generation data
                    self.gen_score += score
                    gen_mean = np_round((self.gen_score / self.gen_episodes), 3)
                    self.gen_scores.append(score)
                    self.gen_mean_scores.append(gen_mean)

                # Update agent data
                self.agent_score += score
                agent_mean = np_round((self.agent_score / self.game.agent_episode), 3)
                self.agent_scores.append(score)
                self.agent_mean_scores.append(agent_mean)
        return agent


    def _record_data(self):
        ''' Record the session data as it currently stands. '''
        self.plotter.plot_data(self.all_scores,
                               self.all_mean_scores,
                               self.gen_scores,
                               self.gen_mean_scores,
                               self.game.generation,
                               self.agent_scores,
                               self.agent_mean_scores,
                               self.game.agent_num,
                               len(self.agents),
                               self.game.agent_episode,
                               self.max_episodes,
                               self.game.top_score)
