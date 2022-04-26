from helper import Direction, Point, TILE_SIZE, WHITE, RED, BLACK, GREEN1, GREEN2

from random import randint as rand_randint
from numpy import array_equal as np_array_equal

from pygame import init as pyg_init
from pygame import RESIZABLE as pyg_RESIZABLE
from pygame import QUIT as pyg_QUIT
from pygame import quit as pyg_quit
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
try:
    FONT = pyg_Font("arial.ttf", int(TILE_SIZE*1.5))
except:
    FONT = pyg_Font("arial", int(TILE_SIZE*1.5))



class MainMenu():
    '''
    A main menu to open the game into, where you can select
    your game mode and change various settings.
    '''
    def __init__(self):
        pass



class SettingsMenu():
    '''
    A menu for changing settings such as game speed, number of agents,
    number of episodes, number of generations, mutation rate, etc.
    '''
    def __init__(self):
        pass



class SnakeGameAI():
    ''' The base logic for the game itself. '''
    def __init__(self, fps=100, tiles_wide=32, tiles_high=24, tiles_margin=4):
        # Initialize input data
        self.fps = fps
        self.margin_value = tiles_margin
        self.width = tiles_wide*TILE_SIZE
        self.height = tiles_high*TILE_SIZE
        self.margin = tiles_margin*TILE_SIZE # Added to the bottom for diplaying data (score, episode, etc)

        # Initialze display
        self.true_display = pyg_set_mode((self.width, self.height + self.margin), pyg_RESIZABLE)
        self.false_display = self.true_display.copy()
        pyg_set_caption("Snake")
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
        if self._is_collision() or (self.frame_iteration > 125*len(self.snake)):
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


    def _is_collision(self, block=None):
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
        for x, y in self.snake:
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
        self.false_display.blit(text, [0, int(self.height+((TILE_SIZE//4)+((self.margin_value//2)*TILE_SIZE)))])

        # Show the highest score
        text = FONT.render(f"Top Score: {self.top_score}", True, WHITE)
        self.false_display.blit(text, [TILE_SIZE*9, int(self.height+((TILE_SIZE//4)+((self.margin_value//2)*TILE_SIZE)))])

        # Show the mean score
        text = FONT.render(f"Mean: {self.mean_score}", True, WHITE)
        self.false_display.blit(text, [TILE_SIZE*20, int(self.height+((TILE_SIZE//4)+((self.margin_value//2)*TILE_SIZE)))])

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
    def __init__(self, fps=10, tiles_wide=32, tiles_high=24, tiles_margin=2):
        # Initialize input data (and score)
        self.fps = fps
        self.width = tiles_wide*TILE_SIZE
        self.height = tiles_high*TILE_SIZE
        self.margin = tiles_margin*TILE_SIZE

        # Initialize internal values
        self.score = 0
        self.direction = Direction.UP

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

        # Move
        self._move() # Update the head
        self.snake.insert(0, self.head)

        # Check if game over
        game_over = False
        if self._is_collision():
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


    def _is_collision(self, block=None):
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
        for x, y in self.snake:
            pyg_rect(self.false_display, GREEN1, [x, y, TILE_SIZE, TILE_SIZE])
            pyg_rect(self.false_display, GREEN2, [x, y, TILE_SIZE, TILE_SIZE], 1)

        # Draw the food block
        pyg_rect(self.false_display, RED, [self.food.x, self.food.y, TILE_SIZE, TILE_SIZE])

        # Draw a line for the margin
        pyg_line(self.false_display, WHITE, (0, self.height), (self.width, self.height), width=2)

        # Show the current score
        text = FONT.render(f"Score: {self.score}", True, WHITE)
        self.false_display.blit(text, [0, int(self.height+(TILE_SIZE//4))])

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
