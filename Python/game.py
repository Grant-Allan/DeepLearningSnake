from agent import Agent
from helper import Direction, Point, TILE_SIZE, WHITE, BLACK, RED, GREEN1, GREEN2

from random import randint as rand_randint
from numpy import array_equal as np_array_equal
from numpy import sqrt as np_sqrt
from math import floor as math_floor
from math import ceil as math_ceil

from pygame import RESIZABLE as pyg_RESIZABLE
from pygame import QUIT as pyg_QUIT
from pygame import quit as pyg_quit
from pygame import KEYDOWN as pyg_KEYDOWN
from pygame import K_ESCAPE as pyg_K_ESCAPE
from pygame import K_UP as pyg_K_UP
from pygame import K_LEFT as pyg_K_LEFT
from pygame import K_DOWN as pyg_K_DOWN
from pygame import K_RIGHT as pyg_K_RIGHT
from pygame import K_w as pyg_K_w
from pygame import K_a as pyg_K_a
from pygame import K_s as pyg_K_s
from pygame import K_d as pyg_K_d
from pygame import font as pyg_font
from pygame import display as pyg_display
from pygame.time import Clock as pyg_Clock
from pygame.event import get as pyg_get
from pygame.transform import scale as pyg_scale
from pygame.draw import rect as pyg_rect
from pygame.draw import line as pyg_line


# Initialize pygame modules as needed
pyg_font.init()
pyg_display.init()

# Fonts
FONT_SIZE = int(TILE_SIZE*1.5)
TITLE_FONT_SIZE = int(TILE_SIZE*3)
try:
    FONT = pyg_font.Font("arial.ttf", FONT_SIZE)
    TITLE_FONT = pyg_font.Font("arial.ttf", TITLE_FONT_SIZE)
except:
    FONT = pyg_font.Font("arial", FONT_SIZE)
    TITLE_FONT = pyg_font.Font("arial", TITLE_FONT_SIZE)



class BackgroundSnake():
    ''' The pre-trained snake that runs in the background of the menus. '''
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
        self.agent = Agent(model_path=r"./Resources/background_model.h5")

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
        #x, y = self.snake[0]
        #pyg_rect(self.false_display, GREEN2, [x, y, TILE_SIZE, TILE_SIZE])
        #pyg_rect(self.false_display, GREEN1, [x, y, TILE_SIZE, TILE_SIZE], 1)
        #for x, y in self.snake[1:]:
        for x, y in self.snake:
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
    ''' The logic for having a Deep Q Learning snake run. '''
    def __init__(self, width, height, margin, fps=100):
        # Initialize input data
        self.fps = fps
        self.width = width
        self.height = height
        self.margin = margin

        # Initialze display
        self.true_display = pyg_display.set_mode((self.width, self.height+self.margin), pyg_RESIZABLE)
        self.false_display = self.true_display.copy()
        pyg_display.set_caption("Snake")
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
        #for x, y in self.snake[1:]:
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
        self.false_display.blit(text, [0, int(self.height+((TILE_SIZE//4)+(self.margin//2)))])

        # Show the highest score
        text = FONT.render(f"Top Score: {self.top_score}", True, WHITE)
        self.false_display.blit(text, [TILE_SIZE*9, int(self.height+((TILE_SIZE//4)+(self.margin//2)))])

        # Show the mean score
        text = FONT.render(f"Mean: {self.mean_score}", True, WHITE)
        self.false_display.blit(text, [TILE_SIZE*20, int(self.height+((TILE_SIZE//4)+(self.margin//2)))])

        # Update the display
        self.true_display.blit(pyg_scale(self.false_display, self.true_display.get_size()), (0, 0))
        pyg_display.flip()


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



class SnakeGameGA():
    ''' The logic for having a Deep Q Learning snake run. '''
    def __init__(self, width, height, margin, population_size, fps=100):
        # Initialize input data
        self.fps = fps
        self.width = width
        self.height = height
        self.margin = margin
        self.population_size = population_size

        # Get x and y ratios for displaying each snake in subsections of the display
        ratio = np_sqrt(population_size)
        self.x_ratio = self.width / math_ceil(ratio)
        self.y_ratio = self.height / math_floor(ratio)

        # Initialze display
        self.true_display = pyg_display.set_mode((self.width, self.height+self.margin), pyg_RESIZABLE)
        self.false_display = self.true_display.copy()
        pyg_display.set_caption("Snake")
        self.clock = pyg_Clock()

        # Initialize game values
        self.reset()
        self.generation = 0
        self.top_score = 0


    def reset(self):
        ''' Reset/Initialize base game state. '''
        self.agents_data = []
        for i in range(self.population_size):
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

            # Colors
            color1 = (rand_randint(0, 255), rand_randint(0, 255), rand_randint(0, 255))
            color2 = (rand_randint(0, 255), rand_randint(0, 255), rand_randint(0, 255))

            # Store agent's data
            # direction = 0, head = 1, snake = 2, score = 3, food = 4, frame = 5, game_over = 6, color tuple = 7
            self.agents_data.append([self.direction, self.head, self.snake, self.score, self.food, self.frame_iteration, False, (color1, color2)])


    def _food_gen(self):
        ''' Randomly place food on the map. '''
        x = rand_randint(0, (self.width-TILE_SIZE) // TILE_SIZE) * TILE_SIZE
        y = rand_randint(0, (self.height-TILE_SIZE) // TILE_SIZE) * TILE_SIZE
        self.food = Point(x, y)

        # Check for conflicting values
        if self.food in self.snake:
            self._food_gen()


    def play_step(self, agents):
        ''' Run a frame of the game. '''
        for i in range(self.population_size):
            # Check for if the game has been closed
            for event in pyg_get():
                if event.type == pyg_QUIT:
                    pyg_quit()
                    quit()
            
            # Increase frame count
            self.agents_data[i][5] += 1

            # Have each agent take their action
            action = agents.get_action(agents.agents[i][0], self)

            # Move
            self.agents_data[i][0], self.agents_data[i][1] = self._move(action, self.agents_data[i][0], self.agents_data[i][1]) # Update the head
            self.agents_data[i][2].insert(0, self.agents_data[i][1])

            # Check if game over
            # If the snake hasn't made enough progress, it's executed
            self.agents_data[i][6] = False
            if self.is_collision() or (self.agents_data[i][5] > 125*len(self.agents_data[i][2])):
                self.agents_data[i][6] = True
                return self.agents_data[i][6], self.agents_data[i][3]

            # Place new food or just move
            if self.agents_data[i][1] == self.agents_data[i][1]:
                self.agents_data[i][3] += 1
                self._food_gen()
            else:
                self.agents_data[i][2].pop()

        # Update ui and clock
        self._update_ui()
        self.clock.tick(self.fps)

        # Return values for the agent to process
        return self.agents_data[i][6], self.score


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

        for agent in self.agents_data:
            # Draw out the snake block by block
            #x, y = self.snake[0]
            #pyg_rect(self.false_display, self.color2, [x, y, TILE_SIZE, TILE_SIZE])
            #pyg_rect(self.false_display, self.color1, [x, y, TILE_SIZE, TILE_SIZE], 1)
            #for x, y in self.snake[1:]:
            for x, y in agent[2]:
                pyg_rect(self.false_display, agent[7][0], [x*self.x_ratio, y*self.y_ratio, TILE_SIZE*self.x_ratio, TILE_SIZE*self.y_ratio])
                pyg_rect(self.false_display, agent[7][1], [x*self.x_ratio, y*self.y_ratio, TILE_SIZE*self.x_ratio, TILE_SIZE*self.y_ratio], 1)

            # Draw the food block
            pyg_rect(self.false_display, RED, [agent[4].x*self.x_ratio, agent[4].y*self.y_ratio, TILE_SIZE*self.x_ratio, TILE_SIZE*self.y_ratio])

        # Draw a line for the margin
        pyg_line(self.false_display, WHITE, (0, self.height), (self.width, self.height), width=2)

        # Show the current generation
        text = FONT.render(f"Generation: {self.generation}", True, WHITE)
        self.false_display.blit(text, [TILE_SIZE*20, int(self.height+(TILE_SIZE//4))])

        # Update the display
        self.true_display.blit(pyg_scale(self.false_display, self.true_display.get_size()), (0, 0))
        pyg_display.flip()


    def _move(self, action, agent_direction, agent_head):
        '''
        Choose a new direction from straight, right, or left, where
        straight is to continue the current direction and right and
        left are to turn in either direction from the perspective
        of what direction the snake is currently heading.
        '''
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(agent_direction)

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
        agent_direction = new_dir

        # Update the head coordinates
        x = agent_head.x
        y = agent_head.y
        if agent_direction == Direction.RIGHT:
            x += TILE_SIZE
        elif agent_direction == Direction.LEFT:
            x -= TILE_SIZE
        elif agent_direction == Direction.DOWN:
            y += TILE_SIZE
        elif agent_direction == Direction.UP:
            y -= TILE_SIZE
        agent_head = Point(x, y)
        return agent_direction, agent_head



class SnakeGameHuman():
    ''' The logic for playing Snake as a human. '''
    def __init__(self, width, height, margin, fps=100):
        # Initialize input data
        self.fps = fps
        self.width = width
        self.height = height
        self.margin = margin

        # Initialze display
        self.true_display = pyg_display.set_mode((self.width, self.height+self.margin), pyg_RESIZABLE)
        self.false_display = self.true_display.copy()
        pyg_display.set_caption("Snake")
        self.clock = pyg_Clock()

        # Initialize internal values
        self.score = 0
        self.direction = Direction.UP
        self.wait = True
        self.escape = False

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
                # Check for exiting out of window
                elif event.key == pyg_K_ESCAPE:
                    self.escape = True

        # Skip if there's no input
        if self.wait:
            self._update_ui()
            self.clock.tick(self.fps)
            return False, 0

        # Move
        self._move() # Update the head
        self.snake.insert(0, self.head)

        # Check if game over
        if self.is_collision() or self.escape:
            return True, self.score

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
        return False, self.score


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
        #for x, y in self.snake[1:]:
        for x, y in self.snake:
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
        self.true_display.blit(pyg_scale(self.false_display, self.true_display.get_size()), (0, 0))
        pyg_display.flip()


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

