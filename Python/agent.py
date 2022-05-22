from collections import deque
from model import QNet, QTrainer
from helper import Direction, Point, TILE_SIZE, MAX_MEMORY, BATCH_SIZE, LR, WIDTH, HEIGHT

from tensorflow.keras.models import load_model as tf_load_model
from math import dist as math_dist
from random import randint as rand_randint
from random import sample as rand_sample
from numpy import expand_dims as np_expand_dims
from numpy import array as np_array
from numpy import argmax as np_argmax


class AgentDQN():
    ''' The snake agent- not the model itself. '''
    def __init__(self, model_path=None):
        # Current episode
        self.episode = 0

        # Internal data
        self.epsilon = 0 # Randomness
        self.gamma = 0.9 # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()

        if model_path == None:
            self.model = QNet.linear_QNet(19, 3)
        else:
            self.model = tf_load_model(model_path)

        self.trainer = QTrainer(self.model, gamma=self.gamma)

        # Colors
        self.color1 = (rand_randint(0, 255), rand_randint(0, 255), rand_randint(0, 255))
        self.color2 = (rand_randint(0, 255), rand_randint(0, 255), rand_randint(0, 255))

        # Graphing and data variables
        self.total_score = 0
        self.top_score = 0
        self.mean_score = 0


    def get_state(self, game):
        ''' Update the agent's state. '''
        point_l = Point(game.head.x - TILE_SIZE, game.head.y)
        point_r = Point(game.head.x + TILE_SIZE, game.head.y)
        point_u = Point(game.head.x, game.head.y - TILE_SIZE)
        point_d = Point(game.head.x, game.head.y + TILE_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # List of states, mainly using binary checks to fill values
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Position relative to food
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down

            # Current head position
            game.head.x / WIDTH,  # x position
            game.head.y / HEIGHT, # y position

            # Current food position
            game.food.x / WIDTH,  # x position
            game.food.y / HEIGHT, # y position

            # Distance from head to middle block
            math_dist([game.snake[len(game.snake)//2].x/WIDTH, game.head.y/HEIGHT],\
                      [game.snake[len(game.snake)//2].y/HEIGHT, game.food.y/HEIGHT]),

            # Current tail position
            math_dist([game.snake[-1].x/WIDTH, game.head.y/HEIGHT],\
                      [game.snake[-1].y/HEIGHT, game.food.y/HEIGHT]),

            # Food distances (Values are scaled to be 0-1)
            # Current distance
            math_dist([game.head.x/WIDTH, game.head.y/HEIGHT],\
                      [game.food.x/WIDTH, game.food.y/HEIGHT]),
            # Previous distance
            math_dist([game.snake[1].x/WIDTH, game.snake[1].y/HEIGHT],\
                      [game.food.x/WIDTH, game.food.y/HEIGHT])
        ]
        return np_array(state, dtype=int)


    def remember(self, state, action, reward, next_state, done):
        ''' Add values to the memory. '''
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached


    def train_long_memory(self):
        ''' The long term memory training done after each episode. '''
        if len(self.memory) > BATCH_SIZE:
            mini_sample = rand_sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, next_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)


    def train_short_memory(self, state, action, reward, next_state, done):
        ''' The short term memory training done after each action. '''
        self.trainer.train_step(state, action, reward, next_state, done)


    def get_action(self, state):
        '''
        Get the current action of the model. If it doesn't have enough
        in the memory to train the model, it chooses a random action.
        Otherwise, it gets an action from the model.
        '''
        # Random moves (exploration via randomness vs exploitation of "safe" moves)
        self.epsilon = 80 - self.episode
        final_move = [0, 0, 0]
        if rand_randint(0, 200) < self.epsilon:
            move = rand_randint(0, 2)
            final_move[move] = 1
        else:
            state0 = np_expand_dims(np_array(state, dtype=float), 0)
            prediction = self.model(state0)
            move = np_argmax(prediction).item()
            final_move[move] = 1

        return final_move



class AgentGA():
    ''' The snake agent- not the model itself. '''
    def __init__(self, population_size, model_path=None):
        # Current generation
        self.generation = 0

        # Internal data
        self.epsilon = 0 # Randomness
        self.gamma = 0.9 # Discount rate
        self.population_size = population_size

        # Generate models and associated colors
        self.agents = []
        for i in range(self.population_size):
            # Model
            if model_path == None:
                model = QNet.linear_QNet(11, 3, hidden_layers=[128, 128], random_model=False)
            else:
                model = tf_load_model(model_path)

            # [model, fitness]
            self.agents.append([model, 0])

        # Overall top score
        self.top_score = 0

        # Internal mean for graphing
        self.mean_score = 0


    def get_parents(self, fitness_threshold):
        '''
        Sort agents by fitness.
        If fittness_threshold is > 1, use it as the number of agents.
        If it's < 1, use it as a percentage.
        '''
        # Sort agents by fitness (highest to lowest)
        self.agents.sort(key=lambda a: a[1], reverse=True)

        # Get the number of breeding agents
        # If it's less than 1, the threshold is treated as a percent
        # Otherwise, it's treated as a set number of parents
        if fitness_threshold <= 1:
            cutoff = (int)(fitness_threshold * self.population_size)
            if not cutoff % 2: cutoff -= 1
            if cutoff < 2: cutoff = 2
        else:
            cutoff = fitness_threshold

        # Get number of times each parent pair needs to breed
        num_children = self.population_size // cutoff

        return self.agents[:cutoff], num_children



    def _get_state(self, game):
        ''' Update the agent's state. '''
        point_l = Point(game.head.x - TILE_SIZE, game.head.y)
        point_r = Point(game.head.x + TILE_SIZE, game.head.y)
        point_u = Point(game.head.x, game.head.y - TILE_SIZE)
        point_d = Point(game.head.x, game.head.y + TILE_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # List of states, using binary checks to fill values
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r, game.body)) or
            (dir_l and game.is_collision(point_l, game.body)) or
            (dir_u and game.is_collision(point_u, game.body)) or
            (dir_d and game.is_collision(point_d, game.body)),

            # Danger right
            (dir_u and game.is_collision(point_r, game.body)) or
            (dir_d and game.is_collision(point_l, game.body)) or
            (dir_l and game.is_collision(point_u, game.body)) or
            (dir_r and game.is_collision(point_d, game.body)),

            # Danger left
            (dir_d and game.is_collision(point_r, game.body)) or
            (dir_u and game.is_collision(point_l, game.body)) or
            (dir_r and game.is_collision(point_u, game.body)) or
            (dir_l and game.is_collision(point_d, game.body)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y   # food down
        ]
        return np_array(state, dtype=float)


    def get_action(self, model, game):
        '''
        Get the current action of the model.=
        '''
        final_move = [0, 0, 0]
        prediction = model(np_expand_dims(self._get_state(game), 0))
        move = np_argmax(prediction).item()
        final_move[move] = 1
        return final_move
