from collections import deque
from model import QNet, QTrainer
from helper import Direction, Point, TILE_SIZE, MAX_MEMORY, BATCH_SIZE, LR

from random import randint as rand_randint
from random import sample as rand_sample
from numpy import expand_dims as np_expand_dims
from numpy import array as np_array
from numpy import argmax as np_argmax


class Agent():
    ''' The snake agent- not the model itself. '''
    def __init__(self, num_states=11, num_actions=3):
        # Current episode
        self.episode = 0

        # Internal data
        self.epsilon = 0 # Randomness
        self.gamma = 0.9 # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = QNet.linear_QNet(input_size=num_states, hidden_sizes=[128, 128], output_size=num_actions, learning_rate=LR)
        self.trainer = QTrainer(self.model, gamma=self.gamma)
        
        # Colors
        self.color1 = (rand_randint(0, 255), rand_randint(0, 255), rand_randint(0, 255))
        self.color2 = (rand_randint(0, 255), rand_randint(0, 255), rand_randint(0, 255))

        # Internal score for fitness testing
        self.top_score = 0

        # Internal mean for graphing
        self.mean = 0


    def get_state(self, game):
        ''' Update the agent's state. '''
        head = game.snake[0]
        point_l = Point(head.x - TILE_SIZE, head.y)
        point_r = Point(head.x + TILE_SIZE, head.y)
        point_u = Point(head.x, head.y - TILE_SIZE)
        point_d = Point(head.x, head.y + TILE_SIZE)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # List of states, using binary checks to fill values
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
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y   # food down
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
