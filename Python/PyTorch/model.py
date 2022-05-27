from helper import LR

from os import makedirs as os_makedirs
from os import remove as os_remove
from os.path import exists as os_exists
from random import randrange
from numpy import array as np_array

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class LinearNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=None, random_model=True):
        super().__init__()

        # 
        # Build the model
        #
        self.layers = nn.ModuleList()
        self.input_size = input_size
        self.output_size = output_size

        # Add hidden layers
        self.num_layers = 0
        if hidden_layers == None:
            if random_model:
                # Randomized hidden layers
                for i in range(randrange(2, 3)):
                    # Get random layer size and use it and the input size to create the layer
                    size = randrange(64, 128)
                    self.layers.append(nn.Linear(input_size, size))

                    # Update input size for next layer and number of layers
                    input_size = size
                    self.num_layers += 1
        else:
            # Predetermined hidden layers
            for size in hidden_layers:
                # Create layer
                self.layers.append(nn.Linear(input_size, size))

                # Update input size for next layer and number of layers
                input_size = size
                self.num_layers += 1
        
        # Output layer
        self.layers.append(nn.Linear(input_size, output_size))
        self.num_layers += 1

        # Initialize weight/bias values
        self.apply(self.initialize_weights)

        # 
        # Set whether or not to use gpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        #
        # Set optimizer and loss
        self.optimizer = optim.Adam(self.parameters(), lr=LR)
        self.criterion = nn.MSELoss()

        #self.model_info()
    

    def initialize_weights(self, layer):
        '''Initialize the layer weight and bias values.'''
        if isinstance(layer, nn.Conv2d):
            #nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("relu"))
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.constant_(layer.weight, 1)
            nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.Linear):
            #nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("relu"))
            #nn.init.constant_(layer.bias, 0)
            nn.init.uniform_(layer.bias, 0, 1)


    def forward(self, x):
        '''Get output from model.'''
        #for layer in self.layers: x = layer(x)
        for i, layer in enumerate(self.layers): x = layer(x) if (i==self.num_layers) else layer(F.relu(x))
        return x

    def save(self, max_episodes, new_score, old_score):
        '''Save the model.'''
        # Get model data
        shapes = []
        for info in self.named_modules():
            if isinstance(info[1], nn.Linear):
                shapes.append(f"-{list(info[1].weight.data.size())}")
        
        # Convert to TorchScript
        model_scripted = torch.jit.script(self)
        
        # Save model
        if not os_exists(r"./models"):
            os_makedirs(r"./models")

        if not os_exists(r"./models/DQN_model_({})_({})__({}).pt".format(max_episodes, ''.join(shapes), old_score)):
            model_scripted.save(r"./models/DQN_model_({})_({})__({}).pt".format(max_episodes, ''.join(shapes), new_score))
        else: # delete existing file to make a new one
            os_remove(r"./models/DQN_model_({})_({})__({}).pt".format(max_episodes, ''.join(shapes), old_score))
            model_scripted.save(r"./models/DQN_model_({})_({})__({}).pt".format(max_episodes, ''.join(shapes), new_score))

    
    def model_info(self):
        '''Get your model's information (layers, sizes, parameters, etc).'''
        # Break
        print("\n==================================================")

        # Print model's state_dict
        print("Model's state_dict:")
        for param_tensor in self.state_dict():
            print(param_tensor, "\t", self.state_dict()[param_tensor].size())

        # Print optimizer's state_dict
        #print("\nOptimizer's state_dict:")
        #for var_name in self.optimizer.state_dict():
        #    print(var_name, "\t", self.optimizer.state_dict()[var_name])

        # Break
        print("==================================================\n")


class QTrainer:
    def __init__(self, model, gamma):
        self.gamma = gamma
        self.model = model

    def train_step(self, state, action, reward, next_state, done):
        '''
        The training function. Handles both long
        and short term memory training.
        '''
        # Should be in the state (n, x)
        state = torch.tensor(np_array(state), dtype=torch.float)
        next_state = torch.tensor(np_array(next_state), dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # If it is in this state: (1, x)
        # Then convert it to (n, x)
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # Predicted Q values with current state
        pred = self.model(state)
        target = pred.clone() # Create a copy we'll modify

        for i in range(len(done)):
            # Check to see if the formula should be used
            # Ternary for the sake of efficiency, but it's basically
            # if done, reward, else calculate a value
            Q_new = reward[i] if done[i] else reward[i] + self.gamma * torch.max(self.model(next_state[i]))

            # Set target
            target[i][torch.argmax(action[i]).item()] = Q_new
    
        # Get the loss from this target/pred set and train the model
        self.model.optimizer.zero_grad()
        loss = self.model.criterion(target, pred)
        loss.backward()
        self.model.optimizer.step()