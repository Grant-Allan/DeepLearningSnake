from helper import LR
from numpy import array, expand_dims, max, argmax
from random import randrange
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError


class LinearNet():
    def linear_QNet(input_size, output_size, random_model=True, hidden_layers=None, model_path=""):
        '''
        Build the model, given an input size, list of hidden layer sizes,
        output size, and learning rate.

        Either load a trained model or create a new one.
        '''
        # Load a trained model or build a new one
        if (model_path):
            model = load_model(model_path)
        else:
            model = Sequential()
            model.add(InputLayer(input_shape=(input_size,)))

            # Get the number and size of hidden layers
            if random_model:
                for i in range(randrange(1, 2)):
                    model.add(Dense(units=randrange(8, 64), activation="relu"))
            else:
                for size in hidden_layers:
                    model.add(Dense(units=size, activation="relu"))

            model.add(Dense(units=output_size))
            model.compile(optimizer=Adam(learning_rate=LR), loss="mean_squared_error", metrics=[RootMeanSquaredError()])
        return model


class QTrainer():
    def __init__(self, model, gamma=0.9):
        self.model = model
        self.gamma = gamma


    def train_step(self, state, action, reward, next_state, done):
        '''
        The training function. Handles both long
        and short term memory training.
        '''
        # Should be in the state (n, x)
        state = array(state, dtype=float)
        next_state = array(next_state, dtype=float)
        action = array(action, dtype=int)
        reward = array(reward, dtype=float)

        if len(state.shape) == 1:
            # If it is in this state: (1, x)
            # Then convert it to (n, x)
            state = expand_dims(state, 0)
            next_state = expand_dims(next_state, 0)
            action = expand_dims(action, 0)
            reward = expand_dims(reward, 0)
            done = (done, )

        # Predicted Q values with current state
        pred = self.model(state)

        target = array(pred, copy=True)
        for i in range(len(done)):
            # Reshape reward, if needed
            if len(reward[i].shape) == 1:
                _reward = expand_dims(reward[i], 0)
            else:
                _reward = reward[i]

            # Check to see if the formula should be used
            if done[i]:
                Q_new = _reward
            else:
                # Reshape next_state, if needed
                if len(next_state[i].shape) == 1:
                    _next_state = expand_dims(next_state[i], 0)
                else:
                    _next_state = next_state[i]

                # Formula
                Q_new = _reward + self.gamma * max(self.model(_next_state))

            # Reshape action, if needed
            if len(action[i].shape) == 1:
                _action = expand_dims(action[i], 0)
            else:
                _action = action[i]

            # Set target
            target[i][argmax(_action).item()] = Q_new

        # Train the model
        self.model.fit(x=state,
                       y=target,
                       epochs=1,
                       batch_size=len(done),
                       verbose=0)
