import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError


class QNet():
    def linear_QNet(model_path="", input_size=11, hidden_sizes=[256], output_size=3, learning_rate=0.001):
        '''
        Build the model, given an input size, list of hidden layer sizes,
        output size, and learning rate.
        
        Either load a trained model or create a new one.
        '''
        # If you want to load a trained model...
        if (model_path):
            model = load_model(model_path)

        # If you want to build a new model
        else:
            model = Sequential()
            model.add(Dense(units=input_size, activation="relu"))
            for layer_size in hidden_sizes:
                model.add(Dense(units=layer_size, activation="relu"))
            model.add(Dense(units=output_size))
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mean_squared_error", metrics=[RootMeanSquaredError()])
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
        state = np.array(state, dtype=float)
        next_state = np.array(next_state, dtype=float)
        action = np.array(action, dtype=int)
        reward = np.array(reward, dtype=float)

        if len(state.shape) == 1:
            # If it is in this state: (1, x)
            # Then convert it to (n, x)
            state = np.expand_dims(state, 0)
            next_state = np.expand_dims(next_state, 0)
            action = np.expand_dims(action, 0)
            reward = np.expand_dims(reward, 0)
            done = (done, )

        # Predicted Q values with current state
        pred = self.model(state)

        target = np.array(pred, copy=True)  
        for i in range(len(done)):
            # Reshape reward, if needed
            if len(reward[i].shape) == 1:
                _reward = np.expand_dims(reward[i], 0)
            else:
                _reward = reward[i]

            # Check to see if the formula should be used
            if done[i]:
                Q_new = _reward
            else:
                # Reshape next_state, if needed
                if len(next_state[i].shape) == 1:
                    _next_state = np.expand_dims(next_state[i], 0)
                else:
                    _next_state = next_state[i]

                # Formula
                Q_new = _reward + self.gamma * np.max(self.model(_next_state))

            # Reshape action, if needed
            if len(action[i].shape) == 1:
                _action = np.expand_dims(action[i], 0)
            else:
                _action = action[i]

            # Set target
            target[i][np.argmax(_action).item()] = Q_new

        # Train the model
        self.model.fit(x=state,
                       y=target,
                       epochs=1,
                       batch_size=len(done),
                       verbose=0)
