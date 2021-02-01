"""
Created on Thu Jan 26 12:29:10 2021

@author: dasan
"""

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam
from collections import deque
from statistics import mean
import numpy as np
import random
import matplotlib.pyplot as plt

LEARNING_RATE = 1e-3
MAX_MEMORY = 1000000
BATCH_SIZE = 20
GAMMA = 0.95
EXPLORATION_DECAY = 0.995
EXPLORATION_MIN = 0.01

class Network:

    def __init__(self, observation_space, action_space):

        self.action_space = action_space
        self.observation_space = observation_space
        self.memory = deque(maxlen=MAX_MEMORY)
        self.exploration_rate = 1.0

        self.model = Sequential()
        self.model.add(Dense(32, input_shape=(observation_space,), activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(self.action_space, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))

    def add_to_memory(self, state, action, reward, next_state, done):
        state_array = np.array(state)
        next_state_array = np.array(next_state)
        self.memory.append((state_array, action, reward, next_state_array, done))

    def take_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(0, self.action_space)
        return self.take_action_without_exploration(state)
    
    def take_action_without_exploration(self, state):
        state = np.array(state)
        state = np.reshape(state, [1, self.observation_space])
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        else:
            
            minibatch = random.sample(self.memory, BATCH_SIZE)
            state_list = np.zeros(shape = (20,4))
            state_next_list = np.zeros(shape = (20,4))
            actions = np.empty(20)
            # rewards = np.empty(20)
            done_list = []
            Q = np.empty(20)
            i = 0
            for state, action, reward, state_next, done in minibatch:
                state_list[i] = state
                state_next_list[i] = state_next
                actions[i] = action
                done_list.append(done)
                Q[i] = reward
                
                i = i+1
            
            Q_next_values = self.model.predict(state_next_list)
            Q_values = self.model.predict(state_list)
            i = 0
            for done in done_list:
                if not done:
                    Q[i] = Q[i] + GAMMA * np.amax(Q_next_values[i])
                    
                Q_values[i][int(actions[i])] = Q[i]
                i = i+1
                
            self.model.fit(state_list, Q_values, verbose=0)
            self.exploration_rate *= EXPLORATION_DECAY
            self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)



import json
from flask import request, current_app
from flask import Flask
app = Flask(__name__)

@app.route('/', methods = ['POST'])
def take_action():
    data = request.get_json(force=True)
    data['action'] = int(current_app.solver.take_action(data['state']))
    return json.dumps(data)

@app.route('/play', methods = ['POST'])
def take_action_no_replay():
    data = request.get_json(force=True)
    data['action'] = int(current_app.solver.take_action_without_exploration(data['state']))
    return json.dumps(data)

@app.route('/solver', methods = ['POST'])
def init_solver():
    data = request.get_json(force=True)

    current_app.solver = Network(data['observation-space'], data['action-space'])
    return 'Solver created'

@app.route('/memory', methods = ['POST'])
def store_to_memory():
    data = request.get_json(force=True)
    current_app.solver.add_to_memory(data['state'], data['action'], data['reward'], data['state_next'], data['done'])
    return 'State added to memory'

@app.route('/replay', methods = ['GET'])
def experience_replay():
    current_app.solver.experience_replay()
    return 'Experienced replayed'

@app.route('/load', methods = ['POST'])
def load_model():
    data = request.get_json(force=True)
    current_app.solver.model = keras.models.load_model(data['text'])
    return 'Model loaded'

if __name__ == "__main__":
    app.run(host="127.0.0.1", port = 8080)