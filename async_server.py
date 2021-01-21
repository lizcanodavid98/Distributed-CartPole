# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 10:31:10 2020

@author: dasan
"""

import asyncio
from aiohttp import web
import nest_asyncio
from random import randint
nest_asyncio.apply()

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

        # self.model = Sequential()
        # self.model.add(Dense(32, input_shape=(observation_space,), activation='relu'))
        # self.model.add(Dense(32, activation='relu'))
        # self.model.add(Dense(self.action_space, activation='linear'))
        # self.model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))
        
        self.model = None

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
            

    def get_model(self):
        return self.model

async def handle_post(request):
    data = await request.json()
    
    # await asyncio.sleep(randint(0,10)/100)
    
    data["action"] = int(solver.take_action(data['state']))
    return web.json_response(data)

async def no_replay_action(request):
    data = await request.json()
    data['action'] = int(solver.take_action_without_exploration(data['state']))
    return web.json_response(data)

async def solver(request):
    data = await request.json()
    
    
    solver.action_space = data['action-space']
    solver.observation_space = data['observation-space']
    
    solver.model = Sequential()
    solver.model.add(Dense(32, input_shape=(solver.observation_space,), activation='relu'))
    solver.model.add(Dense(32, activation='relu'))
    solver.model.add(Dense(solver.action_space, activation='linear'))
    solver.model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))
    
   
    return web.Response(text = 'Solver created.')

async def memory(request):
    data = await request.json()
    state = data['state']
    action = data['action']
    reward = data['reward']
    state_next = data['state_next']
    done = data['done']
    
    solver.add_to_memory(state, action, reward, state_next, done)
    return web.Response(text = 'State added to memory')

async def experience_replay(request):
    solver.experience_replay()
    return web.Response(text = 'Experience replayed.')
    
async def load_model(request):
    data = await request.json()
    model = data['text']
    solver.model = keras.models.load_model(model)
    return web.Response(text = 'Loaded model')

app = web.Application()
app.add_routes([web.post('/', handle_post),
                web.post('/play', no_replay_action),
                web.post('/solver', solver),
                web.post('/memory', memory),
                web.get('/replay', experience_replay),
                web.post('/load', load_model)
                ])

solver = Network(0,0)


if __name__ == '__main__':
    web.run_app(app)