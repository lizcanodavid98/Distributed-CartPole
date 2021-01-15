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


class ScoreEvaluator:

    def __init__(self, max_len, average_of_last_runs, model = None):
        self.max_len = max_len
        self.score_table = deque(maxlen=self.max_len)
        self.model = model
        self.average_of_last_runs = average_of_last_runs

    def store_score(self, episode, step):
        self.score_table.append([episode, step])

    def plot_evaluation(self, title = "Training"):
        print(self.model.summary()) if self.model is not None else print("Model not defined!")
        avg_score = mean(self.score_table[1])
        x = []
        y = []
        for i in range(len(self.score_table)):
            x.append(self.score_table[i][0])
            y.append(self.score_table[i][1])

        average_range = self.average_of_last_runs if self.average_of_last_runs is not None else len(x)
        plt.plot(x, y, label="score per run")
        plt.plot(x[-average_range:], [np.mean(y[-average_range:])] * len(y[-average_range:]), linestyle="--",
                 label="last " + str(average_range) + " runs average")
        title = "CartPole-v1 " + str(title)
        plt.title(title)
        plt.xlabel("Runs")
        plt.ylabel("Score")
        plt.show()
        

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
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        else:
            minibatch = random.sample(self.memory, BATCH_SIZE)
            for state, action, reward, state_next, done in minibatch:
                Q = reward
                if not done:
                    Q = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
                Q_values = self.model.predict(state)
                Q_values[0][action] = Q
                self.model.fit(state, Q_values, verbose=0)
            self.exploration_rate *= EXPLORATION_DECAY
            self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)
            print(self.exploration_rate)

    def get_model(self):
        return self.model

# async def handle(request):
#     name = request.match_info.get('name', "Anonymous")
#     text = "Hello, " + name
#     print("Request received from {}")
#     return web.Response(text=text)

async def handle_post(request):
    data = await request.json()
    
    # await asyncio.sleep(randint(0,10)/100)
    
    data["action"] = int(solver.take_action(data['state']))
    # print(data["action"])
    # print("Post received")
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

async def score_evaluator(request):
    data = await request.json()
    
    
    score_eval.max_len = data['max-len']
    score_eval.average_of_last_runs = data['average_of_last_runs']
    score_eval.model = solver.get_model()
    
    return web.Response(text = 'Score evaluator created')

async def store_score(request):
    data = await request.json()
    
    score_eval.store_score(data['episode'], data['step'])
    return web.Response(text = 'Score stored')

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

async def plot(request):
    data = await request.json()
    text = data['text']
    score_eval.plot_evaluation(title = text)

app = web.Application()
app.add_routes([web.post('/', handle_post),
                web.post('/solver', solver),
                web.post('/score_evaluator', score_evaluator),
                web.post('/memory', memory),
                web.post('/score', store_score),
                web.get('/replay', experience_replay),
                web.post('/plot', plot)
                ])

solver = Network(0,0)
score_eval = ScoreEvaluator(400, 50)

if __name__ == '__main__':
    web.run_app(app)