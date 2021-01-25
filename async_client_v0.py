# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 10:24:06 2020

@author: dasan
"""


import aiohttp, asyncio, nest_asyncio, async_timeout
import json
import gym
from collections import deque
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam
import random


from connectivity import *
    
    
class TrainSolver:

    def __init__(self, max_episodes):
        self.max_episodes = max_episodes
        self.score_table = deque(maxlen=400)
        self.average_of_last_runs = None
        self.play_episodes = 100
        env = gym.make('CartPole-v1')
        observation_space = env.observation_space.shape[0]
        action_space = env.action_space.n 
    
          
    def play(self, play_episodes = 100, model = None):
        self.play_episodes = play_episodes
        env = gym.make('CartPole-v1')
        observation_space = env.observation_space.shape[0]
        action_space = env.action_space.n
        episode = 0
        
        start_solver(session, observation_space, action_space)
        
        if model is None:
            print("Can't load specified model.")
        else:
            load_model(session, model)
        
        
        data = {
            'action': 0,
            'state': [0,0,0,0],
            'state_next': [0,0,0,0],
            'reward': 1.0,
            'done': True,
            'info': {}
        }
        
        while episode < self.play_episodes:
            episode += 1
            state = env.reset()
            
            if type(state) != list:
                data["state"] = state.tolist()
            else:
                data["state"] = state
            
            step = 0
            
            while True:
                step += 1
                
                
                data = get_action(session, data)
                action = data['action']
                
                state_next, reward, done, info = env.step(action)
                if not done:
                    reward = reward
                else:
                    reward = -reward
                
                data["state_next"] = state_next.tolist()
                data["reward"] = reward
                data["done"] = done
                data["info"] = info
                data["action"] = action
                
                state = state_next
                data['state'] = state.tolist()
                
                if done:
                    print('Run: ' + str(episode) + ', score: ' + str(step))
                    break
                
    def train(self):
        env = gym.make('CartPole-v1')
        observation_space = env.observation_space.shape[0]
        action_space = env.action_space.n
        
        episode = 0
        start_solver(session, observation_space, action_space)
        
        while episode < self.max_episodes:
            episode += 1
            state = env.reset()
        
            data = {
            'action': 0,
            'state': [0,0,0,0],
            'state_next': [0,0,0,0],
            'reward': 1.0,
            'done': True,
            'info': {}
            }
            
            if type(state) != list:
                data["state"] = state.tolist()
            else:
                data["state"] = state
                
            step = 0
            
            while True:
    
                    step += 1
                    # env.render()
                    data = get_action(session, data, training = True)
                    state_next, reward, done, info = env.step(data['action'])
                    if not done:
                        reward = reward
                    else:
                        reward = -reward
                        
                    data["state_next"] = state_next.tolist()
                    data["reward"] = reward
                    data["done"] = done
                    data["info"] = info
                    # data["action"] = action
                    
                    store_in_memory(session, data)
                    
                    state = state_next
                    data['state'] = state.tolist()
    
                    if done:
                        print("Run: " + str(episode) + ", score: " + str(step))
                        # self.score_table.append([episode, step])
                        # score_eval.store_score(episode, step)
                        break
                    replay_experience(session)
        
        
        
        
if __name__ == '__main__':
    
    session = new_session(url = 'http://127.0.0.1:8080')
    
    trainer = TrainSolver(100)
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(trainer.play(play_episodes = 10, model = 'cartpole_model_v3.h5'))
    # # loop.run_until_complete(trainer.train())
    # loop.run_until_complete(session.close_session())
  

    trainer.play(play_episodes = 1, model = 'cartpole_model_v3.h5')
    session.close_session()