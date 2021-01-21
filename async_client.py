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

nest_asyncio.apply()
session = aiohttp.ClientSession()

url = 'http://127.0.0.1:8080'




async def fetch_post(session, url, data_json):
    async with session.post(url, data = data_json) as resp:
        pass
        # print(url + ' ' + str(resp.status))
        # print(await resp.text())
        
async def post_json(data_json, url=url):
    await fetch_post(session, url, data_json)
          
async def fetch_post_json(session, url, data_json):
    async with session.post(url, data = data_json) as resp:
        return await resp.json()

async def post_expected_json(data_json, url = url):
    data = await fetch_post_json(session, url, data_json)
    # print("Action: {}\nState: {}".format(data['action'], data['state']))
    return data

async def get_something(url = url):
    await fetch_get(session, url)
    
async def fetch_get(session, url):
    async with session.get(url) as resp:
        pass
        # print(url + ' ' + str(resp.status))
        # print(await resp.text())
    
async def close_session():
    await session.close()
   
    
    
    
class TrainSolver:

    def __init__(self, max_episodes):
        self.max_episodes = max_episodes
        self.score_table = deque(maxlen=400)
        self.average_of_last_runs = None
        self.play_episodes = 100
        env = gym.make('CartPole-v1')
        observation_space = env.observation_space.shape[0]
        action_space = env.action_space.n 
    
          
    async def play(self, play_episodes = 100, model = None):
        self.play_episodes = play_episodes
        env = gym.make('CartPole-v1')
        observation_space = env.observation_space.shape[0]
        action_space = env.action_space.n
        episode = 0
        
        solver_params = json.dumps({
            'observation-space': observation_space, 
            'action-space': action_space
            })
    
        solver_url = url + '/solver'
        await post_json(solver_params, url = solver_url)
        
        if model is None:
            print("Can't load specified model.")
        else:
            load_model_url = url + '/load'
            await post_json(json.dumps({'text':model}), url = load_model_url)
        
        
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
                play_url = url + '/play'
                
                data = await post_expected_json(json.dumps(data), url = play_url)
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
                
    async def train(self):
        env = gym.make('CartPole-v1')
        observation_space = env.observation_space.shape[0]
        action_space = env.action_space.n
        
        episode = 0
        solver_params = json.dumps({
            'observation-space': observation_space, 
            'action-space': action_space
            })
    
        solver_url = url + '/solver'
        await post_json(solver_params, url = solver_url)
        
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
                    data = await post_expected_json(json.dumps(data))
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
                    
                    memory_url = url + '/memory'
                    await post_json(json.dumps(data), url = memory_url)
                    
                    state = state_next
                    data['state'] = state.tolist()
    
                    if done:
                        print("Run: " + str(episode) + ", score: " + str(step))
                        # self.score_table.append([episode, step])
                        # score_eval.store_score(episode, step)
                        break
                    replay_url = url + '/replay'
                    await get_something(url = replay_url)
        
        
        
        
if __name__ == '__main__':
    trainer = TrainSolver(60)
    loop = asyncio.get_event_loop()
    # loop.run_until_complete(trainer.play(play_episodes = 10, model = 'cartpole_model_v3.h5'))
    loop.run_until_complete(trainer.train())
    loop.run_until_complete(close_session())
  
